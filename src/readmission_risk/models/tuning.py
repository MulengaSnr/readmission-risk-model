"""Tune random forest and XGBoost, then compare them with the baselines.

Run from the project root (after ``make data``), one step per process:

    uv run python -m readmission_risk.models.tuning --tune random_forest
    uv run python -m readmission_risk.models.tuning --tune xgboost
    uv run python -m readmission_risk.models.tuning --compare

``make models`` runs all three. Each tuning step logs every fit with its time
and saves its results as soon as it finishes, so a slow or interrupted run is
visible and does not lose completed work. (A first single-process run slowed
to a crawl in the XGBoost stage after the random-forest stage, although the
same fits run at normal speed in a fresh process.)

Design:

- A random search per model, scored by PR-AUC on the *same* patient-grouped
  folds as the baselines. Preprocessing sits inside each pipeline, so it is
  refitted on every training fold.
- No class weighting (``class_weight`` / ``scale_pos_weight``): it distorts
  probabilities, which must stay calibrated, and does not improve ranking.
- Choosing the best of N settings on these folds makes its CV score slightly
  optimistic. The untouched test set (step 6) gives the unbiased estimate.
- Each candidate is compared with logistic regression *fold by fold*. Paired
  differences remove the shared fold-to-fold variation, so they show whether
  a gain is consistent rather than an artefact of one lucky fold.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from readmission_risk.config import PROJECT_ROOT, load_config
from readmission_risk.data.clean import GROUP_COL, TARGET
from readmission_risk.features.build import add_features, build_preprocessor
from readmission_risk.models.baselines import (
    Folds,
    cross_validate_models,
    grouped_folds,
    make_baselines,
    summarise,
)

logger = logging.getLogger(__name__)

# Threads per model fit. The search itself runs fits one at a time, so the
# machine's cores are used inside each fit without oversubscription.
N_JOBS = 4


def make_candidates(seed: int = 42) -> dict[str, tuple[Pipeline, dict[str, Any]]]:
    """Unfitted tree-model pipelines and their search spaces.

    Search spaces are kept to the hyperparameters that matter most for each
    model. ``model__`` prefixes address the estimator step in the pipeline.

    Args:
        seed: Random seed for the estimators.

    Returns:
        Mapping of name to ``(pipeline, param_distributions)``.
    """
    forest = Pipeline(
        [
            ("prep", build_preprocessor(include_race=False)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300, n_jobs=N_JOBS, random_state=seed
                ),
            ),
        ]
    )
    forest_space = {
        "model__max_features": ["sqrt", 0.1, 0.2, 0.3],
        "model__min_samples_leaf": [5, 10, 20, 50, 100],
        "model__max_depth": [None, 12, 20],
    }
    boosted = Pipeline(
        [
            ("prep", build_preprocessor(include_race=False)),
            (
                "model",
                XGBClassifier(
                    tree_method="hist",
                    eval_metric="logloss",
                    n_jobs=N_JOBS,
                    random_state=seed,
                ),
            ),
        ]
    )
    boosted_space = {
        "model__n_estimators": randint(200, 801),
        "model__learning_rate": loguniform(0.01, 0.2),
        "model__max_depth": randint(2, 7),
        "model__min_child_weight": loguniform(1, 50),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.5, 0.5),
        "model__reg_lambda": loguniform(0.1, 10),
    }
    return {
        "random_forest": (forest, forest_space),
        "xgboost": (boosted, boosted_space),
    }


def tune(
    x: pd.DataFrame,
    y: np.ndarray,
    folds: Folds,
    pipeline: Pipeline,
    space: dict[str, Any],
    n_iter: int,
    seed: int = 42,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Random search over ``space``, scored by PR-AUC on the given folds.

    A plain loop rather than ``RandomizedSearchCV`` so that every fit is
    logged with its score and duration while the search runs. Settings are
    drawn with ``ParameterSampler``, exactly as ``RandomizedSearchCV`` does.

    Args:
        x: Feature table.
        y: Binary outcomes.
        folds: ``(train_idx, valid_idx)`` pairs shared with the baselines.
        pipeline: Unfitted pipeline.
        space: Parameter distributions for the pipeline.
        n_iter: Number of settings to try.
        seed: Seed for sampling the settings.

    Returns:
        ``(best_params, trials)`` where ``trials`` has one row per setting,
        best first.
    """
    settings = list(ParameterSampler(space, n_iter=n_iter, random_state=seed))
    rows = []
    for i, params in enumerate(settings, start=1):
        scores, seconds = [], []
        for j, (tr, va) in enumerate(folds, start=1):
            started = time.perf_counter()
            model = clone(pipeline).set_params(**params).fit(x.iloc[tr], y[tr])
            prob = model.predict_proba(x.iloc[va])[:, 1]
            seconds.append(time.perf_counter() - started)
            scores.append(average_precision_score(y[va], prob))
            logger.info(
                "setting %d/%d fold %d/%d  PR-AUC %.4f  %.1f s",
                i, len(settings), j, len(folds), scores[-1], seconds[-1],
            )  # fmt: skip
        rows.append(
            {
                **{k.removeprefix("model__"): v for k, v in params.items()},
                "mean_test_score": float(np.mean(scores)),
                "std_test_score": float(np.std(scores)),
                "mean_fit_time": float(np.mean(seconds)),
            }
        )
        best = max(r["mean_test_score"] for r in rows)
        logger.info(
            "setting %d/%d mean PR-AUC %.4f (best so far %.4f)",
            i, len(settings), rows[-1]["mean_test_score"], best,
        )  # fmt: skip
    order = np.argsort([-r["mean_test_score"] for r in rows], kind="stable")
    trials = pd.DataFrame(rows).iloc[order].reset_index(drop=True)
    return settings[order[0]], trials


def paired_difference(
    results: pd.DataFrame, reference: str, metric: str = "pr_auc"
) -> pd.DataFrame:
    """Fold-by-fold difference of each model's metric against a reference.

    Args:
        results: Output of ``cross_validate_models`` (one row per model/fold).
        reference: Name of the model to compare against.
        metric: Metric column; higher must be better.

    Returns:
        Per model: mean and SD of the difference, and folds where it won.

    Raises:
        KeyError: If ``reference`` is not in ``results``.
    """
    wide = results.pivot(index="fold", columns="model", values=metric)
    if reference not in wide:
        raise KeyError(f"Reference model {reference!r} not in results")
    diff = wide.drop(columns=reference).sub(wide[reference], axis=0)
    return pd.DataFrame(
        {
            "mean_diff": diff.mean(),
            "sd_diff": diff.std(),
            "folds_better": (diff > 0).sum(),
            "n_folds": len(diff),
        }
    ).sort_values("mean_diff", ascending=False)


def _to_builtin(value: Any) -> Any:
    """Convert numpy scalars so parameters can be written as JSON."""
    return value.item() if isinstance(value, np.generic) else value


def load_best_params(path: Path) -> dict[str, dict[str, Any]]:
    """Read the saved best parameters for each tuned model.

    Args:
        path: JSON file written by :func:`save_best_params`.

    Returns:
        Mapping of model name to pipeline parameters.

    Raises:
        FileNotFoundError: If no model has been tuned yet.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run with --tune first")
    return json.loads(path.read_text())


def save_best_params(path: Path, name: str, params: dict[str, Any]) -> None:
    """Add or replace one model's best parameters in a JSON file.

    Args:
        path: JSON file mapping model name to parameters; created if missing.
        name: Model name.
        params: Pipeline parameters (``model__...`` keys).
    """
    saved = load_best_params(path) if path.exists() else {}
    saved[name] = {k: _to_builtin(v) for k, v in params.items()}
    path.write_text(json.dumps(saved, indent=2, sort_keys=True) + "\n")


def tuned_models(
    best_params: dict[str, dict[str, Any]], seed: int = 42
) -> dict[str, Pipeline]:
    """Candidate pipelines with their saved best parameters applied.

    Args:
        best_params: Output of :func:`load_best_params`.
        seed: Random seed for the estimators.

    Returns:
        Mapping of model name to unfitted, configured pipeline.

    Raises:
        KeyError: If a saved name is not one of :func:`make_candidates`.
    """
    candidates = make_candidates(seed)
    unknown = set(best_params) - set(candidates)
    if unknown:
        raise KeyError(f"Unknown models in best parameters: {sorted(unknown)}")
    return {
        name: clone(candidates[name][0]).set_params(**params)
        for name, params in best_params.items()
    }


def main() -> None:
    """Command-line entry point: tune named models, or compare tuned models."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--tune", nargs="+", choices=list(make_candidates()))
    action.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    config = load_config()
    seed = config["seed"]
    train = pd.read_parquet(
        PROJECT_ROOT / config["paths"]["processed"] / "train.parquet"
    )
    x = add_features(train)
    y = train[TARGET].to_numpy()
    folds = grouped_folds(y, train[GROUP_COL], config["cv"]["n_splits"], seed)
    out = PROJECT_ROOT / config["paths"]["metrics"]
    out.mkdir(parents=True, exist_ok=True)
    params_path = out / "best_params.json"

    if args.tune:
        candidates = make_candidates(seed)
        for name in args.tune:
            pipeline, space = candidates[name]
            n_iter = config["tuning"]["n_iter"][name]
            logger.info("Tuning %s: %d settings x %d folds", name, n_iter, len(folds))
            started = time.perf_counter()
            params, trials = tune(x, y, folds, pipeline, space, n_iter, seed)
            trials.to_csv(out / f"tuning_{name}.csv", index=False, float_format="%.5f")
            save_best_params(params_path, name, params)
            logger.info(
                "%s done in %.1f min, best PR-AUC %.4f",
                name, (time.perf_counter() - started) / 60,
                trials.loc[0, "mean_test_score"],
            )  # fmt: skip
        return

    models = {
        "logistic_regression": make_baselines()["logistic_regression"],
        **tuned_models(load_best_params(params_path), seed),
    }
    capacities = config["evaluation"]["capacities"]
    results = cross_validate_models(x, y, folds, models, capacities)
    results.to_csv(out / "model_cv.csv", index=False, float_format="%.5f")
    print(summarise(results).round(4).to_string())
    print(paired_difference(results, "logistic_regression").round(4).to_string())


if __name__ == "__main__":
    main()
