"""Final, one-time evaluation on the held-out test set (step 6a).

Run from the project root (after ``make models``):

    uv run python -m readmission_risk.models.evaluate

The protocol was fixed before the test set was scored:

- XGBoost is the primary model; logistic regression is the transparent
  reference. Both are refitted on the full training set with the settings
  chosen in cross-validation, then scored on the test set once.
- 95% confidence intervals come from a patient-level bootstrap: patients are
  resampled with replacement and keep all their encounters, matching the way
  the data were split. The XGBoost minus logistic-regression difference is
  computed on the same resamples, so it is a paired comparison.
- No recalibration step: out-of-fold predictions on the training set were
  already well calibrated. If the test set disagrees, that is reported, not
  fixed using the test set.
- If XGBoost does not beat logistic regression on the test set, that is
  reported; the model is not switched.

The script refuses to overwrite existing test results unless ``--force`` is
given. Re-running is deterministic, so ``--force`` only reproduces the same
numbers; it must never be used to try something new against the test set.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from readmission_risk.analysis import wilson_interval
from readmission_risk.config import PROJECT_ROOT, load_config
from readmission_risk.data.clean import GROUP_COL, TARGET
from readmission_risk.features.build import add_features
from readmission_risk.models.baselines import make_baselines
from readmission_risk.models.metrics import evaluate
from readmission_risk.models.tuning import load_best_params, tuned_models
from readmission_risk.visualisation import plots

logger = logging.getLogger(__name__)

PRIMARY = "xgboost"
REFERENCE = "logistic_regression"
DIFFERENCE = f"{PRIMARY}_minus_{REFERENCE}"


def final_models(best_params: dict, seed: int = 42) -> dict[str, Pipeline]:
    """The two unfitted pipelines evaluated on the test set, primary first."""
    return {
        PRIMARY: tuned_models({PRIMARY: best_params[PRIMARY]}, seed)[PRIMARY],
        REFERENCE: clone(make_baselines()[REFERENCE]),
    }


def patient_bootstrap(
    y: np.ndarray,
    probs: dict[str, np.ndarray],
    groups: pd.Series,
    n_boot: int = 1000,
    seed: int = 42,
    capacities: Sequence[float] = (0.1, 0.2),
) -> pd.DataFrame:
    """Metrics on resamples of patients (each keeps all their encounters).

    Args:
        y: Binary outcomes.
        probs: Predicted probabilities per model, aligned with ``y``.
        groups: Patient identifiers, aligned with ``y``.
        n_boot: Number of resamples.
        seed: Random seed.
        capacities: Review capacities passed to :func:`evaluate`.

    Returns:
        One row per (resample, model) with every metric.
    """
    codes, uniques = pd.factorize(groups)
    n_patients = len(uniques)
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_boot):
        draws = rng.integers(0, n_patients, n_patients)
        copies = np.bincount(draws, minlength=n_patients)[codes]
        idx = np.repeat(np.arange(len(y)), copies)
        for name, p in probs.items():
            rows.append(
                {"boot": b, "model": name, **evaluate(y[idx], p[idx], capacities)}
            )
    return pd.DataFrame(rows)


def summarise_test(
    point: dict[str, dict[str, float]], boot: pd.DataFrame
) -> pd.DataFrame:
    """Point estimates with 95% percentile intervals, plus the paired difference.

    Args:
        point: Metrics on the full test set per model (from :func:`evaluate`).
        boot: Output of :func:`patient_bootstrap`; must contain ``PRIMARY``
            and ``REFERENCE``.

    Returns:
        Rows of (model, metric) with ``estimate``, ``ci_low`` and ``ci_high``.
        The difference row is primary minus reference (for Brier, negative
        means the primary model is better).
    """
    wide = boot.set_index(["boot", "model"]).unstack("model")
    rows = []
    for metric in wide.columns.levels[0]:
        samples = {m: wide[(metric, m)] for m in point}
        samples[DIFFERENCE] = samples[PRIMARY] - samples[REFERENCE]
        estimates = {m: point[m][metric] for m in point}
        estimates[DIFFERENCE] = point[PRIMARY][metric] - point[REFERENCE][metric]
        for model, values in samples.items():
            low, high = np.percentile(values, [2.5, 97.5])
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "estimate": estimates[model],
                    "ci_low": float(low),
                    "ci_high": float(high),
                }
            )
    return pd.DataFrame(rows)


def calibration_table(
    y: np.ndarray, prob: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Observed readmission rate per decile of predicted risk.

    Args:
        y: Binary outcomes.
        prob: Predicted probabilities.
        n_bins: Number of equal-size bins.

    Returns:
        One row per bin: ``n``, ``mean_pred``, ``observed`` and its Wilson
        95% interval (``ci_low``, ``ci_high``).
    """
    bins = pd.qcut(prob, n_bins, labels=False, duplicates="drop")
    table = (
        pd.DataFrame({"bin": bins, "prob": prob, "y": y})
        .groupby("bin")
        .agg(n=("y", "size"), events=("y", "sum"), mean_pred=("prob", "mean"))
    )
    table["observed"] = table["events"] / table["n"]
    table["ci_low"], table["ci_high"] = wilson_interval(table["events"], table["n"])
    return table.reset_index()


def gains_curve(y: np.ndarray, prob: np.ndarray) -> pd.DataFrame:
    """Share of readmissions reached as the review list grows, highest risk first.

    Args:
        y: Binary outcomes.
        prob: Predicted probabilities.

    Returns:
        ``share_reviewed`` and ``share_reached``, starting at (0, 0).
    """
    order = np.argsort(-prob, kind="stable")
    reached = np.cumsum(y[order]) / y.sum()
    reviewed = np.arange(1, len(y) + 1) / len(y)
    return pd.DataFrame(
        {
            "share_reviewed": np.r_[0.0, reviewed],
            "share_reached": np.r_[0.0, reached],
        }
    )


def save_models(models: dict[str, Pipeline], folder: Path, seed: int) -> None:
    """Save fitted pipelines with the library versions needed to load them."""
    folder.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, folder / f"{name}.joblib")
    meta = {
        "models": list(models),
        "primary": PRIMARY,
        "trained_on": "data/processed/train.parquet (full training set)",
        "seed": seed,
        "versions": {
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "joblib": joblib.__version__,
        },
    }
    (folder / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")


def guard_single_use(results_path: Path, force: bool) -> None:
    """Stop if the test set has already been evaluated, unless forced.

    Raises:
        SystemExit: If ``results_path`` exists and ``force`` is false.
    """
    if results_path.exists() and not force:
        raise SystemExit(
            f"{results_path} exists: the test set has already been evaluated. "
            "Use --force only to reproduce those results."
        )


def main() -> None:
    """Command-line entry point: refit, score the test set once, write reports."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing test results (reproduction only)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    config = load_config()
    seed = config["seed"]
    processed = PROJECT_ROOT / config["paths"]["processed"]
    metrics_dir = PROJECT_ROOT / config["paths"]["metrics"]
    figures = PROJECT_ROOT / config["paths"]["figures"]
    results_path = metrics_dir / "test_metrics.csv"
    guard_single_use(results_path, args.force)

    train = pd.read_parquet(processed / "train.parquet")
    test = pd.read_parquet(processed / "test.parquet")
    y_train, y_test = train[TARGET].to_numpy(), test[TARGET].to_numpy()
    x_train, x_test = add_features(train), add_features(test)

    models = final_models(load_best_params(metrics_dir / "best_params.json"), seed)
    probs = {}
    for name, model in models.items():
        logger.info("Fitting %s on %d training encounters", name, len(train))
        model.fit(x_train, y_train)
        probs[name] = model.predict_proba(x_test)[:, 1]
    save_models(models, PROJECT_ROOT / "models", seed)

    capacities = config["evaluation"]["capacities"]
    point = {name: evaluate(y_test, p, capacities) for name, p in probs.items()}
    n_boot = config["evaluation"]["n_bootstrap"]
    logger.info("Patient bootstrap: %d resamples", n_boot)
    boot = patient_bootstrap(y_test, probs, test[GROUP_COL], n_boot, seed, capacities)
    summary = summarise_test(point, boot)
    summary.to_csv(results_path, index=False, float_format="%.5f")

    calibration = {name: calibration_table(y_test, p) for name, p in probs.items()}
    pd.concat(calibration, names=["model"]).reset_index(level=0).to_csv(
        metrics_dir / "test_calibration.csv", index=False, float_format="%.5f"
    )
    gains = {name: gains_curve(y_test, p) for name, p in probs.items()}

    plots.apply_style()
    labels = {PRIMARY: "XGBoost", REFERENCE: "Logistic regression"}
    fig = plots.plot_calibration(
        {labels[k]: v for k, v in calibration.items()},
        title="Predicted risk matches observed readmission (test set)",
    ).figure
    plots.save(fig, figures / "test_calibration.png")
    fig = plots.plot_gains(
        {labels[k]: v for k, v in gains.items()},
        capacities=capacities,
        title="Share of readmissions reached by reviewing the highest-risk patients",
    ).figure
    plots.save(fig, figures / "test_gains.png")

    logger.info("Test set: %d encounters, %d patients, readmission rate %.2f%%",
                len(test), test[GROUP_COL].nunique(), 100 * y_test.mean())  # fmt: skip
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
