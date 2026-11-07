"""Sensitivity analysis: does the model need the payer (insurance type) feature?

Run from the project root (after ``make models``):

    uv run python -m readmission_risk.models.sensitivity

Why: payer is the fourth most important feature (SHAP, README section 10) and
may stand in for income or social disadvantage. If the model does as well
without it, the simpler and less proxy-prone model is preferable.

Protocol, fixed before running:

- The tuned XGBoost settings are fitted with and without ``payer_code`` on
  the same five patient-grouped folds of the **training set**. The test set
  is not used.
- Decision rule: drop payer if removing it lowers mean PR-AUC by less than
  ``DECISION_MARGIN`` (0.005, about a third of the fold-to-fold SD).
- The evaluated v1.0 model is not changed. A model without payer would be
  adopted in the production project, and any test-set score for it labelled
  as a second use of the test set.
- Out-of-fold predictions are also used to compare recall at 20% capacity by
  sex, age band and race, to see whether payer affects any group.
"""

from __future__ import annotations

import argparse
import logging
import math

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone

from readmission_risk.config import PROJECT_ROOT, load_config
from readmission_risk.data.clean import GROUP_COL, TARGET
from readmission_risk.features.build import add_features, build_preprocessor
from readmission_risk.models.baselines import Folds, grouped_folds
from readmission_risk.models.evaluate import PRIMARY
from readmission_risk.models.fairness import audit_groups, group_metrics
from readmission_risk.models.metrics import evaluate
from readmission_risk.models.tuning import (
    load_best_params,
    paired_difference,
    tuned_models,
)

logger = logging.getLogger(__name__)

FEATURE = "payer_code"
WITH, WITHOUT = "with_payer", "without_payer"
DECISION_MARGIN = 0.005


def make_variants(best_params: dict, seed: int = 42) -> dict[str, BaseEstimator]:
    """The tuned primary model with and without the payer feature."""
    full = tuned_models({PRIMARY: best_params[PRIMARY]}, seed)[PRIMARY]
    reduced = clone(full).set_params(prep=build_preprocessor(exclude=(FEATURE,)))
    return {WITH: full, WITHOUT: reduced}


def cross_validate_oof(
    x: pd.DataFrame,
    y: np.ndarray,
    folds: Folds,
    models: dict[str, BaseEstimator],
    capacities: tuple[float, ...] | list[float] = (0.1, 0.2),
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Per-fold metrics plus out-of-fold predictions for each model.

    Args:
        x: Feature table.
        y: Binary outcomes.
        folds: ``(train_idx, valid_idx)`` pairs.
        models: Unfitted estimators keyed by name.
        capacities: Review capacities passed to :func:`evaluate`.

    Returns:
        ``(results, oof)``: one row per (model, fold), and for each model an
        array holding every row's prediction from the fold that held it out.
    """
    rows, oof = [], {}
    for name, model in models.items():
        oof[name] = np.full(len(y), np.nan)
        for i, (tr, va) in enumerate(folds):
            prob = clone(model).fit(x.iloc[tr], y[tr]).predict_proba(x.iloc[va])[:, 1]
            oof[name][va] = prob
            rows.append({"model": name, "fold": i, **evaluate(y[va], prob, capacities)})
            logger.info("%-14s fold %d  PR-AUC %.4f", name, i, rows[-1]["pr_auc"])
    return pd.DataFrame(rows), oof


def decide(diff: pd.DataFrame, margin: float = DECISION_MARGIN) -> str:
    """Apply the pre-specified rule to the paired PR-AUC difference.

    Args:
        diff: Output of ``paired_difference`` with reference ``WITH``; must
            contain the row ``WITHOUT``.
        margin: Largest acceptable mean PR-AUC loss from dropping payer.

    Returns:
        ``"drop"`` or ``"keep"``.
    """
    return "drop" if diff.loc[WITHOUT, "mean_diff"] > -margin else "keep"


def group_recall(
    y: np.ndarray, oof: dict[str, np.ndarray], groups: pd.DataFrame, capacity: float
) -> pd.DataFrame:
    """Recall and selection rate per group at a single top-``capacity`` cut-off.

    Args:
        y: Binary outcomes.
        oof: Out-of-fold predictions per model.
        groups: Output of ``audit_groups``, aligned with ``y``.
        capacity: Share of all encounters flagged.

    Returns:
        One row per (attribute, group) with recall and selection rate for
        each model, and the recall change from dropping payer.
    """
    k = max(1, math.ceil(capacity * len(y) - 1e-9))
    rows = []
    flagged = {m: p >= np.sort(p)[::-1][k - 1] for m, p in oof.items()}
    for attribute in groups:
        values = groups[attribute].to_numpy()
        for group in pd.unique(values):
            mask = values == group
            row = {"attribute": attribute, "group": group,
                   "n": int(mask.sum()), "events": int(y[mask].sum())}  # fmt: skip
            for m, p in oof.items():
                metrics = group_metrics(y[mask], p[mask], flagged[m][mask])
                row[f"recall_{m}"] = metrics["recall"]
                row[f"selection_{m}"] = metrics["selection_rate"]
            row["recall_change"] = row[f"recall_{WITHOUT}"] - row[f"recall_{WITH}"]
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    """Command-line entry point: CV with and without payer on the training set."""
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    config = load_config()
    seed = config["seed"]
    train = pd.read_parquet(
        PROJECT_ROOT / config["paths"]["processed"] / "train.parquet"
    )
    x, y = add_features(train), train[TARGET].to_numpy()
    folds = grouped_folds(y, train[GROUP_COL], config["cv"]["n_splits"], seed)
    metrics_dir = PROJECT_ROOT / config["paths"]["metrics"]
    variants = make_variants(load_best_params(metrics_dir / "best_params.json"), seed)

    capacities = config["evaluation"]["capacities"]
    results, oof = cross_validate_oof(x, y, folds, variants, capacities)
    results.to_csv(metrics_dir / "payer_sensitivity_cv.csv", index=False,
                   float_format="%.5f")  # fmt: skip
    diffs = {
        metric: paired_difference(results, WITH, metric).loc[WITHOUT]
        for metric in ("pr_auc", "roc_auc", "recall_at_20pct")
    }
    groups = group_recall(y, oof, audit_groups(train).reset_index(drop=True), 0.2)
    groups.to_csv(metrics_dir / "payer_sensitivity_groups.csv", index=False,
                  float_format="%.5f")  # fmt: skip

    print(pd.DataFrame(diffs).T.round(4).to_string())
    print(groups.round(3).to_string(index=False))
    decision = decide(paired_difference(results, WITH, "pr_auc"))
    print(f"Decision (margin {DECISION_MARGIN}): {decision} {FEATURE}")


if __name__ == "__main__":
    main()
