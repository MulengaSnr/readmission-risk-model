"""Baseline models compared in patient-grouped cross-validation.

Run from the project root (after ``make data``):

    uv run python -m readmission_risk.models.baselines

Three baselines, each answering a different question:

- ``no_skill``: predicts the training readmission rate for everyone. Any model
  must beat this to be useful at all.
- ``prior_inpatient``: logistic regression on the number of inpatient stays in
  the previous year, the strongest single signal in the EDA. This is close to
  a rule a ward could apply without a model; a complex model must beat it to
  justify its cost.
- ``logistic_regression``: all engineered features through the shared
  preprocessor. The interpretable benchmark for random forest / XGBoost.

Every model is scored on the *same* folds, so differences are paired
comparisons rather than luck of the split. The test set is not touched.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from readmission_risk.config import PROJECT_ROOT, load_config
from readmission_risk.data.clean import GROUP_COL, TARGET
from readmission_risk.features.build import add_features, build_preprocessor
from readmission_risk.models.metrics import evaluate

logger = logging.getLogger(__name__)

Folds = list[tuple[np.ndarray, np.ndarray]]


def make_baselines() -> dict[str, Pipeline]:
    """Create the unfitted baseline pipelines, keyed by name.

    Logistic regression is unweighted: class weights would distort the
    probabilities we later need to be calibrated, and do not help ranking.

    Returns:
        Ordered mapping of model name to scikit-learn pipeline.
    """
    return {
        "no_skill": Pipeline([("model", DummyClassifier(strategy="prior"))]),
        "prior_inpatient": Pipeline(
            [
                (
                    "select",
                    ColumnTransformer(
                        [("log1p", FunctionTransformer(np.log1p), ["number_inpatient"])]
                    ),
                ),
                ("model", LogisticRegression()),
            ]
        ),
        "logistic_regression": Pipeline(
            [
                ("prep", build_preprocessor(include_race=False)),
                ("model", LogisticRegression(max_iter=2000)),
            ]
        ),
    }


def grouped_folds(
    y: np.ndarray, groups: pd.Series, n_splits: int = 5, seed: int = 42
) -> Folds:
    """Patient-disjoint, outcome-stratified cross-validation folds.

    Args:
        y: Binary outcomes.
        groups: Patient identifiers, aligned with ``y``.
        n_splits: Number of folds.
        seed: Random seed.

    Returns:
        List of ``(train_idx, valid_idx)`` positional index pairs.

    Raises:
        RuntimeError: If a patient appears on both sides of a fold.
    """
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(splitter.split(np.zeros(len(y)), y, groups))
    g = np.asarray(groups)
    for i, (tr, va) in enumerate(folds):
        if np.intersect1d(g[tr], g[va]).size:  # defensive: splitter guarantees it
            raise RuntimeError(f"Fold {i} shares patients between train and valid")
    return folds


def cross_validate_models(
    x: pd.DataFrame,
    y: np.ndarray,
    folds: Folds,
    models: dict[str, BaseEstimator],
    capacities: Sequence[float] = (0.1, 0.2),
) -> pd.DataFrame:
    """Fit and score every model on every fold.

    Each fit uses a fresh clone, so preprocessing is learned from the training
    part of the fold only.

    Args:
        x: Feature table (output of ``add_features``).
        y: Binary outcomes, aligned with ``x``.
        folds: ``(train_idx, valid_idx)`` pairs, shared by all models.
        models: Unfitted estimators keyed by name.
        capacities: Review capacities passed to :func:`evaluate`.

    Returns:
        One row per (model, fold) with all metrics.
    """
    rows = []
    for name, model in models.items():
        for i, (tr, va) in enumerate(folds):
            fitted = clone(model).fit(x.iloc[tr], y[tr])
            prob = fitted.predict_proba(x.iloc[va])[:, 1]
            rows.append({"model": name, "fold": i, **evaluate(y[va], prob, capacities)})
            logger.info("%-20s fold %d  PR-AUC %.4f", name, i, rows[-1]["pr_auc"])
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation of each metric across folds, per model."""
    metrics = results.drop(columns="fold")
    summary = metrics.groupby("model", sort=False).agg(["mean", "std"])
    return summary.sort_values(("pr_auc", "mean"), ascending=False)


def main() -> None:
    """Command-line entry point: grouped CV of the baselines on the training set."""
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config()
    train = pd.read_parquet(
        PROJECT_ROOT / config["paths"]["processed"] / "train.parquet"
    )
    x = add_features(train)
    y = train[TARGET].to_numpy()
    folds = grouped_folds(y, train[GROUP_COL], config["cv"]["n_splits"], config["seed"])
    results = cross_validate_models(
        x, y, folds, make_baselines(), config["evaluation"]["capacities"]
    )
    out = PROJECT_ROOT / config["paths"]["metrics"]
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "baseline_cv.csv", index=False, float_format="%.5f")
    logger.info("Wrote %s", out / "baseline_cv.csv")
    print(summarise(results).round(4).to_string())


if __name__ == "__main__":
    main()
