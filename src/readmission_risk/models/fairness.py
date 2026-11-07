"""Fairness audit of the final models by sex, age band and race (step 6b).

Run from the project root (after ``make evaluate``):

    uv run python -m readmission_risk.models.fairness

Design:

- The operating policy is fixed: flag the top 20% of test encounters by
  predicted risk, using one cut-off for everyone. Each group is measured
  under that same policy, as it would be in use.
- Per group: ROC-AUC (ranking within the group), calibration gap (mean
  predicted minus observed rate), selection rate (share flagged), recall
  (share of the group's readmissions that are flagged, i.e. equal
  opportunity) and precision (share of flagged who are readmitted).
- Recall is also compared with the largest group of each attribute, as a
  paired difference.
- 95% intervals come from the same patient-level bootstrap as the test
  evaluation. The cut-off stays fixed across resamples because it is the
  policy being audited.
- Race is never a model input; it is used here only for auditing. Groups with
  fewer than ``MIN_EVENTS`` readmissions are reported but flagged as too
  small to support conclusions.
"""

from __future__ import annotations

import argparse
import logging
import math
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from readmission_risk.config import PROJECT_ROOT, load_config
from readmission_risk.data.clean import GROUP_COL, TARGET
from readmission_risk.features.build import add_features
from readmission_risk.models.evaluate import PRIMARY, REFERENCE
from readmission_risk.visualisation import plots

logger = logging.getLogger(__name__)

MIN_EVENTS = 50
CAPACITY = 0.2

AGE_BANDS = {
    "[0-10)": "<40", "[10-20)": "<40", "[20-30)": "<40", "[30-40)": "<40",
    "[40-50)": "40-59", "[50-60)": "40-59",
    "[60-70)": "60-79", "[70-80)": "60-79",
    "[80-90)": "80+", "[90-100)": "80+",
}  # fmt: skip
ATTRIBUTE_LABELS = {"sex": "Sex", "age_band": "Age band", "race": "Race"}
METRICS = ("roc_auc", "calibration_gap", "selection_rate", "recall", "precision")


def audit_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Sex, age band and race for each encounter (missing race -> "Unknown").

    Args:
        df: Encounters with ``gender``, ``age`` and ``race`` columns.

    Returns:
        A frame with columns ``sex``, ``age_band`` and ``race``.

    Raises:
        ValueError: If an age value has no band.
    """
    age = df["age"].astype(str).map(AGE_BANDS)
    if age.isna().any():
        raise ValueError(
            f"Unmapped age values: {sorted(df['age'][age.isna()].unique())}"
        )
    return pd.DataFrame(
        {
            "sex": df["gender"].astype(str),
            "age_band": age,
            "race": df["race"].astype("object").fillna("Unknown"),
        },
        index=df.index,
    )


def top_share_threshold(prob: np.ndarray, capacity: float = CAPACITY) -> float:
    """Score cut-off that flags the top ``capacity`` share of encounters."""
    k = max(1, math.ceil(capacity * len(prob) - 1e-9))
    return float(np.sort(prob)[::-1][k - 1])


def group_metrics(y: np.ndarray, prob: np.ndarray, flagged: np.ndarray) -> dict:
    """Audit metrics for one group; undefined metrics are NaN.

    Args:
        y: Binary outcomes for the group.
        prob: Predicted probabilities.
        flagged: Boolean array, True where the policy flags the encounter.

    Returns:
        A dict with the keys in :data:`METRICS`.
    """
    events = y.sum()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        auc = roc_auc_score(y, prob) if 0 < events < len(y) else np.nan
    return {
        "roc_auc": auc,
        "calibration_gap": prob.mean() - y.mean(),
        "selection_rate": flagged.mean(),
        "recall": flagged[y == 1].mean() if events else np.nan,
        "precision": y[flagged].mean() if flagged.any() else np.nan,
    }


def _all_groups(
    y: np.ndarray, prob: np.ndarray, flagged: np.ndarray, groups: pd.DataFrame
) -> dict[tuple[str, str], dict]:
    out = {}
    for attribute in groups:
        values = groups[attribute].to_numpy()
        for group in pd.unique(values):
            mask = values == group
            out[(attribute, group)] = group_metrics(y[mask], prob[mask], flagged[mask])
    return out


def fairness_table(
    y: np.ndarray,
    prob: np.ndarray,
    groups: pd.DataFrame,
    patients: pd.Series,
    n_boot: int = 1000,
    seed: int = 42,
    capacity: float = CAPACITY,
) -> pd.DataFrame:
    """Per-group audit metrics with patient-bootstrap 95% intervals.

    Args:
        y: Binary outcomes.
        prob: Predicted probabilities.
        groups: Output of :func:`audit_groups`, aligned with ``y``.
        patients: Patient identifiers, aligned with ``y``.
        n_boot: Number of bootstrap resamples.
        seed: Random seed.
        capacity: Share of all encounters flagged by the policy.

    Returns:
        One row per (attribute, group, metric) with ``estimate``, ``ci_low``,
        ``ci_high``, the group's ``n`` and ``events``, ``small_group``, and
        ``recall_diff`` rows comparing each group with the attribute's
        largest group.
    """
    y = np.asarray(y)
    prob = np.asarray(prob)
    groups = groups.reset_index(drop=True)
    flagged = prob >= top_share_threshold(prob, capacity)
    point = _all_groups(y, prob, flagged, groups)
    reference = {a: groups[a].value_counts().idxmax() for a in groups}

    codes, uniques = pd.factorize(patients)
    rng = np.random.default_rng(seed)
    boot: dict[tuple[str, str, str], list[float]] = {}
    for _ in range(n_boot):
        draws = rng.integers(0, len(uniques), len(uniques))
        idx = np.repeat(
            np.arange(len(y)), np.bincount(draws, minlength=len(uniques))[codes]
        )
        sample = _all_groups(y[idx], prob[idx], flagged[idx], groups.iloc[idx])
        for (attribute, group), metrics in sample.items():
            for metric, value in metrics.items():
                boot.setdefault((attribute, group, metric), []).append(value)
            ref = sample.get((attribute, reference[attribute]), {}).get(
                "recall", np.nan
            )
            boot.setdefault((attribute, group, "recall_diff"), []).append(
                metrics["recall"] - ref
            )

    rows = []
    for (attribute, group), metrics in point.items():
        mask = groups[attribute].to_numpy() == group
        events = int(y[mask].sum())
        estimates = dict(metrics)
        estimates["recall_diff"] = (
            metrics["recall"] - point[(attribute, reference[attribute])]["recall"]
        )
        for metric, estimate in estimates.items():
            samples = np.asarray(boot[(attribute, group, metric)], dtype=float)
            low, high = (
                np.nanpercentile(samples, [2.5, 97.5])
                if np.isfinite(samples).any()
                else (np.nan, np.nan)
            )
            rows.append(
                {
                    "attribute": attribute,
                    "group": group,
                    "reference_group": reference[attribute],
                    "metric": metric,
                    "estimate": float(estimate),
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "n": int(mask.sum()),
                    "events": events,
                    "small_group": events < MIN_EVENTS,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    """Command-line entry point: audit both final models on the test set."""
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    config = load_config()
    test = pd.read_parquet(PROJECT_ROOT / config["paths"]["processed"] / "test.parquet")
    x, y = add_features(test), test[TARGET].to_numpy()
    groups = audit_groups(test)

    tables, probs = [], {}
    for name in (PRIMARY, REFERENCE):
        path = PROJECT_ROOT / "models" / f"{name}.joblib"
        if not path.exists():
            raise SystemExit(f"{path} not found: run `make evaluate` first")
        prob = probs[name] = joblib.load(path).predict_proba(x)[:, 1]
        logger.info(
            "Auditing %s (%d resamples)", name, config["evaluation"]["n_bootstrap"]
        )
        table = fairness_table(
            y, prob, groups, test[GROUP_COL], config["evaluation"]["n_bootstrap"],
            config["seed"],
        )  # fmt: skip
        tables.append(table.assign(model=name))
    result = pd.concat(tables, ignore_index=True)
    out = PROJECT_ROOT / config["paths"]["metrics"] / "fairness_test.csv"
    result.to_csv(out, index=False, float_format="%.5f")

    primary = result[result["model"] == PRIMARY]
    p = probs[PRIMARY]
    overall_recall = float((p >= top_share_threshold(p))[y == 1].mean())
    plots.apply_style()
    fig = plots.plot_group_metric(
        primary[primary["metric"] == "recall"],
        attribute_labels=ATTRIBUTE_LABELS,
        overall=overall_recall,
        title="Share of each group's readmissions flagged at 20% capacity",
        xlabel="Readmissions flagged (recall)",
        orders={"age_band": ["<40", "40-59", "60-79", "80+"]},
    )
    plots.save(fig, PROJECT_ROOT / config["paths"]["figures"] / "fairness_recall.png")
    plt.close(fig)
    view = primary.pivot_table(index=["attribute", "group", "n", "events"],
                               columns="metric", values="estimate")  # fmt: skip
    print(view.round(3).to_string())


if __name__ == "__main__":
    main()
