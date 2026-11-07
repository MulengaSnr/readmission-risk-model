"""Descriptive statistics used in exploratory analysis and reporting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from readmission_risk.data.clean import TARGET


def wilson_interval(
    events: np.ndarray | int, n: np.ndarray | int, confidence: float = 0.95
) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score confidence interval for a binomial proportion.

    Preferred over the normal ("Wald") interval because it stays inside [0, 1]
    and behaves well for small groups and rates near 0 or 1.

    Args:
        events: Number of positive outcomes.
        n: Number of trials (must be > 0).
        confidence: Confidence level, e.g. 0.95.

    Returns:
        ``(lower, upper)`` bounds as arrays.
    """
    events = np.asarray(events, dtype=float)
    n = np.asarray(n, dtype=float)
    z = stats.norm.ppf(0.5 + confidence / 2)
    p = events / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    # At p = 0 or 1 a bound equals p exactly in theory; clamp so float rounding
    # can never put it on the wrong side (which breaks error bars).
    return np.minimum(centre - half, p), np.maximum(centre + half, p)


def rate_table(
    df: pd.DataFrame, col: str, target: str = TARGET, min_count: int = 0
) -> pd.DataFrame:
    """Readmission rate with a 95% Wilson interval for each level of ``col``.

    Missing values are reported as their own ``"Missing"`` level, because
    missingness itself can be informative.

    Args:
        df: Data containing ``col`` and the binary ``target``.
        col: Column to group by.
        target: Binary outcome column.
        min_count: Drop levels with fewer encounters than this.

    Returns:
        Data frame indexed by level with columns ``n``, ``events``, ``rate``,
        ``ci_low``, ``ci_high``, in the column's natural order.
    """
    key = df[col]
    if not isinstance(key.dtype, pd.CategoricalDtype):
        key = key.astype("object").where(key.notna(), "Missing")
    grouped = df[target].groupby(key, observed=True, sort=True)
    table = pd.DataFrame({"n": grouped.size(), "events": grouped.sum()})
    table = table[table["n"] >= max(min_count, 1)]
    table["rate"] = table["events"] / table["n"]
    table["ci_low"], table["ci_high"] = wilson_interval(table["events"], table["n"])
    table.index.name = col
    return table


def missingness(df: pd.DataFrame) -> pd.Series:
    """Share of missing values per column, for columns with any missing.

    Args:
        df: Any data frame.

    Returns:
        Series sorted from most to least missing.
    """
    share = df.isna().mean()
    return share[share > 0].sort_values(ascending=False)


def chi_square_independence(
    df: pd.DataFrame, col: str, target: str = TARGET
) -> dict[str, float]:
    """Chi-square test of independence between ``col`` and the target.

    Also returns Cramér's V so the size of the association is reported, not
    just its significance (with ~80k rows almost anything is "significant").

    Args:
        df: Data containing both columns.
        col: Categorical column (missing values treated as their own level).
        target: Binary outcome column.

    Returns:
        Dict with ``chi2``, ``dof``, ``p_value`` and ``cramers_v``.
    """
    key = df[col].astype("object").where(df[col].notna(), "Missing")
    table = pd.crosstab(key, df[target])
    chi2, p_value, dof, _ = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    cramers_v = float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))
    return {
        "chi2": float(chi2),
        "dof": int(dof),
        "p_value": float(p_value),
        "cramers_v": cramers_v,
    }
