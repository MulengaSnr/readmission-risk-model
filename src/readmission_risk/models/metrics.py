"""Evaluation metrics, including capacity metrics a care team can act on.

The care team can review only a fixed share of discharges, so besides the
usual ranking and probability metrics we report precision and recall among
the top-ranked ``capacity`` share of encounters.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def capture_at_capacity(
    y_true: ArrayLike, y_score: ArrayLike, capacity: float
) -> tuple[float, float]:
    """Precision and recall if the team reviews only the highest-risk share.

    Ties at the cut-off are shared out in proportion (the expected result of
    breaking ties at random), so a model that gives everyone the same score
    reaches exactly ``capacity`` of the readmissions, the no-skill reference.

    Args:
        y_true: Binary outcomes (1 = readmitted within 30 days).
        y_score: Risk scores; higher means higher risk.
        capacity: Share of encounters reviewed, in ``(0, 1]``.

    Returns:
        ``(precision, recall)`` among the reviewed encounters.

    Raises:
        ValueError: On mismatched or empty inputs, a capacity outside
            ``(0, 1]`` or no positive outcomes.
    """
    y = np.asarray(y_true)
    s = np.asarray(y_score, dtype=float)
    if y.ndim != 1 or y.shape != s.shape or y.size == 0:
        raise ValueError("y_true and y_score must be 1-D arrays of the same length")
    if not 0 < capacity <= 1:
        raise ValueError(f"capacity must be in (0, 1], got {capacity}")
    n_pos = y.sum()
    if n_pos == 0:
        raise ValueError("y_true contains no positive outcomes")

    # Tolerance stops float error (0.3 * 10 = 3.0000000000000004) adding a row.
    k = max(1, math.ceil(capacity * y.size - 1e-9))
    threshold = np.sort(s)[::-1][k - 1]
    above = s > threshold
    tied = s == threshold
    reviewed_from_ties = k - above.sum()
    positives = y[above].sum() + y[tied].sum() * reviewed_from_ties / tied.sum()
    return float(positives / k), float(positives / n_pos)


def evaluate(
    y_true: ArrayLike, y_prob: ArrayLike, capacities: Sequence[float] = (0.1, 0.2)
) -> dict[str, float]:
    """Compute the project's standard metrics for one set of predictions.

    Args:
        y_true: Binary outcomes.
        y_prob: Predicted probability of readmission.
        capacities: Review capacities to report, as shares of encounters.

    Returns:
        A flat dict: ``pr_auc`` (primary metric; equals prevalence for a
        no-skill model), ``roc_auc``, ``brier`` (lower is better) and
        ``precision_at_10pct`` / ``recall_at_10pct`` style keys per capacity.
    """
    metrics = {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }
    for capacity in capacities:
        precision, recall = capture_at_capacity(y_true, y_prob, capacity)
        label = f"{round(capacity * 100)}pct"
        metrics[f"precision_at_{label}"] = precision
        metrics[f"recall_at_{label}"] = recall
    return metrics
