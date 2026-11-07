"""Shared chart style and plotting helpers.

Style rules (applied to every figure in the project):
- one series -> one colour; the overall rate is a muted reference line
- thin bars with a gap, hairline solid grid, recessive axes
- labels and values in ink colours, never the series colour
- every chart answers a question stated in its title
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = "#2a78d6"  # single-series colour (categorical slot 1)
# Categorical order for multi-series charts (fixed, never cycled). Slots 1-2
# pass the palette validator's CVD and contrast checks on SURFACE.
SERIES_ORDER = ("#2a78d6", "#eb6834")


def apply_style() -> None:
    """Set matplotlib defaults for the project's chart style."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.axisbelow": True,
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
        }
    )


def plot_rates(
    table: pd.DataFrame,
    title: str,
    overall: float,
    xlabel: str | None = None,
    horizontal: bool = False,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Bar chart of readmission rate per level with 95% CI whiskers.

    Args:
        table: Output of :func:`readmission_risk.analysis.rate_table`.
        title: Chart title, phrased as the finding or question.
        overall: Overall readmission rate, drawn as a reference line.
        xlabel: Label for the category axis.
        horizontal: Draw horizontal bars (better for long labels).
        ax: Axes to draw on; a new figure is created if omitted.

    Returns:
        The axes.
    """
    if ax is None:
        _, ax = plt.subplots(
            figsize=(7.5, 4.2 if not horizontal else 0.42 * len(table) + 1.2)
        )
    labels = [str(i) for i in table.index]
    pos = range(len(table))
    err = [table["rate"] - table["ci_low"], table["ci_high"] - table["rate"]]
    bar = ax.barh if horizontal else ax.bar
    bar(
        pos, table["rate"], 0.62, color=SERIES, edgecolor=SURFACE, linewidth=2, zorder=2
    )
    if horizontal:
        ax.errorbar(
            table["rate"],
            pos,
            xerr=err,
            fmt="none",
            ecolor=INK_SECONDARY,
            elinewidth=1,
            capsize=0,
            zorder=3,
        )
        ax.axvline(overall, color=MUTED, linewidth=1.2, zorder=1)
        ax.set_yticks(list(pos), labels)
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=None))
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Readmitted within 30 days")
        ax.text(
            overall,
            -0.75,
            f" overall {overall:.1%}",
            color=INK_SECONDARY,
            fontsize=8.5,
            va="bottom",
        )
    else:
        ax.errorbar(
            list(pos),
            table["rate"],
            yerr=err,
            fmt="none",
            ecolor=INK_SECONDARY,
            elinewidth=1,
            capsize=0,
            zorder=3,
        )
        ax.axhline(overall, color=MUTED, linewidth=1.2, zorder=1)
        ax.set_xticks(list(pos), labels)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=None))
        ax.grid(axis="x", visible=False)
        ax.set_ylabel("Readmitted within 30 days")
        # Label the reference line in the right margin, clear of the bars.
        ax.text(
            1.01,
            overall,
            f"overall\n{overall:.1%}",
            transform=ax.get_yaxis_transform(),
            color=INK_SECONDARY,
            fontsize=8.5,
            va="center",
            ha="left",
        )
    if xlabel:
        (ax.set_ylabel if horizontal else ax.set_xlabel)(xlabel)
    ax.set_title(title)
    return ax


def plot_missingness(
    share: pd.Series, title: str, ax: plt.Axes | None = None
) -> plt.Axes:
    """Horizontal bars of the share of missing values per column, labelled.

    Args:
        share: Output of :func:`readmission_risk.analysis.missingness`.
        title: Chart title.
        ax: Axes to draw on; a new figure is created if omitted.

    Returns:
        The axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 0.42 * len(share) + 1.2))
    pos = range(len(share))
    ax.barh(pos, share.to_numpy(), 0.62, color=SERIES, edgecolor=SURFACE, linewidth=2)
    for i, v in enumerate(share.to_numpy()):
        ax.text(
            v + 0.01,
            i,
            f"{v:.1%}" if v >= 0.001 else "<0.1%",
            va="center",
            fontsize=8.5,
            color=INK_SECONDARY,
        )
    ax.set_yticks(list(pos), share.index)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Share of training encounters missing")
    ax.set_title(title)
    return ax


def _series_colours(n: int) -> tuple[str, ...]:
    if n > len(SERIES_ORDER):
        raise ValueError(f"At most {len(SERIES_ORDER)} series, got {n}")
    return SERIES_ORDER[:n]


def plot_calibration(
    tables: dict[str, pd.DataFrame], title: str, ax: plt.Axes | None = None
) -> plt.Axes:
    """Observed vs predicted readmission rate per risk decile, one line per model.

    Args:
        tables: Model label -> output of
            :func:`readmission_risk.models.evaluate.calibration_table`.
        title: Chart title.
        ax: Axes to draw on; a new figure is created if omitted.

    Returns:
        The axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 5.4))
    top = 1.1 * max(
        max(t["mean_pred"].max(), t["ci_high"].max()) for t in tables.values()
    )
    ax.plot([0, top], [0, top], color=MUTED, linewidth=1, zorder=1)
    ax.text(top * 0.97, top * 0.03, "diagonal = perfect calibration",
            color=INK_SECONDARY, fontsize=8.5, ha="right", va="bottom")  # fmt: skip
    for (label, table), colour in zip(
        tables.items(), _series_colours(len(tables)), strict=True
    ):
        err = [
            table["observed"] - table["ci_low"],
            table["ci_high"] - table["observed"],
        ]
        ax.errorbar(table["mean_pred"], table["observed"], yerr=err, fmt="none",
                    ecolor=colour, elinewidth=1, alpha=0.6, zorder=2)  # fmt: skip
        ax.plot(table["mean_pred"], table["observed"], color=colour, linewidth=2,
                marker="o", markersize=6, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=label, zorder=3)  # fmt: skip
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_aspect("equal")
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel("Mean predicted risk (decile of patients)")
    ax.set_ylabel("Observed 30-day readmission rate")
    ax.legend(frameon=False, loc="upper left", labelcolor=INK_SECONDARY)
    ax.set_title(title)
    return ax


def plot_gains(
    curves: dict[str, pd.DataFrame],
    capacities: tuple[float, ...] | list[float],
    title: str,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Cumulative share of readmissions reached as more patients are reviewed.

    The first model's value is labelled at each capacity.

    Args:
        curves: Model label -> output of
            :func:`readmission_risk.models.evaluate.gains_curve`.
        capacities: Review capacities to mark, as shares of patients.
        title: Chart title.
        ax: Axes to draw on; a new figure is created if omitted.

    Returns:
        The axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, zorder=1)
    ax.text(0.62, 0.55, "random order", color=INK_SECONDARY, fontsize=8.5,
            rotation=0, ha="left", va="top")  # fmt: skip
    for (label, curve), colour in zip(
        curves.items(), _series_colours(len(curves)), strict=True
    ):
        ax.plot(curve["share_reviewed"], curve["share_reached"], color=colour,
                linewidth=2, label=label, zorder=3)  # fmt: skip
    first = next(iter(curves.values()))
    for capacity in capacities:
        reached = float(
            np.interp(capacity, first["share_reviewed"], first["share_reached"])
        )
        ax.axvline(capacity, color=GRID, linewidth=1, zorder=0)
        ax.plot(capacity, reached, "o", color=SERIES_ORDER[0], markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)  # fmt: skip
        ax.annotate(f"top {capacity:.0%} → {reached:.0%}", (capacity, reached),
                    xytext=(8, -4), textcoords="offset points", ha="left",
                    va="top", fontsize=8.5, color=INK)  # fmt: skip
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel("Share of discharges reviewed, highest predicted risk first")
    ax.set_ylabel("Share of readmissions reached")
    ax.legend(frameon=False, loc="lower right", labelcolor=INK_SECONDARY)
    ax.set_title(title)
    return ax


def plot_importance(
    table: pd.DataFrame, title: str, ax: plt.Axes | None = None
) -> plt.Axes:
    """Horizontal bars of mean absolute SHAP value, largest at the top.

    Args:
        table: Output of :func:`readmission_risk.models.explain.importance`.
        title: Chart title.
        ax: Axes to draw on; a new figure is created if omitted.

    Returns:
        The axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 0.4 * len(table) + 1.3))
    pos = range(len(table))
    values = table["mean_abs_shap"].to_numpy()
    ax.barh(pos, values, 0.62, color=SERIES, edgecolor=SURFACE, linewidth=2)
    for i, (v, share) in enumerate(zip(values, table["share"], strict=True)):
        ax.text(v + values.max() * 0.01, i, f"{share:.0%}", va="center",
                fontsize=8.5, color=INK_SECONDARY)  # fmt: skip
    ax.set_yticks(list(pos), table["label"])
    ax.invert_yaxis()
    ax.set_xlim(0, values.max() * 1.12)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean |SHAP value| (log-odds); label = share of total")
    ax.set_title(title)
    return ax


def plot_dependence(
    table: pd.DataFrame, title: str, xlabel: str, ax: plt.Axes | None = None
) -> plt.Axes:
    """Median SHAP value per level with interquartile whiskers.

    Args:
        table: Output of :func:`readmission_risk.models.explain.dependence_table`.
        title: Chart title.
        xlabel: Label for the levels.
        ax: Axes to draw on; a new figure is created if omitted.

    Returns:
        The axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 4.2))
    pos = np.arange(len(table))
    ax.axhline(0, color=MUTED, linewidth=1, zorder=1)
    ax.vlines(pos, table["q1"], table["q3"], color=SERIES, linewidth=2, alpha=0.5,
              zorder=2)  # fmt: skip
    ax.plot(pos, table["median"], "o", color=SERIES, markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)  # fmt: skip
    ax.set_xticks(
        pos,
        [f"{lv}\nn={n:,}" for lv, n in zip(table["level"], table["n"], strict=True)],
    )
    ax.grid(axis="x", visible=False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("SHAP value (log-odds of readmission)")
    ax.text(1.0, 0, " no effect", transform=ax.get_yaxis_transform(),
            color=INK_SECONDARY, fontsize=8.5, va="center", ha="left")  # fmt: skip
    ax.set_title(title)
    return ax


def plot_group_metric(
    table: pd.DataFrame,
    attribute_labels: dict[str, str],
    overall: float,
    title: str,
    xlabel: str,
    orders: dict[str, list[str]] | None = None,
) -> plt.Figure:
    """One panel per attribute: a metric per group with 95% CI whiskers.

    Groups flagged ``small_group`` get an asterisk and a footnote.

    Args:
        table: Rows of one metric from
            :func:`readmission_risk.models.fairness.fairness_table`.
        attribute_labels: Attribute column -> panel heading.
        overall: Metric for the whole population, drawn as a reference line.
        title: Figure title.
        xlabel: Label for the metric axis.
        orders: Optional fixed group order per attribute (e.g. age bands);
            other attributes are sorted largest group first.

    Returns:
        The figure.
    """
    attributes = list(attribute_labels)
    sizes = [int((table["attribute"] == a).sum()) for a in attributes]
    fig, axes = plt.subplots(
        len(attributes), 1, figsize=(7.5, 0.42 * sum(sizes) + 1.0 * len(attributes)),
        sharex=True, gridspec_kw={"height_ratios": sizes},
    )  # fmt: skip
    for ax, attribute in zip(axes, attributes, strict=True):
        part = table[table["attribute"] == attribute].sort_values("n", ascending=False)
        if orders and attribute in orders:
            rank = {g: i for i, g in enumerate(orders[attribute])}
            part = part.sort_values("group", key=lambda s: s.map(rank))
        labels = [
            f"{g} *" if small else str(g)
            for g, small in zip(part["group"], part["small_group"], strict=True)
        ]
        rates = part.set_index(pd.Index(labels))[["estimate", "ci_low", "ci_high"]]
        plot_rates(
            rates.rename(columns={"estimate": "rate"}),
            title=attribute_labels[attribute],
            overall=overall,
            horizontal=True,
            ax=ax,
        )
        ax.set_xlabel("")
        ax.title.set_fontsize(10.5)
    axes[-1].set_xlabel(xlabel)
    for ax in axes[1:]:
        for text in ax.texts:  # overall label once, on the top panel
            text.set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.text(0.01, -0.01, "* fewer than 50 readmissions in the test set: too small "
             "to draw conclusions. Whiskers: 95% patient-bootstrap interval.",
             fontsize=8, color=INK_SECONDARY, ha="left", va="top")  # fmt: skip
    fig.tight_layout()
    return fig


def save(fig: plt.Figure, path: Path) -> Path:
    """Save a figure as PNG, creating the folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    return path
