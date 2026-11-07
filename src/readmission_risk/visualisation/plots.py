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
import pandas as pd
from matplotlib.ticker import PercentFormatter

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = "#2a78d6"  # single-series colour (categorical slot 1)


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


def save(fig: plt.Figure, path: Path) -> Path:
    """Save a figure as PNG, creating the folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    return path
