"""Figures for the report: confusion matrices and a reliability diagram (matplotlib, PNG)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8c8b86"
GRID = "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SEQ = LinearSegmentedColormap.from_list(
    "blues", ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)


def _style():
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": GRID,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
        }
    )


def confusion_grid(panels: list[tuple[str, np.ndarray]], labels: list[str], path, xlabel: str, ylabel: str):
    """One heatmap per model, rows normalised to recall, each cell labelled with its share."""
    _style()
    fig, axes = plt.subplots(1, len(panels), figsize=(4.1 * len(panels), 4.2), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (title, cm) in zip(axes, panels, strict=True):
        cm = np.asarray(cm, dtype=float)
        rows = cm.sum(1, keepdims=True)
        share = np.divide(cm, rows, out=np.zeros_like(cm), where=rows > 0)
        ax.imshow(share, cmap=SEQ, vmin=0, vmax=1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if rows[i, 0] == 0:
                    continue
                v = share[i, j]
                ax.text(
                    j,
                    i,
                    f"{v * 100:.0f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=SURFACE if v > 0.55 else (INK if v >= 0.005 else MUTED),
                )
        ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_title(title, fontsize=11, color=INK, loc="left", pad=8)
        ax.set_xlabel(xlabel)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        ax.set_xticks(np.arange(-0.5, len(labels)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(labels)), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.tick_params(which="minor", length=0)
    axes[0].set_ylabel(ylabel)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def reliability_plot(
    before: list[dict], after: list[dict], names: tuple[str, str], path, xlabel: str, ylabel: str
):
    _style()
    fig, ax = plt.subplots(figsize=(5.2, 4.6), constrained_layout=True)
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
    for bins, color, name in ((before, ORANGE, names[0]), (after, BLUE, names[1])):
        x = [b["conf"] for b in bins]
        y = [b["acc"] for b in bins]
        sizes = [max(16, min(120, b["n"] / 4)) for b in bins]
        ax.plot(x, y, color=color, linewidth=2, label=name, zorder=2)
        ax.scatter(x, y, s=sizes, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(color=GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="upper left", labelcolor=INK_2)
    fig.savefig(path, dpi=160)
    plt.close(fig)
