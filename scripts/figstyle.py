"""Shared publication style for all MyerST figures (Nature-style)."""

import matplotlib


def use_style():
    matplotlib.rcParams.update({
        "font.family": "Arial",
        "font.size": 8,
        "axes.linewidth": 0.6,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
    })


def panel_label(ax, letter, x=-0.16, y=1.08):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="left")


def clean_spatial(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_aspect("equal")
    ax.invert_yaxis()


def scalebar(ax, x, y, length_um, label=None, color="black", lw=1.4):
    """Draw a scale bar at data coords (x, y) with length in data units."""
    ax.plot([x, x + length_um], [y, y], color=color, lw=lw,
            solid_capstyle="butt")
    ax.text(x + length_um / 2, y, label or f"{int(length_um)} µm",
            ha="center", va="bottom", fontsize=6.5, color=color)
