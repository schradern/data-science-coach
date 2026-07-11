"""Shared matplotlib style for all course figures (DRY).

Usage:
    from lib.plotstyle import apply_style, save, PRIMARY, SECONDARY, EVENT, PALETTE

    apply_style()
    fig, ax = plt.subplots(...)
    ...
    save(fig, "module/08-eda-visualisierung/assets/my_plot.png")
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# --- Colour palette ---------------------------------------------------------
PRIMARY   = "#2A5C8A"   # clinical dark blue (main colour)
SECONDARY = "#6B7178"   # muted slate for secondary elements
EVENT     = "#B5482E"   # warm contrast red for events / deaths

PALETTE = [PRIMARY, EVENT, "#5B9E6E", SECONDARY, "#C08B3A", "#7B5EA7", "#3A8FA0"]


def apply_style() -> None:
    """Set a clean, clinical matplotlib style (NEJM-inspired).

    Call once at the top of every figure script, before plt.subplots().
    """
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        # clean Helvetica-like sans with full umlaut support
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Nimbus Sans", "Helvetica",
                             "Arial", "DejaVu Sans"],
        "axes.unicode_minus": True,
        # minimal frame: only left/bottom, hairline, muted
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.linewidth": 0.9,
        "axes.edgecolor": "#B8BCC2",
        "axes.axisbelow": True,
        # faint horizontal grid only
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#ECEDEF",
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "axes.prop_cycle": mpl.cycler("color", PALETTE),
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.titlecolor": "#16181C",
        "axes.labelsize": 12,
        "axes.labelcolor": "#3A3E44",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "xtick.color": "#6B7178",
        "ytick.color": "#6B7178",
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "legend.fontsize": 11,
        "lines.linewidth": 1.8,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.25,
        "savefig.facecolor": "white",
        "backend": "Agg",
    })


def save(fig: "plt.Figure", path: str | Path) -> None:
    """Save the figure as PNG (dpi=160, tight, white background) and close it.

    Parent directories are created automatically. Axis labels and titles stay
    in German (display layer); only the code is English.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {target}")
