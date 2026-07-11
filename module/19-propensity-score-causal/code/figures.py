"""Figures for module 19. Run: python module/19-propensity-score-causal/code/figures.py

Writes PNGs to ../assets/. German labels (display), English code.
Requires: scikit-learn, scipy, matplotlib.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from lib.helpers import load_cohort  # noqa: E402
from lib.plotstyle import EVENT, PRIMARY, SECONDARY, apply_style, save  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from python import (  # noqa: E402
    CONTEXT_COVARIATES,
    PS_COVARIATES,
    SMD_THRESHOLD,
    TREATMENT,
    fit_propensity_scores,
    match_with_caliper,
    prepare_data,
    smd,
)

ASSETS = Path(__file__).resolve().parent.parent / "assets"

LABELS = {
    "alter": "Alter",
    "geschlecht_m": "Geschlecht (männlich)",
    "hypertonie": "Hypertonie",
    "raucher_aktiv": "Raucher:in (aktiv)",
    "sofa_score": "SOFA-Score (Mediator)",
    "crp_mg_l": "CRP (mg/l)",
}


def fig_love_plot(df) -> None:
    all_covariates = PS_COVARIATES + CONTEXT_COVARIATES
    before = {v: abs(smd(df[v], df[TREATMENT] == 1)) for v in all_covariates}

    caliper = 0.2 * df["logit_ps"].std()
    matched = match_with_caliper(df, caliper)
    after = {v: abs(smd(matched[v], matched[TREATMENT] == 1)) for v in all_covariates}

    df = df.copy()
    df["iptw"] = np.where(df[TREATMENT] == 1, 1 / df["ps"], 1 / (1 - df["ps"]))
    ipw = {v: abs(smd(df[v], df[TREATMENT] == 1, df["iptw"])) for v in all_covariates}

    # Order: PS-adjusted covariates first (sorted by pre-matching imbalance), mediator/context last.
    ordered = sorted(PS_COVARIATES, key=lambda v: before[v]) + CONTEXT_COVARIATES
    labels = [LABELS[v] for v in ordered]
    y = np.arange(len(ordered))

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    b = [before[v] for v in ordered]
    a = [after[v] for v in ordered]
    w = [ipw[v] for v in ordered]
    ax.hlines(y, [min(x, y2) for x, y2 in zip(b, a)], [max(x, y2) for x, y2 in zip(b, a)],
              color=SECONDARY, lw=1.4, alpha=0.5, zorder=1)
    ax.scatter(b, y, color=EVENT, s=60, label="Vor Matching", zorder=3)
    ax.scatter(a, y, color=PRIMARY, s=60, marker="s", label="Nach Matching (Caliper)", zorder=3)
    ax.scatter(w, y, color="#7B5EA7", s=50, marker="^", label="IPW-gewichtet", zorder=3)
    ax.axvline(SMD_THRESHOLD, color="#9A5B12", ls=":", lw=1.3, zorder=2)
    ax.text(SMD_THRESHOLD + 0.01, 0.96, "Balance-Schwelle (SMD = 0,10)",
            color="#9A5B12", fontsize=9.5, transform=ax.get_xaxis_transform(),
            va="top")
    ax.axhline(len(PS_COVARIATES) - 0.5, color="#D8DADD", lw=1.0, ls="-")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Absolute standardisierte Mittelwertdifferenz |SMD|")
    ax.set_title("Kovariaten-Balance vor/nach Propensity-Score-Matching\n"
                 "(Exposition: Diabetes; unterste 2 Zeilen = Mediator/Kontext, nicht adjustiert)")
    ax.legend(loc="lower right", fontsize=9.5)
    save(fig, ASSETS / "love_plot_smd.png")


def fig_ps_overlap(df) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bins = np.linspace(0, df["ps"].max() * 1.05, 30)
    ax.hist(df.loc[df[TREATMENT] == 0, "ps"], bins=bins, color=SECONDARY, alpha=0.6,
            label=f"Kontrolle (kein Diabetes, n={int((df[TREATMENT] == 0).sum())})", density=True)
    ax.hist(df.loc[df[TREATMENT] == 1, "ps"], bins=bins, color=EVENT, alpha=0.6,
            label=f"Diabetes (n={int((df[TREATMENT] == 1).sum())})", density=True)
    ax.set_xlabel("Geschätzter Propensity Score")
    ax.set_ylabel("Dichte")
    ax.set_title("Overlap der Propensity Scores vor dem Matching")
    ax.legend(loc="upper right")
    save(fig, ASSETS / "ps_overlap.png")


def main() -> None:
    apply_style()
    df = prepare_data()
    df = fit_propensity_scores(df)
    fig_ps_overlap(df)
    fig_love_plot(df)


if __name__ == "__main__":
    main()
