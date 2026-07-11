"""Figures for module 18. Run: python module/18-longitudinal-mixed-models/code/figures.py

Writes PNGs to ../assets/. German labels (display), English code.
Requires: statsmodels, matplotlib.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402

from lib.helpers import SEED, load_cohort, load_vitals  # noqa: E402
from lib.plotstyle import EVENT, PRIMARY, SECONDARY, apply_style, save  # noqa: E402

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def build_data() -> pd.DataFrame:
    vitals = load_vitals()
    cohort = load_cohort()[["patient_id", "diabetes"]]
    return vitals.merge(cohort, on="patient_id", how="left")


def fig_spaghetti(data: pd.DataFrame, mixed) -> None:
    """Per-patient MAP trajectories vs. the population-average (fixed-effect) trend."""
    rng = np.random.default_rng(SEED)
    sample_ids = rng.choice(data["patient_id"].unique(), size=60, replace=False)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for pid in sample_ids:
        sub = data[data["patient_id"] == pid].sort_values("tag")
        ax.plot(sub["tag"], sub["map_mmhg"], color=SECONDARY, alpha=0.35, lw=1.1)

    tag_range = np.arange(data["tag"].min(), data["tag"].max() + 1)
    pop_line = mixed.params["Intercept"] + mixed.params["tag"] * tag_range
    ax.plot(tag_range, pop_line, color=PRIMARY, lw=3,
            label="Populationsmittel (Fixed Effect aus dem Mixed Model)")

    ax.set_xlabel("Tag")
    ax.set_ylabel("MAP (mmHg)")
    ax.set_title("Individuelle MAP-Verläufe vs. Populationsmittel\n"
                 "(60 zufällige Patient:innen, wiederholte Messungen)")
    ax.legend(loc="upper right")
    save(fig, ASSETS / "spaghetti_map.png")


def fig_random_intercepts(mixed) -> None:
    """Distribution of the estimated per-patient random intercepts (BLUPs)."""
    re_values = np.array([float(v.iloc[0]) for v in mixed.random_effects.values()])
    group_var = float(mixed.cov_re.iloc[0, 0])

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(re_values, bins=30, color=PRIMARY, alpha=0.85, edgecolor="white")
    ax.axvline(0, color=EVENT, lw=1.6, ls="--", label="Populationsmittel (0)")
    ax.set_xlabel("Individuelle Abweichung vom Populationsmittel (mmHg)")
    ax.set_ylabel("Anzahl Patient:innen")
    ax.set_title("Verteilung der geschätzten Random Intercepts\n"
                 f"(Between-Patient-Varianz = {group_var:.2f}, "
                 f"SD = {np.sqrt(group_var):.2f} mmHg)")
    ax.legend(loc="upper right")
    save(fig, ASSETS / "random_intercepts.png")


def main() -> None:
    apply_style()
    data = build_data()
    mixed = smf.mixedlm("map_mmhg ~ tag", data=data, groups=data["patient_id"]).fit(reml=True)

    fig_spaghetti(data, mixed)
    fig_random_intercepts(mixed)


if __name__ == "__main__":
    main()
