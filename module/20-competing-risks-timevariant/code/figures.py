"""Figures for module 20. Run: python module/20-competing-risks-timevariant/code/figures.py

Writes PNGs to ../assets/. German labels (display), English code.
Requires: lifelines, matplotlib.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from lifelines import KaplanMeierFitter  # noqa: E402

from lib.plotstyle import EVENT, PRIMARY, SECONDARY, apply_style, save  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from python import ADMIN_HORIZON, aalen_johansen_cif, build_competing_risk_data  # noqa: E402

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def fig_cif_vs_naive_km(df) -> None:
    # Exact Aalen-Johansen (no tie-jitter), starting at 0 on day 0.
    cif = aalen_johansen_cif(df["event_time"], df["event_state"], cause=1)
    cif = pd.concat([pd.Series({0.0: 0.0}), cif]).loc[:ADMIN_HORIZON]

    kmf = KaplanMeierFitter()
    kmf.fit(df["event_time"], event_observed=(df["event_state"] == 1))
    naive_cif = (1 - kmf.survival_function_).loc[:ADMIN_HORIZON]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.step(naive_cif.index, naive_cif.values.ravel(), where="post", color=EVENT, lw=2,
            label="1 − Kaplan-Meier (ignoriert Entlassung als Konkurrenzrisiko)")
    ax.step(cif.index, cif.values,
            where="post", color=PRIMARY, lw=2.4,
            label="Cumulative Incidence Function (korrekt, Aalen-Johansen)")
    ax.set_xlabel("Tage seit Aufnahme")
    ax.set_ylabel("Kumulative Inzidenz Tod")
    ax.set_ylim(0, 0.40)
    ax.set_title("1 − Kaplan-Meier überschätzt das Sterberisiko bei konkurrierenden Risiken")
    ax.legend(loc="upper left", fontsize=9.5)
    save(fig, ASSETS / "cif_vs_naive_km.png")


def fig_state_distribution(df) -> None:
    counts = df["event_state"].map({0: "Zensiert", 1: "Verstorben", 2: "Entlassen"}).value_counts()
    order = ["Entlassen", "Verstorben", "Zensiert"]
    counts = counts.reindex(order)
    colors = [SECONDARY, EVENT, "#C08B3A"]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(counts.index, counts.values, color=colors, width=0.55)
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 5, f"{int(v)}", ha="center", fontweight="bold")
    ax.set_ylabel("Anzahl Patient:innen")
    ax.set_title(f"Endzustände in der Kohorte (n={len(df)})")
    save(fig, ASSETS / "event_states.png")


def main() -> None:
    apply_style()
    df = build_competing_risk_data()
    fig_cif_vs_naive_km(df)
    fig_state_distribution(df)


if __name__ == "__main__":
    main()
