"""Figure for module 34: AUC vs. calibration-in-the-large under a simulated
baseline-risk drift during silent deployment.

The simulation is IMPORTED from code/python.py, never re-implemented. It used
to be a copy, and when python.py was corrected to score the drift months out of
sample (resampling only the held-out test set), the copy here was not – so the
shipped figure silently kept the in-sample methodology the fix had removed.
One implementation, one truth.

Run: python module/34-design-ki-studien/code/figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from lib.helpers import load_cohort  # noqa: E402
from lib.plotstyle import EVENT, PRIMARY, apply_style, save  # noqa: E402

# `python.py` is not an importable identifier – load it by path.
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "modul34_lektion", Path(__file__).resolve().parent / "python.py")
_lektion = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lektion)
simulate_drift = _lektion.simulate_drift

ASSETS = Path(__file__).resolve().parent.parent / "assets"
# The simulation's parameters (FEATURES, N_MONTHS, …) belong to `simulate_drift`
# in python.py. Restating them here would be a copy that nothing keeps in sync.


def fig_silent_deployment_drift(drift: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.6))
    fig.subplots_adjust(top=0.78, bottom=0.3, wspace=0.32)

    ax1.plot(drift["month"], drift["auc"], "o-", color=PRIMARY, lw=2)
    ax1.set_ylim(0.5, 1.0)
    ax1.set_xlabel("Monat im Silent Deployment")
    ax1.set_ylabel("AUC")
    ax1.set_title("Diskriminierung (AUC)", loc="left")

    ax2.plot(drift["month"], drift["observed"], "o-", color=EVENT, lw=2,
             label="beobachtete Sterblichkeit")
    ax2.plot(drift["month"], drift["predicted"], "o-", color=PRIMARY, lw=2,
             label="mittleres vorhergesagtes Risiko")
    ax2.set_ylim(0, 0.4)
    ax2.set_xlabel("Monat im Silent Deployment")
    ax2.set_ylabel("Anteil")
    ax2.set_title("Kalibrierung im Mittel", loc="left")
    ax2.legend(loc="upper left", fontsize=8.6, frameon=False)
    last = drift.iloc[-1]
    ax2.annotate(f"O:E = {last['oe']:.2f}".replace(".", ","), (last["month"], last["observed"]),
                 textcoords="offset points", xytext=(-62, 6), fontsize=9, color=EVENT)

    fig.suptitle("Dasselbe eingefrorene Modell, steigendes Grundrisiko:\n"
                 "AUC bleibt gleich, die Risiken werden zunehmend unterschaetzt",
                 fontsize=12.5, fontweight="bold", x=0.02, ha="left", y=0.98)
    slope0 = f"{drift['slope'].iloc[0]:.2f}".replace(".", ",")
    slope6 = f"{drift['slope'].iloc[-1]:.2f}".replace(".", ",")
    fig.text(0.02, 0.02, "Validierung an den zurueckgehaltenen Patient:innen. Simuliert: Log-Odds des Todes\n"
                        "steigen jeden Monat um 0,15 bei gleichem Merkmalsprofil. Die Kalibrierungs-Steigung\n"
                        f"bleibt nahe 1 ({slope0} -> {slope6}); nur das mittlere Niveau verschiebt sich.",
             fontsize=8.6, color="#6B7178")
    save(fig, ASSETS / "silent_deployment_drift.png")


def main() -> None:
    apply_style()
    df = load_cohort()
    drift = simulate_drift(df)
    fig_silent_deployment_drift(drift)


if __name__ == "__main__":
    main()
