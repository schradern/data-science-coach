"""Figure for module 34: AUC vs. calibration (Brier score) under a simulated
population shift during silent deployment.

The simulation is IMPORTED from code/python.py, never re-implemented. It used
to be a copy, and when python.py was corrected to score the drift months out of
sample (resampling only the held-out test set), the copy here was not — so the
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
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import brier_score_loss, roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from lib.helpers import SEED, load_cohort  # noqa: E402
from lib.plotstyle import EVENT, PRIMARY, apply_style, save  # noqa: E402

# `python.py` is not an importable identifier — load it by path.
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "modul33_lektion", Path(__file__).resolve().parent / "python.py")
_lektion = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lektion)
simulate_drift = _lektion.simulate_drift

ASSETS = Path(__file__).resolve().parent.parent / "assets"
# The simulation's parameters (FEATURES, N_MONTHS, …) belong to `simulate_drift`
# in python.py. Restating them here would be a copy that nothing keeps in sync.


def fig_silent_deployment_drift(drift: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.6))
    fig.subplots_adjust(top=0.78, bottom=0.3, wspace=0.32)

    ax1.plot(drift["monat"], drift["auc"], "o-", color=PRIMARY, lw=2)
    ax1.axhline(drift["auc"].iloc[0], color=PRIMARY, ls=":", lw=1, alpha=0.6)
    ax1.set_ylim(0.7, 0.95)
    ax1.set_xlabel("Monat im Silent Deployment")
    ax1.set_ylabel("AUC")
    ax1.set_title("Diskriminierung (AUC)", loc="left")

    ax2.plot(drift["monat"], drift["brier"], "o-", color=EVENT, lw=2)
    ax2.axhline(drift["brier"].iloc[0], color=EVENT, ls=":", lw=1, alpha=0.6)
    ax2.set_xlabel("Monat im Silent Deployment")
    ax2.set_ylabel("Brier Score (niedriger = besser)")
    ax2.set_title("Kalibrierung (Brier Score)", loc="left")

    fig.suptitle("Dieselbe eingefrorene Vorhersage, eine zunehmend kraenkere Population:\n"
                 "AUC bleibt stabil, aber die Kalibrierung verschlechtert sich",
                 fontsize=12.5, fontweight="bold", x=0.02, ha="left", y=0.98)
    fig.text(0.02, 0.02, "Gepunktete Linie = Monat 0 (retrospektive Validierung). Modell fixiert;\n"
                        "nur die SOFA-Score-Verteilung der ausgehaltenen Testkohorte wurde fuer die\n"
                        "Simulation schrittweise verschoben (staerkeres Gewicht auf hoeheren Scores).",
             fontsize=8.6, color="#6B7178")
    save(fig, ASSETS / "silent_deployment_drift.png")


def main() -> None:
    apply_style()
    df = load_cohort()
    drift = simulate_drift(df)
    fig_silent_deployment_drift(drift)


if __name__ == "__main__":
    main()
