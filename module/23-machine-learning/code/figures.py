"""Figures for module 23. Run: python module/23-machine-learning/code/figures.py

Writes PNGs to ../assets/. German labels (display), English code.

Replaces the module-18 figure block that used to live inside the shared
`data/figures.py` script (old "Module 14" section) – module 23 now owns its
figures directly, consistent with modules 24–27.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV, calibration_curve  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from lib.helpers import SEED, load_cohort  # noqa: E402
from lib.plotstyle import EVENT, PRIMARY, SECONDARY, apply_style, save  # noqa: E402

ASSETS = Path(__file__).resolve().parent.parent / "assets"
FEATURES = ["alter", "sofa_score", "crp_mg_l", "diabetes", "is_sepsis", "is_smoker"]


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)),
    ])


def fig_roc(y_test, proba) -> None:
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc_val = roc_auc_score(y_test, proba)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color=PRIMARY, linewidth=2.2,
            label=f"Logistische Regression (AUC = {auc_val:.3f})")
    ax.plot([0, 1], [0, 1], color=SECONDARY, linestyle="--", linewidth=1.0,
            label="Zufallsklassifikator (AUC = 0.500)")
    ax.fill_between(fpr, tpr, alpha=0.10, color=PRIMARY)
    ax.set_xlabel("Falsch-Positiv-Rate (1 − Spezifität)")
    ax.set_ylabel("Richtig-Positiv-Rate (Sensitivität)")
    ax.set_title(f"ROC-Kurve: 30-Tage-Mortalität\n(Testdaten, N = {len(y_test)})")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    save(fig, ASSETS / "roc_kurve.png")


def fig_calibration(y_test, proba_raw, proba_cal) -> None:
    """Calibration curve: class_weight='balanced' distorts predict_proba(), and
    CalibratedClassifierCV fixes it. Shown side by side so the effect is
    visible directly, not just asserted in prose. See Module 25 for the full
    calibration-in-the-large/slope/DCA workflow that builds on this."""
    brier_raw = brier_score_loss(y_test, proba_raw)
    brier_cal = brier_score_loss(y_test, proba_cal)
    frac_raw, mean_raw = calibration_curve(y_test, proba_raw, n_bins=5, strategy="quantile")
    frac_cal, mean_cal = calibration_curve(y_test, proba_cal, n_bins=5, strategy="quantile")

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], color=SECONDARY, linestyle="--", linewidth=1.0,
            label="Perfekte Kalibrierung")
    ax.plot(mean_raw, frac_raw, "o-", color=EVENT, linewidth=2.0, markersize=7,
            label=f"class_weight='balanced', unkalibriert\n(Brier = {brier_raw:.3f}, Ø vorhergesagt {proba_raw.mean():.0%})")
    ax.plot(mean_cal, frac_cal, "o-", color=PRIMARY, linewidth=2.0, markersize=7,
            label=f"nach CalibratedClassifierCV\n(Brier = {brier_cal:.3f}, Ø vorhergesagt {proba_cal.mean():.0%})")
    ax.set_xlabel("Vorhergesagte Wahrscheinlichkeit")
    ax.set_ylabel("Beobachtete Ereignisrate")
    ax.set_title("Kalibrierungskurve: 'balanced' verzerrt, Rekalibrierung korrigiert\n"
                 f"(30-Tage-Mortalität, Testdaten, beobachtet {y_test.mean():.0%})")
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save(fig, ASSETS / "kalibrierung.png")


def main() -> None:
    apply_style()
    df = load_cohort()
    df["is_sepsis"] = (df["aufnahmegrund"] == "Sepsis").astype(int)
    df["is_smoker"] = (df["raucherstatus"] == "aktiv").astype(int)
    X = df[FEATURES]
    y = df["verstorben_30d"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    proba_raw = pipe.predict_proba(X_test)[:, 1]
    fig_roc(y_test, proba_raw)

    calibrated = CalibratedClassifierCV(build_pipeline(), method="sigmoid", cv=5)
    calibrated.fit(X_train, y_train)
    proba_cal = calibrated.predict_proba(X_test)[:, 1]
    fig_calibration(y_test, proba_raw, proba_cal)


if __name__ == "__main__":
    main()
