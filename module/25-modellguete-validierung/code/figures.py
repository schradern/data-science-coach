"""Figures for module 25. Run: python module/25-modellguete-validierung/code/figures.py

Writes PNGs to ../assets/. German labels (display), English code.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV, calibration_curve  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import brier_score_loss  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

from lib.helpers import SEED, load_cohort, load_labs  # noqa: E402
from lib.plotstyle import EVENT, PRIMARY, SECONDARY, apply_style, save  # noqa: E402

ASSETS = Path(__file__).resolve().parent.parent / "assets"

NUMERIC = ["alter", "sofa_score", "crp_mg_l", "bmi", "leukozyten_g_l", "kreatinin_mg_dl", "laktat_mmol_l"]
CATEGORICAL = ["aufnahmegrund", "raucherstatus"]
BINARY = ["diabetes", "hypertonie"]
TARGET = "verstorben_30d"


def build_pipeline() -> Pipeline:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    pre = ColumnTransformer([
        ("num", numeric, NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("bin", "passthrough", BINARY),
    ])
    return Pipeline([
        ("pre", pre),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def net_benefit(y_true, proba, threshold: float) -> float:
    n = len(y_true)
    pos = proba >= threshold
    tp = int(((pos == 1) & (y_true == 1)).sum())
    fp = int(((pos == 1) & (y_true == 0)).sum())
    return tp / n - (threshold / (1 - threshold)) * fp / n


def fig_calibration(y_test, proba) -> None:
    """Calibration curve on RECALIBRATED probabilities.

    `proba` here must already be the CalibratedClassifierCV output (see
    main()) – class_weight="balanced" inflates raw predict_proba() and would
    make this figure show a miscalibrated model regardless of true quality.
    """
    brier = brier_score_loss(y_test, proba)
    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], color=SECONDARY, lw=1, ls="--", label="Perfekte Kalibrierung")
    ax.plot(mean_pred, frac_pos, "o-", color=PRIMARY, lw=1.8,
            label=f"Logistisches Modell (rekalibriert)\n(Brier Score = {brier:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Vorhergesagte Wahrscheinlichkeit")
    ax.set_ylabel("Beobachteter Ereignisanteil")
    ax.set_title("Kalibrierungskurve: Modell (nach Rekalibrierung) vs. Ideal")
    ax.legend(loc="upper left")
    save(fig, ASSETS / "kalibrierung.png")


def fig_decision_curve(y_test, proba) -> None:
    thresholds = np.linspace(0.01, 0.50, 200)
    base_rate = float(np.mean(y_test))

    nb_model = [net_benefit(y_test, proba, t) for t in thresholds]
    nb_all   = [base_rate - (t / (1 - t)) * (1 - base_rate) for t in thresholds]
    nb_none  = [0.0] * len(thresholds)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(thresholds, nb_model, color=PRIMARY,    lw=2,   label="Logistisches Modell")
    ax.plot(thresholds, nb_all,   color=EVENT,      lw=1.5, ls="-.", label="Alle behandeln")
    ax.plot(thresholds, nb_none,  color=SECONDARY,  lw=1.2, ls="--",  label="Niemanden behandeln")
    ax.axhline(0, color="#CCCCCC", lw=0.8)
    ax.set_xlim(0.01, 0.50)
    ax.set_ylim(-0.05, None)
    ax.set_xlabel("Entscheidungsschwelle")
    ax.set_ylabel("Nettovorteil (Net Benefit)")
    ax.set_title("Decision-Curve-Analyse: Klinischer Nutzen des Modells (rekalibriert)")
    ax.legend(loc="upper right")
    save(fig, ASSETS / "decision_curve.png")


def fig_roc_vs_pr_curve(y_test, proba) -> None:
    from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
    
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc = roc_auc_score(y_test, proba)
    
    precision, recall, _ = precision_recall_curve(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    
    # ROC Curve
    ax1.plot(fpr, tpr, color=PRIMARY, lw=2.2, label=f"ROC-Kurve (AUC = {roc_auc:.3f})")
    ax1.plot([0, 1], [0, 1], color=SECONDARY, lw=1.2, ls="--", label="Zufall (AUC = 0.500)")
    ax1.fill_between(fpr, tpr, alpha=0.10, color=PRIMARY)
    ax1.set_xlabel("1 - Spezifität (FPR)")
    ax1.set_ylabel("Sensitivität (TPR)")
    ax1.set_title("ROC-Kurve: Gute Gesamtleistung")
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # PR Curve
    ax2.plot(recall, precision, color=EVENT, lw=2.2, label=f"PR-Kurve (AUC/AP = {pr_auc:.3f})")
    base_rate = float(np.mean(y_test))
    ax2.axhline(base_rate, color=SECONDARY, lw=1.2, ls="--", label=f"Zufall / Prävalenz (AP = {base_rate:.3f})")
    ax2.fill_between(recall, precision, alpha=0.10, color=EVENT)
    ax2.set_xlabel("Sensitivität (Recall)")
    ax2.set_ylabel("Präzision (PPV)")
    ax2.set_title("Precision-Recall-Kurve: Demaskiert schlechte PPV bei seltenem Event")
    ax2.legend(loc="lower left")
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    save(fig, ASSETS / "roc_vs_pr_curve.png")


def main() -> None:
    apply_style()

    df = load_cohort().merge(load_labs(), on="patient_id", how="left")
    X = df[NUMERIC + CATEGORICAL + BINARY]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED
    )
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]

    # Ranking-based figure (ROC/PR): unaffected by class_weight="balanced",
    # use the raw pipeline's scores.
    fig_roc_vs_pr_curve(y_test.values, proba)

    # Absolute-probability figures (calibration, DCA): class_weight="balanced"
    # inflates predict_proba(), so recalibrate first (see code/python.py for
    # the full explanation and the before/after numbers).
    calibrated = CalibratedClassifierCV(build_pipeline(), method="sigmoid", cv=5)
    calibrated.fit(X_train, y_train)
    proba_cal = calibrated.predict_proba(X_test)[:, 1]
    fig_calibration(y_test.values, proba_cal)
    fig_decision_curve(y_test.values, proba_cal)


if __name__ == "__main__":
    main()
