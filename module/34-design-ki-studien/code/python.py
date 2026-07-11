"""Module 34 - why silent deployment catches what a one-off retrospective
validation misses: simulate the shared cohort's model performance under a
gradual population shift ("die Intensivstation wird im Verlauf kraenker"),
which is exactly the kind of drift a real silent-deployment phase monitors for.

Run: python module/34-design-ki-studien/code/python.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import brier_score_loss, roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from lib.helpers import SEED, load_cohort  # noqa: E402

FEATURES = ["alter", "sofa_score", "crp_mg_l", "diabetes"]
N_MONTHS = 6
N_RESAMPLE = 3000
N_REPEATS = 25


def checklist() -> pd.DataFrame:
    return pd.DataFrame([
        ("Retrospective validation", "AUC, calibration, subgroups", "Historical bias"),
        ("Silent deployment", "Prospective performance, latency, missing inputs", "No patient impact yet"),
        ("Clinical trial", "Patient outcome, process metric, safety endpoint", "Intervention risk"),
        ("Post-market monitoring", "Drift, alert burden, override rate", "Model decay"),
    ], columns=["phase", "primary evidence", "main risk"])


def simulate_drift(df: pd.DataFrame) -> pd.DataFrame:
    """Freeze a model trained on the current cohort, then re-score it on
    resamples that skew toward higher SOFA scores month by month -- a sicker
    population than the one the model was validated on (spectrum shift)."""
    rng = np.random.default_rng(SEED)
    X, y = df[FEATURES], df["verstorben_30d"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y)
    scaler = StandardScaler().fit(X_train)
    model = LogisticRegression(max_iter=1000).fit(scaler.transform(X_train), y_train)

    proba_test = model.predict_proba(scaler.transform(X_test))[:, 1]
    rows = [{
        "monat": 0,
        "auc": roc_auc_score(y_test, proba_test),
        "brier": brier_score_loss(y_test, proba_test),
        "mean_sofa": X_test["sofa_score"].mean(),
        "label": "Retrospektive Validierung",
    }]

    # Every drift month must be scored OUT OF SAMPLE, exactly like month 0.
    # Resample only from the held-out TEST set (never the full cohort, which is
    # ~70% training patients the frozen model already saw). Otherwise the
    # drift-month metrics would be largely in-sample and optimistically biased
    # against the out-of-sample baseline — confounding the whole demo.
    sofa_test = X_test["sofa_score"]
    sofa_mean, sofa_std = sofa_test.mean(), sofa_test.std()
    n_test = len(X_test)
    for month in range(1, N_MONTHS + 1):
        shift = 0.15 * month
        weight = np.exp(shift * (sofa_test - sofa_mean) / sofa_std)
        prob = (weight / weight.sum()).to_numpy()
        aucs, briers, sofas = [], [], []
        for _ in range(N_REPEATS):
            idx = rng.choice(n_test, size=N_RESAMPLE, replace=True, p=prob)
            X_m, y_m = X_test.iloc[idx], y_test.iloc[idx]
            proba_m = model.predict_proba(scaler.transform(X_m))[:, 1]
            aucs.append(roc_auc_score(y_m, proba_m))
            briers.append(brier_score_loss(y_m, proba_m))
            sofas.append(X_m["sofa_score"].mean())
        rows.append({
            "monat": month,
            "auc": float(np.mean(aucs)),
            "brier": float(np.mean(briers)),
            "mean_sofa": float(np.mean(sofas)),
            "label": f"Silent Deployment Monat {month}",
        })
    return pd.DataFrame(rows)


def main() -> None:
    print("=== Clinical AI validation ladder ===")
    print(checklist().to_string(index=False))
    print("\nDecision rule: do not move to patient-facing use before silent deployment")
    print("shows stable data flow, calibration, and subgroup performance.")

    df = load_cohort()
    drift = simulate_drift(df)

    print("\n=== Simuliertes Silent Deployment: Population wird schrittweise kraenker ===")
    print("(dasselbe, eingefrorene Modell -- nur die Population verschiebt sich)")
    print(drift[["monat", "mean_sofa", "auc", "brier"]].round(3).to_string(index=False))

    auc_change = drift["auc"].iloc[-1] - drift["auc"].iloc[0]
    brier_change = drift["brier"].iloc[-1] - drift["brier"].iloc[0]
    print(f"\nAUC von Monat 0 zu Monat {N_MONTHS}: {auc_change:+.3f}")
    print(f"Brier Score von Monat 0 zu Monat {N_MONTHS}: {brier_change:+.3f} "
          f"({brier_change / drift['brier'].iloc[0]:+.0%} relativ)")
    if brier_change > 0.02 and auc_change > -0.02:
        print("-> Die Diskriminierung (AUC) bleibt stabil oder verbessert sich sogar, waehrend sich die")
        print("   Kalibrierung (Brier Score) deutlich verschlechtert: die vorhergesagten Risiken passen")
        print("   nicht mehr zur tatsaechlichen Ereignisrate der (nun kraenkeren) Population. Eine einmalige")
        print("   retrospektive AUC-Zahl haette das nie gezeigt -- genau dafuer beobachtet Silent Deployment")
        print("   die Kalibrierung fortlaufend mit.")
    elif auc_change < -0.02:
        print("-> Sowohl Diskriminierung als auch Kalibrierung verschlechtern sich unter der")
        print("   verschobenen Population.")
    else:
        print("-> In diesem Lauf zeigt sich kein eindeutiger Drift-Effekt -- Ergebnis ist zufallsabhaengig")
        print("   (SEED, Resampling-Staerke); wiederhole mit anderem SEED, um Robustheit zu pruefen.")


if __name__ == "__main__":
    main()
