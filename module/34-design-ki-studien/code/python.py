"""Module 34 - why silent deployment catches what a one-off retrospective
validation misses: freeze a model, then let the true risk drift while the
patients' recorded features look the same ("bei gleichem Merkmalsprofil steigt
das Sterberisiko"), which is exactly the kind of drift a real silent-deployment
phase monitors for.

The drift is a genuine change of P(Y|X): month by month the log-odds of death
rise by a fixed offset for EVERY patient profile (e.g. a more virulent pathogen,
a changed referral pattern -- factors the model does not see). It is simulated
deterministically by up-weighting the observed deaths of the held-out test set:
giving every event the weight exp(delta) multiplies the event odds at each X by
exp(delta). No random resampling, so Python and R produce the same numbers.

Run: python module/34-design-ki-studien/code/python.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from lib.helpers import load_cohort  # noqa: E402

FEATURES = ["alter", "sofa_score", "crp_mg_l", "diabetes"]
TARGET = "verstorben_30d"
# Deterministic split both languages can reproduce without sharing an RNG:
# every third patient (patient_id divisible by 3) is held out for validation.
VALIDATION_EVERY = 3
N_MONTHS = 6
LOGODDS_SHIFT_PER_MONTH = 0.15  # true log-odds of death rise by 0.15 per month at every X


def checklist() -> pd.DataFrame:
    return pd.DataFrame([
        ("Retrospective validation", "AUC, calibration, subgroups", "Historical bias"),
        ("Silent deployment", "Prospective performance, latency, missing inputs", "No patient impact yet"),
        ("Clinical trial", "Patient outcome, process metric, safety endpoint", "Intervention risk"),
        ("Post-market monitoring", "Drift, alert burden, override rate", "Model decay"),
    ], columns=["phase", "primary evidence", "main risk"])


def fit_frozen_model(df: pd.DataFrame):
    """Unpenalised logistic regression on the development patients (same as R's glm)."""
    is_test = df["patient_id"] % VALIDATION_EVERY == 0
    dev, test = df[~is_test], df[is_test]
    X_dev = sm.add_constant(dev[FEATURES].astype(float))
    model = sm.GLM(dev[TARGET], X_dev, family=sm.families.Binomial()).fit()
    return model, test


def weighted_logit(y: np.ndarray, X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Coefficients of a weighted logistic regression (point estimates only)."""
    fit = sm.GLM(y, X, family=sm.families.Binomial(), freq_weights=w).fit()
    return np.asarray(fit.params)


def simulate_drift(df: pd.DataFrame) -> pd.DataFrame:
    """Score one frozen model on the held-out patients while the true baseline
    risk drifts upward month by month (P(Y|X) changes, the model does not)."""
    model, test = fit_frozen_model(df)
    X_test = sm.add_constant(test[FEATURES].astype(float))
    p = np.asarray(model.predict(X_test))
    lp = np.log(p / (1 - p))
    y = test[TARGET].to_numpy()
    X_slope = sm.add_constant(lp)

    rows = []
    for month in range(N_MONTHS + 1):
        delta = LOGODDS_SHIFT_PER_MONTH * month
        w = np.where(y == 1, np.exp(delta), 1.0)
        observed = float(np.sum(w * y) / np.sum(w))
        predicted = float(np.sum(w * p) / np.sum(w))
        brier = float(np.sum(w * (p - y) ** 2) / np.sum(w))
        slope = float(weighted_logit(y, X_slope, w)[1])
        # With weights constant within each class, the weighted AUC equals the
        # unweighted one: a pure baseline-risk shift leaves the ranking untouched.
        rows.append({
            "month": month,
            "observed": observed,
            "predicted": predicted,
            "oe": observed / predicted,
            "slope": slope,
            "auc": float(roc_auc_score(y, p, sample_weight=w)),
            "brier": brier,
        })
    return pd.DataFrame(rows)


def main() -> None:
    print("=== Clinical AI validation ladder ===")
    print(checklist().to_string(index=False))
    print("\nDecision rule: do not move to patient-facing use before silent deployment")
    print("shows stable data flow, calibration, and subgroup performance.")

    df = load_cohort()
    drift = simulate_drift(df)

    print("\n=== Simuliertes Silent Deployment: Grundrisiko steigt, Modell bleibt eingefroren ===")
    n_test = int((df["patient_id"] % VALIDATION_EVERY == 0).sum())
    print(f"(entwickelt an {len(df) - n_test} Patient:innen, bewertet an {n_test} zurueckgehaltenen; "
          f"Log-Odds +{LOGODDS_SHIFT_PER_MONTH} pro Monat)")
    print(drift.round(3).to_string(index=False))

    first, last = drift.iloc[0], drift.iloc[-1]
    print(f"\nAUC von Monat 0 zu Monat {N_MONTHS}: {round(last['auc'] - first['auc'], 3) + 0.0:+.3f}")
    print(f"Kalibrierungs-Steigung von Monat 0 zu Monat {N_MONTHS}: {round(last['slope'] - first['slope'], 3) + 0.0:+.3f}")
    print(f"O:E von Monat 0 zu Monat {N_MONTHS}: {first['oe']:.2f} -> {last['oe']:.2f}")
    print(f"Brier Score von Monat 0 zu Monat {N_MONTHS}: {last['brier'] - first['brier']:+.3f} "
          "(bewegt sich auch mit der Ereignisrate, kein reines Kalibrierungsmass)")
    print("-> Rangfolge (AUC) und Kalibrierungs-Steigung bleiben unveraendert, waehrend die")
    print("   vorhergesagten Risiken immer weiter unter der beobachteten Sterblichkeit liegen")
    print("   (O:E steigt). Eine einmalige retrospektive AUC-Zahl haette das nie gezeigt --")
    print("   genau dafuer beobachtet Silent Deployment die Kalibrierung fortlaufend mit.")


if __name__ == "__main__":
    main()
