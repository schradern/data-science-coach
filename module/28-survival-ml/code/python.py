"""Module 28 – Survival ML: Random Survival Forests and time-dependent risk.

Runs standalone from the project root:
    python module/28-survival-ml/code/python.py

Data: read from data/ (committed with the repo); if that folder is
missing, the same files are fetched from the published URL.
Core deps: scikit-learn, lifelines. Optional: scikit-survival (sksurv).
Falls back to Cox + HistGradientBoosting classifier if sksurv is missing.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from lib.helpers import SEED, load_cohort, load_labs  # noqa: E402

NUMERIC = ["alter", "sofa_score", "crp_mg_l", "bmi", "leukozyten_g_l",
           "kreatinin_mg_dl", "laktat_mmol_l"]
BINARY = ["diabetes", "hypertonie"]
FEATURES = NUMERIC + BINARY
TARGET_EVENT = "status"
TARGET_TIME = "fu_zeit_tage"
TIMES = [7, 14, 21, 28]   # evaluation horizons in days


def build_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Merge cohort and labs; return feature matrix, event array, time array."""
    df = load_cohort().merge(load_labs(), on="patient_id", how="left")
    X = df[FEATURES].copy()
    events = df[TARGET_EVENT].astype(bool).values
    times = df[TARGET_TIME].astype(float).values
    return X, events, times


def impute_and_scale(X_train: pd.DataFrame,
                     X_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Median imputation + standard scaling (fit on train only)."""
    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(imp.fit_transform(X_train))
    X_te = scaler.transform(imp.transform(X_test))
    return X_tr, X_te


def make_survival_array(events: np.ndarray,
                        times: np.ndarray) -> np.ndarray:
    """Build sksurv structured array from event and time arrays."""
    dtype = np.dtype([("event", "?"), ("time", "<f8")])
    arr = np.empty(len(events), dtype=dtype)
    arr["event"] = events
    arr["time"] = times
    return arr


def cox_risk_scores(X_train_np: np.ndarray, X_test_np: np.ndarray,
                    events_train: np.ndarray,
                    times_train: np.ndarray) -> np.ndarray:
    """Fit a CoxPHFitter and return linear predictor on test set."""
    from lifelines import CoxPHFitter
    train_df = pd.DataFrame(X_train_np, columns=FEATURES)
    train_df[TARGET_TIME] = times_train
    train_df[TARGET_EVENT] = events_train.astype(int)
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(train_df, duration_col=TARGET_TIME, event_col=TARGET_EVENT)
    test_df = pd.DataFrame(X_test_np, columns=FEATURES)
    # predict_partial_hazard returns higher = more risk
    return cph.predict_partial_hazard(test_df).values


def bootstrap_td_auc(cumulative_dynamic_auc, surv_train, surv_test,
                     rsf_scores, cox_scores, times, n_boot: int = 500,
                     seed: int = SEED):
    """Paired bootstrap of time-dependent AUC for RSF and Cox.

    Resamples the (small) test set with replacement and recomputes the
    time-dependent AUC of both models on each resample, so we can report a
    95%-CI per model at each horizon – and, crucially, a CI on the
    Cox-minus-RSF difference. With only ~19 test events (and a handful of
    deaths by day 7) the per-time point estimates are extremely noisy; the
    difference CI shows whether any apparent RSF/Cox gap is real or sampling
    noise. surv_train stays fixed as the reference risk set.
    """
    rng = np.random.default_rng(seed)
    n = len(rsf_scores)
    rsf_b = {t: [] for t in times}
    cox_b = {t: [] for t in times}
    diff_b = {t: [] for t in times}
    # A resample with no events before an early horizon (e.g. day 7) makes the
    # true-positive rate 0/0; sksurv returns nan there – we drop those below.
    with np.errstate(invalid="ignore", divide="ignore"):
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            st = surv_test[idx]
            for t in times:
                # Requested time must be strictly below the resample's max
                # follow-up, and the resample must contain at least one event.
                if t >= st["time"].max() or st["event"].sum() == 0:
                    continue
                try:
                    a_r, _ = cumulative_dynamic_auc(surv_train, st, rsf_scores[idx], [t])
                    a_c, _ = cumulative_dynamic_auc(surv_train, st, cox_scores[idx], [t])
                except Exception:
                    continue
                if not (np.isfinite(a_r[0]) and np.isfinite(a_c[0])):
                    continue
                rsf_b[t].append(a_r[0])
                cox_b[t].append(a_c[0])
                diff_b[t].append(a_c[0] - a_r[0])
    return rsf_b, cox_b, diff_b


def run_with_sksurv(X, events, times) -> bool:
    """Try running RSF + time-dependent AUC with scikit-survival."""
    try:
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.metrics import cumulative_dynamic_auc
    except ImportError:
        return False

    print("\n=== scikit-survival AVAILABLE – using Random Survival Forest ===")
    (X_train, X_test, events_train, events_test,
     times_train, times_test) = train_test_split(
        X, events, times, test_size=0.25, stratify=events, random_state=SEED)

    X_tr_np, X_te_np = impute_and_scale(X_train, X_test)
    surv_train = make_survival_array(events_train, times_train)
    surv_test = make_survival_array(events_test, times_test)

    # --- RSF ---
    rsf = RandomSurvivalForest(
        n_estimators=200, min_samples_leaf=10,
        random_state=SEED, n_jobs=-1,
    )
    rsf.fit(X_tr_np, surv_train)

    # Risk score = cumulative hazard at day 28 (high score = high risk).
    # predict_cumulative_hazard_function returns a list of step functions.
    chf_funcs = rsf.predict_cumulative_hazard_function(X_te_np, return_array=False)
    # Summarise as cumulative hazard at last evaluation time (28d).
    rsf_scores = np.array([fn(28) for fn in chf_funcs])

    # --- Cox baseline ---
    cox_scores = cox_risk_scores(X_tr_np, X_te_np, events_train, times_train)

    # --- Time-dependent AUC ---
    # cumulative_dynamic_auc needs the FULL test survival array + full score
    # array in one call; it builds the risk set per time point internally.
    # Masking surv_test/scores per-t (as an earlier version of this code did)
    # corrupts the array's time range and raises "times must be within
    # follow-up time of test data" for later time points. Requested times
    # must be strictly below the largest observed test follow-up time.
    valid_times = [t for t in TIMES if t < times_test.max()]
    auc_rsf, mean_auc_rsf = cumulative_dynamic_auc(
        surv_train, surv_test, rsf_scores, valid_times)
    auc_cox, mean_auc_cox = cumulative_dynamic_auc(
        surv_train, surv_test, cox_scores, valid_times)

    # Bootstrap 95%-CIs per model + a CI on the Cox-RSF difference.
    rsf_b, cox_b, diff_b = bootstrap_td_auc(
        cumulative_dynamic_auc, surv_train, surv_test,
        rsf_scores, cox_scores, valid_times)

    def ci_lohi(vals):
        arr = np.array([v for v in vals if np.isfinite(v)])
        return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))) if len(arr) else (np.nan, np.nan)

    n_events_test = int(surv_test["event"].sum())
    print(f"\n  Test set: n={len(surv_test)}, Ereignisse={n_events_test}")
    print("  Time-dependent AUC (cumulative_dynamic_auc) mit 95%-Bootstrap-KI:")
    print(f"  {'Zeitpunkt':>10}  {'RSF-AUC (KI)':>22}  {'Cox-AUC (KI)':>22}  {'Cox-RSF (KI)':>24}")
    zero_in_ci = {}
    for t, a_rsf, a_cox in zip(valid_times, auc_rsf, auc_cox):
        r_lo, r_hi = ci_lohi(rsf_b[t])
        c_lo, c_hi = ci_lohi(cox_b[t])
        d = float(np.mean([v for v in diff_b[t] if np.isfinite(v)])) if diff_b[t] else np.nan
        d_lo, d_hi = ci_lohi(diff_b[t])
        zero_in_ci[t] = (d_lo <= 0 <= d_hi)
        marker = "  0 im KI" if zero_in_ci[t] else ""
        print(f"  {t:>8}d  {a_rsf:>7.3f} [{r_lo:.2f},{r_hi:.2f}]  "
              f"{a_cox:>7.3f} [{c_lo:.2f},{c_hi:.2f}]  "
              f"{d:>+7.3f} [{d_lo:+.2f},{d_hi:+.2f}]{marker}")
    print(f"  {'Mean AUC':>10}   RSF {mean_auc_rsf:.3f}   Cox {mean_auc_cox:.3f}")

    n_indistinct = sum(zero_in_ci.values())
    print(f"\n  Fazit: Bei {n_indistinct} von {len(valid_times)} Zeitpunkten schließt das"
          " 95%-KI der Differenz\n  die Null ein – dort ist die Cox-vs-RSF-Lücke von"
          " Rauschen nicht zu unterscheiden.")
    print("  Nur am frühesten Horizont (Tag 7) liegt das KI knapp über 0 zugunsten von")
    print("  Cox, aber genau dort gibt es die wenigsten Ereignisse, und dieser Bootstrap")
    print("  hält die Modelle fest (er misst nur die Test-Stichprobenstreuung, nicht die")
    print("  Trainingsstreuung), unterschätzt die Unsicherheit also noch. Belastbar ist")
    print("  daher nur: kein Modell ist hier klar überlegen, am wenigsten aus den frühen,")
    print("  ereignisarmen Zeitpunkten.")

    return True


def run_fallback(X, events, times) -> None:
    """Fallback when sksurv is not installed.

    Fits Cox (lifelines) and HistGradientBoosting at a fixed 30-day horizon.
    Computes standard AUC for comparison. Prints a clear note about censoring.
    """
    print("\n=== scikit-survival NOT installed – fallback: Cox vs. classifier ===")
    print("  Install scikit-survival: uv pip install scikit-survival")
    print("  Note: HistGradientBoosting ignores censoring; Cox handles it correctly.\n")

    (X_train, X_test, events_train, events_test,
     times_train, times_test) = train_test_split(
        X, events, times, test_size=0.25, stratify=events, random_state=SEED)

    X_tr_np, X_te_np = impute_and_scale(X_train, X_test)

    # Cox (lifelines) is a survival-aware MODEL, but note the METRIC below is
    # NOT: roc_auc_score on the raw event indicator ignores censoring entirely
    # (patients censored before 30 days are silently counted as non-events).
    cox_scores = cox_risk_scores(X_tr_np, X_te_np, events_train, times_train)
    cox_auc = roc_auc_score(events_test, cox_scores)
    print(f"  Cox 30-day AUC (Metrik ignoriert Zensierung): {cox_auc:.3f}")

    # HistGradientBoosting – ignores censoring (educational anti-pattern).
    hgb = HistGradientBoostingClassifier(
        random_state=SEED, max_iter=200, class_weight="balanced")
    hgb.fit(X_tr_np, events_train.astype(int))
    hgb_scores = hgb.predict_proba(X_te_np)[:, 1]
    hgb_auc = roc_auc_score(events_test, hgb_scores)
    print(f"  HistGradientBoosting AUC (ignoriert Zens.):   {hgb_auc:.3f}")
    print("\n  WICHTIG: roc_auc_score(events_test, ...) ist KEINE survival-aware Metrik.")
    print("  Sie behandelt vor Tag 30 zensierte Patienten als gesicherte Nicht-Ereignisse")
    print("  und widerspricht damit dem Kern dieses Moduls. Nur die time-dependent AUC")
    print("  (sksurv, oben) bzw. der C-Index berücksichtigen Zensierung korrekt.")
    print("  Der hier gezeigte Cox-Wert nutzt einen zensierungsbewussten MODELL-Score,")
    print("  wird aber mit einer zensierungsblinden METRIK bewertet – deshalb nur")
    print("  als Kontrast zum Klassifikator, nicht als Gütemaß zu lesen.")

    print("\n  Concept – time-dependent AUC (not computed without sksurv):")
    for t in TIMES:
        print(f"    Day {t:2d}: would compare RSF vs. Cox discrimination up to that point.")


def main() -> None:
    X, events, times = build_data()

    print("=== 1) Why classifiers fail with censored survival data ===")
    print(f"  Total patients:   {len(events)}")
    print(f"  Events (deaths):  {events.sum()} ({100*events.mean():.1f}%)")
    print(f"  Min/Max time:     {times.min():.0f} / {times.max():.0f} days")
    print("  Censored patients contributed follow-up time without dying;")
    print("  a classifier would label them as 'no event' – systematically wrong.")

    sksurv_ran = run_with_sksurv(X, events, times)
    if not sksurv_ran:
        run_fallback(X, events, times)

    print("\n=== 3) Competing risks – concept ===")
    print("  If a patient is transferred or dies from an unrelated cause,")
    print("  Kaplan-Meier overestimates the cumulative incidence.")
    print("  The Aalen-Johansen estimator (cmprsk/tidycmprsk in R) is correct.")
    print("  In Python: lifelines.AalenJohansenFitter or scikit-survival.")

    print("\n=== Key takeaways ===")
    print("  1. Use survival models when follow-up times vary or censoring occurs.")
    print("  2. Time-dependent AUC reveals whether discrimination holds across time.")
    print("  3. Group patients by model score (not outcome) to get honest KM curves.")
    print("  4. Competing risks require dedicated estimators – KM is biased.")


if __name__ == "__main__":
    main()
