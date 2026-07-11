"""Figures for module 28. Run: python module/28-survival-ml/code/figures.py

Writes PNGs to ../assets/. German labels (display), English code.
Requires: scikit-learn, lifelines, matplotlib.
Optional: scikit-survival (sksurv) — falls back gracefully.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lifelines import CoxPHFitter, KaplanMeierFitter  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from lib.helpers import SEED, load_cohort, load_labs  # noqa: E402
from lib.plotstyle import EVENT, PALETTE, PRIMARY, SECONDARY, apply_style, save  # noqa: E402

ASSETS = Path(__file__).resolve().parent.parent / "assets"

NUMERIC = ["alter", "sofa_score", "crp_mg_l", "bmi", "leukozyten_g_l",
           "kreatinin_mg_dl", "laktat_mmol_l"]
BINARY = ["diabetes", "hypertonie"]
FEATURES = NUMERIC + BINARY
TARGET_EVENT = "status"
TARGET_TIME = "fu_zeit_tage"
TIMES = [7, 14, 21, 28]


def prepare_data():
    df = load_cohort().merge(load_labs(), on="patient_id", how="left")
    X = df[FEATURES].copy()
    events = df[TARGET_EVENT].astype(bool).values
    times = df[TARGET_TIME].astype(float).values
    return X, events, times, df


def impute_scale(X_train, X_test):
    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(imp.fit_transform(X_train))
    X_te = scaler.transform(imp.transform(X_test))
    return X_tr, X_te, imp, scaler


def cox_risk_scores(X_tr_np, X_te_np, events_train, times_train):
    """Fit CoxPHFitter and return partial hazard on test set."""
    train_df = pd.DataFrame(X_tr_np, columns=FEATURES)
    train_df[TARGET_TIME] = times_train
    train_df[TARGET_EVENT] = events_train.astype(int)
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(train_df, duration_col=TARGET_TIME, event_col=TARGET_EVENT)
    test_df = pd.DataFrame(X_te_np, columns=FEATURES)
    return cph.predict_partial_hazard(test_df).values


def make_survival_array(events, times):
    dtype = np.dtype([("event", "?"), ("time", "<f8")])
    arr = np.empty(len(events), dtype=dtype)
    arr["event"] = events
    arr["time"] = times
    return arr


def fig_rsf_vs_cox(X_tr_np, X_te_np, events_train, events_test,
                   times_train, times_test) -> None:
    """Time-dependent AUC comparison (RSF vs. Cox) or fallback bar chart."""
    fig, ax = plt.subplots(figsize=(7, 4))

    cox_scores = cox_risk_scores(X_tr_np, X_te_np, events_train, times_train)

    try:
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.metrics import cumulative_dynamic_auc

        surv_train = make_survival_array(events_train, times_train)
        surv_test = make_survival_array(events_test, times_test)

        rsf = RandomSurvivalForest(n_estimators=200, min_samples_leaf=10,
                                   random_state=SEED, n_jobs=-1)
        rsf.fit(X_tr_np, surv_train)
        chf_funcs = rsf.predict_cumulative_hazard_function(X_te_np, return_array=False)
        rsf_scores = np.array([fn(28) for fn in chf_funcs])

        # cumulative_dynamic_auc needs the FULL test survival array + full
        # score array in a single call (it builds the risk set per time point
        # internally). Masking surv_test/scores per-t corrupts the array's
        # time range and raises "times must be within follow-up time of test
        # data" for later time points. Requested times must be strictly below
        # the largest observed test follow-up time.
        valid_times = [t for t in TIMES if t < times_test.max()]
        auc_rsf_list, _ = cumulative_dynamic_auc(
            surv_train, surv_test, rsf_scores, valid_times)
        auc_cox_list, _ = cumulative_dynamic_auc(
            surv_train, surv_test, cox_scores, valid_times)

        x = np.arange(len(valid_times))
        width = 0.35
        ax.bar(x - width / 2, auc_rsf_list, width, color=PRIMARY,
               label="Random Survival Forest")
        ax.bar(x + width / 2, auc_cox_list, width, color=SECONDARY,
               label="Cox-Proportional Hazards")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Tag {t}" for t in valid_times])
        ax.set_title("Zeitabhängige AUC: RSF vs. Cox-Baseline")

    except ImportError as exc:
        # scikit-survival is a required dependency for this figure — the
        # RSF-vs-Cox time-dependent AUC comparison is the whole point of the
        # chart, so silently degrading to a different chart (with a mismatched
        # caption) is worse than failing loudly. Install scikit-survival and
        # re-run instead of relying on a fallback here.
        plt.close(fig)
        raise SystemExit(
            "scikit-survival (sksurv) is required to generate rsf_vs_cox.png "
            "— install it with `pip install scikit-survival` and re-run. "
            "This figure specifically compares RSF vs. Cox time-dependent AUC; "
            "there is no meaningful fallback chart for that comparison."
        ) from exc

    ax.set_ylim(0, 1)
    ax.axhline(0.5, color=SECONDARY, lw=0.8, ls="--", label="Zufalls-AUC")
    ax.set_ylabel("AUC")
    # AUC values sit high (~0.8-0.9) in this cohort, so the legend goes in the
    # empty lower-right area instead of the default placement (which overlaps
    # the bars).
    ax.legend(loc="lower right")
    save(fig, ASSETS / "rsf_vs_cox.png")


def fig_risikogruppen_km(risk_scores, events_test, times_test) -> None:
    """Kaplan-Meier curves stratified by model-predicted risk tertile."""
    # Split by tertiles of the risk score (not by outcome — that would be leakage).
    tertile_low = np.percentile(risk_scores, 33)
    tertile_high = np.percentile(risk_scores, 67)

    masks = {
        "Niedrig (< 33. Perz.)": risk_scores < tertile_low,
        "Mittel (33.–67. Perz.)": (risk_scores >= tertile_low) & (risk_scores < tertile_high),
        "Hoch (> 67. Perz.)": risk_scores >= tertile_high,
    }
    colors = [PRIMARY, SECONDARY, EVENT]

    fig, ax = plt.subplots(figsize=(7, 4))
    for (label, mask), color in zip(masks.items(), colors):
        if mask.sum() < 5:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(
            durations=times_test[mask],
            event_observed=events_test[mask],
            label=label,
        )
        kmf.plot_survival_function(ax=ax, color=color, ci_show=True)

    ax.set_xlabel("Zeit (Tage)")
    ax.set_ylabel("Überlebenswahrscheinlichkeit")
    ax.set_title("Kaplan-Meier nach Risikogruppe (Modell-Tertile)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left")
    save(fig, ASSETS / "risikogruppen_km.png")


def main() -> None:
    apply_style()
    X, events, times, _ = prepare_data()

    (X_train, X_test, events_train, events_test,
     times_train, times_test) = train_test_split(
        X, events, times, test_size=0.25, stratify=events, random_state=SEED)

    X_tr_np, X_te_np, imp, scaler = impute_scale(X_train, X_test)

    # Figure 1: RSF vs. Cox (or fallback)
    fig_rsf_vs_cox(X_tr_np, X_te_np, events_train, events_test,
                   times_train, times_test)

    # For KM plot: use Cox scores as risk scores (always available).
    cox_scores = cox_risk_scores(X_tr_np, X_te_np, events_train, times_train)

    # If sksurv available, use RSF scores for KM; else use Cox.
    try:
        from sksurv.ensemble import RandomSurvivalForest
        surv_train = make_survival_array(events_train, times_train)
        rsf = RandomSurvivalForest(n_estimators=200, min_samples_leaf=10,
                                   random_state=SEED, n_jobs=-1)
        rsf.fit(X_tr_np, surv_train)
        chf_funcs = rsf.predict_cumulative_hazard_function(X_te_np, return_array=False)
        risk_scores_km = np.array([fn(28) for fn in chf_funcs])
    except ImportError:
        risk_scores_km = cox_scores

    # Figure 2: KM by risk group
    fig_risikogruppen_km(risk_scores_km, events_test, times_test)


if __name__ == "__main__":
    main()
