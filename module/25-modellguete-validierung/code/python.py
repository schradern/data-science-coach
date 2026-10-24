"""Module 25 – Model quality and clinical validation.

Runs standalone from the project root:
    python module/25-modellguete-validierung/code/python.py

Data: read from data/ (committed with the repo); if that folder is
missing, the same files are fetched from the published URL.
Only scikit-learn / scipy / statsmodels / numpy are required.

IMPORTANT – calibration and class_weight="balanced":
Module 23/24 use class_weight="balanced" so the model does not ignore the
minority class when RANKING patients (AUC, thresholds). But re-weighting the
loss function distorts predict_proba(): the model was trained as if deaths
were as common as survivals, so its probabilities systematically overshoot
the true risk (see the CITL demo below). Ranking (AUC/PR-AUC) is a monotonic
function of the score, so it is unaffected – but Brier score, the
calibration curve, calibration-in-the-large/slope, and Decision-Curve
Analysis all consume the probabilities *as absolute numbers*, so we
recalibrate with CalibratedClassifierCV before computing any of them.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from scipy.special import logit  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV, calibration_curve  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

from lib.helpers import SEED, load_cohort, load_labs  # noqa: E402

NUMERIC = ["alter", "sofa_score", "crp_mg_l", "bmi", "leukozyten_g_l", "kreatinin_mg_dl", "laktat_mmol_l"]
CATEGORICAL = ["aufnahmegrund", "raucherstatus"]
BINARY = ["diabetes", "hypertonie"]
TARGET = "verstorben_30d"


def build_pipeline() -> Pipeline:
    """Logistic regression pipeline with imputation, scaling, encoding."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = OneHotEncoder(handle_unknown="ignore")
    pre = ColumnTransformer([
        ("num", numeric, NUMERIC),
        ("cat", categorical, CATEGORICAL),
        ("bin", "passthrough", BINARY),
    ])
    return Pipeline([
        ("pre", pre),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)),
    ])


def bootstrap_metric_ci(y_true, proba, metric_fn, n_boot: int = 2000,
                        seed: int = SEED) -> tuple[float, float]:
    """95%-bootstrap percentile CI for a ranking metric (AUC or PR-AUC).

    Resamples (y, proba) pairs with replacement. On a ~125-patient test set
    with only ~19 events a single point estimate is far too confident, so we
    report the interval alongside it (as modules 32 and 33 do).
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true)
    proba = np.asarray(proba)
    n = len(y)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        y_b, p_b = y[idx], proba[idx]
        if len(np.unique(y_b)) < 2:
            continue  # a resample with only one class has no defined metric
        boot.append(metric_fn(y_b, p_b))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def net_benefit(y_true, proba, threshold: float) -> float:
    """Net benefit for a single decision threshold p_t.

    NB = TP/N - (p_t / (1 - p_t)) * FP/N
    """
    n = len(y_true)
    predicted_pos = proba >= threshold
    tp = int(((predicted_pos == 1) & (y_true == 1)).sum())
    fp = int(((predicted_pos == 1) & (y_true == 0)).sum())
    return tp / n - (threshold / (1 - threshold)) * fp / n


def bootstrap_optimism(X, y, pipeline, n_boot: int = 200, seed: int = SEED):
    """Bootstrap optimism correction for AUC (Harrell method).

    Returns (auc_apparent, optimism, auc_corrected).
    """
    rng = np.random.default_rng(seed)
    pipeline.fit(X, y)
    auc_apparent = roc_auc_score(y, pipeline.predict_proba(X)[:, 1])

    optimisms = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        X_b = X.iloc[idx].reset_index(drop=True)
        y_b = y.iloc[idx].reset_index(drop=True)
        if y_b.nunique() < 2:
            continue
        try:
            pipeline.fit(X_b, y_b)
            auc_boot   = roc_auc_score(y_b, pipeline.predict_proba(X_b)[:, 1])
            auc_orig_b = roc_auc_score(y,   pipeline.predict_proba(X)[:, 1])
            optimisms.append(auc_boot - auc_orig_b)
        except Exception:
            continue
    optimism = float(np.mean(optimisms))
    return auc_apparent, optimism, auc_apparent - optimism


def calibration_stats(y_true, proba):
    """Compute calibration-in-the-large (intercept) and calibration slope.

    Both are logistic recalibration models on logit(proba) (Van Calster et al.,
    2016; Steyerberg's `val.prob`):

    - Calibration-in-the-large: intercept of a GLM where log_odds enters as an
      OFFSET (coefficient fixed at 1) – "if the model's ranking/shape is taken
      as given, is the overall predicted risk level too high or too low?"
      0 = on average correct, negative = model overestimates risk, positive =
      model underestimates risk.
    - Calibration slope: coefficient of log_odds as a covariate – 1 = ideal,
      <1 = predictions too extreme (overfit), >1 = predictions too
      conservative (underfit / need sharpening).

    NOTE ON A COMMON BUG: fitting `sm.Logit(y_true, ones)` (intercept-only,
    with NO reference to `proba` at all) always returns logit(mean(y_true)),
    completely independent of the model's predictions. It looks plausible
    (some number comes out) but silently ignores whether the model is any
    good – it would report the exact same "calibration" for a well-calibrated
    model and a badly miscalibrated one, as long as both target the same
    mean event rate. The offset formulation below actually depends on `proba`.
    """
    log_odds = logit(np.clip(proba, 1e-6, 1 - 1e-6))
    # Intercept only, log_odds as OFFSET (slope fixed at 1) -> calibration-in-the-large
    citl_model = sm.GLM(y_true, np.ones(len(y_true)),
                        family=sm.families.Binomial(), offset=log_odds).fit()
    citl = float(citl_model.params[0])
    # log_odds as covariate (slope estimated freely) -> calibration slope
    slope_model = sm.Logit(y_true, sm.add_constant(log_odds)).fit(disp=False)
    slope = float(slope_model.params[1])
    return citl, slope


def main() -> None:
    df = load_cohort().merge(load_labs(), on="patient_id", how="left")
    X = df[NUMERIC + CATEGORICAL + BINARY]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]

    # --- 1) Discrimination ---------------------------------------------------
    # Ranking metrics (AUC, PR-AUC) only depend on the ORDER of scores. A
    # monotone rescaling of fixed scores leaves them unchanged; REFITTING with
    # class_weight="balanced" is a different model and can reorder patients
    # (compared below).
    print("=== 1) Diskriminierung: ROC-AUC und PR-AUC ===")
    auc_roc = roc_auc_score(y_test, proba)
    auc_pr  = average_precision_score(y_test, proba)
    event_rate = y_test.mean()
    n_events = int(y_test.sum())
    # Bootstrap 95%-CIs: on n=125 with ~19 events the point estimate alone is
    # misleadingly precise, so we quote it with an interval everywhere.
    auc_lo, auc_hi = bootstrap_metric_ci(y_test, proba, roc_auc_score)
    pr_lo, pr_hi = bootstrap_metric_ci(y_test, proba, average_precision_score)
    print(f"  Testset:                  n={len(y_test)}, Ereignisse={n_events}")
    print(f"  Ereignisrate im Testset:  {event_rate:.3f}")
    print(f"  ROC-AUC:                  {auc_roc:.3f}  (95%-Bootstrap-KI {auc_lo:.3f}–{auc_hi:.3f})")
    print(f"  PR-AUC:                   {auc_pr:.3f}  (95%-Bootstrap-KI {pr_lo:.3f}–{pr_hi:.3f}, Baseline = {event_rate:.3f})")
    unweighted = build_pipeline().set_params(model__class_weight=None).fit(X_train, y_train)
    auc_unweighted = roc_auc_score(y_test, unweighted.predict_proba(X_test)[:, 1])
    print(f"  ROC-AUC ohne class_weight (neu gefittet): {auc_unweighted:.4f} "
          f"vs. mit: {auc_roc:.4f} – anderes Modell, andere Rangfolge")

    # --- 1b) Recalibrate before touching absolute probabilities ---------------
    # class_weight="balanced" reweights the loss as if the two classes were
    # equally common, so predict_proba() no longer reflects the true event
    # rate (verified below). CalibratedClassifierCV refits a monotone mapping
    # (Platt scaling here) from raw score to calibrated probability, via
    # internal cross-validation on the TRAINING data only. The recalibrated
    # probabilities preserve the model's ranking (same AUC) but restore
    # realistic absolute risk levels.
    print("\n=== 1b) Rekalibrierung (class_weight='balanced' verzerrt Wahrscheinlichkeiten) ===")
    print(f"  Mittlere vorhergesagte Mortalität (unkalibriert): {proba.mean():.1%}"
          f"   vs. beobachtet: {event_rate:.1%}")
    calibrated = CalibratedClassifierCV(build_pipeline(), method="sigmoid", cv=5)
    calibrated.fit(X_train, y_train)
    proba_cal = calibrated.predict_proba(X_test)[:, 1]
    auc_roc_cal = roc_auc_score(y_test, proba_cal)
    print(f"  Mittlere vorhergesagte Mortalität (rekalibriert): {proba_cal.mean():.1%}"
          f"   vs. beobachtet: {event_rate:.1%}")
    print(f"  ROC-AUC nach Rekalibrierung: {auc_roc_cal:.3f}  (Ranking bleibt praktisch gleich)")
    print("  -> Ab hier verwenden wir proba_cal für Kalibrierung, Brier Score und DCA.")

    # --- 2) Calibration --------------------------------------------------------
    print("\n=== 2) Kalibrierung (auf rekalibrierten Wahrscheinlichkeiten) ===")
    brier_raw = brier_score_loss(y_test, proba)
    brier = brier_score_loss(y_test, proba_cal)
    frac_pos, mean_pred = calibration_curve(y_test, proba_cal, n_bins=10)
    citl, slope = calibration_stats(y_test.values, proba_cal)
    print(f"  Brier Score (unkalibriert):     {brier_raw:.4f}")
    print(f"  Brier Score (rekalibriert):     {brier:.4f}  (0 = perfekt, {event_rate * (1 - event_rate):.3f} = Nullmodell)")
    print(f"  Calibration-in-the-large:       {citl:+.3f}  (0 = gut, neg. = Überschätzung, pos. = Unterschätzung)")
    print(f"  Calibration Slope:              {slope:.3f}  (1 = gut, <1 = überstreckt, >1 = zu konservativ)")

    # --- 3) Decision-Curve Analysis ------------------------------------------
    # DCA thresholds are interpreted as TRUE risk levels ("treat if risk >=
    # p_t"), so this only makes clinical sense with calibrated probabilities.
    print("\n=== 3) Decision-Curve-Analyse (Net Benefit, rekalibrierte Wahrscheinlichkeiten) ===")
    thresholds = np.linspace(0.05, 0.45, 9)
    print(f"  {'Schwelle':>8}  {'Modell NB':>10}  {'Alle NB':>10}  {'Keiner NB':>10}")
    base_rate = float(y_test.mean())
    for t in thresholds:
        nb_model = net_benefit(y_test.values, proba_cal, t)
        nb_all   = base_rate - (t / (1 - t)) * (1 - base_rate)
        print(f"  {t:>8.2f}  {nb_model:>10.4f}  {nb_all:>10.4f}  {0.0:>10.4f}")

    # --- 4) Bootstrap optimism -----------------------------------------------
    print("\n=== 4) Bootstrap-Optimismus-Korrektur (n_boot=100) ===")
    auc_app, optimism, auc_corr = bootstrap_optimism(X_train, y_train,
                                                      build_pipeline(), n_boot=100)
    print(f"  Apparente Trainings-AUC:  {auc_app:.3f}")
    print(f"  Mittlerer Optimismus:     {optimism:.3f}")
    print(f"  Korrigierte AUC:          {auc_corr:.3f}")
    print(f"\n  (Zum Vergleich: echte Test-AUC = {auc_roc:.3f})")
    print("\nKein Modell ohne Kalibrierung und DCA ins Klinikum.")


if __name__ == "__main__":
    main()
