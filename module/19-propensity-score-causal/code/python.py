"""Module 19 - propensity score matching and weighting.

Uses the shared cohort (`kohorte.csv`). Exposure of interest: `diabetes`.
Outcome: `verstorben_30d`. Per the course DAG (see data/README.md and
module 15): `alter` is a confounder of diabetes -> death and must be
adjusted for; `sofa_score` is a MEDIATOR on the diabetes -> death pathway
(diabetic patients tend to arrive sicker) and must NOT be adjusted for when
estimating the total effect of diabetes - adjusting for it would only give
the direct effect. `geschlecht`, `hypertonie`, `raucherstatus` are baseline
covariates unrelated to diabetes assignment in the data-generating process;
including them in the propensity-score model does not bias the estimate and
can improve precision.

Run from project root:
    python module/19-propensity-score-causal/code/python.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from scipy.spatial.distance import cdist  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from lib.helpers import SEED, load_cohort  # noqa: E402

TREATMENT = "diabetes"
OUTCOME = "verstorben_30d"
# True confounders + precision covariates (NOT the mediator sofa_score).
PS_COVARIATES = ["alter", "geschlecht_m", "hypertonie", "raucher_aktiv"]
# Reported for context only - sofa_score/crp_mg_l are on the causal pathway
# (sofa_score) or unrelated to diabetes (crp_mg_l); we do NOT expect/require
# matching to balance the mediator.
CONTEXT_COVARIATES = ["sofa_score", "crp_mg_l"]
SMD_THRESHOLD = 0.10


def smd(x: pd.Series, treated_mask: pd.Series, weights: pd.Series | None = None) -> float:
    x_arr = x.to_numpy(dtype=float)
    g = treated_mask.to_numpy().astype(bool)
    if weights is None:
        m1, m0 = x_arr[g].mean(), x_arr[~g].mean()
        v1, v0 = x_arr[g].var(ddof=1), x_arr[~g].var(ddof=1)
    else:
        w = weights.to_numpy(dtype=float)
        m1 = np.average(x_arr[g], weights=w[g])
        m0 = np.average(x_arr[~g], weights=w[~g])
        v1 = np.average((x_arr[g] - m1) ** 2, weights=w[g])
        v0 = np.average((x_arr[~g] - m0) ** 2, weights=w[~g])
    return float((m1 - m0) / np.sqrt((v1 + v0) / 2))


def prepare_data() -> pd.DataFrame:
    df = load_cohort().copy()
    df["geschlecht_m"] = (df["geschlecht"] == "maennlich").astype(int)
    df["raucher_aktiv"] = (df["raucherstatus"] == "aktiv").astype(int)
    return df


def fit_propensity_scores(df: pd.DataFrame) -> pd.DataFrame:
    X = df[PS_COVARIATES]
    y = df[TREATMENT]
    # C=np.inf: no penalty, i.e. plain maximum-likelihood logistic regression
    # (matches R's glm() exactly) - scikit-learn's default L2 regularization would
    # otherwise shift the propensity scores slightly and make the two language
    # tracks diverge. (penalty=None does the same but is deprecated since 1.8.)
    ps_model = LogisticRegression(max_iter=2000, C=np.inf).fit(X, y)
    df = df.copy()
    df["ps"] = ps_model.predict_proba(X)[:, 1].clip(1e-6, 1 - 1e-6)
    df["logit_ps"] = np.log(df["ps"] / (1 - df["ps"]))
    return df


def match_with_caliper(df: pd.DataFrame, caliper: float) -> pd.DataFrame:
    """1:1 nearest-neighbor matching on the logit(PS), without replacement,
    dropping any treated patient whose best match exceeds the caliper.

    Returns the matched set with a `pair_id` column identifying each
    treated/control pair. The pair id is required downstream: a matched
    analysis must NOT treat the two members of a pair as independent
    observations - the outcome model needs a cluster-robust SE on `pair_id`.
    """
    treated = df[df[TREATMENT] == 1]
    control = df[df[TREATMENT] == 0]
    dists = cdist(treated[["logit_ps"]], control[["logit_ps"]])

    available = np.ones(len(control), dtype=bool)
    matched_pairs = []
    # Process treated patients from best- to worst-supported (smallest min distance first)
    # so caliper rejections don't starve easy-to-match patients of their best partner.
    order = np.argsort(dists.min(axis=1))
    for i in order:
        d = dists[i].copy()
        d[~available] = np.inf
        j = int(np.argmin(d))
        if d[j] <= caliper:
            matched_pairs.append((treated.index[i], control.index[j]))
            available[j] = False

    treated_idx = [p[0] for p in matched_pairs]
    control_idx = [p[1] for p in matched_pairs]
    pair_ids = list(range(len(matched_pairs)))
    matched = pd.concat([df.loc[treated_idx], df.loc[control_idx]]).copy()
    matched["pair_id"] = pair_ids + pair_ids  # treated block, then control block
    return matched


def main() -> None:
    df = prepare_data()
    n_treated = int(df[TREATMENT].sum())
    n_control = int((1 - df[TREATMENT]).sum())
    print(f"=== Shared cohort: {TREATMENT} as exposure ===")
    print(f"Treated (diabetes=1): {n_treated} | Control (diabetes=0): {n_control}")

    df = fit_propensity_scores(df)
    print("\nPropensity score overlap:")
    print(df.groupby(TREATMENT)["ps"].agg(["min", "median", "max"]).round(3))

    print("\n=== Balance before matching (SMD) ===")
    all_covariates = PS_COVARIATES + CONTEXT_COVARIATES
    before = {v: smd(df[v], df[TREATMENT] == 1) for v in all_covariates}
    for v in all_covariates:
        tag = "" if v in PS_COVARIATES else " [context only - NOT adjusted, see note below]"
        print(f"{v:<14} SMD = {before[v]: .3f}{tag}")

    # --- 1:1 nearest-neighbor matching, no caliper (reproduces the naive approach) ---
    naive_matched = match_with_caliper(df, caliper=np.inf)
    print(f"\n=== 1:1 matching WITHOUT a caliper ({len(naive_matched) // 2} pairs) ===")
    naive_after = {v: smd(naive_matched[v], naive_matched[TREATMENT] == 1) for v in all_covariates}
    for v in PS_COVARIATES:
        flag = " <- exceeds 0.10, add a caliper" if abs(naive_after[v]) > SMD_THRESHOLD else ""
        print(f"{v:<14} SMD before = {before[v]: .3f} | after = {naive_after[v]: .3f}{flag}")

    # --- 1:1 nearest-neighbor matching WITH caliper = 0.2 * SD(logit PS) ---
    caliper = 0.2 * df["logit_ps"].std()
    matched = match_with_caliper(df, caliper=caliper)
    n_pairs = len(matched) // 2
    print(f"\n=== 1:1 matching WITH caliper = 0.2 x SD(logit PS) = {caliper:.3f} "
          f"({n_pairs} of {n_treated} treated matched) ===")
    after = {v: smd(matched[v], matched[TREATMENT] == 1) for v in all_covariates}
    for v in PS_COVARIATES:
        ok = "OK (< 0.10)" if abs(after[v]) <= SMD_THRESHOLD else "STILL IMBALANCED"
        print(f"{v:<14} SMD before = {before[v]: .3f} | after caliper = {after[v]: .3f}  [{ok}]")
    for v in CONTEXT_COVARIATES:
        print(f"{v:<14} SMD before = {before[v]: .3f} | after caliper = {after[v]: .3f}  "
              f"[not a matching target - mediator/unrelated, see note]")
    print("\nNote: sofa_score stays imbalanced even after good confounder balance - that's expected.")
    print("Diabetes causally raises sofa_score (mediator), so matched diabetics are still sicker on")
    print("average. Forcing sofa_score balance would adjust away part of diabetes's real effect.")

    # --- IPW as a second, complementary approach (targets the ATE) ---
    # Unstabilised ATE weights: 1/ps for treated, 1/(1-ps) for controls.
    df["iptw"] = np.where(df[TREATMENT] == 1, 1 / df["ps"], 1 / (1 - df["ps"]))
    # Stabilised weights multiply by the marginal treatment probability P(A=a).
    # Same target estimand (ATE), but a much tighter weight distribution, which
    # makes the estimator less sensitive to a few extreme scores.
    p_treated = df[TREATMENT].mean()
    df["siptw"] = np.where(df[TREATMENT] == 1,
                           p_treated / df["ps"],
                           (1 - p_treated) / (1 - df["ps"]))
    print(f"\n=== Inverse Probability Weighting (IPW, targets the ATE) ===")
    print(f"Unstabilised weight range [{df['iptw'].min():.2f}, {df['iptw'].max():.2f}] "
          f"| stabilised weight range [{df['siptw'].min():.2f}, {df['siptw'].max():.2f}]")
    print("Stabilising leaves the estimand unchanged but shrinks the weight range - "
          "the honest diagnostic for extreme weights.")
    for v in PS_COVARIATES:
        w_smd = smd(df[v], df[TREATMENT] == 1, df["iptw"])
        ok = "OK (< 0.10)" if abs(w_smd) <= SMD_THRESHOLD else "still imbalanced"
        print(f"{v:<14} SMD before = {before[v]: .3f} | IPW-weighted = {w_smd: .3f}  [{ok}]")

    # --- Outcome: two DIFFERENT estimands, each with the CORRECT variance ---
    # Both leave the sofa_score mediator out, so both estimate a TOTAL effect of
    # diabetes - but on different populations:
    #   * matched caliper -> ATT: effect among the treated (the matched diabetics)
    #   * IPW 1/ps        -> ATE: effect in the whole cohort
    # These coincide only under a constant treatment effect; otherwise they can
    # differ for real, not just by chance.
    print(f"\n=== Effect of {TREATMENT} on {OUTCOME}: crude vs. matched (ATT) vs. IPW (ATE) ===")

    def or_ci(fit, term=TREATMENT):
        ci = fit.conf_int().loc[term]
        return np.exp(fit.params[term]), np.exp(ci[0]), np.exp(ci[1])

    crude = smf.logit(f"{OUTCOME} ~ {TREATMENT}", data=df).fit(disp=False)
    est, lo, hi = or_ci(crude)
    print(f"Crude (association)            OR = {est:.2f}  95% CI [{lo:.2f}, {hi:.2f}]")

    # Matched analysis targets the ATT. The two members of a matched pair are NOT
    # independent, so we cluster the SE on pair_id; the naive model-based SE that
    # ignores the pairing is not the reason the CI is wide (sample size is).
    matched_fit = smf.logit(f"{OUTCOME} ~ {TREATMENT}", data=matched).fit(
        disp=False, cov_type="cluster", cov_kwds={"groups": matched["pair_id"]})
    est, lo, hi = or_ci(matched_fit)
    print(f"Matched caliper -> ATT ({n_pairs} pairs) OR = {est:.2f}  95% CI [{lo:.2f}, {hi:.2f}]  "
          f"(cluster-robust SE on the matched pair)")

    # IPW targets the ATE. Non-integer IPW weights are NOT frequency counts, so
    # freq_weights (the naive default) inflates the pseudo-N to ~2N and reports a
    # spuriously narrow CI. The correct variance is a robust (Huber-White HC0)
    # sandwich. Verified against a nonparametric bootstrap that refits the PS and
    # outcome model each resample: sandwich SE(log-OR) ~= 0.302 vs bootstrap
    # ~= 0.305 (see README Stolperstein). statsmodels flags cov_type with
    # var_weights as "not fully supported", but for HC0 it reproduces both the R
    # sandwich (sandwich::vcovHC) and the bootstrap, so we silence that one note.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ipw_fit = smf.glm(f"{OUTCOME} ~ {TREATMENT}", data=df, family=sm.families.Binomial(),
                          var_weights=df["iptw"]).fit(cov_type="HC0")
    est, lo, hi = or_ci(ipw_fit)
    print(f"IPW 1/ps -> ATE               OR = {est:.2f}  95% CI [{lo:.2f}, {hi:.2f}]  "
          f"(robust HC0 sandwich SE)")

    # Sensitivity: stabilised weights (same ATE) and 99th-percentile trimming
    # both show the estimate is not driven by the handful of extreme weights.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ipw_stab = smf.glm(f"{OUTCOME} ~ {TREATMENT}", data=df, family=sm.families.Binomial(),
                           var_weights=df["siptw"]).fit(cov_type="HC0")
        cap = np.percentile(df["iptw"], 99)
        ipw_trim = smf.glm(f"{OUTCOME} ~ {TREATMENT}", data=df, family=sm.families.Binomial(),
                           var_weights=df["iptw"].clip(upper=cap)).fit(cov_type="HC0")
    est, lo, hi = or_ci(ipw_stab)
    print(f"  IPW stabilised (ATE)        OR = {est:.2f}  95% CI [{lo:.2f}, {hi:.2f}]")
    est, lo, hi = or_ci(ipw_trim)
    print(f"  IPW trimmed at 99th pct     OR = {est:.2f}  95% CI [{lo:.2f}, {hi:.2f}]  (cap = {cap:.2f})")

    print("\nMatched (ATT) and IPW (ATE) both leave out the sofa_score mediator, so both are")
    print("TOTAL-effect estimands - but on different target populations (the treated vs. the")
    print("whole cohort). The IPW CI uses a robust sandwich SE; a bootstrap that also refits")
    print("the PS is the gold standard and is only marginally wider here (see README).")


if __name__ == "__main__":
    main()
