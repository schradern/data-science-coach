"""Module 14 - Monte-Carlo benchmark: what imputation methods actually do.

Claim under test: for the model `verstorben_30d ~ bga_ph + alter + sofa_score`
fit on a cohort where `bga_ph` is missing depending on the OUTCOME (MAR,
mechanism below), different ways of handling the missing pH values differ not
just in *bias* but in whether their reported standard error is honest about
their own sampling variability.

    p_missing_i = 1 / (1 + exp(-(-1.7 + 1.4 * verstorben_30d_i)))

Each replicate resamples the WHOLE data-generating process -- a fresh cohort
of n = 500 patients (age, comorbidities, SOFA, the death process, the true
pH, and the missingness mask), via `lib.ground_truth.replay_cohort_with_ph`.
That is what makes this a valid Monte-Carlo study: the oracle (which fits on
the true, unmasked pH) has genuine patient-to-patient sampling variability,
so its empirical SD is a real sampling SD, not zero by construction. Every
other method's SE ratio and coverage are then judged against that same real
sampling variability, for nine competing methods:

  * bias      - mean(beta_hat) - beta_target, where beta_target is the
    POPULATION parameter (fit at n = 400,000, not this replicate's cohort).
  * precision honesty - does the method's own reported SE match how much its
    point estimate actually jumps around across replicates (SE ratio), and
    does its nominal-95% CI actually cover the target 95% of the time?

The headline result is the CONTRAST between the slope `beta_bga_ph` and the
Intercept / predicted reference risk: selecting on the outcome shifts a
logistic intercept, not its slopes (Prentice & Pyke, 1979), so complete-case
analysis should look fine for the odds ratio and be badly miscalibrated for
every absolute risk.

Run:
    cd module/14-fehlende-werte/code
    MPLBACKEND=Agg ../../../.venv/bin/python benchmark.py
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import patsy  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.special import expit  # noqa: E402
from statsmodels.imputation import mice  # noqa: E402

from lib.helpers import SEED  # noqa: E402
from lib.ground_truth import replay_cohort_with_ph  # noqa: E402
from lib.plotstyle import EVENT, PALETTE, PRIMARY, SECONDARY, apply_style, save  # noqa: E402

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Fixed analysis model and reference patient (identical to code/python.py).
# ---------------------------------------------------------------------------
MODEL = "verstorben_30d ~ bga_ph + alter + sofa_score"
MODEL_IND = "verstorben_30d ~ bga_ph + alter + sofa_score + bga_fehlt"
REF = {"bga_ph": 7.38, "alter": 64, "sofa_score": 4}
REF_IND = {**REF, "bga_fehlt": 0}
Z = stats.norm.ppf(0.975)  # 95 % Wald critical value, used for every method alike

# Monte-Carlo replicate count. See the timing note printed at the end of
# main() for how this number was chosen -- it is NOT a silent subsample.
R = 500

# Population target: one huge replay of the SAME data-generating process,
# fit on the TRUE (unmasked) pH. This is the estimand every replicate's
# methods are judged against -- never hardcoded, always recomputed here.
POP_N = 400_000
POP_SEED = [SEED, 0, 1]  # fixed, distinct from every per-replicate seed below

# MICE settings, matching code/python.py's single-draw demonstration.
MICE_M = 20          # multiple-imputation count (Rubin's rules)
MICE_BURNIN = 10      # burn-in cycles before the first stored imputation
MICE_SKIP = 3         # cycles skipped between stored imputations (statsmodels default)

# Performance guard: MICE (m=20, with burn-in/skip cycles) is by far the most
# expensive method per replicate. If the full run would blow the ~10-minute
# budget, MICE is evaluated on a PREFIX subset of the replicates and every
# other method still uses all R -- this is measured, not assumed, and is
# always printed explicitly (see `mice_note` in main()).

PMM_K = 5  # predictive-mean-matching donor pool size


# ---------------------------------------------------------------------------
# A tiny "fitted model" wrapper so a single `summarize()` works for both
# genuine statsmodels results (oracle, complete-case, mean, median, the two
# regression-imputation variants, PMM, the indicator model) AND for a
# Rubin's-rules-pooled MICE result, which has no `.model.data.design_info`
# of its own.
# ---------------------------------------------------------------------------
class PooledFit:
    def __init__(self, params: pd.Series, cov: pd.DataFrame, design_info) -> None:
        self.params = params
        self._cov = cov
        self.model = SimpleNamespace(data=SimpleNamespace(design_info=design_info))

    def cov_params(self) -> pd.DataFrame:
        return self._cov


def summarize(fit, ref: dict) -> dict:
    """Point estimate + Wald SE/CI for beta_bga_ph, Intercept, and the
    predicted risk at the reference patient (delta method on the linear
    predictor, transformed through the logistic link)."""
    design_info = fit.model.data.design_info
    ref_df = pd.DataFrame([ref])
    x_ref = np.asarray(patsy.dmatrix(design_info, ref_df, return_type="dataframe"))[0]
    cov = np.asarray(fit.cov_params())
    idx = list(fit.params.index)
    params = np.asarray(fit.params)

    def se_of(name: str) -> float:
        j = idx.index(name)
        return float(np.sqrt(cov[j, j]))

    beta = float(fit.params["bga_ph"])
    se_beta = se_of("bga_ph")
    intercept = float(fit.params["Intercept"])
    se_intercept = se_of("Intercept")

    eta = float(x_ref @ params)
    var_eta = float(x_ref @ cov @ x_ref)
    se_eta = float(np.sqrt(max(var_eta, 0.0)))
    risk = float(expit(eta))
    se_risk = risk * (1 - risk) * se_eta  # delta method, probability scale

    return dict(
        beta=beta, se_beta=se_beta,
        ci_beta=(beta - Z * se_beta, beta + Z * se_beta),
        intercept=intercept, se_intercept=se_intercept,
        ci_intercept=(intercept - Z * se_intercept, intercept + Z * se_intercept),
        risk=risk, se_risk=se_risk,
        ci_risk=(expit(eta - Z * se_eta), expit(eta + Z * se_eta)),
    )


def combine_rubin(fits: list) -> PooledFit:
    """Rubin's rules, reimplemented (validated against statsmodels'
    `mice.MICE.combine()` on identical data -- same params, same SE to full
    float precision). Needed because `mice.MICE.fit()` only keeps the LAST
    imputed dataset, and this benchmark also needs every intermediate
    imputed value to compute the RMSE-vs-truth metric."""
    m = len(fits)
    idx = fits[0].params.index
    params_mat = np.vstack([f.params.reindex(idx).to_numpy() for f in fits])
    params_mean = params_mat.mean(axis=0)
    cov_within = sum(
        np.asarray(f.cov_params().reindex(index=idx, columns=idx)) for f in fits
    ) / m
    cov_between = np.cov(params_mat, rowvar=False, ddof=1)
    cov_total = cov_within + (1 + 1.0 / m) * cov_between
    params_series = pd.Series(params_mean, index=idx)
    cov_df = pd.DataFrame(cov_total, index=idx, columns=idx)
    return PooledFit(params_series, cov_df, fits[0].model.data.design_info)


# ---------------------------------------------------------------------------
# One Monte-Carlo replicate: draw a FRESH cohort (patients, true pH, and the
# missingness mask) from the data-generating process, apply every method, and
# return a dict of {method_name: summarize(...)} plus RMSE-vs-truth where
# applicable.
# ---------------------------------------------------------------------------
def run_replicate(cohort_seed, draw_seed, run_mice: bool) -> dict:
    df = replay_cohort_with_ph(500, seed=cohort_seed)
    rng = np.random.default_rng(draw_seed)  # for the methods' OWN stochastic draws

    mask = df["bga_ph"].isna().to_numpy()
    truth_missing = df.loc[mask, "bga_ph_wahr"].to_numpy()

    work = df.copy()

    out: dict[str, dict] = {}

    # --- oracle: fit on THIS replicate's true, unmasked pH -------------
    oracle_df = df.assign(bga_ph=df["bga_ph_wahr"])
    out["oracle"] = summarize(smf.logit(MODEL, data=oracle_df).fit(disp=0), REF)

    # --- complete case ------------------------------------------------
    cc = work.dropna(subset=["bga_ph"])
    out["complete_case"] = summarize(smf.logit(MODEL, data=cc).fit(disp=0), REF)

    # --- mean / median imputation --------------------------------------
    mean_val = work["bga_ph"].mean()
    median_val = work["bga_ph"].median()

    mean_work = work.copy()
    mean_work["bga_ph"] = mean_work["bga_ph"].fillna(mean_val)
    out["mean_imputation"] = summarize(smf.logit(MODEL, data=mean_work).fit(disp=0), REF)
    out["mean_imputation"]["rmse"] = float(np.sqrt(np.mean((mean_val - truth_missing) ** 2)))

    median_work = work.copy()
    median_work["bga_ph"] = median_work["bga_ph"].fillna(median_val)
    out["median_imputation"] = summarize(smf.logit(MODEL, data=median_work).fit(disp=0), REF)
    out["median_imputation"]["rmse"] = float(np.sqrt(np.mean((median_val - truth_missing) ** 2)))

    # --- regression imputation (deterministic + single stochastic draw) -
    obs = work.loc[~mask]
    mis = work.loc[mask]
    ols = smf.ols("bga_ph ~ alter + sofa_score + verstorben_30d", data=obs).fit()
    pred_mis = ols.predict(mis).to_numpy()

    det_work = work.copy()
    det_work.loc[mask, "bga_ph"] = pred_mis
    out["regression_deterministic"] = summarize(smf.logit(MODEL, data=det_work).fit(disp=0), REF)
    out["regression_deterministic"]["rmse"] = float(np.sqrt(np.mean((pred_mis - truth_missing) ** 2)))

    sigma_hat = float(np.sqrt(ols.mse_resid))
    stoch_draw = pred_mis + rng.normal(0, sigma_hat, size=mask.sum())
    stoch_work = work.copy()
    stoch_work.loc[mask, "bga_ph"] = stoch_draw
    out["regression_stochastic"] = summarize(smf.logit(MODEL, data=stoch_work).fit(disp=0), REF)
    out["regression_stochastic"]["rmse"] = float(np.sqrt(np.mean((stoch_draw - truth_missing) ** 2)))

    # --- predictive mean matching (single imputation, k=5 donors) -------
    pred_obs = ols.predict(obs).to_numpy()
    obs_vals = obs["bga_ph"].to_numpy()
    pmm_draw = np.empty(mask.sum())
    for i, pm in enumerate(pred_mis):
        donor_idx = np.argsort(np.abs(pred_obs - pm))[:PMM_K]
        chosen = rng.integers(0, PMM_K)
        pmm_draw[i] = obs_vals[donor_idx[chosen]]
    pmm_work = work.copy()
    pmm_work.loc[mask, "bga_ph"] = pmm_draw
    out["pmm"] = summarize(smf.logit(MODEL, data=pmm_work).fit(disp=0), REF)
    out["pmm"]["rmse"] = float(np.sqrt(np.mean((pmm_draw - truth_missing) ** 2)))

    # --- missing-indicator method ---------------------------------------
    ind_work = work.copy()
    ind_work["bga_fehlt"] = mask.astype(int)
    ind_work["bga_ph"] = ind_work["bga_ph"].fillna(median_val)
    out["missing_indicator"] = summarize(smf.logit(MODEL_IND, data=ind_work).fit(disp=0), REF_IND)

    # --- MICE (m = 20, Rubin's rules) -----------------------------------
    if run_mice:
        # MICEData draws from the GLOBAL numpy random state (not a
        # Generator). Reseed it from `rng` -- deterministic given the
        # replicate's own draw seed, independent of every other draw made
        # above -- so nothing here uses a bare integer literal other than
        # SEED itself.
        np.random.seed(int(rng.integers(0, 2**31 - 1)))
        imp_data = work[["verstorben_30d", "bga_ph", "alter", "sofa_score"]].copy()
        imputer = mice.MICEData(imp_data)
        imputer.set_imputer("bga_ph", "alter + sofa_score + verstorben_30d")
        imputer.update_all(MICE_BURNIN)
        fits, imputed_draws = [], []
        for _ in range(MICE_M):
            imputer.update_all(MICE_SKIP + 1)
            fits.append(smf.logit(MODEL, data=imputer.data).fit(disp=0))
            imputed_draws.append(imputer.data.loc[mask, "bga_ph"].to_numpy())
        pooled = combine_rubin(fits)
        out["mice"] = summarize(pooled, REF)
        imputed_mean = np.mean(imputed_draws, axis=0)  # posterior mean across the m draws
        out["mice"]["rmse"] = float(np.sqrt(np.mean((imputed_mean - truth_missing) ** 2)))

    return out


# ---------------------------------------------------------------------------
# Aggregation and reporting
# ---------------------------------------------------------------------------
METHOD_LABELS = {
    "oracle": "oracle / full data",
    "complete_case": "complete case",
    "mean_imputation": "mean imputation",
    "median_imputation": "median imputation",
    "regression_deterministic": "regression imputation (deterministic)",
    "regression_stochastic": "regression imputation (stochastic, single)",
    "pmm": "PMM (single, k=5)",
    "mice": "MICE (m=20, Rubin's rules)",
    "missing_indicator": "missing-indicator method",
}
METHOD_ORDER = list(METHOD_LABELS)
SINGLE_IMPUTATION = {
    "mean_imputation", "median_imputation", "regression_deterministic",
    "regression_stochastic", "pmm", "missing_indicator",
}
RMSE_METHODS = [
    "mean_imputation", "median_imputation", "regression_deterministic",
    "regression_stochastic", "pmm", "mice",
]


def compute_metrics(betas: np.ndarray, ses: np.ndarray, ci_lo: np.ndarray,
                     ci_hi: np.ndarray, target: float) -> dict:
    emp_sd = float(betas.std(ddof=1))
    mean_se = float(ses.mean())
    se_ratio = mean_se / emp_sd if emp_sd > 1e-12 else float("nan")
    coverage = float(np.mean((ci_lo <= target) & (target <= ci_hi)))
    return dict(bias=float(betas.mean() - target), emp_sd=emp_sd, mean_se=mean_se,
                se_ratio=se_ratio, coverage=coverage)


def print_table(title: str, target: float, rows: dict[str, dict]) -> None:
    print(f"\n{title}  (target = {target:+.4f})")
    header = f"  {'method':42s}{'bias':>10}{'emp. SD':>10}{'mean SE':>10}{'SE ratio':>10}{'coverage':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for key in METHOD_ORDER:
        m = rows[key]
        ratio = f"{m['se_ratio']:>10.2f}" if np.isfinite(m["se_ratio"]) else f"{'n/a':>10}"
        print(f"  {METHOD_LABELS[key]:42s}{m['bias']:>+10.4f}{m['emp_sd']:>10.4f}"
              f"{m['mean_se']:>10.4f}{ratio}{m['coverage']:>10.1%}")


def main() -> None:
    # Announce the runtime BEFORE doing anything slow. This script takes about
    # five minutes; a learner who sees no output for five minutes assumes it
    # hung and kills it.
    print("=" * 88)
    print("HINWEIS: Dieses Skript rechnet ~5 Minuten (500 Monte-Carlo-Kohorten x 9 Verfahren,")
    print("         inklusive multipler Imputation mit m = 20). Es schreibt am Ende die")
    print("         Abbildung assets/imputation_coverage.png. Bitte nicht abbrechen.")
    print("=" * 88, flush=True)

    t_start = time.time()

    # --------------------------------------------------------------
    # Population target: fit the FIXED analysis model on a single huge
    # replay of the data-generating process, using the TRUE (unmasked) pH.
    # Never hardcoded -- recomputed here every run.
    # --------------------------------------------------------------
    pop_df = replay_cohort_with_ph(POP_N, seed=POP_SEED)
    pop_df = pop_df.assign(bga_ph=pop_df["bga_ph_wahr"])
    target_fit = smf.logit(MODEL, data=pop_df).fit(disp=0)
    target = summarize(target_fit, REF)

    print("=" * 88)
    print("Monte-Carlo benchmark: what imputation methods actually do to")
    print(f"  {MODEL}")
    print("=" * 88)
    print(f"Each replicate resamples a FRESH cohort of n = 500 patients from the full")
    print(f"data-generating process (patients, true pH, and the missingness mask).")
    print(f"Missingness mechanism: p = 1/(1+exp(-(-1.7 + 1.4*verstorben_30d)))")
    print(f"Population target (fit on {POP_N:,} freshly replayed patients with the TRUE pH,")
    print(f"  recomputed here, not hardcoded):")
    print(f"  beta_bga_ph = {target['beta']:+.4f}   Intercept = {target['intercept']:+.4f}   "
          f"risk @ ref patient = {target['risk']:.4f}")

    # --------------------------------------------------------------
    # Decide the MICE replicate budget from measured per-replicate timing
    # (never a silent subsample -- always printed and commented).
    # --------------------------------------------------------------
    probe_cohort_seed = [SEED, R]      # a stream never used by the real loop
    probe_draw_seed = [SEED, R, 1]
    t0 = time.time()
    run_replicate(probe_cohort_seed, probe_draw_seed, run_mice=True)
    per_rep_with_mice = time.time() - t0
    t0 = time.time()
    run_replicate(probe_cohort_seed, probe_draw_seed, run_mice=False)
    per_rep_without_mice = time.time() - t0
    mice_only = max(per_rep_with_mice - per_rep_without_mice, 1e-6)

    projected_full = per_rep_without_mice * R + mice_only * R
    # MICE always runs all R replicates. A timing-based cap would make the
    # reported coverage depend on machine speed — a fast machine runs full R and
    # prints one set of numbers, a loaded or slow machine runs fewer and prints
    # different ones. The module asserts specific coverage values and
    # tools/check_numbers.py verifies them, so determinism must win over the
    # ~5-minute wall-clock. The projection is still printed so a slow run is no
    # surprise.
    mice_r = R
    print(f"\nMICE runs all R = {R} replicates (measured {per_rep_with_mice:.2f} s/replicate; "
          f"projected total {projected_full / 60:.1f} min).")

    # --------------------------------------------------------------
    # Monte-Carlo loop. Each replicate resamples the WHOLE cohort (not just
    # the missingness mask): `[SEED, rep]` seeds the patients + true pH +
    # mask, `[SEED, rep, 1]` seeds each method's own stochastic imputation
    # draws (regression noise, PMM donor choice, MICE reseed).
    # --------------------------------------------------------------
    records: dict[str, dict[str, list]] = {
        key: {"beta": [], "se_beta": [], "ci_beta": [],
              "intercept": [], "se_intercept": [], "ci_intercept": [],
              "risk": [], "se_risk": [], "ci_risk": [], "rmse": []}
        for key in METHOD_LABELS
    }

    for rep in range(R):
        cohort_seed = [SEED, rep]
        draw_seed = [SEED, rep, 1]
        run_mice = rep < mice_r
        result = run_replicate(cohort_seed, draw_seed, run_mice=run_mice)
        for key, summ in result.items():
            rec = records[key]
            rec["beta"].append(summ["beta"]); rec["se_beta"].append(summ["se_beta"])
            rec["ci_beta"].append(summ["ci_beta"])
            rec["intercept"].append(summ["intercept"]); rec["se_intercept"].append(summ["se_intercept"])
            rec["ci_intercept"].append(summ["ci_intercept"])
            rec["risk"].append(summ["risk"]); rec["se_risk"].append(summ["se_risk"])
            rec["ci_risk"].append(summ["ci_risk"])
            if "rmse" in summ:
                rec["rmse"].append(summ["rmse"])

    elapsed = time.time() - t_start
    print(f"Completed R = {R} replicates ({mice_r} with MICE) in {elapsed / 60:.1f} min.")

    # --------------------------------------------------------------
    # Metrics per estimand.
    # --------------------------------------------------------------
    def metrics_for(field: str, ci_field: str, target_value: float) -> dict[str, dict]:
        out = {}
        for key, rec in records.items():
            betas = np.asarray(rec[field])
            ses = np.asarray(rec[f"se_{field}"] if field != "risk" else rec["se_risk"])
            cis = np.asarray(rec[ci_field])
            out[key] = compute_metrics(betas, ses, cis[:, 0], cis[:, 1], target_value)
        return out

    beta_metrics = metrics_for("beta", "ci_beta", target["beta"])
    intercept_metrics = metrics_for("intercept", "ci_intercept", target["intercept"])
    risk_metrics = metrics_for("risk", "ci_risk", target["risk"])

    print_table("1) beta_bga_ph -- the slope", target["beta"], beta_metrics)
    print_table("2) Intercept", target["intercept"], intercept_metrics)
    print_table("3) predicted 30-day risk @ reference patient (pH 7.38, 64y, SOFA 4)",
                target["risk"], risk_metrics)

    # --------------------------------------------------------------
    # RMSE of imputed values vs. the truth (missing rows only).
    # --------------------------------------------------------------
    print(f"\n4) RMSE of imputed bga_ph values vs. bga_ph_wahr (missing rows only)")
    header = f"  {'method':42s}{'mean RMSE':>12}{'SD RMSE':>12}{'n replicates':>14}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for key in RMSE_METHODS:
        rmse = np.asarray(records[key]["rmse"])
        print(f"  {METHOD_LABELS[key]:42s}{rmse.mean():>12.4f}{rmse.std(ddof=1):>12.4f}{len(rmse):>14d}")

    # --------------------------------------------------------------
    # Interpretive footer -- every statement below is derived from the
    # numbers this run just produced, not asserted in advance.
    # --------------------------------------------------------------
    print("\n" + "=" * 88)
    print("Interpretation")
    print("=" * 88)

    oracle_cov = beta_metrics["oracle"]["coverage"]
    oracle_sd = beta_metrics["oracle"]["emp_sd"]
    oracle_ratio = beta_metrics["oracle"]["se_ratio"]
    print(f"  Oracle now resamples a FRESH cohort every replicate (patients, true pH, mask),")
    print(f"  so its empirical SD is a genuine sampling SD, not zero by construction:")
    print(f"  emp. SD = {oracle_sd:.4f}, SE ratio = {oracle_ratio:.2f}, coverage = {oracle_cov:.1%}")
    print(f"  (this is the harness's own self-consistency check: with a correctly specified")
    print(f"  model and a valid Monte-Carlo design, oracle coverage should sit near 95%).")

    unbiased = [k for k in METHOD_ORDER
                if abs(beta_metrics[k]["bias"]) < 0.15 * abs(target["beta"]) and k != "oracle"]
    liars = [k for k in METHOD_ORDER
             if np.isfinite(beta_metrics[k]["se_ratio"]) and beta_metrics[k]["se_ratio"] < 0.9]
    print(f"\n  Roughly unbiased for beta_bga_ph: {', '.join(METHOD_LABELS[k] for k in unbiased) or '(none)'}")
    print(f"  Understate their own precision (SE ratio < 0.9, i.e. reported SE undersells true")
    print(f"  sampling variability): {', '.join(METHOD_LABELS[k] for k in liars) or '(none)'}")

    single_imputation_liars = [k for k in liars if k in SINGLE_IMPUTATION]
    if not single_imputation_liars:
        print(f"  NOTE: no single-imputation method landed below SE ratio 0.9 -- that contradicts")
        print(f"  the expectation that treating one guessed value as observed data undersells")
        print(f"  uncertainty. Investigate before trusting this run.")
    else:
        print(f"  As expected, single-imputation methods appear here: "
              f"{', '.join(METHOD_LABELS[k] for k in single_imputation_liars)}.")

    cc_beta_cov = beta_metrics["complete_case"]["coverage"]
    cc_int_cov = intercept_metrics["complete_case"]["coverage"]
    cc_risk_cov = risk_metrics["complete_case"]["coverage"]
    print(f"\n  Complete-case coverage for beta_bga_ph: {cc_beta_cov:.1%}"
          f"  vs. for the Intercept: {cc_int_cov:.1%}"
          f"  vs. for the reference risk: {cc_risk_cov:.1%}.")
    print(f"  That gap is the point: selecting on the OUTCOME shifts a logistic intercept, not")
    print(f"  its slopes (Prentice & Pyke, 1979) -- complete-case should stay near-nominal for the")
    print(f"  slope while collapsing for the intercept and every absolute risk it predicts.")

    print(f"\n  Single-imputation methods (mean/median/regression/PMM/indicator) all treat one")
    print(f"  guessed value as if it were observed data: none of them add back the between-")
    print(f"  imputation uncertainty, so their SE ratio and coverage tend to undershoot -- MICE's")
    print(f"  Rubin's-rules SE is the one built to be honest about that extra uncertainty.")
    mice_cov = beta_metrics["mice"]["coverage"]
    median_cov = beta_metrics["median_imputation"]["coverage"]
    detreg_cov = beta_metrics["regression_deterministic"]["coverage"]
    print(f"  Measured coverage for beta_bga_ph -- MICE: {mice_cov:.1%}, median imputation: "
          f"{median_cov:.1%}, deterministic regression imputation: {detreg_cov:.1%}.")

    # --------------------------------------------------------------
    # Figure: coverage per method for beta_bga_ph.
    # --------------------------------------------------------------
    make_figure(beta_metrics)


def make_figure(beta_metrics: dict[str, dict]) -> None:
    import matplotlib.pyplot as plt

    apply_style()

    order = list(reversed(METHOD_ORDER))  # oracle on top
    labels_de = {
        "oracle": "Oracle (volle Wahrheit)",
        "complete_case": "Complete Case",
        "mean_imputation": "Mittelwert-Imputation",
        "median_imputation": "Median-Imputation",
        "regression_deterministic": "Regressions-Imputation (deterministisch)",
        "regression_stochastic": "Regressions-Imputation (stochastisch, single)",
        "pmm": "PMM (single, k=5)",
        "mice": "MICE (m=20, Rubin's rules)",
        "missing_indicator": "Missing-Indicator-Methode",
    }
    coverage = [beta_metrics[k]["coverage"] for k in order]
    colors = []
    for k in order:
        if k in ("oracle",):
            colors.append(PRIMARY)
        elif k == "complete_case":
            colors.append(SECONDARY)
        elif k == "mice":
            colors.append(PALETTE[2])  # multiple imputation -- distinct colour
        else:
            colors.append(EVENT)  # single-imputation methods

    fig, ax = plt.subplots(figsize=(9, 5.5))

    nominal = 0.95
    tol = 2 * np.sqrt(nominal * (1 - nominal) / R)
    ax.axvspan(nominal - tol, nominal + tol, color="#ECEDEF", zorder=0,
               label=f"Monte-Carlo-Toleranz (±2·SE, R={R})")
    ax.axvline(nominal, color=SECONDARY, linewidth=1.2, linestyle="--", zorder=1)

    bars = ax.barh(range(len(order)), coverage, color=colors, edgecolor="none",
                    height=0.6, zorder=2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([labels_de[k] for k in order])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("95%-CI-Abdeckung für beta(bga_ph)")
    ax.set_title("Abdeckung des 95%-Konfidenzintervalls je Imputationsmethode")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)

    for bar, cov, k in zip(bars, coverage, order):
        single = " (single)" if k in SINGLE_IMPUTATION else ""
        ax.text(min(cov + 0.015, 1.0), bar.get_y() + bar.get_height() / 2,
                f"{cov:.0%}{single}", va="center", fontsize=9)

    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor=PRIMARY, label="Oracle (volle Daten)"),
        Patch(facecolor=SECONDARY, label="Complete Case"),
        Patch(facecolor=EVENT, label="Single-Imputation-Methoden"),
        Patch(facecolor=PALETTE[2], label="Multiple Imputation (MICE)"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=9)

    save(fig, Path(__file__).resolve().parents[1] / "assets" / "imputation_coverage.png")


if __name__ == "__main__":
    main()
