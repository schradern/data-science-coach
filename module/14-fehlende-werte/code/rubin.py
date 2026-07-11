"""Module 14 - Rubin's rules, derived by hand.

Run: python module/14-fehlende-werte/code/rubin.py

`code/python.py` already shows *that* multiple imputation beats median
imputation. This script proves *how* MI turns m completed datasets into one
answer: every pooling quantity below (Qbar, Ubar, B, T, SE, r, lambda, the
Barnard-Rubin degrees of freedom, fmi, relative efficiency, the CI) is
computed with plain arithmetic from the m point estimates and variances --
no `mice::pool()`, no `statsmodels.imputation.mice.MICE` pooling call. Those
libraries are only used afterwards, in step 4, to check the hand computation
against a trusted implementation.

Six steps:
  1) Build m completed datasets with statsmodels' chained-equations imputer.
  2) Fit the analysis model on each one; collect Q_l (the bga_ph coefficient)
     and U_l (its variance).
  3) Pool Q_l, U_l into Rubin's rules by hand, including the Barnard-Rubin
     (1999) degrees of freedom -- the modern replacement for Rubin's (1987)
     df, which behaves badly whenever the complete-data df is not huge.
  4) Validate: reproduce the SAME hand-computed numbers through statsmodels'
     own pooling arithmetic (applied to the identical fitted models), then
     also run an independent library MICE chain and show where and why it
     disagrees (different random draws; no Barnard-Rubin df in statsmodels).
  5) Show *why* the (1+1/m) correction exists: recompute the total variance
     with and without it and watch the inflation shrink as m grows.
  6) Ask how large m needs to be: the White/Royston/Wood (2011) rule of
     thumb, and an empirical Monte Carlo showing the pooled SE stabilising
     as m grows.

Estimand throughout: the coefficient of `bga_ph` in
`verstorben_30d ~ bga_ph + alter + sofa_score` (logistic). `bga_ph` is
~19.8 % missing and missing depending on the OUTCOME (MAR given outcome), so
the imputation model below includes `verstorben_30d` -- exactly what makes
this missingness ignorable.
"""
from __future__ import annotations

import math

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from scipy import stats  # noqa: E402
from statsmodels.imputation import mice  # noqa: E402
from statsmodels.imputation.mice import MICEData  # noqa: E402

from lib.helpers import SEED, load_cohort, load_labs  # noqa: E402

warnings.filterwarnings("ignore")

MODEL = "verstorben_30d ~ bga_ph + alter + sofa_score"
IMPUTER_FORMULA = "alter + sofa_score + verstorben_30d"
M = 20                       # number of imputations for the main analysis
N = 500                      # cohort size
K = 4                        # complete-data parameters: intercept + 3 slopes
BANNER = "=" * 74


def load() -> pd.DataFrame:
    df = load_cohort().merge(load_labs(), on="patient_id", how="left")
    return df[["verstorben_30d", "bga_ph", "alter", "sofa_score"]].copy()


# ---------------------------------------------------------------------------
# Steps 1-2: build m completed datasets, fit the analysis model on each.
# ---------------------------------------------------------------------------
def run_mice_chain(work: pd.DataFrame, m: int, seed: int):
    """One chained-equations MICE run of length m.

    Reseeds numpy's global RNG (MICEData draws from it directly, not from a
    passed-in Generator), builds a fresh imputer, and calls `update_all()`
    once per imputation, snapshotting `imp.data` each time -- exactly the
    m completed datasets a real MI analysis would carry forward.

    Returns (imp, datasets, fits, Q, U) where Q_l/U_l are the bga_ph
    coefficient and its squared standard error from fitting MODEL on
    datasets[l].
    """
    np.random.seed(seed)
    imp = MICEData(work)
    imp.set_imputer("bga_ph", IMPUTER_FORMULA)

    datasets: list[pd.DataFrame] = []
    fits = []
    for _ in range(m):
        imp.update_all()
        d = imp.data.copy()
        fit = smf.logit(MODEL, data=d).fit(disp=0)
        datasets.append(d)
        fits.append(fit)

    Q = np.array([f.params["bga_ph"] for f in fits])
    U = np.array([f.bse["bga_ph"] ** 2 for f in fits])
    return imp, datasets, fits, Q, U


def section_1_2_build_and_fit(work: pd.DataFrame):
    print(BANNER)
    print("1) Build m completed datasets, fit the analysis model on each")
    print(BANNER)
    n_missing = int(work["bga_ph"].isna().sum())
    print(f"  bga_ph missing: {n_missing}/{len(work)} ({n_missing / len(work):.1%})")
    print(f"  imputation model: bga_ph ~ {IMPUTER_FORMULA}  (includes the OUTCOME -> MAR ignorable)")
    print(f"  m = {M} completed datasets via MICEData chained equations, seed = SEED")

    imp, datasets, fits, Q, U = run_mice_chain(work, M, SEED)

    # One concrete cell, followed across the m datasets, to make the between-
    # imputation *uncertainty* tangible before it becomes the abstraction B.
    missing_idx = work.index[work["bga_ph"].isna()]
    row = missing_idx[0]
    vals = np.array([d.loc[row, "bga_ph"] for d in datasets])
    print(f"\n  e.g. row {row} (bga_ph originally missing): imputed value across the {M}")
    print(f"  datasets ranges {vals.min():.2f} to {vals.max():.2f} (sd={vals.std(ddof=1):.2f}).")
    print("  That spread across datasets is exactly what B measures below.")

    print(f"\n2) Fit '{MODEL}' on each of the {M} completed datasets")
    print(f"  {'l':>3}{'Q_l = b(bga_ph)':>18}{'U_l = se^2':>14}")
    for l, (q, u) in enumerate(zip(Q, U), start=1):
        print(f"  {l:>3}{q:>18.4f}{u:>14.4f}")

    return imp, datasets, fits, Q, U


# ---------------------------------------------------------------------------
# Step 3: Rubin's rules, computed by hand from Q and U.
# ---------------------------------------------------------------------------
def rubin_pool(Q: np.ndarray, U: np.ndarray, m: int, n: int = N, k: int = K) -> dict:
    """Hand-rolled Rubin (1987) + Barnard-Rubin (1999) pooling.

    Q: array of m point estimates Q_l.
    U: array of m within-imputation variances U_l (i.e. squared SEs).
    No library pooling function is used anywhere in this function.
    """
    Qbar = Q.mean()                                    # pooled point estimate
    Ubar = U.mean()                                     # within-imputation variance
    B = ((Q - Qbar) ** 2).sum() / (m - 1)                # between-imputation variance
    T = Ubar + (1 + 1 / m) * B                           # total variance
    SE = np.sqrt(T)

    r = (1 + 1 / m) * B / Ubar                           # relative increase in variance
    # lambda and fmi are DIFFERENT quantities, even though both get called
    # "fraction of missing information" informally: lambda is a pure
    # variance-share ratio; fmi (below) additionally corrects for finite m
    # via the Barnard-Rubin df, so fmi != lambda in general.
    lam = (1 + 1 / m) * B / T                            # == r / (r + 1)

    nu_old = (m - 1) / lam ** 2                          # Rubin (1987) df -- obsolete
    nu_com = n - k                                        # complete-data df
    nu_obs = ((nu_com + 1) / (nu_com + 3)) * nu_com * (1 - lam)
    nu_br = (nu_old * nu_obs) / (nu_old + nu_obs)          # Barnard-Rubin (1999) df

    fmi = (r + 2 / (nu_br + 3)) / (r + 1)                  # what mice::pool() prints as `fmi`
    RE = 1 / (1 + lam / m)                                 # relative efficiency vs m = infinity

    t_stat = Qbar / SE
    p_value = 2 * stats.t.sf(abs(t_stat), df=nu_br)
    t_crit = stats.t.ppf(0.975, df=nu_br)
    ci_lo, ci_hi = Qbar - t_crit * SE, Qbar + t_crit * SE

    return {
        "m": m, "Qbar": Qbar, "Ubar": Ubar, "B": B, "T": T, "SE": SE,
        "r": r, "lambda": lam, "nu_old": nu_old, "nu_com": nu_com, "nu_obs": nu_obs,
        "nu_br": nu_br, "fmi": fmi, "RE": RE, "t": t_stat, "p": p_value,
        "ci_lo": ci_lo, "ci_hi": ci_hi,
    }


def section_3_hand_rubin(pooled: dict) -> None:
    print("\n" + BANNER)
    print("3) Rubin's rules, by hand, from Q_l and U_l alone")
    print(BANNER)
    print(f"  Qbar  (pooled estimate)        = {pooled['Qbar']:.4f}")
    print(f"  Ubar  (within-imputation var.) = {pooled['Ubar']:.4f}")
    print(f"  B     (between-imputation var.)= {pooled['B']:.4f}")
    print(f"  T     (total variance)         = {pooled['T']:.4f}")
    print(f"  SE    = sqrt(T)                = {pooled['SE']:.4f}")
    print(f"  r     (relative variance incr.)= {pooled['r']:.4f}")

    print("\n  lambda (variance share) and fmi (mice's finite-m-corrected FMI)")
    print("  are NOT the same number -- printed separately on purpose:")
    print(f"    lambda = (1+1/m)*B/T          = {pooled['lambda']:.4f}")
    print(f"    fmi    = (r + 2/(nu_BR+3))/(r+1) = {pooled['fmi']:.4f}")

    print("\n  degrees of freedom -- Barnard-Rubin (1999) shrinks Rubin's (1987) nu_old:")
    print(f"    nu_old (Rubin 1987, (m-1)/lambda^2)     = {pooled['nu_old']:.1f}")
    print(f"    nu_com (complete-data, n-k = {N}-{K})        = {pooled['nu_com']}")
    print(f"    nu_obs                                  = {pooled['nu_obs']:.1f}")
    print(f"    nu_BR  (Barnard-Rubin, side by side)    = {pooled['nu_br']:.1f}")

    print(f"\n  RE (relative efficiency vs m=infinity) = {pooled['RE']:.4f}")
    print(f"  t = Qbar/SE = {pooled['t']:.3f}, df = nu_BR = {pooled['nu_br']:.1f}"
          f" -> two-sided p = {pooled['p']:.4f}")
    print(f"  95% CI = Qbar +/- t_(0.975, nu_BR)*SE = [{pooled['ci_lo']:.3f}, {pooled['ci_hi']:.3f}]")


# ---------------------------------------------------------------------------
# Step 4: validate the hand computation against library pooling.
# ---------------------------------------------------------------------------
def section_4_validate(imp: MICEData, fits: list, pooled: dict) -> None:
    print("\n" + BANNER)
    print("4) Validate against the library -- do not fudge disagreements")
    print(BANNER)

    # (a) SAME m completed datasets: feed the exact fitted models from step 2
    # into statsmodels' own MICE.combine() pooling arithmetic. This isolates
    # the MATH -- same Q_l, U_l, same formula -- so it must match the hand
    # computation to floating-point noise.
    mice_obj = mice.MICE(MODEL, sm.Logit, imp)
    mice_obj.results_list = fits
    mice_obj.exog_names = fits[0].model.exog_names
    mice_obj.endog_names = fits[0].model.endog_names
    combined = mice_obj.combine()
    idx = list(combined.exog_names).index("bga_ph")
    lib_Qbar, lib_SE = combined.params[idx], combined.bse[idx]
    lib_lambda = combined.frac_miss_info[idx]

    print("  (a) same m completed datasets, statsmodels' own combine():")
    print(f"      hand        Qbar={pooled['Qbar']:.6f}  SE={pooled['SE']:.6f}")
    print(f"      statsmodels Qbar={lib_Qbar:.6f}  SE={lib_SE:.6f}")
    match = np.isclose(pooled["Qbar"], lib_Qbar) and np.isclose(pooled["SE"], lib_SE)
    print(f"      match to floating-point noise: {match}")
    print(f"      statsmodels attribute 'frac_miss_info' = {lib_lambda:.4f}"
          f"  vs our lambda = {pooled['lambda']:.4f}")
    print("      -> statsmodels' 'frac_miss_info' IS our lambda (a variance-share ratio),")
    print("         NOT the Barnard-Rubin-corrected fmi that mice::pool() reports; statsmodels")
    print("         does not expose that quantity at all.")

    # (b) A genuinely independent library MICE chain, as specced literally:
    # mice.MICE(...).fit(n_imputations=20, n_burnin=10). It continues the
    # MCMC state of `imp` with its own burn-in/skip schedule, so it draws a
    # DIFFERENT set of m completed datasets than step 1-3 used, even at the
    # same seed. Expect Qbar/SE close but not identical -- and no Barnard-
    # Rubin df at all: statsmodels' MICEResults runs a plain Wald z-test.
    mi = mice.MICE(MODEL, sm.Logit, imp, fit_kwds={"disp": 0})
    mi_res = mi.fit(n_burnin=10, n_imputations=M)
    idx2 = list(mi_res.exog_names).index("bga_ph")
    mi_Qbar, mi_SE, mi_p = mi_res.params[idx2], mi_res.bse[idx2], mi_res.pvalues[idx2]

    print("\n  (b) fresh, independent mice.MICE(...).fit(n_imputations=20, n_burnin=10):")
    print(f"      hand        Qbar={pooled['Qbar']:.4f}  SE={pooled['SE']:.4f}"
          f"  df=nu_BR={pooled['nu_br']:.1f} (t-test)  p={pooled['p']:.4f}")
    print(f"      statsmodels Qbar={mi_Qbar:.4f}  SE={mi_SE:.4f}"
          f"  df=infinity (z-test, no Barnard-Rubin df)  p={mi_p:.4f}")
    print("      These differ for two real reasons, reported as observed, not fudged:")
    print("        (i)  a fresh MCMC chain draws different completed datasets than ours;")
    print("        (ii) statsmodels' MICEResults has no degrees-of-freedom concept at all")
    print("             -- it is a simpler, always-liberal Wald z-test, not Barnard-Rubin.")


# ---------------------------------------------------------------------------
# Step 5: why the (1 + 1/m) correction exists.
# ---------------------------------------------------------------------------
def section_5_why_the_correction(pooled: dict) -> None:
    print("\n" + BANNER)
    print("5) Why the (1 + 1/m) term -- SE inflation shrinks as m grows")
    print(BANNER)
    Ubar, B = pooled["Ubar"], pooled["B"]

    # Print the intermediate quantities the exercises quote, so every number in
    # loesungen.md is reproduced by a script rather than derived on paper.
    m = pooled["m"]
    lam = pooled["lambda"]
    print(f"  Intermediates for Aufgabe 7:  (1 + 1/m)*B = {(1 + 1 / m) * B:.4f}"
          f"   lambda^2 = {lam ** 2:.4f}")
    print(f"  Setting B = 0 (what a single imputation reports):"
          f"  T = Ubar = {Ubar:.4f}   SE = {math.sqrt(Ubar):.4f}")
    print()
    print("  Reusing the SAME Ubar and B measured above (from m=20), varying only")
    print("  the (1+1/m) correction factor itself:")
    print(f"  {'m':>5}{'T, no corr. (Ubar+B)':>23}{'T, corrected':>15}"
          f"{'SE, no corr.':>15}{'SE, corrected':>16}{'SE inflation':>15}")
    for m_val in (2, 5, 20, 100):
        T_plain = Ubar + B
        T_corr = Ubar + (1 + 1 / m_val) * B
        se_plain, se_corr = np.sqrt(T_plain), np.sqrt(T_corr)
        inflation = 100 * (se_corr / se_plain - 1)
        print(f"  {m_val:>5}{T_plain:>23.4f}{T_corr:>15.4f}"
              f"{se_plain:>15.4f}{se_corr:>16.4f}{inflation:>14.2f}%")
    print("\n  (1+1/m) exists because Qbar averages only m Monte Carlo draws of the")
    print("  imputation model; the correction vanishes as m -> infinity.")


# ---------------------------------------------------------------------------
# Step 6: how large must m be?
# ---------------------------------------------------------------------------
def section_6_how_large_m(work: pd.DataFrame, pooled: dict) -> None:
    print("\n" + BANNER)
    print("6) How large must m be?")
    print(BANNER)
    lam = pooled["lambda"]
    m_needed = 100 * lam
    verdict = "suffices" if M >= m_needed else "does NOT suffice"
    print(f"  White, Royston & Wood (2011) rule of thumb: m >= 100*lambda")
    print(f"  measured lambda = {lam:.4f} -> 100*lambda = {m_needed:.1f}")
    print(f"  m = {M} {verdict} the rule of thumb here.")

    print("\n  Empirically: repeat the ENTIRE pooling pipeline for m in {2,5,10,20,50},")
    print("  across REPS=10 different seeds per m, and look at how much the pooled SE")
    print("  itself varies -- SE is a statistic of a random imputation process too.")
    m_grid = [2, 5, 10, 20, 50]
    reps = 10
    print(f"  {'m':>4}{'mean(SE)':>12}{'sd(SE)':>10}{'cv(SE) %':>10}")
    for m_val in m_grid:
        ses = []
        for rep in range(reps):
            # Derived seed: SEED (=42) is the only literal seed in this file;
            # every replicate seed below is computed from it, never written
            # as a bare literal.
            seed_r = SEED * 1000 + m_val * 10 + rep
            _, _, _, Q_r, U_r = run_mice_chain(work, m_val, seed_r)
            pooled_r = rubin_pool(Q_r, U_r, m_val)
            ses.append(pooled_r["SE"])
        ses = np.array(ses)
        cv = 100 * ses.std(ddof=1) / ses.mean()
        print(f"  {m_val:>4}{ses.mean():>12.4f}{ses.std(ddof=1):>10.4f}{cv:>10.2f}")
    print("\n  The Monte Carlo sd(SE) shrinks as m grows: at small m the pooled SE you")
    print("  happen to get is noticeably seed-dependent; by m=50 it has largely settled.")


def main() -> None:
    work = load()
    imp, _datasets, fits, Q, U = section_1_2_build_and_fit(work)
    pooled = rubin_pool(Q, U, M)
    section_3_hand_rubin(pooled)
    section_4_validate(imp, fits, pooled)
    section_5_why_the_correction(pooled)
    section_6_how_large_m(work, pooled)


if __name__ == "__main__":
    main()
