"""Module 12 – Regression models: OLS, logistic regression, and Cox regression.

Runs standalone from the project root:
    python module/12-regression/code/python.py

Data: read from data/ (committed with the repo); if that folder is
missing, the same files are fetched from the published URL.
Packages: statsmodels, lifelines  (pip install statsmodels lifelines)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from lifelines import CoxPHFitter  # noqa: E402

from lib.ground_truth import (  # noqa: E402
    true_hazard_ratios,
    true_odds_ratios,
    true_odds_ratios_for,
)
from lib.helpers import SEED, load_cohort  # noqa: E402

pd.set_option("display.width", 100)
pd.set_option("display.float_format", "{:.4f}".format)

LOGIT_FORMULA = (
    "verstorben_30d ~ age_centred + sofa_score + crp_mg_l"
    " + diabetes + sepsis + active_smoker"
)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive binary indicators and centre age as in the data generator.

    Column names that stay German (clinical schema):
        raucherstatus, aufnahmegrund, alter, verweildauer_tage, verstorben_30d,
        sofa_score, crp_mg_l, diabetes, hypertonie
    New derived columns use English names to follow code conventions.
    """
    df = df.copy()
    df["active_smoker"] = (df["raucherstatus"] == "aktiv").astype(int)
    df["sepsis"]        = (df["aufnahmegrund"] == "Sepsis").astype(int)
    # Centre age on 64 – matches the synthetic data generator intercept
    df["age_centred"]   = df["alter"] - 64
    return df


# ---------------------------------------------------------------------------
# Section 1 – OLS: length of stay
# ---------------------------------------------------------------------------

def fit_ols(df: pd.DataFrame) -> None:
    """Ordinary Least Squares for a continuous outcome (length of stay).

    Note: verweildauer_tage is right-skewed; OLS is illustrative here.
    A Gamma-GLM would be more appropriate for production use.
    """
    print("=" * 65)
    print("SECTION 1 – LINEAR REGRESSION: length of stay (days)")
    print("=" * 65)
    print("Note: length of stay is right-skewed; OLS is illustrative.")
    print("A Gamma-GLM or log-transform would be more appropriate.\n")

    formula = "verweildauer_tage ~ sofa_score + age_centred + sepsis + diabetes"
    result = smf.ols(formula, data=df).fit()

    print(f"R²: {result.rsquared:.4f}   Adjusted R²: {result.rsquared_adj:.4f}\n")

    coef_table = pd.DataFrame({
        "β̂":            result.params,
        "SE":            result.bse,
        "95% CI lower":  result.conf_int()[0],
        "95% CI upper":  result.conf_int()[1],
        "p":             result.pvalues,
    })
    print("Coefficients:")
    print(coef_table.to_string())

    sofa_coef = result.params["sofa_score"]
    print(
        f"\nInterpretation: each additional SOFA point is associated with"
        f" {sofa_coef:.2f} extra days of stay, adjusting for the other predictors."
    )

    # The sepsis/diabetes rows above are DIRECT effects – they hold sofa_score
    # fixed, and sofa_score is a mediator. Reading "sepsis: p = 0.42" as "sepsis
    # does not prolong the stay" is exactly the overadjustment error §5 warns
    # about. Show the total effect next to it instead of leaving the trap open.
    total = smf.ols("verweildauer_tage ~ age_centred + sepsis + diabetes", data=df).fit()
    print("\nDirect vs. total effect (sofa_score is a MEDIATOR, see README §4):")
    print(f"{'':12s}{'direct (sofa in model)':>26}{'total (sofa omitted)':>24}")
    for term in ("sepsis", "diabetes"):
        print(f"{term:12s}"
              f"{f'{result.params[term]:+.2f} d (p={result.pvalues[term]:.3f})':>26}"
              f"{f'{total.params[term]:+.2f} d (p={total.pvalues[term]:.3f})':>24}")
    print("Sepsis lengthens the stay entirely THROUGH the SOFA score. Adjusting")
    print("for SOFA removes the effect you were asking about.")


# ---------------------------------------------------------------------------
# Section 2 – Logistic regression: 30-day mortality
# ---------------------------------------------------------------------------

def fit_logistic(df: pd.DataFrame) -> None:
    """Logistic regression for a binary outcome (30-day mortality).

    Reports:
    - Estimated log-odds coefficients (beta_hat)
    - Odds ratios with 95 % confidence intervals
    - Truth comparison against the known generative betas
    - Model fit statistics (pseudo-R², AIC, EPV)
    """
    print("\n" + "=" * 65)
    print("SECTION 2 – LOGISTIC REGRESSION: 30-day mortality")
    print("=" * 65)

    result = smf.logit(LOGIT_FORMULA, data=df).fit(disp=False)

    # Odds ratios and 95 % CI on the exponentiated scale
    ci = result.conf_int()
    or_table = pd.DataFrame({
        "Log-odds (β̂)":   result.params,
        "OR":              np.exp(result.params),
        "OR 95% CI lower": np.exp(ci[0]),
        "OR 95% CI upper": np.exp(ci[1]),
        "p":               result.pvalues,
    })
    print("\nEstimated coefficients and odds ratios:")
    print(or_table.to_string())

    # ------------------------------------------------------------------
    # Truth comparison – against the true ODDS ratios.
    #
    # The generative model in data/generate_data.py is a Weibull proportional-
    # hazards process, so its coefficients are log-HAZARDS. Exponentiating them
    # gives true hazard ratios, not the odds ratios this logistic model
    # estimates. lib/ground_truth.py obtains the true ORs by replaying the same
    # data-generating process at N = 2 000 000 and fitting this exact model.
    # ------------------------------------------------------------------
    true_or = true_odds_ratios()
    true_hr = true_hazard_ratios()["direct"]

    print("\n--- Truth comparison: estimated OR vs. true OR ---")
    print(f"{'Predictor':<16}{'OR_hat':>10}{'95% CI':>20}{'true OR':>10}"
          f"{'true HR':>10}{'covered?':>10}")
    print("-" * 76)
    covered = 0
    for predictor, or_true in true_or.items():
        est = float(np.exp(result.params[predictor]))
        lo, hi = float(np.exp(ci.loc[predictor, 0])), float(np.exp(ci.loc[predictor, 1]))
        hit = lo <= or_true <= hi
        covered += hit
        print(f"{predictor:<16}{est:>10.3f}{f'[{lo:.3f}, {hi:.3f}]':>20}"
              f"{or_true:>10.3f}{true_hr[predictor]:>10.3f}{'yes' if hit else 'NO':>10}")

    print(
        f"\nThe 95 % CI covers the true OR for {covered}/{len(true_or)} predictors."
        "\nRemaining deviations are sampling variance (N=500, 78 events), not bias."
        "\n"
        "\nWhy two 'true' columns? The cohort is generated by a Weibull hazard"
        "\nmodel, so exp(beta) is a true HAZARD ratio (compare it in Module 17's"
        "\nCox model). A logistic model estimates an ODDS ratio, which at a 15.6 %"
        "\nevent rate sits systematically further from 1. Comparing an estimated OR"
        "\nagainst a true HR would make an unbiased estimate look biased."
    )

    # Model fit
    n_events = int(df["verstorben_30d"].sum())
    n_pred = len(true_or)                  # intercept is not a predictor
    epv = n_events / n_pred
    print(f"\nPseudo-R² (McFadden): {result.prsquared:.4f}")
    print(f"Log-likelihood:       {result.llf:.2f}")
    print(f"AIC:                  {result.aic:.2f}")
    print(f"\nEvents per variable (EPV): {n_events} events / {n_pred} predictors = {epv:.1f}")
    if epv < 10:
        print("  WARNING: EPV < 10 – model may be over-parametrised.")
    else:
        print("  EPV >= 10 – model complexity is defensible.")


def follow_up_sensitivity(df: pd.DataFrame) -> None:
    """Refit the logistic model on patients whose 30-day outcome is known.

    `verstorben_30d` counts an observed death only: a patient lost to
    follow-up on day 10 enters as 0 although a death on day 20 would go
    unseen. `fu_vollstaendig_30d` marks who was followed to day 30 (or died
    before). If the ORs barely move without the incomplete cases, the
    0-coding does not drive the result; if they move, a survival model
    (Section 3, Module 17) is the honest analysis.
    """
    complete = df[df["fu_vollstaendig_30d"] == 1]
    n_dropped = len(df) - len(complete)
    full = smf.logit(LOGIT_FORMULA, data=df).fit(disp=False)
    known = smf.logit(LOGIT_FORMULA, data=complete).fit(disp=False)
    print(f"\n--- Sensitivity: only patients with complete 30-day follow-up "
          f"({len(complete)} of {len(df)}, {n_dropped} dropped) ---")
    print(f"{'Predictor':<16}{'OR all':>10}{'OR complete':>14}")
    for predictor in full.params.index.drop("Intercept"):
        print(f"{predictor:<16}{np.exp(full.params[predictor]):>10.3f}"
              f"{np.exp(known.params[predictor]):>14.3f}")
    print(f"Deaths among the {n_dropped} dropped patients: "
          f"{int(df.loc[df['fu_vollstaendig_30d'] == 0, 'verstorben_30d'].sum())} "
          "(by construction: an unobserved death cannot be counted).")


# ---------------------------------------------------------------------------
# Section 3 – Cox regression: time-to-event analysis
# ---------------------------------------------------------------------------

def confounder_vs_mediator(df: pd.DataFrame) -> None:
    """Crude -> +confounder -> +mediator, for sepsis and for diabetes.

    README §4 and the exercises quote these numbers. Printing them here means no
    number in the lesson exists only on paper: the reader can reproduce every
    cell, and a future edit that breaks one is caught by tools/check_numbers.py.
    """
    print("\n" + "=" * 65)
    print("SECTION 2b – CONFOUNDER vs. MEDIATOR (README §4, Aufgabe 3)")
    print("=" * 65)
    print("alter confounds diabetes (DAG arrow alter -> diabetes) but not sepsis")
    print("(no arrow alter -> sepsis; the sepsis OR still moves because ORs are")
    print("non-collapsible). sofa_score is a MEDIATOR (do not adjust, if the total")
    print("effect is the question).")
    print("True ORs replayed at N = 400 000 (Monte-Carlo error ~1 %, far below the")
    print("width of the N=500 confidence intervals they sit next to).\n")

    # Never hardcode a truth: recompute it from the generating process, at a
    # cheaper N because these auxiliary values are compared against N=500 CIs
    # that are orders of magnitude wider than the Monte-Carlo error.
    spezifikationen = [
        ("crude", ()),
        ("+ alter", ("age_centred",)),
        ("+ alter + sofa", ("age_centred", "sofa_score")),
    ]

    for exposure in ("sepsis", "diabetes"):
        print(f"  {exposure}")
        print(f"    {'model':18s}{'OR_hat':>9}{'95% CI':>20}{'true OR':>10}")
        for label, extra in spezifikationen:
            terms = " + ".join((exposure,) + extra)
            fit = smf.logit(f"verstorben_30d ~ {terms}", data=df).fit(disp=False)
            est = float(np.exp(fit.params[exposure]))
            lo, hi = np.exp(fit.conf_int().loc[exposure])
            wahr = true_odds_ratios_for((exposure,) + extra, n=400_000)[exposure]
            print(f"    {label:18s}{est:>9.2f}{f'[{lo:.2f}, {hi:.2f}]':>20}{wahr:>10.3f}")
        print()

    print("Sepsis: 5.75 -> 1.92 when the mediator enters; against the true total effect")
    print("the overadjustment bias is demonstrable (both CIs come from the same data, so")
    print("their overlap is no formal test). Diabetes shows the same pattern,")
    print("but its CI [0.52, 1.87] contains both the true total and direct effects:")
    print("at 78 events the bias can be derived, not shown.")


def fit_cox(df: pd.DataFrame) -> None:
    """Cox proportional-hazards regression for time-to-event data.

    Uses fu_zeit_tage (follow-up time until event/censoring) as the time
    variable and status (1 = died, 0 = censored) as the event indicator.
    verweildauer_tage (length of stay) is NOT a time-to-event variable, it
    is generated independently of the survival process – see data/README.md.
    """
    print("\n" + "=" * 65)
    print("SECTION 3 – COX REGRESSION: time-to-event analysis")
    print("=" * 65)
    print(
        "Duration: fu_zeit_tage | Event: status (1 = died, 0 = censored)"
        "\nPatients without the event are right-censored (not a data loss).\n"
    )

    cox_cols = [
        "fu_zeit_tage", "status",
        "sofa_score", "age_centred", "sepsis", "active_smoker",
    ]
    cox_df = df[cox_cols].copy()

    cph = CoxPHFitter()
    cph.fit(
        cox_df,
        duration_col="fu_zeit_tage",
        event_col="status",
    )

    summary_cols = ["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]
    hr_table = cph.summary[summary_cols].copy()
    hr_table.columns = ["HR", "HR 95% CI lower", "HR 95% CI upper", "p"]
    print("Hazard ratios (HR) with 95 % CI:")
    print(hr_table.to_string())

    print(
        "\nHR interpretation:"
        "\n  HR > 1 → elevated hazard (event occurs sooner / more often)"
        "\n  HR < 1 → reduced hazard (protective)"
        "\n  HR = 1 → no difference from reference"
        "\n\nNote: the proportional-hazards assumption should be checked with"
        "\nSchoenfeld residuals – use cph.check_assumptions(cox_df) in lifelines."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    df_raw = load_cohort()
    df = prepare_features(df_raw)

    mortality_rate = df["verstorben_30d"].mean()
    print(f"Cohort: {len(df)} patients | 30-day mortality: {mortality_rate:.1%}")
    print(
        "This dataset has a KNOWN ground truth: a Weibull proportional-hazards"
        "\nprocess (data/generate_data.py). Its coefficients are log-HAZARDS, so"
        "\nwe compare the Cox model against them directly, and the logistic model"
        "\nagainst the true ODDS ratios from lib/ground_truth.py."
    )

    fit_ols(df)
    fit_logistic(df)
    follow_up_sensitivity(df)
    confounder_vs_mediator(df)
    fit_cox(df)

    print("\nDone.")


if __name__ == "__main__":
    main()
