"""Module 18 - mixed-effects models for longitudinal data.

Uses the shared cohort's repeated MAP measurements (`vitalwerte.csv`, 1-4
measurements per patient - an unbalanced design, as real ICU data usually
is) merged with baseline characteristics (`kohorte.csv`). Demonstrates why
naive OLS on repeated measures can get the *standard error* wrong even when
the point estimate looks unremarkable - that's the part the printed
coefficients alone would hide.

Run from project root:
    python module/18-longitudinal-mixed-models/code/python.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402

from lib.helpers import load_cohort, load_vitals  # noqa: E402

FORMULA = "map_mmhg ~ tag + diabetes"


def build_longitudinal_data() -> pd.DataFrame:
    """Merge repeated MAP measurements with baseline patient characteristics."""
    vitals = load_vitals()
    cohort = load_cohort()[["patient_id", "sofa_score", "alter", "diabetes"]]
    return vitals.merge(cohort, on="patient_id", how="left")


def report_incomplete_series(data: pd.DataFrame) -> None:
    """Break incomplete measurement series down by cause: early discharge
    (a patient still alive who simply left before day 4 -- MAR/MCAR-style
    missingness, an LMM handles this fine) vs. death within the 30-day
    follow-up (the day-3/day-4 MAP does not exist for that patient -- this
    is truncation by death, not missingness; see README section 4).
    """
    cohort = load_cohort()[["patient_id", "verweildauer_tage", "verstorben_30d"]]
    n_per_patient = data.groupby("patient_id").size()
    n_max = n_per_patient.max()
    incomplete_ids = n_per_patient[n_per_patient < n_max].index
    incomplete = cohort[cohort["patient_id"].isin(incomplete_ids)]
    n_incomplete = len(incomplete)
    n_died = int(incomplete["verstorben_30d"].sum())
    n_discharged = n_incomplete - n_died
    print(f"Rows: {len(data)} | patients: {data['patient_id'].nunique()} | "
          f"measurements per patient: {n_per_patient.min()}-{n_max} "
          f"({n_incomplete} patients have fewer than {n_max})")
    print(f"  Of these {n_incomplete} incomplete series: {n_discharged} are patients who "
          f"survived (early discharge, verweildauer_tage < {n_max} -- ordinary MAR/MCAR-style "
          "missingness) and")
    print(f"  {n_died} are patients who died within 30 days (verstorben_30d == 1 -- "
          "truncation by death, NOT missingness; see the pitfall in the README).")


def main() -> None:
    data = build_longitudinal_data()

    print("=== Repeated MAP measurements (long format) ===")
    print(data.head())
    report_incomplete_series(data)

    ols = smf.ols(FORMULA, data=data).fit()
    mixed = smf.mixedlm(FORMULA, data=data, groups=data["patient_id"]).fit(reml=True)

    print("\n=== Naive OLS (treats every row as independent) ===")
    print(pd.DataFrame({"coef": ols.params, "se": ols.bse}).round(3))

    print("\n=== Random-intercept mixed model (accounts for repeated measures per patient) ===")
    mixed_out = pd.DataFrame({"coef": mixed.params, "se": mixed.bse}).drop("Group Var")
    print(mixed_out.round(3))
    group_var = float(mixed.cov_re.iloc[0, 0])
    icc = group_var / (group_var + mixed.scale)
    print(f"Patient-level (random intercept) variance: {group_var:.3f} | "
          f"residual variance: {mixed.scale:.3f} | ICC = {icc:.3f}")

    print("\n=== Where the point estimates agree but the standard errors don't ===")
    for term, kind in [("tag", "time-varying (changes within a patient)"),
                       ("diabetes", "patient-level (constant across a patient's 1-4 rows)")]:
        ratio = mixed.bse[term] / ols.bse[term]
        print(f"{term:10s} [{kind}]:")
        print(f"    OLS coef={ols.params[term]: .3f} se={ols.bse[term]:.3f} | "
              f"mixed coef={mixed.params[term]: .3f} se={mixed.bse[term]:.3f} "
              f"(mixed/OLS SE ratio = {ratio:.2f}x)")
    print("\nInterpretation: for the patient-level predictor `diabetes`, every row of the same")
    print("patient repeats the same value. OLS treats those repeats as independent new evidence")
    print("(pseudo-replication), so its SE for `diabetes` is too small - the mixed model corrects")
    print("this. For the time-varying predictor `tag`, the naive SE is not automatically too small;")
    print("whether OLS over- or understates it depends on how within-patient the variation is.")
    print("Either way, the mixed model's SE is the trustworthy one because it is derived from the")
    print("correct error structure instead of assuming independence it doesn't have.")
    print("\nEstimand caveat: this model is fit on whoever HAS a MAP measurement on a given day.")
    print("For patients who died before day 3, that measurement does not exist -- so what the")
    print("model actually estimates is the mean MAP trajectory among patients still alive and")
    print("in hospital, not among everyone admitted. No missing-data assumption (MCAR/MAR) fixes")
    print("that; it is a truncation-by-death problem, not a missingness problem.")

    print("\n=== Random slope: does the MAP trend over time also vary by patient? ===")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        smf.mixedlm(FORMULA, data=data, groups=data["patient_id"], re_formula="~tag").fit(reml=True)
    if any("singular" in str(w.message).lower() for w in caught):
        print("Random-slope covariance is singular: with only up to 4 measurements per patient,")
        print("the data don't support estimating a patient-specific day-trend on top of the random")
        print("intercept - this is the 'Singular Fit' pitfall below, seen live. The simpler")
        print("random-intercept-only model above is the right choice here.")
    else:
        print("Random-slope model converged without a singular-fit warning.")


if __name__ == "__main__":
    main()
