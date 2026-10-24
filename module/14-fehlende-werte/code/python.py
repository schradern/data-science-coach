"""Module 14 - Missing data: when complete-case is safe, and when it is not.

Run: python module/14-fehlende-werte/code/python.py

The cohort is synthetic, so the truth behind `bga_ph` is known. That lets this
script *demonstrate* bias instead of asserting it.

Two missingness mechanisms, two different lessons:

  laktat_mmol_l   missing depends on `sofa_score` -- a covariate the analysis
                  model conditions on. Complete-case COEFFICIENTS are unbiased.
                  The marginal mean is not.

  bga_ph          missing depends on `verstorben_30d` -- the OUTCOME. The odds
                  ratios survive, but the intercept, every predicted absolute
                  risk, and the cohort's apparent mortality do not.
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
from statsmodels.imputation import mice  # noqa: E402

from lib.helpers import SEED, load_cohort, load_labs  # noqa: E402

warnings.filterwarnings("ignore")

MODEL = "verstorben_30d ~ bga_ph + alter + sofa_score"
# Reference patient for the absolute-risk comparison: pH 7.38, 64 years, SOFA 4.
REF = {"bga_ph": 7.38, "alter": 64, "sofa_score": 4}


def predicted_risk(params: pd.Series) -> float:
    """30-day risk the model predicts for the reference patient."""
    z = params["Intercept"] + sum(params[k] * v for k, v in REF.items())
    return float(1 / (1 + np.exp(-z)))


def or_per_01_ph_drop(params: pd.Series) -> float:
    """Odds ratio for a 0.1-unit fall in pH (worsening acidosis)."""
    return float(np.exp(-0.1 * params["bga_ph"]))


def load() -> pd.DataFrame:
    df = load_cohort().merge(load_labs(), on="patient_id", how="left")
    truth = pd.read_csv(ROOT / "data" / "bga_ph_wahrheit.csv")
    return df.merge(truth, on="patient_id", how="left")


def section_1_profile(df: pd.DataFrame) -> None:
    print("=" * 74)
    print("1) Missingness profile – who is missing, not just how many")
    print("=" * 74)
    for col in ["bmi", "laktat_mmol_l", "bga_ph"]:
        print(f"  {col:16s} {df[col].isna().mean():6.1%} missing")

    work = df.assign(
        laktat_fehlt=df["laktat_mmol_l"].isna().astype(int),
        bga_fehlt=df["bga_ph"].isna().astype(int),
    )
    print("\n  Regress the missingness indicator on severity and on the outcome:")
    for label, col in [("laktat_mmol_l", "laktat_fehlt"), ("bga_ph", "bga_fehlt")]:
        m = smf.logit(f"{col} ~ sofa_score + verstorben_30d", data=work).fit(disp=0)
        print(f"    {label:14s} sofa b={m.params['sofa_score']:+.3f} (p={m.pvalues['sofa_score']:.4f})"
              f"   outcome b={m.params['verstorben_30d']:+.3f} (p={m.pvalues['verstorben_30d']:.4f})")
    print("\n  -> lactate is missing by SEVERITY; the blood gas is missing by OUTCOME.")


def section_2_selected_cohort(df: pd.DataFrame) -> None:
    print("\n" + "=" * 74)
    print("2) The cohort you keep is not the cohort you had")
    print("=" * 74)
    cc = df.dropna(subset=["bga_ph"])
    print(f"  full cohort           N={len(df):3d}   30-day mortality {df['verstorben_30d'].mean():.1%}")
    print(f"  complete cases (pH)   N={len(cc):3d}   30-day mortality {cc['verstorben_30d'].mean():.1%}")
    print("  -> dropping rows with a missing blood gas silently drops the dead.")


def section_3_what_survives(df: pd.DataFrame) -> None:
    print("\n" + "=" * 74)
    print("3) Complete-case: what survives, what breaks")
    print("=" * 74)

    truth = smf.logit(MODEL, data=df.assign(bga_ph=df["bga_ph_wahr"])).fit(disp=0)
    cc = smf.logit(MODEL, data=df.dropna(subset=["bga_ph"])).fit(disp=0)

    print(f"  {'':22s}{'Intercept':>11}{'OR / -0.1 pH':>15}{'risk @ ref':>13}")
    for name, fit in [("full data (truth)", truth), ("complete-case", cc)]:
        p = fit.params
        print(f"  {name:22s}{p['Intercept']:>11.2f}{or_per_01_ph_drop(p):>15.2f}{predicted_risk(p):>13.3f}")

    r_true, r_cc = predicted_risk(truth.params), predicted_risk(cc.params)
    print("\n  odds ratio:     essentially unbiased")
    print(f"  absolute risk:  {r_cc:.3f} vs {r_true:.3f}   ({100 * (r_cc / r_true - 1):+.0f} %)")
    print("\n  Selecting on the OUTCOME shifts a logistic intercept, not its slopes.")
    print("  The odds ratios look fine while every predicted risk is too low.")


def section_4_lactate_contrast(df: pd.DataFrame) -> None:
    print("\n" + "=" * 74)
    print("4) Contrast: lactate is missing by covariate, not by outcome")
    print("=" * 74)
    cc = df.dropna(subset=["laktat_mmol_l"])
    fit = smf.logit("verstorben_30d ~ laktat_mmol_l + alter + sofa_score", data=cc).fit(disp=0)
    lo, hi = fit.conf_int().loc["laktat_mmol_l"]
    print(f"  complete-case b(laktat) = {fit.params['laktat_mmol_l']:+.3f}"
          f"   (95 % CI {lo:+.3f} to {hi:+.3f})")
    print("  The driver of the missingness (sofa_score) is IN the model, so this")
    print("  coefficient is unbiased. Complete-case is the correct analysis here.")
    print("\n  The marginal summary, however, is not:")
    print(f"    mean lactate over observed values only : {df['laktat_mmol_l'].mean():.2f} mmol/l")
    print("    Sicker patients get lactate drawn more often, so a 'Table 1' mean")
    print("    computed on the observed values overstates the cohort's true mean.")


def section_5_imputation(df: pd.DataFrame) -> None:
    print("\n" + "=" * 74)
    print("5) Median imputation attenuates. Multiple imputation does not.")
    print("=" * 74)

    truth = smf.logit(MODEL, data=df.assign(bga_ph=df["bga_ph_wahr"])).fit(disp=0)
    cc = smf.logit(MODEL, data=df.dropna(subset=["bga_ph"])).fit(disp=0)

    med = df.copy()
    med["bga_ph"] = med["bga_ph"].fillna(med["bga_ph"].median())
    fit_med = smf.logit(MODEL, data=med).fit(disp=0)

    # Multiple imputation. The imputation model MUST contain the outcome:
    # that is precisely what makes this missingness ignorable (MAR).
    np.random.seed(SEED)
    work = df[["verstorben_30d", "bga_ph", "alter", "sofa_score"]].copy()
    imputer = mice.MICEData(work)
    imputer.set_imputer("bga_ph", "alter + sofa_score + verstorben_30d")
    # fit_kwds reaches each per-imputation Logit fit. Without disp=0 the optimizer
    # prints a convergence banner 20 times and buries the table below.
    mi_fit = mice.MICE(MODEL, sm.Logit, imputer, fit_kwds={"disp": 0}).fit(
        n_imputations=20, n_burnin=10)
    mi_params = pd.Series(np.asarray(mi_fit.params), index=truth.params.index)
    mi_se = pd.Series(np.asarray(mi_fit.bse), index=truth.params.index)

    print(f"  {'':28s}{'b(bga_ph)':>12}{'SE':>8}{'OR / -0.1 pH':>15}{'risk @ ref':>13}")
    rows = [
        ("full data (truth)", truth.params, truth.bse),
        ("complete-case", cc.params, cc.bse),
        ("median imputation", fit_med.params, fit_med.bse),
        ("multiple imputation (m=20)", mi_params, mi_se),
    ]
    for name, p, se in rows:
        print(f"  {name:28s}{p['bga_ph']:>12.2f}{se['bga_ph']:>8.2f}"
              f"{or_per_01_ph_drop(p):>15.2f}{predicted_risk(p):>13.3f}")

    r_true = predicted_risk(truth.params)
    print("\n  Absolute risk, relative error against the truth:")
    for name, p, _ in rows[1:]:
        print(f"    {name:28s} {100 * (predicted_risk(p) / r_true - 1):+5.0f} %")

    n_imputed = int(df["bga_ph"].isna().sum())
    print(f"\n  Median imputation invents {n_imputed} patients with an average pH and shrinks")
    print("  the effect toward zero (regression dilution). It also claims a certainty")
    print("  it does not have – note its standard error. Rubin's rules add the")
    print("  between-imputation variance back in.")


def section_5b_sabotaged_imputation(df: pd.DataFrame) -> None:
    """Exercise 6 quotes these numbers; compute them here rather than on paper.

    Drop the outcome from the imputation model and the missingness stops being
    ignorable: the pH-outcome association is diluted away, exactly as median
    imputation dilutes it. The absolute risk, meanwhile, survives.
    """
    print("\n" + "=" * 74)
    print("5b) What happens if the outcome is left OUT of the imputation model")
    print("=" * 74)

    truth = smf.logit(MODEL, data=df.assign(bga_ph=df["bga_ph_wahr"])).fit(disp=0)

    def pooled(imputer_formula: str) -> pd.Series:
        np.random.seed(SEED)
        work = df[["verstorben_30d", "bga_ph", "alter", "sofa_score"]].copy()
        imputer = mice.MICEData(work)
        imputer.set_imputer("bga_ph", imputer_formula)
        fit = mice.MICE(MODEL, sm.Logit, imputer, fit_kwds={"disp": 0}).fit(
            n_imputations=20, n_burnin=10)
        return pd.Series(np.asarray(fit.params), index=truth.params.index)

    with_outcome = pooled("alter + sofa_score + verstorben_30d")
    without_outcome = pooled("alter + sofa_score")

    print(f"  {'':34s}{'OR / -0.1 pH':>14}{'risk @ ref':>13}")
    for name, p in [("full data (truth)", truth.params),
                    ("MI, outcome IN imputation model", with_outcome),
                    ("MI, outcome OMITTED", without_outcome)]:
        print(f"  {name:34s}{or_per_01_ph_drop(p):>14.2f}{predicted_risk(p):>13.3f}")
    print("\n  Omitting the outcome breaks the ODDS RATIO, not the absolute risk –")
    print("  the mirror image of what complete-case analysis does (section 3).")


def section_6_report(df: pd.DataFrame) -> None:
    print("\n" + "=" * 74)
    print("6) The sentence that belongs in the paper")
    print("=" * 74)
    n_miss = int(df["bga_ph"].isna().sum())
    pct = df["bga_ph"].isna().mean()
    print(f'  "Der arterielle pH fehlte bei {n_miss} von {len(df)} Patient:innen ({pct:.1%}).')
    print("   Das Fehlen hing mit der 30-Tage-Mortalität zusammen, nicht mit Alter")
    print("   oder SOFA-Score. Die Hauptanalyse verwendet multiple Imputation")
    print("   (m = 20, Rubins Regeln; das Imputationsmodell enthält das Outcome).")
    print("   Eine Complete-Case-Analyse als Sensitivitätsanalyse lieferte")
    print('   vergleichbare Odds Ratios, aber zu niedrige absolute Risiken."')


def main() -> None:
    df = load()
    section_1_profile(df)
    section_2_selected_cohort(df)
    section_3_what_survives(df)
    section_4_lactate_contrast(df)
    section_5_imputation(df)
    section_5b_sabotaged_imputation(df)
    section_6_report(df)


if __name__ == "__main__":
    main()
