"""The cohort's true effects — on the scale you are actually estimating.

`data/generate_data.py` draws survival times from a **Weibull proportional-
hazards** model. Its coefficients therefore live on the **log-hazard** scale:

    eta = -1.95 + 0.045*(alter-64) + 0.28*sofa_score + 0.0035*crp_mg_l
                + 0.45*diabetes + 0.6*(aufnahmegrund=="Sepsis")
                + 0.4*(raucherstatus=="aktiv")

Two consequences that are easy to get wrong, and that this module exists to
prevent:

1.  exp(beta) is a true **hazard ratio**, NOT a true odds ratio. A Cox model
    (Module 17) may be compared with exp(beta) directly. A logistic model
    (Module 12) may not: with a 15.6 % event rate the true OR is systematically
    further from 1 than the true HR. There is no closed form for the true OR of
    a discretised Weibull process, so `true_odds_ratios()` obtains it the only
    honest way -- by replaying the data-generating process at large N.

2.  The coefficients above are **direct** effects, i.e. conditional on
    `sofa_score`. Because `sofa_score` is a mediator of diabetes and sepsis,
    the **total** effect of those two is larger. `true_hazard_ratios()` returns
    both, so a lesson never has to guess which one it means.

Everything here is deterministic given SEED_TRUTH, but "deterministic" is not
"exact": these are Monte-Carlo estimates of the truth and they carry a
Monte-Carlo error. Order of magnitude, as relative standard deviation across
seeds — bounds rounded UP, because a bound measured on five seeds is itself an
estimate and quoting it to three digits would be the very false precision this
module exists to prevent:

    quantity                       continuous predictors   binary predictors
    true ORs        (N = 2e6)      < 0.2 %                 < 0.6 %
    true HRs direct (N = 2e5)      < 0.2 %                 < 2 %
    true HRs total  (N = 2e5)      < 0.2 %                 < 2 %

The binary predictors (diabetes, sepsis, active_smoker) converge slowly at a
15.6 % event rate; the continuous ones converge fast. The third decimal of a
binary predictor's hazard ratio is therefore noise, not signal.

That is acceptable here and nowhere else. Every lesson uses these numbers for
exactly one purpose: to ask whether an N = 500 confidence interval — 20 to 40 %
wide — covers the truth. A 2 % error in the target cannot change that answer.
Do not reuse them where 2 % matters.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# The data-generating process. `data/generate_data.py` imports every constant
# below, so the generator and this replay of it cannot drift apart. They did
# drift once (see the covariate block), and a logistic OR is non-collapsible,
# so the "true" ORs were silently wrong by up to 3.7 %.
#
# Change a number here and you change the cohort. Nothing else defines them.
# ---------------------------------------------------------------------------

# Log-HAZARD scale, direct effects (i.e. conditional on sofa_score).
TRUE_LOG_HAZARD: dict[str, float] = {
    "age_centred": 0.045,
    "sofa_score": 0.28,
    "crp_mg_l": 0.0035,
    "diabetes": 0.45,
    "sepsis": 0.60,
    "active_smoker": 0.40,
}

SURV_B0 = -1.95              # intercept of the linear predictor eta
BASELINE_SCALE = 115.0       # Weibull scale at eta = 0
WEIBULL_SHAPE = 1.3          # Weibull shape k
SEED_TRUTH = 20240601

# N is chosen per estimator, not per taste. The Cox fits cost ~20 s at N = 2e6
# and Module 12's lesson script calls them, so the hazard ratios stay at 2e5 and
# accept the ~1.3 % error documented above. The logistic fit is cheap, and the
# odds ratios are the numbers Module 12 compares its own estimates against, so
# they get 2e6 (worst case 0.51 %, down from 1.5 % at 2e5).
N_TRUTH_OR = 2_000_000
N_TRUTH_HR = 200_000

# Written by `python lib/ground_truth.py`. `tools/check_repo.py` compares the
# literals hardcoded in Module 12's r.R against it, because R cannot import
# this module and a second copy of a number is a second chance to be wrong.
TRUTH_SNAPSHOT = Path(__file__).resolve().parent / "ground_truth.json"

# ---------------------------------------------------------------------------
# Blood-gas pH (Module 14). Verbatim from data/generate_data.py's
# `_bga_ph_truth_and_observed` -- single source of truth so the Module-12
# Monte-Carlo benchmark and the data generator can never drift apart.
# ---------------------------------------------------------------------------
BGA_INTERCEPT = 7.40
BGA_SOFA = -0.006
BGA_DEAD = -0.024
BGA_NOISE_SD = 0.050
BGA_MISS_B0 = -1.7      # logit of missingness among survivors
BGA_MISS_DEAD = 1.4     # extra logit of missingness among the deceased

# ---------------------------------------------------------------------------
# Covariate constants. `data/generate_data.py` imports these, and `_replay_dgp`
# below uses the same names, so the two cannot drift apart.
#
# They did drift once: the replay drew age ~ N(64, 14) clipped to [18, 95] with
# a Sepsis share of 0.20, while the generator used N(64, 16) clipped to [18, 98]
# and 0.18. A logistic odds ratio is **not collapsible** — it depends on the
# covariate distribution — so the "true ORs" handed to Module 12 were wrong by
# up to ~3.7 %. Silently. Constants now live here and only here.
# ---------------------------------------------------------------------------
AGE_MEAN, AGE_SD, AGE_MIN, AGE_MAX = 64, 16, 18, 98

REASONS = ["Pneumonie", "Sepsis", "Herzinsuffizienz", "COPD-Exazerbation", "Sonstiges"]
REASON_P = [0.26, 0.18, 0.22, 0.16, 0.18]

SMOKING = ["nie", "ehemalig", "aktiv"]
SMOKING_P = [0.45, 0.33, 0.22]

DIAB_B0, DIAB_SLOPE, DIAB_AGE_REF = 0.05, 0.006, 40
DIAB_MIN, DIAB_MAX = 0.02, 0.45

SOFA_POISSON_MEAN, SOFA_SEPSIS_BONUS = 3, 3
SOFA_MIN, SOFA_MAX = 0, 20

CRP_SHAPE, CRP_SCALE, CRP_MIN, CRP_MAX = 2.0, 35, 1, 400

ADMIN_HORIZON_DAYS = 30
LOST_TO_FU_RATE = 0.15
CENSOR_LO, CENSOR_HI = 3, 40


def _replay_dgp(n: int, seed: int) -> pd.DataFrame:
    """Replays data/generate_data.py's cohort at arbitrary N.

    Only the draws that enter the survival model are replayed (sex, height,
    weight and hypertension do not). The RNG stream therefore differs from the
    generator's, which is irrelevant: this function is seeded independently and
    is used only to estimate *population* quantities, never to reproduce the
    seed-42 sample. What must match — and now does, by construction — are the
    marginal distributions of every covariate that enters `eta`.
    """
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(AGE_MEAN, AGE_SD, n), AGE_MIN, AGE_MAX).round()
    p_diab = np.clip(DIAB_B0 + (age - DIAB_AGE_REF) * DIAB_SLOPE, DIAB_MIN, DIAB_MAX)
    diabetes = (rng.random(n) < p_diab).astype(int)
    reason = rng.choice(REASONS, n, p=REASON_P)
    smoking = rng.choice(SMOKING, n, p=SMOKING_P)
    sofa = np.clip(rng.poisson(SOFA_POISSON_MEAN, n) + diabetes
                   + (reason == "Sepsis") * SOFA_SEPSIS_BONUS, SOFA_MIN, SOFA_MAX)
    crp = np.clip(rng.gamma(CRP_SHAPE, CRP_SCALE, n), CRP_MIN, CRP_MAX).round(1)

    sepsis = (reason == "Sepsis").astype(int)
    active_smoker = (smoking == "aktiv").astype(int)
    eta = (SURV_B0
           + TRUE_LOG_HAZARD["age_centred"] * (age - AGE_MEAN)
           + TRUE_LOG_HAZARD["sofa_score"] * sofa
           + TRUE_LOG_HAZARD["crp_mg_l"] * crp
           + TRUE_LOG_HAZARD["diabetes"] * diabetes
           + TRUE_LOG_HAZARD["sepsis"] * sepsis
           + TRUE_LOG_HAZARD["active_smoker"] * active_smoker)

    scale = BASELINE_SCALE * np.exp(-eta / WEIBULL_SHAPE)
    t_event = scale * (-np.log(rng.random(n))) ** (1 / WEIBULL_SHAPE)
    lost = rng.random(n) < LOST_TO_FU_RATE
    censor = np.where(lost, rng.uniform(CENSOR_LO, CENSOR_HI, n), np.inf)
    fu = np.clip(np.minimum(np.minimum(t_event, censor), ADMIN_HORIZON_DAYS), 1, None)
    status = (t_event <= np.minimum(censor, ADMIN_HORIZON_DAYS)).astype(int)
    fu_days = np.round(fu)

    return pd.DataFrame({
        "fu_zeit_tage": fu_days,
        "status": status,
        "verstorben_30d": ((status == 1) & (fu_days <= ADMIN_HORIZON_DAYS)).astype(int),
        "age_centred": age - AGE_MEAN,
        "sofa_score": sofa,
        "crp_mg_l": crp,
        "diabetes": diabetes,
        "sepsis": sepsis,
        "active_smoker": active_smoker,
    })


def replay_cohort_with_ph(n: int, seed) -> pd.DataFrame:
    """One fresh cohort from the DGP, with true pH and the observed (masked) pH.

    Replays `_replay_dgp(n, seed)` and attaches `alter` (uncentred age) plus
    the blood-gas pH mechanism from `data/generate_data.py`'s
    `_bga_ph_truth_and_observed`: `bga_ph_wahr` is the true pH for every
    patient, `bga_ph` is the same value with MAR-on-the-outcome missingness
    applied (NaN where missing).

    `seed` is expanded into two independent child streams via
    `np.random.SeedSequence.spawn` -- one for `_replay_dgp`, one (a single
    `np.random.default_rng`) for BOTH the pH noise and the missingness mask.
    This is not optional: naively reusing the same `seed` for a second,
    separately-constructed `np.random.default_rng` would replay the exact
    same underlying draw sequence `_replay_dgp` already consumed internally
    (age, diabetes, sofa, ...), making the "pH noise" numerically identical
    (up to scale) to `alter`'s draws -- a spurious deterministic collinearity
    between `bga_ph` and `alter` that causes the downstream logistic fit to
    hit quasi-perfect separation. Spawned child seed sequences are
    statistically independent by construction, so this cannot happen, while
    the whole replicate (cohort + pH + mask) stays fully and reproducibly
    determined by `seed`.
    """
    seed_seq = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    child_dgp, child_ph = seed_seq.spawn(2)

    df = _replay_dgp(n, child_dgp)
    df["alter"] = df["age_centred"] + AGE_MEAN

    rng = np.random.default_rng(child_ph)
    sofa = df["sofa_score"].to_numpy()
    dead = df["verstorben_30d"].to_numpy()

    truth = BGA_INTERCEPT + BGA_SOFA * sofa + BGA_DEAD * dead
    truth = truth + rng.normal(0, BGA_NOISE_SD, n)
    truth = np.clip(truth, 6.90, 7.60).round(3)

    p_missing = 1 / (1 + np.exp(-(BGA_MISS_B0 + BGA_MISS_DEAD * dead)))
    observed = truth.astype(float).copy()
    observed[rng.random(n) < p_missing] = np.nan

    df["bga_ph_wahr"] = truth
    df["bga_ph"] = observed
    return df


@lru_cache(maxsize=32)
def true_odds_ratios_for(terms: tuple[str, ...], n: int = 0) -> dict[str, float]:
    """True ORs of `verstorben_30d ~ <terms>` under the data-generating process.

    A true OR depends on which covariates the model adjusts for, so every caller
    must state its own specification. `alter` is accepted as an alias for
    `age_centred` (the figures use the uncentred variable).

    `n` overrides the replay size. Use the default for headline numbers; a lesson
    that needs several auxiliary truths may pass a smaller n to stay fast. At
    n = 400 000 the relative Monte-Carlo error is still ~1 %, which is far below
    the width of any N=500 confidence interval it will be compared against.
    """
    import statsmodels.formula.api as smf

    df = _replay_dgp(n or N_TRUTH_OR, SEED_TRUTH)
    df["alter"] = df["age_centred"] + AGE_MEAN

    fit = smf.logit(f"verstorben_30d ~ {' + '.join(terms)}", data=df).fit(disp=0)
    return {t: float(np.exp(fit.params[t])) for t in terms}


def true_odds_ratios() -> dict[str, float]:
    """True ORs of the SOFA-adjusted logistic model used in Module 12.

    These are what Module 12's logistic regression actually estimates. They are
    NOT exp(TRUE_LOG_HAZARD): a hazard ratio and an odds ratio differ, and at a
    15.6 % event rate the OR sits further from 1.
    """
    return true_odds_ratios_for(tuple(TRUE_LOG_HAZARD))


@lru_cache(maxsize=1)
def true_hazard_ratios() -> dict[str, dict[str, float]]:
    """True hazard ratios, separately for the DIRECT and the TOTAL effect.

    direct = adjusted for `sofa_score` -> recovers exp(TRUE_LOG_HAZARD).
    total  = `sofa_score` omitted      -> includes the mediated path,
             so diabetes and sepsis are larger. Alter and CRP are unmediated
             and barely move.
    """
    from lifelines import CoxPHFitter

    df = _replay_dgp(N_TRUTH_HR, SEED_TRUTH)
    covars = list(TRUE_LOG_HAZARD)

    def fit(cols: list[str]) -> dict[str, float]:
        cph = CoxPHFitter().fit(df[["fu_zeit_tage", "status"] + cols],
                                duration_col="fu_zeit_tage", event_col="status")
        return {c: float(np.exp(cph.params_[c])) for c in cols}

    direct = fit(covars)
    total = fit([c for c in covars if c != "sofa_score"])
    return {"direct": direct, "total": total}


def write_snapshot() -> Path:
    """Persist the truths so non-Python lessons can be checked against them.

    Module 12's `r.R` cannot `import lib.ground_truth`, so it hardcodes the true
    ORs and HRs. `tools/check_repo.py` diffs those literals against this file.
    """
    hr = true_hazard_ratios()

    # Auf 6 Nachkommastellen runden. Die Modellfits stimmen über Plattformen
    # hinweg nur bis etwa zur 14. Stelle überein (macOS/Accelerate vs.
    # Linux/OpenBLAS in der CI). Der Byte-Abgleich in numbers.yml würde an diesem
    # Gleitkomma-Rauschen scheitern, obwohl jede genutzte Stelle (Prosa und R
    # nutzen ≤3, check_repo prüft mit Toleranz 0,0005) identisch ist. 6 Stellen
    # liegen weit über jedem Bedarf und weit über dem Rauschen.
    def r6(d: dict[str, float]) -> dict:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in d.items()}

    payload = {
        "seed": SEED_TRUTH,
        "n_or": N_TRUTH_OR,
        "n_hr": N_TRUTH_HR,
        "true_or": r6(true_odds_ratios()),
        "true_hr_direct": r6(hr["direct"]),
        "true_hr_total": r6(hr["total"]),
    }
    TRUTH_SNAPSHOT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                              encoding="utf-8")
    return TRUTH_SNAPSHOT


if __name__ == "__main__":
    print(f"Replaying the DGP at N = {N_TRUTH_OR:,} (odds ratios) and N = {N_TRUTH_HR:,} (hazard ratios)\n")
    hr = true_hazard_ratios()
    orr = true_odds_ratios()
    print(f"{'Prädiktor':<16}{'beta (log-hazard)':>19}{'true HR direct':>16}"
          f"{'true HR total':>15}{'true OR (logit)':>17}")
    print("-" * 83)
    for k, b in TRUE_LOG_HAZARD.items():
        tot = hr["total"].get(k)
        tot_s = f"{tot:.3f}" if tot is not None else "  (mediator)"
        print(f"{k:<16}{b:>19.4f}{hr['direct'][k]:>16.3f}{tot_s:>15}{orr[k]:>17.3f}")
    print("\nexp(beta) is a HAZARD ratio. Module 12 estimates ODDS ratios.")
    print(f"\nSnapshot written: {write_snapshot().name}")
