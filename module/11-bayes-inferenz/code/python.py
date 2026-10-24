"""Module 11 - Bayesian inference (conjugate Beta-Binomial, no MCMC)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402
from statsmodels.stats.proportion import proportion_confint  # noqa: E402

from lib.helpers import SEED, load_cohort  # noqa: E402


def beta_summary(a: float, b: float) -> tuple[float, float, float]:
    """Posterior mean and equal-tailed 95% credible interval of Beta(a, b)."""
    mean = a / (a + b)
    lo, hi = stats.beta.ppf([0.025, 0.975], a, b)
    return mean, lo, hi


def main() -> None:
    df = load_cohort()
    n = len(df)
    k = int(df["verstorben_30d"].sum())

    # -----------------------------------------------------------------
    # 1) Beta-Binomial posterior for the overall 30-day mortality rate.
    #    Uniform prior Beta(1, 1): every rate equally plausible a priori.
    #    Posterior = Beta(1 + k, 1 + n - k).
    # -----------------------------------------------------------------
    print("\n1) Overall 30-day mortality – posterior Beta(1+k, 1+n-k)")
    a_post, b_post = 1 + k, 1 + n - k
    mean, lo, hi = beta_summary(a_post, b_post)
    print(f"data: k={k} deaths / n={n} patients (observed {k / n:.4f})")
    print(f"posterior Beta({a_post}, {b_post})")
    print(f"posterior mean={mean:.4f}, 95% CrI [{lo:.4f}, {hi:.4f}]")

    # Frequentist counterpart from module 13 for the honest contrast.
    w_lo, w_hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    print(f"frequentist Wilson 95% CI [{w_lo:.4f}, {w_hi:.4f}]")

    # -----------------------------------------------------------------
    # 2) Prior sensitivity: with n=500 the prior barely matters; with a
    #    tiny interim sample it dominates.
    #    weak prior  = Beta(1, 1)    (uniform, ~0 pseudo-patients)
    #    strong prior= Beta(20, 80)  (mean 0.20, worth 100 pseudo-patients)
    # -----------------------------------------------------------------
    print("\n2) Prior sensitivity – weak Beta(1,1) vs strong Beta(20,80)")
    priors = {"weak Beta(1,1)": (1, 1), "strong Beta(20,80)": (20, 80)}

    interim = df.head(30)
    ni, ki = len(interim), int(interim["verstorben_30d"].sum())
    print(f"full cohort:      k={k}/n={n}")
    print(f"interim (first {ni}): k={ki}/n={ni}")
    for label, (a0, b0) in priors.items():
        m_full, lo_f, hi_f = beta_summary(a0 + k, b0 + n - k)
        m_int, lo_i, hi_i = beta_summary(a0 + ki, b0 + ni - ki)
        print(f"  {label:>19}: full  mean={m_full:.4f} CrI[{lo_f:.4f},{hi_f:.4f}]  "
              f"interim mean={m_int:.4f} CrI[{lo_i:.4f},{hi_i:.4f}]")

    # -----------------------------------------------------------------
    # 3) Comparing two groups the Bayesian way: Sepsis vs non-Sepsis.
    #    Draw from each Beta posterior (uniform prior) and look at the
    #    posterior of the DIFFERENCE in mortality.
    # -----------------------------------------------------------------
    print("\n3) Two groups – Sepsis vs non-Sepsis, posterior of the difference")
    sep = df["aufnahmegrund"] == "Sepsis"
    k1, n1 = int(df.loc[sep, "verstorben_30d"].sum()), int(sep.sum())
    k0, n0 = int(df.loc[~sep, "verstorben_30d"].sum()), int((~sep).sum())
    m1, l1, h1 = beta_summary(1 + k1, 1 + n1 - k1)
    m0, l0, h0 = beta_summary(1 + k0, 1 + n0 - k0)
    print(f"Sepsis:     k={k1}/n={n1}, posterior mean={m1:.4f} CrI[{l1:.4f},{h1:.4f}]")
    print(f"non-Sepsis: k={k0}/n={n0}, posterior mean={m0:.4f} CrI[{l0:.4f},{h0:.4f}]")

    rng = np.random.default_rng(SEED)
    draws = 200_000
    p1 = rng.beta(1 + k1, 1 + n1 - k1, draws)
    p0 = rng.beta(1 + k0, 1 + n0 - k0, draws)
    diff = p1 - p0
    d_lo, d_hi = np.quantile(diff, [0.025, 0.975])
    p_gt0 = float(np.mean(diff > 0))
    print(f"difference: mean={diff.mean():.4f}, 95% CrI [{d_lo:.4f}, {d_hi:.4f}]")
    print(f"P(Sepsis mortality > non-Sepsis mortality) = {p_gt0:.4f}")

    # -----------------------------------------------------------------
    # 4) ROPE – region of practical equivalence.
    #    A difference of at most +/- 0.05 (5 percentage points) is treated
    #    as clinically negligible. Report the posterior mass inside it.
    # -----------------------------------------------------------------
    print("\n4) ROPE [-0.05, +0.05] on the mortality difference")
    rope_lo, rope_hi = -0.05, 0.05
    in_sep = float(np.mean((diff > rope_lo) & (diff < rope_hi)))
    print(f"Sepsis vs non-Sepsis: posterior mass inside ROPE = {in_sep:.4f}")

    # Contrast: hypertension vs no hypertension (a near-null effect).
    htn = df["hypertonie"] == 1
    kh, nh = int(df.loc[htn, "verstorben_30d"].sum()), int(htn.sum())
    kn, nn = int(df.loc[~htn, "verstorben_30d"].sum()), int((~htn).sum())
    ph = rng.beta(1 + kh, 1 + nh - kh, draws)
    pn = rng.beta(1 + kn, 1 + nn - kn, draws)
    dh = ph - pn
    in_htn = float(np.mean((dh > rope_lo) & (dh < rope_hi)))
    dh_lo, dh_hi = np.quantile(dh, [0.025, 0.975])
    print(f"Hypertonie: k={kh}/n={nh} vs k={kn}/n={nn}")
    print(f"Hypertonie vs none: diff mean={dh.mean():.4f}, 95% CrI [{dh_lo:.4f}, {dh_hi:.4f}]")
    print(f"Hypertonie vs none: posterior mass inside ROPE = {in_htn:.4f}")


if __name__ == "__main__":
    main()
