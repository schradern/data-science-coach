"""Module 10 – Inferential statistics and hypothesis testing (Python / scipy.stats).

Runs standalone from the project root:
    python module/10-inferenzstatistik/code/python.py

Data: read from data/ (committed with the repo); if that folder is
missing, the same files are fetched from the published URL.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from lib.helpers import SEED, load_cohort, load_labs  # noqa: E402

pd.set_option("display.width", 110)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Helper: confidence interval for a mean (t-distribution)
# ---------------------------------------------------------------------------
def confidence_interval_mean(
    data: pd.Series, level: float = 0.95
) -> tuple[float, float]:
    """Return (lower, upper) confidence interval for the mean.

    Uses the t-distribution; missing values are dropped silently.

    Args:
        data:  Numeric series.
        level: Confidence level, default 0.95.

    Returns:
        Tuple (lower, upper).
    """
    x = data.dropna().to_numpy()
    se = stats.sem(x)
    t_crit = stats.t.ppf((1 + level) / 2, df=len(x) - 1)
    mean = x.mean()
    return float(mean - t_crit * se), float(mean + t_crit * se)


# ---------------------------------------------------------------------------
# Helper: Cohen's d (pooled standard deviation)
# ---------------------------------------------------------------------------
def cohens_d(group_a: pd.Series, group_b: pd.Series) -> float:
    """Compute pooled Cohen's d (positive when group_a > group_b).

    Args:
        group_a: First group values.
        group_b: Second group values.

    Returns:
        Cohen's d (float).
    """
    a = group_a.dropna().to_numpy()
    b = group_b.dropna().to_numpy()
    pooled_sd = np.sqrt(
        ((len(a) - 1) * a.std(ddof=1) ** 2 + (len(b) - 1) * b.std(ddof=1) ** 2)
        / (len(a) + len(b) - 2)
    )
    return float((a.mean() - b.mean()) / pooled_sd)


# ---------------------------------------------------------------------------
# Helper: rank-biserial r (effect size for Mann-Whitney-U)
# ---------------------------------------------------------------------------
def rank_biserial_r(u_stat: float, n1: int, n2: int) -> float:
    """Convert Mann-Whitney U statistic to rank-biserial correlation r.

    r = 2*U / (n1*n2) - 1, where U is the U statistic for group 1 (n1) as
    returned by scipy.stats.mannwhitneyu(group1, group2, ...). Convention:
    r > 0 means group 1 tends to have stochastically higher values than
    group 2; |r| ~ 0.1 small, 0.3 medium, 0.5 large.

    Args:
        u_stat: U statistic from mannwhitneyu (for the first sample passed).
        n1:     Size of first group.
        n2:     Size of second group.

    Returns:
        Rank-biserial r in [-1, 1].
    """
    return float(2 * u_stat / (n1 * n2) - 1)


# ---------------------------------------------------------------------------
# Helper: odds ratio with 95-% CI (Woolf / log-transform method)
# ---------------------------------------------------------------------------
def odds_ratio_ci(
    contingency: pd.DataFrame, level: float = 0.95
) -> tuple[float, float, float]:
    """Compute odds ratio and CI from a 2×2 contingency table.

    Rows: exposure (0 = no, 1 = yes); columns: outcome (0 = no, 1 = yes).

    Args:
        contingency: 2×2 DataFrame produced by pd.crosstab.
        level:       Confidence level, default 0.95.

    Returns:
        Tuple (or_value, lower, upper).
    """
    a = contingency.iloc[1, 1]   # exposed, outcome
    b = contingency.iloc[1, 0]   # exposed, no outcome
    c = contingency.iloc[0, 1]   # unexposed, outcome
    d = contingency.iloc[0, 0]   # unexposed, no outcome
    or_val = (a * d) / (b * c)
    z = stats.norm.ppf((1 + level) / 2)
    log_se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    lower = np.exp(np.log(or_val) - z * log_se)
    upper = np.exp(np.log(or_val) + z * log_se)
    return float(or_val), float(lower), float(upper)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def main() -> None:
    cohort = load_cohort()
    labs = load_labs()
    df = cohort.merge(labs, on="patient_id", how="left")

    # -----------------------------------------------------------------------
    # 0) Shapiro-Wilk: DESCRIPTIVE normality diagnostic for laktat_mmol_l
    #    NOT a test-selection rule – see module 21 ("Normalitätstest-Autopilot").
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("0) Shapiro-Wilk: descriptive normality diagnostic for laktat_mmol_l")
    print("=" * 60)

    laktat_all = df["laktat_mmol_l"].dropna()
    w_stat, p_sw = stats.shapiro(laktat_all)
    print(f"  n={len(laktat_all)}, W={w_stat:.4f}, p={p_sw:.4e}")
    print(f"  Skewness={laktat_all.skew():.2f}")
    print(f"  Normal distribution rejected (p<0.05): {p_sw < 0.05}")
    print("  NOTE: Shapiro-Wilk is a DESCRIPTIVE diagnostic, not a test-selection")
    print("  rule (it over-rejects at large N, misses departures at small N).")
    print("  Choose the test from the estimand, the shape seen in a plot, and")
    print("  robustness (Welch-t is robust at N>30) – see module 21. laktat is")
    print("  strongly right-skewed, so we report BOTH Welch-t and Mann-Whitney-U.")

    # -----------------------------------------------------------------------
    # 1) Welch-t-test: laktat – Sepsis vs. non-Sepsis
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("1) Welch-t-test: laktat_mmol_l – Sepsis vs. nicht-Sepsis")
    print("=" * 60)

    sepsis = df.loc[df["aufnahmegrund"] == "Sepsis", "laktat_mmol_l"].dropna()
    no_sepsis = df.loc[df["aufnahmegrund"] != "Sepsis", "laktat_mmol_l"].dropna()

    print(f"  Sepsis      n={len(sepsis):>3}  mean={sepsis.mean():.2f}  SD={sepsis.std():.2f}")
    print(f"  kein Sepsis n={len(no_sepsis):>3}  mean={no_sepsis.mean():.2f}  SD={no_sepsis.std():.2f}")

    t_stat, p_t = stats.ttest_ind(sepsis, no_sepsis, equal_var=False)
    d = cohens_d(sepsis, no_sepsis)
    ci_sepsis = confidence_interval_mean(sepsis)
    ci_no_sep = confidence_interval_mean(no_sepsis)

    print(f"\n  Welch-t={t_stat:.3f},  p={p_t:.4f}")
    print(f"  95-%-CI Sepsis:      [{ci_sepsis[0]:.2f}, {ci_sepsis[1]:.2f}]")
    print(f"  95-%-CI kein Sepsis: [{ci_no_sep[0]:.2f}, {ci_no_sep[1]:.2f}]")
    print(f"  Effect size Cohen's d = {d:.3f}  (0.2 small, 0.5 medium, 0.8 large)")
    print()
    print("  Correct interpretation of p:")
    print("  p = P(data as extreme or more | H0 true) – NOT P(H0 true | data).")
    print("  Always report effect size + CI alongside p.")

    # -----------------------------------------------------------------------
    # 2) Mann-Whitney-U (nonparametric – preferred for skewed laktat)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("2) Mann-Whitney-U: laktat_mmol_l – Sepsis vs. nicht-Sepsis")
    print("   (nonparametric, robust against skew and outliers)")
    print("=" * 60)

    # use_continuity defaults to True (matches R's wilcox.test correct=TRUE),
    # so this p-value equals the R script's. Keep the same setting in both
    # languages; see the README "Stolperstein" on continuity correction.
    u_stat, p_mw = stats.mannwhitneyu(sepsis, no_sepsis, alternative="two-sided")
    r_rb = rank_biserial_r(u_stat, len(sepsis), len(no_sepsis))

    print(f"  U={u_stat:.0f},  p={p_mw:.4f}")
    print(f"  Rank-biserial r={r_rb:.3f}  (|r|~0.1 small, 0.3 medium, 0.5 large)")
    print("  Interpretation: r > 0 means higher values in Sepsis group stochastically.")

    # -----------------------------------------------------------------------
    # 3) Chi-square test: Diabetes × 30-day mortality
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("3) Chi-square: Diabetes × 30-day mortality (verstorben_30d)")
    print("=" * 60)

    ct = pd.crosstab(
        df["diabetes"],
        df["verstorben_30d"],
        rownames=["Diabetes"],
        colnames=["Verstorben_30d"],
    )
    print("\n  Contingency table:")
    print(ct.to_string())

    chi2, p_chi2, dof, _ = stats.chi2_contingency(ct, correction=False)
    n_total = ct.values.sum()
    cramers_v = float(np.sqrt(chi2 / (n_total * (min(ct.shape) - 1))))

    print(f"\n  χ²={chi2:.3f},  df={dof},  p={p_chi2:.4f}")
    print(f"  Cramér's V={cramers_v:.3f}  (0.1 small, 0.3 medium, 0.5 large)")

    or_val, or_lo, or_hi = odds_ratio_ci(ct)
    print(f"\n  Odds Ratio (Diabetes=1 vs 0) = {or_val:.2f}")
    print(f"  95-%-CI OR: [{or_lo:.2f}, {or_hi:.2f}]")
    print("  Note: OR ≠ RR (relative risk). At low event rates they are similar;")
    print("  at high event rates (>10 %) OR overestimates RR.")

    # -----------------------------------------------------------------------
    # 4) Multiple testing – Bonferroni correction for 5 lab markers
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("4) Multiple testing: Bonferroni correction (5 lab markers)")
    print("=" * 60)

    lab_markers = [
        "leukozyten_g_l",
        "haemoglobin_g_dl",
        "kreatinin_mg_dl",
        "laktat_mmol_l",
        "natrium_mmol_l",
    ]
    n_tests = len(lab_markers)
    results = []

    for marker in lab_markers:
        grp_dead = df.loc[df["verstorben_30d"] == 1, marker].dropna()
        grp_alive = df.loc[df["verstorben_30d"] == 0, marker].dropna()
        _, p_raw = stats.ttest_ind(grp_dead, grp_alive, equal_var=False)
        d_val = cohens_d(grp_dead, grp_alive)
        results.append({"marker": marker, "p_raw": p_raw, "cohens_d": d_val})

    result_df = pd.DataFrame(results)
    result_df["p_bonferroni"] = np.minimum(result_df["p_raw"] * n_tests, 1.0)
    result_df["sig_raw"] = result_df["p_raw"] < 0.05
    result_df["sig_bonferroni"] = result_df["p_bonferroni"] < 0.05

    print(f"\n  Lab markers vs. 30-day mortality (Welch-t, α=0.05):")
    print(result_df.to_string(index=False, float_format="{:.4f}".format))
    print(
        f"\n  Without correction:    {result_df['sig_raw'].sum()} / {n_tests} significant"
    )
    print(
        f"  After Bonferroni:      {result_df['sig_bonferroni'].sum()} / {n_tests} significant"
    )
    print(f"  Bonferroni-corrected α = {0.05 / n_tests:.4f}")
    print(
        "  -> Multiple testing inflates the false-positive rate."
        " Bonferroni is conservative;"
    )
    print(
        "     Benjamini-Hochberg (FDR) is a common alternative in multi-marker studies."
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
