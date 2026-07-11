"""Module 14 - MICE in practice: passive imputation, bounds, auxiliary
variables and convergence diagnostics.

Run: python module/14-fehlende-werte/code/mice_praxis.py

Four lessons that complement code/python.py's missingness-mechanism story:

  1. Passive imputation      -- gewicht_kg and bmi are the same information
                                 twice (bmi is a deterministic function of
                                 weight and height). Imputing both as
                                 independent MICE targets invents patients
                                 whose derived BMI contradicts their own
                                 weight and height.
  2. Bounds                  -- an unbounded normal imputer for bga_ph can
                                 propose values outside the physiologically
                                 possible range; predictive mean matching
                                 (PMM) cannot, because it only ever returns
                                 values that were actually observed.
  3. Auxiliary variables      -- the imputation model may (and often must)
                                 be richer than the analysis model
                                 (Meng 1994, "uncongeniality").
  4. Convergence diagnostics  -- MICE is a Gibbs sampler; it must be
                                 inspected, not trusted.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import statsmodels.api as sm  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from patsy import dmatrix  # noqa: E402
from scipy import stats  # noqa: E402
from statsmodels.imputation import mice  # noqa: E402

from lib.helpers import SEED, load_cohort, load_labs  # noqa: E402
from lib.plotstyle import EVENT, PALETTE, PRIMARY, SECONDARY, apply_style, save  # noqa: E402

warnings.filterwarnings("ignore")

ASSETS = Path(__file__).resolve().parents[1] / "assets"

MODEL = "verstorben_30d ~ bga_ph + alter + sofa_score"
PH_LOW, PH_HIGH = 6.90, 7.60
K_PMM = 5


def seed_int(stream: int) -> int:
    """Deterministic 31-bit seed for a named stream, derived from SEED.

    statsmodels' MICEData consumes the *global* numpy RNG internally (it
    calls np.random.* directly, see statsmodels.imputation.mice), so it
    cannot be handed a Generator. We derive a per-stream integer seed from
    SEED with default_rng and feed that into np.random.seed() right before
    each MICE run, keeping every run reproducible without ever writing
    another bare literal seed.
    """
    return int(np.random.default_rng([SEED, stream]).integers(0, 2**31 - 1))


def load() -> pd.DataFrame:
    df = load_cohort().merge(load_labs(), on="patient_id", how="left")
    truth = pd.read_csv(ROOT / "data" / "bga_ph_wahrheit.csv")
    return df.merge(truth, on="patient_id", how="left")


# ===========================================================================
# 1) Passive imputation -- the derived-variable trap
# ===========================================================================
def section_1_passive(df: pd.DataFrame) -> None:
    print("=" * 74)
    print("1) Passive imputation -- the derived-variable trap")
    print("=" * 74)

    miss_w = df["gewicht_kg"].isna()
    miss_b = df["bmi"].isna()
    same_rows = bool((miss_w == miss_b).all())
    height_complete = bool(df["groesse_cm"].notna().all())
    print(f"  gewicht_kg missing on {int(miss_w.sum())} patients, "
          f"bmi missing on {int(miss_b.sum())} patients")
    print(f"  missing on exactly the same rows : {same_rows}")
    print(f"  groesse_cm is complete            : {height_complete}")
    assert same_rows, "gewicht_kg and bmi are not missing on the same rows -- lesson invalid"
    assert height_complete, "groesse_cm has missing values -- lesson invalid"

    cols = ["groesse_cm", "gewicht_kg", "bmi", "alter", "sofa_score", "crp_mg_l"]
    work = df[cols].copy()
    idx_miss = work.index[miss_w]

    def true_bmi(gewicht: pd.Series) -> pd.Series:
        return gewicht / (work.loc[gewicht.index, "groesse_cm"] / 100) ** 2

    # --- naive: gewicht_kg and bmi imputed as two INDEPENDENT MICE targets.
    # The default imputer formula for each variable is "all other columns",
    # so gewicht_kg's model includes bmi as a predictor and vice versa --
    # exactly the circular setup that produces the inconsistency.
    np.random.seed(seed_int(0))
    naive = mice.MICEData(work.copy(), k_pmm=K_PMM)
    naive.update_all(10)  # burn-in
    naive_snaps = []
    for _ in range(20):
        naive.update_all(1)
        snap = naive.data.loc[idx_miss, ["gewicht_kg", "bmi"]].copy()
        snap["bmi_von_gewicht"] = true_bmi(snap["gewicht_kg"])
        naive_snaps.append(snap)
    naive_pairs = pd.concat(naive_snaps, ignore_index=True)
    naive_incons = (naive_pairs["bmi"] - naive_pairs["bmi_von_gewicht"]).abs()

    # --- passive: impute gewicht_kg only, then DERIVE bmi deterministically.
    # statsmodels has no passive-imputation mechanism (no formula-driven
    # "~ I(...)" method the way R's mice offers) -- this is a genuine
    # feature gap, not an oversight. The workaround: drop bmi from the MICE
    # run entirely and recompute it by hand from the imputed weight and the
    # (always observed) height after every draw.
    np.random.seed(seed_int(1))
    passive_cols = ["groesse_cm", "gewicht_kg", "alter", "sofa_score", "crp_mg_l"]
    passive = mice.MICEData(work[passive_cols].copy(), k_pmm=K_PMM)
    passive.update_all(10)
    passive_snaps = []
    for _ in range(20):
        passive.update_all(1)
        gewicht = passive.data.loc[idx_miss, "gewicht_kg"]
        bmi_derived = true_bmi(gewicht)
        passive_snaps.append(pd.DataFrame({
            "gewicht_kg": gewicht.to_numpy(),
            "bmi": bmi_derived.to_numpy(),
            "bmi_von_gewicht": bmi_derived.to_numpy(),
        }))
    passive_pairs = pd.concat(passive_snaps, ignore_index=True)
    passive_incons = (passive_pairs["bmi"] - passive_pairs["bmi_von_gewicht"]).abs()

    print(f"\n  naive   |bmi_imp - gewicht_imp/(h/100)^2| : mean={naive_incons.mean():.4f}"
          f"  max={naive_incons.max():.4f}")
    print(f"  passive |bmi_imp - gewicht_imp/(h/100)^2| : mean={passive_incons.mean():.10f}"
          f"  max={passive_incons.max():.10f}")
    print("\n  Naive MICE invents patients whose BMI contradicts their own weight")
    print("  and height. Passive imputation cannot, by construction.")

    _figure_passive(naive_pairs, passive_pairs)


def _figure_passive(naive_pairs: pd.DataFrame, passive_pairs: pd.DataFrame) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.2), sharex=True, sharey=True)

    all_vals = pd.concat([
        naive_pairs[["bmi", "bmi_von_gewicht"]],
        passive_pairs[["bmi", "bmi_von_gewicht"]],
    ])
    lo, hi = float(all_vals.min().min()), float(all_vals.max().max())
    pad = 0.04 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    panels = [
        ("Naiv: bmi und gewicht_kg\nunabhängig imputiert", naive_pairs, PRIMARY),
        ("Passiv: bmi aus imputiertem\ngewicht_kg abgeleitet", passive_pairs, "#5B9E6E"),
    ]
    for ax, (title, data, color) in zip(axes, panels):
        ax.plot([lo, hi], [lo, hi], color=SECONDARY, linestyle="--", linewidth=1.3,
                label="Identität", zorder=1)
        ax.scatter(data["bmi_von_gewicht"], data["bmi"], s=20, alpha=0.5,
                   color=color, edgecolor="none", zorder=2)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("berechneter BMI aus imputiertem Gewicht (kg/m²)")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")

    axes[0].set_ylabel("imputierter BMI (kg/m²)")
    axes[0].legend(loc="upper left", fontsize=9.5)
    fig.suptitle("Passive Imputation vermeidet die BMI-Inkonsistenz",
                 fontsize=14, fontweight="bold", x=0.01, y=1.0, ha="left")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    save(fig, ASSETS / "passive_imputation.png")


# ===========================================================================
# 2) Bounds -- imputed values must be physiologically possible
# ===========================================================================
def section_2_bounds(df: pd.DataFrame) -> None:
    print("\n" + "=" * 74)
    print("2) Bounds -- imputed values must be physiologically possible")
    print("=" * 74)
    print(f"  bga_ph is bounded to [{PH_LOW:.2f}, {PH_HIGH:.2f}] by construction "
          "(data/generate_data.py clips it).")

    formula = "alter + sofa_score + verstorben_30d"
    work = df[["verstorben_30d", "bga_ph", "alter", "sofa_score"]].copy()
    obs = work.dropna(subset=["bga_ph"])
    miss = work[work["bga_ph"].isna()]
    idx_miss = miss.index
    n_miss = len(miss)

    # --- unbounded normal / linear-regression imputer -----------------------
    # This is Rubin's classic Bayesian linear-regression ("norm") algorithm:
    # draw sigma^2 from its posterior, draw beta from its conditional
    # posterior given sigma, then predict + add Gaussian noise. statsmodels'
    # own MICEData always matches to an observed donor internally (its
    # impute() method calls impute_pmm() unconditionally -- there is no
    # "norm" option at all), so this Bayesian linear imputer is implemented
    # by hand.
    X_obs = dmatrix(formula, data=obs, return_type="dataframe")
    X_miss = dmatrix(formula, data=miss, return_type="dataframe")
    y_obs = obs["bga_ph"].to_numpy()
    ols = sm.OLS(y_obs, X_obs).fit()
    beta_hat = ols.params.to_numpy()
    xtx_inv = np.linalg.inv(X_obs.T.to_numpy() @ X_obs.to_numpy())
    rss, dfree = float(ols.ssr), ols.df_resid
    resid_sd = np.sqrt(rss / dfree)

    rng = np.random.default_rng([SEED, 2])
    norm_draws = []
    for _ in range(20):
        g = rng.chisquare(dfree)
        sigma_star = np.sqrt(rss / g)
        beta_star = rng.multivariate_normal(beta_hat, (sigma_star ** 2) * xtx_inv)
        mean_miss = X_miss.to_numpy() @ beta_star
        norm_draws.append(mean_miss + rng.normal(0.0, sigma_star, size=n_miss))
    norm_draws = np.concatenate(norm_draws)

    norm_oob = (norm_draws < PH_LOW) | (norm_draws > PH_HIGH)
    print(f"\n  norm imputer (m=20, unbounded): {int(norm_oob.sum())} / {len(norm_draws)} "
          f"draws outside [{PH_LOW:.2f}, {PH_HIGH:.2f}]")
    print(f"    draw range: {norm_draws.min():.3f} to {norm_draws.max():.3f}")
    if norm_oob.sum() > 0:
        print(f"    out-of-bounds values range {norm_draws[norm_oob].min():.3f} "
              f"to {norm_draws[norm_oob].max():.3f}")
    else:
        lo_gap = norm_draws.min() - PH_LOW
        hi_gap = PH_HIGH - norm_draws.max()
        print(f"    none of these draws actually crossed the boundary; the closest one sat "
              f"{min(lo_gap, hi_gap):.3f} pH units from the nearer bound.")
        print("    This cohort's pH distribution is narrow relative to the physiological")
        print("    range, so the norm imputer rarely breaches it in practice -- but it")
        print("    still assigns nonzero probability mass beyond [6.90, 7.60]:")
        mean_miss_hat = X_miss.to_numpy() @ beta_hat
        p_below = stats.norm.cdf(PH_LOW, loc=mean_miss_hat, scale=resid_sd)
        p_above = 1 - stats.norm.cdf(PH_HIGH, loc=mean_miss_hat, scale=resid_sd)
        expected_violations = 20 * float((p_below + p_above).sum())
        print(f"    expected out-of-bounds draws under this model over m=20: "
              f"{expected_violations:.4f} (not exactly 0)")

    # --- PMM, donor pool k = 5 -----------------------------------------------
    np.random.seed(seed_int(3))
    pmm_imp = mice.MICEData(work.copy(), k_pmm=K_PMM)
    pmm_imp.set_imputer("bga_ph", formula=formula)
    pmm_imp.update_all(10)
    pmm_snaps = []
    for _ in range(20):
        pmm_imp.update_all(1)
        pmm_snaps.append(pmm_imp.data.loc[idx_miss, "bga_ph"].to_numpy())
    pmm_draws = np.concatenate(pmm_snaps)
    pmm_oob = (pmm_draws < PH_LOW) | (pmm_draws > PH_HIGH)
    print(f"\n  PMM imputer  (m=20, k={K_PMM}): {int(pmm_oob.sum())} / {len(pmm_draws)} "
          f"draws outside [{PH_LOW:.2f}, {PH_HIGH:.2f}]")
    print(f"    draw range: {pmm_draws.min():.3f} to {pmm_draws.max():.3f} "
          "-- always inside the observed support")

    print("\n  Rule: PMM respects the support and the marginal distribution of the")
    print("  observed data; an unbounded normal imputer does not. The cost: PMM")
    print("  cannot extrapolate beyond the observed range, even when that would be")
    print("  the physiologically correct thing to do.")


# ===========================================================================
# 3) Auxiliary variables and congeniality (Meng 1994)
# ===========================================================================
def section_3_auxiliary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 74)
    print("3) Auxiliary variables and congeniality (Meng 1994)")
    print("=" * 74)

    truth = smf.logit(MODEL, data=df.assign(bga_ph=df["bga_ph_wahr"])).fit(disp=0)
    beta_true = float(truth.params["bga_ph"])
    print(f"  true beta_bga_ph on the full (uncensored) data: {beta_true:+.3f}")

    variants = [
        ("a) alter + sofa_score",
         ["verstorben_30d", "bga_ph", "alter", "sofa_score"],
         "alter + sofa_score"),
        ("b) + verstorben_30d",
         ["verstorben_30d", "bga_ph", "alter", "sofa_score"],
         "alter + sofa_score + verstorben_30d"),
        # laktat_mmol_l is itself ~17% missing, so folding it in as an
        # auxiliary predictor makes this a genuinely MULTIVARIATE MICE
        # problem: laktat_mmol_l must be imputed too, inside the same
        # chained-equations run, before it can serve as a predictor for
        # bga_ph.
        ("c) + laktat, crp, leukozyten",
         ["verstorben_30d", "bga_ph", "alter", "sofa_score",
          "laktat_mmol_l", "crp_mg_l", "leukozyten_g_l"],
         "alter + sofa_score + verstorben_30d + laktat_mmol_l + crp_mg_l + leukozyten_g_l"),
    ]

    print(f"\n  {'imputation model':30s}{'beta_bga_ph':>13}{'SE':>9}{'lambda':>9}")
    results = {}
    for i, (label, cols, formula) in enumerate(variants):
        np.random.seed(seed_int(4 + i))
        work = df[cols].copy()
        imputer = mice.MICEData(work, k_pmm=K_PMM)
        imputer.set_imputer("bga_ph", formula=formula)
        fit = mice.MICE(MODEL, sm.Logit, imputer, fit_kwds={"disp": 0}).fit(
            n_imputations=20, n_burnin=10)
        idx = truth.params.index
        beta_s = pd.Series(np.asarray(fit.params), index=idx)
        se_s = pd.Series(np.asarray(fit.bse), index=idx)
        lam_s = pd.Series(np.asarray(fit.frac_miss_info), index=idx)
        results[label] = (float(beta_s["bga_ph"]), float(se_s["bga_ph"]), float(lam_s["bga_ph"]))
        beta, se, lam = results[label]
        print(f"  {label:30s}{beta:>13.3f}{se:>9.3f}{lam:>9.3f}")

    print(f"\n  {'true (full data)':30s}{beta_true:>13.3f}")

    label_a, label_b, label_c = (v[0] for v in variants)
    beta_a, _, lam_a = results[label_a]
    beta_b, _, lam_b = results[label_b]
    beta_c, _, lam_c = results[label_c]

    print(f"\n  |beta_a - true| = {abs(beta_a - beta_true):.3f}   "
          f"(a) is biased toward zero: {abs(beta_a) < abs(beta_true)}")
    print(f"  |beta_b - true| = {abs(beta_b - beta_true):.3f}   "
          f"(b) recovers the true effect much more closely")
    print(f"  |beta_c - true| = {abs(beta_c - beta_true):.3f}")

    if lam_c <= lam_b:
        print(f"\n  (c) has an equal-or-lower fraction of missing information than (b): "
              f"{lam_c:.3f} <= {lam_b:.3f}")
    else:
        print(f"\n  (c) does NOT lower the fraction of missing information: "
              f"{lam_c:.3f} > {lam_b:.3f}")
        print("  laktat_mmol_l is itself ~17 % missing, so folding it in adds its own")
        print("  imputation uncertainty without buying much extra predictive power for")
        print("  pH -- a legitimate negative result: richer is not automatically better.")


# ===========================================================================
# 4) Convergence diagnostics -- MICE is a Gibbs sampler
# ===========================================================================
def _rhat(chain_matrix: np.ndarray) -> float:
    """Gelman-Rubin potential scale reduction factor (R-hat).

    chain_matrix has shape (n_iter, n_chains): one traced scalar per chain
    per iteration (here: the chain's mean imputed bga_ph after that
    iteration). R-hat close to 1 indicates the chains have mixed.
    """
    n, m = chain_matrix.shape
    chain_means = chain_matrix.mean(axis=0)
    grand_mean = chain_means.mean()
    b = n / (m - 1) * np.sum((chain_means - grand_mean) ** 2)
    w = chain_matrix.var(axis=0, ddof=1).mean()
    var_hat = (n - 1) / n * w + b / n
    return float(np.sqrt(var_hat / w))


def section_4_convergence(df: pd.DataFrame) -> None:
    print("\n" + "=" * 74)
    print("4) Convergence diagnostics -- MICE is a Gibbs sampler")
    print("=" * 74)

    formula = "alter + sofa_score + verstorben_30d"
    work = df[["verstorben_30d", "bga_ph", "alter", "sofa_score"]].copy()
    idx_miss = work.index[work["bga_ph"].isna()]
    n_chains, n_iter = 5, 20

    chain_means = np.zeros((n_iter, n_chains))
    chain_sds = np.zeros((n_iter, n_chains))
    final_snaps = []  # pooled final-iteration imputed values, all chains
    for c in range(n_chains):
        np.random.seed(seed_int(10 + c))
        imp = mice.MICEData(work.copy(), k_pmm=K_PMM)
        imp.set_imputer("bga_ph", formula=formula)
        for it in range(n_iter):
            imp.update_all(1)
            vals = imp.data.loc[idx_miss, "bga_ph"].to_numpy()
            chain_means[it, c] = vals.mean()
            chain_sds[it, c] = vals.std()
        final_snaps.append(imp.data.loc[idx_miss, "bga_ph"].to_numpy())
    final_draws = np.concatenate(final_snaps)

    rhat_mean = _rhat(chain_means)
    print(f"  maxit={n_iter}, m={n_chains} chains, tracking mean and SD of imputed bga_ph")
    print(f"  Gelman-Rubin R-hat on the chain-mean trace: {rhat_mean:.3f}")
    healthy = rhat_mean < 1.1
    if healthy:
        print(f"  R-hat < 1.1 -- chains intermingle with no drift/trend: convergence looks healthy.")
    else:
        print(f"  R-hat >= 1.1 -- chains have not mixed; more iterations or burn-in are needed.")

    _figure_convergence(chain_means, chain_sds)

    observed = df["bga_ph"].dropna().to_numpy()
    _figure_density(observed, final_draws)
    print(f"\n  observed bga_ph  mean={observed.mean():.3f}  n={len(observed)}")
    print(f"  imputed  bga_ph  mean={final_draws.mean():.3f}  n={len(final_draws)} "
          f"(pooled over {n_chains} chains)")
    print("  The imputed density sits lower than the observed one: missing patients")
    print("  died more often, and dying patients are more acidotic. A perfectly")
    print("  overlapping density would be evidence the imputation model ignored the")
    print("  outcome (verstorben_30d) that drives the missingness.")


def _figure_convergence(chain_means: np.ndarray, chain_sds: np.ndarray) -> None:
    apply_style()
    n_iter, n_chains = chain_means.shape
    it = np.arange(1, n_iter + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 7.5), sharex=True)
    for c in range(n_chains):
        color = PALETTE[c % len(PALETTE)]
        ax1.plot(it, chain_means[:, c], color=color, marker="o", markersize=3,
                  label=f"Kette {c + 1}")
        ax2.plot(it, chain_sds[:, c], color=color, marker="o", markersize=3,
                  label=f"Kette {c + 1}")
    ax1.set_ylabel("Mittelwert (imputiert)")
    ax1.set_title("Konvergenz der MICE-Ketten für bga_ph", fontsize=13.5)
    ax2.set_ylabel("Standardabweichung (imputiert)")
    ax2.set_xlabel("Iteration")
    ax1.legend(ncol=n_chains, fontsize=8.5, loc="upper right")
    fig.tight_layout()
    save(fig, ASSETS / "mice_konvergenz.png")


def _figure_density(observed: np.ndarray, imputed: np.ndarray) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    sns.kdeplot(observed, ax=ax, color=PRIMARY, linewidth=2.3,
                label=f"beobachtet (n={len(observed)})")
    sns.kdeplot(imputed, ax=ax, color=EVENT, linewidth=2.3,
                label=f"imputiert (n={len(imputed)}, gepoolt über 5 Ketten)")
    ax.set_xlabel("arterieller pH (bga_ph)")
    ax.set_ylabel("Dichte")
    ax.set_title("Beobachteter vs. imputierter arterieller pH", fontsize=13.5)
    ax.legend()
    save(fig, ASSETS / "mice_dichte.png")


def main() -> None:
    df = load()
    section_1_passive(df)
    section_2_bounds(df)
    section_3_auxiliary(df)
    section_4_convergence(df)


if __name__ == "__main__":
    main()
