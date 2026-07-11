"""Module 14 - MNAR sensitivity analysis: delta-adjustment / tipping point.

Run: python module/14-fehlende-werte/code/mnar_delta.py

Multiple imputation with the outcome in the imputation model (module/12's main
analysis) is only valid under MAR. In real life MAR can never be verified from
the data alone -- it is an assumption about a process we cannot observe. The
honest response is a sensitivity analysis under assumed MNAR departures.

Delta-adjustment (pattern-mixture form): impute under MAR as usual, then shift
only the IMPUTED values of `bga_ph` by a constant delta (never the observed
values), refit, and pool with Rubin's rules. delta = 0 recovers the ordinary
MAR analysis. delta is the assumed systematic pH difference between patients
whose blood gas was never measured and otherwise-identical patients whose
blood gas was measured.

We sweep delta over a grid and ask: how large would that systematic
difference have to be before the reported association between acidosis and
30-day mortality stops being statistically supported (95% CI of the pooled OR
crosses 1)? That threshold is the "tipping point".
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from statsmodels.imputation import mice  # noqa: E402

from lib.helpers import SEED, load_cohort, load_labs  # noqa: E402
from lib.plotstyle import apply_style, save, EVENT, PRIMARY, SECONDARY  # noqa: E402

warnings.filterwarnings("ignore")

MODEL = "verstorben_30d ~ bga_ph + alter + sofa_score"

# MICE settings -- mirror code/python.py's section 5 (n_burnin=10) and the
# MICE class default (n_skip=3, i.e. 4 Gibbs updates per drawn imputation).
M = 20
N_BURNIN = 10
N_SKIP = 3

DELTA_STEP = 0.01
PRIMARY_DELTAS = np.round(np.arange(-0.10, 0.10 + 1e-9, DELTA_STEP), 2)
WIDE_DELTAS = np.round(np.arange(-0.25, 0.25 + 1e-9, DELTA_STEP), 2)

FIGURE_PATH = ROOT / "module" / "14-fehlende-werte" / "assets" / "mnar_tipping_point.png"


# ---------------------------------------------------------------------------
# Data and seeding
# ---------------------------------------------------------------------------

def load() -> pd.DataFrame:
    df = load_cohort().merge(load_labs(), on="patient_id", how="left")
    return df[["verstorben_30d", "bga_ph", "alter", "sofa_score"]].copy()


def _oracle_odds_ratio() -> float:
    """OR per 0.1 pH drop on the full, un-masked truth. Only possible because
    the cohort is synthetic — this is the target the MAR analysis should hit."""
    import statsmodels.formula.api as smf

    truth = pd.read_csv(ROOT / "data" / "bga_ph_wahrheit.csv")
    df = load_cohort().merge(truth, on="patient_id").rename(columns={"bga_ph_wahr": "bga_ph"})
    fit = smf.logit(MODEL, data=df).fit(disp=0)
    return float(np.exp(-0.1 * fit.params["bga_ph"]))


def conditional_sd_of_ph() -> float:
    """Residual SD of pH given the analysis covariates.

    The natural yardstick for delta: a shift of `delta` pH units is a shift of
    `delta / conditional_sd` residual standard deviations. Judging delta against
    assay noise would be a category error — assay noise is random, delta is a
    *systematic* difference between measured and unmeasured patients.
    """
    import statsmodels.formula.api as smf

    truth = pd.read_csv(ROOT / "data" / "bga_ph_wahrheit.csv")
    df = load_cohort().merge(truth, on="patient_id")
    resid = smf.ols("bga_ph_wahr ~ alter + sofa_score + verstorben_30d", data=df).fit().resid
    return float(np.std(resid, ddof=4))


def clean_delta(delta: float) -> float:
    """Round to the 0.01 grid and normalise away floating-point -0.0."""
    d = round(delta, 2)
    return 0.0 if d == 0 else d


def grid_index(delta: float) -> int:
    """Position of `delta` on the fixed -0.25..0.25 grid (0.01 steps).

    Using a fixed grid (rather than the position within whichever sweep is
    currently running) means every delta gets the same seed regardless of
    whether the primary +-0.10 sweep or the widened +-0.25 sweep produced it.
    """
    return int(round((clean_delta(delta) + 0.25) / DELTA_STEP))


def derive_seed(index: int) -> int:
    """Derive a legacy numpy global seed from SEED and a stream index.

    MICEData draws from the global `np.random` state internally (it has no
    Generator-based API), so we cannot hand it a `default_rng` object
    directly. Instead we use `default_rng([SEED, index])` -- exactly the
    prescribed per-stream derivation -- to draw ONE integer that seeds the
    legacy global state deterministically per delta.
    """
    rng = np.random.default_rng([SEED, index])
    return int(rng.integers(0, 2**31 - 1))


# ---------------------------------------------------------------------------
# One delta: m=20 fresh MICE imputations, shift imputed cells, pool
# ---------------------------------------------------------------------------

def run_delta(work: pd.DataFrame, delta: float, m: int = M) -> dict:
    """Multiple imputation + delta-shift + Rubin's-rules pooling for one delta.

    A FRESH set of m imputations is drawn for every delta (as specified),
    using a delta-specific seed so the whole sweep is reproducible. Only the
    imputed cells of `bga_ph` are shifted by delta; observed cells are
    asserted to be untouched on every single completed dataset.
    """
    np.random.seed(derive_seed(grid_index(delta)))

    imputer = mice.MICEData(work)
    imputer.set_imputer("bga_ph", "alter + sofa_score + verstorben_30d")
    imputer.update_all(N_BURNIN)

    ix_obs = imputer.ix_obs["bga_ph"]
    ix_miss = imputer.ix_miss["bga_ph"]
    observed_original = work["bga_ph"].values[ix_obs]

    betas = np.empty(m)
    variances = np.empty(m)
    for j in range(m):
        imputer.update_all(N_SKIP + 1)
        completed = imputer.data.copy()

        # Only imputed cells may move. This is the load-bearing guarantee of
        # the whole delta-adjustment method.
        assert np.array_equal(completed["bga_ph"].values[ix_obs], observed_original), (
            "observed bga_ph values changed during imputation -- delta leaked "
            "into observed data"
        )
        completed.loc[completed.index[ix_miss], "bga_ph"] = (
            completed["bga_ph"].values[ix_miss] + delta
        )

        fit = smf.logit(MODEL, data=completed).fit(disp=0)
        betas[j] = fit.params["bga_ph"]
        variances[j] = fit.bse["bga_ph"] ** 2

    # Rubin's rules, by hand.
    q_bar = betas.mean()
    u_bar = variances.mean()
    b = betas.var(ddof=1)
    t = u_bar + (1 + 1 / m) * b
    se = np.sqrt(t)

    or_point = float(np.exp(-0.1 * q_bar))
    or_lo = float(np.exp(-0.1 * (q_bar + 1.96 * se)))
    or_hi = float(np.exp(-0.1 * (q_bar - 1.96 * se)))

    return {
        "delta": clean_delta(delta),
        "q_bar": q_bar,
        "u_bar": u_bar,
        "b": b,
        "se": se,
        "OR": or_point,
        "ci_lo": or_lo,
        "ci_hi": or_hi,
        "includes_one": or_lo <= 1.0 <= or_hi,
    }


def sweep(work: pd.DataFrame, deltas: np.ndarray) -> pd.DataFrame:
    rows = [run_delta(work, d) for d in deltas]
    return pd.DataFrame(rows).sort_values("delta").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def section_1_setup(work: pd.DataFrame) -> None:
    print("=" * 78)
    print("1) Delta-adjustment: shift only the imputed values, never the observed")
    print("=" * 78)
    n_miss = int(work["bga_ph"].isna().sum())
    print(f"  bga_ph missing: {n_miss} / {len(work)} ({n_miss / len(work):.1%})")
    print(f"  Analysis model: {MODEL}")
    print(f"  m = {M} imputations, n_burnin = {N_BURNIN}, n_skip = {N_SKIP} (fresh MI per delta)")
    print("\n  bga_ph_imputed(delta) = bga_ph_imputed(MAR) + delta")
    print("  delta = 0 recovers the ordinary MAR multiple-imputation analysis.")


def section_2_sweep(df: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("2) Sweep delta over [-0.10, +0.10] in steps of 0.01 (21 values)")
    print("=" * 78)
    print(f"  {'delta':>7}{'OR / -0.1 pH':>15}{'95% CI':>22}{'CI includes 1?':>18}")
    for _, r in df.iterrows():
        marker = "<-- MAR" if r["delta"] == 0 else ""
        print(
            f"  {r['delta']:+7.2f}{r['OR']:15.3f}"
            f"   ({r['ci_lo']:.3f}, {r['ci_hi']:.3f})"
            f"{str(r['includes_one']):>18}  {marker}"
        )

    row0 = df.loc[np.isclose(df["delta"], 0.0)].iloc[0]
    print(
        f"\n  delta = 0.00 (MAR baseline): OR = {row0['OR']:.2f}, "
        f"95% CI = ({row0['ci_lo']:.2f}, {row0['ci_hi']:.2f})"
    )
    # Sanity-check delta = 0 against the ORACLE (the full, un-masked truth),
    # not against a hardcoded number: a literal reference goes stale the moment
    # anyone retunes the generating process.
    oracle_or = _oracle_odds_ratio()
    print(f"  Oracle (full truth, no missingness):      OR = {oracle_or:.2f}")
    print("  Multiple imputation is stochastic, so exact reproduction is not")
    print("  expected -- but delta = 0 must land in the oracle's neighbourhood.")
    ratio = row0["OR"] / oracle_or
    if not (0.5 <= ratio <= 1.5):
        raise AssertionError(
            f"delta=0 OR ({row0['OR']:.2f}) is {ratio:.2f}x the oracle OR "
            f"({oracle_or:.2f}) -- the delta shift is very likely leaking into "
            "the observed values."
        )
    if bool(row0["includes_one"]):
        print(
            "\n  NOTE: even at delta = 0, the pooled 95 % CI already includes OR = 1.\n"
            "  This is a genuine, seed-stable feature of this cohort (N=500, ~78\n"
            "  deaths) and not a bug in the delta shift: the FULL-DATA truth model\n"
            "  (data/bga_ph_wahrheit.csv, no missingness at all) already has a 95 %\n"
            "  CI for this OR that barely excludes 1, and Rubin's between-imputation\n"
            "  variance widens it further. The MAR point estimate (~1.7-1.9) looks\n"
            "  reassuring, but its own uncertainty was never small to begin with."
        )


def find_literal_tipping_point(df: pd.DataFrame) -> pd.Series | None:
    """Smallest |delta| at which the pooled 95% CI includes OR = 1."""
    hits = df[df["includes_one"]]
    if hits.empty:
        return None
    return hits.loc[hits["delta"].abs().idxmin()]


def significance_region(df: pd.DataFrame) -> tuple[float, float] | None:
    """Contiguous range of delta for which the CI EXCLUDES 1 (association holds)."""
    sig = df[~df["includes_one"]].sort_values("delta")
    if sig.empty:
        return None
    return float(sig["delta"].min()), float(sig["delta"].max())


def section_3_tipping_point(work: pd.DataFrame, df: pd.DataFrame) -> dict:
    print("\n" + "=" * 78)
    print("3) Tipping point")
    print("=" * 78)

    literal = find_literal_tipping_point(df)
    widened = False
    if literal is None:
        print("  The 95% CI never includes OR = 1 within +-0.10.")
        print("  Widening the sweep to +-0.25 to locate the tipping point ...")
        extra_deltas = np.setdiff1d(WIDE_DELTAS, df["delta"].values)
        extra = sweep(work, extra_deltas)
        df = pd.concat([df, extra], ignore_index=True).sort_values("delta").reset_index(drop=True)
        widened = True
        literal = find_literal_tipping_point(df)
        if literal is None:
            print("  The 95% CI STILL never includes OR = 1 within +-0.25.")
            print("  Reporting plainly: no tipping point was found in the searched range.")
        else:
            print(f"  Found within the widened range: delta = {literal['delta']:+.2f}.")

    region = significance_region(df)

    print(
        f"\n  Literal definition (smallest |delta| with CI including 1): "
        f"delta = {literal['delta']:+.2f}" if literal is not None else "\n  No tipping point found."
    )
    if literal is not None and literal["delta"] == 0:
        print(
            "  This is delta = 0 itself: the MAR analysis's own 95% CI already\n"
            "  includes 1 (see section 2), so no MNAR departure is even required to\n"
            "  lose the conclusion in the direction that weakens it further."
        )

    if region is not None:
        lo, hi = region
        print(
            f"\n  Significance boundary (secondary, more informative number): the\n"
            f"  pooled 95% CI EXCLUDES 1 -- i.e. the acidosis-mortality association is\n"
            f"  statistically supported -- only for delta in [{lo:+.2f}, {hi:+.2f}].\n"
            f"  Outside that window (including delta = 0) the finding is not robust."
        )
        # The boundary closest to 0 is the meaningful "how far must the MNAR
        # assumption move before/after the conclusion holds" number.
        boundary = lo if abs(lo) < abs(hi) else hi
    else:
        print("\n  The pooled 95% CI never excludes 1 anywhere in the searched range:")
        print("  the association is not statistically supported for ANY assumed delta here.")
        boundary = None

    return {
        "df": df,
        "widened": widened,
        "literal_tipping": literal,
        "significance_region": region,
        "boundary_delta": boundary,
    }


def section_4_direction(df: pd.DataFrame) -> str:
    print("\n" + "=" * 78)
    print("4) Which direction of delta strengthens, which weakens the association?")
    print("=" * 78)
    print(
        "  Patients without a measured blood gas died more often than patients\n"
        "  with one. Reasoning through both MNAR stories:\n"
        "\n"
        "    delta < 0  ->  unmeasured patients are assumed MORE acidotic (lower\n"
        "                   pH) than the MAR imputation predicts. That is exactly\n"
        "                   consistent with them dying more often -- it reinforces\n"
        "                   the acidosis-mortality story.\n"
        "    delta > 0  ->  unmeasured patients are assumed LESS acidotic (higher\n"
        "                   pH) than the MAR imputation predicts, despite dying\n"
        "                   more often. That contradicts the acidosis-mortality\n"
        "                   story -- their excess mortality is then unexplained by\n"
        "                   pH, so the pH effect must shrink to accommodate them."
    )

    or_neg = df.loc[np.isclose(df["delta"], -0.01), "OR"].iloc[0]
    or_zero = df.loc[np.isclose(df["delta"], 0.00), "OR"].iloc[0]
    or_pos = df.loc[np.isclose(df["delta"], 0.01), "OR"].iloc[0]
    print(
        f"\n  Computed, not guessed: OR(delta=-0.01) = {or_neg:.3f}  "
        f"OR(delta=0) = {or_zero:.3f}  OR(delta=+0.01) = {or_pos:.3f}"
    )
    conservative = "positive (delta > 0)" if or_pos < or_neg else "negative (delta < 0)"
    print(f"  -> OR increases as delta decreases, decreases as delta increases.")
    print(f"  -> The CONSERVATIVE (association-weakening) direction is delta {conservative}.")
    print("     That is the direction a reviewer will ask about: 'what if the patients")
    print("     you didn't measure were actually healthier on this value than you assumed?'")
    return conservative


def section_5_plausibility(tipping: dict) -> None:
    print("\n" + "=" * 78)
    print("5) Clinical plausibility of the tipping-point delta")
    print("=" * 78)

    boundary = tipping["boundary_delta"]
    literal = tipping["literal_tipping"]
    delta_for_verdict = boundary if boundary is not None else (
        literal["delta"] if literal is not None else None
    )

    if delta_for_verdict is None:
        print("  No finite tipping-point delta was found -- no plausibility judgement to make.")
        return

    mag = abs(delta_for_verdict)
    sd = conditional_sd_of_ph()
    in_sd = mag / sd

    print(f"  Relevant magnitude: |delta| = {mag:.2f} pH units.")
    print(f"  Conditional SD of pH given alter, sofa_score, outcome: {sd:.3f}.")
    print(f"  -> the tipping point is {in_sd:.2f} conditional standard deviations.")
    print()
    print("  Do NOT judge delta against blood-gas assay noise. Assay noise is")
    print("  RANDOM measurement error; delta is a SYSTEMATIC difference in the")
    print("  true mean pH between measured and unmeasured patients, after")
    print("  adjusting for everything the imputation model already knows. The")
    print("  honest yardstick is the residual spread, not the analyser's precision.")

    if in_sd < 0.2:
        verdict = (
            f"FRAGILE: {in_sd:.2f} SD is a small systematic shift. A bias this "
            "modest would overturn the conclusion, so the finding does not "
            "survive a realistic departure from MAR."
        )
    elif in_sd < 0.5:
        verdict = (
            f"MODERATELY ROBUST: it takes a systematic shift of {in_sd:.2f} SD -- "
            "unmeasured patients differing from the imputation model by nearly "
            "half a residual standard deviation, in the direction that CONTRADICTS "
            "their higher mortality -- before the conclusion breaks. That is a "
            "specific, arguable assumption, not a trivial one."
        )
    else:
        verdict = (
            f"ROBUST: only a systematic shift of {in_sd:.2f} SD or more overturns "
            "the conclusion. A departure from MAR that large would very likely be "
            "visible in the other measured markers (lactate, SOFA)."
        )

    print(f"\n  delta = {delta_for_verdict:+.2f} ({in_sd:.2f} SD): {verdict}")


def section_6_report(tipping: dict, conservative: str) -> None:
    print("\n" + "=" * 78)
    print("6) The sentence that belongs in the paper")
    print("=" * 78)
    literal = tipping["literal_tipping"]
    region = tipping["significance_region"]
    lit_txt = f"{literal['delta']:+.2f}" if literal is not None else "nicht gefunden"
    region_txt = (
        f"[{region[0]:+.2f}, {region[1]:+.2f}]" if region is not None else "keinen Bereich"
    )
    print(
        f'  "Eine Delta-Adjustment-Sensitivitätsanalyse (m = {M}, 21 Werte von delta\n'
        f"   zwischen -0,10 und +0,10 pH-Einheiten, Rubin's rules) zeigt: das gepoolte\n"
        f"   Odds Ratio je -0,1 pH-Abfall verlässt den signifikanten Bereich (KI\n"
        f"   schließt 1 nicht ein) nur für delta in {region_txt}. Der kleinste |delta|,\n"
        f"   bei dem das 95-%-KI die 1 einschließt, ist delta = {lit_txt}. Die\n"
        f"   konservative, den Effekt abschwächende Richtung ist delta {conservative}:\n"
        f"   sie entspricht der Annahme, unbeobachtete Patient:innen seien trotz\n"
        f'   höherer Mortalität pH-mäßig unauffälliger gewesen als von der\n'
        f'   MAR-Imputation vorhergesagt."'
    )


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(df: pd.DataFrame, tipping: dict) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 5.5))

    df = df.sort_values("delta")
    ax.plot(df["delta"], df["OR"], color=PRIMARY, linewidth=2.0, marker="o", markersize=3.5,
            label="Gepooltes OR (Rubin's rules)")
    ax.fill_between(df["delta"], df["ci_lo"], df["ci_hi"], color=PRIMARY, alpha=0.18,
                     label="95 %-Konfidenzintervall")

    ax.axhline(1.0, color=SECONDARY, linestyle="--", linewidth=1.2, zorder=1)

    ax.axvline(0.0, color="#16181C", linestyle="-", linewidth=1.3, zorder=1)
    ymax = ax.get_ylim()[1]
    ax.text(0.0, ymax, "MAR-Annahme", rotation=90, va="top", ha="right",
            fontsize=10, color="#16181C")

    # Der Kipppunkt ist das ERSTE δ, dessen KI die 1 einschließt — nicht das
    # letzte, bei dem sie es noch nicht tut (`boundary_delta`). Dort steht die
    # Schlussfolgerung noch, sie fällt einen Schritt später.
    literal = tipping.get("literal_tipping")
    if literal is not None:
        kipp = float(literal["delta"])
        if df["delta"].min() <= kipp <= df["delta"].max():
            ax.axvline(kipp, color=EVENT, linestyle=":", linewidth=1.6, zorder=1)
            ax.text(kipp, ymax, f"Kipppunkt (δ={kipp:+.2f})", rotation=90,
                    va="top", ha="right", fontsize=10, color=EVENT)

    ax.set_xlabel("δ — pH-Verschiebung der imputierten Werte")
    ax.set_ylabel("gepooltes OR je 0,1 pH-Abfall")
    ax.set_title("MNAR-Sensitivitätsanalyse: Delta-Adjustment und Kipppunkt")
    ax.legend(loc="upper right")

    save(fig, FIGURE_PATH)


# ---------------------------------------------------------------------------
def main() -> None:
    work = load()
    section_1_setup(work)

    df = sweep(work, PRIMARY_DELTAS)
    section_2_sweep(df)

    tipping = section_3_tipping_point(work, df)
    df = tipping["df"]

    conservative = section_4_direction(df)
    section_5_plausibility(tipping)
    section_6_report(tipping, conservative)

    make_figure(df, tipping)


if __name__ == "__main__":
    main()
