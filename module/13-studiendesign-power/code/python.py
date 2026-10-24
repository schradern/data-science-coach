"""Module 13 - Study design, power, and precision."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from statsmodels.stats.power import TTestIndPower  # noqa: E402
from statsmodels.stats.proportion import proportion_confint  # noqa: E402

from lib.helpers import load_cohort  # noqa: E402


def main() -> None:
    df = load_cohort()
    power = TTestIndPower()

    print("\n1) Sample size per group for two-sample t-test")
    rows = []
    for effect in [0.2, 0.3, 0.5, 0.8]:
        n = power.solve_power(effect_size=effect, alpha=0.05, power=0.8, ratio=1.0)
        rows.append({"cohens_d": effect, "n_per_group": n})
    print(pd.DataFrame(rows).round(1).to_string(index=False))

    print("\n2) Power at the observed diabetes group sizes for a PLANNING effect d=0.5")
    n_diabetes = int(df["diabetes"].sum())
    n_no = len(df) - n_diabetes
    ratio = n_no / n_diabetes
    # NOTE: this is power for an *a-priori* planning effect (d=0.5) at the
    # observed sample sizes – NOT "observed/post-hoc power" computed from the
    # measured effect, which is a deterministic function of the p-value and
    # therefore uninformative (Hoenig & Heisey 2001; see module's Fallstricke).
    power_at_observed_n = power.power(effect_size=0.5, nobs1=n_diabetes, ratio=ratio, alpha=0.05)
    print(f"n diabetes={n_diabetes}, n no diabetes={n_no}, power={power_at_observed_n:.3f}")

    print("\n3) Precision of 30-day mortality rate")
    events = int(df["verstorben_30d"].sum())
    n = len(df)
    lo, hi = proportion_confint(events, n, alpha=0.05, method="wilson")
    print(f"events={events}/{n} ({events/n:.3f}), Wilson 95% CI [{lo:.3f}, {hi:.3f}]")

    print("\n4) Events per variable")
    for parameters in [3, 6, 10, 12]:
        print(f"{parameters:2d} parameters -> EPV={events / parameters:.1f}")


if __name__ == "__main__":
    main()
