"""Module 17 - Classic survival analysis."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from lifelines import CoxPHFitter, KaplanMeierFitter  # noqa: E402
from lifelines.statistics import logrank_test  # noqa: E402

from lib.helpers import load_cohort  # noqa: E402

# Real time-to-event columns: fu_zeit_tage (follow-up time) + status
# (1 = death observed, 0 = censored). verweildauer_tage (length of stay) is
# a separate, purely descriptive column and must NOT be used as survival time.


def main() -> None:
    df = load_cohort()
    df["sepsis"] = (df["aufnahmegrund"] == "Sepsis").astype(int)

    print("\n1) Kaplan-Meier survival at fixed times")
    for label, g in [("Sepsis", df[df["sepsis"] == 1]), ("Non-sepsis", df[df["sepsis"] == 0])]:
        km = KaplanMeierFitter(label=label)
        km.fit(g["fu_zeit_tage"], event_observed=g["status"])
        print(label, "- median survival:", km.median_survival_time_)
        print(km.survival_function_at_times([10, 20, 30]).round(3))

    print("\n2) Log-rank test")
    a = df[df["sepsis"] == 1]
    b = df[df["sepsis"] == 0]
    lr = logrank_test(
        a["fu_zeit_tage"],
        b["fu_zeit_tage"],
        event_observed_A=a["status"],
        event_observed_B=b["status"],
    )
    print(lr.summary.round(4))

    print("\n3) Cox proportional hazards model")
    cox_df = df[["fu_zeit_tage", "status", "sepsis", "alter", "sofa_score"]].copy()
    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col="fu_zeit_tage", event_col="status")
    print(cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].round(3))

    print("\n4) Proportional-hazards assumption check")
    cph.check_assumptions(cox_df, p_value_threshold=0.05, show_plots=False)


if __name__ == "__main__":
    main()
