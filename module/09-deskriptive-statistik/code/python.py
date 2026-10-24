"""Module 09 – Descriptive statistics and 'Table 1' (Python / pandas + tableone).

Runs standalone from the project root:
    python module/09-deskriptive-statistik/code/python.py

Data: read from data/ (committed with the repo); if that folder is
missing, the same files are fetched from the published URL.
Package required: pip install tableone
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tableone import TableOne  # noqa: E402

from lib.helpers import SEED, load_cohort, load_labs  # noqa: E402
from lib.plotstyle import apply_style, save, PRIMARY, EVENT, SECONDARY  # noqa: E402

apply_style()
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)
# Figures belong next to the lesson, in assets/ – never in code/.
FIGURE_DIR = Path(__file__).resolve().parent.parent / "assets"
FIGURE_DIR.mkdir(exist_ok=True)


def main() -> None:
    cohort = load_cohort()
    labs   = load_labs()

    # Merge cohort + labs (left join keeps all patients).
    df = cohort.merge(labs, on="patient_id", how="left")
    assert len(df) == len(cohort), "JOIN multiplied rows – check for duplicate keys!"

    # Standardise gender encoding (learned in module 06).
    df["geschlecht"] = df["geschlecht"].replace({"w": "weiblich"})

    # ------------------------------------------------------------------
    # 1) Location measures: mean vs median
    # ------------------------------------------------------------------
    print("=== 1) Location measures: mean vs. median ===")

    # Length of stay is right-skewed – outliers pull the mean upward.
    los = df["verweildauer_tage"]
    print(f"Verweildauer – mean:   {los.mean():.1f} days  ← pulled by long-stay outliers")
    print(f"Verweildauer – median: {los.median():.0f} days  ← typical value")
    print(f"Verweildauer – SD:     {los.std():.1f}")
    print(f"Verweildauer – IQR:    {los.quantile(0.25):.0f} – {los.quantile(0.75):.0f}")

    # Age is approximately normal – mean makes sense here.
    age = df["alter"]
    print(f"\nAlter – mean ± SD:     {age.mean():.1f} ± {age.std():.1f} years")
    print(f"Alter – median [IQR]:   {age.median():.0f}"
          f" [{age.quantile(0.25):.0f}; {age.quantile(0.75):.0f}] years")

    # ------------------------------------------------------------------
    # 2) Spread measures and percentiles
    # ------------------------------------------------------------------
    print("\n=== 2) Spread measures and percentiles ===")

    # CRP is right-skewed – report median [IQR], not mean ± SD.
    crp = df["crp_mg_l"]
    print("CRP mg/l – descriptive summary:")
    print(crp.describe().round(1))
    print(f"Percentiles 10/50/90: "
          f"{crp.quantile(0.10):.1f} / {crp.quantile(0.50):.1f} / {crp.quantile(0.90):.1f} mg/l")

    sofa = df["sofa_score"]
    print(f"\nSOFA-Score – mean ± SD:    {sofa.mean():.1f} ± {sofa.std():.1f}")
    print(f"SOFA-Score – median [IQR]:  {sofa.median():.0f}"
          f" [{sofa.quantile(0.25):.0f}; {sofa.quantile(0.75):.0f}]")

    # ------------------------------------------------------------------
    # 3) Frequencies for categorical variables
    # ------------------------------------------------------------------
    print("\n=== 3) Frequencies – categorical variables ===")

    print("Aufnahmegrund (absolute and relative):")
    freq = df["aufnahmegrund"].value_counts()
    rel  = df["aufnahmegrund"].value_counts(normalize=True).mul(100)
    print(pd.DataFrame({"n": freq, "%": rel.round(1)}))

    print("\nGeschlecht:")
    print(df["geschlecht"].value_counts())

    print("\nDiabetes (0 = nein, 1 = ja):")
    print(df["diabetes"].value_counts().sort_index().rename({0: "nein", 1: "ja"}))
    print(f"Proportion with diabetes: {df['diabetes'].mean():.1%}")

    # ------------------------------------------------------------------
    # 4) Manual group comparison (survivors vs. non-survivors)
    # ------------------------------------------------------------------
    print("\n=== 4) Manual group comparison ===")

    continuous_vars = ["alter", "sofa_score", "crp_mg_l", "verweildauer_tage",
                       "kreatinin_mg_dl", "laktat_mmol_l"]

    print("Median [Q1; Q3] per group (0=survived, 1=died):")
    group_summary = (
        df.groupby("verstorben_30d")[continuous_vars]
          .agg(lambda x: f"{x.median():.1f} [{x.quantile(0.25):.1f}; {x.quantile(0.75):.1f}]")
    )
    print(group_summary.T)

    print("\nDiabetes n (%) per group:")
    diabetes_by_group = (
        df.groupby("verstorben_30d")["diabetes"]
          .agg(n=("sum"), proportion=lambda x: f"{x.mean():.1%}")
    )
    print(diabetes_by_group)

    # ------------------------------------------------------------------
    # 5) Table 1 using the tableone package
    # ------------------------------------------------------------------
    print("\n=== 5) Table 1 (tableone package) ===")

    columns = [
        "alter", "geschlecht", "aufnahmegrund", "diabetes", "hypertonie",
        "raucherstatus", "sofa_score", "crp_mg_l", "verweildauer_tage",
        "kreatinin_mg_dl", "laktat_mmol_l",
    ]
    categorical = ["geschlecht", "aufnahmegrund", "diabetes", "hypertonie", "raucherstatus"]
    # Skewed/ordinal variables: median [Q1,Q3] and Kruskal-Wallis instead of
    # tableone's default mean (SD) and t-test/ANOVA (matches gtsummary in r.R).
    nonnormal = ["sofa_score", "crp_mg_l", "verweildauer_tage",
                 "kreatinin_mg_dl", "laktat_mmol_l"]

    # groupby splits columns by 30-day mortality; p-value is descriptive, not a goal.
    table1 = TableOne(
        df,
        columns=columns,
        categorical=categorical,
        nonnormal=nonnormal,
        groupby="verstorben_30d",
        pval=True,    # descriptive group comparison
        missing=True, # always show missing – never hide them
        rename={
            "verstorben_30d":    "Verstorben (30d)",
            "alter":             "Alter (Jahre)",
            "geschlecht":        "Geschlecht",
            "aufnahmegrund":     "Aufnahmegrund",
            "diabetes":          "Diabetes",
            "hypertonie":        "Hypertonie",
            "raucherstatus":     "Raucherstatus",
            "sofa_score":        "SOFA-Score",
            "crp_mg_l":          "CRP (mg/l)",
            "verweildauer_tage": "Verweildauer (Tage)",
            "kreatinin_mg_dl":   "Kreatinin (mg/dl)",
            "laktat_mmol_l":     "Laktat (mmol/l)",
        },
        label_suffix=False,
    )
    print(table1.tabulate(tablefmt="simple"))

    # ------------------------------------------------------------------
    # 6) Statistical Process Control (SPC) Run Chart
    # ------------------------------------------------------------------
    print("\n=== 6) SPC Run Chart ===")

    # Monthly average wait times in emergency department (24 months)
    months = np.arange(1, 25)
    wait_times = [43.5, 47.2, 41.8, 46.0, 48.5, 42.1, 45.2, 49.0, 44.1, 46.5, 43.0, 45.5,
                  34.5, 31.2, 33.8, 30.5, 29.1, 32.4, 35.0, 31.8, 30.0, 32.5, 33.1, 28.5]

    median_baseline = np.median(wait_times[:12])
    print(f"Baseline-Median (first 12 months): {median_baseline:.2f} minutes")

    # Check if there is a shift (>= 6 consecutive points above/below median)
    below_median = [t < median_baseline for t in wait_times]
    max_consecutive = 0
    current_consecutive = 0
    for val in below_median:
        if val:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0

    print(f"Max consecutive points below median: {max_consecutive} (Shift if >= 6)")

    import matplotlib.pyplot as plt  # noqa: E402  (local import – headless-safe)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(months, wait_times, "o-", color=PRIMARY, label="Messwert")
    ax.axhline(median_baseline, color=SECONDARY, linestyle="--", label="Baseline-Median")
    ax.plot(months[12:], wait_times[12:], "o-", color=EVENT,
            label="Shift (Prozessverbesserung)")
    ax.set_xlabel("Monat")
    ax.set_ylabel("Wartezeit (Minuten)")
    ax.set_title("SPC Run Chart: Wartezeit Notaufnahme")
    ax.legend()
    save(fig, FIGURE_DIR / "spc_run_chart_demo.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
