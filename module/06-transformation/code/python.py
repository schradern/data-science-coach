"""Module 06 — Data cleaning and transformation (Python / pandas).

Runs standalone from the project root:
    python module/06-transformation/code/python.py

Data: read from data/ (committed with the repo); if that folder is
missing, the same files are fetched from the published URL.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from lib.helpers import SEED, load_cohort, load_labs, load_vitals  # noqa: E402

pd.set_option("display.width", 100)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load cohort, labs and vitals from the shared helpers."""
    return load_cohort(), load_labs(), load_vitals()


def main() -> None:
    cohort, labs, vitals = load_data()

    # ------------------------------------------------------------------
    # 1) Standardise categories ('w' -> 'weiblich')
    # ------------------------------------------------------------------
    print("=== 1) Standardise categories ===")
    print("before:", sorted(cohort["geschlecht"].unique()))
    cohort["geschlecht"] = cohort["geschlecht"].replace({"w": "weiblich"})
    print("after: ", sorted(cohort["geschlecht"].unique()))

    # ------------------------------------------------------------------
    # 2) Count missing values — understand before acting
    # ------------------------------------------------------------------
    # Three columns, three different mechanisms. Naming all three here keeps the
    # printed output, the figure and the README's table in agreement.
    print("\n=== 2) Missing values — understand first ===")
    print(f"{'column':16s}{'missing':>9}{'share':>9}   mechanism")
    mechanisms = {
        "bmi": "MCAR — random",
        "gewicht_kg": "MCAR — random",
        "laktat_mmol_l": "depends on sofa_score (a covariate)",
        "bga_ph": "depends on verstorben_30d (the OUTCOME)",
    }
    merged = cohort.merge(labs, on="patient_id", how="left")
    for col, mechanism in mechanisms.items():
        n_missing = int(merged[col].isna().sum())
        print(f"{col:16s}{n_missing:>9d}{n_missing / len(merged):>9.1%}   {mechanism}")
    print("Only the mechanism decides whether dropna() is harmless — see Module 14.")

    # ------------------------------------------------------------------
    # 3) Join: cohort + labs (LEFT JOIN keeps all patients)
    # ------------------------------------------------------------------
    print("\n=== 3) Join: cohort + labs ===")
    df = cohort.merge(labs, on="patient_id", how="left")
    # A left-join on a 1:1 key must not increase the row count.
    assert len(df) == len(cohort), "JOIN multiplied rows — check for duplicate keys!"
    print("Shape after join:", df.shape)

    # ------------------------------------------------------------------
    # 4) Five core verbs: filter, select, derive, group, summarise
    # ------------------------------------------------------------------
    print("\n=== 4) Five core verbs ===")
    sepsis_pts = (
        df.query("aufnahmegrund == 'Sepsis'")
          .loc[:, ["patient_id", "alter", "laktat_mmol_l", "sofa_score"]]
          .assign(high_lactate=lambda d: d["laktat_mmol_l"] > 2.0)
    )
    print(f"Sepsis patients: {len(sepsis_pts)}"
          f"  | lactate > 2 mmol/l: {int(sepsis_pts['high_lactate'].sum())}")

    print("\nMedian lactate per admission type:")
    print(
        df.groupby("aufnahmegrund")["laktat_mmol_l"]
          .median()
          .sort_values(ascending=False)
          .round(2)
    )

    # ------------------------------------------------------------------
    # 5) Reshape long → wide (heart rate per day)
    # ------------------------------------------------------------------
    print("\n=== 5) Reshape long → wide (heart rate per day) ===")
    hr_wide = vitals.pivot_table(
        index="patient_id", columns="tag", values="herzfrequenz"
    )
    hr_wide.columns = [f"hf_tag{t}" for t in hr_wide.columns]
    print(hr_wide.head(3))

    print("\nDone.")


if __name__ == "__main__":
    main()
