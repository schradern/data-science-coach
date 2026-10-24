"""Module 01 – Introduction & learning path.

Runs standalone from the project root:
    python module/01-einfuehrung/code/python.py

Data: read from data/ (committed with the repo); if that folder is
missing, the same files are fetched from the published URL.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from lib.helpers import SEED, load_cohort  # noqa: E402


def summarise_cohort(df) -> None:
    """Print key statistics that frame the clinical question."""
    n_patients = len(df)
    n_cols = df.shape[1]
    mortality_rate = df["verstorben_30d"].mean()
    n_deceased = df["verstorben_30d"].sum()

    print(f"Dataset loaded:   {n_patients} patients, {n_cols} columns")
    print(f"Columns:          {list(df.columns)}")
    print(f"\n30-day mortality: {mortality_rate:.1%}  ({n_deceased} of {n_patients} patients)")


def main() -> None:
    print("=== Module 01 – First look at the cohort ===\n")

    cohort = load_cohort()
    summarise_cohort(cohort)

    # Admission reasons – the first grouping a clinician would ask for.
    print("\nAdmission reasons (by frequency):")
    print(cohort["aufnahmegrund"].value_counts().to_string())

    # Mortality by admission reason – the clinical question in miniature.
    print("\n30-day mortality by admission reason:")
    mortality_by_reason = (
        cohort.groupby("aufnahmegrund")["verstorben_30d"]
        .mean()
        .sort_values(ascending=False)
        .map(lambda r: f"{r:.1%}")
    )
    print(mortality_by_reason.to_string())

    print(f"\nSeed: {SEED}  – results are fully reproducible.")
    print("\nDone. Next: Module 02 – Tools & reproducible environment.")


if __name__ == "__main__":
    main()
