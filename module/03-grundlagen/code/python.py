"""Module 03 — Programming fundamentals for data.

Runs standalone from the project root:
    python module/03-grundlagen/code/python.py

Data: read from data/ (committed with the repo); if that folder is
missing, the same files are fetched from the published URL.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from lib.helpers import SEED, load_cohort  # noqa: E402

pd.set_option("display.width", 100)

# Constant instead of magic number — change in one place only.
FEVER_THRESHOLD = 38.0   # degrees Celsius


def has_fever(temperature: float, threshold: float = FEVER_THRESHOLD) -> bool:
    """Return True when the temperature meets or exceeds the fever threshold."""
    return temperature >= threshold


def bmi_category(bmi: float) -> str:
    """Classify a BMI value according to WHO categories."""
    if bmi >= 30:
        return "adipös"
    elif bmi >= 25:
        return "übergewichtig"
    elif bmi >= 18.5:
        return "normalgewichtig"
    else:
        return "untergewichtig"


def assess_map(map_mmhg: float) -> str:
    """Classify mean arterial pressure (MAP) by clinical thresholds."""
    if map_mmhg < 65:
        return "Schock-Grenzwert unterschritten"
    elif map_mmhg <= 90:
        return "Normbereich"
    else:
        return "Erhöht"


def main() -> None:
    # ---------------------------------------------------------------------- #
    #  1) Variables and data types                                            #
    # ---------------------------------------------------------------------- #
    print("=== 1) Variables and data types ===")

    age: int = 67                    # integer  — patient age
    temperature: float = 38.9       # float    — body temperature
    admission_reason: str = "Sepsis"  # str    — diagnosis
    has_diabetes: bool = True        # bool     — comorbidity flag

    print(f"age:              {age}  ({type(age).__name__})")
    print(f"temperature:      {temperature}  ({type(temperature).__name__})")
    print(f"admission_reason: {admission_reason}  ({type(admission_reason).__name__})")
    print(f"has_diabetes:     {has_diabetes}  ({type(has_diabetes).__name__})")

    # ---------------------------------------------------------------------- #
    #  2) Vectors and lists                                                   #
    # ---------------------------------------------------------------------- #
    print("\n=== 2) Vectors and lists ===")

    # List of body temperatures, as handed over by nursing staff.
    temperatures = [36.8, 37.2, 38.9, 36.5, 39.4, 37.1]

    print("Measurements:", temperatures)
    print("Count:", len(temperatures))
    print("Maximum:", max(temperatures))
    mean_temp = sum(temperatures) / len(temperatures)
    print(f"Mean: {mean_temp:.2f}")

    # Indexing — Python counts from 0.
    print(f"First measurement (index 0): {temperatures[0]}")
    print(f"Last measurement  (index -1): {temperatures[-1]}")

    # Dictionary: key-value pairs — e.g. one patient as a structured record.
    patient = {
        "id": 42,
        "age": 71,
        "aufnahmegrund": "Herzinsuffizienz",
        "sofa_score": 5,
    }
    print(f"\nPatient {patient['id']}: {patient['aufnahmegrund']}, "
          f"age {patient['age']}, SOFA {patient['sofa_score']}")

    # ---------------------------------------------------------------------- #
    #  3) Functions                                                           #
    # ---------------------------------------------------------------------- #
    print("\n=== 3) Functions ===")

    print(f"has_fever(38.9) -> {has_fever(38.9)}")   # True
    print(f"has_fever(37.2) -> {has_fever(37.2)}")   # False

    # Apply function to the entire list via list comprehension.
    fever_readings = [t for t in temperatures if has_fever(t)]
    print(f"Fever readings: {fever_readings}")
    print(f"Fraction with fever: {len(fever_readings) / len(temperatures):.0%}")

    # BMI categorisation.
    sample_bmis = [17.5, 22.3, 27.1, 34.8]
    for b in sample_bmis:
        print(f"  BMI {b:5.1f} -> {bmi_category(b)}")

    # ---------------------------------------------------------------------- #
    #  4) Control flow                                                        #
    # ---------------------------------------------------------------------- #
    print("\n=== 4) Control flow ===")

    # if/elif/else — fever severity tiers.
    for temp in [37.2, 38.4, 39.6]:
        if temp >= 39.0:
            level = "Hohes Fieber — ärztliche Beurteilung erforderlich"
        elif temp >= FEVER_THRESHOLD:
            level = "Fieber"
        else:
            level = "Kein Fieber"
        print(f"  {temp} °C -> {level}")

    # MAP assessment — Surviving Sepsis Campaign threshold.
    print("\nMAP assessment (sample values):")
    for map_val in [58, 72, 95, 63, 88]:
        print(f"  MAP {map_val:3d} mmHg -> {assess_map(map_val)}")

    # ---------------------------------------------------------------------- #
    #  5) DataFrames — tables in code                                        #
    # ---------------------------------------------------------------------- #
    print("\n=== 5) DataFrames — tables in code ===")

    cohort = load_cohort()

    print(f"Shape (rows x cols): {cohort.shape}")
    print("\nData types per column:")
    print(cohort.dtypes)

    # Select columns and show first rows.
    print("\nFirst 5 rows (selected columns):")
    print(cohort[["patient_id", "alter", "aufnahmegrund", "sofa_score",
                  "verstorben_30d"]].head())

    # Filter rows — only patients with Sepsis.
    septic = cohort[cohort["aufnahmegrund"] == "Sepsis"]
    print(f"\nSeptic patients: {len(septic)}")
    print(f"Deceased within 30 days: {septic['verstorben_30d'].sum()}")

    # Derive a new column — BMI category for the whole dataset.
    cohort["bmi_cat"] = cohort["bmi"].apply(
        lambda b: bmi_category(b) if pd.notna(b) else "unbekannt"
    )
    print("\nBMI category distribution:")
    print(cohort["bmi_cat"].value_counts())

    # Descriptive statistics.
    print("\nAge — descriptive statistics:")
    print(cohort["alter"].describe().round(1))

    # Group comparison.
    print("\nMean age by admission reason:")
    print(
        cohort.groupby("aufnahmegrund")["alter"]
        .mean()
        .sort_values(ascending=False)
        .round(1)
    )

    print(f"\nSeed used: {SEED}\nDone.")


if __name__ == "__main__":
    main()
