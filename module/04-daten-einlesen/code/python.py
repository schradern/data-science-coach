"""Module 04 – Reading & extracting data (Python / pandas).

Runs standalone from the project root:
    python module/04-daten-einlesen/code/python.py

Data: the same companion files the lesson reads (kohorte_de.csv,
einschluss.xlsx, beispiel.json, fhir_bundle.json), taken from data/ when the
repo is cloned and otherwise fetched from the published URL.
Packages: pandas, openpyxl (both in requirements.txt). The API demo below
uses `requests` if installed and falls back to the local dataset otherwise
(`requests` is pinned in requirements.txt; the fallback keeps the lesson
runnable behind a hospital firewall).
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from lib.helpers import DATA_BASE_URL, SEED, load_cohort  # noqa: E402

pd.set_option("display.width", 100)
pd.set_option("display.max_columns", 10)


def _source(name: str) -> Path | str:
    """Local copy in data/ if present, else the published URL of the same file."""
    local = ROOT / "data" / name
    return local if local.exists() else DATA_BASE_URL + name


def main() -> None:
    data_dir = ROOT / "data"

    # ----------------------------------------------------------------------- #
    #  1) CSV – standard case: explicit options                               #
    # ----------------------------------------------------------------------- #
    print("=== 1) CSV – standard case (UTF-8, comma) ===")
    cohort = pd.read_csv(
        data_dir / "kohorte.csv",
        sep=",",
        encoding="utf-8",
        decimal=".",
        dtype={"patient_id": int, "diabetes": int,
               "hypertonie": int, "verstorben_30d": int},
        na_values=["", "NA", "N/A", "-"],
    )
    print(f"Shape: {cohort.shape}  |  Columns: {list(cohort.columns)}")
    print(cohort.head(3))

    # ----------------------------------------------------------------------- #
    #  2) CSV – German format: semicolon, latin-1, decimal comma              #
    # ----------------------------------------------------------------------- #
    print("\n=== 2) CSV – German format (semicolon, latin-1, decimal comma) ===")
    cohort_de = pd.read_csv(
        _source("kohorte_de.csv"),
        sep=";",
        encoding="latin-1",
        decimal=",",
    )
    print(f"Shape: {cohort_de.shape}")
    print(cohort_de[["patient_id", "geschlecht", "gewicht_kg", "bmi"]].head(3))
    print(f"gewicht_kg read as {cohort_de['gewicht_kg'].dtype} "
          f"(decimal comma understood)")

    # Without encoding="latin-1" the umlaut in "männlich" is an invalid byte.
    try:
        pd.read_csv(_source("kohorte_de.csv"), sep=";", decimal=",")
    except UnicodeDecodeError as exc:
        print(f"Read as UTF-8 instead: UnicodeDecodeError ({exc.reason})")

    # ----------------------------------------------------------------------- #
    #  3) Excel – one workbook, three sheets                                  #
    # ----------------------------------------------------------------------- #
    print("\n=== 3) Excel – einschluss.xlsx ===")
    xl_source = _source("einschluss.xlsx")
    print(f"Sheets: {pd.ExcelFile(xl_source).sheet_names}")
    sheet_labs = pd.read_excel(xl_source, sheet_name="Labor", usecols="A:F")
    print(f"Sheet 'Labor': {sheet_labs.shape}")
    print(sheet_labs.head(3))

    # ----------------------------------------------------------------------- #
    #  4) JSON – flat and nested structures                                   #
    # ----------------------------------------------------------------------- #
    print("\n=== 4) JSON – flat and nested ===")

    # Flat JSON (e.g. REDCap export).
    df_flat = pd.read_json(_source("beispiel.json"))
    print(f"Flat JSON (beispiel.json): {df_flat.shape}")
    print(df_flat.head(3))

    # Nested JSON (FHIR bundle).
    fhir_source = _source("fhir_bundle.json")
    if isinstance(fhir_source, Path):
        bundle = json.loads(fhir_source.read_text(encoding="utf-8"))
    else:
        with urllib.request.urlopen(fhir_source, timeout=30) as response:
            bundle = json.load(response)
    df_fhir = pd.json_normalize(bundle["entry"], sep="_")
    print(f"\nNested JSON (fhir_bundle.json) normalised: {df_fhir.shape}")
    print(f"Columns: {list(df_fhir.columns[:6])} …")

    # ----------------------------------------------------------------------- #
    #  5) Web API with offline fallback                                       #
    # ----------------------------------------------------------------------- #
    print("\n=== 5) Web API with offline fallback ===")
    API_URL = "https://disease.sh/v3/covid-19/countries?allowNull=false&limit=10"

    try:
        import requests  # noqa: PLC0415

        # Some APIs reject the library's default User-Agent with HTTP 403.
        response = requests.get(API_URL, timeout=5,
                                headers={"User-Agent": "data-science-coach/1.0"})
        response.raise_for_status()
        df_api = pd.DataFrame(response.json())
        print(f"API call succeeded: {len(df_api)} countries, "
              f"columns: {list(df_api.columns[:5])} …")
        print(df_api[["country", "cases", "deaths"]].head(5))
    except Exception as exc:  # noqa: BLE001
        print(f"Note: API unreachable ({type(exc).__name__}: {exc}).")
        print("Offline fallback: using local cohort dataset.")
        df_api = load_cohort()
        print(f"  -> {df_api.shape[0]} rows from kohorte.csv loaded.")

    # ----------------------------------------------------------------------- #
    #  6) Summary: explicit, safe CSV import with missing-value check        #
    # ----------------------------------------------------------------------- #
    print("\n=== 6) kohorte.csv – explicit and safe ===")
    cohort_final = pd.read_csv(
        data_dir / "kohorte.csv",
        sep=",",
        encoding="utf-8",
        decimal=".",
        dtype={"patient_id": int, "diabetes": int,
               "hypertonie": int, "verstorben_30d": int},
        na_values=["", "NA", "N/A", "-"],
    )
    print(f"Shape: {cohort_final.shape}")
    missing = cohort_final.isna().sum()
    print("Missing values per column (>0 only):")
    print(missing[missing > 0])

    print(f"\nSeed: {SEED}\nDone.")


if __name__ == "__main__":
    main()
