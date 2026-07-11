"""Module 04 — Reading & extracting data (Python / pandas).

Runs standalone from the project root:
    python module/04-daten-einlesen/code/python.py

Data: read from data/ (committed with the repo); if that folder is
missing, the same files are fetched from the published URL.
Packages: pandas, openpyxl (both in requirements.txt). The API demo below
uses `requests` if installed and falls back to the local dataset otherwise
(`requests` is pinned in requirements.txt; the fallback keeps the lesson
runnable behind a hospital firewall).
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from lib.helpers import SEED, load_cohort  # noqa: E402

pd.set_option("display.width", 100)
pd.set_option("display.max_columns", 10)


# ---------------------------------------------------------------------------
# Helper: produce a synthetic "German-format CSV" (semicolon, decimal comma)
# ---------------------------------------------------------------------------

def _write_german_csv(path: Path) -> None:
    """Write a semicolon / decimal-comma version of a cohort subset."""
    df = load_cohort()[["patient_id", "alter", "bmi", "crp_mg_l"]].head(10).copy()
    path.write_text(df.to_csv(sep=";", decimal=",", index=False), encoding="latin-1")


def main() -> None:
    data_dir = ROOT / "data"

    # ----------------------------------------------------------------------- #
    #  1) CSV — standard case: explicit options                               #
    # ----------------------------------------------------------------------- #
    print("=== 1) CSV — standard case (UTF-8, comma) ===")
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
    #  2) CSV — German format: semicolon, latin-1, decimal comma              #
    # ----------------------------------------------------------------------- #
    print("\n=== 2) CSV — German format (semicolon, latin-1, decimal comma) ===")
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False,
                                     mode="w", encoding="latin-1") as tmp:
        german_path = Path(tmp.name)
    _write_german_csv(german_path)

    cohort_de = pd.read_csv(
        german_path,
        sep=";",
        encoding="latin-1",
        decimal=",",
    )
    print(f"Shape: {cohort_de.shape}  |  Types:\n{cohort_de.dtypes}")
    print(cohort_de.head(3))
    german_path.unlink(missing_ok=True)

    # ----------------------------------------------------------------------- #
    #  3) Excel — create a synthetic file, then read it back                  #
    # ----------------------------------------------------------------------- #
    print("\n=== 3) Excel — create and read ===")
    xl_tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
    xl_tmp.close()
    xl_path = Path(xl_tmp.name)

    with pd.ExcelWriter(xl_path, engine="openpyxl") as writer:
        cohort[["patient_id", "alter", "aufnahmegrund", "sofa_score"]].head(20).to_excel(
            writer, sheet_name="Kohorte", index=False
        )
        cohort[["patient_id", "crp_mg_l", "verweildauer_tage"]].head(20).to_excel(
            writer, sheet_name="Klinik", index=False
        )

    sheet_cohort = pd.read_excel(xl_path, sheet_name="Kohorte")
    print(f"Sheet 'Kohorte': {sheet_cohort.shape}")
    print(sheet_cohort.head(3))

    sheet_clinic = pd.read_excel(xl_path, sheet_name="Klinik")
    print(f"Sheet 'Klinik': {sheet_clinic.shape}")
    xl_path.unlink(missing_ok=True)

    # ----------------------------------------------------------------------- #
    #  4) JSON — flat and nested structures                                   #
    # ----------------------------------------------------------------------- #
    print("\n=== 4) JSON — flat and nested ===")

    # Flat JSON (e.g. REDCap export).
    flat_records = cohort[["patient_id", "alter", "aufnahmegrund"]].head(5).to_dict(orient="records")
    json_text    = json.dumps(flat_records, ensure_ascii=False, indent=2)
    df_flat      = pd.read_json(io.StringIO(json_text))
    print(f"Flat JSON read: {df_flat.shape}")
    print(df_flat)

    # Nested JSON (FHIR-like structure).
    fhir_bundle = {
        "resourceType": "Bundle",
        "entry": [
            {"resource": {"id": str(r["patient_id"]),
                          "birthYear": 2024 - r["alter"],
                          "diagnosis": {"code": r["aufnahmegrund"]}}}
            for r in flat_records[:3]
        ],
    }
    df_fhir = pd.json_normalize(fhir_bundle["entry"], sep="_")
    print("\nNested JSON (FHIR-like) normalised:")
    print(df_fhir)

    # ----------------------------------------------------------------------- #
    #  5) Web API with offline fallback                                       #
    # ----------------------------------------------------------------------- #
    print("\n=== 5) Web API with offline fallback ===")
    API_URL = "https://disease.sh/v3/covid-19/countries?allowNull=false&limit=10"

    try:
        import requests  # noqa: PLC0415

        response = requests.get(API_URL, timeout=5)
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
    print("\n=== 6) kohorte.csv — explicit and safe ===")
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
