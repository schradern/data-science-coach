"""Module 07 - parse a FHIR Bundle (Patient/Observation) into an analysis table.

Run: python module/07-fhir-omop-praxis/code/python.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from lib.helpers import DATA_BASE_URL  # noqa: E402


def load_fhir_bundle() -> dict:
    """Local-first, URL-fallback loader for the FHIR Bundle JSON.

    Mirrors the pattern in lib.helpers._load(): reads data/fhir_bundle.json
    when a local clone has it, otherwise fetches it from DATA_BASE_URL, so
    this script runs offline in the repo and online without one.
    """
    path = ROOT / "data" / "fhir_bundle.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    with urlopen(DATA_BASE_URL + "fhir_bundle.json") as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    bundle = load_fhir_bundle()

    # 1. Unpack the FHIR Bundle into a flat table (one JSON object per resource).
    flat = pd.json_normalize(bundle["entry"], sep="_")

    # 2. Separate Patient and Observation resources (different fields).
    patients = (
        flat[flat["resource_resourceType"] == "Patient"]
        [["resource_id", "resource_gender", "resource_birthDate"]]
        .rename(columns={"resource_id": "patient_id", "resource_gender": "gender",
                         "resource_birthDate": "birth_date"})
    )

    observations = flat[flat["resource_resourceType"] == "Observation"].copy()
    observations["patient_id"] = observations["resource_subject_reference"].str.split("/").str[-1]

    # 3. Pivot observations from long (one row per measurement) to wide (one column
    #    per concept) -- this is the step that turns nested FHIR JSON into an
    #    analysable table.
    obs_wide = observations.pivot(index="patient_id", columns="resource_code_text",
                                  values="resource_valueQuantity_value")
    obs_wide = obs_wide.rename(columns={"SOFA score": "sofa_score", "C-reactive protein": "crp_mg_l"})

    analysis = patients.merge(obs_wide, on="patient_id", how="left")

    print("=== Analysetabelle aus dem FHIR-Bundle (Patient + Observation) ===")
    print(analysis.to_string(index=False))

    print("\nOMOP-Gedanke: 'SOFA score' und 'C-reactive protein' sind hier Freitext")
    print("(code.text). Fuer eine multizentrische Analyse muesste jede Observation vor")
    print("dem Zusammenfuehren auf eine standardisierte Concept-ID (LOINC -> OMOP")
    print("measurement_concept_id) gemappt werden, nicht auf den Rohtext gefiltert.")


if __name__ == "__main__":
    main()
