"""Shared Python helpers (DRY) used by every module.

So no module repeats data-loading logic or hard-codes paths.

Note on language: code is English (identifiers, comments, docstrings); the
dataset is a German clinical schema, so column names stay German and are
accessed by string key, e.g. df["aufnahmegrund"].

Example:
    from lib.helpers import load_cohort, SEED
    df = load_cohort()
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SEED = 42
# Project root = one directory above 'lib/'.
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# Every published URL lives in exactly one file: config.json at the project
# root. Until the course is published, `data_base_url` still holds the literal
# placeholder, and the URL fallback below refuses to run rather than fetching
# a 404 and reporting it as a network error.
PLACEHOLDER = "__SET_ME__"
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
DATA_BASE_URL = CONFIG["data_base_url"]
DATA_BASE_URL_CONFIGURED = PLACEHOLDER not in DATA_BASE_URL


def _load(name: str) -> pd.DataFrame:
    """Local-first, URL-fallback loader.

    Reads from the local `data/` directory when the clone has the file
    (normal course/repo usage). Otherwise falls back to fetching the same
    file from DATA_BASE_URL, so notebooks work in the browser/Colab without
    a local `data/` directory.
    """
    path = DATA / name
    if path.exists():
        return pd.read_csv(path)

    if not DATA_BASE_URL_CONFIGURED:
        raise FileNotFoundError(
            f"'{name}' is missing locally ({path}) and no download URL is set.\n"
            "Either run:  python data/generate_data.py\n"
            "or set 'owner'/'repo' and data_base_url in config.json."
        )
    try:
        return pd.read_csv(DATA_BASE_URL + name)
    except Exception as exc:
        raise FileNotFoundError(
            f"Could not load '{name}' locally ({path} is missing) or from "
            f"{DATA_BASE_URL + name}. Run first: python data/generate_data.py, "
            "or check config.json / your internet connection."
        ) from exc


def load_cohort() -> pd.DataFrame:
    """Patient master table (one row per patient)."""
    return _load("kohorte.csv")


def load_vitals() -> pd.DataFrame:
    """Vital signs in long format (several measurements per patient)."""
    return _load("vitalwerte.csv")


def load_labs() -> pd.DataFrame:
    """Laboratory values in wide format (with missing values)."""
    return _load("labor.csv")


def load_notes() -> pd.DataFrame:
    """Synthetic free-text clinical notes with label `verschlechterung`."""
    return _load("notizen.csv")
