"""Module 02 – Tools & reproducible environment.

Runs standalone from the project root:
    python module/02-werkzeuge/code/python.py

Data: read from data/ (committed with the repo); if that folder is
missing, the same files are fetched from the published URL.
Checks that all key packages are installed and the dataset is reachable.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from lib.helpers import SEED, load_cohort  # noqa: E402


def package_version(name: str) -> str:
    """Return version string for an installed package, or 'NOT INSTALLED'."""
    try:
        mod = importlib.import_module(name)
        return getattr(mod, "__version__", "installed")
    except ImportError:
        return "NOT INSTALLED"


def main() -> None:
    print("=== Module 02 – Environment check ===\n")

    print(f"Python version:   {sys.version.split()[0]}")
    print(f"Project root:     {ROOT}")
    print(f"Seed:             {SEED}")

    packages = [
        "pandas", "numpy", "matplotlib", "seaborn",
        "scipy", "statsmodels", "sklearn",
    ]
    print("\nInstalled packages:")
    all_ok = True
    for pkg in packages:
        version = package_version(pkg)
        ok = version != "NOT INSTALLED"
        if not ok:
            all_ok = False
        status = "OK" if ok else "MISSING"
        display = "scikit-learn" if pkg == "sklearn" else pkg
        print(f"  {display:<20} {version:<20} [{status}]")

    print("\nDataset check:")
    try:
        cohort = load_cohort()
        print(f"  kohorte.csv  -> {cohort.shape[0]} rows × {cohort.shape[1]} cols  [OK]")
    except FileNotFoundError as exc:
        print(f"  ERROR: {exc}")
        all_ok = False

    print()
    if all_ok:
        print("All checks passed. Environment is ready.")
    else:
        print("Some packages are missing. Install with:")
        print("  uv pip install -r requirements.txt")


if __name__ == "__main__":
    main()
