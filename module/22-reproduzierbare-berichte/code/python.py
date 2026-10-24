"""Module 22 – Reproducible reports and workflows (Python).

Demonstrates three pillars of reproducible clinical data analysis:
  1. Seed setting (numpy global + random_state per call)
  2. Package version logging into a Markdown manifest
  3. Reproducible random sampling on the course cohort

Runs standalone from the project root:
    python module/22-reproduzierbare-berichte/code/python.py

Data: read from data/ (committed with the repo); if that folder is
missing, the same files are fetched from the published URL.
"""
from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from lib.helpers import SEED, load_cohort  # noqa: E402


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_package_versions(packages: list[str]) -> dict[str, str]:
    """Return installed version strings for the requested package names.

    Args:
        packages: PyPI distribution names (e.g. ``"scikit-learn"``).

    Returns:
        Mapping from package name to version string or ``"not installed"``.
    """
    versions: dict[str, str] = {}
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "not installed"
    return versions


def write_manifest(versions: dict[str, str], output_path: Path) -> None:
    """Write a simple Markdown package manifest.

    The manifest records the seed and all package versions so the analysis
    environment can be reconstructed later.

    Args:
        versions: Mapping from package name to version string.
        output_path: Destination ``.md`` file (created or overwritten).
    """
    lines = [
        "# Paket-Manifest\n\n",
        f"Erzeugt von Modul 22 · Seed: {SEED}\n\n",
        "| Paket | Version |\n",
        "|-------|---------|\n",
    ]
    for pkg, version in versions.items():
        lines.append(f"| `{pkg}` | {version} |\n")
    output_path.write_text("".join(lines), encoding="utf-8")
    print(f"Manifest geschrieben: {output_path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ---------------------------------------------------------------------- #
    # 1) Set the seed – must happen before any random operation               #
    # ---------------------------------------------------------------------- #
    np.random.seed(SEED)
    print(f"=== Seed gesetzt: {SEED} ===\n")

    # ---------------------------------------------------------------------- #
    # 2) Load data                                                             #
    # ---------------------------------------------------------------------- #
    cohort = load_cohort()
    print(f"Kohorte geladen: {cohort.shape[0]} Patient:innen, "
          f"{cohort.shape[1]} Spalten")

    # ---------------------------------------------------------------------- #
    # 3) Reproducible random sample                                            #
    # random_state=SEED makes this independent of the numpy global state      #
    # ---------------------------------------------------------------------- #
    print("\n=== Reproduzierbare Stichprobe (n=50) ===")
    sample = cohort.sample(n=50, random_state=SEED)
    print(f"Mittleres Alter:       {sample['alter'].mean():.1f} Jahre")
    print(f"30-Tage-Mortalitaet:  {sample['verstorben_30d'].mean():.1%}")
    print(f"Erste patient_id:      {sample['patient_id'].iloc[0]}")
    print("(Dieser Wert aendert sich nicht, solange Seed = 42 gesetzt ist.)\n")

    # ---------------------------------------------------------------------- #
    # 4) Log package versions                                                  #
    # ---------------------------------------------------------------------- #
    relevant_packages = ["numpy", "pandas", "scipy", "statsmodels", "scikit-learn"]
    versions = get_package_versions(relevant_packages)

    print("=== Paket-Manifest ===")
    for pkg, version in versions.items():
        print(f"  {pkg:20s}: {version}")

    # Write manifest next to this script (inside code/)
    manifest_path = Path(__file__).parent / "manifest.md"
    write_manifest(versions, manifest_path)

    # ---------------------------------------------------------------------- #
    # 5) Descriptive summary of the sample                                     #
    # ---------------------------------------------------------------------- #
    print("\n=== Deskriptive Zusammenfassung der Stichprobe ===")
    summary = (
        sample
        .groupby("aufnahmegrund")
        .agg(
            n=("patient_id", "count"),
            alter_median=("alter", "median"),
            mortalitaet=("verstorben_30d", "mean"),
        )
        .round({"alter_median": 0, "mortalitaet": 3})
    )
    print(summary.to_string())

    print("\n=== Reproduzierbarkeits-Check ===")
    print("Ergebnis ist reproduzierbar, wenn:")
    print("  - Seed = 42 gesetzt (numpy + random_state)")
    print("  - Datensatz unveraendert (gleicher generate_data.py-Lauf)")
    print("  - Paketversionen wie im Manifest")
    print("\nFertig.")


if __name__ == "__main__":
    main()
