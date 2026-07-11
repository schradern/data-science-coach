"""Module 32 — Model persistence, drift monitoring, fairness and governance.

Runs standalone from the project root:
    python module/32-einsatz-governance/code/python.py

Data: read from data/ (committed with the repo); if that folder is
missing, the same files are fetched from the published URL.
Requires: scikit-learn, joblib (bundled with scikit-learn), numpy, pandas.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from lib.helpers import SEED, load_cohort, load_notes  # noqa: E402

TARGET = "verschlechterung"

# Drift simulation: flip this fraction of test labels per monitoring step.
# Represents concept drift — the text-outcome relationship has changed
# (e.g. new treatment protocols, different documentation practices).
DRIFT_FLIP_RATES = [0.00, 0.05, 0.10, 0.15, 0.20]

# Bootstrap resamples for subgroup-AUC confidence intervals (Section 4).
N_BOOTSTRAP = 1000


# ---------------------------------------------------------------------------
# 1. Build and train the text pipeline
# ---------------------------------------------------------------------------

def build_text_pipeline() -> Pipeline:
    """TF-IDF + logistic regression — all preprocessing inside the pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True,
        )),
        ("model", LogisticRegression(
            max_iter=1000, class_weight="balanced", C=1.0,
        )),
    ])


def prepare_data():
    """Load and split the clinical notes dataset."""
    df = load_notes()
    texts, y = df["notiz"], df[TARGET]
    return train_test_split(texts, y, test_size=0.25, stratify=y, random_state=SEED)


# ---------------------------------------------------------------------------
# 2. joblib persistence
# ---------------------------------------------------------------------------

def demo_persistence(pipe: Pipeline) -> None:
    """Save and reload the pipeline, verifying sklearn version."""
    print("=== 2) Model persistence with joblib ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "klinisches_modell_v1.joblib"
        artefact = {
            "pipeline": pipe,
            "sklearn_version": sklearn.__version__,
            "target": TARGET,
        }
        joblib.dump(artefact, model_path)
        print(f"  Saved: {model_path.name}  ({model_path.stat().st_size / 1024:.1f} KB)")

        loaded = joblib.load(model_path)
        version_ok = loaded["sklearn_version"] == sklearn.__version__
        print(f"  Version match: {version_ok}  (sklearn {sklearn.__version__})")
        print("  Pipeline re-loaded successfully.")


# ---------------------------------------------------------------------------
# 3. Drift monitoring simulation
# ---------------------------------------------------------------------------

def simulate_concept_drift(y_test: pd.Series, flip_rate: float,
                            rng: np.random.Generator) -> pd.Series:
    """Simulate concept drift by flipping a fraction of test labels.

    Rationale: concept drift means the relationship between text and outcome
    has changed (e.g. new treatment protocols alter which phrases predict
    worsening). From the model's perspective this is indistinguishable from
    label noise — it sees the same text but the 'correct' answer has shifted.
    """
    y_noisy = y_test.copy()
    n_flip = int(flip_rate * len(y_test))
    if n_flip > 0:
        flip_idx = rng.choice(len(y_test), size=n_flip, replace=False)
        y_noisy.iloc[flip_idx] = 1 - y_noisy.iloc[flip_idx]
    return y_noisy


def demo_drift(pipe: Pipeline, X_test: pd.Series, y_test: pd.Series) -> None:
    """Show AUC degradation as concept drift accumulates over monitoring periods."""
    print("\n=== 3) Drift monitoring simulation ===")
    print("  (Concept drift: label-flip simulates changing text-outcome relationship)")
    rng = np.random.default_rng(SEED)
    results = []
    for flip_rate in DRIFT_FLIP_RATES:
        y_drifted = simulate_concept_drift(y_test, flip_rate, rng)
        proba = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_drifted, proba)
        results.append((flip_rate, auc))
        flag = " ← below threshold!" if auc < 0.70 else ""
        print(f"  Drift {flip_rate:.0%}: AUC = {auc:.3f}{flag}")

    baseline = results[0][1]
    drop = baseline - results[-1][1]
    print(f"\n  AUC drop over full drift range: {drop:.3f}")
    print("  Without labels in production, only covariate shift is detectable.")


# ---------------------------------------------------------------------------
# 4. Fairness: subgroup AUC with bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_auc_ci(y: np.ndarray, proba: np.ndarray, n_boot: int = N_BOOTSTRAP,
                      seed: int = SEED) -> tuple[float, float]:
    """95%-bootstrap CI for AUC: resample (y, proba) pairs with replacement.

    The module calls subgroup CIs "Pflicht" (mandatory) — a point-estimate AUC
    gap between subgroups is often just sampling noise in a small test split.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    proba = np.asarray(proba)
    n = len(y)
    boot_aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        y_b, p_b = y[idx], proba[idx]
        if len(np.unique(y_b)) < 2:
            continue  # a resample with only one class has no AUC
        boot_aucs.append(roc_auc_score(y_b, p_b))
    lo, hi = np.percentile(boot_aucs, [2.5, 97.5])
    return float(lo), float(hi)


def _report_subgroup_auc(test_data: pd.DataFrame, proba_all: np.ndarray,
                          column: str, groups: list[str], title: str) -> None:
    """Print AUC + bootstrap 95%-CI per group, then state whether the CIs overlap."""
    print(f"\n  -- {title} --")
    summary: dict[str, tuple[float, float, float, int]] = {}
    for group in groups:
        mask = (test_data[column] == group).values
        if mask.sum() < 5:
            print(f"  {group}: too few samples (n={mask.sum()}) — skip")
            continue
        y_g = test_data.loc[mask, TARGET].to_numpy()
        p_g = proba_all[mask]
        auc_g = roc_auc_score(y_g, p_g)
        lo, hi = bootstrap_auc_ci(y_g, p_g)
        summary[group] = (auc_g, lo, hi, int(mask.sum()))
        print(f"  AUC {group:12s}: {auc_g:.3f}  [95%-CI {lo:.3f}-{hi:.3f}]  (n={mask.sum()})")

    if len(summary) == 2:
        (g1, (a1, lo1, hi1, n1)), (g2, (a2, lo2, hi2, n2)) = summary.items()
        overlap = lo1 <= hi2 and lo2 <= hi1
        if overlap:
            print(f"  -> CIs overlap: the {g1}/{g2} gap is NOT confidently established as real.")
        else:
            print(f"  -> CIs do not overlap: the {g1}/{g2} gap looks like a genuine disparity.")


def demo_fairness(pipe: Pipeline) -> None:
    """Compute AUC by Geschlecht and Altersgruppe subgroup, each with a bootstrap CI."""
    print("\n=== 4) Fairness: subgroup AUC with bootstrap 95%-CI ===")
    cohort = load_cohort()
    # Normalise the 'w' → 'weiblich' dirty encoding from the raw data.
    cohort["geschlecht_clean"] = cohort["geschlecht"].replace({"w": "weiblich"})

    notes = load_notes()
    merged = notes.merge(
        cohort[["patient_id", "geschlecht_clean", "alter"]], on="patient_id", how="left"
    )
    _, test_idx = train_test_split(
        merged.index, test_size=0.25, stratify=merged[TARGET], random_state=SEED
    )
    test_data = merged.loc[test_idx].copy()
    # Clinically common age cut for a worked age-subgroup example (elderly vs. younger).
    test_data["altersgruppe"] = np.where(test_data["alter"] < 65, "<65 Jahre", ">=65 Jahre")

    proba_all = pipe.predict_proba(test_data["notiz"])[:, 1]
    overall_auc = roc_auc_score(test_data[TARGET], proba_all)
    print(f"  Overall AUC: {overall_auc:.3f}")

    _report_subgroup_auc(test_data, proba_all, "geschlecht_clean",
                          ["weiblich", "maennlich"], "Nach Geschlecht")
    _report_subgroup_auc(test_data, proba_all, "altersgruppe",
                          ["<65 Jahre", ">=65 Jahre"], "Nach Altersgruppe (Schwelle 65 Jahre)")

    print("\n  A subgroup AUC difference is only worth investigating once its CI excludes")
    print("  the other group's CI — otherwise it's likely small-sample noise, not a real gap.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== 1) Train text classifier ===")
    X_train, X_test, y_train, y_test = prepare_data()
    pipe = build_text_pipeline()
    pipe.fit(X_train, y_train)
    test_auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
    print(f"  Baseline test AUC: {test_auc:.3f}")

    demo_persistence(pipe)
    demo_drift(pipe, X_test, y_test)
    demo_fairness(pipe)

    print("\n=== 5) Governance note ===")
    print("  Clinical decision support → MDR 2017/745: typically Class IIa, but under")
    print("  Rule 11 software informing decisions that may cause death or irreversible")
    print("  deterioration is Class III (serious deterioration → IIb). A 30-day-mortality")
    print("  CDSS is a IIb/III candidate. EU AI Act: high-risk. Not legal advice.")
    print("  Before deployment: clinical evaluation study, CE marking, PMS plan.")
    print("  This model is for teaching purposes only — not for clinical use.")


if __name__ == "__main__":
    main()
