"""Figures for module 32. Run: python module/32-einsatz-governance/code/figures.py

Writes PNGs to ../assets/. German labels (display), English code.
Requires only scikit-learn, matplotlib, numpy, pandas.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from lib.helpers import SEED, load_cohort, load_notes  # noqa: E402
from lib.plotstyle import EVENT, PRIMARY, SECONDARY, apply_style, save  # noqa: E402

ASSETS = Path(__file__).resolve().parent.parent / "assets"
TARGET = "verschlechterung"

# Drift is simulated by label-flipping: represents concept drift
# (text-outcome relationship changes, e.g. new documentation practice).
# Chosen so the AUC curve degrades monotonically and crosses the 0.70
# acceptance line at the final monitoring point (matches the README caption).
DRIFT_FLIP_RATES = [0.00, 0.05, 0.10, 0.15, 0.20]

# Bootstrap resamples for the subgroup-AUC confidence intervals.
N_BOOTSTRAP = 1000


def bootstrap_auc_ci(y: np.ndarray, proba: np.ndarray, n_boot: int = N_BOOTSTRAP,
                      seed: int = SEED) -> tuple[float, float]:
    """95%-bootstrap CI for AUC: resample (y, proba) pairs with replacement."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    proba = np.asarray(proba)
    n = len(y)
    boot_aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        y_b, p_b = y[idx], proba[idx]
        if len(np.unique(y_b)) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_b, p_b))
    lo, hi = np.percentile(boot_aucs, [2.5, 97.5])
    return float(lo), float(hi)


def build_and_fit_pipeline(X_train: pd.Series, y_train: pd.Series) -> Pipeline:
    """Fit TF-IDF + logistic regression on training data."""
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True,
        )),
        ("model", LogisticRegression(
            max_iter=1000, class_weight="balanced", C=1.0,
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def simulate_concept_drift(y_test: pd.Series, flip_rate: float,
                            rng: np.random.Generator) -> pd.Series:
    """Simulate concept drift by flipping a fraction of test labels.

    Label-flipping models the scenario where the text-outcome relationship has
    changed (new documentation language, updated treatment protocols).
    """
    y_noisy = y_test.copy()
    n_flip = int(flip_rate * len(y_test))
    if n_flip > 0:
        flip_idx = rng.choice(len(y_test), size=n_flip, replace=False)
        y_noisy.iloc[flip_idx] = 1 - y_noisy.iloc[flip_idx]
    return y_noisy


def fig_drift_monitoring(pipe: Pipeline, X_test: pd.Series, y_test: pd.Series) -> None:
    """Line plot of AUC vs. simulated concept drift level over 5 monitoring periods."""
    rng = np.random.default_rng(SEED)
    aucs = []
    for flip_rate in DRIFT_FLIP_RATES:
        y_drifted = simulate_concept_drift(y_test, flip_rate, rng)
        proba = pipe.predict_proba(X_test)[:, 1]
        aucs.append(roc_auc_score(y_drifted, proba))

    labels = [f"T{i+1}\n({fr:.0%} Drift)" for i, fr in enumerate(DRIFT_FLIP_RATES)]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(range(len(DRIFT_FLIP_RATES)), aucs, marker="o", color=PRIMARY, lw=2, label="AUC")
    ax.axhline(aucs[0], color=SECONDARY, lw=0.9, ls="--",
               label=f"Baseline T1 ({aucs[0]:.2f})")
    ax.axhline(0.70, color=EVENT, lw=0.9, ls=":",
               label="Akzeptanzgrenze (0.70)")
    ax.fill_between(range(len(DRIFT_FLIP_RATES)), aucs, aucs[0],
                    where=[a < aucs[0] for a in aucs],
                    alpha=0.15, color=EVENT)
    ax.set_xticks(range(len(DRIFT_FLIP_RATES)))
    ax.set_xticklabels(labels)
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("Monitoring-Zeitpunkt (Anteil Konzept-Drift)")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("AUC-Verlauf unter simuliertem Konzept-Drift")
    ax.legend(loc="lower left")
    save(fig, ASSETS / "drift_monitoring.png")


def fig_subgruppen_auc(pipe: Pipeline) -> None:
    """Bar chart of AUC by Geschlecht subgroup, with bootstrap 95%-CI error bars."""
    cohort = load_cohort()
    cohort["geschlecht_clean"] = cohort["geschlecht"].replace({"w": "weiblich"})
    notes = load_notes()
    merged = notes.merge(
        cohort[["patient_id", "geschlecht_clean"]], on="patient_id", how="left"
    )

    _, test_idx = train_test_split(
        merged.index, test_size=0.25, stratify=merged[TARGET], random_state=SEED
    )
    test_data = merged.loc[test_idx].copy()

    proba_all = pipe.predict_proba(test_data["notiz"])[:, 1]
    overall_auc = roc_auc_score(test_data[TARGET], proba_all)

    # Display labels with correct German umlauts — "maennlich" is only the
    # internal data-encoding value, not what should appear on the chart.
    display_name = {"weiblich": "Weiblich", "maennlich": "Männlich"}

    groups, aucs, err_lo, err_hi, cis = [], [], [], [], []
    for group in ["weiblich", "maennlich"]:
        mask = test_data["geschlecht_clean"] == group
        if mask.sum() < 5:
            continue
        y_g = test_data.loc[mask, TARGET].to_numpy()
        p_g = proba_all[mask.values]
        auc_g = roc_auc_score(y_g, p_g)
        lo, hi = bootstrap_auc_ci(y_g, p_g)
        groups.append(display_name[group])
        aucs.append(auc_g)
        err_lo.append(auc_g - lo)
        err_hi.append(hi - auc_g)
        cis.append((lo, hi))

    # Flag a bar only if its CI does NOT overlap the overall AUC — a point
    # estimate more than 0.05 away from the mean can still be pure noise if
    # the CI is wide, so colour-coding must be based on the interval, not
    # the point estimate alone.
    colors = [EVENT if not (lo <= overall_auc <= hi) else PRIMARY for lo, hi in cis]

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.bar(groups, aucs, color=colors, width=0.5,
           yerr=[err_lo, err_hi], capsize=6, ecolor="#3A3A3A", error_kw={"lw": 1.3})
    ax.axhline(overall_auc, color=SECONDARY, lw=1.2, ls="--",
               label=f"Gesamt-AUC ({overall_auc:.2f})")
    for i, (g, a, (lo, hi)) in enumerate(zip(groups, aucs, cis)):
        ax.text(i, hi + 0.02, f"{a:.2f}\n[{lo:.2f}-{hi:.2f}]", ha="center", fontsize=8.5)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Subgruppenleistung nach Geschlecht (mit 95%-Bootstrap-CI)")
    ax.legend(loc="lower right")
    save(fig, ASSETS / "subgruppen_auc.png")


def main() -> None:
    apply_style()

    notes = load_notes()
    texts, y = notes["notiz"], notes[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=0.25, stratify=y, random_state=SEED
    )

    pipe = build_and_fit_pipeline(X_train, y_train)

    fig_drift_monitoring(pipe, X_test, y_test)
    fig_subgruppen_auc(pipe)


if __name__ == "__main__":
    main()
