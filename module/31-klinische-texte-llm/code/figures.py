"""Figures for module 31. Run: python module/31-klinische-texte-llm/code/figures.py

Writes PNGs to ../assets/. German labels (display), English code.
Requires only scikit-learn and matplotlib.
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
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from lib.helpers import SEED, load_notes  # noqa: E402
from lib.plotstyle import EVENT, PRIMARY, SECONDARY, apply_style, save  # noqa: E402

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def build_and_fit_pipeline(X_train, y_train) -> Pipeline:
    """Fit TF-IDF + logistic regression on training data."""
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=1.0,
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def fig_top_tokens(pipe: Pipeline) -> None:
    """Horizontal signed bar chart of the 15 most predictive tokens."""
    feature_names = pipe.named_steps["tfidf"].get_feature_names_out()
    coef = pipe.named_steps["model"].coef_[0]
    n = 15
    idx = np.argsort(np.abs(coef))[::-1][:n]
    tokens = feature_names[idx]
    values = coef[idx]

    # Sort by coefficient for visual clarity (negative → positive)
    order = np.argsort(values)
    tokens_sorted = tokens[order]
    values_sorted = values[order]

    colors = [EVENT if v > 0 else PRIMARY for v in values_sorted]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.barh(tokens_sorted, values_sorted, color=colors, height=0.7)
    ax.axvline(0, color=SECONDARY, lw=0.9, ls="--")
    ax.set_xlabel("Log-Odds-Koeffizient (logistische Regression)")
    ax.set_title("Prädiktivste Tokens für klinische Verschlechterung")

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=EVENT, label="Verschlechterung (positiv)"),
        Patch(facecolor=PRIMARY, label="stabil (negativ)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")
    save(fig, ASSETS / "top_tokens.png")


def fig_text_roc(pipe: Pipeline, X_test, y_test) -> None:
    """ROC curve for the text classifier on the held-out test set."""
    proba = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, color=PRIMARY, lw=2, label=f"TF-IDF + LogReg (AUC = {auc:.2f})")
    ax.plot([0, 1], [0, 1], color=SECONDARY, lw=0.9, ls="--", label="Zufallsklassifikator")
    ax.set_xlabel("Falsch-Positiv-Rate (1 – Spezifität)")
    ax.set_ylabel("Richtig-Positiv-Rate (Sensitivität)")
    ax.set_title("ROC-Kurve: Textklassifikator für Verschlechterung")
    ax.legend(loc="lower right")
    save(fig, ASSETS / "text_roc.png")


def main() -> None:
    apply_style()
    df = load_notes()
    texts, y = df["notiz"], df["verschlechterung"]

    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=0.25, stratify=y, random_state=SEED
    )

    pipe = build_and_fit_pipeline(X_train, y_train)

    fig_top_tokens(pipe)
    fig_text_roc(pipe, X_test, y_test)


if __name__ == "__main__":
    main()
