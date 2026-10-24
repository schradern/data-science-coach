"""Module 31 – Clinical text classification with TF-IDF and logistic regression.

Runs standalone from the project root:
    python module/31-klinische-texte-llm/code/python.py

Data: read from data/ (committed with the repo); if that folder is
missing, the same files are fetched from the published URL.
Only scikit-learn and standard library are required.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from lib.helpers import SEED, load_notes  # noqa: E402

TARGET = "verschlechterung"


def load_data() -> tuple[pd.Series, pd.Series]:
    """Return (texts, labels) from the clinical notes dataset."""
    df = load_notes()
    return df["notiz"], df[TARGET]


def build_pipeline(C: float = 1.0, ngram_max: int = 2) -> Pipeline:
    """TF-IDF vectoriser followed by logistic regression – both in one pipeline.

    Keeping the vectoriser inside the pipeline ensures IDF weights are estimated
    only on training documents, preventing information leakage into the test set.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, ngram_max),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=C,
        )),
    ])


def top_tokens(pipe: Pipeline, n: int = 15) -> pd.DataFrame:
    """Return the n tokens with the largest absolute log-odds coefficients."""
    feature_names = pipe.named_steps["tfidf"].get_feature_names_out()
    coef = pipe.named_steps["model"].coef_[0]
    idx = np.argsort(np.abs(coef))[::-1][:n]
    return pd.DataFrame({
        "token": feature_names[idx],
        "koeffizient": coef[idx],
    }).sort_values("koeffizient", ascending=False)


def main() -> None:
    texts, y = load_data()

    print("=== 1) Dataset overview ===")
    print(f"  {len(texts)} notes  |  {y.mean():.1%} positive (Verschlechterung)")

    print("\n=== 2) Train/test split ===")
    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=0.25, stratify=y, random_state=SEED
    )

    print("\n=== 3) 5-fold cross-validated AUC (training set only) ===")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    pipe_cv = build_pipeline()
    cv_aucs = cross_val_score(pipe_cv, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"  CV-AUC: {cv_aucs.mean():.3f} ± {cv_aucs.std():.3f}")

    print("\n=== 4) Final model – fit on full training set, evaluate on held-out test ===")
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, proba)
    print(f"  Test-AUC: {test_auc:.3f}")
    print()
    print(classification_report(
        y_test, pipe.predict(X_test),
        target_names=["stabil", "Verschlechterung"],
    ))

    print("\n=== 5) Most predictive tokens ===")
    top = top_tokens(pipe, n=15)
    print(top.to_string(index=False))

    print("\n=== 6) Vocabulary size ===")
    vocab_size = len(pipe.named_steps["tfidf"].vocabulary_)
    print(f"  Vocabulary: {vocab_size} unique tokens/bigrams")
    print("\nA bag-of-words model ignores word order and negation context.")
    print("Embeddings (sentence-transformers) capture semantics – see README, section 5.")


if __name__ == "__main__":
    main()
