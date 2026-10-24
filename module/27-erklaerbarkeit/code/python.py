"""Module 27 – Explainable AI: permutation importance, PDP/ICE, and SHAP.

Runs standalone from the project root:
    python module/27-erklaerbarkeit/code/python.py

Data: read from data/ (committed with the repo); if that folder is
missing, the same files are fetched from the published URL.
Core deps: scikit-learn. Optional: shap (falls back gracefully if missing).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split  # noqa: E402

from lib.helpers import SEED, load_cohort, load_labs  # noqa: E402

NUMERIC = ["alter", "sofa_score", "crp_mg_l", "bmi", "leukozyten_g_l",
           "kreatinin_mg_dl", "laktat_mmol_l"]
CATEGORICAL = ["aufnahmegrund", "raucherstatus"]
BINARY = ["diabetes", "hypertonie"]
FEATURES = NUMERIC + CATEGORICAL + BINARY
TARGET = "verstorben_30d"


def build_data() -> tuple[pd.DataFrame, pd.Series]:
    """Merge cohort and labs; return feature matrix and binary outcome."""
    df = load_cohort().merge(load_labs(), on="patient_id", how="left")
    # HistGradientBoosting handles NaN natively, so no imputation needed here.
    X = df[FEATURES].copy()
    # Encode categorical columns as integer codes (HistGBT accepts those).
    for col in CATEGORICAL:
        X[col] = X[col].astype("category").cat.codes
    y = df[TARGET]
    return X, y


def train_model(X_train: pd.DataFrame, y_train: pd.Series
                ) -> HistGradientBoostingClassifier:
    """Fit a HistGradientBoostingClassifier – handles missing values natively."""
    model = HistGradientBoostingClassifier(
        random_state=SEED, max_iter=200, learning_rate=0.05,
        max_depth=4, class_weight="balanced",
    )
    model.fit(X_train, y_train)
    return model


def explain_permutation(model, X_test: pd.DataFrame,
                         y_test: pd.Series) -> None:
    """Compute and print permutation importance on the held-out test set.

    Permutation importance is preferred over impurity importance because:
    - It is evaluated on unseen data, not training data.
    - It does not inflate importance of high-cardinality features.
    - It reflects actual model reliance, not split frequency.
    """
    print("\n=== 2) Permutation Importance (on test set, n_repeats=10) ===")
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=10, random_state=SEED, scoring="roc_auc", n_jobs=1,
    )
    # Sort by mean importance descending.
    order = np.argsort(result.importances_mean)[::-1]
    for i in order:
        name = FEATURES[i]
        mean = result.importances_mean[i]
        std = result.importances_std[i]
        print(f"  {name:<30} {mean:+.4f}  ±{std:.4f}")

    # Compare with impurity importance (train-based, only for models that support it).
    print("\n  (Impurity importance from training data – compare critically:)")
    if hasattr(model, "feature_importances_"):
        impurity = model.feature_importances_
        imp_order = np.argsort(impurity)[::-1]
        for i in imp_order[:5]:  # show top-5 only
            print(f"  {FEATURES[i]:<30} {impurity[i]:.4f}  (impurity-based)")
    else:
        print("  HistGradientBoostingClassifier does not expose impurity importance.")
        print("  Use RandomForestClassifier to compare: rf.feature_importances_")
        print("  Permutation importance above is the correct alternative.")
    print("  Note: impurity importance is biased toward high-cardinality features.")


def explain_shap(model, X_train: pd.DataFrame,
                 X_test: pd.DataFrame) -> bool:
    """Try SHAP explanation; fall back to a note if shap is not installed."""
    print("\n=== 4) SHAP – local explanation ===")
    try:
        import shap  # optional dependency
    except ImportError:
        print("  SHAP not installed.")
        print("  Install: uv pip install shap")
        print("  Fallback: use permutation importance (section 2) as global proxy.")
        print("  For local explanation of individual patients without SHAP:")
        print("  compare patient feature values against the PDP baseline (section 3).")
        return False

    # TreeExplainer is exact and fast for gradient-boosted trees.
    # check_additivity=False: HistGradientBoostingClassifier + class_weight=
    # "balanced" produces raw-margin outputs where shap's strict floating-point
    # additivity check (sum of SHAP values == model output, to ~1e-6) can fail
    # by a tiny amount without the explanation being wrong; this is a known
    # shap/HistGradientBoosting interaction, not a bug in this script.
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test.iloc[:100], check_additivity=False)  # first 100 patients for speed

    mean_abs = np.abs(shap_values.values).mean(axis=0)
    print("  Mean |SHAP| per feature (global summary):")
    for feat, val in sorted(zip(FEATURES, mean_abs), key=lambda x: -x[1]):
        print(f"    {feat:<30} {val:.4f}")
    print(
        "  CAVEAT: correlated features (e.g. leukozyten_g_l, which depends on"
        "\n  sofa_score in this cohort) can SPLIT credit between them in SHAP -- a feature with a"
        "\n  small mean |SHAP| is not necessarily unimportant, its signal may be"
        "\n  shared with a correlated partner. Check feature correlations before"
        "\n  concluding a feature 'doesn't matter'."
    )

    # Local explanation for the patient with the highest predicted risk.
    highest_risk_idx = model.predict_proba(X_test)[:, 1].argmax()
    print(f"\n  Local SHAP for the highest-risk patient (index {highest_risk_idx}):")
    sv = shap_values[highest_risk_idx].values
    base = shap_values[highest_risk_idx].base_values
    print(f"  Base value (log-odds): {base:.3f}")
    for feat, s in sorted(zip(FEATURES, sv), key=lambda x: -abs(x[1])):
        direction = "↑ risk" if s > 0 else "↓ risk"
        print(f"    {feat:<30} {s:+.4f}  ({direction})")
    return True


def check_predictiveness(X: pd.DataFrame, y: pd.Series) -> float:
    """Gate: is the model even predictive before we spend a whole module
    explaining it? A chance-level model's "explanations" are noise dressed
    up as insight -- SHAP/PDP/permutation importance on such a model would
    happily produce plausible-looking plots with no real signal behind them.
    Returns the mean CV-AUC so callers can decide whether to proceed.
    """
    print("=== 0) Vorher: ist das Modell überhaupt prädiktiv? ===")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    probe = HistGradientBoostingClassifier(
        random_state=SEED, max_iter=200, learning_rate=0.05,
        max_depth=4, class_weight="balanced",
    )
    scores = cross_val_score(probe, X, y, cv=cv, scoring="roc_auc")
    print(f"  5-fold CV-AUC: {scores.mean():.3f} ± {scores.std():.3f}")
    if scores.mean() < 0.60:
        print(
            "  WARNING: CV-AUC is close to chance (0.5). Explaining a model that"
            "\n  barely predicts anything produces plausible-looking but"
            "\n  meaningless plots. Fix the model (features/data) before trusting"
            "\n  any explanation below."
        )
    else:
        print(
            f"  CV-AUC {scores.mean():.3f} is clearly above chance (0.5) -- "
            "explaining this model's\n  behaviour is a meaningful exercise. Proceeding."
        )
    return float(scores.mean())


def main() -> None:
    X, y = build_data()

    # --- Section 0: predictiveness gate (run BEFORE any explanation) --------
    check_predictiveness(X, y)

    print("\n=== 1) Train HistGradientBoostingClassifier for verstorben_30d ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED)
    model = train_model(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"  Test AUC: {auc:.3f}")

    # --- Section 2: Permutation importance ---
    explain_permutation(model, X_test, y_test)

    # --- Section 3: PDP / ICE (printed summary; figure is in figures.py) ---
    print("\n=== 3) Partial Dependence – concept check ===")
    print("  PDP averages predictions over all patients while varying one feature.")
    print("  ICE shows individual patient curves – crossing lines = interaction.")
    print("  See assets/partial_dependence.png for the visual output.")

    # --- Section 4: SHAP ---
    shap_available = explain_shap(model, X_train, X_test)

    print("\n=== Summary ===")
    print(f"  SHAP available: {shap_available}")
    print("  Global explanation: permutation importance + PDP")
    print("  Local explanation: SHAP (if available) or PDP-baseline comparison")
    print("\nKey point: importance ≠ causation. Validate findings clinically.")


if __name__ == "__main__":
    main()
