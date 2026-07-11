"""Figures for module 27. Run: python module/27-erklaerbarkeit/code/figures.py

Writes PNGs to ../assets/. German labels (display), English code.
Requires: scikit-learn, matplotlib. Optional: shap (adds shap_beeswarm.png
if installed; skipped with a note otherwise).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.inspection import PartialDependenceDisplay, permutation_importance  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from lib.helpers import SEED, load_cohort, load_labs  # noqa: E402
from lib.plotstyle import EVENT, PRIMARY, SECONDARY, apply_style, save  # noqa: E402

ASSETS = Path(__file__).resolve().parent.parent / "assets"

NUMERIC = ["alter", "sofa_score", "crp_mg_l", "bmi", "leukozyten_g_l",
           "kreatinin_mg_dl", "laktat_mmol_l"]
CATEGORICAL = ["aufnahmegrund", "raucherstatus"]
BINARY = ["diabetes", "hypertonie"]
FEATURES = NUMERIC + CATEGORICAL + BINARY


def prepare_data():
    df = load_cohort().merge(load_labs(), on="patient_id", how="left")
    X = df[FEATURES].copy()
    for col in CATEGORICAL:
        X[col] = X[col].astype("category").cat.codes.astype(float)
    # Convert all integer columns to float to avoid sklearn PDP FutureWarning.
    for col in X.columns:
        if X[col].dtype in ("int64", "int32", "int8", "int16"):
            X[col] = X[col].astype(float)
    y = df["verstorben_30d"]
    return X, y


def fit_model(X_train, y_train):
    model = HistGradientBoostingClassifier(
        random_state=SEED, max_iter=200, learning_rate=0.05,
        max_depth=4, class_weight="balanced",
    )
    model.fit(X_train, y_train)
    return model


def fig_permutation_importance(model, X_test: pd.DataFrame,
                               y_test: pd.Series) -> None:
    """Horizontal bar chart of permutation importance with error bars."""
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=10, random_state=SEED, scoring="roc_auc", n_jobs=1,
    )
    means = result.importances_mean
    stds = result.importances_std

    # Sort ascending so the most important feature appears at the top.
    order = np.argsort(means)
    sorted_names = [FEATURES[i] for i in order]
    sorted_means = means[order]
    sorted_stds = stds[order]

    # German-friendly feature label mapping.
    label_map = {
        "alter": "Alter (Jahre)",
        "sofa_score": "SOFA-Score",
        "crp_mg_l": "CRP (mg/l)",
        "bmi": "BMI",
        "leukozyten_g_l": "Leukozyten (G/l)",
        "kreatinin_mg_dl": "Kreatinin (mg/dl)",
        "laktat_mmol_l": "Laktat (mmol/l)",
        "aufnahmegrund": "Aufnahmegrund",
        "raucherstatus": "Raucherstatus",
        "diabetes": "Diabetes",
        "hypertonie": "Hypertonie",
    }
    display_names = [label_map.get(n, n) for n in sorted_names]

    colors = [EVENT if m > 0.005 else SECONDARY for m in sorted_means]

    fig, ax = plt.subplots(figsize=(7, 5))
    y_pos = np.arange(len(sorted_names))
    ax.barh(y_pos, sorted_means, xerr=sorted_stds,
            color=colors, height=0.6, error_kw={"linewidth": 0.8, "capsize": 3})
    ax.axvline(0, color="#CCCCCC", lw=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names, fontsize=10)
    ax.set_xlabel("Mittlerer AUC-Abfall (Permutation Importance)")
    ax.set_title("Permutation Importance auf dem Testset\n(positiv = Modell verlässt sich darauf)")
    ax.grid(axis="x")
    save(fig, ASSETS / "permutation_importance.png")


def fig_partial_dependence(model, X_test: pd.DataFrame) -> None:
    """PDP + ICE for sofa_score and alter in a two-panel figure."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # PartialDependenceDisplay uses feature names if X is a DataFrame.
    # NOTE: use "linewidth", not the "lw" alias -- newer matplotlib raises
    # "Got both 'linewidth' and 'lw'" if PartialDependenceDisplay's internal
    # defaults and our kwargs use different aliases for the same property.
    display = PartialDependenceDisplay.from_estimator(
        model, X_test,
        features=["sofa_score", "alter"],
        kind="both",       # "average" = PDP only; "both" adds ICE lines
        subsample=80,
        random_state=SEED,
        ax=axes,
        line_kw={"color": PRIMARY, "linewidth": 2.0, "label": "PDP (Durchschnitt)"},
        ice_lines_kw={"color": SECONDARY, "alpha": 0.15, "linewidth": 0.6},
    )

    titles = ["SOFA-Score", "Alter (Jahre)"]
    for i, ax in enumerate(axes):
        ax.set_title(f"PDP & ICE: {titles[i]}")
        ax.set_ylabel("Vorhergesagtes Sterberisiko (Wahrsch.)")
        ax.set_xlabel(titles[i])

    fig.suptitle("Partial Dependence & ICE — globaler vs. individueller Effekt",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, ASSETS / "partial_dependence.png")


LABEL_MAP = {
    "alter": "Alter (Jahre)",
    "sofa_score": "SOFA-Score",
    "crp_mg_l": "CRP (mg/l)",
    "bmi": "BMI",
    "leukozyten_g_l": "Leukozyten (G/l)",
    "kreatinin_mg_dl": "Kreatinin (mg/dl)",
    "laktat_mmol_l": "Laktat (mmol/l)",
    "aufnahmegrund": "Aufnahmegrund (codiert)",
    "raucherstatus": "Raucherstatus (codiert)",
    "diabetes": "Diabetes",
    "hypertonie": "Hypertonie",
}


def fig_shap_beeswarm(model, X_train: pd.DataFrame, X_test: pd.DataFrame) -> bool:
    """SHAP beeswarm summary plot. Returns True if shap was available and the
    figure was generated, False if shap is not installed (caller can skip the
    README reference / print a note instead of failing)."""
    try:
        import shap
    except ImportError:
        print("  shap not installed -- skipping shap_beeswarm.png "
              "(install: uv pip install shap)")
        return False

    explainer = shap.Explainer(model, X_train)
    # check_additivity=False: see code/python.py for why this is needed with
    # HistGradientBoostingClassifier + class_weight="balanced".
    shap_values = explainer(X_test.iloc[:150], check_additivity=False)
    shap_values.feature_names = [LABEL_MAP.get(f, f) for f in FEATURES]

    fig = plt.figure(figsize=(7.5, 5.5))
    # beeswarm jitters overlapping points vertically and draws that jitter from
    # numpy's GLOBAL RNG, which nothing else in this file seeds. Without this the
    # figure differs on every run and the committed PNG is never reproducible.
    np.random.seed(SEED)
    shap.plots.beeswarm(shap_values, show=False, max_display=11)
    ax = plt.gca()
    ax.set_title("SHAP-Werte: Beitrag jedes Merkmals zur Vorhersage\n"
                 "(150 zufällige Testpatient:innen, HistGradientBoosting)",
                 loc="left", fontweight="bold")
    ax.set_xlabel("SHAP-Wert (Beitrag zum log-odds-Score, negativ = senkt Risiko)")
    fig = plt.gcf()
    save(fig, ASSETS / "shap_beeswarm.png")
    return True


def main() -> None:
    apply_style()
    X, y = prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED)
    model = fit_model(X_train, y_train)

    fig_permutation_importance(model, X_test, y_test)
    fig_partial_dependence(model, X_test)
    fig_shap_beeswarm(model, X_train, X_test)


if __name__ == "__main__":
    main()
