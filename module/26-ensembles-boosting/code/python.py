"""Module 26 — Tree ensembles and gradient boosting.

Runs standalone from the project root:
    python module/26-ensembles-boosting/code/python.py

Data: read from data/ (committed with the repo); if that folder is
missing, the same files are fetched from the published URL.
Requires scikit-learn. xgboost / lightgbm are used if available (install with
`uv pip install xgboost lightgbm`; on macOS you also need the OpenMP runtime,
`brew install libomp`, and must run Python with
`DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib` so the two libraries can find
it); otherwise HistGradientBoostingClassifier is used as a fallback and this
script still runs and teaches the same concepts.

IMPORTANT — leakage: imputation and one-hot encoding are NOT fit on the full
dataset up front (that would be the exact leakage Module 24 warns against).
Both live inside a Pipeline/ColumnTransformer that cross_val_score() re-fits
on the training fold only, every fold, every model.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import warnings  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from lib.helpers import SEED, load_cohort, load_labs  # noqa: E402

NUMERIC = ["alter", "sofa_score", "crp_mg_l", "bmi", "leukozyten_g_l", "kreatinin_mg_dl", "laktat_mmol_l"]
CATEGORICAL = ["aufnahmegrund", "raucherstatus"]
BINARY = ["diabetes", "hypertonie"]
TARGET = "verstorben_30d"


def load_data():
    """Return the RAW feature frame (imputation/encoding not yet applied) and y."""
    df = load_cohort().merge(load_labs(), on="patient_id", how="left")
    X = df[NUMERIC + CATEGORICAL + BINARY]
    y = df[TARGET]
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Impute + one-hot-encode. Wrapped in a Pipeline below so this is only
    ever fit on a training fold, never on the full dataset (no leakage)."""
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("bin", "passthrough", BINARY),
    ])


def build_pipeline(model) -> Pipeline:
    """Preprocessing + model as ONE pipeline, so cross_val_score()/train_test_split
    based fitting always re-estimates the imputer/encoder per fold, not once
    on everything up front."""
    return Pipeline([("pre", build_preprocessor()), ("model", model)])


def build_native_missing_pipeline(model) -> Pipeline:
    """For HistGradientBoosting: keep NaN (it handles missing values natively,
    no imputer needed) but still one-hot-encode the categoricals inside the
    pipeline (fit per fold). Numeric/binary columns pass through untouched."""
    pre = ColumnTransformer([
        ("num", "passthrough", NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("bin", "passthrough", BINARY),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def load_boosters(y):
    """Return dict of optional boosting models (xgboost, lightgbm)."""
    models = {}
    pos_weight = float((y == 0).sum()) / float((y == 1).sum())

    # --- XGBoost -------------------------------------------------------------
    try:
        import xgboost as xgb  # noqa: F401
    except Exception as exc:
        print(f"  XGBoost übersprungen: {exc.__class__.__name__}.")
        xgb = None

    if xgb is not None:
        try:
            models["XGBoost"] = xgb.XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                scale_pos_weight=pos_weight,
                eval_metric="auc", random_state=SEED, verbosity=0,
                use_label_encoder=False,
            )
        except TypeError:
            # Older XGBoost versions don't have use_label_encoder param
            models["XGBoost"] = xgb.XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                scale_pos_weight=pos_weight,
                eval_metric="auc", random_state=SEED, verbosity=0,
            )

    # --- LightGBM ------------------------------------------------------------
    try:
        import lightgbm as lgb  # noqa: F401
    except Exception as exc:
        print(f"  LightGBM übersprungen: {exc.__class__.__name__}.")
        lgb = None

    if lgb is not None:
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            is_unbalance=True, random_state=SEED, verbose=-1,
        )

    return models


def main() -> None:
    X, y = load_data()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    n_events = int(y.sum())
    n_predictors = len(NUMERIC) + len(CATEGORICAL) + len(BINARY)
    print(f"Cohort: {len(y)} patients, {n_events} events ({y.mean():.1%}), "
          f"{n_predictors} raw predictors -> EPV (events per variable) "
          f"~{n_events / n_predictors:.1f}.")
    print(
        "EPV ~10 is the traditional regression rule of thumb; tree ensembles"
        "\ncan fit far more effective parameters than that from the same data,"
        "\nso with ~500 rows, regularization (depth, learning rate, n_estimators,"
        "\nclass_weight) matters even more than usual -- see sections 1 and 3."
    )

    # --- 1) Decision tree: depth vs. overfitting -----------------------------
    # Split RAW rows first, THEN fit the imputer/encoder on the training rows
    # only -- exactly the leakage-safe order Module 24 teaches.
    print("\n=== 1) Entscheidungsbaum: Tiefe vs. Überanpassung ===")
    print(f"  {'Tiefe':>6}  {'Train-AUC':>10}  {'Val-AUC':>10}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED
    )
    pre = build_preprocessor()
    X_train_imp = pre.fit_transform(X_train)   # fit on TRAIN only
    X_test_imp  = pre.transform(X_test)        # TEST only ever transformed
    for depth in [1, 2, 3, 4, 5, 6, 8, 10, None]:
        tree = DecisionTreeClassifier(max_depth=depth, random_state=SEED)
        tree.fit(X_train_imp, y_train)
        tr_auc  = roc_auc_score(y_train, tree.predict_proba(X_train_imp)[:, 1])
        val_auc = roc_auc_score(y_test,  tree.predict_proba(X_test_imp)[:, 1])
        depth_label = str(depth) if depth is not None else "None"
        print(f"  {depth_label:>6}  {tr_auc:>10.3f}  {val_auc:>10.3f}")

    # --- 2) Model comparison via CV ------------------------------------------
    # Every model below is wrapped in a Pipeline, so cross_val_score() re-fits
    # imputation/encoding on each training fold only -- no leakage, whether
    # the model needs pre-imputed input (tree/RF/boosters) or handles missing
    # values natively (HistGradientBoosting).
    print("\n=== 2) Kreuzvalidierter AUC-Vergleich (alle Modelle: Pipeline in CV) ===")

    tree_cv = DecisionTreeClassifier(max_depth=4, random_state=SEED)
    rf      = RandomForestClassifier(
        n_estimators=300, max_features="sqrt",
        class_weight="balanced", random_state=SEED, n_jobs=-1,
    )
    hgb     = HistGradientBoostingClassifier(
        max_iter=300, max_depth=5, learning_rate=0.05,
        class_weight="balanced", random_state=SEED,
    )

    models_fixed = {
        "Entscheidungsbaum (Tiefe 4)": build_pipeline(tree_cv),
        "Random Forest":               build_pipeline(rf),
        "HistGradBoost (sklearn)":     build_native_missing_pipeline(hgb),
    }

    results = {}
    for name, pipe in models_fixed.items():
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
        results[name] = scores
        print(f"  {name:<32} AUC = {scores.mean():.3f} ± {scores.std():.3f}")

    # Optional boosters (imputed/encoded input, same leakage-safe pipeline)
    boosters = load_boosters(y)
    if boosters:
        print("\n  --- Optionale Boosting-Bibliotheken ---")
    else:
        print(
            "\n  xgboost / lightgbm nicht installiert -- sklearn HistGradBoost als"
            "\n  Ersatz. Installieren mit: uv pip install xgboost lightgbm"
            "\n  (macOS: zusätzlich `brew install libomp`, dann Python mit"
            "\n  DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib starten)."
        )

    for name, model in boosters.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(build_pipeline(model), X, y, cv=cv, scoring="roc_auc")
        results[name] = scores
        print(f"  {name:<32} AUC = {scores.mean():.3f} ± {scores.std():.3f}")

    # --- 3) Regularisation intuition -----------------------------------------
    print("\n=== 3) Regularisierung: Lernrate × Iterationen ===")
    configs = [(0.30, 50), (0.10, 150), (0.05, 300), (0.01, 1500)]
    print(f"  {'lr':>5}  {'n_iter':>7}  {'CV-AUC':>8}")
    for lr, n_iter in configs:
        m = HistGradientBoostingClassifier(
            max_iter=n_iter, learning_rate=lr,
            class_weight="balanced", random_state=SEED,
        )
        auc = cross_val_score(build_native_missing_pipeline(m), X, y, cv=cv, scoring="roc_auc").mean()
        print(f"  {lr:>5.2f}  {n_iter:>7}  {auc:>8.3f}")

    # --- Honest, DATA-DRIVEN conclusion (never hardcode the winner) ---------
    tree_mean = results["Entscheidungsbaum (Tiefe 4)"].mean()
    others = {k: v.mean() for k, v in results.items() if k != "Entscheidungsbaum (Tiefe 4)"}
    best_name = max(others, key=others.get)
    best_mean = others[best_name]
    gap = best_mean - tree_mean
    print(f"\n=== Fazit (aus den obigen Zahlen abgeleitet, nicht vorab festgelegt) ===")
    if gap > 0.02:
        print(
            f"Bestes Ensemble/Boosting-Modell ({best_name}, AUC={best_mean:.3f}) übertrifft"
            f"\nden einzelnen Baum (AUC={tree_mean:.3f}) um {gap:.3f} -- auf dieser Kohorte"
            "\nlohnt sich der Mehraufwand."
        )
    elif gap > -0.02:
        print(
            f"Bestes Ensemble/Boosting-Modell ({best_name}, AUC={best_mean:.3f}) liegt in"
            f"\netwa gleichauf mit dem einzelnen Baum (AUC={tree_mean:.3f}, Differenz"
            f" {gap:+.3f})."
            "\nBei nur ~500 Zeilen und wenigen Prädiktoren ist das ein plausibles,"
            "\nehrliches Ergebnis -- mehr Modellkomplexität zahlt sich erst bei mehr"
            "\nDaten/stärkeren Nichtlinearitäten sicher aus. Vergleiche IMMER die"
            "\ntatsächlichen Zahlen, nie eine angenommene Rangfolge."
        )
    else:
        print(
            f"Der einzelne Baum (AUC={tree_mean:.3f}) schneidet hier sogar besser ab als"
            f"\n{best_name} (AUC={best_mean:.3f}). Das kann bei kleinen, wenig nichtlinearen"
            "\nDatensätzen passieren -- ein weiterer Grund, nie eine Modellklasse"
            "\nblind zu bevorzugen, sondern immer zu messen."
        )


if __name__ == "__main__":
    main()
