"""Module 24 – Prediction workflow, data leakage, tuning and thresholds.

Runs standalone from the project root:
    python module/24-praediktion-workflow/code/python.py

Data: read from data/ (committed with the repo); if that folder is
missing, the same files are fetched from the published URL.
Only scikit-learn is required.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.feature_selection import SelectKBest, f_classif  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import (  # noqa: E402
    GridSearchCV, StratifiedKFold, cross_val_predict, cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve  # noqa: E402

from lib.helpers import SEED, load_cohort, load_labs  # noqa: E402

NUMERIC = ["alter", "sofa_score", "crp_mg_l", "bmi", "leukozyten_g_l", "kreatinin_mg_dl", "laktat_mmol_l"]
CATEGORICAL = ["aufnahmegrund", "raucherstatus"]
BINARY = ["diabetes", "hypertonie"]
TARGET = "verstorben_30d"


def build_data():
    df = load_cohort().merge(load_labs(), on="patient_id", how="left")
    X = df[NUMERIC + CATEGORICAL + BINARY]
    y = df[TARGET]
    return X, y


def build_pipeline(C: float = 1.0) -> Pipeline:
    """Imputation, scaling and encoding live INSIDE the pipeline, so they are
    re-fitted on each training fold only – this is what prevents leakage."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = OneHotEncoder(handle_unknown="ignore")
    pre = ColumnTransformer([
        ("num", numeric, NUMERIC),
        ("cat", categorical, CATEGORICAL),
        ("bin", "passthrough", BINARY),
    ])
    return Pipeline([
        ("pre", pre),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced", C=C)),
    ])


def main() -> None:
    X, y = build_data()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    print("=== 1) The leakage trap: feature selection OUTSIDE vs INSIDE the CV ===")
    # Demonstration: add 300 pure-noise features, then keep the 12 "best" ones.
    # If selection sees the whole dataset, noise that happens to correlate with
    # the outcome leaks in -> the score looks great but is fiction.
    rng = np.random.default_rng(SEED)
    # Keep the raw numeric columns (with their missing values) and append 300
    # noise columns. Imputation and scaling stay OUT of this matrix so that in
    # the honest branch they can be re-fitted per fold inside the pipeline.
    X_num_raw = X[NUMERIC].to_numpy(dtype=float)
    noise = rng.normal(size=(len(y), 300))
    X_aug = np.hstack([X_num_raw, noise])

    # WRONG: impute, scale and select using ALL rows, then cross-validate.
    # Every preprocessing step here has already seen the validation rows.
    X_all = SimpleImputer(strategy="median").fit_transform(X_aug)
    X_all = StandardScaler().fit_transform(X_all)
    selected = SelectKBest(f_classif, k=12).fit_transform(X_all, y)
    leaky_auc = cross_val_score(LogisticRegression(max_iter=1000), selected, y,
                                cv=cv, scoring="roc_auc").mean()
    # CORRECT: imputation, scaling AND selection all sit in the pipeline, so
    # they are re-fit on each training fold only – nothing sees the fold's
    # validation rows in advance.
    honest_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                            ("scale", StandardScaler()),
                            ("select", SelectKBest(f_classif, k=12)),
                            ("model", LogisticRegression(max_iter=1000))])
    honest_auc = cross_val_score(honest_pipe, X_aug, y, cv=cv, scoring="roc_auc").mean()
    print(f"  leaked  CV-AUC: {leaky_auc:.3f}   (too optimistic – selection saw the test rows)")
    print(f"  honest  CV-AUC: {honest_auc:.3f}   (trustworthy)")

    print("\n=== 2) Train/test split + hyperparameter tuning ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED)
    grid = GridSearchCV(build_pipeline(), {"model__C": [0.01, 0.1, 1, 10]},
                        cv=cv, scoring="roc_auc")
    grid.fit(X_train, y_train)
    print(f"  best C: {grid.best_params_['model__C']}   inner CV-AUC: {grid.best_score_:.3f}")

    # The test set is touched exactly ONCE, at the very end.
    test_auc = roc_auc_score(y_test, grid.predict_proba(X_test)[:, 1])
    print(f"  held-out test AUC: {test_auc:.3f}")

    print("\n=== 3) Choosing a decision threshold (not just 0.5) ===")
    # Choose the threshold on the TRAINING data (out-of-fold predictions) and
    # apply it once to the test set. Picking it on the test set would be one
    # more way of optimising on the final yardstick.
    oof = cross_val_predict(grid.best_estimator_, X_train, y_train, cv=cv,
                            method="predict_proba")[:, 1]
    fpr, tpr, thr = roc_curve(y_train, oof)
    youden_thr = float(thr[int(np.argmax(tpr - fpr))])
    proba = grid.predict_proba(X_test)[:, 1]

    def sens_spec(threshold: float) -> tuple[float, float]:
        tn, fp, fn, tp = confusion_matrix(y_test, (proba >= threshold).astype(int)).ravel()
        return tp / (tp + fn), tn / (tn + fp)

    s05, sp05 = sens_spec(0.5)
    sY, spY = sens_spec(youden_thr)
    print(f"  threshold 0.50          -> sensitivity {s05:.2f}, specificity {sp05:.2f}")
    print(f"  threshold {youden_thr:.2f} (Youden) -> sensitivity {sY:.2f}, specificity {spY:.2f}"
          "   (threshold chosen on training folds, evaluated on test)")
    print("\nThe threshold is a clinical decision, not a statistical default.")


if __name__ == "__main__":
    main()
