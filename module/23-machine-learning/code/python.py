"""Module 23 – Introduction to machine learning: first classifier.

Runs standalone from the project root:
    python module/23-machine-learning/code/python.py

Data: read from data/ (committed with the repo); if that folder is
missing, the same files are fetched from the published URL.
Only scikit-learn is required.

Code is English (identifiers, comments, docstrings).
Dataset column names stay German (e.g. df["aufnahmegrund"]).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import (  # noqa: E402
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from lib.helpers import SEED, load_cohort  # noqa: E402

pd.set_option("display.width", 100)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Derive predictors and target from the raw cohort table.

    Binary indicator columns use English identifiers; the source columns
    stay German (aufnahmegrund, raucherstatus) because that is the dataset schema.
    """
    df = df.copy()
    df["is_sepsis"] = (df["aufnahmegrund"] == "Sepsis").astype(int)
    df["is_smoker"] = (df["raucherstatus"] == "aktiv").astype(int)

    feature_cols = ["alter", "sofa_score", "crp_mg_l", "diabetes",
                    "is_sepsis", "is_smoker"]
    X = df[feature_cols]
    y = df["verstorben_30d"]
    return X, y


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(pipeline: Pipeline,
                   X_test: pd.DataFrame,
                   y_test: pd.Series) -> None:
    """Print AUC, classification report and confusion matrix on the held-out test set."""
    print("\n" + "=" * 60)
    print("EVALUATION ON THE TEST SET")
    print("=" * 60)

    probs = pipeline.predict_proba(X_test)[:, 1]   # P(verstorben = 1)
    preds = pipeline.predict(X_test)
    n_events = int(y_test.sum())

    auc = roc_auc_score(y_test, probs)
    print(f"\nROC-AUC (test): {auc:.3f}")
    print(
        f"\nNote: with only ~{n_events} events in the test set ({y_test.mean():.0%} of "
        f"{len(y_test)}) the 95 % CI of the AUC is wide. One or two misclassified"
        "\ncases shift AUC by 0.05-0.10. Draw no firm conclusions from a single split."
    )

    print("\nClassification report (threshold = 0.5):")
    print(classification_report(y_test, preds, target_names=["lebt", "verstorben"]))

    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    print("Confusion matrix:")
    print(f"  TN={tn:3d}  FP={fp:3d}")
    print(f"  FN={fn:3d}  TP={tp:3d}")

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    print(f"\nSensitivity (recall on deaths): {sensitivity:.2f}   Specificity: {specificity:.2f}")
    if fn > tp:
        print(
            "\nAt threshold 0.5 more positive cases are missed than caught (FN > TP)."
            "\nclass_weight='balanced' pulls the decision boundary toward the"
            "\nminority class but does not guarantee FN <= TP at threshold 0.5."
        )
    else:
        print(
            "\nAt threshold 0.5, class_weight='balanced' has shifted the decision"
            f"\nboundary enough that TP ({tp}) >= FN ({fn}): sensitivity is high"
            f" ({sensitivity:.0%}) at the cost of specificity ({specificity:.0%})."
            "\nThat trade-off is a deliberate effect of class_weight, not a fixed law -"
            "\nthe threshold must still be chosen explicitly for the clinical setting."
        )


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def check_calibration(pipeline: Pipeline,
                      X_test: pd.DataFrame,
                      y_test: pd.Series) -> None:
    """Print a text calibration table: predicted probability vs. observed rate.

    Uses n_bins=5 quantile bins to avoid empty bins on a small test set.
    """
    print("\n" + "=" * 60)
    print("CALIBRATION CHECK")
    print("=" * 60)
    print(
        "Calibration: do predicted probabilities match observed event rates?"
        "\nIdeal: each predicted probability bin lies near the diagonal."
    )

    probs = pipeline.predict_proba(X_test)[:, 1]
    n_bins = 5

    print(
        f"\nMean predicted mortality: {probs.mean():.1%}   "
        f"Observed mortality: {y_test.mean():.1%}"
    )
    print(
        "\nSTOLPERSTEIN: this gap is not (only) small-sample noise. This pipeline"
        "\nwas fitted with class_weight='balanced', which reweights the loss as if"
        "\ndeaths were as common as survivals. That shifts the default 0.5"
        "\nthreshold towards sensitivity but means predict_proba() systematically"
        "\noverstates absolute risk. AUC/ranking is almost unaffected (the weights"
        "\nmainly shift the intercept, a near-monotonic rescaling) – but any"
        "\nabsolute probability (Brier score, calibration curve, decision-curve"
        "\nnet benefit) IS affected. Module 25 recalibrates with"
        "\nCalibratedClassifierCV before computing any of those; see this module's"
        "\nassets/kalibrierung.png for the before/after comparison."
    )

    try:
        frac_pos, mean_probs = calibration_curve(
            y_test, probs, n_bins=n_bins, strategy="quantile"
        )
        print(f"\nCalibration curve ({n_bins} quantile bins):")
        print(f"{'Pred. prob.':>14} {'Obs. rate':>11}")
        for mp, fp in zip(mean_probs, frac_pos):
            bar = "#" * int(fp * 30)
            print(f"  {mp:6.3f}  →  {fp:6.3f}  {bar}")
        print(
            f"\nCAVEAT: with only ~{int(y_test.sum())} events in the test set,"
            "\nbin estimates are ALSO very uncertain on top of the class_weight"
            "\neffect above. Reliable calibration assessment requires external"
            "\nvalidation data with many more events."
        )
    except ValueError as exc:
        print(f"Calibration curve could not be computed: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """End-to-end ML pipeline: split, train, cross-validate, calibrate."""
    print("Module 23 – Machine Learning: 30-day mortality classifier")
    print("=" * 60)

    df = load_cohort()
    X, y = prepare_features(df)

    print(f"\nCohort: {len(df)} patients")
    print(f"Features: {list(X.columns)}")
    print(f"Target: verstorben_30d  (base rate: {y.mean():.1%})")
    print(
        f"\nAccuracy paradox: a model that always predicts 'lebt' achieves"
        f" {1 - y.mean():.1%} accuracy – and misses every death."
        f"\n→ Use AUC, not accuracy, as the primary metric for imbalanced outcomes."
    )

    # --- Train/test split (stratified) ---
    print("\n" + "=" * 60)
    print("TRAIN / TEST SPLIT  (80 / 20, stratified, seed=42)")
    print("=" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    print(f"Training : {len(X_train)} patients  |  mortality: {y_train.mean():.1%}")
    print(f"Test     : {len(X_test)} patients  |  mortality: {y_test.mean():.1%}")

    # --- Pipeline: StandardScaler + LogisticRegression ---
    print("\n" + "=" * 60)
    print("MODEL  (Pipeline: StandardScaler + LogisticRegression)")
    print("=" * 60)
    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",   # upweight the minority class (~16 %)
            random_state=SEED,
        )),
    ])
    pipeline.fit(X_train, y_train)
    print("Pipeline fitted successfully.")
    print(
        "Logistic regression: transparent, interpretable, and – WITHOUT class"
        "\nweighting – calibrated by default. class_weight='balanced' prevents the"
        "\nmodel from ignoring deaths at the 0.5 threshold, but it reweights the"
        "\nloss function, so predict_proba() no longer reflects the true event rate"
        "\n(see the calibration check below). The scaler is fitted on training data"
        "\nonly – no leakage into the test set."
    )

    # --- Evaluation on test set ---
    evaluate_model(pipeline, X_test, y_test)

    # --- Cross-validation (on training data only) ---
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION  (StratifiedKFold, 5 folds)")
    print("=" * 60)
    print(
        "CV uses all training data alternately for fitting and validation."
        "\nThe test set is never touched during CV – it remains a clean holdout."
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    auc_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"\nAUC per fold : {np.round(auc_scores, 3)}")
    print(f"Mean CV-AUC  : {auc_scores.mean():.3f} ± {auc_scores.std():.3f}")

    # --- Calibration check ---
    check_calibration(pipeline, X_test, y_test)

    # --- Honest limits ---
    print("\n" + "=" * 60)
    print("HONEST LIMITS OF THIS MODEL")
    print("=" * 60)
    print(
        "1. Synthetic data: real cohorts have more noise, missing values, and"
        "\n   measurement errors than this dataset.\n"
        "2. No external validation: a model must be validated at another site"
        "\n   or time period before any clinical deployment.\n"
        "3. Calibration: class_weight='balanced' inflates predict_proba() (see"
        "\n   above), and the ~16 test events are too few for stable bins even"
        "\n   after recalibration.\n"
        "4. Fairness not checked: subgroup AUC can vary widely from global AUC.\n"
        "5. Association ≠ causation: the model finds statistical patterns –"
        "\n   some may reflect confounders, not causal pathways.\n"
        "\nThis script is a teaching example, not a clinical decision tool."
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
