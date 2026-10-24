"""Module 30 – Deep learning intro: MLP on tabular clinical data.

Runs standalone from the project root:
    python module/30-deep-learning/code/python.py

Data: read from data/ (committed with the repo); if that folder is
missing, the same files are fetched from the published URL.
Only scikit-learn is required (torch is NOT imported – CNN code in README is
prose-only and clearly marked as illustrative).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import warnings  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import (  # noqa: E402
    StratifiedKFold, cross_val_score, train_test_split,
)
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

# On small imbalanced data the MLP optimiser may not fully converge within
# max_iter; suppress the warning – this is expected behaviour and part of
# the teaching point (MLPs need more data than we have here).
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from lib.helpers import SEED, load_cohort, load_labs  # noqa: E402

NUMERIC = [
    "alter", "sofa_score", "crp_mg_l", "bmi",
    "leukozyten_g_l", "kreatinin_mg_dl", "laktat_mmol_l",
]
CATEGORICAL = ["aufnahmegrund", "raucherstatus"]
BINARY = ["diabetes", "hypertonie"]
TARGET = "verstorben_30d"


def build_data():
    """Merge cohort + labs and return X, y."""
    df = load_cohort().merge(load_labs(), on="patient_id", how="left")
    X = df[NUMERIC + CATEGORICAL + BINARY]
    y = df[TARGET]
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Shared imputation + scaling + encoding for all models."""
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("bin", "passthrough", BINARY),
    ])


def make_lr(pre: ColumnTransformer) -> Pipeline:
    return Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                   random_state=SEED)),
    ])


def make_gb(pre: ColumnTransformer) -> Pipeline:
    return Pipeline([
        ("pre", pre),
        ("clf", GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                            random_state=SEED)),
    ])


def make_mlp_es(pre: ColumnTransformer, alpha: float = 0.01) -> Pipeline:
    """MLP with early_stopping for learning-curve demonstration (section 2).

    Not used in CV: early_stopping holds back 15% of training rows per fold,
    which starves the model on our 375-row training set.
    """
    return Pipeline([
        ("pre", pre),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=alpha,
            early_stopping=True,
            validation_fraction=0.15,
            max_iter=300,
            random_state=SEED,
        )),
    ])


def make_mlp(pre: ColumnTransformer, alpha: float = 0.01) -> Pipeline:
    """MLP with fixed iterations – used for CV comparisons.

    early_stopping is disabled so the full fold training set is used.
    max_iter=500 prevents ConvergenceWarnings on this small dataset.
    """
    return Pipeline([
        ("pre", pre),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=alpha,
            early_stopping=False,
            max_iter=500,
            random_state=SEED,
        )),
    ])


def summarise_folds(scores: np.ndarray) -> dict:
    """Mean, sample SD and two approximate 95% t-intervals for per-fold scores.

    The naive interval treats the k fold scores as independent. They are not:
    the training sets overlap, so the naive interval is too narrow, and no
    unbiased estimator of the k-fold variance exists (Bengio & Grandvalet
    2004). The corrected interval inflates the variance by the Nadeau-Bengio
    factor (1/k + n_test/n_train, with n_test/n_train = 1/(k-1) for k-fold).
    Both are descriptive approximations, not exact confidence intervals.
    """
    k = len(scores)
    mean = scores.mean()
    sd = scores.std(ddof=1)
    t_crit = stats.t.ppf(0.975, df=k - 1)
    half_naive = t_crit * sd * np.sqrt(1 / k)
    half_corrected = t_crit * sd * np.sqrt(1 / k + 1 / (k - 1))
    return {"k": k, "mean": mean, "sd": sd,
            "ci_low": mean - half_naive, "ci_high": mean + half_naive,
            "nb_low": mean - half_corrected, "nb_high": mean + half_corrected}


def main() -> None:
    X, y = build_data()
    pre = build_preprocessor()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # ── 1) Train/test split ────────────────────────────────────────────────
    print("=== 1) Train/test split (75/25, stratified) ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED)
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}")
    print(f"  Event rate train: {y_train.mean():.1%}  test: {y_test.mean():.1%}")

    # ── 2) MLP: train and read learning curve ──────────────────────────────
    print("\n=== 2) MLP training + learning curve ===")
    # Use early_stopping variant here so we can read loss_curve_ and
    # validation_scores_. This is a single train/test context, not CV,
    # so the 15% validation split is not a problem.
    mlp_pipe = make_mlp_es(build_preprocessor())
    mlp_pipe.fit(X_train, y_train)
    mlp = mlp_pipe.named_steps["mlp"]
    print(f"  Stopped after {mlp.n_iter_} epochs")
    print(f"  Best validation score (accuracy): {mlp.best_validation_score_:.3f}")
    print(f"  Final training loss: {mlp.loss_curve_[-1]:.4f}")
    test_auc_mlp = roc_auc_score(y_test, mlp_pipe.predict_proba(X_test)[:, 1])
    print(f"  Test AUC: {test_auc_mlp:.3f}")

    # ── 3) AUC comparison: LR vs GB vs MLP ────────────────────────────────
    # A single point estimate per model hides the fold-to-fold noise. With
    # only 5 folds on ~500 patients, that noise can be as large as the gap
    # between models – so we report per-fold scores, mean +/- SD, and a
    # 95% CI for every model, AND the *paired* per-fold difference between
    # models (paired is the right comparison here because every model sees
    # the exact same 5 folds, so fold-to-fold difficulty cancels out).
    print("\n=== 3) AUC comparison via 5-fold CV (per-fold, not just the mean) ===")
    models = {
        "Logistische Regression": make_lr(build_preprocessor()),
        "Gradient Boosting":      make_gb(build_preprocessor()),
        "MLP":                    make_mlp(build_preprocessor()),
    }
    fold_scores = {}
    for name, pipe in models.items():
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
        fold_scores[name] = scores
        summary = summarise_folds(scores)
        fold_str = ", ".join(f"{s:.3f}" for s in scores)
        print(f"  {name}")
        print(f"    Folds (k={summary['k']}): {fold_str}")
        print(f"    Mean = {summary['mean']:.3f}  SD = {summary['sd']:.3f}  "
              f"naive 95%-CI = [{summary['ci_low']:.3f}, {summary['ci_high']:.3f}]  "
              f"corrected = [{summary['nb_low']:.3f}, {summary['nb_high']:.3f}]")

    # Never hardcode the winner: derive it from the actually computed means.
    means = {name: scores.mean() for name, scores in fold_scores.items()}
    order = sorted(means.items(), key=lambda t: -t[1])
    winner, winner_auc = order[0]
    print(f"\n  Highest mean CV-AUC (point estimate only): {winner} ({winner_auc:.3f})")

    # Paired per-fold differences vs. the winner – this is the honest test
    # of whether the ranking is actually established, or within noise.
    print("\n  Paired per-fold differences vs. "
          f"{winner} (same 5 folds for every model):")
    comparisons = []
    for name, _ in order[1:]:
        diff = fold_scores[winner] - fold_scores[name]
        summary = summarise_folds(diff)
        comparisons.append((name, summary))
        diff_str = ", ".join(f"{d:+.3f}" for d in diff)
        print(f"    {winner} − {name}: {diff_str}")
        print(f"      Mean diff = {summary['mean']:+.3f}  SD = {summary['sd']:.3f}  "
              f"naive 95%-CI = [{summary['ci_low']:+.3f}, {summary['ci_high']:+.3f}]  "
              f"corrected = [{summary['nb_low']:+.3f}, {summary['nb_high']:+.3f}]")

    # Rewrite the conclusion from the corrected intervals – do not assume a
    # fixed narrative. The naive interval assumes independent folds and is
    # only printed for comparison.
    print("\n  Befund (aus den korrigierten Intervallen, nicht aus den Punktschätzern):")
    for name, summary in comparisons:
        corrected_excludes_zero = summary["nb_low"] > 0 or summary["nb_high"] < 0
        naive_excludes_zero = summary["ci_low"] > 0 or summary["ci_high"] < 0
        if corrected_excludes_zero:
            print(f"    {winner} vor {name}: auch das korrigierte Intervall schließt 0 aus –")
            print("    ein deutlicher Hinweis auf einen echten Unterschied, kein exakter Beweis.")
        elif naive_excludes_zero:
            print(f"    {winner} vs. {name}: nur das naive Intervall schließt 0 aus. Weil die")
            print("    Folds korreliert sind, ist das kein belastbarer Beleg – nicht unterscheidbar.")
        else:
            print(f"    {winner} und {name} sind bei k=5 Folds NICHT unterscheidbar.")
    print("  MLPs need large datasets to justify their parameter count; on")
    print("  small clinical tables, simpler models are often at least on par.")

    # ── 4) Regularisation sweep ────────────────────────────────────────────
    print("\n=== 4) Regularisation: alpha sweep ===")
    for alpha in [0.0001, 0.001, 0.01, 0.1, 1.0]:
        pipe = make_mlp(build_preprocessor(), alpha=alpha)
        auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean()
        print(f"  alpha={alpha:<7}  CV-AUC={auc:.3f}")

    print(f"\nDone. Honest finding on this {len(X)}-patient dataset ({y.sum()} events):")
    print(f"  {winner} had the highest mean CV-AUC ({winner_auc:.3f}), but only the")
    print("  gap(s) whose corrected paired interval excludes 0 (printed above) are")
    print("  supported at this sample size – see per-model output for which.")
    print("  In any case: deep learning does not automatically win on small")
    print("  tabular clinical data – a fixed-architecture MLP needs far more than")
    print(f"  {len(X)} rows to reliably beat a well-regularised linear or")
    print("  tree-ensemble baseline.")


if __name__ == "__main__":
    main()
