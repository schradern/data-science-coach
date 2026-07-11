"""Figures for module 26. Run: python module/26-ensembles-boosting/code/figures.py

Writes PNGs to ../assets/. German labels (display), English code.

Leakage note: like code/python.py, all preprocessing (imputation, one-hot
encoding) lives inside a Pipeline that cross_val_score()/manual fit-transform
only ever applies to a training split/fold, never to the full dataset up
front.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import warnings  # noqa: E402

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
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
from lib.plotstyle import EVENT, PALETTE, PRIMARY, SECONDARY, apply_style, save  # noqa: E402

ASSETS = Path(__file__).resolve().parent.parent / "assets"

NUMERIC = ["alter", "sofa_score", "crp_mg_l", "bmi", "leukozyten_g_l", "kreatinin_mg_dl", "laktat_mmol_l"]
CATEGORICAL = ["aufnahmegrund", "raucherstatus"]
BINARY = ["diabetes", "hypertonie"]
TARGET = "verstorben_30d"


def load_data():
    """Return the RAW feature frame (no imputation/encoding yet) and y."""
    df = load_cohort().merge(load_labs(), on="patient_id", how="left")
    X = df[NUMERIC + CATEGORICAL + BINARY]
    y = df[TARGET]
    return X, y


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("bin", "passthrough", BINARY),
    ])


def build_pipeline(model) -> Pipeline:
    return Pipeline([("pre", build_preprocessor()), ("model", model)])


def build_native_missing_pipeline(model) -> Pipeline:
    """HistGradientBoosting: keep NaN in numeric columns (handled natively),
    still one-hot-encode categoricals inside the pipeline."""
    pre = ColumnTransformer([
        ("num", "passthrough", NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("bin", "passthrough", BINARY),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def fig_baum_tiefe(X_train, X_val, y_train, y_val) -> None:
    """Train vs validation AUC over decision tree depth (overfitting curve)."""
    depths = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]
    train_aucs, val_aucs = [], []
    for d in depths:
        tree = DecisionTreeClassifier(max_depth=d, random_state=SEED)
        tree.fit(X_train, y_train)
        train_aucs.append(roc_auc_score(y_train, tree.predict_proba(X_train)[:, 1]))
        val_aucs.append(roc_auc_score(y_val,   tree.predict_proba(X_val)[:, 1]))

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(depths, train_aucs, "o-", color=PRIMARY, label="Trainings-AUC")
    ax.plot(depths, val_aucs,   "s-", color=EVENT,   label="Validierungs-AUC")
    ax.axvline(depths[int(np.argmax(val_aucs))], color=SECONDARY, lw=0.9,
               ls="--", label=f"Optimale Tiefe = {depths[int(np.argmax(val_aucs))]}")
    ax.set_xlabel("Baumtiefe (max_depth)")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Überanpassung: AUC über Baumtiefe")
    ax.legend(loc="lower right")
    save(fig, ASSETS / "baum_tiefe.png")


def fig_modellvergleich(X, y, cv) -> None:
    """CV-AUC per model with error bars. Title/caption never hardcode a
    winner -- build.py/README quote whatever this run actually produced."""
    tree = DecisionTreeClassifier(max_depth=4, random_state=SEED)
    rf   = RandomForestClassifier(
        n_estimators=300, max_features="sqrt",
        class_weight="balanced", random_state=SEED, n_jobs=-1,
    )
    hgb  = HistGradientBoostingClassifier(
        max_iter=300, max_depth=5, learning_rate=0.05,
        class_weight="balanced", random_state=SEED,
    )

    model_specs = [
        ("Entscheidungsbaum\n(Tiefe 4)", build_pipeline(tree)),
        ("Random\nForest",               build_pipeline(rf)),
        ("HistGradBoost\n(sklearn)",      build_native_missing_pipeline(hgb)),
    ]

    # Try optional boosters
    pos_weight = float((y == 0).sum()) / float((y == 1).sum())

    try:
        import xgboost as xgb
    except Exception:
        xgb = None

    if xgb is not None:
        try:
            xgb_m = xgb.XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                scale_pos_weight=pos_weight,
                eval_metric="auc", random_state=SEED, verbosity=0,
                use_label_encoder=False,
            )
        except TypeError:
            xgb_m = xgb.XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                scale_pos_weight=pos_weight,
                eval_metric="auc", random_state=SEED, verbosity=0,
            )
        model_specs.append(("XGBoost", build_pipeline(xgb_m)))

    try:
        import lightgbm as lgb
    except Exception:
        lgb = None

    if lgb is not None:
        lgb_m = lgb.LGBMClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            is_unbalance=True, random_state=SEED, verbose=-1,
        )
        model_specs.append(("LightGBM", build_pipeline(lgb_m)))

    names, means, stds = [], [], []
    for name, pipe in model_specs:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
        names.append(name)
        means.append(scores.mean())
        stds.append(scores.std())

    x = np.arange(len(names))
    colors = [PRIMARY, PALETTE[2], PALETTE[4], EVENT, PALETTE[5]][:len(names)]

    # Data-driven verdict for the title -- never a hardcoded winner.
    tree_mean = means[0]
    other_pairs = list(zip(names[1:], means[1:]))
    best_name, best_mean = max(other_pairs, key=lambda t: t[1])
    best_name_plain = best_name.replace("\n", " ").strip()
    gap = best_mean - tree_mean
    if gap > 0.02:
        verdict = f"{best_name_plain} übertrifft den einzelnen Baum"
    elif gap > -0.02:
        verdict = "Ensembles liegen etwa gleichauf mit dem einzelnen Baum"
    else:
        verdict = "der einzelne Baum schneidet hier nicht schlechter ab"

    fig, ax = plt.subplots(figsize=(max(7, len(names) * 1.5), 4.5))
    bars = ax.bar(x, means, yerr=stds, color=colors, width=0.55,
                  capsize=5, error_kw={"lw": 1.5})
    ax.axhline(0.5, color=SECONDARY, lw=0.8, ls="--", label="Zufallsklassifikator")
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.01,
                f"{m:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("Kreuzvalidierte ROC-AUC (±1 SD)")
    ax.set_title(f"Modellvergleich: Baum-Ensembles & Gradient Boosting\n({verdict}, n≈{len(y)})")
    ax.legend(loc="lower right")
    save(fig, ASSETS / "modellvergleich_auc.png")


def main() -> None:
    apply_style()
    X, y = load_data()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED
    )
    pre = build_preprocessor()
    X_train_imp = pre.fit_transform(X_train)   # fit on TRAIN only
    X_test_imp  = pre.transform(X_test)
    fig_baum_tiefe(X_train_imp, X_test_imp, y_train.values, y_test.values)
    fig_modellvergleich(X, y, cv)


if __name__ == "__main__":
    main()
