"""Figures for module 30 – deep learning intro.

Run: python module/30-deep-learning/code/figures.py

Writes PNGs to assets/. German labels (display layer), English code.
torch is NOT imported – all code is runnable with sklearn only.
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
from sklearn.ensemble import GradientBoostingClassifier  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)

from lib.helpers import SEED, load_cohort, load_labs  # noqa: E402
from lib.plotstyle import EVENT, PRIMARY, SECONDARY, PALETTE, apply_style, save  # noqa: E402

ASSETS = Path(__file__).resolve().parent.parent / "assets"

NUMERIC = [
    "alter", "sofa_score", "crp_mg_l", "bmi",
    "leukozyten_g_l", "kreatinin_mg_dl", "laktat_mmol_l",
]
CATEGORICAL = ["aufnahmegrund", "raucherstatus"]
BINARY = ["diabetes", "hypertonie"]
TARGET = "verstorben_30d"


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("bin", "passthrough", BINARY),
    ])


def fig_lernkurve(X_train, y_train) -> None:
    """Plot MLP training loss curve and validation accuracy curve."""
    mlp_pipe = Pipeline([
        ("pre", build_preprocessor()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=0.01,
            early_stopping=True,
            validation_fraction=0.15,
            max_iter=300,
            random_state=SEED,
        )),
    ])
    mlp_pipe.fit(X_train, y_train)
    mlp = mlp_pipe.named_steps["mlp"]

    loss_curve = mlp.loss_curve_
    val_scores = mlp.validation_scores_
    epochs = range(1, len(loss_curve) + 1)

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(epochs, loss_curve, color=PRIMARY, lw=1.8, label="Trainingsverlust")
    ax1.set_xlabel("Epoche")
    ax1.set_ylabel("Verlust (log loss)", color=PRIMARY)
    ax1.tick_params(axis="y", labelcolor=PRIMARY)

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_scores, color=EVENT, lw=1.8, linestyle="--",
             label="Validierungsgüte (Genauigkeit)")
    ax2.set_ylabel("Validierungsgüte", color=EVENT)
    ax2.tick_params(axis="y", labelcolor=EVENT)
    ax2.grid(False)

    # Mark early-stopping point
    best_ep = int(np.argmax(val_scores)) + 1
    ax2.axvline(best_ep, color=SECONDARY, lw=1, ls=":")
    ax2.text(best_ep + 0.5, min(val_scores) + 0.01,
             f"Early stop\n(Epoche {best_ep})", color=SECONDARY, fontsize=9)

    ax1.set_title("MLP-Lernkurve: Verlust und Validierungsgüte über Epochen")
    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right")

    save(fig, ASSETS / "lernkurve.png")


def fig_auc_vergleich(X, y) -> None:
    """Bar chart comparing 5-fold CV AUC: LR vs GB vs MLP."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    def lr_pipe():
        return Pipeline([("pre", build_preprocessor()),
                         ("clf", LogisticRegression(max_iter=1000,
                                                    class_weight="balanced",
                                                    random_state=SEED))])

    def gb_pipe():
        return Pipeline([("pre", build_preprocessor()),
                         ("clf", GradientBoostingClassifier(n_estimators=200,
                                                             max_depth=3,
                                                             random_state=SEED))])

    def mlp_pipe():
        # early_stopping=False for CV: avoids shrinking each fold's training set
        # max_iter=500 prevents ConvergenceWarnings on this small dataset
        return Pipeline([("pre", build_preprocessor()),
                         ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32),
                                               alpha=0.01,
                                               early_stopping=False,
                                               max_iter=500,
                                               random_state=SEED))])

    models = [
        ("Logistische\nRegression", lr_pipe(), PRIMARY),
        ("Gradient\nBoosting", gb_pipe(), PALETTE[2]),
        ("MLP\n(Neuronales Netz)", mlp_pipe(), EVENT),
    ]

    names, aucs, colors = [], [], []
    for name, pipe, color in models:
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
        names.append(name)
        aucs.append(scores.mean())
        colors.append(color)
        print(f"  {name.replace(chr(10), ' ')}: AUC={scores.mean():.3f}")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(names, aucs, color=colors, width=0.55)
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, auc + 0.005,
                f"{auc:.3f}", ha="center", fontweight="bold", fontsize=11)

    ax.axhline(0.5, color=SECONDARY, lw=0.8, ls="--")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Kreuzvalidierte ROC-AUC (5-fach, stratifiziert)")

    # Data-driven verdict – NEVER hardcode who "wins": derive it from the
    # computed aucs so the title can't drift out of sync with the bars.
    order = sorted(zip(names, aucs), key=lambda t: -t[1])
    best_name, best_auc = order[0]
    runner_up_name, runner_up_auc = order[1]
    gap = best_auc - runner_up_auc
    best_flat = best_name.replace("\n", " ")
    runner_up_flat = runner_up_name.replace("\n", " ")
    verdict = (f"{best_flat} liegt vorn" if gap > 0.01
               else f"{best_flat} und {runner_up_flat} liegen gleichauf")

    ax.set_title("AUC-Vergleich: Logistische Regression vs. Gradient Boosting vs. MLP\n"
                 f"(synthetische Kohorte, n≈{len(X)} – {verdict})")

    save(fig, ASSETS / "auc_vergleich.png")


def main() -> None:
    apply_style()
    df = load_cohort().merge(load_labs(), on="patient_id", how="left")
    X = df[NUMERIC + CATEGORICAL + BINARY]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED)

    print("Erzeuge Lernkurve …")
    fig_lernkurve(X_train, y_train)

    print("Erzeuge AUC-Vergleich …")
    fig_auc_vergleich(X, y)

    print("Fertig – alle Abbildungen in", ASSETS)


if __name__ == "__main__":
    main()
