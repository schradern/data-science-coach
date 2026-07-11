"""Figures for module 24. Run: python module/24-praediktion-workflow/code/figures.py

Writes PNGs to ../assets/. German labels (display), English code.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.feature_selection import SelectKBest, f_classif  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from lib.helpers import SEED, load_cohort, load_labs  # noqa: E402
from lib.plotstyle import EVENT, PRIMARY, SECONDARY, apply_style, save  # noqa: E402

ASSETS = Path(__file__).resolve().parent.parent / "assets"
NUMERIC = ["alter", "sofa_score", "crp_mg_l", "bmi", "leukozyten_g_l", "kreatinin_mg_dl", "laktat_mmol_l"]


def fig_leakage(X_num, y, cv):
    rng = np.random.default_rng(SEED)
    X_aug = np.hstack([StandardScaler().fit_transform(X_num), rng.normal(size=(len(y), 300))])
    leaky = cross_val_score(LogisticRegression(max_iter=1000),
                            SelectKBest(f_classif, k=12).fit_transform(X_aug, y),
                            y, cv=cv, scoring="roc_auc").mean()
    honest = cross_val_score(Pipeline([("s", SelectKBest(f_classif, k=12)),
                                       ("m", LogisticRegression(max_iter=1000))]),
                             X_aug, y, cv=cv, scoring="roc_auc").mean()
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar(["mit Leakage\n(Auswahl auf allen Daten)", "korrekt\n(Auswahl in der Pipeline)"],
           [leaky, honest], color=[EVENT, PRIMARY], width=0.6)
    for i, v in enumerate([leaky, honest]):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontweight="bold")
    ax.axhline(0.5, color=SECONDARY, lw=0.8, ls="--")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Kreuzvalidierte ROC-AUC")
    ax.set_title("Data Leakage täuscht eine bessere Modellgüte vor")
    save(fig, ASSETS / "leakage_vergleich.png")


def fig_threshold(proba, y_test):
    fpr, tpr, thr = roc_curve(y_test, proba)
    sens, spec = tpr, 1 - fpr
    youden = int(np.argmax(tpr - fpr))
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(thr, sens, color=PRIMARY, label="Sensitivität")
    ax.plot(thr, spec, color=EVENT, label="Spezifität")
    ax.axvline(thr[youden], color=SECONDARY, ls="--", lw=1)
    ax.text(thr[youden], 0.05, f" Youden = {thr[youden]:.2f}", color=SECONDARY, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Entscheidungsschwelle")
    ax.set_ylabel("Anteil")
    ax.set_title("Die Schwelle bestimmt Sensitivität und Spezifität")
    ax.legend(loc="center right")
    save(fig, ASSETS / "schwellenwahl.png")


def main() -> None:
    apply_style()
    df = load_cohort().merge(load_labs(), on="patient_id", how="left")
    y = df["verstorben_30d"]
    X_num = SimpleImputer(strategy="median").fit_transform(df[NUMERIC])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fig_leakage(X_num, y, cv)

    X_tr, X_te, y_tr, y_te = train_test_split(X_num, y, test_size=0.25, stratify=y, random_state=SEED)
    model = Pipeline([("scale", StandardScaler()),
                      ("m", LogisticRegression(max_iter=1000, class_weight="balanced"))]).fit(X_tr, y_tr)
    fig_threshold(model.predict_proba(X_te)[:, 1], y_te)


if __name__ == "__main__":
    main()
