"""Figures for module 29 — unsupervised learning and clinical phenotyping.

Run: python module/29-unueberwacht-phenotyping/code/figures.py

Writes PNGs to assets/. German labels (display layer), English code.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from lib.helpers import SEED, load_cohort, load_labs  # noqa: E402
from lib.plotstyle import PALETTE, PRIMARY, SECONDARY, EVENT, apply_style, save  # noqa: E402

ASSETS = Path(__file__).resolve().parent.parent / "assets"
NUMERIC = [
    "alter", "sofa_score", "crp_mg_l", "bmi",
    "leukozyten_g_l", "kreatinin_mg_dl", "laktat_mmol_l",
]


def build_features() -> tuple[np.ndarray, pd.DataFrame]:
    """Load, impute and scale numeric clinical features. Return (X_scaled, df)."""
    df = load_cohort().merge(load_labs(), on="patient_id", how="left")
    prep = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    X = prep.fit_transform(df[NUMERIC])
    return X, df


def fig_silhouette_k(X: np.ndarray) -> None:
    """Bar chart of silhouette scores for k = 2..8, plus inertia on twin axis."""
    k_values = list(range(2, 9))
    silhouettes, inertias = [], []

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        labels = km.fit_predict(X)
        silhouettes.append(silhouette_score(X, labels))
        inertias.append(km.inertia_)

    fig, ax1 = plt.subplots(figsize=(7, 4))

    bars = ax1.bar(k_values, silhouettes, color=PRIMARY, alpha=0.85, width=0.6,
                   label="Silhouettenkoeffizient")
    # Highlight the best k
    best_idx = int(np.argmax(silhouettes))
    bars[best_idx].set_color(EVENT)
    ax1.set_xlabel("Anzahl Cluster k")
    ax1.set_ylabel("Silhouettenkoeffizient", color=PRIMARY)
    ax1.tick_params(axis="y", labelcolor=PRIMARY)
    ax1.set_ylim(0, max(silhouettes) * 1.25)
    ax1.set_xticks(k_values)

    ax2 = ax1.twinx()
    ax2.plot(k_values, inertias, color=SECONDARY, marker="o", lw=1.5,
             linestyle="--", label="Inertia (Elbow)")
    ax2.set_ylabel("Inertia", color=SECONDARY)
    ax2.tick_params(axis="y", labelcolor=SECONDARY)
    ax2.grid(False)

    ax1.set_title("Silhouettenkoeffizient und Inertia je k\n"
                  "(roter Balken = bestes k nach Silhouette)")
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    save(fig, ASSETS / "silhouette_k.png")


def fig_pca_cluster(X: np.ndarray, df) -> None:
    """2D PCA scatter coloured by k-Means cluster (k=3).

    Cluster names are assigned by ranking clusters on mean SOFA (illness
    severity) AFTER clustering, not by hardcoding cluster ID -> name. Raw
    k-Means cluster IDs (0, 1, 2, ...) are arbitrary and are not guaranteed
    to come out in severity order, so a fixed {0: "mild", 1: "moderate", ...}
    mapping would silently mislabel the plot whenever the ID order and the
    severity order disagree (as they do for this cohort/seed).
    """
    km = KMeans(n_clusters=3, random_state=SEED, n_init=10)
    labels = km.fit_predict(X)

    pca = PCA(n_components=2, random_state=SEED)
    X_2d = pca.fit_transform(X)
    var = pca.explained_variance_ratio_

    sofa_by_cluster = pd.Series(df["sofa_score"].values).groupby(labels).mean()
    severity_order = sofa_by_cluster.sort_values().index.tolist()
    severity_labels = ["leichter Verlauf", "mittelschwer", "Hochrisiko"]
    cluster_colors = {c: col for c, col in zip(severity_order, [PRIMARY, PALETTE[2], EVENT])}
    cluster_names = {c: f"Cluster {i + 1} ({severity_labels[i]})"
                      for i, c in enumerate(severity_order)}
    cluster_markers = {c: m for c, m in zip(severity_order, ["o", "s", "^"])}

    fig, ax = plt.subplots(figsize=(7, 5))
    for c in severity_order:
        mask = labels == c
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   color=cluster_colors[c], label=cluster_names[c],
                   marker=cluster_markers[c],
                   alpha=0.65, s=28, linewidths=0)

    ax.set_xlabel(f"Hauptkomponente 1 ({var[0]:.1%} Varianz erklärt)")
    ax.set_ylabel(f"Hauptkomponente 2 ({var[1]:.1%} Varianz erklärt)")
    ax.set_title("2D-PCA-Projektion eingefärbt nach k-Means-Cluster (k=3)")
    ax.legend(loc="best", markerscale=1.4)

    save(fig, ASSETS / "pca_cluster.png")


def fig_pca_scale_effect() -> None:
    """Show the effect of scaling on PCA (Scale Dependency)."""
    df = load_cohort().merge(load_labs(), on="patient_id", how="left").dropna(subset=["alter", "kreatinin_mg_dl"])
    X_raw = df[["alter", "kreatinin_mg_dl"]].values
    
    # Left: Without scaling
    pca_raw = PCA(n_components=2, random_state=SEED)
    X_raw_2d = pca_raw.fit_transform(X_raw)
    
    # Right: With scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    pca_scaled = PCA(n_components=2, random_state=SEED)
    X_scaled_2d = pca_scaled.fit_transform(X_scaled)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    
    # Without scaling scatter
    sc1 = ax1.scatter(X_raw_2d[:, 0], X_raw_2d[:, 1], c=X_raw[:, 1], cmap="coolwarm", alpha=0.7)
    ax1.set_xlabel("PC 1 (Dominiert vom Alter)")
    ax1.set_ylabel("PC 2")
    ax1.set_title("Ohne Standardisierung\n(Alter dominiert wegen großer Varianz)")
    fig.colorbar(sc1, ax=ax1, label="Kreatinin (mg/dl)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # With scaling scatter
    sc2 = ax2.scatter(X_scaled_2d[:, 0], X_scaled_2d[:, 1], c=X_raw[:, 1], cmap="coolwarm", alpha=0.7)
    ax2.set_xlabel("PC 1")
    ax2.set_ylabel("PC 2")
    ax2.set_title("Mit Standardisierung (StandardScaler)\n(Beide Merkmale tragen gleichwertig bei)")
    fig.colorbar(sc2, ax=ax2, label="Kreatinin (mg/dl)")
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    save(fig, ASSETS / "pca_scale_effect.png")


def main() -> None:
    apply_style()
    X, df = build_features()
    fig_silhouette_k(X)
    fig_pca_cluster(X, df)
    fig_pca_scale_effect()
    print("Fertig — alle Abbildungen in", ASSETS)


if __name__ == "__main__":
    main()
