"""Module 29 — Unsupervised learning and clinical phenotyping.

Runs standalone from the project root:
    python module/29-unueberwacht-phenotyping/code/python.py

Data: read from data/ (committed with the repo); if that folder is
missing, the same files are fetched from the published URL.
Guaranteed packages: sklearn, scipy. Optional: umap-learn (falls back to TSNE).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cluster import KMeans, AgglomerativeClustering  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from scipy.cluster.hierarchy import linkage, fcluster  # noqa: E402

from lib.helpers import SEED, load_cohort, load_labs  # noqa: E402

NUMERIC = [
    "alter", "sofa_score", "crp_mg_l", "bmi",
    "leukozyten_g_l", "kreatinin_mg_dl", "laktat_mmol_l",
]


def build_features() -> tuple[np.ndarray, pd.DataFrame]:
    """Load, impute and scale numeric clinical features. Return (X_scaled, df)."""
    df = load_cohort().merge(load_labs(), on="patient_id", how="left")
    prep = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
    ])
    X = prep.fit_transform(df[NUMERIC])
    return X, df


def choose_k(X: np.ndarray) -> int:
    """Print elbow (inertia) and silhouette scores for k = 2..8. Return the
    silhouette-best k (NOT necessarily the k used downstream — see note)."""
    print("=== 1) Choosing k: inertia and silhouette ===")
    best_k, best_sil = 2, -1.0
    sils: dict[int, float] = {}
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        sils[k] = sil
        print(f"  k={k}  inertia={km.inertia_:.1f}  silhouette={sil:.3f}")
        if sil > best_sil:
            best_sil, best_k = sil, k
    print(f"  -> best k by silhouette: {best_k}  (score={best_sil:.3f})")
    spread = max(sils.values()) - min(sils.values())
    if spread < 0.05:
        print(
            f"  Note: silhouette values are low and close together "
            f"(range {min(sils.values()):.3f}-{max(sils.values()):.3f}) -> weak, "
            "overlapping cluster structure; no k separates the cohort cleanly. "
            "The rest of this script still uses k=3 below for clinical "
            "interpretability (three severity tiers), a deliberate pedagogical "
            "choice, not the statistically 'best' k."
        )
    return best_k


def cluster_kmeans(X: np.ndarray, k: int = 3) -> np.ndarray:
    """Fit k-Means and return labels."""
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    return km.fit_predict(X)


def cluster_hierarchical(X: np.ndarray, k: int = 3) -> np.ndarray:
    """Ward hierarchical clustering; return labels for k clusters."""
    Z = linkage(X, method="ward")
    return fcluster(Z, t=k, criterion="maxclust")


def reduce_dimensions(X: np.ndarray) -> tuple[np.ndarray, str]:
    """Reduce to 2D using UMAP (preferred) or TSNE as fallback.

    Falls back to TSNE both when umap-learn is missing (ImportError) and
    when an installed umap-learn is incompatible with the installed
    scikit-learn version (e.g. umap-learn passing a `check_array()` keyword
    that an older/newer sklearn doesn't accept -> TypeError). That is a real
    version-skew failure, not just a "package missing" one, so both are
    caught here rather than only ImportError.
    """
    try:
        import umap  # type: ignore
        reducer = umap.UMAP(n_components=2, random_state=SEED)
        X_2d = reducer.fit_transform(X)
        method = "UMAP"
    except (ImportError, TypeError) as exc:
        from sklearn.manifold import TSNE
        X_2d = TSNE(n_components=2, random_state=SEED, perplexity=30).fit_transform(X)
        method = "t-SNE (UMAP unavailable/incompatible; see note)"
        print(f"  Note: UMAP failed ({type(exc).__name__}: {exc}); "
              f"falling back to {method}")
    print(f"  Dimensionality reduction: {method}")
    return X_2d, method


def profile_clusters(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Print mean feature values per cluster (incl. length-of-stay and
    30-day mortality) and return the profile table."""
    df = df.copy()
    df["cluster"] = labels
    cols = NUMERIC + ["verweildauer_tage", "verstorben_30d"]
    profile = df.groupby("cluster")[cols].mean().round(2)
    print(profile.T.to_string())
    return profile


def name_phenotypes(profile: pd.DataFrame) -> dict[int, str]:
    """Rank clusters by mean SOFA (illness severity) and assign qualitative
    names AFTER reading the profile.

    k-Means cluster IDs (0, 1, 2, ...) are arbitrary and can be permuted
    between runs/seeds, so names must never be hardcoded against a fixed ID
    (e.g. {1: "mild", 2: "moderate", 3: "severe"}); that silently mislabels
    clusters whenever the ID order doesn't match severity order.
    """
    order = profile["sofa_score"].sort_values().index.tolist()
    names = ["Leichter Verlauf", "Mittelschwer", "Hochrisiko"]
    return dict(zip(order, names[: len(order)]))


def main() -> None:
    X, df = build_features()

    print("=== 1) Choosing k ===")
    best_k = choose_k(X)

    print(f"\n=== 2) k-Means clustering (k=3; silhouette-best was k={best_k}, "
          "see note above) ===")
    labels_km = cluster_kmeans(X, k=3)
    sil = silhouette_score(X, labels_km)
    print(f"  k-Means silhouette: {sil:.3f}")

    print("\n=== 3) Hierarchical clustering (Ward, k=3) ===")
    labels_hier = cluster_hierarchical(X, k=3)
    sil_h = silhouette_score(X, labels_hier)
    print(f"  Hierarchical silhouette: {sil_h:.3f}")

    # ARI is label-permutation-invariant, unlike a raw `labels_km != labels_hier`
    # count: cluster ID numbers are arbitrary between two independent
    # clustering runs, so comparing raw IDs directly is not meaningful.
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(labels_km, labels_hier)
    print(f"  Agreement (Adjusted Rand Index): {ari:.3f}  "
          "(0 = random labelling, 1 = perfect agreement; this is low-to-moderate)")

    print("\n=== 4) Dimensionality reduction ===")
    pca = PCA(n_components=2, random_state=SEED)
    X_pca = pca.fit_transform(X)
    print(f"  PCA PC1+PC2 explain {pca.explained_variance_ratio_.sum():.1%} of variance")
    X_2d, method = reduce_dimensions(X)

    print("\n=== 5) Clinical phenotype profiles (k-Means, k=3) ===")
    profile = profile_clusters(df, labels_km)
    phenotype_names = name_phenotypes(profile)
    counts = pd.Series(labels_km).map(phenotype_names).value_counts()
    print(f"\n  Phenotype names (assigned AFTER reading the profile, ranked by "
          f"mean SOFA): {phenotype_names}")
    print(f"  Group sizes:\n{counts.to_string()}")
    print("\nNote: cluster labels are exploratory, not diagnoses.")
    print("External validation required before any clinical use.")


if __name__ == "__main__":
    main()
