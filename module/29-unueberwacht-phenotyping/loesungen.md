# Musterlösungen · Unüberwachtes Lernen und Phänotypisierung von Patient:innen

## Aufgabe 1 – k wählen und begründen

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

scores = {}
for k in range(2, 9):
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
    scores[k] = silhouette_score(X, labels)

best_k = max(scores, key=scores.get)
print(f"Highest silhouette at k={best_k}: {scores[best_k]:.3f}")
```

In dieser Kohorte ist der Silhouettenkoeffizient bei **k=2 am höchsten (0,130)**,
knapp gefolgt von k=7 (0,123) und k=5 (0,122); k=3 liegt mit 0,115 sogar
**unter** dem Durchschnitt der getesteten k-Werte. Alle Werte liegen im
schmalen Band 0,10–0,13, das Elbow-Diagramm der Inertia fällt glatt und
gleichmäßig ab, ohne ausgeprägten Knick bei einem bestimmten k. Das ist in
Summe ein Hinweis auf eine **schwache, überlappende Clusterstruktur**: Die
Kohorte zerfällt nicht in klar getrennte Gruppen, unabhängig vom gewählten k.

Die statistische Antwort (k=2) reicht trotzdem nicht als alleinige
Begründung: Zwei Gruppen wären für eine differenzierte Risikostratifizierung
oft zu grob. Im weiteren Verlauf dieses Moduls wird bewusst mit **k=3**
weitergearbeitet (leicht / mittelschwer / hochrisiko), weil drei
Schweregrad-Stufen klinisch interpretierbar sind und sich in Mortalität und
Verweildauer unterscheiden lassen (siehe Aufgabe 4), eine explizit **klinisch
begründete**, keine rein datengetriebene Entscheidung. Wichtig für die
Publikationspraxis: Diese Wahl muss im Methodenteil offen benannt werden
("wir wählen k=3 aus klinischen Gründen, obwohl k=2 den höchsten
Silhouettenwert hat"), nicht als "das Elbow/die Silhouette bestätigt k=3"
verkauft werden, das wäre hier schlicht falsch.

## Aufgabe 2 – Auswirkung der Skalierung

```python
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import numpy as np

X_raw = SimpleImputer(strategy="median").fit_transform(df[NUMERIC])
labels_raw   = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_raw)
labels_scaled = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X)  # standardized

import pandas as pd
df["cluster_raw"]    = labels_raw
df["cluster_scaled"] = labels_scaled
print(df.groupby("cluster_raw")[NUMERIC].mean().round(1))
print(df.groupby("cluster_scaled")[NUMERIC].mean().round(1))
```

Ohne Skalierung dominiert CRP (mg/l, Wertebereich 0–300) die euklidische
Distanz, Cluster werden fast ausschließlich nach CRP-Niveau gebildet. Mit
Skalierung trägt jedes Merkmal gleichgewichtig bei; die Cluster spiegeln dann
ein komplexeres klinisches Muster wider.

## Aufgabe 3 – Hierarchisches vs. k-Means

```python
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
import numpy as np

Z = linkage(X, method="ward")
labels_hier = fcluster(Z, t=3, criterion="maxclust") - 1  # 0-indexed, matches KMeans
labels_km   = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X)

# ARI is the right primary agreement metric: it's invariant to label
# permutation, unlike a raw (labels_km != labels_hier) count.
ari = adjusted_rand_score(labels_km, labels_hier)
print(f"Adjusted Rand Index: {ari:.3f}")

# A raw mismatch count is only valid AFTER aligning cluster IDs (Hungarian
# algorithm): "cluster 0" from k-Means and "cluster 0" from hierarchical
# clustering have no necessary relationship, comparing them directly with
# != would give a number that depends on arbitrary label numbering, not on
# actual (dis)agreement.
def align_labels(a: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
    cost = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            cost[i, j] = -np.sum((a == i) & (b == j))
    row, col = linear_sum_assignment(cost)
    mapping = dict(zip(row, col))
    return np.array([mapping[label] for label in b])

aligned_hier = align_labels(labels_km, labels_hier, k=3)
mismatch = np.sum(labels_km != aligned_hier)
print(f"Patients in different clusters (after label alignment): {mismatch} / {len(labels_km)}")
```

In dieser Kohorte: **ARI = 0,243** (niedrig bis mäßig, weit unter dem oft
zitierten Schwellenwert 0,7 für "stabile" Cluster), und nach
Label-Angleichung landen **177 von 500 Patient:innen (35 %)** in
unterschiedlichen Clustern je nach Verfahren. Das zeigt: Die Cluster sind
**nicht robust**, die Wahl des Algorithmus (k-Means vs. hierarchisch)
beeinflusst spürbar, wer in welche Gruppe fällt, ein wichtiges Caveat im
Methodik-Abschnitt jeder Phänotypisierungsstudie und konsistent mit der
schwachen Clusterstruktur aus Aufgabe 1 (niedrige Silhouettenwerte für
jedes k).

## Aufgabe 4 – Phänotyp-Profil

```python
import pandas as pd

df["cluster"] = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X)
profile = df.groupby("cluster")[
    ["alter", "sofa_score", "crp_mg_l", "laktat_mmol_l", "verweildauer_tage", "verstorben_30d"]
].mean().round(2)
print(profile.T)

# Names are assigned AFTER reading the profile, ranked by mean SOFA — never
# hardcode {0: "mild", 1: "severe", ...}: k-Means cluster IDs are arbitrary
# and not guaranteed to come out in severity order.
severity_order = profile["sofa_score"].sort_values().index.tolist()
phenotype_names = dict(zip(severity_order, ["Leichter Verlauf", "Mittelschwer", "Hochrisiko"]))
df["phenotype"] = df["cluster"].map(phenotype_names)
```

Beispielhafter Befund in dieser Kohorte (k=3, seed=42): ein Cluster mit
niedrigem SOFA (2,4) und niedriger Mortalität (5 %) -> "Leichter Verlauf";
ein Cluster mit hohem SOFA (6,0), hohem Laktat (3,2 mmol/l) und hoher
Mortalität (30 %) -> "Hochrisiko"; ein dazwischenliegender Cluster (SOFA 3,3,
Mortalität 16 %) -> "Mittelschwer". Verweildauer unterscheidet sich zwischen
den Clustern nur wenig (6,2–7,2 Tage), sie trägt hier kaum zur Trennung bei,
ein Hinweis darauf, dass die Cluster primär akute Schwere und nicht
Krankenhausverlauf abbilden.

Die Benennung ist **interpretativ** und muss explizit begründet werden.
Einschränkungen: (1) Cluster-Nummern sind arbiträr und können zwischen Runs
variieren; (2) die Grenzen zwischen Clustern sind weich, nicht medizinisch
definiert; (3) externe Validierung (z. B. an einer zweiten Kohorte) ist
notwendig, bevor man die Phänotypen als klinisch real behandelt.

## Bonus – UMAP vs. PCA

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

try:
    import umap
    X_umap = umap.UMAP(n_components=2, random_state=42).fit_transform(X)
    method = "UMAP"
except (ImportError, TypeError):
    # ImportError: not installed. TypeError: installed umap-learn can be
    # incompatible with the installed scikit-learn version (version skew).
    X_umap = X_pca
    method = "PCA (UMAP nicht verfügbar/inkompatibel)"

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, coords, title in zip(axes, [X_pca, X_umap], ["PCA", method]):
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=df["cluster"], cmap="Set1", s=15)
    ax.set_title(title)
plt.tight_layout()
plt.show()
```

UMAP bewahrt lokale Nachbarschaftsstrukturen, Cluster erscheinen oft kompakter
und visuell klarer getrennt. PCA ist linear: die Achsen haben eine direkte
Bedeutung (Ladungen), das Bild ist reproduzierbar ohne Hyperparameter. UMAP
ist nicht linear und von `n_neighbors` / `min_dist` abhängig, verschiedene
Hyperparameter können sehr verschiedene Layouts erzeugen, was die
Interpretierbarkeit einschränkt. Wichtig beim Lesen eines UMAP-Plots: nur die
**lokale** Nachbarschaft ist verlässlich, Abstände zwischen weit entfernten
Clustern und ihre relative Größe/Dichte im Bild sind **keine** verlässlichen
Aussagen über tatsächliche Ähnlichkeit oder Gruppengröße.
