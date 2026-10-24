# Module 29 – Unsupervised learning and clinical phenotyping (R, parallel to Python).
#   Rscript module/29-unueberwacht-phenotyping/code/r.R
# Data: read from data/ (committed with the repo); if that folder is
# missing, the same files are fetched from the published URL.
# Code is English; German dataset column names are kept as-is.

suppressPackageStartupMessages({
  missing_pkgs <- setdiff(c("tidyverse", "recipes"), rownames(installed.packages()))
  if (length(missing_pkgs) > 0) {
    message("Missing R packages: ", paste(missing_pkgs, collapse = ", "))
    message("This lesson cannot run without them; nothing was computed.")
    message("Full per-module package list: DEPENDENCIES-R.md")
    message("Install with: install.packages(c('tidyverse', 'recipes'))")
    quit(save = "no", status = 1)
  }
  library(tidyverse)
  library(recipes)
  library(cluster)    # silhouette(), pam()
  if (requireNamespace("factoextra", quietly = TRUE)) {
    library(factoextra)
  }
})

script <- normalizePath(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1]))
root   <- dirname(dirname(dirname(dirname(script))))
source(file.path(root, "lib", "helpers.R"))
set.seed(SEED)

# ── 1) Load and preprocess ────────────────────────────────────────────────────
df <- load_cohort() |> left_join(load_labs(), by = "patient_id")

numeric_cols <- c("alter", "sofa_score", "crp_mg_l", "bmi",
                  "leukozyten_g_l", "kreatinin_mg_dl", "laktat_mmol_l")

rec <- recipe(~ ., data = df[numeric_cols]) |>
  step_impute_median(all_predictors()) |>
  step_normalize(all_predictors())

X_scaled <- rec |> prep() |> bake(new_data = NULL) |> as.matrix()

cat("── 1) Preprocessing done:", nrow(X_scaled), "patients,",
    ncol(X_scaled), "features\n")

# ── 2) Choose k: silhouette and elbow ─────────────────────────────────────────
cat("\n── 2) Silhouette scores for k = 2..8 ──\n")
sil_scores <- numeric(7)
for (k in 2:8) {
  km  <- kmeans(X_scaled, centers = k, nstart = 10, iter.max = 100)
  sil <- silhouette(km$cluster, dist(X_scaled))
  sil_scores[k - 1] <- mean(sil[, 3])
  cat(sprintf("  k=%d  silhouette=%.3f  inertia=%.1f\n",
              k, mean(sil[, 3]), km$tot.withinss))
}
best_k <- which.max(sil_scores) + 1
cat(sprintf("  -> Best k by silhouette: %d (score=%.3f)\n", best_k, max(sil_scores)))
if ((max(sil_scores) - min(sil_scores)) < 0.05) {
  cat("  Note: silhouette values are low and close together -> weak, overlapping\n")
  cat("  cluster structure; no k separates the cohort cleanly. The rest of this\n")
  cat("  script still uses k=3 below for clinical interpretability (three severity\n")
  cat("  tiers), a deliberate pedagogical choice, not the statistically 'best' k.\n")
}

# ── 3) k-Means with k=3 (chosen for clinical interpretability, see note above) ─
cat("\n── 3) k-Means (k=3) ──\n")
km3 <- kmeans(X_scaled, centers = 3, nstart = 10, iter.max = 100)
df$cluster_km <- factor(km3$cluster)
sil3 <- mean(silhouette(km3$cluster, dist(X_scaled))[, 3])
cat(sprintf("  Silhouette (k=3): %.3f\n", sil3))

# ── 4) Hierarchical clustering (Ward) ─────────────────────────────────────────
cat("\n── 4) Hierarchical clustering (Ward) ──\n")
dist_mat  <- dist(X_scaled, method = "euclidean")
hc        <- hclust(dist_mat, method = "ward.D2")
df$cluster_hier <- factor(cutree(hc, k = 3))

sil_h <- mean(silhouette(as.integer(df$cluster_hier), dist_mat)[, 3])
cat(sprintf("  Ward silhouette (k=3): %.3f\n", sil_h))

# Agreement between methods. A contingency table alone doesn't give a single
# comparable number, and comparing raw cluster IDs directly (e.g. counting
# km != hier) is NOT valid: cluster ID numbers are arbitrary between two
# independent clustering runs. The Adjusted Rand Index (ARI) is invariant to
# label permutation and is the standard way to compare two clusterings.
adjusted_rand_index <- function(a, b) {
  tab <- table(a, b)
  n <- sum(tab)
  sum_comb <- function(x) sum(choose(x, 2))
  index    <- sum_comb(tab)
  expected <- sum_comb(rowSums(tab)) * sum_comb(colSums(tab)) / choose(n, 2)
  max_idx  <- 0.5 * (sum_comb(rowSums(tab)) + sum_comb(colSums(tab)))
  if (max_idx == expected) return(0)
  (index - expected) / (max_idx - expected)
}

tab <- table(km = df$cluster_km, hier = df$cluster_hier)
cat("  k-Means vs hierarchical – contingency table:\n")
print(tab)
ari <- adjusted_rand_index(df$cluster_km, df$cluster_hier)
cat(sprintf("  Agreement (Adjusted Rand Index): %.3f  (0 = random, 1 = perfect)\n", ari))

# ── 5) Dimensionality reduction: PCA ──────────────────────────────────────────
cat("\n── 5) PCA (2D) ──\n")
pca_res  <- prcomp(X_scaled, scale. = FALSE)
var_exp  <- summary(pca_res)$importance[2, 1:2]
cat(sprintf("  PC1: %.1f%%  PC2: %.1f%%  (total: %.1f%%)\n",
            var_exp[1] * 100, var_exp[2] * 100, sum(var_exp) * 100))

# Optional: UMAP
tryCatch({
  library(umap)
  umap_res <- umap(X_scaled, random_state = SEED)
  cat("  UMAP: erfolgreich (umap-Paket verfügbar)\n")
}, error = function(e) {
  cat("  UMAP nicht verfügbar – PCA-2D als Fallback\n")
})

# ── 6) Phenotype profiles ──────────────────────────────────────────────────────
cat("\n── 6) Cluster phenotype profiles ──\n")
profile <- df |>
  group_by(cluster_km) |>
  summarise(across(all_of(c(numeric_cols, "verweildauer_tage", "verstorben_30d")),
                   \(x) mean(x, na.rm = TRUE)), .groups = "drop") |>
  mutate(across(where(is.numeric), \(x) round(x, 2)))
print(as.data.frame(profile))

cat("\nHinweis: Cluster sind explorative Gruppen – keine Diagnosen.\n")
cat("Externe Validierung vor klinischem Einsatz erforderlich.\n")
