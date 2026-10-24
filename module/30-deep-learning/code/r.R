# Module 30 – Deep learning intro with nnet / tidymodels (R, parallel to Python).
#   Rscript module/30-deep-learning/code/r.R
# Data: read from data/ (committed with the repo); if that folder is
# missing, the same files are fetched from the published URL.
# Code is English; German dataset column names are kept as-is.
# torch is NOT used – nnet provides a single-hidden-layer MLP.

suppressPackageStartupMessages({
  missing_pkgs <- setdiff(c("tidyverse", "tidymodels", "nnet"), rownames(installed.packages()))
  if (length(missing_pkgs) > 0) {
    message("Missing R packages: ", paste(missing_pkgs, collapse = ", "))
    message("This lesson cannot run without them; nothing was computed.")
    message("Full per-module package list: DEPENDENCIES-R.md")
    message("Install with: install.packages(c('tidyverse', 'tidymodels', 'nnet'))")
    quit(save = "no", status = 1)
  }
  library(tidyverse)
  library(tidymodels)
  library(nnet)      # single-layer MLP backend
})

script <- normalizePath(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1]))
root   <- dirname(dirname(dirname(dirname(script))))
source(file.path(root, "lib", "helpers.R"))
set.seed(SEED)

# ── Data ──────────────────────────────────────────────────────────────────────
df <- load_cohort() |>
  left_join(load_labs(), by = "patient_id") |>
  mutate(verstorben_30d = factor(verstorben_30d, levels = c(1, 0)))

split  <- initial_split(df, prop = 0.75, strata = verstorben_30d)
train  <- training(split)
folds  <- vfold_cv(train, v = 5, strata = verstorben_30d)

# ── Recipe: impute + dummy + normalise (all preprocessing in one place) ───────
rec <- recipe(
    verstorben_30d ~ alter + sofa_score + crp_mg_l + bmi +
      leukozyten_g_l + kreatinin_mg_dl + laktat_mmol_l +
      aufnahmegrund + raucherstatus + diabetes + hypertonie,
    data = train
  ) |>
  step_impute_median(all_numeric_predictors()) |>
  step_dummy(all_nominal_predictors()) |>
  step_normalize(all_numeric_predictors())

# ── Model specifications ───────────────────────────────────────────────────────
spec_lr <- logistic_reg(penalty = 0.01, mixture = 0) |>
  set_engine("glmnet") |>
  set_mode("classification")

spec_nnet <- mlp(hidden_units = 64, penalty = 0.01, epochs = 200) |>
  set_engine("nnet", MaxNWts = 2000) |>
  set_mode("classification")

# Note: Gradient Boosting (xgboost) requires the xgboost package.
# If not installed, it is skipped gracefully below.

metric <- metric_set(roc_auc)

# fold_aucs(): per-fold roc_auc as a named-by-id numeric vector, so folds
# can be paired across models by fold id (not just by position).
fold_aucs <- function(res, colname) {
  collect_metrics(res, summarize = FALSE) |>
    filter(.metric == "roc_auc") |>
    select(id, !!colname := .estimate)
}

# summarise_folds(): mean, sample SD and two approximate 95% t-intervals
# (df = k-1). The naive one treats the folds as independent; they are not
# (training sets overlap), so it is too narrow (Bengio & Grandvalet 2004).
# The corrected one inflates the variance by the Nadeau-Bengio factor
# 1/k + n_test/n_train = 1/k + 1/(k-1). Both are descriptive approximations.
summarise_folds <- function(x) {
  k  <- length(x)
  m  <- mean(x)
  sd_ <- sd(x)
  t_crit <- qt(0.975, df = k - 1)
  hw <- t_crit * sd_ * sqrt(1 / k)
  hw_nb <- t_crit * sd_ * sqrt(1 / k + 1 / (k - 1))
  list(k = k, mean = m, sd = sd_, ci_low = m - hw, ci_high = m + hw,
       nb_low = m - hw_nb, nb_high = m + hw_nb)
}

print_summary <- function(label, x) {
  s <- summarise_folds(x)
  cat(sprintf("  %s\n", label))
  cat(sprintf("    Folds (k=%d): %s\n", s$k, paste(sprintf("%.3f", x), collapse = ", ")))
  cat(sprintf("    Mean = %.3f  SD = %.3f  naive 95%%-CI = [%.3f, %.3f]  corrected = [%.3f, %.3f]\n",
              s$mean, s$sd, s$ci_low, s$ci_high, s$nb_low, s$nb_high))
}

# ── Cross-validation: logistic regression ────────────────────────────────────
wf_lr  <- workflow() |> add_recipe(rec) |> add_model(spec_lr)
res_lr <- fit_resamples(wf_lr, folds, metrics = metric, control = control_resamples(save_pred = TRUE))
fold_lr <- fold_aucs(res_lr, "lr")

# ── Cross-validation: MLP (nnet) ─────────────────────────────────────────────
wf_mlp  <- workflow() |> add_recipe(rec) |> add_model(spec_nnet)
res_mlp <- fit_resamples(wf_mlp, folds, metrics = metric, control = control_resamples(save_pred = TRUE))
fold_mlp <- fold_aucs(res_mlp, "mlp")

# ── Optional: Gradient Boosting ───────────────────────────────────────────────
fold_gb <- NULL
tryCatch({
  library(xgboost)
  spec_gb <- boost_tree(trees = 200, tree_depth = 3) |>
    set_engine("xgboost") |>
    set_mode("classification")
  wf_gb  <- workflow() |> add_recipe(rec) |> add_model(spec_gb)
  res_gb <- fit_resamples(wf_gb, folds, metrics = metric, control = control_resamples(save_pred = TRUE))
  fold_gb <- fold_aucs(res_gb, "gb")
}, error = function(e) {
  cat("── Gradient Boosting: xgboost nicht verfügbar, übersprungen.\n")
})

# ── AUC comparison: per fold, not just the mean ───────────────────────────────
# Same rationale as code/python.py: with k=5 folds on ~500 patients, the
# fold-to-fold SD can rival the gap between models' means. Report per-fold
# scores + a 95% CI per model, and – the right comparison, since every
# model sees the exact same folds – the *paired* per-fold difference.
cat("\n=== AUC comparison via 5-fold CV (per-fold, not just the mean) ===\n")
fold_tbl <- fold_lr |> inner_join(fold_mlp, by = "id")
if (!is.null(fold_gb)) fold_tbl <- fold_tbl |> inner_join(fold_gb, by = "id")

print_summary("Logistische Regression", fold_tbl$lr)
if (!is.null(fold_gb)) print_summary("Gradient Boosting", fold_tbl$gb)
print_summary("MLP (nnet)", fold_tbl$mlp)

means <- c("Logistische Regression" = mean(fold_tbl$lr), "MLP (nnet)" = mean(fold_tbl$mlp))
if (!is.null(fold_gb)) means["Gradient Boosting"] <- mean(fold_tbl$gb)
sorted_means <- sort(means, decreasing = TRUE)
winner <- names(sorted_means)[1]
winner_col <- c("Logistische Regression" = "lr", "Gradient Boosting" = "gb", "MLP (nnet)" = "mlp")[[winner]]
cat(sprintf("\n  Highest mean CV-AUC (point estimate only): %s (%.3f)\n", winner, sorted_means[1]))

# Paired per-fold differences vs. the winner. The conclusion comes from the
# corrected interval; the naive one is printed for comparison only.
cat(sprintf("\n  Paired per-fold differences vs. %s (same folds for every model):\n", winner))
other_names <- setdiff(names(sorted_means), winner)
for (name in other_names) {
  col <- c("Logistische Regression" = "lr", "Gradient Boosting" = "gb", "MLP (nnet)" = "mlp")[[name]]
  diff <- fold_tbl[[winner_col]] - fold_tbl[[col]]
  s <- summarise_folds(diff)
  cat(sprintf("    %s - %s: %s\n", winner, name,
              paste(sprintf("%+.3f", diff), collapse = ", ")))
  cat(sprintf("      Mean diff = %+.3f  SD = %.3f  naive 95%%-CI = [%+.3f, %+.3f]  corrected = [%+.3f, %+.3f]\n",
              s$mean, s$sd, s$ci_low, s$ci_high, s$nb_low, s$nb_high))
  if (s$nb_low > 0 || s$nb_high < 0) {
    cat("      Auch das korrigierte Intervall schliesst 0 aus -> deutlicher Hinweis, kein exakter Beweis.\n")
  } else if (s$ci_low > 0 || s$ci_high < 0) {
    cat("      Nur das naive Intervall schliesst 0 aus -> bei korrelierten Folds nicht belastbar.\n")
  } else {
    cat(sprintf("      %s und %s sind bei k=%d Folds NICHT unterscheidbar.\n", winner, name, s$k))
  }
}
cat("\n  Befund: nur Unterschiede, deren korrigiertes paarweises Intervall 0 ausschliesst,\n")
cat("  sind bei dieser Fold-Zahl gestuetzt. Beide Intervalle sind Naeherungen.\n")
cat("  Deep Learning gewinnt auf kleinen klinischen Tabellendaten nicht automatisch.\n")
