# Module 26 — tree ensembles and gradient boosting (R, parallel to Python).
#   Rscript module/26-ensembles-boosting/code/r.R
# Data: read from data/ (committed with the repo); if that folder is
# missing, the same files are fetched from the published URL.
# Code is English; the dataset schema (column names) stays German.

suppressPackageStartupMessages({
  missing_pkgs <- setdiff(c("tidyverse", "tidymodels"), rownames(installed.packages()))
  if (length(missing_pkgs) > 0) {
    message("Missing R packages: ", paste(missing_pkgs, collapse = ", "))
    message("This lesson cannot run without them; nothing was computed.")
    message("Full per-module package list: DEPENDENCIES-R.md")
    message("Install with: install.packages(c('tidyverse', 'tidymodels'))")
    quit(save = "no", status = 1)
  }
  library(tidyverse)
  library(tidymodels)
})

script <- normalizePath(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1]))
root   <- dirname(dirname(dirname(dirname(script))))
source(file.path(root, "lib", "helpers.R"))
set.seed(SEED)

data <- load_cohort() |>
  left_join(load_labs(), by = "patient_id") |>
  mutate(verstorben_30d = factor(verstorben_30d, levels = c(1, 0)))

split  <- initial_split(data, prop = 0.75, strata = verstorben_30d)
train  <- training(split)
folds  <- vfold_cv(train, v = 5, strata = verstorben_30d)

rec <- recipe(verstorben_30d ~ alter + sofa_score + crp_mg_l + bmi +
                leukozyten_g_l + kreatinin_mg_dl + laktat_mmol_l +
                aufnahmegrund + raucherstatus + diabetes + hypertonie,
              data = train) |>
  step_impute_median(all_numeric_predictors()) |>
  step_dummy(all_nominal_predictors()) |>
  step_zv(all_predictors())

# --- 1) Decision tree (single) -----------------------------------------------
tree_spec <- decision_tree(tree_depth = 4, mode = "classification") |>
  set_engine("rpart")

tree_wf  <- workflow() |> add_recipe(rec) |> add_model(tree_spec)
tree_res <- fit_resamples(tree_wf, folds, metrics = metric_set(roc_auc))
cat("Entscheidungsbaum (Tiefe 4):\n")
print(collect_metrics(tree_res))

# --- 2) Random Forest --------------------------------------------------------
# Balanced class weights computed FROM THE DATA, mirroring sklearn's
# class_weight="balanced" (weight_c = n / (n_classes * n_c)). A hardcoded 10:1
# neither matches the ~84/16 base rate (~5.4:1) nor the Python side, so the two
# forests would not be comparable. table() is in factor-level order c(1, 0),
# and ranger reads class.weights in factor-level order.
class_counts  <- table(train$verstorben_30d)
class_weights <- as.numeric(nrow(train) / (2 * class_counts))

rf_spec <- rand_forest(trees = 300, mtry = tune(), min_n = 5,
                       mode = "classification") |>
  set_engine("ranger", class.weights = class_weights,
             seed = SEED)

rf_wf  <- workflow() |> add_recipe(rec) |> add_model(rf_spec)
rf_grid <- tibble(mtry = c(3, 5, 7))
rf_res  <- tune_grid(rf_wf, resamples = folds, grid = rf_grid,
                     metrics = metric_set(roc_auc))
cat("\nRandom Forest (beste mtry):\n")
print(show_best(rf_res, metric = "roc_auc"))

# --- 3) Gradient Boosting (xgboost if available) ----------------------------
if (requireNamespace("xgboost", quietly = TRUE)) {
  xgb_spec <- boost_tree(trees = 300, tree_depth = 5, learn_rate = 0.05,
                         mode = "classification") |>
    set_engine("xgboost")

  xgb_wf  <- workflow() |> add_recipe(rec) |> add_model(xgb_spec)
  xgb_res <- fit_resamples(xgb_wf, folds, metrics = metric_set(roc_auc))
  cat("\nGradient Boosting (XGBoost):\n")
  print(collect_metrics(xgb_res))
} else {
  cat("\nxgboost nicht installiert. Installieren mit: install.packages('xgboost')\n")
}
