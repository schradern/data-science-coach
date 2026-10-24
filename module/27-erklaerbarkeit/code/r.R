# Module 27 – Explainable AI with R (parallel to Python).
#   Rscript module/27-erklaerbarkeit/code/r.R
# Data: read from data/ (committed with the repo); if that folder is
# missing, the same files are fetched from the published URL.
# Packages: tidymodels, ranger, vip, pdp (install if missing).
# Code is English; dataset schema (column names) stays German.

suppressPackageStartupMessages({
  missing_pkgs <- setdiff(c("tidyverse", "tidymodels", "ranger"), rownames(installed.packages()))
  if (length(missing_pkgs) > 0) {
    message("Missing R packages: ", paste(missing_pkgs, collapse = ", "))
    message("This lesson cannot run without them; nothing was computed.")
    message("Full per-module package list: DEPENDENCIES-R.md")
    message("Install with: install.packages(c('tidyverse', 'tidymodels', 'ranger'))")
    quit(save = "no", status = 1)
  }
  library(tidyverse)
  library(tidymodels)
})

script <- normalizePath(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1]))
root   <- dirname(dirname(dirname(dirname(script))))
source(file.path(root, "lib", "helpers.R"))
set.seed(SEED)

# ---- Data -------------------------------------------------------------------
data <- load_cohort() |>
  left_join(load_labs(), by = "patient_id") |>
  mutate(verstorben_30d = factor(verstorben_30d, levels = c(1, 0)))

split  <- initial_split(data, prop = 0.75, strata = verstorben_30d)
train  <- training(split)
test   <- testing(split)

# ---- Recipe + Model ---------------------------------------------------------
rec <- recipe(verstorben_30d ~ alter + sofa_score + crp_mg_l + bmi +
                leukozyten_g_l + kreatinin_mg_dl + laktat_mmol_l +
                aufnahmegrund + raucherstatus + diabetes + hypertonie,
              data = train) |>
  step_impute_median(all_numeric_predictors()) |>
  step_dummy(all_nominal_predictors()) |>
  step_normalize(all_numeric_predictors())

# Use ranger (Random Forest) – supports both permutation importance and PDPs.
spec <- rand_forest(trees = 300, mtry = 4) |>
  set_engine("ranger", importance = "permutation", seed = SEED) |>
  set_mode("classification")

wf   <- workflow() |> add_recipe(rec) |> add_model(spec)
fit  <- wf |> fit(data = train)

# ---- 1) AUC on test set -----------------------------------------------------
preds <- augment(fit, new_data = test)
cat("\nTest AUC:\n")
print(roc_auc(preds, truth = verstorben_30d, .pred_1))

# ---- 2) Permutation Importance (vip package) --------------------------------
if (requireNamespace("vip", quietly = TRUE)) {
  library(vip)
  cat("\n--- Permutation Importance (ranger, permutation method) ---\n")
  # Extract the fitted ranger model from the workflow.
  ranger_fit <- extract_fit_parsnip(fit)$fit
  # vip uses the importance stored in the ranger object directly.
  print(vi(ranger_fit) |> slice_head(n = 10))
} else {
  cat("\nvip package not installed. Install with: install.packages('vip')\n")
  cat("Permutation importance is computed inside ranger when importance='permutation'.\n")
}

# ---- 3) Partial Dependence (pdp package) ------------------------------------
if (requireNamespace("pdp", quietly = TRUE)) {
  library(pdp)
  cat("\n--- Partial Dependence: sofa_score ---\n")
  # Use the ranger model directly via pdp::partial().
  ranger_fit <- extract_fit_parsnip(fit)$fit
  baked_test  <- rec |> prep() |> bake(new_data = test)
  pd_sofa <- partial(ranger_fit, pred.var = "sofa_score",
                     train = baked_test, type = "classification",
                     prob = TRUE, which.class = 1)
  print(pd_sofa)
} else {
  cat("\npdp package not installed. Install with: install.packages('pdp')\n")
  cat("Concept: vary one feature across its range while averaging over all others.\n")
}

# ---- 4) SHAP (DALEX / DALEXtra) --------------------------------------------
if (requireNamespace("DALEXtra", quietly = TRUE)) {
  library(DALEX)
  library(DALEXtra)
  cat("\n--- SHAP (via DALEX Shapley) ---\n")
  explainer <- explain_tidymodels(
    fit, data = select(test, -verstorben_30d),
    y = as.numeric(test$verstorben_30d == "1"),
    label = "RandomForest", verbose = FALSE
  )
  # Shapley values for the first test patient (one local explanation).
  shap_vals <- predict_parts(explainer, new_observation = test[1, ],
                              type = "shap", B = 15)
  print(shap_vals)
} else {
  cat("\nDALEXtra not installed. Install: install.packages(c('DALEX','DALEXtra'))\n")
  cat("SHAP concept: each feature receives a contribution to the individual prediction.\n")
}

cat("\nKey reminder: importance ≠ causation. Validate clinically.\n")
