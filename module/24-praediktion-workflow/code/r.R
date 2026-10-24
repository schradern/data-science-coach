# Module 24 – prediction workflow with tidymodels (R, parallel to Python).
#   Rscript module/24-praediktion-workflow/code/r.R
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
root <- dirname(dirname(dirname(dirname(script))))
source(file.path(root, "lib", "helpers.R"))
set.seed(SEED)

data <- load_cohort() |>
  left_join(load_labs(), by = "patient_id") |>
  mutate(verstorben_30d = factor(verstorben_30d, levels = c(1, 0)))  # event first

# Split + resampling
split <- initial_split(data, prop = 0.75, strata = verstorben_30d)
train <- training(split)
folds <- vfold_cv(train, v = 5, strata = verstorben_30d)

# Recipe: imputation, dummies and scaling live in the recipe, so they are
# re-estimated on each fold only -> no leakage.
rec <- recipe(verstorben_30d ~ alter + sofa_score + crp_mg_l + bmi +
                leukozyten_g_l + kreatinin_mg_dl + laktat_mmol_l +
                aufnahmegrund + raucherstatus + diabetes + hypertonie,
              data = train) |>
  step_impute_median(all_numeric_predictors()) |>
  step_dummy(all_nominal_predictors()) |>
  step_normalize(all_numeric_predictors())

spec <- logistic_reg(penalty = tune(), mixture = 0) |>
  set_engine("glmnet")

wf <- workflow() |> add_recipe(rec) |> add_model(spec)

tuned <- tune_grid(wf, resamples = folds,
                   grid = tibble(penalty = 10^seq(-3, 1, length.out = 6)),
                   metrics = metric_set(roc_auc))

best <- select_best(tuned, metric = "roc_auc")
final <- finalize_workflow(wf, best) |> last_fit(split)
cat("held-out test AUC:\n")
print(collect_metrics(final))
