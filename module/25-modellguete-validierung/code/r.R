# Module 25 — model quality and clinical validation (R, parallel to Python).
#   Rscript module/25-modellguete-validierung/code/r.R
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
test   <- testing(split)

rec <- recipe(verstorben_30d ~ alter + sofa_score + crp_mg_l + bmi +
                leukozyten_g_l + kreatinin_mg_dl + laktat_mmol_l +
                aufnahmegrund + raucherstatus + diabetes + hypertonie,
              data = train) |>
  step_impute_median(all_numeric_predictors()) |>
  step_dummy(all_nominal_predictors()) |>
  step_normalize(all_numeric_predictors())

spec <- logistic_reg() |> set_engine("glm")
wf   <- workflow() |> add_recipe(rec) |> add_model(spec)
fit  <- wf |> fit(train)

# Predictions on test set
proba <- predict(fit, test, type = "prob")$.pred_1
y_test <- as.integer(as.character(test$verstorben_30d))

# --- 1) Discrimination -------------------------------------------------------
# Step-wise average precision (PR-AUC), matching sklearn's definition:
# AP = sum (recall_n - recall_{n-1}) * precision_n.
average_precision <- function(y, score) {
  ord <- order(score, decreasing = TRUE)
  y   <- y[ord]
  tp  <- cumsum(y)
  fp  <- cumsum(1 - y)
  precision <- tp / (tp + fp)
  recall    <- tp / sum(y)
  sum(c(recall[1], diff(recall)) * precision)
}

# 95%-bootstrap percentile CI for a ranking metric. On n=125 with ~19 events
# the point estimate alone is far too confident, so we report the interval too.
bootstrap_metric_ci <- function(y, score, metric_fn, n_boot = 2000) {
  n <- length(y)
  vals <- numeric(0)
  for (b in seq_len(n_boot)) {
    idx <- sample.int(n, n, replace = TRUE)
    yb  <- y[idx]
    if (length(unique(yb)) < 2) next  # undefined metric on a one-class resample
    vals <- c(vals, metric_fn(yb, score[idx]))
  }
  stats::quantile(vals, c(0.025, 0.975), names = FALSE)
}

n_events <- sum(y_test)
cat(sprintf("Testset: n=%d, Ereignisse=%d\n", length(y_test), n_events))

# pROC for ROC-AUC + DeLong 95%-CI
if (requireNamespace("pROC", quietly = TRUE)) {
  roc_obj <- pROC::roc(y_test, proba, quiet = TRUE, direction = "<")
  auc_ci  <- as.numeric(pROC::ci.auc(roc_obj))  # c(lower, auc, upper), DeLong
  cat(sprintf("ROC-AUC: %.3f  (95%%-DeLong-KI %.3f-%.3f)\n",
              auc_ci[2], auc_ci[1], auc_ci[3]))
} else {
  ci <- bootstrap_metric_ci(y_test, proba, function(y, s) {
    # rank-based AUC (Mann-Whitney U) so we still get a CI without pROC
    r <- rank(s); n1 <- sum(y == 1); n0 <- sum(y == 0)
    (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
  })
  cat(sprintf("ROC-AUC 95%%-Bootstrap-KI: %.3f-%.3f (pROC fuer Punktschaetzer installieren)\n",
              ci[1], ci[2]))
}

# PR-AUC point estimate + bootstrap 95%-CI
pr_auc <- average_precision(y_test, proba)
pr_ci  <- bootstrap_metric_ci(y_test, proba, average_precision)
cat(sprintf("PR-AUC:  %.3f  (95%%-Bootstrap-KI %.3f-%.3f, Baseline = %.3f)\n",
            pr_auc, pr_ci[1], pr_ci[2], mean(y_test)))

# --- 2) Calibration ----------------------------------------------------------
# NOTE: the recipe/glmnet spec above has NO class weighting, so `proba` here
# is already reasonably calibrated. The Python script uses
# class_weight="balanced" (for consistent ranking behaviour with modules
# 18/19) and therefore recalibrates with CalibratedClassifierCV BEFORE
# computing Brier/CITL/slope/DCA -- see code/python.py for that step and why
# it is necessary (class_weight="balanced" inflates predicted risk).
brier <- mean((proba - y_test)^2)
cat(sprintf("Brier Score: %.4f\n", brier))

log_odds <- log(pmax(proba, 1e-6) / pmax(1 - proba, 1e-6))

# Calibration slope: log_odds as a freely estimated covariate.
slope_mod <- glm(y_test ~ log_odds, family = binomial)
cat(sprintf("Calibration Slope:        %.3f\n", coef(slope_mod)[2]))

# Calibration-in-the-large: log_odds as an OFFSET (coefficient fixed at 1),
# intercept-only model -- this is what actually matches the textbook
# definition (Steyerberg / val.prob) and is what code/python.py now computes
# via statsmodels' GLM(..., offset=log_odds). Do NOT use glm(y ~ 1) alone --
# that ignores `proba` entirely and always returns logit(mean(y)).
citl_mod <- glm(y_test ~ offset(log_odds), family = binomial)
cat(sprintf("Calibration-in-the-large: %.3f\n", coef(citl_mod)[1]))

# --- 3) Net benefit (simple manual loop) -------------------------------------
base_rate <- mean(y_test)
thresholds <- seq(0.05, 0.45, by = 0.10)
cat("\nDecision-Curve-Analyse (Auszug):\n")
cat(sprintf("  %8s  %10s  %10s\n", "Schwelle", "Modell NB", "Alle NB"))
for (t in thresholds) {
  pos     <- as.integer(proba >= t)
  tp      <- sum(pos == 1 & y_test == 1)
  fp      <- sum(pos == 1 & y_test == 0)
  n       <- length(y_test)
  nb_mod  <- tp / n - (t / (1 - t)) * fp / n
  nb_all  <- base_rate - (t / (1 - t)) * (1 - base_rate)
  cat(sprintf("  %8.2f  %10.4f  %10.4f\n", t, nb_mod, nb_all))
}
cat("\nHinweis: Paket 'dcurves' liefert vollstaendige DCA-Plots.\n")
