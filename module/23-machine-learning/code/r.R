# Module 23 — Introduction to machine learning: first classifier (R / glm).
#
# Runs standalone from the project root:
#   Rscript module/23-machine-learning/code/r.R
#
# Data: read from data/ (committed with the repo); if that folder is
# missing, the same files are fetched from the published URL.
# Only base R + tidyverse is required; tidymodels sketch is shown but NOT run.
#
# Code is English (identifiers, comments).
# Dataset column names stay German (e.g. cohort$aufnahmegrund).
#
# NOTE ON PARITY WITH code/python.py: the Python pipeline uses
# class_weight="balanced" to counter the accuracy paradox at ~16 % mortality.
# Base R's glm() has no built-in class_weight argument, so we replicate the
# same idea manually via `weights=` (glm's case weights), using the identical
# formula scikit-learn uses for class_weight="balanced":
#   w_i = n_samples / (n_classes * n_samples_in_class(i))
# Without this, glm() silently reproduces the accuracy paradox this module
# warns about (see the confusion matrix below without weighting: verified by
# running this script with `weights = NULL` -- 9 of 16 deaths in the test set
# are missed, sensitivity ~44 %, far below the 81% the weighted Python model
# gets at the same threshold).

suppressPackageStartupMessages(library(tidyverse))

# ---------------------------------------------------------------------------
# Project root + helpers
# ---------------------------------------------------------------------------
script <- normalizePath(
  sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1])
)
root <- dirname(dirname(dirname(dirname(script))))
source(file.path(root, "lib", "helpers.R"))
set.seed(SEED)   # SEED = 42, defined in helpers.R

# ---------------------------------------------------------------------------
# Load and prepare data
# ---------------------------------------------------------------------------
cohort <- load_cohort() |>
  mutate(
    is_sepsis = as.integer(aufnahmegrund == "Sepsis"),
    is_smoker = as.integer(raucherstatus == "aktiv")
  )

cat("Cohort:", nrow(cohort), "patients\n")
cat("30-day mortality:", round(mean(cohort$verstorben_30d) * 100, 1), "%\n")
cat(
  "Accuracy paradox: always predicting 'lebt' gives",
  round((1 - mean(cohort$verstorben_30d)) * 100, 1),
  "% accuracy — and misses every death.\n\n"
)

# ---------------------------------------------------------------------------
# 1) Stratified train/test split (80 / 20)
# ---------------------------------------------------------------------------
cat("=================================================================\n")
cat("1) TRAIN / TEST SPLIT  (80 / 20, stratified, seed =", SEED, ")\n")
cat("=================================================================\n")

# Stratify manually: preserve event rate in both sets.
idx_pos   <- which(cohort$verstorben_30d == 1)
idx_neg   <- which(cohort$verstorben_30d == 0)
train_idx <- c(
  sample(idx_pos, size = floor(0.8 * length(idx_pos))),
  sample(idx_neg, size = floor(0.8 * length(idx_neg)))
)

train_df <- cohort[train_idx, ]
test_df  <- cohort[-train_idx, ]

cat("Training:", nrow(train_df), "| mortality:",
    round(mean(train_df$verstorben_30d) * 100, 1), "%\n")
cat("Test    :", nrow(test_df),  "| mortality:",
    round(mean(test_df$verstorben_30d)  * 100, 1), "%\n\n")

# ---------------------------------------------------------------------------
# 2) Logistic regression (glm, equivalent to baseline Pipeline in Python)
# ---------------------------------------------------------------------------
cat("=================================================================\n")
cat("2) MODEL  (logistic regression via glm)\n")
cat("=================================================================\n")

# Case weights == scikit-learn's class_weight="balanced": each class gets
# weight n / (2 * n_class), so the minority class (deaths) counts more.
n_train   <- nrow(train_df)
n_pos     <- sum(train_df$verstorben_30d == 1)
n_neg     <- sum(train_df$verstorben_30d == 0)
w_pos     <- n_train / (2 * n_pos)
w_neg     <- n_train / (2 * n_neg)
case_wts  <- ifelse(train_df$verstorben_30d == 1, w_pos, w_neg)
cat("Class weights (balanced): lebt =", round(w_neg, 3),
    " verstorben =", round(w_pos, 3), "\n")

# suppressWarnings(): non-integer case weights trigger R's harmless
# "non-integer #successes in a binomial glm!" notice (quasi-likelihood
# weighting is standard here; this is not a data or model problem).
fit <- suppressWarnings(glm(
  verstorben_30d ~ alter + sofa_score + crp_mg_l + diabetes + is_sepsis + is_smoker,
  data    = train_df,
  weights = case_wts,
  family  = binomial(link = "logit")
))

cat("\nCoefficients:\n")
print(round(coef(fit), 4))
cat(
  "\nNote: scale predictors before glm for comparable coefficient magnitudes.",
  "\nHere we skip that step to keep the code readable; Python uses StandardScaler",
  "\ninside a Pipeline so the scaler never sees test data.\n",
  "\nNote: case weights fix the ranking/threshold behaviour (accuracy paradox)",
  "\nbut, exactly like Python's class_weight='balanced', they also distort",
  "\npredict(type='response') as an absolute probability -- see Module 25 for",
  "\nrecalibration before trusting these probabilities as calibrated risk.\n"
)

# ---------------------------------------------------------------------------
# 3) Evaluation on the test set
# ---------------------------------------------------------------------------
cat("\n=================================================================\n")
cat("3) EVALUATION ON THE TEST SET\n")
cat("=================================================================\n")

probs <- predict(fit, newdata = test_df, type = "response")
preds <- as.integer(probs >= 0.5)

# Confusion matrix
cm <- table(Predicted = preds, Observed = test_df$verstorben_30d)
cat("\nConfusion matrix:\n")
print(cm)

tp <- sum(preds == 1 & test_df$verstorben_30d == 1)
fn <- sum(preds == 0 & test_df$verstorben_30d == 1)
tn <- sum(preds == 0 & test_df$verstorben_30d == 0)
fp <- sum(preds == 1 & test_df$verstorben_30d == 0)
sensitivity <- tp / (tp + fn)
specificity <- tn / (tn + fp)
cat(sprintf("\nSensitivity: %.2f   Specificity: %.2f  (threshold = 0.5, weighted glm)\n",
            sensitivity, specificity))

# AUC via Wilcoxon statistic — equivalent to the area under the ROC curve.
# No external package required: AUC = U / (n1 * n0).
u_stat <- wilcox.test(
  probs[test_df$verstorben_30d == 1],
  probs[test_df$verstorben_30d == 0]
)$statistic
n_pos <- sum(test_df$verstorben_30d == 1)
n_neg <- sum(test_df$verstorben_30d == 0)
auc   <- as.numeric(u_stat) / (n_pos * n_neg)
cat("\nROC-AUC (test, Wilcoxon method):", round(auc, 3), "\n")
cat(
  "\nNote: with only ~", n_pos, "events in the test set the 95 % CI of the AUC",
  "\nis wide. Draw no firm conclusions from a single split.\n"
)

# ---------------------------------------------------------------------------
# 4) Simple 5-fold cross-validation (on training data only)
# ---------------------------------------------------------------------------
cat("\n=================================================================\n")
cat("4) CROSS-VALIDATION  (5-fold, stratified, training data only)\n")
cat("=================================================================\n")

k          <- 5L
folds      <- rep(1:k, length.out = nrow(train_df))
fold_order <- c(
  sample(which(train_df$verstorben_30d == 1)),
  sample(which(train_df$verstorben_30d == 0))
)
fold_assignment <- integer(nrow(train_df))
fold_assignment[fold_order] <- folds

cv_auc <- numeric(k)
for (fold in seq_len(k)) {
  cv_train <- train_df[fold_assignment != fold, ]
  cv_val   <- train_df[fold_assignment == fold, ]

  cv_n_pos <- sum(cv_train$verstorben_30d == 1)
  cv_n_neg <- sum(cv_train$verstorben_30d == 0)
  cv_wts   <- ifelse(cv_train$verstorben_30d == 1,
                     nrow(cv_train) / (2 * cv_n_pos),
                     nrow(cv_train) / (2 * cv_n_neg))
  cv_fit   <- suppressWarnings(glm(
    verstorben_30d ~ alter + sofa_score + crp_mg_l + diabetes + is_sepsis + is_smoker,
    data = cv_train, weights = cv_wts, family = binomial
  ))
  cv_probs <- predict(cv_fit, newdata = cv_val, type = "response")

  n1 <- sum(cv_val$verstorben_30d == 1)
  n0 <- sum(cv_val$verstorben_30d == 0)
  if (n1 > 0 && n0 > 0) {
    u  <- wilcox.test(
      cv_probs[cv_val$verstorben_30d == 1],
      cv_probs[cv_val$verstorben_30d == 0]
    )$statistic
    cv_auc[fold] <- as.numeric(u) / (n1 * n0)
  } else {
    cv_auc[fold] <- NA_real_
  }
}

cat("\nAUC per fold:", round(cv_auc, 3), "\n")
cat("Mean CV-AUC :", round(mean(cv_auc, na.rm = TRUE), 3),
    "±", round(sd(cv_auc, na.rm = TRUE), 3), "\n")

# ---------------------------------------------------------------------------
# 5) tidymodels workflow sketch (commented — requires tidymodels package)
# ---------------------------------------------------------------------------
cat("\n=================================================================\n")
cat("5) TIDYMODELS SKETCH  (not executed — install.packages('tidymodels'))\n")
cat("=================================================================\n")
cat(
  "tidymodels mirrors scikit-learn's Pipeline + StratifiedKFold pattern.\n",
  "Preprocessing lives in a recipe(), so it is re-estimated per fold only.\n\n"
)
cat(
  "# library(tidymodels)\n",
  "# rec <- recipe(verstorben_30d ~ alter + sofa_score + crp_mg_l +\n",
  "#                diabetes + is_sepsis + is_smoker, data = train_df) |>\n",
  "#   step_normalize(all_numeric_predictors())\n",
  "# spec    <- logistic_reg() |> set_engine('glm')\n",
  "# wf      <- workflow() |> add_recipe(rec) |> add_model(spec)\n",
  "# folds   <- vfold_cv(train_df, v = 5, strata = verstorben_30d)\n",
  "# results <- fit_resamples(wf, resamples = folds,\n",
  "#                          metrics = metric_set(roc_auc))\n",
  "# collect_metrics(results)\n"
)

cat("\nDone.\n")
