# Module 28 – Survival ML with R (parallel to Python).
#   Rscript module/28-survival-ml/code/r.R
# Data: read from data/ (committed with the repo); if that folder is
# missing, the same files are fetched from the published URL.
# Packages: tidymodels, survival, survminer, randomForestSRC (optional).
# Code is English; dataset schema (column names) stays German.

suppressPackageStartupMessages({
  library(tidyverse)
  library(survival)
})

script <- normalizePath(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1]))
root   <- dirname(dirname(dirname(dirname(script))))
source(file.path(root, "lib", "helpers.R"))
set.seed(SEED)

# ---- Data -------------------------------------------------------------------
# Median-impute all numeric predictors with missingness (incl. bmi) instead of
# listwise-dropping rows – mirrors python.py's SimpleImputer(strategy="median").
data <- load_cohort() |>
  left_join(load_labs(), by = "patient_id") |>
  mutate(across(c(bmi, leukozyten_g_l, haemoglobin_g_dl, kreatinin_mg_dl,
                  laktat_mmol_l, natrium_mmol_l), ~ ifelse(is.na(.), median(., na.rm=TRUE), .)))

features <- c("alter", "sofa_score", "crp_mg_l", "bmi",
              "leukozyten_g_l", "kreatinin_mg_dl", "laktat_mmol_l",
              "diabetes", "hypertonie")

# 75/25 split (stratified by event)
idx_train <- sample(nrow(data), floor(0.75 * nrow(data)))
train <- data[idx_train, ]
test  <- data[-idx_train, ]

# ---- 1) Cox proportional hazards (baseline) ---------------------------------
cat("=== 1) Cox proportional hazards ===\n")
formula_str <- paste("Surv(fu_zeit_tage, status) ~",
                     paste(features, collapse = " + "))
cox_fit <- coxph(as.formula(formula_str), data = train, x = TRUE)
cat("Concordance (C-statistic, TRAIN, for reference only):",
    summary(cox_fit)$concordance[1], "\n")

# Risk scores on test set
cox_lp <- predict(cox_fit, newdata = test, type = "lp")  # linear predictor

# Test-set concordance – summary(cox_fit)$concordance above is computed on the
# TRAINING data the model was fit on and is optimistic; the number that matters
# for model comparison is out-of-sample, on the held-out test set.
cox_test_conc <- survival::concordance(cox_fit, newdata = test)
cat("Cox test C-statistic:", cox_test_conc$concordance, "\n")

# ---- 2) Random Survival Forest (randomForestSRC) ----------------------------
cat("\n=== 2) Random Survival Forest ===\n")
if (requireNamespace("randomForestSRC", quietly = TRUE)) {
  library(randomForestSRC)
  rsf_fit <- rfsrc(as.formula(formula_str), data = train,
                   ntree = 200, seed = SEED, importance = "permute")
  cat("RSF Variable Importance (top 5):\n")
  vi <- sort(rsf_fit$importance, decreasing = TRUE)
  print(head(vi, 5))
  rsf_pred <- predict(rsf_fit, newdata = test)
  rsf_scores <- rsf_pred$predicted  # higher = higher risk
  # reverse = TRUE: a higher score means higher risk, i.e. shorter survival.
  rsf_conc <- survival::concordance(Surv(fu_zeit_tage, status) ~ rsf_scores,
                                    data = test, reverse = TRUE)
  cat("\nRSF test C-statistic:", rsf_conc$concordance, "\n")
  have_rsf <- TRUE
} else {
  cat("randomForestSRC not installed. Install: install.packages('randomForestSRC')\n")
  cat("Using Cox linear predictor as the risk score for downstream analyses.\n")
  rsf_scores <- cox_lp
  have_rsf <- FALSE
}

# ---- 2b) Time-dependent AUC (parallel to python.py) -------------------------
# python.py teaches cumulative_dynamic_auc; the R track must teach the SAME
# estimand, not only Harrell's C-index (a single time-averaged number). timeROC
# gives the cumulative/dynamic AUC at each horizon WITH a confidence interval
# (iid = TRUE), so – as in python.py – we can judge whether an apparent
# Cox-vs-RSF gap is real or just sampling noise on a small test set.
require_pkgs("timeROC")   # aborts loudly (exit 1) if timeROC is not installed
cat("\n=== 2b) Time-dependent AUC (timeROC, with 95% CI) ===\n")
eval_times <- c(7, 14, 21, 28)
eval_times <- eval_times[eval_times < max(test$fu_zeit_tage)]
cat(sprintf("Test set: n = %d, events = %d\n", nrow(test), sum(test$status)))

td_auc_with_ci <- function(marker, label) {
  td <- timeROC::timeROC(T = test$fu_zeit_tage, delta = test$status,
                         marker = marker, cause = 1,
                         times = eval_times, iid = TRUE)
  ci <- confint(td, level = 0.95)$CI_AUC  # S3 method confint.ipcwsurvivalROC; percent scale
  # The normal-approximation CI can spill past [0, 1] with few events; clip it,
  # since an AUC is bounded – the width still shows how uncertain it is.
  ci <- pmin(pmax(ci / 100, 0), 1)
  for (i in seq_along(eval_times)) {
    cat(sprintf("  %-4s %2dd: AUC %.3f  (95%%-KI %.3f-%.3f)\n",
                label, eval_times[i], td$AUC[i], ci[i, 1], ci[i, 2]))
  }
}

td_auc_with_ci(cox_lp, "Cox")
if (have_rsf) {
  td_auc_with_ci(rsf_scores, "RSF")
  cat("Interpretation: overlapping CIs at a horizon mean the Cox/RSF gap there\n",
      "  is not distinguishable from noise (cf. python.py's difference CIs).\n", sep = "")
} else {
  cat("(RSF-AUC uebersprungen: randomForestSRC nicht installiert - nur Cox gezeigt.)\n")
}

# ---- 3) Kaplan-Meier by risk tertile ----------------------------------------
cat("\n=== 3) Kaplan-Meier curves by risk group ===\n")
# Tertile cut by score (not by outcome – that would be leakage).
breaks <- unique(quantile(rsf_scores, c(0, 1/3, 2/3, 1), na.rm = TRUE))
if (length(breaks) == 4) {
  test$risikogruppe <- cut(rsf_scores, breaks = breaks,
                           labels = c("Niedrig", "Mittel", "Hoch"),
                           include.lowest = TRUE)
} else {
  test$risikogruppe <- cut(rank(rsf_scores, ties.method = "first"),
                           breaks = 3,
                           labels = c("Niedrig", "Mittel", "Hoch"),
                           include.lowest = TRUE)
}
km_fit <- survfit(Surv(fu_zeit_tage, status) ~ risikogruppe, data = test)
cat("Kaplan-Meier summary by risk group:\n")
print(km_fit)

# Log-rank test for group separation.
lr_test <- survdiff(Surv(fu_zeit_tage, status) ~ risikogruppe, data = test)
cat("\nLog-rank test p-value:", 1 - pchisq(lr_test$chisq, df = 2), "\n")

# ---- 4) Competing risks (concept) -------------------------------------------
cat("\n=== 4) Competing risks – concept ===\n")
if (requireNamespace("cmprsk", quietly = TRUE)) {
  library(cmprsk)
  cat("cmprsk available. Example:\n")
  cat("  cuminc(ftime, fstatus, group)\n")
  cat("  where fstatus = 1 (event), 2 (competing event), 0 (censored)\n")
  cat("  Returns cumulative incidence function per group.\n")
} else {
  cat("cmprsk not installed. Install: install.packages('cmprsk')\n")
  cat("Concept: when a patient dies of another cause (competing event),\n")
  cat("  Kaplan-Meier overestimates cumulative incidence.\n")
  cat("  Aalen-Johansen estimator (cuminc) is the correct alternative.\n")
}

cat("\nKey reminder: use survival-aware models whenever follow-up varies or censoring exists.\n")
