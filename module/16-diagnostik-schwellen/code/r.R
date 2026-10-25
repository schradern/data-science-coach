# Module 16 - Diagnostic accuracy, threshold choice, and net benefit.

script <- normalizePath(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1]))
root <- dirname(dirname(dirname(dirname(script))))
source(file.path(root, "lib", "helpers.R"))

suppressPackageStartupMessages(library(tidyverse))
require_pkgs("pROC")
suppressPackageStartupMessages(library(pROC))

metrics_at_threshold <- function(y, score, threshold) {
  pred <- score >= threshold
  tp <- sum(pred & y == 1)
  fp <- sum(pred & y == 0)
  fn <- sum(!pred & y == 1)
  tn <- sum(!pred & y == 0)
  sens <- tp / (tp + fn)
  spec <- tn / (tn + fp)
  ppv <- tp / (tp + fp)
  npv <- tn / (tn + fn)
  tibble(threshold, TP = tp, FP = fp, FN = fn, TN = tn,
         sensitivity = sens, specificity = spec, PPV = ppv, NPV = npv,
         `LR+` = sens / (1 - spec), `LR-` = (1 - sens) / spec)
}

# Net benefit of a model at threshold probability pt (Vickers & Elkin 2006):
#   NB = TP/N - FP/N * pt/(1-pt), "treat" when prob >= pt.
net_benefit <- function(y, prob, pt) {
  n <- length(y)
  pred <- prob >= pt
  tp <- sum(pred & y == 1)
  fp <- sum(pred & y == 0)
  tp / n - fp / n * (pt / (1 - pt))
}
# Net benefit of "treat everyone".
net_benefit_all <- function(prevalence, pt) prevalence - (1 - prevalence) * (pt / (1 - pt))

df <- left_join(load_cohort(), load_labs(), by = "patient_id") |>
  drop_na(laktat_mmol_l)
y <- df$verstorben_30d
score <- df$laktat_mmol_l

# ---------------------------------------------------------------------------
# 1) ROC, AUC and Youden threshold
# ---------------------------------------------------------------------------
cat("\n1) ROC and Youden threshold\n")
roc_obj <- roc(y, score, quiet = TRUE, direction = "<")
auc_val <- as.numeric(auc(roc_obj))

# Youden index computed at the OBSERVED cutoffs with the same convention as
# scikit-learn (pred = score >= t), so the reported threshold matches Python's.
# (pROC's coords("best") reports the midpoint between adjacent values instead.)
cutoffs <- sort(unique(score))
youden_j <- vapply(cutoffs, function(t) {
  pred <- score >= t
  sens <- sum(pred & y == 1) / sum(y == 1)
  spec <- sum(!pred & y == 0) / sum(y == 0)
  sens + spec - 1
}, numeric(1))
youden <- cutoffs[which.max(youden_j)]
pred_y <- score >= youden
sens_y <- sum(pred_y & y == 1) / sum(y == 1)
spec_y <- sum(!pred_y & y == 0) / sum(y == 0)

# 95% CI for the AUC: DeLong (pROC's default); python.py bootstraps instead,
# so the two intervals differ in the last digit.
auc_ci <- as.numeric(ci.auc(roc_obj))
# Wilson intervals for sensitivity/specificity (prop.test without correction)
wilson <- function(k, n) sprintf("[%.3f, %.3f]", prop.test(k, n, correct = FALSE)$conf.int[1],
                                  prop.test(k, n, correct = FALSE)$conf.int[2])
cat(sprintf("AUC=%.3f  95%% CI (DeLong) [%.3f, %.3f]\n", auc_val, auc_ci[1], auc_ci[3]))
cat(sprintf("Youden threshold=%.2f, sensitivity=%.3f %s, specificity=%.3f %s  (Wilson 95%% CI)\n",
            youden, sens_y, wilson(sum(pred_y & y == 1), sum(y == 1)),
            spec_y, wilson(sum(!pred_y & y == 0), sum(y == 0))))
cat("NOTE: the Youden threshold here is chosen AND evaluated on the same data\n")
cat("(in-sample) – optimistic. Validate on held-out data (see modules 24/25).\n")

# ---------------------------------------------------------------------------
# 2) Metrics at clinically simple thresholds
# ---------------------------------------------------------------------------
cat("\n2) Metrics at clinically simple thresholds\n")
res <- bind_rows(lapply(c(1.5, 2.0, 2.5, youden), function(t) {
  metrics_at_threshold(y, score, t)
}))
print(as.data.frame(round(res, 3)))

# ---------------------------------------------------------------------------
# 3) Prevalence
# ---------------------------------------------------------------------------
cat("\n3) Prevalence\n")
prevalence <- mean(y)
cat(sprintf("Observed event rate=%.3f\n", prevalence))

# ---------------------------------------------------------------------------
# 4) Net benefit / decision-curve analysis (Vickers & Elkin 2006)
# ---------------------------------------------------------------------------
# Youden gives ONE threshold on the raw score and ignores prevalence and the
# relative cost of false positives vs. false negatives. A decision curve encodes
# exactly that trade-off in the threshold probability pt. We need predicted
# PROBABILITIES, so we fit a simple logistic model of the outcome on laktat.
cat("\n4) Net benefit (decision-curve analysis)\n")
model <- glm(verstorben_30d ~ laktat_mmol_l, data = df, family = binomial)
prob <- predict(model, type = "response")

cat("  Net benefit at selected threshold probabilities:\n")
cat("   pt   NB_model  NB_all   NB_none\n")
for (pt in c(0.10, 0.15, 0.20, 0.25, 0.30)) {
  cat(sprintf("  %.2f   %+.4f  %+.4f  0.0000\n",
              pt, net_benefit(y, prob, pt), net_benefit_all(prevalence, pt)))
}
cat("  Reading: where NB_model is above BOTH treat-all and treat-none, the\n")
cat("  model adds decision value at that threshold probability.\n")
cat("  (The styled decision-curve figure is produced by code/python.py.)\n")
