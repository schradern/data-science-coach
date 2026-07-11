# Module 18 - mixed-effects models for longitudinal data.
#   Rscript module/18-longitudinal-mixed-models/code/r.R
#
# Uses the shared cohort's repeated MAP measurements (vitalwerte.csv, 1-4
# measurements per patient) merged with baseline characteristics
# (kohorte.csv) - the same data as code/python.py.

if (!requireNamespace("lme4", quietly = TRUE)) {
  message("Missing R package: lme4")
  message("Install with: install.packages('lme4')")
  quit(save = "no", status = 1)
}

suppressPackageStartupMessages(library(lme4))

script <- normalizePath(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1]))
root   <- dirname(dirname(dirname(dirname(script))))
source(file.path(root, "lib", "helpers.R"))
set.seed(SEED)

vitals <- load_vitals()
cohort <- load_cohort()[, c("patient_id", "sofa_score", "alter", "diabetes")]
data <- merge(vitals, cohort, by = "patient_id")

cat("=== Repeated MAP measurements (long format) ===\n")
print(head(data))
n_per_patient <- table(data$patient_id)
n_max <- max(n_per_patient)
n_incomplete <- sum(n_per_patient < n_max)
cat(sprintf(
  "Rows: %d | patients: %d | measurements per patient: %d-%d (%d patients have fewer than %d)\n",
  nrow(data), length(unique(data$patient_id)), min(n_per_patient), n_max,
  n_incomplete, n_max
))

# Break the incomplete series down by cause: early discharge (still alive --
# ordinary MAR/MCAR-style missingness, an LMM handles this fine) vs. death
# within the 30-day follow-up (the day-3/day-4 MAP does not exist for that
# patient -- truncation by death, not missingness; see README section 4).
death_info <- load_cohort()[, c("patient_id", "verweildauer_tage", "verstorben_30d")]
incomplete_ids <- names(n_per_patient)[n_per_patient < n_max]
incomplete <- death_info[death_info$patient_id %in% as.integer(incomplete_ids), ]
n_died <- sum(incomplete$verstorben_30d)
n_discharged <- n_incomplete - n_died
cat(sprintf(
  "  Of these %d incomplete series: %d are patients who survived (early discharge,\n",
  n_incomplete, n_discharged
))
cat("  verweildauer_tage < 4 -- ordinary MAR/MCAR-style missingness) and\n")
cat(sprintf(
  "  %d are patients who died within 30 days (verstorben_30d == 1 -- truncation by\n",
  n_died
))
cat("  death, NOT missingness; see the pitfall in the README).\n")

ols <- lm(map_mmhg ~ tag + diabetes, data = data)
mixed <- lmer(map_mmhg ~ tag + diabetes + (1 | patient_id), data = data, REML = TRUE)

ols_summary <- summary(ols)$coefficients[, c("Estimate", "Std. Error")]
mixed_summary <- summary(mixed)$coefficients[, c("Estimate", "Std. Error")]

cat("\n=== Naive OLS (treats every row as independent) ===\n")
print(round(ols_summary, 3))

cat("\n=== Random-intercept mixed model (accounts for repeated measures per patient) ===\n")
print(round(mixed_summary, 3))
var_comp <- as.data.frame(VarCorr(mixed))
group_var <- var_comp$vcov[var_comp$grp == "patient_id"]
resid_var <- var_comp$vcov[var_comp$grp == "Residual"]
cat(sprintf("Patient-level (random intercept) variance: %.3f | residual variance: %.3f | ICC = %.3f\n",
            group_var, resid_var, group_var / (group_var + resid_var)))

cat("\n=== Where the point estimates agree but the standard errors don't ===\n")
for (term in c("tag", "diabetes")) {
  kind <- if (term == "tag") "time-varying (changes within a patient)" else
    "patient-level (constant across a patient's 1-4 rows)"
  ratio <- mixed_summary[term, "Std. Error"] / ols_summary[term, "Std. Error"]
  cat(sprintf("%-10s [%s]:\n", term, kind))
  cat(sprintf("    OLS coef=% .3f se=%.3f | mixed coef=% .3f se=%.3f (mixed/OLS SE ratio = %.2fx)\n",
              ols_summary[term, "Estimate"], ols_summary[term, "Std. Error"],
              mixed_summary[term, "Estimate"], mixed_summary[term, "Std. Error"], ratio))
}
cat("\nInterpretation: for the patient-level predictor diabetes, every row of the same patient\n")
cat("repeats the same value. OLS treats those repeats as independent new evidence\n")
cat("(pseudo-replication), so its SE for diabetes is too small - the mixed model corrects this.\n")
cat("For the time-varying predictor tag, the naive SE is not automatically too small; whether\n")
cat("OLS over- or understates it depends on how within-patient the variation is. Either way,\n")
cat("the mixed model's SE is the trustworthy one because it reflects the correct error structure.\n")
cat("\nEstimand caveat: this model is fit on whoever HAS a MAP measurement on a given day. For\n")
cat("patients who died before day 3, that measurement does not exist -- so what the model\n")
cat("actually estimates is the mean MAP trajectory among patients still alive and in hospital,\n")
cat("not among everyone admitted. No missing-data assumption (MCAR/MAR) fixes that; it is a\n")
cat("truncation-by-death problem, not a missingness problem.\n")

cat("\n=== Random slope: does the MAP trend over time also vary by patient? ===\n")
mixed_slope <- suppressWarnings(
  lmer(map_mmhg ~ tag + diabetes + (tag | patient_id), data = data, REML = TRUE)
)
if (isSingular(mixed_slope, tol = 1e-4)) {
  cat("Random-slope covariance is singular (boundary fit): with only up to 4 measurements per\n")
  cat("patient, the data don't support estimating a patient-specific day-trend on top of the\n")
  cat("random intercept - this is the 'Singular Fit' pitfall below, seen live. The simpler\n")
  cat("random-intercept-only model above is the right choice here.\n")
} else {
  cat("Random-slope model converged without a singular-fit warning.\n")
}
