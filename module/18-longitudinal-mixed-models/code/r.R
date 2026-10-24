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

# Break the incomplete series down by what actually ended them. A series is
# cut short by death only if the patient died inside the measurement window;
# a patient who left on day 2 and died on day 22 has an ordinary
# early-discharge gap, not truncation by death.
death_info <- load_cohort()[, c("patient_id", "verweildauer_tage", "fu_zeit_tage", "status")]
incomplete_ids <- as.integer(names(n_per_patient)[n_per_patient < n_max])
incomplete <- death_info[death_info$patient_id %in% incomplete_ids, ]
died <- incomplete[incomplete$status == 1, ]
in_window <- died$fu_zeit_tage < n_max
cat(sprintf(
  "  %d of these %d series belong to a short stay (verweildauer_tage < %d); %d of the patients died within 30 days,\n",
  sum(incomplete$verweildauer_tage < n_max), n_incomplete, n_max, nrow(died)
))
cat(sprintf(
  "  but only %d died inside the measurement window (death days of the others: %s).\n",
  sum(in_window), paste(sort(died$fu_zeit_tage[!in_window]), collapse = ", ")
))
cat("  The vital-sign series here are simulated independently of the death time, so they cannot\n")
cat("  serve as evidence of truncation by death; the README explains the concept.\n")

# Plausibility check: a measurement dated after the recorded death day. Real
# records contain such contradictions too (wrong date, late documentation, a
# mistyped death date) - find them and clarify them, do not silently delete.
checked <- merge(vitals, death_info, by = "patient_id")
after <- checked[checked$status == 1 & checked$tag > checked$fu_zeit_tage,
                 c("patient_id", "tag", "fu_zeit_tage")]
names(after)[3] <- "death_day"
cat(sprintf("\nPlausibility check: %d measurements in %d patients are dated after the recorded death day:\n",
            nrow(after), length(unique(after$patient_id))))
print(after[order(after$patient_id, after$tag), ], row.names = FALSE)

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
cat("\nEstimand caveat: this model is fit on whoever HAS a MAP measurement on a given day.\n")
cat("In real ICU data, a patient who dies on day 2 has no day-3 MAP at all -- the model\n")
cat("then estimates the mean MAP trajectory among patients still alive and in hospital,\n")
cat("not among everyone admitted. No missing-data assumption (MCAR/MAR) fixes that; it is\n")
cat("truncation by death. This toy dataset does not simulate it (see above).\n")

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
