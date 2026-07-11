# Module 14 - Missing data: when complete-case is safe, and when it is not.
#   Rscript module/14-fehlende-werte/code/r.R
# Data: read from data/ (committed with the repo); if that folder is
# missing, the same files are fetched from the published URL.
#
# Mirrors code/python.py: same estimands, same numbers.
#   laktat_mmol_l  missing depends on sofa_score (a covariate IN the model)
#                  -> complete-case coefficients unbiased, marginal mean not.
#   bga_ph         missing depends on verstorben_30d (the OUTCOME)
#                  -> odds ratios unbiased, intercept and absolute risk are not.

script <- normalizePath(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1]))
root <- dirname(dirname(dirname(dirname(script))))
source(file.path(root, "lib", "helpers.R"))

require_pkgs("tidyverse", "mice")
suppressPackageStartupMessages({
  library(tidyverse)
  library(mice)
})
set.seed(SEED)

MODEL <- verstorben_30d ~ bga_ph + alter + sofa_score
REF <- list(bga_ph = 7.38, alter = 64, sofa_score = 4)

df <- left_join(load_cohort(), load_labs(), by = "patient_id") |>
  left_join(readr::read_csv(file.path(root, "data", "bga_ph_wahrheit.csv"),
                            show_col_types = FALSE), by = "patient_id")

predicted_risk <- function(cf) {
  z <- cf[["(Intercept)"]] + cf[["bga_ph"]] * REF$bga_ph +
    cf[["alter"]] * REF$alter + cf[["sofa_score"]] * REF$sofa_score
  1 / (1 + exp(-z))
}
or_per_01 <- function(cf) exp(-0.1 * cf[["bga_ph"]])

# ---------------------------------------------------------------------------
cat("\n=== 1) Missingness profile — who is missing, not just how many ===\n")
for (v in c("bmi", "laktat_mmol_l", "bga_ph")) {
  cat(sprintf("  %-16s %5.1f%% missing\n", v, 100 * mean(is.na(df[[v]]))))
}

work <- df |> mutate(laktat_fehlt = as.integer(is.na(laktat_mmol_l)),
                     bga_fehlt    = as.integer(is.na(bga_ph)))
for (pair in list(c("laktat_mmol_l", "laktat_fehlt"), c("bga_ph", "bga_fehlt"))) {
  f <- glm(as.formula(paste(pair[2], "~ sofa_score + verstorben_30d")),
           data = work, family = binomial)
  s <- summary(f)$coefficients
  cat(sprintf("  %-14s sofa b=%+.3f (p=%.4f)   outcome b=%+.3f (p=%.4f)\n",
              pair[1], s["sofa_score", 1], s["sofa_score", 4],
              s["verstorben_30d", 1], s["verstorben_30d", 4]))
}
cat("  -> lactate is missing by SEVERITY; the blood gas is missing by OUTCOME.\n")

# ---------------------------------------------------------------------------
cat("\n=== 2) The cohort you keep is not the cohort you had ===\n")
cc_df <- df |> drop_na(bga_ph)
cat(sprintf("  full cohort           N=%3d   30-day mortality %.1f%%\n",
            nrow(df), 100 * mean(df$verstorben_30d)))
cat(sprintf("  complete cases (pH)   N=%3d   30-day mortality %.1f%%\n",
            nrow(cc_df), 100 * mean(cc_df$verstorben_30d)))

# ---------------------------------------------------------------------------
cat("\n=== 3) Complete-case: what survives, what breaks ===\n")
truth_fit <- glm(MODEL, data = df |> mutate(bga_ph = bga_ph_wahr), family = binomial)
cc_fit    <- glm(MODEL, data = cc_df, family = binomial)

cat(sprintf("  %-22s %10s %14s %12s\n", "", "Intercept", "OR / -0.1 pH", "risk @ ref"))
for (nm in c("full data (truth)", "complete-case")) {
  cf <- coef(if (nm == "full data (truth)") truth_fit else cc_fit)
  cat(sprintf("  %-22s %10.2f %14.2f %12.3f\n",
              nm, cf[["(Intercept)"]], or_per_01(cf), predicted_risk(cf)))
}
r_true <- predicted_risk(coef(truth_fit)); r_cc <- predicted_risk(coef(cc_fit))
cat(sprintf("\n  absolute risk: %.3f vs %.3f  (%+.0f %%)\n", r_cc, r_true, 100 * (r_cc / r_true - 1)))
cat("  Selecting on the OUTCOME shifts a logistic intercept, not its slopes.\n")

# ---------------------------------------------------------------------------
cat("\n=== 4) Contrast: lactate is missing by covariate, not by outcome ===\n")
lac <- glm(verstorben_30d ~ laktat_mmol_l + alter + sofa_score,
           data = df |> drop_na(laktat_mmol_l), family = binomial)
ci <- confint.default(lac)["laktat_mmol_l", ]
cat(sprintf("  complete-case b(laktat) = %+.3f   (95%% CI %+.3f to %+.3f)\n",
            coef(lac)[["laktat_mmol_l"]], ci[1], ci[2]))
cat("  The driver (sofa_score) is IN the model -> this coefficient is unbiased.\n")
cat(sprintf("  But the marginal mean is not: observed lactate = %.2f mmol/l\n",
            mean(df$laktat_mmol_l, na.rm = TRUE)))

# ---------------------------------------------------------------------------
cat("\n=== 5) Median imputation attenuates. Multiple imputation does not. ===\n")
med_fit <- glm(MODEL, family = binomial,
               data = df |> mutate(bga_ph = replace_na(bga_ph, median(bga_ph, na.rm = TRUE))))

# The imputation model MUST contain the outcome: that is what makes this MAR.
imp_data <- df |> select(verstorben_30d, bga_ph, alter, sofa_score)
imp <- mice(imp_data, m = 20, method = "pmm", seed = SEED, printFlag = FALSE)
mi <- pool(with(imp, glm(verstorben_30d ~ bga_ph + alter + sofa_score, family = binomial)))
mi_sum <- summary(mi)
mi_cf <- setNames(mi_sum$estimate, mi_sum$term)
mi_se <- setNames(mi_sum$std.error, mi_sum$term)

cat(sprintf("  %-28s %11s %7s %14s %12s\n", "", "b(bga_ph)", "SE", "OR / -0.1 pH", "risk @ ref"))
rows <- list(
  list("full data (truth)", coef(truth_fit), summary(truth_fit)$coefficients[, 2]),
  list("complete-case", coef(cc_fit), summary(cc_fit)$coefficients[, 2]),
  list("median imputation", coef(med_fit), summary(med_fit)$coefficients[, 2]),
  list("multiple imputation (m=20)", mi_cf, mi_se)
)
for (r in rows) {
  cf <- r[[2]]; se <- r[[3]]
  cat(sprintf("  %-28s %11.2f %7.2f %14.2f %12.3f\n",
              r[[1]], cf[["bga_ph"]], se[["bga_ph"]], or_per_01(cf), predicted_risk(cf)))
}
cat("\n  Absolute risk, relative error against the truth:\n")
for (r in rows[-1]) {
  cat(sprintf("    %-28s %+5.0f %%\n", r[[1]], 100 * (predicted_risk(r[[2]]) / r_true - 1)))
}
cat("\n  Median imputation shrinks the effect toward zero and reports a standard\n")
cat("  error it has not earned. Rubin's rules add the between-imputation variance.\n")
