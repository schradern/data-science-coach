# Module 12 — Regression models (R / stats + survival)
#
# Runs standalone from the project root:
#   Rscript module/12-regression/code/r.R
#
# Data: read from data/ (committed with the repo); if that folder is
# missing, the same files are fetched from the published URL.
# Packages: tidyverse, survival (standard CRAN — no extra install needed)

suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library(survival))

# Resolve project root and load shared helpers
.this_script <- normalizePath(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1]))
.root        <- dirname(dirname(dirname(dirname(.this_script))))
source(file.path(.root, "lib", "helpers.R"))

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
df <- load_cohort() |>
  mutate(
    active_smoker = as.integer(raucherstatus == "aktiv"),
    sepsis        = as.integer(aufnahmegrund == "Sepsis"),
    age_centred   = alter - 64        # centre on 64, as in the data generator
  )

cat("Cohort:", nrow(df), "patients\n")
cat("30-day mortality:", round(mean(df$verstorben_30d) * 100, 1), "%\n")
cat("This dataset has a KNOWN ground truth — we compare estimates directly.\n\n")

# ---------------------------------------------------------------------------
# Section 1 — Logistic regression: 30-day mortality
# ---------------------------------------------------------------------------
cat("=================================================================\n")
cat("SECTION 1 — LOGISTIC REGRESSION: 30-day mortality\n")
cat("=================================================================\n")

fit_logistic <- glm(
  verstorben_30d ~ age_centred + sofa_score + crp_mg_l +
                   diabetes + sepsis + active_smoker,
  data   = df,
  family = binomial(link = "logit")
)

# Odds ratios and 95 % CI (log scale first, then exponentiate)
or_est <- exp(coef(fit_logistic))
or_ci  <- exp(suppressMessages(confint(fit_logistic)))

or_table <- data.frame(
  OR         = round(or_est, 4),
  ci_lower   = round(or_ci[, 1], 4),
  ci_upper   = round(or_ci[, 2], 4),
  p          = round(summary(fit_logistic)$coefficients[, 4], 4)
)
cat("\nOdds ratios with 95 % CI:\n")
print(or_table)

# ------------------------------------------------------------------
# Truth comparison — against the true ODDS ratios.
#
# data/generate_data.py draws survival times from a Weibull proportional-hazards
# process, so its coefficients are log-HAZARDS: exp(beta) is a true hazard
# ratio, not the odds ratio this model estimates. The true ORs below come from
# lib/ground_truth.py, which replays the same process at N = 2 000 000 and fits
# this exact model. Comparing an estimated OR against a true HR would make an
# unbiased estimate look biased. Keep these in sync with lib/ground_truth.py:
#   ./.venv/bin/python lib/ground_truth.py
# ------------------------------------------------------------------
true_hr <- c("age_centred" = 1.046, "sofa_score" = 1.320, "crp_mg_l" = 1.004,
             "diabetes" = 1.549, "sepsis" = 1.835, "active_smoker" = 1.506)
true_or <- c("age_centred" = 1.053, "sofa_score" = 1.387, "crp_mg_l" = 1.004,
             "diabetes" = 1.710, "sepsis" = 2.046, "active_smoker" = 1.601)

preds <- names(true_or)
covered <- 0L
cat("\n--- Truth comparison: estimated OR vs. true OR ---\n")
cat(sprintf("%-16s%10s%20s%10s%10s%10s\n",
            "Predictor", "OR_hat", "95% CI", "true OR", "true HR", "covered?"))
cat(strrep("-", 76), "\n")
for (p in preds) {
  est <- or_est[[p]]; lo <- or_ci[p, 1]; hi <- or_ci[p, 2]
  hit <- lo <= true_or[[p]] && true_or[[p]] <= hi
  covered <- covered + hit
  cat(sprintf("%-16s%10.3f%20s%10.3f%10.3f%10s\n", p, est,
              sprintf("[%.3f, %.3f]", lo, hi), true_or[[p]], true_hr[[p]],
              ifelse(hit, "yes", "NO")))
}
cat(sprintf("\nThe 95%% CI covers the true OR for %d/%d predictors.\n", covered, length(preds)))
cat("Remaining deviations are sampling variance (N=500, 78 events), not bias.\n\n")
cat("Why two 'true' columns? exp(beta) from the generating model is a HAZARD\n")
cat("ratio — compare it against Module 17's Cox model. A logistic model\n")
cat("estimates an ODDS ratio, which at a 15.6 % event rate sits further from 1.\n")

# Pseudo-R² (McFadden)
pseudo_r2 <- 1 - (fit_logistic$deviance / fit_logistic$null.deviance)
cat("\nPseudo-R² (McFadden):", round(pseudo_r2, 4), "\n")
cat("AIC:", round(AIC(fit_logistic), 2), "\n")

# EPV check
n_events <- sum(df$verstorben_30d)
n_pred   <- length(preds)
epv      <- n_events / n_pred
cat(sprintf("\nEvents per variable (EPV): %d / %d = %.1f\n", n_events, n_pred, epv))
if (epv < 10) {
  cat("  WARNING: EPV < 10 — model may be over-parametrised.\n")
} else {
  cat("  EPV >= 10 — model complexity is defensible.\n")
}

# ---------------------------------------------------------------------------
# Section 2 — OLS: length of stay
# ---------------------------------------------------------------------------
cat("\n=================================================================\n")
cat("SECTION 2 — LINEAR REGRESSION: length of stay (days)\n")
cat("=================================================================\n")
cat("Note: verweildauer_tage is right-skewed; OLS is illustrative here.\n\n")

fit_ols <- lm(
  verweildauer_tage ~ sofa_score + age_centred + sepsis + diabetes,
  data = df
)
ols_summary <- summary(fit_ols)
cat("R²:", round(ols_summary$r.squared, 4),
    "  Adjusted R²:", round(ols_summary$adj.r.squared, 4), "\n\n")
cat("Coefficients:\n")
print(round(ols_summary$coefficients, 4))

sofa_coef <- coef(fit_ols)["sofa_score"]
cat(sprintf(
  "\nInterpretation: each additional SOFA point is associated with %.2f extra days.\n",
  sofa_coef
))

# ---------------------------------------------------------------------------
# Section 3 — Cox regression: time-to-event analysis
# ---------------------------------------------------------------------------
cat("\n=================================================================\n")
cat("SECTION 3 — COX REGRESSION: time-to-event analysis\n")
cat("=================================================================\n")
cat("Duration: fu_zeit_tage | Event: status (1 = died, 0 = censored)\n")
cat("Patients without the event are right-censored (not a data loss).\n")
cat("Note: verweildauer_tage (length of stay) is NOT used here — it is not\n")
cat("a time-to-event variable (see data/README.md).\n\n")

fit_cox <- coxph(
  Surv(fu_zeit_tage, status) ~
    sofa_score + age_centred + sepsis + active_smoker,
  data = df
)

cox_summary <- summary(fit_cox)
hr_table <- data.frame(
  HR       = round(cox_summary$conf.int[, "exp(coef)"], 4),
  ci_lower = round(cox_summary$conf.int[, "lower .95"], 4),
  ci_upper = round(cox_summary$conf.int[, "upper .95"], 4),
  p        = round(cox_summary$coefficients[, "Pr(>|z|)"], 4)
)
cat("Hazard ratios (HR) with 95 % CI:\n")
print(hr_table)

cat(
  "\nNote: check the proportional-hazards assumption with cox.zph(fit_cox).",
  "\nIf p < 0.05 for a predictor, the PH assumption may be violated.\n"
)

cat("\nDone.\n")
