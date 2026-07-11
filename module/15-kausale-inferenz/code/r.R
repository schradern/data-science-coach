# Module 15 - Causal inference basics with propensity scores.

script <- normalizePath(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1]))
root <- dirname(dirname(dirname(dirname(script))))
source(file.path(root, "lib", "helpers.R"))
# Robust (sandwich) variance estimator for the weighted IPTW outcome model.
require_pkgs("sandwich", "lmtest")

suppressPackageStartupMessages(library(tidyverse))

smd <- function(x, group, weights = NULL) {
  g <- group == 1
  if (is.null(weights)) {
    m1 <- mean(x[g]); m0 <- mean(x[!g])
    v1 <- var(x[g]); v0 <- var(x[!g])
  } else {
    w <- weights
    m1 <- weighted.mean(x[g], w[g]); m0 <- weighted.mean(x[!g], w[!g])
    v1 <- weighted.mean((x[g] - m1)^2, w[g])
    v0 <- weighted.mean((x[!g] - m0)^2, w[!g])
  }
  (m1 - m0) / sqrt((v1 + v0) / 2)
}

df <- load_cohort() |> mutate(diabetes = as.integer(diabetes))

# DAG (see README §2): alter -> diabetes, alter -> sofa_score, alter ->
# verstorben_30d (alter is a CONFOUNDER: adjust for it) and
# diabetes -> sofa_score -> verstorben_30d (sofa_score is a MEDIATOR on the
# diabetes path: do NOT adjust for it when estimating diabetes's TOTAL effect
# on mortality, or the adjustment blocks part of the very effect we want to
# measure).
cat("\n1) Crude vs adjusted (total effect) diabetes effect\n")
crude <- glm(verstorben_30d ~ diabetes, data = df, family = binomial)
adj <- glm(verstorben_30d ~ diabetes + alter, data = df, family = binomial)
print(exp(cbind(OR = coef(crude), confint.default(crude)))["diabetes", ])
print(exp(cbind(OR = coef(adj), confint.default(adj)))["diabetes", ])

cat("\n1b) Teaching moment: (wrongly) adjusting for the mediator SOFA too\n")
adj_mediator <- glm(verstorben_30d ~ diabetes + alter + sofa_score, data = df, family = binomial)
print(exp(cbind(OR = coef(adj_mediator), confint.default(adj_mediator)))["diabetes", ])
cat("-> Adjusting for the mediator shifts the estimate toward null: it now answers\n")
cat("   'what is left once the SOFA-mediated pathway is blocked', not diabetes's\n")
cat("   total effect on mortality. Different question!\n")

cat("\n2) Propensity score and IPTW (confounders only: alter)\n")
ps_fit <- glm(diabetes ~ alter, data = df, family = binomial)
df <- df |>
  mutate(ps = predict(ps_fit, type = "response"),
         iptw = if_else(diabetes == 1, 1 / ps, 1 / (1 - ps)))
print(summary(df$iptw))

cat("\n3) Balance before and after IPTW\n")
cat(sprintf("alter raw SMD %.3f weighted SMD %.3f\n",
            smd(df$alter, df$diabetes), smd(df$alter, df$diabetes, df$iptw)))
cat(sprintf("(sofa_score raw SMD %.3f - NOT a PS covariate: it's a mediator, so it is\n",
            smd(df$sofa_score, df$diabetes)))
cat(" expected to stay imbalanced by design; that is not evidence of a broken PS model.)\n")

cat("\n4) Weighted outcome model (IPTW -> ATE), robust variance\n")
# IPTW weights are not frequency counts. quasibinomial + model-based SE and
# freq-weight SEs are both wrong; IPW needs a robust HC0 sandwich SE. Verified
# against a bootstrap that refits the PS: sandwich SE(log-OR) ~= 0.296 vs
# bootstrap ~= 0.297. This matches Python's var_weights + cov_type='HC0'.
w_fit <- glm(verstorben_30d ~ diabetes, data = df, family = quasibinomial, weights = iptw)
w_ci <- exp(lmtest::coefci(w_fit, vcov. = sandwich::vcovHC(w_fit, type = "HC0"))["diabetes", ])
cat(sprintf("IPTW weighted OR=%.2f 95%% CI [%.2f, %.2f]  (robust HC0 SE)\n",
            exp(coef(w_fit)[["diabetes"]]), w_ci[[1]], w_ci[[2]]))
cat("(The naive freq-weight SE would give a spuriously narrow CI ~[1.27, 2.44]; the robust\n")
cat(" CI now correctly includes 1 - the weighted estimate is not significant here.)\n")
