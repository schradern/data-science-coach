# Module 19 - propensity score matching and weighting.
#   Rscript module/19-propensity-score-causal/code/r.R
#
# Uses the shared cohort (kohorte.csv). Exposure: diabetes. Outcome:
# verstorben_30d. Per the course DAG (see data/README.md and module 15):
# alter is a confounder and must be adjusted for; sofa_score is a MEDIATOR
# on the diabetes -> death pathway and must NOT be adjusted for when
# estimating the total effect of diabetes.

script <- normalizePath(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1]))
root   <- dirname(dirname(dirname(dirname(script))))
source(file.path(root, "lib", "helpers.R"))
# Robust/cluster-robust variance estimators for the IPW and matched analyses.
require_pkgs("sandwich", "lmtest")
set.seed(SEED)

TREATMENT <- "diabetes"
OUTCOME <- "verstorben_30d"
PS_COVARIATES <- c("alter", "geschlecht_m", "hypertonie", "raucher_aktiv")
CONTEXT_COVARIATES <- c("sofa_score", "crp_mg_l")
SMD_THRESHOLD <- 0.10

smd <- function(x, treated_mask, weights = NULL) {
  g <- as.logical(treated_mask)
  if (is.null(weights)) {
    m1 <- mean(x[g]); m0 <- mean(x[!g])
    v1 <- var(x[g]);  v0 <- var(x[!g])
  } else {
    w <- weights
    m1 <- weighted.mean(x[g], w[g]); m0 <- weighted.mean(x[!g], w[!g])
    v1 <- sum(w[g] * (x[g] - m1)^2) / sum(w[g])
    v0 <- sum(w[!g] * (x[!g] - m0)^2) / sum(w[!g])
  }
  (m1 - m0) / sqrt((v1 + v0) / 2)
}

# Returns the matched set with a `pair_id` column. The two members of a pair
# are NOT independent, so a matched outcome model needs a cluster-robust SE on
# pair_id (see below) - not the naive model-based SE.
match_with_caliper <- function(df, caliper) {
  treated <- df[df[[TREATMENT]] == 1, ]
  control <- df[df[[TREATMENT]] == 0, ]
  dists <- outer(treated$logit_ps, control$logit_ps, function(a, b) abs(a - b))
  available <- rep(TRUE, nrow(control))
  order_idx <- order(apply(dists, 1, min))
  t_idx <- integer(0); c_idx <- integer(0)
  for (i in order_idx) {
    d <- dists[i, ]
    d[!available] <- Inf
    j <- which.min(d)
    if (d[j] <= caliper) {
      t_idx <- c(t_idx, i)
      c_idx <- c(c_idx, j)
      available[j] <- FALSE
    }
  }
  pair_ids <- seq_along(t_idx)
  out <- rbind(treated[t_idx, ], control[c_idx, ])
  out$pair_id <- c(pair_ids, pair_ids)  # treated block, then control block
  out
}

df <- load_cohort()
df$geschlecht_m <- as.integer(df$geschlecht == "maennlich")
df$raucher_aktiv <- as.integer(df$raucherstatus == "aktiv")

n_treated <- sum(df[[TREATMENT]])
n_control <- sum(1 - df[[TREATMENT]])
cat("=== Shared cohort: diabetes as exposure ===\n")
cat(sprintf("Treated (diabetes=1): %d | Control (diabetes=0): %d\n", n_treated, n_control))

ps_formula <- as.formula(paste(TREATMENT, "~", paste(PS_COVARIATES, collapse = " + ")))
ps_model <- glm(ps_formula, data = df, family = binomial)
df$ps <- pmin(pmax(predict(ps_model, type = "response"), 1e-6), 1 - 1e-6)
df$logit_ps <- log(df$ps / (1 - df$ps))

cat("\nPropensity score overlap:\n")
print(aggregate(ps ~ get(TREATMENT), data = df,
                 FUN = function(x) round(c(min = min(x), median = median(x), max = max(x)), 3)))

cat("\n=== Balance before matching (SMD) ===\n")
all_covariates <- c(PS_COVARIATES, CONTEXT_COVARIATES)
before <- sapply(all_covariates, function(v) smd(df[[v]], df[[TREATMENT]] == 1))
for (v in all_covariates) {
  tag <- if (v %in% PS_COVARIATES) "" else " [context only - NOT adjusted, see note below]"
  cat(sprintf("%-14s SMD = % .3f%s\n", v, before[v], tag))
}

naive_matched <- match_with_caliper(df, caliper = Inf)
cat(sprintf("\n=== 1:1 matching WITHOUT a caliper (%d pairs) ===\n", nrow(naive_matched) / 2))
naive_after <- sapply(all_covariates, function(v) smd(naive_matched[[v]], naive_matched[[TREATMENT]] == 1))
for (v in PS_COVARIATES) {
  flag <- if (abs(naive_after[v]) > SMD_THRESHOLD) " <- exceeds 0.10, add a caliper" else ""
  cat(sprintf("%-14s SMD before = % .3f | after = % .3f%s\n", v, before[v], naive_after[v], flag))
}

caliper <- 0.2 * sd(df$logit_ps)
matched <- match_with_caliper(df, caliper)
n_pairs <- nrow(matched) / 2
cat(sprintf("\n=== 1:1 matching WITH caliper = 0.2 x SD(logit PS) = %.3f (%d of %d treated matched) ===\n",
            caliper, n_pairs, n_treated))
after <- sapply(all_covariates, function(v) smd(matched[[v]], matched[[TREATMENT]] == 1))
for (v in PS_COVARIATES) {
  ok <- if (abs(after[v]) <= SMD_THRESHOLD) "OK (< 0.10)" else "STILL IMBALANCED"
  cat(sprintf("%-14s SMD before = % .3f | after caliper = % .3f  [%s]\n", v, before[v], after[v], ok))
}
for (v in CONTEXT_COVARIATES) {
  cat(sprintf("%-14s SMD before = % .3f | after caliper = % .3f  [not a matching target - mediator/unrelated, see note]\n",
              v, before[v], after[v]))
}
cat("\nNote: sofa_score stays imbalanced even after good confounder balance - that's expected.\n")
cat("Diabetes causally raises sofa_score (mediator), so matched diabetics are still sicker on\n")
cat("average. Forcing sofa_score balance would adjust away part of diabetes's real effect.\n")

# Unstabilised ATE weights, plus stabilised weights P(A=a)/P(A=a|X): same
# estimand (ATE), tighter weight distribution.
df$iptw <- ifelse(df[[TREATMENT]] == 1, 1 / df$ps, 1 / (1 - df$ps))
p_treated <- mean(df[[TREATMENT]])
df$siptw <- ifelse(df[[TREATMENT]] == 1, p_treated / df$ps, (1 - p_treated) / (1 - df$ps))
cat(sprintf("\n=== Inverse Probability Weighting (IPW, targets the ATE) ===\n"))
cat(sprintf("Unstabilised weight range [%.2f, %.2f] | stabilised weight range [%.2f, %.2f]\n",
            min(df$iptw), max(df$iptw), min(df$siptw), max(df$siptw)))
cat("Stabilising leaves the estimand unchanged but shrinks the weight range.\n")
for (v in PS_COVARIATES) {
  w_smd <- smd(df[[v]], df[[TREATMENT]] == 1, df$iptw)
  ok <- if (abs(w_smd) <= SMD_THRESHOLD) "OK (< 0.10)" else "still imbalanced"
  cat(sprintf("%-14s SMD before = % .3f | IPW-weighted = % .3f  [%s]\n", v, before[v], w_smd, ok))
}

# Two DIFFERENT estimands, each with the CORRECT variance:
#   matched caliper -> ATT (effect in the treated), cluster-robust SE on pair_id
#   IPW 1/ps        -> ATE (effect in the whole cohort), robust HC0 sandwich SE
cat(sprintf("\n=== Effect of %s on %s: crude vs. matched (ATT) vs. IPW (ATE) ===\n", TREATMENT, OUTCOME))

or_ci_robust <- function(fit, V) {
  ci <- exp(lmtest::coefci(fit, vcov. = V)[TREATMENT, ])
  c(OR = exp(coef(fit)[[TREATMENT]]), lo = ci[[1]], hi = ci[[2]])
}

crude <- glm(as.formula(paste(OUTCOME, "~", TREATMENT)), data = df, family = binomial)
crude_ci <- exp(confint.default(crude)[TREATMENT, ])
cat(sprintf("Crude (association)  OR = %.2f 95%% CI [%.2f, %.2f]\n",
            exp(coef(crude)[TREATMENT]), crude_ci[1], crude_ci[2]))

# Matched analysis (ATT): cluster the SE on the matched pair - the two members of
# a pair are not independent, so the naive model-based SE is not the right one.
matched_fit <- glm(as.formula(paste(OUTCOME, "~", TREATMENT)), data = matched, family = binomial)
m <- or_ci_robust(matched_fit, sandwich::vcovCL(matched_fit, cluster = matched$pair_id))
cat(sprintf("Matched caliper -> ATT (%d pairs) OR = %.2f 95%% CI [%.2f, %.2f]  (cluster-robust SE on pair)\n",
            n_pairs, m[["OR"]], m[["lo"]], m[["hi"]]))

# IPW (ATE): non-integer weights are not frequency counts. quasibinomial +
# model-based SE and freq-weight SEs are both wrong; the correct variance is a
# robust HC0 sandwich (verified against a bootstrap that refits the PS: sandwich
# SE(log-OR) ~= 0.302 vs bootstrap ~= 0.305). This matches Python's var_weights +
# cov_type='HC0'.
ipw_fit <- glm(as.formula(paste(OUTCOME, "~", TREATMENT)), data = df, family = quasibinomial,
               weights = iptw)
i <- or_ci_robust(ipw_fit, sandwich::vcovHC(ipw_fit, type = "HC0"))
cat(sprintf("IPW 1/ps -> ATE      OR = %.2f 95%% CI [%.2f, %.2f]  (robust HC0 sandwich SE)\n",
            i[["OR"]], i[["lo"]], i[["hi"]]))

# Sensitivity: stabilised weights (same ATE) and 99th-percentile trimming.
ipw_stab <- glm(as.formula(paste(OUTCOME, "~", TREATMENT)), data = df, family = quasibinomial,
                weights = siptw)
s <- or_ci_robust(ipw_stab, sandwich::vcovHC(ipw_stab, type = "HC0"))
cat(sprintf("  IPW stabilised (ATE) OR = %.2f 95%% CI [%.2f, %.2f]\n", s[["OR"]], s[["lo"]], s[["hi"]]))
cap <- quantile(df$iptw, 0.99)
df$iptw_trim <- pmin(df$iptw, cap)
ipw_trim <- glm(as.formula(paste(OUTCOME, "~", TREATMENT)), data = df, family = quasibinomial,
                weights = iptw_trim)
tr <- or_ci_robust(ipw_trim, sandwich::vcovHC(ipw_trim, type = "HC0"))
cat(sprintf("  IPW trimmed 99th pct OR = %.2f 95%% CI [%.2f, %.2f]  (cap = %.2f)\n",
            tr[["OR"]], tr[["lo"]], tr[["hi"]], cap))

cat("\nMatched (ATT) and IPW (ATE) both leave out the sofa_score mediator (total-effect\n")
cat("estimands) but target different populations - the treated vs. the whole cohort.\n")
