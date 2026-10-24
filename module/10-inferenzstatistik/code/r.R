# Module 10 – Inferential statistics and hypothesis testing (R / base stats).
#
# Runs standalone from the project root:
#   Rscript module/10-inferenzstatistik/code/r.R
#
# Data: read from data/ (committed with the repo); if that folder is
# missing, the same files are fetched from the published URL.
# Code identifiers and comments are English; dataset column names stay German.

suppressPackageStartupMessages(library(tidyverse))

# Resolve project root relative to this script and load helpers.
script <- normalizePath(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1]))
root   <- dirname(dirname(dirname(dirname(script))))
source(file.path(root, "lib", "helpers.R"))

set.seed(SEED)

cohort <- load_cohort()
labs   <- load_labs()
df     <- left_join(cohort, labs, by = "patient_id")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

#' Pooled Cohen's d for two independent groups.
#'
#' @param a Numeric vector, first group (NAs removed internally).
#' @param b Numeric vector, second group.
#' @return Cohen's d (positive when mean(a) > mean(b)).
cohens_d <- function(a, b) {
  a <- na.omit(a)
  b <- na.omit(b)
  pooled_sd <- sqrt(
    ((length(a) - 1) * var(a) + (length(b) - 1) * var(b)) /
    (length(a) + length(b) - 2)
  )
  (mean(a) - mean(b)) / pooled_sd
}

#' Rank-biserial correlation r from Mann-Whitney U statistic.
#'
#' r = 2*U / (n1*n2) - 1. r > 0 means group 1 (first argument to
#' wilcox.test) tends to have stochastically higher values than group 2.
#'
#' @param u  U statistic from wilcox.test (for the first sample passed).
#' @param n1 Size of first group.
#' @param n2 Size of second group.
#' @return Rank-biserial r in [-1, 1].
rank_biserial_r <- function(u, n1, n2) {
  2 * u / (n1 * n2) - 1
}

# ---------------------------------------------------------------------------
# 0) Shapiro-Wilk: DESCRIPTIVE normality diagnostic for laktat_mmol_l
#    NOT a test-selection rule – see module 21 ("Normalitätstest-Autopilot").
# ---------------------------------------------------------------------------
cat("============================================================\n")
cat("0) Shapiro-Wilk: descriptive normality diagnostic for laktat_mmol_l\n")
cat("============================================================\n")

laktat_all <- df$laktat_mmol_l |> na.omit()
sw_res     <- shapiro.test(laktat_all)

cat(sprintf("  n=%d, W=%.4f, p=%.4e\n", length(laktat_all), sw_res$statistic, sw_res$p.value))
# Adjusted Fisher-Pearson skewness, the same estimator as pandas .skew()
n_lac <- length(laktat_all)
skew_lac <- n_lac / ((n_lac - 1) * (n_lac - 2)) *
  sum(((laktat_all - mean(laktat_all)) / sd(laktat_all))^3)
cat(sprintf("  Skewness=%.2f\n", skew_lac))
cat(sprintf("  Normal distribution rejected (p<0.05): %s\n",
            ifelse(sw_res$p.value < 0.05, "TRUE", "FALSE")))
cat("  NOTE: Shapiro-Wilk is a DESCRIPTIVE diagnostic, not a test-selection\n")
cat("  rule (it over-rejects at large N). Choose the test from the estimand,\n")
cat("  the shape seen in a plot, and robustness – see module 21. laktat is\n")
cat("  strongly right-skewed, so we report BOTH Welch-t and Mann-Whitney-U.\n")

# ---------------------------------------------------------------------------
# 1) Welch-t-test: laktat_mmol_l – Sepsis vs. non-Sepsis
# ---------------------------------------------------------------------------
cat("\n============================================================\n")
cat("1) Welch-t-test: laktat_mmol_l – Sepsis vs. nicht-Sepsis\n")
cat("============================================================\n")

sepsis    <- df |> filter(aufnahmegrund == "Sepsis") |> pull(laktat_mmol_l) |> na.omit()
no_sepsis <- df |> filter(aufnahmegrund != "Sepsis") |> pull(laktat_mmol_l) |> na.omit()

cat(sprintf("  Sepsis      n=%d  mean=%.2f  SD=%.2f\n",
            length(sepsis), mean(sepsis), sd(sepsis)))
cat(sprintf("  kein Sepsis n=%d  mean=%.2f  SD=%.2f\n",
            length(no_sepsis), mean(no_sepsis), sd(no_sepsis)))

t_res <- t.test(sepsis, no_sepsis, var.equal = FALSE)
d     <- cohens_d(sepsis, no_sepsis)

cat(sprintf("\n  Welch-t=%.3f,  p=%.4f\n", t_res$statistic, t_res$p.value))
cat(sprintf("  95-%%-CI difference: [%.2f, %.2f]\n", t_res$conf.int[1], t_res$conf.int[2]))
cat(sprintf("  95-%%-CI Sepsis:     [%.2f, %.2f]\n",
            t.test(sepsis)$conf.int[1], t.test(sepsis)$conf.int[2]))
cat(sprintf("  Effect size Cohen's d = %.3f  (0.2 small, 0.5 medium, 0.8 large)\n", d))
cat("\n  Correct interpretation:\n")
cat("  p = P(data as extreme | H0 true) – NOT P(H0 true | data).\n")
cat("  Always report effect size + CI alongside p.\n")

# ---------------------------------------------------------------------------
# 2) Mann-Whitney-U (nonparametric – preferred for skewed laktat)
# ---------------------------------------------------------------------------
cat("\n============================================================\n")
cat("2) Mann-Whitney-U: laktat_mmol_l – Sepsis vs. nicht-Sepsis\n")
cat("   (nonparametric, robust against skew and outliers)\n")
cat("============================================================\n")

# correct = TRUE is the default and matches scipy's use_continuity=True, so this
# p-value equals the Python script's. Keep the same setting in both languages.
mw_res <- wilcox.test(sepsis, no_sepsis, alternative = "two.sided", correct = TRUE)
r_rb   <- rank_biserial_r(mw_res$statistic, length(sepsis), length(no_sepsis))

cat(sprintf("  W=%.0f,  p=%.4f\n", mw_res$statistic, mw_res$p.value))
cat(sprintf("  Rank-biserial r=%.3f  (|r|~0.1 small, 0.3 medium, 0.5 large)\n", r_rb))
cat("  r > 0: stochastically higher values in Sepsis group.\n")

# ---------------------------------------------------------------------------
# 3) Chi-square: Diabetes × 30-day mortality
# ---------------------------------------------------------------------------
cat("\n============================================================\n")
cat("3) Chi-square: Diabetes x 30-day mortality (verstorben_30d)\n")
cat("============================================================\n")

ct <- table(Diabetes = df$diabetes, Verstorben_30d = df$verstorben_30d)
cat("\n  Contingency table:\n")
print(ct)

chi_res  <- chisq.test(ct, correct = FALSE)
n_total  <- sum(ct)
k        <- min(dim(ct)) - 1
cramers  <- sqrt(chi_res$statistic / (n_total * k))

cat(sprintf("\n  chi2=%.3f,  df=%d,  p=%.4f\n",
            chi_res$statistic, chi_res$parameter, chi_res$p.value))
cat(sprintf("  Cramér's V=%.3f  (0.1 small, 0.3 medium, 0.5 large)\n", cramers))

# Odds Ratio with 95-% CI (Woolf log-transform method)
a      <- ct[2, 2]   # Diabetes=1, verstorben=1
b      <- ct[2, 1]   # Diabetes=1, verstorben=0
cc     <- ct[1, 2]   # Diabetes=0, verstorben=1
d_cell <- ct[1, 1]   # Diabetes=0, verstorben=0
or_val <- (a * d_cell) / (b * cc)
log_se <- sqrt(1/a + 1/b + 1/cc + 1/d_cell)
or_lo  <- exp(log(or_val) - 1.96 * log_se)
or_hi  <- exp(log(or_val) + 1.96 * log_se)

cat(sprintf("\n  Odds Ratio (Diabetes=1 vs 0) = %.2f\n", or_val))
cat(sprintf("  95-%%-CI OR: [%.2f, %.2f]\n", or_lo, or_hi))
cat("  Note: OR != RR. At high event rates OR overestimates RR.\n")

# ---------------------------------------------------------------------------
# 4) Multiple testing – Bonferroni correction (5 lab markers)
# ---------------------------------------------------------------------------
cat("\n============================================================\n")
cat("4) Multiple testing: Bonferroni correction (5 lab markers)\n")
cat("============================================================\n")

lab_markers <- c("leukozyten_g_l", "haemoglobin_g_dl", "kreatinin_mg_dl",
                 "laktat_mmol_l", "natrium_mmol_l")
n_tests <- length(lab_markers)

results <- map_dfr(lab_markers, function(m) {
  grp_dead  <- df |> filter(verstorben_30d == 1) |> pull(!!sym(m)) |> na.omit()
  grp_alive <- df |> filter(verstorben_30d == 0) |> pull(!!sym(m)) |> na.omit()
  p_raw     <- t.test(grp_dead, grp_alive, var.equal = FALSE)$p.value
  d_val     <- cohens_d(grp_dead, grp_alive)
  tibble(marker = m, p_raw = p_raw, cohens_d = d_val)
})

results <- results |>
  mutate(
    p_bonferroni  = pmin(p_raw * n_tests, 1),
    sig_raw       = p_raw < 0.05,
    sig_bonferroni = p_bonferroni < 0.05
  )

cat("\n  Lab markers vs. 30-day mortality (Welch-t, alpha=0.05):\n")
print(results, digits = 4)
cat(sprintf("\n  Without correction: %d / %d significant\n", sum(results$sig_raw), n_tests))
cat(sprintf("  After Bonferroni:   %d / %d significant\n", sum(results$sig_bonferroni), n_tests))
cat(sprintf("  Bonferroni-corrected alpha = %.4f\n", 0.05 / n_tests))

# Benjamini-Hochberg for comparison
p_bh <- p.adjust(results$p_raw, method = "BH")
cat(sprintf("  After Benjamini-Hochberg:  %d / %d significant\n",
            sum(p_bh < 0.05), n_tests))
cat("  -> Bonferroni is conservative; BH controls FDR and is less strict.\n")

cat("\nDone.\n")
