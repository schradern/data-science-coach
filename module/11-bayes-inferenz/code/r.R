# Module 11 - Bayesian inference (conjugate Beta-Binomial, no MCMC).
#   Rscript module/11-bayes-inferenz/code/r.R
# Data: read from data/ (committed with the repo); if that folder is
# missing, the same files are fetched from the published URL.
#
# Mirrors code/python.py: same estimands, same numbers.
#   Beta(a, b) posterior for a mortality rate; equal-tailed 95% credible
#   interval via qbeta(); the two-group difference, P(diff > 0) and the
#   ROPE mass via Monte-Carlo draws from rbeta() (set.seed for reproducibility).

script <- normalizePath(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1]))
root <- dirname(dirname(dirname(dirname(script))))
source(file.path(root, "lib", "helpers.R"))

set.seed(SEED)

# Posterior mean and equal-tailed 95% credible interval of Beta(a, b).
beta_summary <- function(a, b) {
  ci <- qbeta(c(0.025, 0.975), a, b)
  c(mean = a / (a + b), lo = ci[1], hi = ci[2])
}

df <- load_cohort()
n <- nrow(df)
k <- sum(df$verstorben_30d)

# ---------------------------------------------------------------------------
# 1) Beta-Binomial posterior for the overall 30-day mortality rate.
#    Uniform prior Beta(1, 1); posterior = Beta(1 + k, 1 + n - k).
# ---------------------------------------------------------------------------
cat("\n1) Overall 30-day mortality - posterior Beta(1+k, 1+n-k)\n")
a_post <- 1 + k
b_post <- 1 + n - k
s <- beta_summary(a_post, b_post)
cat(sprintf("data: k=%d deaths / n=%d patients (observed %.4f)\n", k, n, k / n))
cat(sprintf("posterior Beta(%d, %d)\n", a_post, b_post))
cat(sprintf("posterior mean=%.4f, 95%% CrI [%.4f, %.4f]\n", s["mean"], s["lo"], s["hi"]))

# Frequentist Wilson score interval for the honest contrast (module 10/13).
z <- qnorm(0.975)
phat <- k / n
denom <- 1 + z^2 / n
centre <- (phat + z^2 / (2 * n)) / denom
halfw <- z * sqrt(phat * (1 - phat) / n + z^2 / (4 * n^2)) / denom
cat(sprintf("frequentist Wilson 95%% CI [%.4f, %.4f]\n", centre - halfw, centre + halfw))

# ---------------------------------------------------------------------------
# 2) Prior sensitivity: with n=500 the prior barely matters; with a tiny
#    interim sample it dominates. weak = Beta(1,1), strong = Beta(20,80).
# ---------------------------------------------------------------------------
cat("\n2) Prior sensitivity - weak Beta(1,1) vs strong Beta(20,80)\n")
priors <- list("weak Beta(1,1)" = c(1, 1), "strong Beta(20,80)" = c(20, 80))
interim <- head(df, 30)
ni <- nrow(interim)
ki <- sum(interim$verstorben_30d)
cat(sprintf("full cohort:      k=%d/n=%d\n", k, n))
cat(sprintf("interim (first %d): k=%d/n=%d\n", ni, ki, ni))
for (label in names(priors)) {
  a0 <- priors[[label]][1]
  b0 <- priors[[label]][2]
  f <- beta_summary(a0 + k, b0 + n - k)
  i <- beta_summary(a0 + ki, b0 + ni - ki)
  cat(sprintf("  %19s: full  mean=%.4f CrI[%.4f,%.4f]  interim mean=%.4f CrI[%.4f,%.4f]\n",
              label, f["mean"], f["lo"], f["hi"], i["mean"], i["lo"], i["hi"]))
}

# ---------------------------------------------------------------------------
# 3) Comparing two groups the Bayesian way: Sepsis vs non-Sepsis.
#    Draw from each Beta posterior and look at the posterior of the DIFFERENCE.
# ---------------------------------------------------------------------------
cat("\n3) Two groups - Sepsis vs non-Sepsis, posterior of the difference\n")
sep <- df$aufnahmegrund == "Sepsis"
k1 <- sum(df$verstorben_30d[sep]);  n1 <- sum(sep)
k0 <- sum(df$verstorben_30d[!sep]); n0 <- sum(!sep)
s1 <- beta_summary(1 + k1, 1 + n1 - k1)
s0 <- beta_summary(1 + k0, 1 + n0 - k0)
cat(sprintf("Sepsis:     k=%d/n=%d, posterior mean=%.4f CrI[%.4f,%.4f]\n", k1, n1, s1["mean"], s1["lo"], s1["hi"]))
cat(sprintf("non-Sepsis: k=%d/n=%d, posterior mean=%.4f CrI[%.4f,%.4f]\n", k0, n0, s0["mean"], s0["lo"], s0["hi"]))

draws <- 200000
p1 <- rbeta(draws, 1 + k1, 1 + n1 - k1)
p0 <- rbeta(draws, 1 + k0, 1 + n0 - k0)
diff <- p1 - p0
d_ci <- quantile(diff, c(0.025, 0.975))
cat(sprintf("difference: mean=%.4f, 95%% CrI [%.4f, %.4f]\n", mean(diff), d_ci[1], d_ci[2]))
cat(sprintf("P(Sepsis mortality > non-Sepsis mortality) = %.4f\n", mean(diff > 0)))

# ---------------------------------------------------------------------------
# 4) ROPE - region of practical equivalence (+/- 0.05 on the difference).
# ---------------------------------------------------------------------------
cat("\n4) ROPE [-0.05, +0.05] on the mortality difference\n")
rope_lo <- -0.05
rope_hi <- 0.05
in_sep <- mean(diff > rope_lo & diff < rope_hi)
cat(sprintf("Sepsis vs non-Sepsis: posterior mass inside ROPE = %.4f\n", in_sep))

# Contrast: hypertension vs no hypertension (a near-null effect).
htn <- df$hypertonie == 1
kh <- sum(df$verstorben_30d[htn]);  nh <- sum(htn)
kn <- sum(df$verstorben_30d[!htn]); nn <- sum(!htn)
ph <- rbeta(draws, 1 + kh, 1 + nh - kh)
pn <- rbeta(draws, 1 + kn, 1 + nn - kn)
dh <- ph - pn
in_htn <- mean(dh > rope_lo & dh < rope_hi)
dh_ci <- quantile(dh, c(0.025, 0.975))
cat(sprintf("Hypertonie: k=%d/n=%d vs k=%d/n=%d\n", kh, nh, kn, nn))
cat(sprintf("Hypertonie vs none: diff mean=%.4f, 95%% CrI [%.4f, %.4f]\n", mean(dh), dh_ci[1], dh_ci[2]))
cat(sprintf("Hypertonie vs none: posterior mass inside ROPE = %.4f\n", in_htn))
