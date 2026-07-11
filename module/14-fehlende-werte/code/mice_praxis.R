# Module 14 - MICE in practice: passive imputation, bounds, auxiliary
# variables and convergence diagnostics.
#   Rscript module/14-fehlende-werte/code/mice_praxis.R
#
# Mirrors code/mice_praxis.py: same four lessons, same estimands.
#   1. Passive imputation      -- gewicht_kg and bmi are the same information
#                                  twice (bmi is a deterministic function of
#                                  weight and height). mice's native passive
#                                  imputation ("~ I(...)" method) avoids the
#                                  inconsistency that independent imputation
#                                  of both variables creates.
#   2. Bounds                  -- an unbounded normal imputer for bga_ph can
#                                  propose values outside the physiologically
#                                  possible range; PMM cannot.
#   3. Auxiliary variables      -- the imputation model may (and often must)
#                                  be richer than the analysis model
#                                  (Meng 1994, "uncongeniality").
#   4. Convergence diagnostics  -- MICE is a Gibbs sampler; it must be
#                                  inspected, not trusted.

script <- normalizePath(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1]))
root <- dirname(dirname(dirname(dirname(script))))
source(file.path(root, "lib", "helpers.R"))

require_pkgs("tidyverse", "mice")
suppressPackageStartupMessages({
  library(tidyverse)
  library(mice)
})
set.seed(SEED)

# Ensure UTF-8 glyphs (ä, ö, ü, ß, ²) render correctly in the saved figures.
# Some R installations start in the "C" locale, under which grid's text
# layout silently drops non-ASCII glyphs even though the underlying strings
# are valid UTF-8 (verified: charToRaw() is correct, only the rendered PNG
# is affected). Try a few common UTF-8 locale names; harmless if none is
# available on a given system.
for (.loc in c("en_US.UTF-8", "C.UTF-8", "en_GB.UTF-8", "de_DE.UTF-8")) {
  if (isTRUE(nzchar(suppressWarnings(tryCatch(Sys.setlocale("LC_CTYPE", .loc),
                                               error = function(e) ""))))) break
}

ASSETS <- file.path(root, "module", "14-fehlende-werte", "assets")

MODEL <- verstorben_30d ~ bga_ph + alter + sofa_score
PH_LOW <- 6.90
PH_HIGH <- 7.60
K_PMM <- 5

# --- shared course plot style (mirrors lib/plotstyle.py, no R equivalent
# exists yet, so the palette is reproduced here by hand) -------------------
PRIMARY <- "#2A5C8A"
SECONDARY <- "#6B7178"
EVENT <- "#B5482E"
PALETTE <- c(PRIMARY, EVENT, "#5B9E6E", SECONDARY, "#C08B3A", "#7B5EA7", "#3A8FA0")

course_theme <- function() {
  theme_minimal(base_size = 12) +
    theme(
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.y = element_line(color = "#ECEDEF", linewidth = 0.4),
      axis.line = element_line(color = "#B8BCC2", linewidth = 0.4),
      axis.ticks = element_blank(),
      plot.title = element_text(face = "bold", size = 13, hjust = 0),
      plot.title.position = "plot",
      strip.text = element_text(face = "bold", size = 10.5, hjust = 0),
      legend.position = "top",
      legend.title = element_blank()
    )
}

save_plot <- function(p, name, width, height) {
  path <- file.path(ASSETS, name)
  ggsave(path, p, width = width, height = height, dpi = 200, bg = "white")
  cat(sprintf("  saved: %s\n", path))
}

# --- data -------------------------------------------------------------------
cohort <- load_cohort()
labs <- load_labs()
truth <- readr::read_csv(file.path(root, "data", "bga_ph_wahrheit.csv"), show_col_types = FALSE)
df <- left_join(cohort, labs, by = "patient_id") |> left_join(truth, by = "patient_id")

# ===========================================================================
cat("\n==========================================================================\n")
cat("1) Passive imputation -- the derived-variable trap\n")
cat("==========================================================================\n")

miss_w <- is.na(df$gewicht_kg)
miss_b <- is.na(df$bmi)
same_rows <- all(miss_w == miss_b)
height_complete <- all(!is.na(df$groesse_cm))
cat(sprintf("  gewicht_kg missing on %d patients, bmi missing on %d patients\n",
            sum(miss_w), sum(miss_b)))
cat(sprintf("  missing on exactly the same rows : %s\n", same_rows))
cat(sprintf("  groesse_cm is complete            : %s\n", height_complete))
stopifnot("gewicht_kg and bmi are not missing on the same rows -- lesson invalid" = same_rows)
stopifnot("groesse_cm has missing values -- lesson invalid" = height_complete)

d1 <- df |> select(groesse_cm, gewicht_kg, bmi, alter, sofa_score, crp_mg_l)
miss_idx1 <- which(miss_w)

# --- naive: gewicht_kg and bmi imputed as two INDEPENDENT mice targets.
# The default predictor matrix uses "all other columns", so gewicht_kg's
# model includes bmi and bmi's model includes gewicht_kg -- exactly the
# circular setup that produces the inconsistency.
imp_naive <- mice(d1, m = 20, seed = SEED * 1000L + 0, printFlag = FALSE)
naive_long <- complete(imp_naive, "long")
naive_sub <- naive_long |>
  filter(.id %in% miss_idx1) |>
  mutate(bmi_von_gewicht = gewicht_kg / (groesse_cm / 100)^2,
         incons = abs(bmi - bmi_von_gewicht),
         methode = "Naiv: bmi und gewicht_kg\nunabhängig imputiert")

# --- passive: mice's native passive imputation. bmi's method becomes a
# formula that is RE-EVALUATED from the (possibly just-imputed) gewicht_kg
# and the always-observed groesse_cm on every iteration, instead of being
# drawn as its own random variable.
#
# pred["gewicht_kg", "bmi"] <- 0 is required: bmi is a deterministic
# function of gewicht_kg, so if gewicht_kg's own imputation model were
# allowed to use bmi as a predictor, the two variables would feed back into
# each other (gewicht_kg partially "explained" by its own transformation),
# which is both conceptually circular and numerically near-collinear.
ini <- mice(d1, maxit = 0)
meth <- ini$method
meth["bmi"] <- "~ I(gewicht_kg / (groesse_cm/100)^2)"
pred <- ini$predictorMatrix
pred["gewicht_kg", "bmi"] <- 0

imp_passive <- mice(d1, method = meth, predictorMatrix = pred, m = 20,
                     seed = SEED * 1000L + 1, printFlag = FALSE)
passive_long <- complete(imp_passive, "long")
passive_sub <- passive_long |>
  filter(.id %in% miss_idx1) |>
  mutate(bmi_von_gewicht = gewicht_kg / (groesse_cm / 100)^2,
         incons = abs(bmi - bmi_von_gewicht),
         methode = "Passiv: bmi aus imputiertem\ngewicht_kg abgeleitet")

cat(sprintf("\n  naive   |bmi_imp - gewicht_imp/(h/100)^2| : mean=%.4f  max=%.4f\n",
            mean(naive_sub$incons), max(naive_sub$incons)))
cat(sprintf("  passive |bmi_imp - gewicht_imp/(h/100)^2| : mean=%.10f  max=%.10f\n",
            mean(passive_sub$incons), max(passive_sub$incons)))
cat("\n  Naive MICE invents patients whose BMI contradicts their own weight\n")
cat("  and height. Passive imputation cannot, by construction.\n")

comb1 <- bind_rows(naive_sub, passive_sub) |>
  mutate(methode = factor(methode, levels = c(unique(naive_sub$methode), unique(passive_sub$methode))))
p1 <- ggplot(comb1, aes(x = bmi_von_gewicht, y = bmi)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = SECONDARY, linewidth = 0.7) +
  geom_point(aes(color = methode), alpha = 0.45, size = 1.6) +
  scale_color_manual(values = c(PRIMARY, "#5B9E6E")) +
  facet_wrap(~methode) +
  coord_equal() +
  labs(title = "Passive Imputation vermeidet die BMI-Inkonsistenz",
       x = "berechneter BMI aus imputiertem Gewicht (kg/m²)",
       y = "imputierter BMI (kg/m²)") +
  course_theme() +
  theme(legend.position = "none")
save_plot(p1, "passive_imputation.png", width = 11, height = 6.2)

# ===========================================================================
cat("\n==========================================================================\n")
cat("2) Bounds -- imputed values must be physiologically possible\n")
cat("==========================================================================\n")
cat(sprintf("  bga_ph is bounded to [%.2f, %.2f] by construction (data/generate_data.py clips it).\n",
            PH_LOW, PH_HIGH))

work2 <- df |> select(verstorben_30d, bga_ph, alter, sofa_score)
miss_idx2 <- which(is.na(work2$bga_ph))

# --- unbounded normal / linear-regression imputer ("norm" method) ----------
imp_norm <- mice(work2, m = 20, method = "norm", seed = SEED * 1000L + 2, printFlag = FALSE)
norm_vals <- complete(imp_norm, "long") |> filter(.id %in% miss_idx2) |> pull(bga_ph)
norm_oob <- norm_vals < PH_LOW | norm_vals > PH_HIGH
cat(sprintf("\n  norm imputer (m=20, unbounded): %d / %d draws outside [%.2f, %.2f]\n",
            sum(norm_oob), length(norm_vals), PH_LOW, PH_HIGH))
cat(sprintf("    draw range: %.3f to %.3f\n", min(norm_vals), max(norm_vals)))
if (sum(norm_oob) > 0) {
  cat(sprintf("    out-of-bounds values range %.3f to %.3f\n",
              min(norm_vals[norm_oob]), max(norm_vals[norm_oob])))
} else {
  lo_gap <- min(norm_vals) - PH_LOW
  hi_gap <- PH_HIGH - max(norm_vals)
  cat(sprintf("    none of these draws actually crossed the boundary; the closest one sat\n"))
  cat(sprintf("    %.3f pH units from the nearer bound.\n", min(lo_gap, hi_gap)))
  cat("    This cohort's pH distribution is narrow relative to the physiological\n")
  cat("    range, so the norm imputer rarely breaches it in practice -- but it\n")
  cat("    still assigns nonzero probability mass beyond [6.90, 7.60]:\n")
  lm_fit <- lm(bga_ph ~ alter + sofa_score + verstorben_30d, data = work2 |> filter(!is.na(bga_ph)))
  resid_sd <- summary(lm_fit)$sigma
  pred_miss <- predict(lm_fit, newdata = work2[miss_idx2, ])
  p_below <- pnorm(PH_LOW, mean = pred_miss, sd = resid_sd)
  p_above <- 1 - pnorm(PH_HIGH, mean = pred_miss, sd = resid_sd)
  expected_violations <- 20 * sum(p_below + p_above)
  cat(sprintf("    expected out-of-bounds draws under this model over m=20: %.4f (not exactly 0)\n",
              expected_violations))
}

# --- PMM, donor pool k = 5 --------------------------------------------------
imp_pmm <- mice(work2, m = 20, method = "pmm", donors = K_PMM,
                 seed = SEED * 1000L + 3, printFlag = FALSE)
pmm_vals <- complete(imp_pmm, "long") |> filter(.id %in% miss_idx2) |> pull(bga_ph)
pmm_oob <- pmm_vals < PH_LOW | pmm_vals > PH_HIGH
cat(sprintf("\n  PMM imputer  (m=20, k=%d): %d / %d draws outside [%.2f, %.2f]\n",
            K_PMM, sum(pmm_oob), length(pmm_vals), PH_LOW, PH_HIGH))
cat(sprintf("    draw range: %.3f to %.3f -- always inside the observed support\n",
            min(pmm_vals), max(pmm_vals)))

cat("\n  Rule: PMM respects the support and the marginal distribution of the\n")
cat("  observed data; an unbounded normal imputer does not. The cost: PMM\n")
cat("  cannot extrapolate beyond the observed range, even when that would be\n")
cat("  the physiologically correct thing to do.\n")

# ===========================================================================
cat("\n==========================================================================\n")
cat("3) Auxiliary variables and congeniality (Meng 1994)\n")
cat("==========================================================================\n")

truth_fit <- glm(MODEL, data = df |> mutate(bga_ph = bga_ph_wahr), family = binomial)
beta_true <- coef(truth_fit)[["bga_ph"]]
cat(sprintf("  true beta_bga_ph on the full (uncensored) data: %+.3f\n", beta_true))

run_variant <- function(cols, bga_predictors, seed_stream) {
  w <- df |> select(all_of(cols))
  ini <- mice(w, maxit = 0)
  pred <- ini$predictorMatrix
  pred["bga_ph", ] <- 0
  pred["bga_ph", bga_predictors] <- 1
  imp <- mice(w, predictorMatrix = pred, m = 20, seed = SEED * 1000L + seed_stream, printFlag = FALSE)
  # NOTE: the analysis-model formula is written out literally here (not as
  # the MODEL variable) because with.mids() evaluates its expression using
  # the formula's own stored environment as a lookup fallback; a formula
  # object created earlier at top level does not see the per-imputation
  # completed data frame, which raises "object 'verstorben_30d' not found".
  fit <- pool(with(imp, glm(verstorben_30d ~ bga_ph + alter + sofa_score, family = binomial)))
  s <- summary(fit)
  list(beta = s$estimate[s$term == "bga_ph"],
       se = s$std.error[s$term == "bga_ph"],
       lambda = fit$pooled$lambda[fit$pooled$term == "bga_ph"])
}

# laktat_mmol_l is itself ~17% missing, so folding it in as an auxiliary
# predictor makes variant (c) a genuinely MULTIVARIATE mice problem:
# laktat_mmol_l must be imputed too, inside the same chained-equations run,
# before it can serve as a predictor for bga_ph.
a <- run_variant(c("verstorben_30d", "bga_ph", "alter", "sofa_score"),
                  c("alter", "sofa_score"), 4)
b <- run_variant(c("verstorben_30d", "bga_ph", "alter", "sofa_score"),
                  c("alter", "sofa_score", "verstorben_30d"), 5)
cc <- run_variant(c("verstorben_30d", "bga_ph", "alter", "sofa_score",
                     "laktat_mmol_l", "crp_mg_l", "leukozyten_g_l"),
                   c("alter", "sofa_score", "verstorben_30d", "laktat_mmol_l", "crp_mg_l", "leukozyten_g_l"), 6)

cat(sprintf("\n  %-30s%13s%9s%9s\n", "imputation model", "beta_bga_ph", "SE", "lambda"))
cat(sprintf("  %-30s%13.3f%9.3f%9.3f\n", "a) alter + sofa_score", a$beta, a$se, a$lambda))
cat(sprintf("  %-30s%13.3f%9.3f%9.3f\n", "b) + verstorben_30d", b$beta, b$se, b$lambda))
cat(sprintf("  %-30s%13.3f%9.3f%9.3f\n", "c) + laktat, crp, leukozyten", cc$beta, cc$se, cc$lambda))
cat(sprintf("\n  %-30s%13.3f\n", "true (full data)", beta_true))

cat(sprintf("\n  |beta_a - true| = %.3f   (a) is biased toward zero: %s\n",
            abs(a$beta - beta_true), abs(a$beta) < abs(beta_true)))
cat(sprintf("  |beta_b - true| = %.3f   (b) recovers the true effect much more closely\n",
            abs(b$beta - beta_true)))
cat(sprintf("  |beta_c - true| = %.3f\n", abs(cc$beta - beta_true)))

if (cc$lambda <= b$lambda) {
  cat(sprintf("\n  (c) has an equal-or-lower fraction of missing information than (b): %.3f <= %.3f\n",
              cc$lambda, b$lambda))
} else {
  cat(sprintf("\n  (c) does NOT lower the fraction of missing information: %.3f > %.3f\n",
              cc$lambda, b$lambda))
  cat("  laktat_mmol_l is itself ~17 % missing, so folding it in adds its own\n")
  cat("  imputation uncertainty without buying much extra predictive power for\n")
  cat("  pH -- a legitimate negative result: richer is not automatically better.\n")
}

# ===========================================================================
cat("\n==========================================================================\n")
cat("4) Convergence diagnostics -- MICE is a Gibbs sampler\n")
cat("==========================================================================\n")

work4 <- df |> select(verstorben_30d, bga_ph, alter, sofa_score)
miss_idx4 <- which(is.na(work4$bga_ph))
n_chains <- 5
n_iter <- 20

imp4 <- mice(work4, m = n_chains, maxit = n_iter, method = "pmm", donors = K_PMM,
             seed = SEED * 1000L + 10, printFlag = FALSE)

chain_mean <- imp4$chainMean["bga_ph", , ]  # iteration x chain
chain_sd <- sqrt(imp4$chainVar["bga_ph", , ])

rhat <- function(chain_matrix) {
  # Gelman-Rubin potential scale reduction factor (R-hat) on the trace of
  # the chain-mean statistic. Close to 1 indicates the chains have mixed.
  n <- nrow(chain_matrix)
  m <- ncol(chain_matrix)
  chain_means <- colMeans(chain_matrix)
  grand_mean <- mean(chain_means)
  b <- n / (m - 1) * sum((chain_means - grand_mean)^2)
  w <- mean(apply(chain_matrix, 2, var))
  var_hat <- (n - 1) / n * w + b / n
  sqrt(var_hat / w)
}
rhat_mean <- rhat(chain_mean)

cat(sprintf("  maxit=%d, m=%d chains, tracking mean and SD of imputed bga_ph\n", n_iter, n_chains))
cat(sprintf("  Gelman-Rubin R-hat on the chain-mean trace: %.3f\n", rhat_mean))
if (rhat_mean < 1.1) {
  cat("  R-hat < 1.1 -- chains intermingle with no drift/trend: convergence looks healthy.\n")
} else {
  cat("  R-hat >= 1.1 -- chains have not mixed; more iterations or burn-in are needed.\n")
}

conv_long <- bind_rows(
  as.data.frame(chain_mean) |>
    mutate(iteration = row_number(), stat = "Mittelwert (imputiert)") |>
    pivot_longer(starts_with("Chain"), names_to = "chain", values_to = "wert"),
  as.data.frame(chain_sd) |>
    mutate(iteration = row_number(), stat = "Standardabweichung (imputiert)") |>
    pivot_longer(starts_with("Chain"), names_to = "chain", values_to = "wert")
) |>
  mutate(stat = factor(stat, levels = c("Mittelwert (imputiert)", "Standardabweichung (imputiert)")),
         chain = str_replace(chain, "^Chain", "Kette"))

p2 <- ggplot(conv_long, aes(x = iteration, y = wert, color = chain)) +
  geom_line(linewidth = 0.7) +
  geom_point(size = 1.3) +
  facet_wrap(~stat, ncol = 1, scales = "free_y") +
  scale_color_manual(values = PALETTE[seq_len(n_chains)]) +
  labs(title = "Konvergenz der MICE-Ketten für bga_ph", x = "Iteration", y = NULL) +
  course_theme()
save_plot(p2, "mice_konvergenz.png", width = 8.5, height = 7.5)

observed_vals <- work4$bga_ph[!is.na(work4$bga_ph)]
imputed_vals4 <- complete(imp4, "long") |> filter(.id %in% miss_idx4) |> pull(bga_ph)

dens_df <- bind_rows(
  tibble(bga_ph = observed_vals, gruppe = sprintf("beobachtet (n=%d)", length(observed_vals))),
  tibble(bga_ph = imputed_vals4,
         gruppe = sprintf("imputiert (n=%d, gepoolt über 5 Ketten)", length(imputed_vals4)))
)
p3 <- ggplot(dens_df, aes(x = bga_ph, color = gruppe)) +
  geom_density(linewidth = 1.1) +
  scale_color_manual(values = c(PRIMARY, EVENT)) +
  labs(title = "Beobachteter vs. imputierter arterieller pH",
       x = "arterieller pH (bga_ph)", y = "Dichte") +
  course_theme()
save_plot(p3, "mice_dichte.png", width = 8.5, height = 5.2)

cat(sprintf("\n  observed bga_ph  mean=%.3f  n=%d\n", mean(observed_vals), length(observed_vals)))
cat(sprintf("  imputed  bga_ph  mean=%.3f  n=%d (pooled over %d chains)\n",
            mean(imputed_vals4), length(imputed_vals4), n_chains))
cat("  The imputed density sits lower than the observed one: missing patients\n")
cat("  died more often, and dying patients are more acidotic. A perfectly\n")
cat("  overlapping density would be evidence the imputation model ignored the\n")
cat("  outcome (verstorben_30d) that drives the missingness.\n")
