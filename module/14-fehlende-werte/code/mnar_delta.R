# Module 14 - MNAR sensitivity analysis: delta-adjustment / tipping point.
#   Rscript module/14-fehlende-werte/code/mnar_delta.R
#
# Mirrors code/mnar_delta.py: same estimand, same sweep, same Rubin's-rules
# pooling (here via mice::pool()).
#
# Multiple imputation with the outcome in the imputation model (module/12's
# main analysis) is only valid under MAR. MAR can never be verified from the
# data alone. The honest response is a sensitivity analysis under assumed
# MNAR departures.
#
# Delta-adjustment (pattern-mixture form): impute under MAR as usual, then
# shift only the IMPUTED values of bga_ph by a constant delta (never the
# observed values), refit, and pool. delta = 0 recovers the ordinary MAR
# analysis. delta is the assumed systematic pH difference between patients
# whose blood gas was never measured and otherwise-identical patients whose
# blood gas was measured.

script <- normalizePath(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1]))
root <- dirname(dirname(dirname(dirname(script))))
source(file.path(root, "lib", "helpers.R"))

require_pkgs("tidyverse", "mice")
suppressPackageStartupMessages({
  library(tidyverse)
  library(mice)
})

MODEL <- verstorben_30d ~ bga_ph + alter + sofa_score

# MICE settings -- mirror code/r.R's multiple-imputation section (m = 20,
# method = "pmm").
M <- 20L
DELTA_STEP <- 0.01
PRIMARY_DELTAS <- round(seq(-0.10, 0.10, by = DELTA_STEP), 2)
WIDE_DELTAS    <- round(seq(-0.25, 0.25, by = DELTA_STEP), 2)

# Note: this script produces no figure; the figure is owned by mnar_delta.py.

work <- left_join(load_cohort(), load_labs(), by = "patient_id") |>
  select(verstorben_30d, bga_ph, alter, sofa_score)

# OR per 0.1 pH drop on the full, un-masked truth. Only possible because the
# cohort is synthetic -- this is the target the MAR analysis should hit.
oracle_odds_ratio <- function() {
  truth <- readr::read_csv(file.path(root, "data", "bga_ph_wahrheit.csv"),
                           show_col_types = FALSE)
  d <- left_join(load_cohort(), truth, by = "patient_id") |>
    mutate(bga_ph = bga_ph_wahr)
  fit <- glm(MODEL, data = d, family = binomial)
  exp(-0.1 * coef(fit)[["bga_ph"]])
}

# Residual SD of pH given the analysis covariates. The natural yardstick for
# delta: judging it against assay noise would be a category error -- assay noise
# is random, delta is a systematic between-group difference.
conditional_sd_of_ph <- function() {
  truth <- readr::read_csv(file.path(root, "data", "bga_ph_wahrheit.csv"),
                           show_col_types = FALSE)
  d <- left_join(load_cohort(), truth, by = "patient_id")
  res <- resid(lm(bga_ph_wahr ~ alter + sofa_score + verstorben_30d, data = d))
  sqrt(sum(res^2) / (length(res) - 4L))
}

# -----------------------------------------------------------------------
# Data and seeding
# -----------------------------------------------------------------------

# Round to the 0.01 grid and normalise away floating-point "-0".
clean_delta <- function(delta) {
  d <- round(delta, 2)
  if (d == 0) 0 else d
}

# Position of `delta` on the fixed -0.25..0.25 grid (0.01 steps). Using a
# fixed grid means every delta gets the same seed regardless of whether the
# primary +-0.10 sweep or the widened +-0.25 sweep produced it.
grid_index <- function(delta) {
  as.integer(round((clean_delta(delta) + 0.25) / DELTA_STEP))
}

# seed_i <- SEED * 1000L + i  is the prescribed per-stream derivation for R.
seed_for <- function(delta) SEED * 1000L + grid_index(delta)

# -----------------------------------------------------------------------
# One delta: m = 20 fresh MICE imputations, shift imputed cells, pool
# -----------------------------------------------------------------------

run_delta <- function(delta, m = M) {
  miss_idx <- is.na(work$bga_ph)
  observed_original <- work$bga_ph[!miss_idx]

  imp <- mice(work, m = m, method = "pmm", seed = seed_for(delta), printFlag = FALSE)

  # Shift only the imputed cells of bga_ph, in every imputed data set. Assert
  # observed cells are untouched -- the load-bearing guarantee of the whole
  # delta-adjustment method.
  for (k in seq_len(m)) {
    completed_k <- complete(imp, k)
    stopifnot(identical(completed_k$bga_ph[!miss_idx], observed_original))
    imp$imp$bga_ph[[k]] <- imp$imp$bga_ph[[k]] + delta
  }

  fit <- with(imp, glm(verstorben_30d ~ bga_ph + alter + sofa_score, family = binomial))
  s <- summary(pool(fit))
  b  <- s$estimate[s$term == "bga_ph"]
  se <- s$std.error[s$term == "bga_ph"]

  or_point <- exp(-0.1 * b)
  or_lo <- exp(-0.1 * (b + 1.96 * se))
  or_hi <- exp(-0.1 * (b - 1.96 * se))

  list(delta = clean_delta(delta), b = b, se = se,
       OR = or_point, ci_lo = or_lo, ci_hi = or_hi,
       includes_one = (or_lo <= 1.0 && or_hi >= 1.0))
}

sweep <- function(deltas) {
  rows <- lapply(deltas, run_delta)
  df <- bind_rows(lapply(rows, as.data.frame))
  df[order(df$delta), ]
}

# -----------------------------------------------------------------------
# Sections
# -----------------------------------------------------------------------

section_1_setup <- function() {
  cat(strrep("=", 78), "\n")
  cat("1) Delta-adjustment: shift only the imputed values, never the observed\n")
  cat(strrep("=", 78), "\n")
  n_miss <- sum(is.na(work$bga_ph))
  cat(sprintf("  bga_ph missing: %d / %d (%.1f%%)\n", n_miss, nrow(work), 100 * n_miss / nrow(work)))
  cat("  Analysis model: verstorben_30d ~ bga_ph + alter + sofa_score\n")
  cat(sprintf("  m = %d imputations, method = pmm (fresh MI per delta)\n", M))
  cat("\n  bga_ph_imputed(delta) = bga_ph_imputed(MAR) + delta\n")
  cat("  delta = 0 recovers the ordinary MAR multiple-imputation analysis.\n")
}

section_2_sweep <- function(df) {
  cat("\n", strrep("=", 78), "\n", sep = "")
  cat("2) Sweep delta over [-0.10, +0.10] in steps of 0.01 (21 values)\n")
  cat(strrep("=", 78), "\n")
  cat(sprintf("  %7s%15s%22s%18s\n", "delta", "OR / -0.1 pH", "95% CI", "CI includes 1?"))
  for (i in seq_len(nrow(df))) {
    r <- df[i, ]
    marker <- if (r$delta == 0) "<-- MAR" else ""
    cat(sprintf("  %+7.2f%15.3f   (%.3f, %.3f)%18s  %s\n",
                r$delta, r$OR, r$ci_lo, r$ci_hi, as.character(r$includes_one), marker))
  }

  row0 <- df[df$delta == 0, ][1, ]
  cat(sprintf("\n  delta = 0.00 (MAR baseline): OR = %.2f, 95%% CI = (%.2f, %.2f)\n",
              row0$OR, row0$ci_lo, row0$ci_hi))
  # Sanity-check delta = 0 against the ORACLE (the full, un-masked truth), not
  # against a hardcoded number: a literal reference goes stale the moment anyone
  # retunes the generating process.
  oracle_or <- oracle_odds_ratio()
  cat(sprintf("  Oracle (full truth, no missingness):      OR = %.2f\n", oracle_or))
  cat("  Multiple imputation is stochastic, so exact reproduction is not\n")
  cat("  expected -- but delta = 0 must land in the oracle's neighbourhood.\n")
  ratio <- row0$OR / oracle_or
  if (!(ratio >= 0.5 && ratio <= 1.5)) {
    stop(sprintf(
      "delta=0 OR (%.2f) is %.2fx the oracle OR (%.2f) -- the delta shift is ",
      row0$OR, ratio, oracle_or
    ), "very likely leaking into the observed values.", call. = FALSE)
  }
  if (isTRUE(row0$includes_one)) {
    cat("\n  NOTE: even at delta = 0, the pooled 95 % CI already includes OR = 1.\n")
    cat("  This is a genuine, seed-stable feature of this cohort (N=500, ~78\n")
    cat("  deaths) and not a bug in the delta shift: the FULL-DATA truth model\n")
    cat("  (data/bga_ph_wahrheit.csv, no missingness at all) already has a 95 %\n")
    cat("  CI for this OR that barely excludes 1, and Rubin's between-imputation\n")
    cat("  variance widens it further. The MAR point estimate (~1.7-1.9) looks\n")
    cat("  reassuring, but its own uncertainty was never small to begin with.\n")
  }
}

find_literal_tipping_point <- function(df) {
  hits <- df[df$includes_one, ]
  if (nrow(hits) == 0) return(NULL)
  hits[which.min(abs(hits$delta)), ]
}

significance_region <- function(df) {
  sig <- df[!df$includes_one, ]
  if (nrow(sig) == 0) return(NULL)
  c(min(sig$delta), max(sig$delta))
}

section_3_tipping_point <- function(df) {
  cat("\n", strrep("=", 78), "\n", sep = "")
  cat("3) Tipping point\n")
  cat(strrep("=", 78), "\n")

  literal <- find_literal_tipping_point(df)
  widened <- FALSE
  if (is.null(literal)) {
    cat("  The 95% CI never includes OR = 1 within +-0.10.\n")
    cat("  Widening the sweep to +-0.25 to locate the tipping point ...\n")
    extra_deltas <- setdiff(WIDE_DELTAS, df$delta)
    extra <- sweep(extra_deltas)
    df <- rbind(df, extra)
    df <- df[order(df$delta), ]
    widened <- TRUE
    literal <- find_literal_tipping_point(df)
    if (is.null(literal)) {
      cat("  The 95% CI STILL never includes OR = 1 within +-0.25.\n")
      cat("  Reporting plainly: no tipping point was found in the searched range.\n")
    } else {
      cat(sprintf("  Found within the widened range: delta = %+.2f.\n", literal$delta))
    }
  }

  region <- significance_region(df)

  if (!is.null(literal)) {
    cat(sprintf("\n  Literal definition (smallest |delta| with CI including 1): delta = %+.2f\n",
                literal$delta))
    if (literal$delta == 0) {
      cat("  This is delta = 0 itself: the MAR analysis's own 95% CI already\n")
      cat("  includes 1 (see section 2), so no MNAR departure is even required to\n")
      cat("  lose the conclusion in the direction that weakens it further.\n")
    }
  } else {
    cat("\n  No tipping point found.\n")
  }

  boundary <- NULL
  if (!is.null(region)) {
    lo <- region[1]; hi <- region[2]
    cat(sprintf("\n  Significance boundary (secondary, more informative number): the\n"))
    cat("  pooled 95% CI EXCLUDES 1 -- i.e. the acidosis-mortality association is\n")
    cat(sprintf("  statistically supported -- only for delta in [%+.2f, %+.2f].\n", lo, hi))
    cat("  Outside that window (including delta = 0) the finding is not robust.\n")
    boundary <- if (abs(lo) < abs(hi)) lo else hi
  } else {
    cat("\n  The pooled 95% CI never excludes 1 anywhere in the searched range:\n")
    cat("  the association is not statistically supported for ANY assumed delta here.\n")
  }

  list(df = df, widened = widened, literal_tipping = literal,
       significance_region = region, boundary_delta = boundary)
}

section_4_direction <- function(df) {
  cat("\n", strrep("=", 78), "\n", sep = "")
  cat("4) Which direction of delta strengthens, which weakens the association?\n")
  cat(strrep("=", 78), "\n")
  cat("  Patients without a measured blood gas died more often than patients\n")
  cat("  with one. Reasoning through both MNAR stories:\n\n")
  cat("    delta < 0  ->  unmeasured patients are assumed MORE acidotic (lower\n")
  cat("                   pH) than the MAR imputation predicts. That is exactly\n")
  cat("                   consistent with them dying more often -- it reinforces\n")
  cat("                   the acidosis-mortality story.\n")
  cat("    delta > 0  ->  unmeasured patients are assumed LESS acidotic (higher\n")
  cat("                   pH) than the MAR imputation predicts, despite dying\n")
  cat("                   more often. That contradicts the acidosis-mortality\n")
  cat("                   story -- their excess mortality is then unexplained by\n")
  cat("                   pH, so the pH effect must shrink to accommodate them.\n")

  or_neg  <- df$OR[df$delta == -0.01]
  or_zero <- df$OR[df$delta == 0]
  or_pos  <- df$OR[df$delta == 0.01]
  cat(sprintf("\n  Computed, not guessed: OR(delta=-0.01) = %.3f  OR(delta=0) = %.3f  OR(delta=+0.01) = %.3f\n",
              or_neg, or_zero, or_pos))
  conservative <- if (or_pos < or_neg) "positive (delta > 0)" else "negative (delta < 0)"
  cat("  -> OR increases as delta decreases, decreases as delta increases.\n")
  cat(sprintf("  -> The CONSERVATIVE (association-weakening) direction is delta %s.\n", conservative))
  cat("     That is the direction a reviewer will ask about: 'what if the patients\n")
  cat("     you didn't measure were actually healthier on this value than you assumed?'\n")
  conservative
}

section_5_plausibility <- function(tipping) {
  cat("\n", strrep("=", 78), "\n", sep = "")
  cat("5) Clinical plausibility of the tipping-point delta\n")
  cat(strrep("=", 78), "\n")

  boundary <- tipping$boundary_delta
  literal <- tipping$literal_tipping
  delta_for_verdict <- if (!is.null(boundary)) boundary else if (!is.null(literal)) literal$delta else NA

  if (is.na(delta_for_verdict)) {
    cat("  No finite tipping-point delta was found -- no plausibility judgement to make.\n")
    return(invisible(NULL))
  }

  mag <- abs(delta_for_verdict)
  cat(sprintf("  Relevant magnitude: |delta| = %.2f pH units.\n", mag))
  sd_ph <- conditional_sd_of_ph()
  in_sd <- mag / sd_ph
  cat(sprintf("  Conditional SD of pH given alter, sofa_score, outcome: %.3f.\n", sd_ph))
  cat(sprintf("  -> the tipping point is %.2f conditional standard deviations.\n\n", in_sd))
  cat("  Do NOT judge delta against blood-gas assay noise. Assay noise is\n")
  cat("  RANDOM measurement error; delta is a SYSTEMATIC difference in the\n")
  cat("  true mean pH between measured and unmeasured patients, after\n")
  cat("  adjusting for everything the imputation model already knows. The\n")
  cat("  honest yardstick is the residual spread, not the analyser's precision.\n")

  if (in_sd < 0.2) {
    verdict <- sprintf(paste(
      "FRAGILE: %.2f SD is a small systematic shift. A bias this modest would",
      "overturn the conclusion, so the finding does not survive a realistic",
      "departure from MAR."
    ), in_sd)
  } else if (in_sd < 0.5) {
    verdict <- sprintf(paste(
      "MODERATELY ROBUST: it takes a systematic shift of %.2f SD -- unmeasured",
      "patients differing from the imputation model by nearly half a residual",
      "standard deviation, in the direction that CONTRADICTS their higher",
      "mortality -- before the conclusion breaks. That is a specific, arguable",
      "assumption, not a trivial one."
    ), in_sd)
  } else {
    verdict <- sprintf(paste(
      "ROBUST: only a systematic shift of %.2f SD or more overturns the",
      "conclusion. A departure from MAR that large would very likely be visible",
      "in the other measured markers (lactate, SOFA)."
    ), in_sd)
  }

  cat(sprintf("\n  delta = %+.2f (%.2f SD): %s\n", delta_for_verdict, in_sd, verdict))
}

section_6_report <- function(tipping, conservative) {
  cat("\n", strrep("=", 78), "\n", sep = "")
  cat("6) The sentence that belongs in the paper\n")
  cat(strrep("=", 78), "\n")
  literal <- tipping$literal_tipping
  region <- tipping$significance_region
  lit_txt <- if (!is.null(literal)) sprintf("%+.2f", literal$delta) else "nicht gefunden"
  region_txt <- if (!is.null(region)) sprintf("[%+.2f, %+.2f]", region[1], region[2]) else "keinen Bereich"
  cat(sprintf(paste0(
    '  "Eine Delta-Adjustment-Sensitivitätsanalyse (m = %d, 21 Werte von delta\n',
    "   zwischen -0,10 und +0,10 pH-Einheiten, Rubin's rules via mice::pool()) zeigt:\n",
    "   das gepoolte Odds Ratio je -0,1 pH-Abfall verlässt den signifikanten Bereich\n",
    "   (KI schließt 1 nicht ein) nur für delta in %s. Der kleinste |delta|, bei\n",
    "   dem das 95-%%-KI die 1 einschließt, ist delta = %s. Die konservative, den\n",
    "   Effekt abschwächende Richtung ist delta %s: sie entspricht der Annahme,\n",
    "   unbeobachtete Patient:innen seien trotz höherer Mortalität pH-mäßig\n",
    '   unauffälliger gewesen als von der MAR-Imputation vorhergesagt."\n'
  ), M, region_txt, lit_txt, conservative))
}

# -----------------------------------------------------------------------
main <- function() {
  section_1_setup()

  df <- sweep(PRIMARY_DELTAS)
  section_2_sweep(df)

  tipping <- section_3_tipping_point(df)

  conservative <- section_4_direction(tipping$df)
  section_5_plausibility(tipping)
  section_6_report(tipping, conservative)
}

main()
