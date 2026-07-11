# Module 09 — Descriptive statistics and 'Table 1' (R / gtsummary)
#
# Runs standalone from the project root:
#   Rscript module/09-deskriptive-statistik/code/r.R
#
# Data: read from data/ (committed with the repo); if that folder is
# missing, the same files are fetched from the published URL.
# Packages: tidyverse, gtsummary

suppressPackageStartupMessages({
  library(tidyverse)
  library(gtsummary)
})

# Resolve project root relative to this script.
.script <- normalizePath(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1]))
.root   <- dirname(dirname(dirname(dirname(.script))))
source(file.path(.root, "lib", "helpers.R"))

cohort <- load_cohort()
labs   <- load_labs()

# Merge cohort + labs (left join keeps all patients).
df <- left_join(cohort, labs, by = "patient_id")
stopifnot(nrow(df) == nrow(cohort))

# Standardise gender encoding (learned in module 06).
df <- df |> mutate(geschlecht = if_else(geschlecht == "w", "weiblich", geschlecht))

# ------------------------------------------------------------------
# 1) Location measures: mean vs. median
# ------------------------------------------------------------------
cat("=== 1) Location measures: mean vs. median ===\n")

# Length of stay is right-skewed — outliers pull the mean upward.
cat(sprintf("Verweildauer – mean:   %.1f days  <- pulled by outliers\n", mean(df$verweildauer_tage)))
cat(sprintf("Verweildauer – median: %.0f days  <- typical value\n",      median(df$verweildauer_tage)))
cat(sprintf("Verweildauer – SD:     %.1f\n",                             sd(df$verweildauer_tage)))
q_los <- quantile(df$verweildauer_tage, c(0.25, 0.75))
cat(sprintf("Verweildauer – IQR:    %.0f – %.0f\n", q_los[1], q_los[2]))

# Age is approximately normal — mean is appropriate.
cat(sprintf("\nAlter – mean +/- SD:    %.1f +/- %.1f years\n", mean(df$alter), sd(df$alter)))
q_age <- quantile(df$alter, c(0.25, 0.75))
cat(sprintf("Alter – median [IQR]:   %.0f [%.0f; %.0f] years\n",
            median(df$alter), q_age[1], q_age[2]))

# ------------------------------------------------------------------
# 2) Spread measures and percentiles
# ------------------------------------------------------------------
cat("\n=== 2) Spread measures and percentiles ===\n")

# CRP is right-skewed — report median [IQR].
cat("CRP mg/l — descriptive summary:\n")
print(summary(df$crp_mg_l))
pct <- quantile(df$crp_mg_l, c(0.10, 0.50, 0.90), na.rm = TRUE)
cat(sprintf("Percentiles 10/50/90: %.1f / %.1f / %.1f mg/l\n", pct[1], pct[2], pct[3]))

qs <- quantile(df$sofa_score, c(0.25, 0.75))
cat(sprintf("\nSOFA-Score – mean +/- SD:   %.1f +/- %.1f\n",  mean(df$sofa_score), sd(df$sofa_score)))
cat(sprintf("SOFA-Score – median [IQR]:  %.0f [%.0f; %.0f]\n", median(df$sofa_score), qs[1], qs[2]))

# ------------------------------------------------------------------
# 3) Frequencies — categorical variables
# ------------------------------------------------------------------
cat("\n=== 3) Frequencies — categorical variables ===\n")

cat("Aufnahmegrund (absolute and relative):\n")
tbl_abs <- table(df$aufnahmegrund)
tbl_rel <- round(prop.table(tbl_abs) * 100, 1)
print(data.frame(n = as.integer(tbl_abs), pct = as.numeric(tbl_rel), row.names = names(tbl_abs)))

cat("\nGeschlecht:\n")
print(table(df$geschlecht))

cat(sprintf("\nProportion with diabetes: %.1f %%\n", mean(df$diabetes) * 100))

# ------------------------------------------------------------------
# 4) Manual group comparison (survivors vs. non-survivors)
# ------------------------------------------------------------------
cat("\n=== 4) Manual group comparison ===\n")

continuous_vars <- c("alter", "sofa_score", "crp_mg_l", "verweildauer_tage",
                     "kreatinin_mg_dl", "laktat_mmol_l")

cat("Median [Q1; Q3] per group (0=survived, 1=died):\n")
df |>
  group_by(verstorben_30d) |>
  summarise(across(
    all_of(continuous_vars),
    ~ sprintf("%.1f [%.1f; %.1f]",
              median(.x, na.rm = TRUE),
              quantile(.x, 0.25, na.rm = TRUE),
              quantile(.x, 0.75, na.rm = TRUE)),
    .names = "{.col}"
  )) |>
  print()

cat("\nDiabetes n (%) per group:\n")
df |>
  group_by(verstorben_30d) |>
  summarise(
    n_diabetes = sum(diabetes),
    proportion = sprintf("%.1f %%", mean(diabetes) * 100)
  ) |>
  print()

# ------------------------------------------------------------------
# 5) Table 1 using gtsummary
# ------------------------------------------------------------------
cat("\n=== 5) Table 1 (gtsummary) ===\n")

df_table <- df |>
  mutate(
    verstorben_30d = factor(verstorben_30d, levels = c(0, 1),
                            labels = c("Überlebt", "Verstorben")),
    diabetes   = factor(diabetes,   levels = c(0, 1), labels = c("nein", "ja")),
    hypertonie = factor(hypertonie, levels = c(0, 1), labels = c("nein", "ja"))
  )

table1 <- df_table |>
  select(alter, geschlecht, aufnahmegrund, diabetes, hypertonie,
         raucherstatus, sofa_score, crp_mg_l, verweildauer_tage,
         kreatinin_mg_dl, laktat_mmol_l, verstorben_30d) |>
  tbl_summary(
    by      = verstorben_30d,
    missing = "ifany",          # always show missing — never hide them
    label   = list(
      alter              ~ "Alter (Jahre)",
      geschlecht         ~ "Geschlecht",
      aufnahmegrund      ~ "Aufnahmegrund",
      diabetes           ~ "Diabetes",
      hypertonie         ~ "Hypertonie",
      raucherstatus      ~ "Raucherstatus",
      sofa_score         ~ "SOFA-Score",
      crp_mg_l           ~ "CRP (mg/l)",
      verweildauer_tage  ~ "Verweildauer (Tage)",
      kreatinin_mg_dl    ~ "Kreatinin (mg/dl)",
      laktat_mmol_l      ~ "Laktat (mmol/l)"
    ),
    statistic = list(
      all_continuous()  ~ "{median} [{p25}; {p75}]",  # right-skewed vars -> median [IQR]
      all_categorical() ~ "{n} ({p}%)"
    )
  ) |>
  add_p() |>       # group comparison (chi-squared / Wilcoxon)
  add_overall()    # add overall "Gesamt" column

print(table1)

# ------------------------------------------------------------------
# 6) Statistical Process Control (SPC) Run Chart
# ------------------------------------------------------------------
cat("\n=== 6) SPC Run Chart ===\n")

# Monthly average wait times in emergency department (24 months)
df_spc <- tibble(
  monat = 1:24,
  wartezeit = c(43.5, 47.2, 41.8, 46.0, 48.5, 42.1, 45.2, 49.0, 44.1, 46.5, 43.0, 45.5,
                34.5, 31.2, 33.8, 30.5, 29.1, 32.4, 35.0, 31.8, 30.0, 32.5, 33.1, 28.5)
)

median_baseline <- median(df_spc$wartezeit[1:12])
cat(sprintf("Baseline-Median (first 12 months): %.2f minutes\n", median_baseline))

# Check if there is a shift (>= 6 consecutive points below median)
below_median <- df_spc$wartezeit < median_baseline
runs <- rle(below_median)
max_consecutive <- max(runs$lengths[runs$values])
cat(sprintf("Max consecutive points below median: %d (Shift if >= 6)\n", max_consecutive))

p_spc <- ggplot(df_spc, aes(x = monat, y = wartezeit)) +
  geom_line(color = "#2A5C8A") +
  geom_point(color = "#2A5C8A") +
  geom_hline(yintercept = median_baseline, linetype = "dashed", color = "gray") +
  geom_point(data = df_spc |> dplyr::filter(monat > 12), color = "#B5482E", size = 2.5) +
  geom_line(data = df_spc |> dplyr::filter(monat > 12), color = "#B5482E") +
  labs(x = "Monat", y = "Wartezeit (Minuten)", title = "SPC Run Chart: Wartezeit Notaufnahme") +
  theme_minimal()

# Figures belong next to the lesson, in assets/ — never in code/.
assets_dir <- file.path(dirname(dirname(.script)), "assets")
dir.create(assets_dir, showWarnings = FALSE)
path_spc <- file.path(assets_dir, "spc_run_chart_demo_r.png")
ggsave(path_spc, p_spc, width = 8, height = 4, dpi = 120)
cat("Saved:", path_spc, "\n")

cat("\nDone.\n")
