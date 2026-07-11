# Module 06 — Data cleaning and transformation (R / tidyverse)
#
# Runs standalone from the project root:
#   Rscript module/06-transformation/code/r.R
#
# Data: read from data/ (committed with the repo); if that folder is
# missing, the same files are fetched from the published URL.

suppressPackageStartupMessages(library(tidyverse))

# Resolve project root relative to this script.
.script <- normalizePath(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1]))
.root   <- dirname(dirname(dirname(dirname(.script))))
source(file.path(.root, "lib", "helpers.R"))

cohort <- load_cohort()
labs   <- load_labs()
vitals <- load_vitals()

# ------------------------------------------------------------------
# 1) Standardise categories ('w' -> 'weiblich')
# ------------------------------------------------------------------
cat("=== 1) Standardise categories ===\n")
cat("before:", paste(sort(unique(cohort$geschlecht)), collapse = ", "), "\n")
cohort <- cohort |> mutate(geschlecht = if_else(geschlecht == "w", "weiblich", geschlecht))
cat("after: ", paste(sort(unique(cohort$geschlecht)), collapse = ", "), "\n")

# ------------------------------------------------------------------
# 2) Count missing values — understand before acting
# ------------------------------------------------------------------
# Three columns, three different mechanisms. Naming all three here keeps the
# printed output, the figure and the README's table in agreement.
cat("\n=== 2) Missing values — understand first ===\n")
merged <- left_join(cohort, labs, by = "patient_id")
mechanisms <- c(
  bmi           = "MCAR — random",
  gewicht_kg    = "MCAR — random",
  laktat_mmol_l = "depends on sofa_score (a covariate)",
  bga_ph        = "depends on verstorben_30d (the OUTCOME)"
)
cat(sprintf("%-16s%9s%9s   %s\n", "column", "missing", "share", "mechanism"))
for (col in names(mechanisms)) {
  n_miss <- sum(is.na(merged[[col]]))
  cat(sprintf("%-16s%9d%8.1f%%   %s\n",
              col, n_miss, 100 * n_miss / nrow(merged), mechanisms[[col]]))
}
cat("Only the mechanism decides whether dropna() is harmless — see Module 14.\n")

# ------------------------------------------------------------------
# 3) Join: cohort + labs (LEFT JOIN keeps all patients)
# ------------------------------------------------------------------
cat("\n=== 3) Join: cohort + labs ===\n")
df <- left_join(cohort, labs, by = "patient_id")
# A left-join on a 1:1 key must not increase the row count.
stopifnot(nrow(df) == nrow(cohort))
cat("Shape after join:", nrow(df), "x", ncol(df), "\n")

# ------------------------------------------------------------------
# 4) Five core verbs: filter, select, derive, group, summarise
# ------------------------------------------------------------------
cat("\n=== 4) Five core verbs ===\n")
sepsis_pts <- df |>
  filter(aufnahmegrund == "Sepsis") |>
  select(patient_id, alter, laktat_mmol_l, sofa_score) |>
  mutate(high_lactate = laktat_mmol_l > 2.0)
cat(sprintf("Sepsis patients: %d  | lactate > 2 mmol/l: %d\n",
            nrow(sepsis_pts), sum(sepsis_pts$high_lactate, na.rm = TRUE)))

cat("\nMedian lactate per admission type:\n")
df |>
  group_by(aufnahmegrund) |>
  summarise(lactate_median = median(laktat_mmol_l, na.rm = TRUE)) |>
  arrange(desc(lactate_median)) |>
  print()

# ------------------------------------------------------------------
# 5) Reshape long -> wide (heart rate per day)
# ------------------------------------------------------------------
cat("\n=== 5) Reshape long -> wide (heart rate per day) ===\n")
hr_wide <- vitals |>
  select(patient_id, tag, herzfrequenz) |>
  pivot_wider(names_from = tag, values_from = herzfrequenz, names_prefix = "hf_tag")
print(head(hr_wide, 3))

cat("\nDone.\n")
