# Module 01 – Introduction & learning path (R, parallel to Python).
#   Rscript module/01-einfuehrung/code/r.R
# Data: read from data/ (committed with the repo); if that folder is
# missing, the same files are fetched from the published URL.
# Code is English; the dataset schema (column names) stays German.

suppressPackageStartupMessages(library(dplyr))

script <- normalizePath(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1]))
root   <- dirname(dirname(dirname(dirname(script))))
source(file.path(root, "lib", "helpers.R"))
set.seed(SEED)

cohort <- load_cohort()

cat("=== Module 01 – First look at the cohort ===\n\n")
cat("Dataset loaded:  ", nrow(cohort), "patients,", ncol(cohort), "columns\n")
cat("Columns:         ", paste(names(cohort), collapse = ", "), "\n")

mortality_rate <- mean(cohort$verstorben_30d)
n_deceased     <- sum(cohort$verstorben_30d)
cat(sprintf("\n30-day mortality: %.1f%%  (%d of %d patients)\n",
            mortality_rate * 100, n_deceased, nrow(cohort)))

cat("\nAdmission reasons (by frequency):\n")
print(sort(table(cohort$aufnahmegrund), decreasing = TRUE))

cat("\n30-day mortality by admission reason:\n")
cohort |>
  group_by(aufnahmegrund) |>
  summarise(mortality = mean(verstorben_30d), .groups = "drop") |>
  arrange(desc(mortality)) |>
  mutate(mortality = sprintf("%.1f%%", mortality * 100)) |>
  print()

cat(sprintf("\nSeed: %d  – results are fully reproducible.\n", SEED))
cat("Done. Next: Module 02 – Tools & reproducible environment.\n")
