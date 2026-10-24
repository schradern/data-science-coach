# Module 02 – Tools & reproducible environment (R, parallel to Python).
#   Rscript module/02-werkzeuge/code/r.R
# Data: read from data/ (committed with the repo); if that folder is
# missing, the same files are fetched from the published URL.
# Checks that key packages are installed and the dataset is reachable.
# Code is English; the dataset schema (column names) stays German.

script <- normalizePath(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1]))
root   <- dirname(dirname(dirname(dirname(script))))
source(file.path(root, "lib", "helpers.R"))

cat("=== Module 02 – Environment check (R) ===\n\n")
cat("R version:  ", R.version.string, "\n")
cat("Seed:       ", SEED, "\n")

required_packages <- c("tidyverse", "gtsummary", "broom", "survival", "readxl", "jsonlite")
cat("\nInstalled packages:\n")
all_ok <- TRUE

for (pkg in required_packages) {
  if (requireNamespace(pkg, quietly = TRUE)) {
    ver <- as.character(packageVersion(pkg))
    cat(sprintf("  %-20s %s  [OK]\n", pkg, ver))
  } else {
    cat(sprintf("  %-20s %-20s [MISSING]\n", pkg, "---"))
    all_ok <- FALSE
  }
}

suppressPackageStartupMessages(library(tidyverse))

cat("\nDataset check:\n")
tryCatch({
  cohort <- load_cohort()
  cat(sprintf("  kohorte.csv  -> %d rows x %d cols  [OK]\n", nrow(cohort), ncol(cohort)))
}, error = function(e) {
  cat("  ERROR:", conditionMessage(e), "\n")
  all_ok <<- FALSE
})

cat("\n")
if (all_ok) {
  cat("All checks passed. Environment is ready.\n")
} else {
  cat("Some packages are missing. Install with:\n")
  cat('  install.packages(c("tidyverse","gtsummary","broom","survival","readxl","jsonlite"))\n')
  cat('  install.packages(c("tidymodels","recipes","ranger","nnet","tidytext"))  # later ML modules\n')
}
