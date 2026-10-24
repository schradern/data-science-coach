# Module 22 – Reproducible reports and workflows (R)
#
# Demonstrates:
#   1. set.seed() for reproducibility
#   2. Reproducible slice_sample() using SEED
#   3. sessionInfo() and per-package version logging
#   4. Descriptive summary grouped by clinical variable
#
# Runs standalone from the project root:
#   Rscript module/22-reproduzierbare-berichte/code/r.R
#
# Data: read from data/ (committed with the repo); if that folder is
# missing, the same files are fetched from the published URL.

suppressPackageStartupMessages(library(tidyverse))

# ---------------------------------------------------------------------------
# Locate project root robustly (works via Rscript and in RStudio)
# ---------------------------------------------------------------------------
.this_script <- tryCatch(
  normalizePath(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1])),
  error = function(e) NULL
)

root <- if (!is.null(.this_script) && !is.na(.this_script)) {
  # Four levels up: code/ -> 22-.../ -> module/ -> project root
  dirname(dirname(dirname(dirname(.this_script))))
} else {
  normalizePath(getwd())
}

source(file.path(root, "lib", "helpers.R"))

# ---------------------------------------------------------------------------
# 1) Set the seed – before any random operation
# ---------------------------------------------------------------------------
set.seed(SEED)
cat(sprintf("=== Seed gesetzt: %d ===\n\n", SEED))

# ---------------------------------------------------------------------------
# 2) Load data
# ---------------------------------------------------------------------------
cohort <- load_cohort()
cat(sprintf("Kohorte geladen: %d Patient:innen, %d Spalten\n",
            nrow(cohort), ncol(cohort)))

# ---------------------------------------------------------------------------
# 3) Reproducible random sample
# ---------------------------------------------------------------------------
cat("\n=== Reproduzierbare Stichprobe (n=50) ===\n")
sample_50 <- cohort |> slice_sample(n = 50)
cat(sprintf("Mittleres Alter:       %.1f Jahre\n", mean(sample_50$alter)))
cat(sprintf("30-Tage-Mortalitaet:  %.1f %%\n", mean(sample_50$verstorben_30d) * 100))
cat(sprintf("Erste patient_id:      %d\n", sample_50$patient_id[1]))
cat("(Dieser Wert aendert sich nicht, solange set.seed(42) gesetzt ist.)\n\n")

# ---------------------------------------------------------------------------
# 4) Log package versions
# ---------------------------------------------------------------------------
cat("=== Paket-Manifest ===\n")
si <- sessionInfo()
cat(sprintf("R-Version:  %s\n", si$R.version$version.string))

key_packages <- c("tidyverse", "dplyr", "tidyr", "ggplot2", "broom")
for (pkg in key_packages) {
  ver <- tryCatch(
    as.character(packageVersion(pkg)),
    error = function(e) "nicht installiert"
  )
  cat(sprintf("  %-20s: %s\n", pkg, ver))
}

cat("\nTipp: sessionInfo() am Ende eines R-Markdown- oder Quarto-Dokuments\n")
cat("      gibt alle Versionen automatisch aus – Best Practice fuer Publikationen.\n")

# ---------------------------------------------------------------------------
# 5) Descriptive summary grouped by admission reason
# ---------------------------------------------------------------------------
cat("\n=== Deskriptive Zusammenfassung der Stichprobe ===\n")
summary_tbl <- sample_50 |>
  group_by(aufnahmegrund) |>
  summarise(
    n            = n(),
    alter_median = median(alter),
    mortalitaet  = round(mean(verstorben_30d), 3),
    .groups      = "drop"
  )
print(as.data.frame(summary_tbl))

cat("\n=== Reproduzierbarkeits-Prinzipien (R) ===\n")
cat("  - set.seed() vor jeder Zufallsoperation\n")
cat("  - relative Pfade via load_cohort() / load_labs()\n")
cat("  - sessionInfo() im Bericht dokumentieren\n")
cat("  - .Rmd / .qmd: Text und Code in einer Datei\n")
cat("\nFertig.\n")
