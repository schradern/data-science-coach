# Module 32 – Deployment, monitoring, fairness and governance (R).
#   Rscript module/32-einsatz-governance/code/r.R
# Data: read from data/ (committed with the repo); if that folder is
# missing, the same files are fetched from the published URL.
# Code is English; dataset column names stay German.
# This R script demonstrates the conceptual workflow.
# The primary implementation for this module is code/python.py.

suppressPackageStartupMessages({
  library(tidyverse)
})

script <- normalizePath(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1]))
root   <- dirname(dirname(dirname(dirname(script))))
source(file.path(root, "lib", "helpers.R"))
set.seed(SEED)

notes  <- load_notes()
cohort <- load_cohort()

# Normalise geschlecht: "w" -> "weiblich"
cohort <- cohort |> mutate(geschlecht_clean = case_when(
  geschlecht == "w" ~ "weiblich",
  TRUE ~ geschlecht
))

cat("=== Governance overview ===\n")
cat("Clinical prediction models fall under:\n")
cat("  - MDR 2017/745: Medical Device Regulation (Class IIa+)\n")
cat("  - EU AI Act: high-risk if the model is (part of) a medical device\n")
cat("    requiring notified-body assessment (Art. 6(1), Annex I)\n\n")

cat("=== Dataset overview ===\n")
cat(sprintf("  Notes: %d  |  positive rate: %.1f%%\n",
            nrow(notes), 100 * mean(notes$verschlechterung)))

cat("\n=== Subgroup distribution by Geschlecht ===\n")
merged <- notes |>
  left_join(cohort |> select(patient_id, geschlecht_clean), by = "patient_id")

merged |>
  group_by(geschlecht_clean) |>
  summarise(
    n        = n(),
    positiv  = sum(verschlechterung),
    rate_pct = round(100 * mean(verschlechterung), 1),
    .groups  = "drop"
  ) |>
  print()

cat("\nNote: Full text modelling and AUC subgroup analysis is in code/python.py.\n")
cat("R-based clinical NLP (tidytext + textrecipes) follows the same pipeline principle.\n")
