# Module 03 — Programming fundamentals for data (R, parallel to Python).
#   Rscript module/03-grundlagen/code/r.R
# Data: read from data/ (committed with the repo); if that folder is
# missing, the same files are fetched from the published URL.
# Code is English; the dataset schema (column names) stays German.

suppressPackageStartupMessages(library(dplyr))

script <- normalizePath(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1]))
root   <- dirname(dirname(dirname(dirname(script))))
source(file.path(root, "lib", "helpers.R"))
set.seed(SEED)

# Constant instead of magic number.
FEVER_THRESHOLD <- 38.0  # degrees Celsius

# ---------------------------------------------------------------------- #
#  Helper functions                                                       #
# ---------------------------------------------------------------------- #

has_fever <- function(temperature, threshold = FEVER_THRESHOLD) {
  # Return TRUE when temperature >= threshold.
  # Vectorised automatically in R.
  temperature >= threshold
}

bmi_category <- function(bmi) {
  # Classify BMI values according to WHO categories (vectorised via case_when).
  case_when(
    is.na(bmi) ~ "unbekannt",
    bmi >= 30  ~ "adipös",
    bmi >= 25  ~ "übergewichtig",
    bmi >= 18.5 ~ "normalgewichtig",
    .default   = "untergewichtig"
  )
}

assess_map <- function(map_mmhg) {
  # Classify mean arterial pressure by clinical thresholds (vectorised).
  case_when(
    map_mmhg < 65  ~ "Schock-Grenzwert unterschritten",
    map_mmhg <= 90 ~ "Normbereich",
    .default       = "Erhöht"
  )
}

# ---------------------------------------------------------------------- #
#  1) Variables and data types                                            #
# ---------------------------------------------------------------------- #
cat("=== 1) Variables and data types ===\n")

age             <- 67L          # integer — patient age
temperature     <- 38.9         # numeric — body temperature
admission_reason <- "Sepsis"    # character — diagnosis
has_diabetes    <- TRUE         # logical — comorbidity flag

cat("age:             ", age, " (", class(age), ")\n", sep = "")
cat("temperature:     ", temperature, " (", class(temperature), ")\n", sep = "")
cat("admission_reason:", admission_reason, " (", class(admission_reason), ")\n", sep = "")
cat("has_diabetes:    ", has_diabetes, " (", class(has_diabetes), ")\n", sep = "")

# ---------------------------------------------------------------------- #
#  2) Vectors and lists                                                   #
# ---------------------------------------------------------------------- #
cat("\n=== 2) Vectors and lists ===\n")

# Vector — all elements share the same type.
temperatures <- c(36.8, 37.2, 38.9, 36.5, 39.4, 37.1)

cat("Measurements:", paste(temperatures, collapse = ", "), "\n")
cat("Count:", length(temperatures), "\n")
cat("Maximum:", max(temperatures), "\n")
cat(sprintf("Mean: %.2f\n", mean(temperatures)))

# Indexing — R counts from 1.
cat("First measurement (index 1):", temperatures[1], "\n")
cat("Last measurement:", temperatures[length(temperatures)], "\n")

# List — can hold mixed types (analogous to dict in Python).
patient <- list(
  id             = 42L,
  age            = 71L,
  aufnahmegrund  = "Herzinsuffizienz",
  sofa_score     = 5L
)
cat(sprintf("\nPatient %d: %s, age %d, SOFA %d\n",
            patient$id, patient$aufnahmegrund, patient$age, patient$sofa_score))

# ---------------------------------------------------------------------- #
#  3) Functions                                                           #
# ---------------------------------------------------------------------- #
cat("\n=== 3) Functions ===\n")

cat("has_fever(38.9) ->", has_fever(38.9), "\n")  # TRUE
cat("has_fever(37.2) ->", has_fever(37.2), "\n")  # FALSE

# In R, has_fever() works directly on a whole vector.
fever_readings <- temperatures[has_fever(temperatures)]
cat("Fever readings:", paste(fever_readings, collapse = ", "), "\n")
cat(sprintf("Fraction with fever: %.0f%%\n", mean(has_fever(temperatures)) * 100))

# BMI categorisation (vectorised via case_when).
sample_bmis <- c(17.5, 22.3, 27.1, 34.8)
cats <- bmi_category(sample_bmis)
for (i in seq_along(sample_bmis)) {
  cat(sprintf("  BMI %5.1f -> %s\n", sample_bmis[i], cats[i]))
}

# ---------------------------------------------------------------------- #
#  4) Control flow                                                        #
# ---------------------------------------------------------------------- #
cat("\n=== 4) Control flow ===\n")

# if/else — fever severity tiers (scalar decisions).
for (temp in c(37.2, 38.4, 39.6)) {
  if (temp >= 39.0) {
    level <- "Hohes Fieber — ärztliche Beurteilung erforderlich"
  } else if (temp >= FEVER_THRESHOLD) {
    level <- "Fieber"
  } else {
    level <- "Kein Fieber"
  }
  cat(sprintf("  %.1f °C -> %s\n", temp, level))
}

# MAP assessment — vectorised operations are idiomatic in R.
cat("\nMAP assessment (sample values):\n")
map_values <- c(58, 72, 95, 63, 88)
results    <- assess_map(map_values)
for (i in seq_along(map_values)) {
  cat(sprintf("  MAP %3d mmHg -> %s\n", map_values[i], results[i]))
}

# ---------------------------------------------------------------------- #
#  5) DataFrames — tables in code                                         #
# ---------------------------------------------------------------------- #
cat("\n=== 5) DataFrames — tables in code ===\n")

cohort <- load_cohort()

cat("Shape (rows x cols):", nrow(cohort), "x", ncol(cohort), "\n")
cat("\nData types per column (glimpse):\n")
glimpse(cohort)

cat("\nFirst 5 rows (selected columns):\n")
print(cohort |> select(patient_id, alter, aufnahmegrund, sofa_score, verstorben_30d) |> head(5))

# Filter rows — only patients with Sepsis.
septic <- cohort |> filter(aufnahmegrund == "Sepsis")
cat(sprintf("\nSeptic patients: %d\n", nrow(septic)))
cat(sprintf("Deceased within 30 days: %d\n", sum(septic$verstorben_30d)))

# Derive a new column — BMI category.
cohort <- cohort |> mutate(bmi_cat = bmi_category(bmi))
cat("\nBMI category distribution:\n")
print(table(cohort$bmi_cat))

# Descriptive statistics.
cat("\nAge — descriptive statistics:\n")
print(summary(cohort$alter))

# Group comparison.
cat("\nMean age by admission reason:\n")
cohort |>
  group_by(aufnahmegrund) |>
  summarise(mean_age = round(mean(alter), 1), .groups = "drop") |>
  arrange(desc(mean_age)) |>
  print()

cat(sprintf("\nSeed used: %d\nDone.\n", SEED))
