# Module 04 — Reading & extracting data (R / readr, readxl, jsonlite).
#   Rscript module/04-daten-einlesen/code/r.R
# Data: read from data/ (committed with the repo); if that folder is
# missing, the same files are fetched from the published URL.
# Packages: tidyverse (readr), readxl, jsonlite, httr2 (optional for API)
# Code is English; the dataset schema (column names) stays German.

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
})

script <- normalizePath(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1]))
root   <- dirname(dirname(dirname(dirname(script))))
source(file.path(root, "lib", "helpers.R"))

data_dir <- file.path(root, "data")

# ---------------------------------------------------------------------------
# 1) CSV — standard case: explicit options
# ---------------------------------------------------------------------------
cat("=== 1) CSV — standard case (UTF-8, comma) ===\n")

cohort <- read_csv(
  file.path(data_dir, "kohorte.csv"),
  col_types = cols(
    patient_id     = col_integer(),
    alter          = col_integer(),
    diabetes       = col_integer(),
    hypertonie     = col_integer(),
    verstorben_30d = col_integer()
  ),
  na = c("", "NA", "N/A", "-"),
  show_col_types = FALSE
)
cat("Shape:", nrow(cohort), "x", ncol(cohort), "\n")
print(head(cohort, 3))

# ---------------------------------------------------------------------------
# 2) CSV — German format: semicolon, latin-1, decimal comma
# ---------------------------------------------------------------------------
cat("\n=== 2) CSV — German format (semicolon, latin-1, decimal comma) ===\n")

# Build a small demo CSV in German hospital format.
demo_de_path <- file.path(tempdir(), "kohorte_de_demo.csv")
subset_for_demo <- cohort |> select(patient_id, alter, bmi, crp_mg_l) |> head(10)
write.table(subset_for_demo, file = demo_de_path, sep = ";", dec = ",",
            row.names = FALSE, fileEncoding = "latin1")

cohort_de <- read_delim(
  demo_de_path,
  delim = ";",
  locale = locale(encoding = "latin1", decimal_mark = ","),
  show_col_types = FALSE
)
cat("Shape:", nrow(cohort_de), "x", ncol(cohort_de), "\n")
print(head(cohort_de, 3))
file.remove(demo_de_path)

# ---------------------------------------------------------------------------
# 3) Excel — create a synthetic file, then read it back
# ---------------------------------------------------------------------------
cat("\n=== 3) Excel — create and read ===\n")

xl_path <- file.path(tempdir(), "einschluss_demo.xlsx")

if (requireNamespace("writexl", quietly = TRUE)) {
  writexl::write_xlsx(
    list(
      Kohorte = cohort |> select(patient_id, alter, aufnahmegrund, sofa_score) |> head(20),
      Klinik  = cohort |> select(patient_id, crp_mg_l, verweildauer_tage) |> head(20)
    ),
    path = xl_path
  )

  if (requireNamespace("readxl", quietly = TRUE)) {
    sheet_cohort <- readxl::read_excel(xl_path, sheet = "Kohorte")
    cat("Sheet 'Kohorte':", nrow(sheet_cohort), "x", ncol(sheet_cohort), "\n")
    print(head(sheet_cohort, 3))

    sheet_clinic <- readxl::read_excel(xl_path, sheet = "Klinik")
    cat("Sheet 'Klinik':", nrow(sheet_clinic), "x", ncol(sheet_clinic), "\n")
  } else {
    cat("readxl not installed — Excel demo skipped.\n")
  }
  file.remove(xl_path)
} else {
  cat("writexl not installed — Excel demo skipped.\n")
  cat("  Tip: install.packages('writexl') to enable.\n")
}

# ---------------------------------------------------------------------------
# 4) JSON — flat and nested structures
# ---------------------------------------------------------------------------
cat("\n=== 4) JSON — flat and nested ===\n")

if (requireNamespace("jsonlite", quietly = TRUE)) {
  library(jsonlite)

  # Flat JSON (REDCap-like export).
  small <- cohort |> select(patient_id, alter, aufnahmegrund) |> head(5)
  json_text <- toJSON(small, pretty = TRUE)
  df_flat   <- fromJSON(json_text) |> as_tibble()
  cat("Flat JSON read:", nrow(df_flat), "x", ncol(df_flat), "\n")
  print(df_flat)

  # Nested JSON (FHIR-like structure).
  fhir_bundle <- list(
    resourceType = "Bundle",
    entry = lapply(seq_len(3), function(i) {
      list(resource = list(
        id        = as.character(small$patient_id[i]),
        birthYear = 2024L - small$alter[i],
        diagnosis = list(code = small$aufnahmegrund[i])
      ))
    })
  )
  df_fhir <- fromJSON(toJSON(fhir_bundle, auto_unbox = TRUE), flatten = TRUE)$entry |>
    as_tibble()
  cat("\nNested JSON (FHIR-like) normalised:\n")
  print(df_fhir)
} else {
  cat("jsonlite not installed — JSON demo skipped.\n")
}

# ---------------------------------------------------------------------------
# 5) Web API with offline fallback
# ---------------------------------------------------------------------------
cat("\n=== 5) Web API with offline fallback ===\n")

api_url <- "https://disease.sh/v3/covid-19/countries?allowNull=false&limit=10"

df_api <- tryCatch({
  if (!requireNamespace("httr2", quietly = TRUE)) stop("httr2 not available")
  library(httr2)
  response <- request(api_url) |> req_timeout(5) |> req_perform()
  df       <- fromJSON(resp_body_string(response)) |> as_tibble()
  cat("API call succeeded:", nrow(df), "countries\n")
  print(df |> select(country, cases, deaths) |> head(5))
  df
}, error = function(e) {
  cat("Note: API unreachable (", conditionMessage(e), ").\n", sep = "")
  cat("Offline fallback: using local cohort dataset.\n")
  k <- load_cohort()
  cat("  ->", nrow(k), "rows from kohorte.csv loaded.\n")
  k
})

# ---------------------------------------------------------------------------
# 6) Summary: explicit, safe CSV import with missing-value check
# ---------------------------------------------------------------------------
cat("\n=== 6) kohorte.csv — explicit and safe ===\n")

cohort_final <- read_csv(
  file.path(data_dir, "kohorte.csv"),
  col_types = cols(
    patient_id        = col_integer(),
    alter             = col_integer(),
    geschlecht        = col_character(),
    groesse_cm        = col_integer(),
    gewicht_kg        = col_double(),
    bmi               = col_double(),
    aufnahmegrund     = col_character(),
    diabetes          = col_integer(),
    hypertonie        = col_integer(),
    raucherstatus     = col_character(),
    sofa_score        = col_integer(),
    crp_mg_l          = col_double(),
    verweildauer_tage = col_integer(),
    verstorben_30d    = col_integer()
  ),
  na = c("", "NA", "N/A", "-"),
  show_col_types = FALSE
)

cat("Shape:", nrow(cohort_final), "x", ncol(cohort_final), "\n")
missing <- colSums(is.na(cohort_final))
cat("Missing values per column (>0 only):\n")
print(missing[missing > 0])

cat("\nDone.\n")
