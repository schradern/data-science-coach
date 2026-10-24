# Module 04 – Reading & extracting data (R / readr, readxl, jsonlite).
#   Rscript module/04-daten-einlesen/code/r.R
# Data: the same companion files the lesson reads (kohorte_de.csv,
# einschluss.xlsx, beispiel.json, fhir_bundle.json), taken from data/ when the
# repo is cloned and otherwise fetched from the published URL.
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

# Local copy in data/ if present, else the published URL of the same file.
source_of <- function(name) {
  local <- file.path(data_dir, name)
  if (file.exists(local)) local else paste0(DATA_BASE_URL, name)
}

# ---------------------------------------------------------------------------
# 1) CSV – standard case: explicit options
# ---------------------------------------------------------------------------
cat("=== 1) CSV – standard case (UTF-8, comma) ===\n")

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
# 2) CSV – German format: semicolon, latin-1, decimal comma
# ---------------------------------------------------------------------------
cat("\n=== 2) CSV – German format (semicolon, latin-1, decimal comma) ===\n")

cohort_de <- read_delim(
  source_of("kohorte_de.csv"),
  delim = ";",
  locale = locale(encoding = "latin1", decimal_mark = ","),
  show_col_types = FALSE
)
cat("Shape:", nrow(cohort_de), "x", ncol(cohort_de), "\n")
print(head(select(cohort_de, patient_id, geschlecht, gewicht_kg, bmi), 3))

# ---------------------------------------------------------------------------
# 3) Excel – one workbook, three sheets
# ---------------------------------------------------------------------------
cat("\n=== 3) Excel – einschluss.xlsx ===\n")

if (requireNamespace("readxl", quietly = TRUE)) {
  xl_path <- source_of("einschluss.xlsx")
  if (startsWith(xl_path, "http")) {          # readxl cannot read a URL
    tmp <- tempfile(fileext = ".xlsx")
    download.file(xl_path, tmp, mode = "wb", quiet = TRUE)
    xl_path <- tmp
  }
  cat("Sheets:", paste(readxl::excel_sheets(xl_path), collapse = ", "), "\n")
  sheet_labs <- readxl::read_excel(xl_path, sheet = "Labor", range = readxl::cell_cols("A:F"))
  cat("Sheet 'Labor':", nrow(sheet_labs), "x", ncol(sheet_labs), "\n")
  print(head(sheet_labs, 3))
} else {
  cat("readxl not installed – Excel demo skipped.\n")
}

# ---------------------------------------------------------------------------
# 4) JSON – flat and nested structures
# ---------------------------------------------------------------------------
cat("\n=== 4) JSON – flat and nested ===\n")

if (requireNamespace("jsonlite", quietly = TRUE)) {
  library(jsonlite)

  # Flat JSON (REDCap-like export).
  df_flat <- fromJSON(source_of("beispiel.json")) |> as_tibble()
  cat("Flat JSON (beispiel.json):", nrow(df_flat), "x", ncol(df_flat), "\n")
  print(head(df_flat, 3))

  # Nested JSON (FHIR bundle).
  bundle  <- fromJSON(source_of("fhir_bundle.json"), flatten = TRUE)
  df_fhir <- as_tibble(bundle$entry)
  cat("\nNested JSON (fhir_bundle.json) normalised:", nrow(df_fhir), "x", ncol(df_fhir), "\n")
  cat("Columns:", paste(head(names(df_fhir), 6), collapse = ", "), "...\n")
} else {
  cat("jsonlite not installed – JSON demo skipped.\n")
}

# ---------------------------------------------------------------------------
# 5) Web API with offline fallback
# ---------------------------------------------------------------------------
cat("\n=== 5) Web API with offline fallback ===\n")

api_url <- "https://disease.sh/v3/covid-19/countries?allowNull=false&limit=10"

df_api <- tryCatch({
  if (!requireNamespace("httr2", quietly = TRUE)) stop("httr2 not available")
  library(httr2)
  response <- request(api_url) |> req_user_agent("data-science-coach/1.0") |>
    req_timeout(5) |> req_perform()
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
cat("\n=== 6) kohorte.csv – explicit and safe ===\n")

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
