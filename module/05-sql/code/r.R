# Module 05 — SQL for data extraction (R / DBI + duckdb)
#
# Runs standalone from the project root:
#   Rscript module/05-sql/code/r.R
#
# Data: read from data/ (committed with the repo); if that folder is
# missing, the same files are fetched from the published URL.
# Packages: DBI, duckdb, dplyr
# Install:  install.packages(c("DBI", "duckdb"))

suppressPackageStartupMessages({
  library(dplyr)
})

# Resolve project root relative to this script.
.script <- normalizePath(
  sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1])
)
.root <- dirname(dirname(dirname(dirname(.script))))
source(file.path(.root, "lib", "helpers.R"))

# ---------------------------------------------------------------------------
# Set up DBI + DuckDB
# ---------------------------------------------------------------------------
if (!requireNamespace("DBI", quietly = TRUE) || !requireNamespace("duckdb", quietly = TRUE)) {
  cat("DBI or duckdb not installed.\n")
  cat("This lesson cannot run without them; nothing was computed.\n")
  cat("Run: install.packages(c('DBI', 'duckdb'))\n")
  cat("Full per-module package list: DEPENDENCIES-R.md\n")
  quit(status = 1)
}

library(DBI)
library(duckdb)

con <- dbConnect(duckdb::duckdb(), dbdir = ":memory:")

# Register the datasets as DuckDB tables. Uses the shared, local-first/
# URL-fallback loaders from helpers.R (not a hardcoded local path), then
# hands the resulting data frames to DuckDB via duckdb_register() — no
# read_csv_auto() on a file path needed.
duckdb::duckdb_register(con, "kohorte", load_cohort())
duckdb::duckdb_register(con, "labor", load_labs())
duckdb::duckdb_register(con, "vitalwerte", load_vitals())

cat("=== Table sizes ===\n")
for (tbl in c("kohorte", "labor", "vitalwerte")) {
  n <- dbGetQuery(con, sprintf("SELECT COUNT(*) AS n FROM %s", tbl))$n
  cat(sprintf("  %s: %d rows\n", tbl, n))
}

# ---------------------------------------------------------------------------
# 1) SELECT / WHERE / ORDER BY
# ---------------------------------------------------------------------------
cat("\n=== 1) SELECT / WHERE / ORDER BY — Sepsis, SOFA >= 6 ===\n")
res1 <- dbGetQuery(con, "
  SELECT patient_id, alter, aufnahmegrund, sofa_score
  FROM kohorte
  WHERE aufnahmegrund = 'Sepsis'
    AND sofa_score >= 6
  ORDER BY sofa_score DESC
  LIMIT 10
")
print(res1)

# ---------------------------------------------------------------------------
# 2) GROUP BY — count, mean SOFA, mortality rate per admission type
# ---------------------------------------------------------------------------
cat("\n=== 2) GROUP BY — aggregation per Aufnahmegrund ===\n")
res2 <- dbGetQuery(con, "
  SELECT
      aufnahmegrund,
      COUNT(*)                              AS anzahl,
      ROUND(AVG(sofa_score), 1)            AS sofa_mittel,
      ROUND(AVG(crp_mg_l), 1)             AS crp_mittel,
      ROUND(AVG(verstorben_30d) * 100, 1) AS mortalitaet_pct
  FROM kohorte
  GROUP BY aufnahmegrund
  ORDER BY mortalitaet_pct DESC
")
print(res2)

# ---------------------------------------------------------------------------
# 3) LEFT JOIN — cohort + labs; row count must not increase
# ---------------------------------------------------------------------------
cat("\n=== 3) LEFT JOIN — kohorte + labor (first 5 rows) ===\n")
res3 <- dbGetQuery(con, "
  SELECT
      k.patient_id,
      k.aufnahmegrund,
      k.alter,
      k.sofa_score,
      l.laktat_mmol_l,
      l.kreatinin_mg_dl
  FROM kohorte AS k
  LEFT JOIN labor AS l ON k.patient_id = l.patient_id
  LIMIT 5
")
print(res3)

# Safety check: LEFT JOIN on a 1:1 table must not multiply rows.
n_join   <- dbGetQuery(con,
  "SELECT COUNT(*) AS n FROM kohorte AS k LEFT JOIN labor AS l ON k.patient_id = l.patient_id")$n
n_cohort <- dbGetQuery(con, "SELECT COUNT(*) AS n FROM kohorte")$n
stopifnot(n_join == n_cohort)
cat(sprintf("  Row-count check OK: %d == %d\n", n_join, n_cohort))

# ---------------------------------------------------------------------------
# 4) JOIN + GROUP BY — median lactate per admission type
# ---------------------------------------------------------------------------
cat("\n=== 4) JOIN + GROUP BY — median lactate per Aufnahmegrund ===\n")
res4 <- dbGetQuery(con, "
  SELECT
      k.aufnahmegrund,
      COUNT(*)                            AS gesamt,
      COUNT(l.laktat_mmol_l)             AS mit_laktat,
      ROUND(MEDIAN(l.laktat_mmol_l), 2)  AS laktat_median,
      ROUND(AVG(l.laktat_mmol_l), 2)     AS laktat_mittel
  FROM kohorte AS k
  LEFT JOIN labor AS l ON k.patient_id = l.patient_id
  GROUP BY k.aufnahmegrund
  ORDER BY laktat_median DESC
")
print(res4)

# ---------------------------------------------------------------------------
# 5) NULL shares in labor
# ---------------------------------------------------------------------------
cat("\n=== 5) NULL shares in labor ===\n")
res5 <- dbGetQuery(con, "
  SELECT
      COUNT(*)                                                        AS gesamt,
      COUNT(laktat_mmol_l)                                           AS laktat_vorhanden,
      COUNT(*) - COUNT(laktat_mmol_l)                                AS laktat_fehlend,
      ROUND(100.0 * (COUNT(*) - COUNT(laktat_mmol_l)) / COUNT(*), 1) AS laktat_fehlend_pct
  FROM labor
")
print(res5)

# ---------------------------------------------------------------------------
# 6) CTE — high-risk Sepsis patients (SOFA >= 8)
# ---------------------------------------------------------------------------
cat("\n=== 6) CTE — lactate in high-risk Sepsis (SOFA >= 8) ===\n")
res6 <- dbGetQuery(con, "
  WITH high_risk AS (
      SELECT patient_id
      FROM kohorte
      WHERE sofa_score >= 8
        AND aufnahmegrund = 'Sepsis'
  )
  SELECT
      COUNT(*)                           AS n_hochrisiko,
      ROUND(MEDIAN(l.laktat_mmol_l), 2) AS laktat_median,
      ROUND(AVG(l.laktat_mmol_l), 2)    AS laktat_mittel
  FROM labor AS l
  INNER JOIN high_risk AS h ON l.patient_id = h.patient_id
  WHERE l.laktat_mmol_l IS NOT NULL
")
print(res6)

# ---------------------------------------------------------------------------
# 7) Three-way JOIN — cohort + labs + vitals at admission day
# ---------------------------------------------------------------------------
cat("\n=== 7) Three-way JOIN — kohorte + labor + vitalwerte day 0 (Sepsis) ===\n")
res7 <- dbGetQuery(con, "
  SELECT
      k.patient_id,
      k.aufnahmegrund,
      k.sofa_score,
      l.laktat_mmol_l,
      v.herzfrequenz AS hf_tag0,
      v.map_mmhg     AS map_tag0
  FROM kohorte AS k
  LEFT JOIN labor      AS l ON k.patient_id = l.patient_id
  LEFT JOIN vitalwerte AS v ON k.patient_id = v.patient_id AND v.tag = 0
  WHERE k.aufnahmegrund = 'Sepsis'
  LIMIT 8
")
print(res7)

# ---------------------------------------------------------------------------
# 8) Hand SQL result to dplyr for further processing
# ---------------------------------------------------------------------------
cat("\n=== 8) SQL result passed to dplyr ===\n")
df_base <- dbGetQuery(con, "
  SELECT k.aufnahmegrund, k.sofa_score, l.laktat_mmol_l
  FROM kohorte AS k
  LEFT JOIN labor AS l ON k.patient_id = l.patient_id
  WHERE l.laktat_mmol_l IS NOT NULL
") |> as_tibble()

summary_tbl <- df_base |>
  group_by(aufnahmegrund) |>
  summarise(
    n              = n(),
    sofa_mean      = round(mean(sofa_score), 2),
    lactate_median = round(median(laktat_mmol_l), 2),
    .groups = "drop"
  ) |>
  arrange(desc(lactate_median))

print(summary_tbl)

dbDisconnect(con, shutdown = TRUE)
cat("\nDone.\n")
