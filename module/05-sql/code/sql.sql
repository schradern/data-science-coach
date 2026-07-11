-- Module 05 -- SQL for data extraction
-- Reference queries for DuckDB (executed via python.py or the DuckDB CLI)
--
-- python.py/r.R load the course datasets with the shared pandas/readr
-- loaders (local-first, URL-fallback) and hand them to DuckDB as in-memory
-- tables via con.register()/duckdb_register() — no file path needed here.
--
-- DuckDB can also query a CSV file (local or remote) directly:
--   read_csv_auto('/path/to/kohorte.csv')          -- local path
--   INSTALL httpfs; LOAD httpfs;                    -- once, for URLs
--   read_csv_auto('https://.../kohorte.csv')        -- remote path
-- but that extra INSTALL/LOAD step isn't needed with the register()
-- approach used in this course.
--
-- This file is the readable reference; python.py runs all queries.

------------------------------------------------------------------------
-- 1) Basic query: SELECT, WHERE, ORDER BY, LIMIT
------------------------------------------------------------------------

SELECT
    patient_id,
    alter,
    aufnahmegrund,
    sofa_score
FROM kohorte
WHERE aufnahmegrund = 'Sepsis'
  AND sofa_score >= 6
ORDER BY sofa_score DESC
LIMIT 10;


------------------------------------------------------------------------
-- 2) Aggregation: count, mean SOFA, mortality rate per admission type
------------------------------------------------------------------------

SELECT
    aufnahmegrund,
    COUNT(*)                              AS anzahl,
    ROUND(AVG(sofa_score), 1)            AS sofa_mittel,
    ROUND(AVG(crp_mg_l), 1)             AS crp_mittel,
    ROUND(AVG(verstorben_30d) * 100, 1) AS mortalitaet_pct
FROM kohorte
GROUP BY aufnahmegrund
ORDER BY mortalitaet_pct DESC;


------------------------------------------------------------------------
-- 3) LEFT JOIN: link cohort and lab table
------------------------------------------------------------------------

SELECT
    k.patient_id,
    k.aufnahmegrund,
    k.alter,
    k.sofa_score,
    l.laktat_mmol_l,
    l.kreatinin_mg_dl
FROM kohorte AS k
LEFT JOIN labor AS l ON k.patient_id = l.patient_id
LIMIT 10;


------------------------------------------------------------------------
-- 4) JOIN + GROUP BY: median lactate per admission type
------------------------------------------------------------------------

SELECT
    k.aufnahmegrund,
    COUNT(*)                            AS gesamt,
    COUNT(l.laktat_mmol_l)             AS mit_laktat,
    ROUND(MEDIAN(l.laktat_mmol_l), 2)  AS laktat_median,
    ROUND(AVG(l.laktat_mmol_l), 2)     AS laktat_mittel
FROM kohorte AS k
LEFT JOIN labor AS l ON k.patient_id = l.patient_id
GROUP BY k.aufnahmegrund
ORDER BY laktat_median DESC;


------------------------------------------------------------------------
-- 5) Detect missing values: NULL shares per column (labor)
------------------------------------------------------------------------

-- Portable: repeat COUNT(*) instead of referencing the alias `gesamt` within
-- the same SELECT list. DuckDB allows reusing a SELECT alias in the same list,
-- but PostgreSQL/SQLite do not — and this query is meant to move to a real
-- clinic server. COUNT(*) - COUNT(col) counts the NULLs in `col`.
SELECT
    COUNT(*)                                      AS gesamt,
    COUNT(laktat_mmol_l)                          AS laktat_vorhanden,
    COUNT(*) - COUNT(laktat_mmol_l)               AS laktat_fehlend,
    ROUND(100.0 * (COUNT(*) - COUNT(laktat_mmol_l)) / COUNT(*), 1) AS laktat_fehlend_pct
FROM labor;


------------------------------------------------------------------------
-- 6) CTE: first identify high-risk patients, then summarise lactate
------------------------------------------------------------------------

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
WHERE l.laktat_mmol_l IS NOT NULL;


------------------------------------------------------------------------
-- 7) Vitals: trend — mean heart rate per patient and day
------------------------------------------------------------------------

SELECT
    patient_id,
    tag,
    ROUND(AVG(herzfrequenz), 1) AS hf_mittel,
    ROUND(AVG(map_mmhg), 1)    AS map_mittel
FROM vitalwerte
GROUP BY patient_id, tag
ORDER BY patient_id, tag
LIMIT 20;


------------------------------------------------------------------------
-- 8) Three-way JOIN: cohort + labs + vitals at admission day (tag 0)
------------------------------------------------------------------------

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
LIMIT 10;
