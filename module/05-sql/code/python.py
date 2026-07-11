"""Module 05 — SQL for data extraction using DuckDB.

Runs standalone from the project root:
    python module/05-sql/code/python.py

Data: read from data/ (committed with the repo); if that folder is
missing, the same files are fetched from the published URL.
Package required: pip install duckdb

DuckDB lets us run SQL directly on CSV files — no database server needed.
The queries mirror the examples in sql.sql.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import duckdb  # noqa: E402
import pandas as pd  # noqa: E402

from lib.helpers import SEED, load_cohort, load_labs, load_vitals  # noqa: E402

pd.set_option("display.width", 100)
pd.set_option("display.max_columns", 10)


def connect() -> duckdb.DuckDBPyConnection:
    """Open an in-memory DuckDB connection and register the datasets as tables.

    Uses the shared, local-first/URL-fallback loaders from lib.helpers (not a
    hardcoded local path), then hands the resulting DataFrames to DuckDB via
    con.register() — no `read_csv_auto()` on a file path needed.
    """
    con = duckdb.connect(":memory:")
    con.register("kohorte", load_cohort())
    con.register("labor", load_labs())
    con.register("vitalwerte", load_vitals())
    return con


def main() -> None:
    con = connect()

    print("=== Table sizes ===")
    for name in ("kohorte", "labor", "vitalwerte"):
        n = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        print(f"  {name}: {n} rows")

    # ------------------------------------------------------------------
    # 1) SELECT / WHERE / ORDER BY / LIMIT
    # ------------------------------------------------------------------
    print("\n=== 1) SELECT / WHERE / ORDER BY — Sepsis, SOFA >= 6 ===")
    df1 = con.execute("""
        SELECT patient_id, alter, aufnahmegrund, sofa_score
        FROM kohorte
        WHERE aufnahmegrund = 'Sepsis'
          AND sofa_score >= 6
        ORDER BY sofa_score DESC
        LIMIT 10
    """).df()
    print(df1.to_string(index=False))

    # ------------------------------------------------------------------
    # 2) GROUP BY — count, mean SOFA, mortality rate per admission type
    # ------------------------------------------------------------------
    print("\n=== 2) GROUP BY — aggregation per Aufnahmegrund ===")
    df2 = con.execute("""
        SELECT
            aufnahmegrund,
            COUNT(*)                              AS anzahl,
            ROUND(AVG(sofa_score), 1)            AS sofa_mittel,
            ROUND(AVG(crp_mg_l), 1)             AS crp_mittel,
            ROUND(AVG(verstorben_30d) * 100, 1) AS mortalitaet_pct
        FROM kohorte
        GROUP BY aufnahmegrund
        ORDER BY mortalitaet_pct DESC
    """).df()
    print(df2.to_string(index=False))

    # ------------------------------------------------------------------
    # 3) LEFT JOIN — cohort + labs; row count must stay the same
    # ------------------------------------------------------------------
    print("\n=== 3) LEFT JOIN — kohorte + labor (first 5 rows) ===")
    df3 = con.execute("""
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
    """).df()
    print(df3.to_string(index=False))

    # Safety check: a LEFT JOIN on a 1:1 table must not multiply rows.
    n_join   = con.execute(
        "SELECT COUNT(*) FROM kohorte AS k LEFT JOIN labor AS l ON k.patient_id = l.patient_id"
    ).fetchone()[0]
    n_cohort = con.execute("SELECT COUNT(*) FROM kohorte").fetchone()[0]
    assert n_join == n_cohort, f"Row count changed after JOIN: {n_join} != {n_cohort}"
    print(f"  Row-count check OK: {n_join} == {n_cohort}")

    # ------------------------------------------------------------------
    # 4) JOIN + GROUP BY — median lactate per admission type
    # ------------------------------------------------------------------
    print("\n=== 4) JOIN + GROUP BY — median lactate per Aufnahmegrund ===")
    df4 = con.execute("""
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
    """).df()
    print(df4.to_string(index=False))

    # ------------------------------------------------------------------
    # 5) NULL analysis — how many lactate values are missing?
    # ------------------------------------------------------------------
    print("\n=== 5) NULL shares in labor ===")
    df5 = con.execute("""
        SELECT
            COUNT(*)                                                       AS gesamt,
            COUNT(laktat_mmol_l)                                          AS laktat_vorhanden,
            COUNT(*) - COUNT(laktat_mmol_l)                               AS laktat_fehlend,
            ROUND(100.0 * (COUNT(*) - COUNT(laktat_mmol_l)) / COUNT(*), 1) AS laktat_fehlend_pct
        FROM labor
    """).df()
    print(df5.to_string(index=False))

    # ------------------------------------------------------------------
    # 6) CTE — high-risk Sepsis patients (SOFA >= 8)
    # ------------------------------------------------------------------
    print("\n=== 6) CTE — lactate in high-risk Sepsis (SOFA >= 8) ===")
    df6 = con.execute("""
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
    """).df()
    print(df6.to_string(index=False))

    # ------------------------------------------------------------------
    # 7) Three-way JOIN — cohort + labs + vitals at admission day
    # ------------------------------------------------------------------
    print("\n=== 7) Three-way JOIN — kohorte + labor + vitalwerte day 0 (Sepsis) ===")
    df7 = con.execute("""
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
    """).df()
    print(df7.to_string(index=False))

    # ------------------------------------------------------------------
    # 8) Hand SQL result to pandas for further processing
    # ------------------------------------------------------------------
    print("\n=== 8) SQL result passed to pandas for groupby summary ===")
    df_base = con.execute("""
        SELECT k.aufnahmegrund, k.sofa_score, l.laktat_mmol_l
        FROM kohorte AS k
        LEFT JOIN labor AS l ON k.patient_id = l.patient_id
        WHERE l.laktat_mmol_l IS NOT NULL
    """).df()

    summary = (
        df_base
        .groupby("aufnahmegrund")
        .agg(n=("sofa_score", "count"),
             sofa_mean=("sofa_score", "mean"),
             lactate_median=("laktat_mmol_l", "median"))
        .round(2)
        .sort_values("lactate_median", ascending=False)
    )
    print(summary)

    con.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
