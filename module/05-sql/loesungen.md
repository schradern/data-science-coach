# Musterlösungen · Datenbank-Abfragen mit SQL

> Vergleiche zuerst mit deinem eigenen Versuch. Es gibt oft mehrere richtige Wege.
> Alle Abfragen laufen via DuckDB über die registrierten Tabellen (`kohorte`, `labor`, `vitalwerte`).
> Führe zuerst dieses Setup aus (siehe README §2), dann bleibt `con` für alle
> folgenden Aufgaben verfügbar:
> ```python
> import duckdb
> import pandas as pd
>
> con = duckdb.connect(":memory:")
> con.register("kohorte", pd.read_csv("{{DATEN}}kohorte.csv"))
> con.register("labor", pd.read_csv("{{DATEN}}labor.csv"))
> con.register("vitalwerte", pd.read_csv("{{DATEN}}vitalwerte.csv"))
> ```

## Aufgabe 1 – SELECT und WHERE

```python
result = con.execute("""
    SELECT patient_id, alter, aufnahmegrund, crp_mg_l, sofa_score
    FROM kohorte
    WHERE crp_mg_l > 100 AND sofa_score < 5
    ORDER BY crp_mg_l DESC
""").df()
print(len(result), "patients")
print(result["aufnahmegrund"].value_counts())
```
```sql
-- SQL-Variante (direkt in DuckDB-CLI)
SELECT aufnahmegrund, COUNT(*) AS n
FROM kohorte
WHERE crp_mg_l > 100 AND sofa_score < 5
GROUP BY aufnahmegrund
ORDER BY n DESC;
```

## Aufgabe 2 – GROUP BY

```sql
SELECT
    raucherstatus,
    COUNT(*)                              AS anzahl,
    ROUND(AVG(verweildauer_tage), 1)     AS verweildauer_mittel,
    ROUND(MEDIAN(crp_mg_l), 1)           AS crp_median,
    ROUND(AVG(verstorben_30d) * 100, 1) AS mortalitaet_pct
FROM kohorte
GROUP BY raucherstatus
ORDER BY mortalitaet_pct DESC;
```
```python
result = con.execute("""
    SELECT raucherstatus,
           COUNT(*) AS anzahl,
           ROUND(AVG(verweildauer_tage), 1)     AS verweildauer_mittel,
           ROUND(MEDIAN(crp_mg_l), 1)           AS crp_median,
           ROUND(AVG(verstorben_30d) * 100, 1) AS mortalitaet_pct
    FROM kohorte
    GROUP BY raucherstatus
    ORDER BY mortalitaet_pct DESC
""").df()
print(result)
```

## Aufgabe 3 – JOIN

```sql
-- Ohne Kreatininwert
SELECT COUNT(*) AS ohne_kreatinin
FROM kohorte AS k
LEFT JOIN labor AS l ON k.patient_id = l.patient_id
WHERE l.kreatinin_mg_dl IS NULL;

-- Kreatinin > 2.0
SELECT COUNT(*) AS kreatinin_hoch
FROM kohorte AS k
LEFT JOIN labor AS l ON k.patient_id = l.patient_id
WHERE l.kreatinin_mg_dl > 2.0;
```
```python
result = con.execute("""
    SELECT
        COUNT(*) FILTER (WHERE l.kreatinin_mg_dl IS NULL)    AS ohne_kreatinin,
        COUNT(*) FILTER (WHERE l.kreatinin_mg_dl > 2.0)      AS kreatinin_hoch
    FROM kohorte AS k
    LEFT JOIN labor AS l ON k.patient_id = l.patient_id
""").df()
print(result)
```

## Aufgabe 4 – NULL und HAVING

```sql
SELECT
    k.aufnahmegrund,
    COUNT(l.laktat_mmol_l)            AS n_laktat,
    ROUND(AVG(l.laktat_mmol_l), 2)   AS laktat_mittel
FROM kohorte AS k
LEFT JOIN labor AS l ON k.patient_id = l.patient_id
WHERE l.laktat_mmol_l IS NOT NULL
GROUP BY k.aufnahmegrund
HAVING COUNT(l.laktat_mmol_l) >= 30
ORDER BY laktat_mittel DESC;
```

## Aufgabe 5 – Dreifach-JOIN

```sql
SELECT
    k.patient_id,
    k.aufnahmegrund,
    l.laktat_mmol_l,
    v.herzfrequenz AS hf_tag1
FROM kohorte AS k
LEFT JOIN labor      AS l ON k.patient_id = l.patient_id
LEFT JOIN vitalwerte AS v ON k.patient_id = v.patient_id AND v.tag = 1
WHERE k.diabetes = 1
  AND l.laktat_mmol_l IS NOT NULL
  AND v.herzfrequenz IS NOT NULL
LIMIT 20;
```
```python
result = con.execute("""
    SELECT COUNT(*) AS n_vollstaendig
    FROM kohorte AS k
    LEFT JOIN labor      AS l ON k.patient_id = l.patient_id
    LEFT JOIN vitalwerte AS v ON k.patient_id = v.patient_id AND v.tag = 1
    WHERE k.diabetes = 1
      AND l.laktat_mmol_l IS NOT NULL
      AND v.herzfrequenz  IS NOT NULL
""").df()
print(result)
```

## Bonus – CTE

```sql
WITH schwer_krank AS (
    SELECT patient_id, 'schwer' AS gruppe
    FROM kohorte WHERE sofa_score >= 7
),
rest AS (
    SELECT patient_id, 'leicht' AS gruppe
    FROM kohorte WHERE sofa_score < 7
),
alle AS (
    SELECT * FROM schwer_krank
    UNION ALL
    SELECT * FROM rest
)
SELECT
    a.gruppe,
    COUNT(*)                           AS n,
    ROUND(MEDIAN(l.kreatinin_mg_dl), 2) AS kreatinin_median,
    ROUND(MEDIAN(l.laktat_mmol_l), 2)   AS laktat_median
FROM alle AS a
LEFT JOIN labor AS l ON a.patient_id = l.patient_id
GROUP BY a.gruppe
ORDER BY kreatinin_median DESC;
```
**Beobachtung:** Schwer kranke Patient:innen (SOFA ≥ 7, n=49) zeigen einen
höheren medianen Laktatwert (2,1 vs. 1,8 mmol/l) als der Rest, das Kreatinin
unterscheidet sich in dieser Kohorte dagegen kaum (0,93 vs. 1,05 mg/dl). Nur
die Laktat-Messwahrscheinlichkeit (nicht der Kreatininwert) ist im Datensatz
absichtlich an den SOFA-Score gekoppelt (siehe `data/README.md`), ein gutes
Beispiel dafür, dass nicht jede klinisch plausible Assoziation im Datensatz
tatsächlich eingebaut ist, immer nachrechnen statt nur vermuten.
