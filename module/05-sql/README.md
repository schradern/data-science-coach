# 05 · Datenbank-Abfragen mit SQL

> **Teil 2 – Datenimport, Datenbereinigung und Datenmanagement** · Dauer: ~55 min · Voraussetzung: Modul 04 (Datenimport aus Dateien und APIs)

## Lernziele
- Das **relationale Datenmodell** verstehen: Tabellen, Schlüssel, Beziehungen.
- Die wichtigsten SQL-Befehle sicher anwenden: `SELECT`, `WHERE`, `JOIN`,
 `GROUP BY` und **Aggregatfunktionen** (`COUNT`, `AVG`, `MAX`, `SUM`).

- SQL direkt über **DuckDB** auf CSV-Dateien ausführen, ohne Datenbankserver.
- Abfragen aus **Python** (`duckdb`) und **R** (`DBI` + `duckdb`) heraus starten.
- Den Unterschied zwischen `LEFT JOIN` und `INNER JOIN` verstehen und richtig
 wählen.

## Klinischer Aufhänger
Ihr Rechenzentrum gibt dir schreibgeschützten Zugriff auf die Klinikdatenbank, nur via SQL-Query. Um die Frage *„Unterscheidet sich das mittlere Laktat je nach
Aufnahmegrund?"* zu beantworten, musst du `kohorte` und `labor` verbinden und
gruppiert aggregieren. Mit SQL geht das in vier Zeilen.

## 1 Relationales Denken

Eine **relationale Datenbank** speichert Daten in mehreren Tabellen. Jede Tabelle
hat einen **Primärschlüssel** (eindeutige ID je Zeile). Tabellen werden über
**Fremdschlüssel** verknüpft.

![Abbildung: Datenmodell, die Tabelle `kohorte` enthält eine Zeile pro Patient:in (Primärschlüssel `patient_id`). Die Messtabellen `vitalwerte`, `labor` und `notizen` verweisen über den Fremdschlüssel `patient_id` zurück und können viele Zeilen je Patient:in enthalten (1:∞).](assets/datenmodell.svg)

In unserem Datensatz ist `patient_id` der gemeinsame Schlüssel zwischen
`kohorte.csv`, `labor.csv` und `vitalwerte.csv`.

Mit **DuckDB** führst du SQL direkt auf CSV-Dateien aus, ohne Datenbankserver,
ideal zum Erkunden.

**Stolperstein:** Hat eine Tabelle mehrere Zeilen pro Patient:in (etwa vier
Messtage in `vitalwerte`), vervielfacht ein JOIN auf `kohorte` deren Zeilen.
Prüfe `COUNT(*)` vor und nach dem JOIN.
**Darüber hinaus:** DuckDB unterstützt `WINDOW`-Funktionen (`ROW_NUMBER()`,
`LAG()`, `LEAD()`), mit denen sich Verlaufsdaten direkt in SQL analysieren lassen,
ohne erst nach Python oder R exportieren zu müssen.

## 2 SELECT, WHERE, ORDER BY

Die grundlegendste SQL-Abfrage:

```sql
SELECT patient_id, alter, aufnahmegrund, sofa_score
FROM kohorte
WHERE aufnahmegrund = 'Sepsis'
  AND sofa_score >= 6
ORDER BY sofa_score DESC
LIMIT 10;
```

<div class="code-verbalization" markdown="1">
🗣 **Code-Verbalisierung:**

* `SELECT patient_id, alter, ...`: Wähle die Spalten `patient_id`, `alter`, `aufnahmegrund` und `sofa_score` aus.
* `FROM kohorte`: Beziehe diese Spalten aus der Tabelle namens `kohorte`.
* `WHERE aufnahmegrund = 'Sepsis' AND sofa_score >= 6`: Filter die Zeilen so, dass nur Patient:innen mit dem Aufnahmegrund `'Sepsis'` **und** einem `sofa_score` von mindestens 6 übrig bleiben.
* `ORDER BY sofa_score DESC`: Sortiere das Ergebnis nach dem `sofa_score` in absteigender Reihenfolge (höchster Score zuerst).
* `LIMIT 10`: Begrenze die Ausgabe auf maximal 10 Zeilen.
</div>

- `SELECT` – welche Spalten?
- `FROM` – aus welcher Tabelle?
- `WHERE` – welche Zeilen? (Bedingungen mit `AND`, `OR`, `NOT`)
- `ORDER BY` – sortieren (Standard: aufsteigend, `DESC` für absteigend)
- `LIMIT` – maximale Zeilenzahl

DuckDB kann keine Web-Adresse direkt als Tabelle öffnen, wohl aber ein
`pandas`-DataFrame oder R-`data.frame`, das du zuvor ganz normal (Modul 04)
geladen hast. Das Muster ist deshalb immer: **erst mit `pandas`/`readr` von
`{{DATEN}}` laden, dann bei DuckDB registrieren.** Die `con`-Verbindung aus
diesem Abschnitt bleibt für den Rest des Moduls offen, alle folgenden
Abfragen bauen darauf auf.

### In Python (DuckDB)

```python
import duckdb
import pandas as pd

cohort = pd.read_csv("{{DATEN}}kohorte.csv")
labor  = pd.read_csv("{{DATEN}}labor.csv")

con = duckdb.connect(":memory:")
con.register("kohorte", cohort)
con.register("labor", labor)

result = con.execute("""
    SELECT patient_id, alter, aufnahmegrund, sofa_score
    FROM kohorte
    WHERE aufnahmegrund = 'Sepsis'
      AND sofa_score >= 6
    ORDER BY sofa_score DESC
    LIMIT 10
""").df()
print(result)
```

### In R (DBI + duckdb)

```r
library(DBI)
library(duckdb)
library(readr)

cohort <- read_csv("{{DATEN}}kohorte.csv", show_col_types = FALSE)
labor  <- read_csv("{{DATEN}}labor.csv", show_col_types = FALSE)

con <- dbConnect(duckdb::duckdb(), dbdir = ":memory:")
duckdb::duckdb_register(con, "kohorte", cohort)
duckdb::duckdb_register(con, "labor", labor)

result <- dbGetQuery(con, "
  SELECT patient_id, alter, aufnahmegrund, sofa_score
  FROM kohorte
  WHERE aufnahmegrund = 'Sepsis'
    AND sofa_score >= 6
  ORDER BY sofa_score DESC
  LIMIT 10
")
print(result)
```

---

**Stolperstein:** `dbConnect()`/`duckdb.connect()` verbindet nur zu einer
DuckDB-Instanz, es lädt noch keine Daten. Erst `con.register()` /
`duckdb_register()` macht ein DataFrame unter einem Tabellennamen für SQL
sichtbar. Vergisst du diesen Schritt, meldet DuckDB `Catalog Error: Table
with name kohorte does not exist`.

## 3 Aggregation mit GROUP BY

Aggregatfunktionen fassen mehrere Zeilen zusammen:

| Funktion | Bedeutung |
|----------|-----------|
| `COUNT(*)` | Anzahl Zeilen |
| `AVG(x)` | Mittelwert |
| `MEDIAN(x)` | Median (DuckDB unterstützt dies nativ) |
| `MIN(x)` / `MAX(x)` | Minimum / Maximum |
| `SUM(x)` | Summe |

`GROUP BY` teilt die Tabelle in Gruppen auf, bevor die Aggregatfunktion
angewendet wird. `HAVING` filtert anschließend auf Gruppenebene (analog zu
`WHERE`, aber nach der Aggregation).

```sql
SELECT
    aufnahmegrund,
    COUNT(*)               AS anzahl,
    ROUND(AVG(sofa_score), 1) AS sofa_mittel,
    ROUND(AVG(verstorben_30d) * 100, 1) AS mortalitaet_pct
FROM kohorte
GROUP BY aufnahmegrund
HAVING COUNT(*) >= 50
ORDER BY mortalitaet_pct DESC;
```

**Stolperstein:** `AVG(verstorben_30d)` ergibt eine Proportion nur, weil das Feld
mit `0`/`1` kodiert ist. Bei Textfeldern (`ja`/`nein`) schlägt das still fehl.
Immer den Datentyp prüfen, bevor man arithmetische Aggregatfunktionen einsetzt.
**Darüber hinaus:** `MEDIAN(x)` ist in Standard-SQL nicht definiert; in DuckDB
funktioniert es nativ. In anderen Systemen (PostgreSQL, SQLite) braucht man
`PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY x)`, gut zu wissen, wenn man die
Abfrage später auf einen echten Klinikserver überträgt.

## 4 JOIN: Tabellen verbinden

Ein `JOIN` verbindet zwei Tabellen über einen gemeinsamen Schlüssel:

```sql
SELECT k.patient_id, k.aufnahmegrund, k.alter, l.laktat_mmol_l
FROM kohorte  AS k
LEFT JOIN labor AS l ON k.patient_id = l.patient_id
WHERE l.laktat_mmol_l IS NOT NULL;
```

| JOIN-Typ | Ergebnis |
|----------|----------|
| `INNER JOIN` | Nur Zeilen, die in **beiden** Tabellen vorhanden sind |
| `LEFT JOIN` | Alle Zeilen aus der **linken** Tabelle; rechts: NULL wenn kein Match |
| `RIGHT JOIN` | Alle aus der **rechten**; selten nötig (swap stattdessen) |
| `FULL OUTER JOIN` | Alle Zeilen beider Tabellen |

![Abbildung: `INNER JOIN` gegen `LEFT JOIN` von `kohorte` mit `labor`. Der `INNER JOIN` behält nur die drei Patient:innen mit vorhandenem Laborwert, die beiden ohne Wert verschwinden ohne Fehlermeldung. Der `LEFT JOIN` behält alle fünf; die fehlenden Laktatwerte erscheinen sichtbar als `NULL`.](assets/join_typen.svg)

**Faustregel:** In der klinischen Forschung fast immer `LEFT JOIN`: du willst
alle Patient:innen behalten, auch wenn ein Laborwert fehlt.

**Stolperstein:** Ein `INNER JOIN` auf `labor` verwirft stillschweigend alle
Patient:innen ohne Laborwert, bei ~17 % fehlenden Laktatwerten schrumpft die
Kohorte entsprechend, ganz ohne Fehlermeldung. Zähle `COUNT(*)` vor und nach
dem JOIN.
**Darüber hinaus:** Nach einem `LEFT JOIN` sind fehlende Werte der rechten Tabelle
als `NULL` erkennbar. Mit `WHERE l.laktat_mmol_l IS NULL` lassen sich genau die
Patient:innen ohne Laborwert selektieren, nützlich zur Missingness-Analyse.

Der vollständige Code mit allen Abfragen steht in [`code/sql.sql`](code/sql.sql),
[`code/python.py`](code/python.py) und [`code/r.R`](code/r.R).

## 5 Abfragen kombinieren

SQL-Abfragen können verschachtelt werden (**Subqueries**) oder mit `WITH` als
**Common Table Expressions (CTEs)** lesbar strukturiert werden:

```sql
-- CTE: first identify Sepsis patients, then evaluate their lactate
WITH sepsis_pts AS (
    SELECT patient_id
    FROM kohorte
    WHERE aufnahmegrund = 'Sepsis'
)
SELECT
    ROUND(AVG(l.laktat_mmol_l), 2) AS laktat_mittel,
    COUNT(*)                         AS n
FROM labor AS l
INNER JOIN sepsis_pts AS s ON l.patient_id = s.patient_id
WHERE l.laktat_mmol_l IS NOT NULL;
```

**Stolperstein:** Verschachtelte Subqueries lesen und debuggen sich schlecht. CTEs
benennen jeden Schritt, leichter zu schreiben und sechs Monate später noch
nachzuvollziehen.
**Darüber hinaus:** Mehrere CTEs lassen sich verketten (`WITH a AS (…), b AS (…)`),
was komplexe Analysepipelines direkt in SQL abbildbar macht, ähnlich wie `dplyr`-Pipes
in R.

## Fallstricke und Merksätze
- **`NULL` ist nicht `0`.** `WHERE laktat IS NOT NULL` explizit formulieren.
 Aggregatfunktionen ignorieren `NULL` automatisch, aber `COUNT(*)` zählt
 alle Zeilen, `COUNT(spalte)` nur Nicht-NULL-Werte.

- **`LEFT JOIN` vs. `INNER JOIN`:** Ein `INNER JOIN` auf `labor` schließt
 Patient:innen ohne Laborwert still aus, das ist fast immer ein Fehler.

- **Reihenfolge der Klauseln:** `SELECT → FROM → WHERE → GROUP BY → HAVING →
 ORDER BY → LIMIT`. SQL prüft `HAVING` erst nach dem Gruppieren.

- **DuckDB und Daten:** Ein DataFrame erst mit `pandas`/`readr` laden (auch von
 einer Web-Adresse) und dann mit `con.register()` / `duckdb_register()`
 sichtbar machen, keine Datei nötig. Alternativ liest DuckDBs `read_csv_auto()`
 auch direkt CSV-Pfade, lokal ohne Zusatzschritt, von einer URL erst nach
 `INSTALL httpfs; LOAD httpfs;`.
- **Spaltenaliase** (`AS anzahl`) verbessern Lesbarkeit und sind in `ORDER BY`
 referenzierbar.

## Selbstcheck
1. **Warum schließt ein `INNER JOIN` auf `labor` Patient:innen ohne Laborwert aus, und wann ist das ein Fehler?**, `INNER JOIN` gibt nur Zeilen zurück, die in beiden Tabellen einen Match haben; fehlt der Laborwert, fällt die Patient:in raus, bei klinischen Kohorten fast immer unbeabsichtigt.
2. **Was ist der Unterschied zwischen `WHERE` und `HAVING`?**, `WHERE` filtert vor der Aggregation auf Zeilenebene; `HAVING` filtert nach `GROUP BY` auf Gruppenebene.
3. **Was gibt `COUNT(laktat_mmol_l)` vs. `COUNT(*)` zurück, wenn ~17 % der Laktatwerte fehlen?**, `COUNT(laktat_mmol_l)` zählt nur Nicht-NULL-Werte (~83 % der Zeilen); `COUNT(*)` zählt alle Zeilen unabhängig von NULL.

