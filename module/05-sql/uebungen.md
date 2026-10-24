# Übungen · Datenbank-Abfragen mit SQL

Nutze DuckDB in Python oder R, um die folgenden Abfragen auf den Datensätzen
`kohorte`, `labor` und `vitalwerte` auszuführen. Lade sie zuerst mit
`pandas`/`readr` von `{{DATEN}}` und registriere sie bei DuckDB (siehe
README §2 „In Python (DuckDB)“ / „In R (DBI + duckdb)“). Lösungen:
[`loesungen.md`](loesungen.md).

## Aufgabe 1 – SELECT und WHERE
Finde alle Patient:innen mit einem `crp_mg_l`-Wert über 100 und einem
`sofa_score` unter 5. Wie viele sind es? Welcher Aufnahmegrund ist am häufigsten?

## Aufgabe 2 – GROUP BY
Berechne pro `raucherstatus` die mittlere Verweildauer (`verweildauer_tage`),
das mediane CRP (`crp_mg_l`) und die 30-Tage-Mortalität als Prozentwert.
Sortiere absteigend nach Mortalität.

## Aufgabe 3 – JOIN
Verbinde `kohorte` und `labor` via `LEFT JOIN`. Wie viele Patient:innen haben
keinen Kreatininwert (NULL)? Wie viele haben ein Kreatinin über 2,0 mg/dl?

## Aufgabe 4 – NULL und HAVING
Filtere im Labor nur Zeilen mit vorhandenem Laktatwert (`laktat_mmol_l IS NOT NULL`).
Berechne dann den mittleren Laktatwert je `aufnahmegrund`. Zeige nur
Gruppen mit mindestens 30 Messwerten (`HAVING COUNT(*) >= 30`).

## Aufgabe 5 – Dreifach-JOIN
Verknüpfe alle drei Tabellen: Kohorte + Labor + Vitalwerte (nur `tag = 1`, also
der erste Tag nach dem Aufnahmetag `tag = 0`).
Gib für Patient:innen mit `diabetes = 1` aus: `patient_id`, `aufnahmegrund`,
`laktat_mmol_l`, `herzfrequenz` (Tag 1). Wie viele haben vollständige Daten?

## Bonus – CTE
Definiere mit `WITH` zunächst eine Gruppe „schwer krank“: `sofa_score >= 7`
und beliebiger Aufnahmegrund. Berechne dann für diese Gruppe das mediane
Kreatinin und Laktat im Vergleich zum Rest.
