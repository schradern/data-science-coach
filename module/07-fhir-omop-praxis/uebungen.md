# Übungen · Patientendaten-Extraktion via FHIR und OMOP

## Übung 1: FHIR-Abfrage für Vitalparameter (Python oder R)
Du willst alle Blutdruckmessungen (LOINC-Code: `85354-9` für Blutdruck-Paneel) von einem öffentlichen Testserver laden.

1. Schreibe ein kurzes Python- oder R-Skript, das die Observation-Ressourcen abfragt.
2. Extrahiere die Patient:innen-ID (`subject.reference`) und das Datum (`effectiveDateTime`).

## Übung 2: OMOP-SQL-Abfrage für Geburtskohorten und Outcome
Schreibe eine SQL-Abfrage für eine OMOP-Datenbank:

1. Wähle alle Patient:innen aus der Tabelle `person`, die vor dem Jahr 1960 geboren wurden.
2. Verknüpfe diese mit der Tabelle `death` (mittels `LEFT JOIN`), um das Sterbedatum (`death_date`) zu ermitteln.
3. Gib die Spalten `person_id`, `year_of_birth` und `death_date` aus.
