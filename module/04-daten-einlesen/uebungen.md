# Übungen · Datenimport aus Dateien und APIs

Code in Englisch, Prosa/Antworten Deutsch. Lösungen: [`loesungen.md`](loesungen.md).

## Aufgabe 1 – CSV explizit einlesen
Lese `{{DATEN}}kohorte.csv` mit **allen** Optionen explizit:
`sep`, `encoding`, `decimal`, `dtype` für die Integer-Spalten, `na_values`.
Wie viele Spalten hat die Datei? Welche Spalten haben fehlende Werte?

## Aufgabe 2 – Nur bestimmte Spalten laden
Lese `{{DATEN}}labor.csv` und wähle dabei **nur** die Spalten
`patient_id`, `kreatinin_mg_dl` und `laktat_mmol_l` beim Einlesen aus
(nicht danach filtern, direkt mit `usecols` / `col_select`).
Wie viele Zeilen hat die resultierende Tabelle?

## Aufgabe 3 – JSON erzeugen und wieder einlesen
Konvertiere die ersten 10 Zeilen der Kohorte (Spalten `patient_id`, `alter`,
`aufnahmegrund`) in einen JSON-String. Lies den JSON-String anschließend
wieder als DataFrame/Tibble ein. Stimmt die Form?

## Aufgabe 4 – Verschachtelte Struktur normalisieren
Gegeben ist folgendes JSON (kannst du direkt im Skript als String definieren):

```json
[
  {"id": 1, "vitals": {"hr": 82, "map": 75}, "dx": "Sepsis"},
  {"id": 2, "vitals": {"hr": 91, "map": 68}, "dx": "Pneumonie"},
  {"id": 3, "vitals": {"hr": 74, "map": 82}, "dx": "COPD-Exazerbation"}
]
```

Normalisiere die verschachtelte `vitals`-Struktur so, dass `hr` und `map`
eigene Spalten werden.

## Aufgabe 5 – API-Fallback-Funktion
Schreibe eine Funktion `load_api_or_local(url, local_file)`, die:

1. Versucht, die URL abzurufen (Timeout: 5 s).
2. Bei Fehler die lokale Datei einliest und einen Hinweis ausgibt.
3. Immer einen DataFrame zurückgibt.

Teste sie mit der disease.sh-API und `{{DATEN}}kohorte.csv` als Fallback.

## Bonus
Lese `{{DATEN}}vitalwerte.csv` ein. Prüfe, ob alle `patient_id`-Werte aus
`kohorte.csv` in `vitalwerte.csv` vorkommen. Was fällt auf?
(Tipp: Mengenoperationen mit Sets in Python oder `anti_join` in R.)
