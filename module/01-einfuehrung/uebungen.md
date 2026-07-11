# Übungen · Einführung und Lernpfad

Code in Englisch, Prosa/Antworten Deutsch. Lösungen: [`loesungen.md`](loesungen.md).

## Aufgabe 1 – Datensatz erkunden
Lade `kohorte.csv` mit `load_cohort()`. Wie viele Patient:innen enthält der
Datensatz? Wie viele Spalten? Gib die Spaltenübersicht aus.

## Aufgabe 2 – Mortalität nach Aufnahmegrund
Berechne die 30-Tage-Mortalität (`verstorben_30d`) getrennt nach `aufnahmegrund`.
Welcher Aufnahmegrund hat die höchste Sterblichkeit?

## Aufgabe 3 – Altersverteilung
Berechne Mittelwert und Median des Alters in der Kohorte. Was sagt der Unterschied
zwischen Mittelwert und Median über die Verteilung aus?

## Aufgabe 4 – Lernpfad wählen
Lies Abschnitt 3 „Kursüberblick und Lernpfade" oben in diesem Modul. Welcher
der sieben Lernpfade passt am besten zu deinem aktuellen Ziel? Notiere kurz,
warum.

## Bonus
Schreibe eine Funktion `check_datasets()`, die beim Aufruf automatisch prüft,
ob alle vier Datensätze (`kohorte.csv`, `vitalwerte.csv`, `labor.csv`,
`notizen.csv`) vorhanden sind, und eine hilfreiche Fehlermeldung ausgibt,
wenn eine Datei fehlt.
