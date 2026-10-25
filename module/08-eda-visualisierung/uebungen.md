# Übungen · Explorative Datenanalyse und Datenvisualisierung

Nutze den Kohorten- und Labordatensatz. Versuche jede Aufgabe zuerst in **Python**,
dann in **R**. Code in Englisch, Antworten auf Deutsch. Lösungen: [`loesungen.md`](loesungen.md).

## Aufgabe 1 – Erste Erkundung
Lade `kohorte.csv` und `labor.csv` wie in der Lektion (Web-Adresse, im Projektordner auch `load_cohort()` / `load_labs()`) und führe sie zusammen.
Gib für die Spalten `bmi`, `kreatinin_mg_dl` und `haemoglobin_g_dl` den Mittelwert,
Median und die Standardabweichung aus. Welche Variable ist am stärksten rechtsschief?

## Aufgabe 2 – Häufigkeiten und fehlende Werte
Wie ist `raucherstatus` verteilt (absolut und relativ)? Welche Variablen haben
überhaupt fehlende Werte? Wie hoch ist der Anteil bei `laktat_mmol_l`?

## Aufgabe 3 – Ausreißer
Wende die IQR-Methode auf `kreatinin_mg_dl` an. Wie viele Ausreißer nach oben
gibt es? Schau dir die Werte an: klinisch plausibel?

## Aufgabe 4 – Histogramm
Erstelle ein Histogramm der `verweildauer_tage`, getrennt nach `aufnahmegrund`.
Welcher Aufnahmegrund hat die längste Verweildauer? Gibt es auffällige Häufungen?

## Aufgabe 5 – Boxplot
Erstelle einen Boxplot des `sofa_score` je `raucherstatus`. Sortiere die Gruppen
nach dem Median (absteigend). Was fällt auf?

## Aufgabe 6 – Streudiagramm und Korrelation
Erstelle ein Streudiagramm von `crp_mg_l` (x) vs. `verweildauer_tage` (y),
eingefärbt nach `verstorben_30d`. Berechne danach den Pearson-Korrelationskoeffizienten.
Entspricht das Bild dem Koeffizienten?

## Bonus
Erstelle eine Korrelationsmatrix für die numerischen klinischen Messwerte in der
zusammengeführten Tabelle (ohne IDs und ohne die Spalten zur Nachbeobachtung wie
`status` oder `fu_zeit_tage`). Welche Variable korreliert am stärksten mit `verstorben_30d`?
Überrascht dich das Ergebnis?
