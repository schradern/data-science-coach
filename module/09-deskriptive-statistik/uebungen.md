# Übungen · Deskriptive Statistik und die Table 1

Nutze den Kohorten- und Labordatensatz. Versuche jede Aufgabe zuerst in
**Python**, dann in **R**. Code in Englisch, Antworten auf Deutsch.
Lösungen: [`loesungen.md`](loesungen.md).

## Aufgabe 1 – Mittelwert oder Median?
Lade die Kohorte wie in der Lektion (`kohorte.csv` über die Web-Adresse, im Projektordner auch `load_cohort()`). Berechne für `crp_mg_l`, `sofa_score`
und `verweildauer_tage` jeweils Mittelwert und Median.

a) Wo weichen sie stark voneinander ab? Was sagt das über die Verteilung? 
b) Welches Lagemaß würdest du in einer Publikation berichten, und warum?

## Aufgabe 2 – Häufigkeiten und Anteile
a) Wie viele Patient:innen haben als Aufnahmegrund „Herzinsuffizienz“?
 Wie viel Prozent der Kohorte ist das (1 Dezimalstelle)? 
b) Wie groß ist der Anteil der Patient:innen mit Hypertonie? 
c) Erstelle eine Kreuztabelle: Raucherstatus × Diabetes (absolut und relativ).

## Aufgabe 3 – Gruppenvergleich manuell
Vergleiche die folgenden Variablen zwischen Patient:innen, die in 30 Tagen
verstorben sind (`verstorben_30d == 1`) und Überlebenden:

- `alter`
- `sofa_score`
- `crp_mg_l`
- `verweildauer_tage`

Berechne Median [IQR] je Gruppe. Welche Variable zeigt den auffälligsten
Unterschied?

## Aufgabe 4 – Table 1 reproduzieren
Erstelle mit `tableone` (Python) oder `gtsummary` (R) eine Table 1 für die
folgenden Variablen, gruppiert nach `aufnahmegrund` (statt nach Mortalität):
`alter`, `geschlecht`, `diabetes`, `sofa_score`, `crp_mg_l`.

Was fällt dir beim Vergleich der Aufnahmegruppen auf?

## Aufgabe 5 – Fehlende Werte in Table 1
Verbinde Kohorte und Labor wie in der Lektion (`kohorte.csv` und `labor.csv`). Erstelle eine
Table 1 und aktiviere die Anzeige fehlender Werte (`missing=True` in tableone,
`missing="ifany"` in gtsummary).

Welche Variablen haben fehlende Werte, und wie viele? Ist das Muster der
fehlenden Werte klinisch erklärbar?

## Bonus
Berechne den **Interquartilsabstand (IQR)** von `laktat_mmol_l` getrennt
nach Aufnahmegrund. Bei welcher Gruppe ist die Streuung am größten?
Erstelle zusätzlich ein Boxplot, das dies visualisiert (Bezug zu Modul 08 (Explorative Datenanalyse und Datenvisualisierung)).
