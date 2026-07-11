# Übungen · Bayesianische Inferenz

## Übung 1: Posterior von Hand

Vor einer kleinen Pilotstudie glaubst du, die Komplikationsrate eines Eingriffs
liege eher niedrig. Du wählst als Prior **Beta(2, 8)** (Mittel 0,20, aber schwach,
nur 10 fiktive Fälle wert). Dann beobachtest du **5 Komplikationen bei 20
Patient:innen**.

1. Schreibe den Posterior als Beta-Verteilung mit konkreten Parametern.
2. Berechne das Posterior-Mittel.
3. Vergleiche das Posterior-Mittel mit dem reinen Datenanteil (5/20). In welche Richtung zieht der Prior, und warum?

## Übung 2: Warum der Prior an der großen Kohorte kaum zählt

Du wendest denselben Prior **Beta(2, 8)** auf einen viel größeren Datensatz an:
**100 Komplikationen bei 400 Patient:innen**.

1. Schreibe den Posterior und berechne sein Mittel.
2. Berechne, um wie viele Prozentpunkte der Prior das Posterior-Mittel gegenüber dem reinen Datenanteil (100/400) verschiebt.
3. Erkläre in zwei Sätzen, warum derselbe Prior hier viel weniger bewegt als in Übung 1.

## Übung 3: Praktische Äquivalenz beurteilen

Zwei Behandlungsarme werden bayesianisch verglichen. Die Posterior-Verteilung der
**Differenz der Komplikationsraten** (Arm A minus Arm B) hat Mittel **0,01** mit
95-%-Credible-Interval **[−0,03; 0,05]**. Die klinisch vorab festgelegte ROPE ist
**±0,05** (Unterschiede unter 5 Prozentpunkten gelten als praktisch gleich).

1. Schließt das Credible-Interval die Null ein? Was bedeutet das?
2. Liegt das gesamte Credible-Interval innerhalb der ROPE? Was folgt daraus für die Aussage „praktisch äquivalent"?
3. Formuliere den Befund in einem Satz für eine klinische Leserin — ohne das Wort „signifikant".
