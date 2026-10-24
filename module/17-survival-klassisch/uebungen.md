# Übungen · Klassische Überlebenszeitanalyse

## Aufgabe 1 – Ereignis und Zeit

Definiere für den Kursdatensatz:

- Zeitvariable
- Ereignisvariable
- Zensierungsinterpretation

Der Datensatz enthält auch `verweildauer_tage` (Verweildauer im Krankenhaus).
Wäre diese Spalte als Zeitvariable für eine Mortalitäts-Survival-Analyse
geeignet? Begründe.

## Aufgabe 2 – Kaplan-Meier

Erstelle Kaplan-Meier-Schätzungen für Sepsis vs. Nicht-Sepsis. Berichte die
Überlebenswahrscheinlichkeit bei Tag 10 und Tag 20.

## Aufgabe 3 – Log-rank

Führe einen Log-rank-Test für Sepsis vs. Nicht-Sepsis durch und interpretiere ihn.

## Aufgabe 4 – Cox-Modell

Fitte ein Cox-Modell:

```text
Surv(fu_zeit_tage, status) ~ sepsis + alter + sofa_score
```

Berichte HR und 95-%-KI.

## Aufgabe 5 – Annahmen

Prüfe die Proportional-Hazards-Annahme oder beschreibe, wie du sie prüfen würdest.
Was würdest du tun, wenn sie verletzt ist?
