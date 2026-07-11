# Übungen · Kausale Inferenz und Directed Acyclic Graphs

## Aufgabe 1 – DAG in Worten

Beschreibe einen DAG für:
Diabetes, Alter, SOFA, Aufnahmegrund und 30-Tage-Mortalität.

Markiere Confounder, mögliche Mediatoren und Variablen, für die du nicht blind
adjustieren würdest.

## Aufgabe 2 – Crude vs. adjustiert

Berechne das rohe OR für Diabetes und Mortalität. Berechne danach ein
adjustiertes OR nur mit dem Confounder Alter (totaler Effekt). Berechne
zusätzlich zum Vergleich ein drittes Modell, das zusätzlich für SOFA
adjustiert. Wie unterscheiden sich die drei OR, und welche Frage beantwortet
jede davon (Aufgabe 1 hilft)?

## Aufgabe 3 – Propensity Score

Schätze den Propensity Score für Diabetes. Welche Variable(n) gehören ins
PS-Modell, und welche nicht (Aufgabe 1)? Berechne IPTW.

## Aufgabe 4 – Balance

Berechne die Standardized Mean Difference für Alter vor und nach IPTW. Ist
Balance besser? Berechne zum Vergleich auch die SMD für SOFA, für das das
PS-Modell *nicht* adjustiert, was beobachtest du, und warum ist das kein
Problem?

## Aufgabe 5 – Interpretation

Formuliere einen Satz, der die kausale Grenze der Analyse klar benennt.
