# Übungen · Maschinelles Lernen für Überlebenszeiten

Code in Englisch, Prosa/Antworten Deutsch. Lösungen: [`loesungen.md`](loesungen.md).

## Aufgabe 1 – Zensierung verstehen
Erkläre in drei Sätzen: Was ist ein **zensierter** Patient in diesem Datensatz,
und warum würde ein einfacher Klassifikator (z. B. `LogisticRegression`) für
diese Person ein falsches Label verwenden?

## Aufgabe 2 – Cox als Baseline
Passe mit `lifelines.CoxPHFitter` ein Cox-Modell an `fu_zeit_tage`/`status`
an (**nicht** an `verweildauer_tage`, das ist nur die deskriptive
Verweildauer, keine Ereigniszeit). Berechne die **Konkordanz** (C-Statistik)
auf dem Testset. Was ist der Unterschied zwischen Konkordanz und AUC bei
binärer Klassifikation?

## Aufgabe 3 – Risikogruppen und KM
Teile die Testpatienten nach dem Cox-Score in drei gleich große **Tertile**
(niedrig / mittel / hoch). Plotte Kaplan-Meier-Kurven je Gruppe mit `lifelines`.
Trennen sich die Kurven deutlich? Führe den Log-Rank-Test durch.

## Aufgabe 4 – RSF (falls sksurv installiert)
Trainiere einen `RandomSurvivalForest` und berechne die zeitabhängige AUC bei
Tag 7 und Tag 21. Vergleiche mit der Cox-AUC. Liegt eines der beiden Modelle zu
einem Zeitpunkt klar vorn, und reicht die Testmenge, um das zu belegen?

## Bonus – Konkurrierende Risiken konzeptuell
Angenommen, 10 % der Patient:innen werden in eine andere Klinik verlegt und
sterben dort, also **außerhalb** des Beobachtungsfensters. Wie würde die
KM-Kurve verzerrt sein, wenn man diese als Nicht-Ereignis behandelt?
Was wäre die korrekte Analysestrategie?
