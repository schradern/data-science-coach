# Übungen · Bewertung der Modellgüte und klinische Validierung

Code in Englisch, Prosa/Antworten Deutsch. Lösungen: [`loesungen.md`](loesungen.md).

## Aufgabe 1 – AUC-Vergleich bei Klassenungleichheit
Berechne sowohl ROC-AUC als auch PR-AUC für das logistische Modell auf dem Testset.
Bei welchem Modellwert liegt die PR-AUC-Baseline (Zufallsklassifikator)?
Warum ist es bei ~15 % Ereignisrate irreführend, nur die ROC-AUC zu berichten?

## Aufgabe 2 – Erst rekalibrieren, dann Kalibrierung prüfen
Vergleiche die mittlere vorhergesagte Mortalität von `proba` (roh, aus dem
`class_weight="balanced"`-Modell) und `proba_cal` (nach `CalibratedClassifierCV`)
mit der beobachteten Ereignisrate im Testset. Welche der beiden würdest du für
eine Kalibrierungskurve verwenden, und warum?

## Aufgabe 3 – Kalibrierungskurve interpretieren
Zeichne die Kalibrierungskurve (10 Bins, auf `proba_cal`) und markiere, in
welchen Risikobereichen das Modell über- bzw. unterschätzt. Berechne den Brier
Score. Liegt das Modell näher an der Diagonalen oder beim Null-Modell (immer
mittlere Ereignisrate)?

## Aufgabe 4 – Net Benefit manuell
Berechne den Net Benefit (auf `proba_cal`) für Schwellen 0,05 / 0,10 / 0,20 / 0,30 für
(a) das Modell, (b) alle behandeln, (c) niemanden behandeln.
Bei welchen Schwellen ist das Modell klinisch nützlich?

## Aufgabe 5 – Bootstrap-Optimismus
Schätze den AUC-Optimismus mit 100 Bootstrap-Stichproben auf den Trainingsdaten.
Wie groß ist der Unterschied zwischen apparenter und korrigierter AUC?
Wann ist dieser Unterschied besonders groß?

## Bonus – Sigmoid vs. isotonische Rekalibrierung
Wende `CalibratedClassifierCV(method="isotonic")` UND `method="sigmoid"` auf das
trainierte `class_weight="balanced"`-Modell an und vergleiche Brier Score und
Kalibrierungskurve der beiden Methoden. Welchen Nachteil hat die isotonische
Methode bei kleinen Datensätzen (hier: ~59 Trainingsereignisse)?
