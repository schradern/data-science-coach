# Übungen · Diagnostische Genauigkeit und Schwellenwerte

## Aufgabe 1 – 2x2-Tabelle

Nutze Laktat ≥ 2,0 mmol/l als positiven Test für 30-Tage-Mortalität. Berechne
TP, FP, FN, TN.

## Aufgabe 2 – Diagnostische Kennzahlen

Berechne Sensitivität, Spezifität, PPV, NPV, LR+ und LR-.

## Aufgabe 3 – Youden-Schwelle

Berechne die ROC-Kurve und finde die Schwelle mit maximalem Youden-Index.
Vergleiche sie mit 2,0 mmol/l.

## Aufgabe 4 – Prävalenz

Erkläre, was mit PPV und NPV passieren würde, wenn dieselbe Testleistung in einer
Kohorte mit deutlich niedrigerer Mortalität eingesetzt wird.

## Aufgabe 5 – Bericht

Schreibe einen kurzen Diagnostik-Abschnitt für Laktat als Marker.

## Aufgabe 6 – Net Benefit

Fitte ein logistisches Modell `verstorben_30d ~ laktat_mmol_l` und berechne den
Net Benefit des Modells sowie der Strategie „alle behandeln" für die
Schwellenwahrscheinlichkeiten $p_t \in \{0{,}10;\ 0{,}15;\ 0{,}20;\ 0{,}25\}$.

- In welchem Bereich liegt das Modell über **beiden** Referenzstrategien?
- Erkläre, warum die Youden-Schwelle diese Prävalenz-/Kosten-Abwägung nicht
  leisten kann.
