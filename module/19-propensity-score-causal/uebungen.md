# Übungen · Propensity Score Matching und Weighting

## Übung 1: SMD manuell berechnen
Nach einer Matching-Analyse liegen für das Merkmal „Alter“ in den beiden Gruppen folgende Kennzahlen vor:

- **Behandlungsgruppe:** Mittelwert = 68,5 Jahre, Standardabweichung (SD) = 8,2 Jahre
- **Kontrollgruppe (gemacht):** Mittelwert = 69,1 Jahre, Standardabweichung (SD) = 7,8 Jahre

1. Berechne die gepoolte Standardabweichung (Pooled SD) unter der vereinfachten Annahme gleicher Gruppengrößen:

```text
Pooled SD = sqrt((SD_treated² + SD_control²) / 2)
```

2. Berechne die absolute Standardized Mean Difference (SMD).
3. Beurteile, ob das Alter nach dem Matching ausreichend balanciert ist.

## Übung 2: Overlap-Verlust interpretieren
Betrachte das folgende Diagramm, das die Verteilung der Propensity Scores in einer Studie zeigt:

- Die Behandlungsgruppe hat fast ausschließlich Scores zwischen 0,7 und 0,95.
- Die Kontrollgruppe hat fast ausschließlich Scores zwischen 0,05 und 0,3.

1. Was bedeutet dieser Befund für ein geplantes 1:1 Matching?
2. Welche Patient:innen können nach dem Matching noch analysiert werden?
3. Wie ändert sich die Zielpopulation (Generalisierbarkeit) deiner Studie?

## Übung 3: IPW-Gewichte von Hand berechnen
In der Kohorte sind 94 von 500 Patient:innen Diabetiker:innen; die marginale Prävalenz beträgt also `P(A=1) = 0,188` (und `P(A=0) = 0,812`). Für drei Patient:innen liegen die geschätzten Propensity Scores (`PS = P(Diabetes = 1 | Confounder)`) vor:

| Patient:in | Diabetes (A) | PS |
|---|---|---|
| A | 1 | 0,80 |
| B | 1 | 0,10 |
| C | 0 | 0,95 |

1. Berechne für jede:n das **unstabilisierte** IPW-Gewicht (`1/PS` für Behandelte, `1/(1−PS)` für Kontrollen).
2. Berechne die **stabilisierten** Gewichte (`P(A=a) / P(A=a | X)`).
3. Welche:r Patient:in dominiert die Schätzung am stärksten? Was verrät der Fall über eine mögliche Positivity-Verletzung – und behebt die Stabilisierung dieses Problem?

## Übung 4: Effektive Stichprobengröße bei extremen Gewichten
Nach dem IPW-Schritt hat eine Mini-Analyse fünf gewichtete Patient:innen. Die Kish-Formel für die effektive Stichprobengröße lautet `N_eff = (Σ wᵢ)² / Σ wᵢ²`.

1. Berechne `N_eff` für die Gewichte {1, 1, 1, 1, 12}.
2. Berechne `N_eff` für die ausgeglichenen Gewichte {1, 1, 1, 1, 1}.
3. Was folgt aus dem Vergleich für die Präzision (den Standardfehler) der IPW-Schätzung, und wie hängt das mit der Empfehlung zusammen, Gewichte zu stabilisieren oder zu trimmen?
