# Übungen · Studiendesign zur Validierung klinischer KI-Systeme

## Übung 1: Validierungsstrategie für ein radiologisches KI-System
Du hast einen Deep-Learning-Algorithmus entwickelt, der auf Röntgen-Thorax-Aufnahmen Anzeichen einer Pneumonie klassifiziert (AUC = 0,92 in retrospektiven Testdaten). Nun soll das System in der Notaufnahme implementiert werden.

1. Entwerfe einen dreiphasigen Validierungsplan (retrospektiv, prospektiv-stumm, interventionell).
2. Welche Endpunkte würdest du für die interventionelle Phase wählen (unterscheide technische, klinische und prozessuale Endpunkte)?

## Übung 2: Die Kontaminations-Problematik bei Randomisierung
Ein CDSS warnt auf der Intensivstation vor akutem Nierenversagen. Du planst eine randomisierte Studie, bei der Patient:innen 1:1 in die Gruppen „KI-Überwachung“ und „Standardüberwachung“ eingeteilt werden.

1. Erkläre, warum eine Randomisierung auf Patient:innenebene bei diesem Studiendesign zu einer Verzerrung (Kontamination) der Ergebnisse führen kann.
2. Welches Studiendesign löst dieses Problem?

## Übung 3: Alarm-Müdigkeit und Spectrum Bias — den positiven Vorhersagewert berechnen
Ein Sepsis-Frühwarnsystem läuft auf einer Intensivstation. In der validierten Population beträgt die Sepsis-Prävalenz (pro überwachtem Patient:innen-Zeitfenster) 5 %. Das Modell hat eine Sensitivität von 90 % und eine Spezifität von 80 %.

1. Lege eine Vierfeldertafel für 1 000 Zeitfenster an und berechne den positiven Vorhersagewert (PPV). Welcher Anteil aller Alarme ist ein Fehlalarm?
2. Dasselbe (eingefrorene) Modell wird nun auf einer Normalstation eingesetzt, wo die Prävalenz nur 1 % beträgt (Spectrum Bias). Berechne den PPV erneut.
3. Was folgt aus dem Vergleich für die Alarm-Müdigkeit?

## Übung 4: AUC vs. Brier-Score — Diskriminierung und Kalibrierung von Hand
Ein im Training gut kalibriertes Sepsis-Modell wird im Schattenbetrieb auf eine kränker gewordene Population angewendet. Für fünf Patient:innen liegen die vorhergesagte Risikowahrscheinlichkeit `p` und das tatsächliche Outcome `y` (1 = Sepsis) vor:

| Patient:in | p | y |
|---|---|---|
| P1 | 0,1 | 0 |
| P2 | 0,2 | 0 |
| P3 | 0,3 | 1 |
| P4 | 0,4 | 1 |
| P5 | 0,5 | 1 |

1. Beurteile die **Diskriminierung**: Trennt das Modell Erkrankte von Gesunden in der Rangfolge? Was wäre die AUC?
2. Berechne den **Brier-Score** `= (1/n) · Σ (pᵢ − yᵢ)²` und vergleiche das mittlere vorhergesagte Risiko mit der beobachteten Ereignisrate.
3. Welche Modelleigenschaft ist hier gestört, welche nicht? Warum ist dieser Fall im Live-Einsatz gefährlich, und welche Phase des Studienplans deckt ihn auf?
