# Musterlösungen · Studiendesign zur Validierung klinischer KI-Systeme

## Übung 1: Validierungsstrategie für ein radiologisches KI-System

1. **Dreiphasiger Validierungsplan:**
    * **Phase 1 (Retrospektiv):** Test des Modells auf einem unabhängigen, historischen Datensatz eines anderen Krankenhauses (externe Validierung), um die rein mathematische Generalisierbarkeit zu prüfen.
    * **Phase 2 (Prospektiv-stumm / Silent Deployment):** Der Algorithmus analysiert im Hintergrund alle eingehenden Röntgenbilder der Notaufnahme. Die Befunde werden protokolliert, aber den Ärzt:innen nicht angezeigt. Nach 3 Monaten wird die Übereinstimmung mit dem späteren Goldstandard (Radiologiebericht) geprüft. Dies zeigt, ob das Modell unter Live-Bedingungen (z. B. schwankende Bildqualität) stabil läuft.
    * **Phase 3 (Interventionell):** Das System wird aktiv geschaltet und liefert Vorschläge an die behandelnden Ärzt:innen in der Notaufnahme.

2. **Endpunkte für die interventionelle Phase:**
    * **Technische Endpunkte:** Prospektive Sensitivität, Spezifität und Falsch-Positiv-Rate im Vergleich zum radiologischen Konsil.
    * **Prozessuale Endpunkte:** Zeitspanne von der Bildaufnahme bis zur Einleitung der Antibiose (Time-to-Antibiotics), Liegedauer in der Notaufnahme.
    * **Klinische Endpunkte:** Rate an Fehldiagnosen, 30-Tage-Mortalität der Pneumonie-Patient:innen.

---

## Übung 2: Die Kontaminations-Problematik bei Randomisierung

1. **Verzerrung durch Kontamination:**
   Wenn Ärzt:innen auf derselben Station für Patient:in A (KI-Gruppe) Alarme erhalten und daraufhin bestimmte Handlungsmuster erlernen (z. B. frühzeitige Flüssigkeitsgaben bei minimalem Kreatininanstieg), werden sie dieses veränderte Verhalten unbewusst auch bei Patient:in B (Kontrollgruppe) anwenden, obwohl für Patient:in B keine Alarme angezeigt werden. Der Lerneffekt der Ärzt:innen „kontaminiert“ die Kontrollgruppe, wodurch der messbare Unterschied zwischen den Gruppen schrumpft.

2. **Lösungs-Studiendesign:**
   Eine **cluster-randomisierte kontrollierte Studie (cRCT)**. Hierbei werden nicht einzelne Patient:innen randomisiert, sondern ganze Stationen oder Krankenhäuser (Cluster). Beispielsweise nutzt Intensivstation 1 das CDSS, während Intensivstation 2 als Kontrolle dient. Alternativ kann ein *Stepped-Wedge-Design* gewählt werden, bei dem die Stationen das System zeitversetzt und in randomisierter Reihenfolge einführen.

---

## Übung 3: Alarm-Müdigkeit und Spectrum Bias – den positiven Vorhersagewert berechnen

1. **Intensivstation (Prävalenz 5 %):** 1 000 Zeitfenster → 50 Sepsis, 950 keine Sepsis.

    - Sensitivität 90 % → `TP = 0,90 · 50 = 45`; `FN = 5`.
    - Spezifität 80 % → `TN = 0,80 · 950 = 760`; `FP = 190`.
    - Alarme `= TP + FP = 45 + 190 = 235`. `PPV = TP / (TP + FP) = 45 / 235 ≈ 0,19` (**19 %**). Fehlalarm-Anteil `= 190 / 235 ≈ 81 %`.

2. **Normalstation (Prävalenz 1 %):** 1 000 Zeitfenster → 10 Sepsis, 990 keine.

    - `TP = 0,90 · 10 = 9`; `FN = 1`. `FP = 0,20 · 990 = 198`; `TN = 792`.
    - Alarme `= 9 + 198 = 207`. `PPV = 9 / 207 ≈ 0,043` (**4,3 %**). Fast **96 %** der Alarme sind falsch.

3. **Konsequenz:** Sensitivität und Spezifität bleiben unverändert, aber der PPV bricht von 19 % auf 4,3 % ein, sobald die Prävalenz sinkt. Auf der Normalstation ist praktisch jeder Alarm ein Fehlalarm – genau die Konstellation, die zu **Alarm-Müdigkeit** führt: Das Personal beginnt, auch die seltenen echten Warnungen zu ignorieren. Der PPV hängt eben nicht nur vom Modell ab, sondern von der Prävalenz der Einsatzpopulation. Ein an Intensivpatient:innen validiertes Modell darf nicht ungeprüft auf ein anderes Spektrum übertragen werden. Streng genommen zeigt die Rechnung nur den Prävalenzeffekt bei unveränderter Sensitivität und Spezifität; echter Spectrum Bias kommt meist noch hinzu, weil sich im anderen Patient:innenspektrum auch Sensitivität und Spezifität verschieben.

---

## Übung 4: AUC vs. Brier-Score – Diskriminierung und Kalibrierung von Hand

1. **Diskriminierung:** Die drei höchsten Vorhersagen (0,3 / 0,4 / 0,5) gehören genau den drei Erkrankten, die beiden niedrigsten (0,1 / 0,2) den beiden Gesunden. Jede:r Erkrankte rangiert über jeder/jedem Gesunden – die Rangfolge ist perfekt, **AUC = 1,0**. Die Diskriminierung ist einwandfrei.
2. **Brier-Score:**

    - `(0,1 − 0)² = 0,01`
    - `(0,2 − 0)² = 0,04`
    - `(0,3 − 1)² = 0,49`
    - `(0,4 − 1)² = 0,36`
    - `(0,5 − 1)² = 0,25`

    Summe `= 1,15`; `Brier = 1,15 / 5 = 0,23` (hoch; ein Teil davon geht allerdings schon auf die hohe Ereignisrate zurück, der Brier Score allein trennt das nicht von Fehlkalibrierung). Mittleres vorhergesagtes Risiko `= (0,1 + 0,2 + 0,3 + 0,4 + 0,5) / 5 = 0,30`; beobachtete Ereignisrate `= 3 / 5 = 0,60`. Das Modell sagt im Schnitt 30 % voraus, wo real 60 % eintreten: `O:E = 0,60 / 0,30 = 2,0`.
3. **Diagnose:** Gestört ist die **Kalibrierung** (die Wahrscheinlichkeiten sind systematisch zu niedrig), intakt ist die **Diskriminierung** (AUC = 1,0). Genau das ist gefährlich: Eine reine AUC-Überwachung würde „alles in Ordnung“ melden, während das Modell das Risiko halbiert und im Live-Einsatz zu spät oder zu selten warnt (Unter-Triage). Aufgedeckt wird das im **Schattenbetrieb / Silent Deployment** durch den Kalibrierungs-Check (O:E, Kalibrierungskurve) – nicht durch die AUC allein (vgl. README Abschnitt 2).
