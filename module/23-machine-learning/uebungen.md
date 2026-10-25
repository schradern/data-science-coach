# Übungen · Einführung in das maschinelle Lernen

Nutze die Kohortendaten. Versuche jede Aufgabe in **Python**, dann in **R**.
Lösungen: [`loesungen.md`](loesungen.md).

## Aufgabe 1 – Genauigkeitsparadox
Implementiere einen naiven Klassifikator, der immer die häufigere Klasse
(„lebt“) vorhersagt. Berechne Accuracy und ROC-AUC. Erkläre in zwei Sätzen,
warum Accuracy bei diesem Outcome kein geeignetes Gütemaß ist und was AUC
stattdessen misst.

## Aufgabe 2 – Split-Varianz
Führe den stratifizierten Train/Test-Split mit drei verschiedenen Seeds
(z. B. 0, 42, 99) durch und trainiere das vollständige Pipeline-Modell jeweils
neu. Wie stark schwankt der Test-AUC? Was sagt das über die Verlässlichkeit eines
einzelnen Splits aus, und warum ist Kreuzvalidierung robuster?

## Aufgabe 3 – Merkmalssätze vergleichen
Trainiere drei Modelle mit unterschiedlichen Merkmalssätzen (immer mit
5-facher stratifizierter Kreuzvalidierung auf den Trainingsdaten):

1. Nur `sofa_score`
2. `sofa_score` + `alter`
3. `sofa_score` + `alter` + `crp_mg_l` + `diabetes` + `is_sepsis` + `is_smoker`

Vergleiche den mittleren CV-AUC. Welches Prinzip, Underfitting vs. Overfitting, illustriert der Vergleich? Wann kann mehr Information schaden?

## Aufgabe 4 – Schwellenwert und klinische Abwägung
Berechne Sensitivität und Spezifität für die Schwellenwerte 0,10, 0,15, 0,20,
0,30 und 0,50. Bei Schwelle 0,50 ist die Sensitivität dank `class_weight=
"balanced"` bereits recht hoch, auf Kosten der Spezifität (viele Fehlalarme).
Welchen Schwellenwert würdest du in einem Setting mit begrenzten
ICU-Kapazitäten wählen, und welchen in einem Setting, in dem Fehlalarme sehr
kostspielig sind? Begründe kurz.

## Aufgabe 5 – Kalibrierungskurve interpretieren
Lies die Kalibrierungsausgabe des Python-Skripts (`check_calibration`): Sie
zeigt eine mittlere vorhergesagte Mortalität von ~46 %, obwohl im Testset nur
~16 % tatsächlich sterben. Was bedeutet das klinisch, wenn ein Modell für eine
Gruppe eine so viel höhere Mortalität vorhersagt, als tatsächlich eintritt?
Welche Konsequenz hätte das für eine ICU-Zuweisung? Erkläre, was
`class_weight="balanced"` an der Basisrate ändert, die das Modell „sieht“, und
nenne eine statistische Methode, mit der man die Kalibrierung post-hoc
korrigieren kann.

## Bonus – Bias nach Subgruppe
Berechne den ROC-AUC getrennt für Patient:innen mit und ohne Diabetes
(auf den Testdaten). Gibt es Unterschiede? Diskutiere: Was würde eine große
AUC-Diskrepanz zwischen Subgruppen bedeuten, und warum reicht der globale AUC
allein nicht als ethische Mindestprüfung?
