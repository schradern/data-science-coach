# Übungen · Workflow für Prädiktionsmodelle und Data Leakage

Code in Englisch, Prosa/Antworten Deutsch. Lösungen: [`loesungen.md`](loesungen.md).

## Aufgabe 1 – Ehrliche Kreuzvalidierung
Baue eine `Pipeline` (Imputation + Skalierung + logistische Regression) und
berechne die stratifizierte 5-fache CV-AUC für `verstorben_30d`. Warum ist die
Pipeline hier entscheidend?

## Aufgabe 2 – Leakage sichtbar machen
Skaliere die Merkmale **einmal auf dem Gesamtdatensatz** und vergleiche die
CV-AUC mit der Pipeline-Variante. Erkläre, warum der Unterschied bei reiner
Skalierung klein, bei Merkmalsauswahl aber groß ist.

## Aufgabe 3 – Tuning ohne Selbstbetrug
Stimme `C` mit `GridSearchCV` ab und berichte die Güte auf einem **separaten**
Testset. Wie unterscheidet sich die innere CV-AUC von der Test-AUC?

## Aufgabe 4 – Schwelle nach Nutzen
Wähle eine Schwelle, die mindestens 80 % Sensitivität erreicht. Welche
Spezifität bezahlst du dafür? In welcher klinischen Situation ist dieser Tausch
gerechtfertigt?

## Bonus – Gruppen-Leakage
Konstruiere mit `vitalwerte.csv` (mehrere Zeilen je Patient:in) ein Beispiel, in
dem zufälliges Splitten Zeilen derselben Person auf Train und Test verteilt.
Nutze `StratifiedGroupKFold` und vergleiche die AUC.
