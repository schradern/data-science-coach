# Übungen · Erklärbarkeit von Machine-Learning-Modellen

Code in Englisch, Prosa/Antworten Deutsch. Lösungen: [`loesungen.md`](loesungen.md).

## Aufgabe 1 – Permutation vs. Impurity
Trainiere einen `RandomForestClassifier` (100 Bäume) auf `verstorben_30d` und
vergleiche `feature_importances_` (Impurity) mit `permutation_importance` auf
dem Testset. Welche Variable rangiert bei der Impurity-Methode künstlich hoch,
und warum?

## Aufgabe 2 – PDP lesen
Erstelle einen PDP für `crp_mg_l`. Beschreibe in zwei Sätzen: Ab welchem CRP-Wert
steigt das Sterberisiko im Modell markant? Wie ändert sich die Aussage, wenn du
`kind="both"` nutzt und die ICE-Linien siehst?

## Aufgabe 3 – SHAP lokal
Falls `shap` installiert ist: Berechne SHAP-Werte für die fünf Patient:innen mit
dem höchsten vorhergesagten Sterberisiko. Welches Merkmal treibt den Score
bei den meisten dieser Fälle am stärksten? Falls SHAP nicht verfügbar: Erkläre
in Prosa, wie du denselben Vergleich mit Permutation Importance und PDP-Baseline
annähern würdest.

## Aufgabe 4 – Korrelierte Merkmale
`leukozyten_g_l` korreliert mit `sofa_score` (r ≈ 0,45). Entferne
`leukozyten_g_l` aus den Merkmalen, trainiere das Modell neu und vergleiche die
Permutation Importance von `sofa_score` vorher und nachher. Erkläre das Ergebnis
in zwei Sätzen.

## Bonus – Globale Sicht vs. klinische Realität
Wähle das aus der PI wichtigste Merkmal. Suche in der Literatur (oder begründe
theoretisch), ob dieser Zusammenhang kausal oder assoziativ ist. Was muss man
aus diesem Unterschied für den klinischen Einsatz ableiten?
