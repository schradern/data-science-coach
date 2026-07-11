# Übungen · Baum-Ensembles und Gradient Boosting

Code in Englisch, Prosa/Antworten Deutsch. Lösungen: [`loesungen.md`](loesungen.md).

## Aufgabe 1 – Overfitting beim einzelnen Baum
Trainiere `DecisionTreeClassifier` für `max_depth` von 1 bis 15 und zeichne
Trainings- vs. Validierungs-AUC (nach Train/Test-Split). Bei welcher Tiefe
beginnt die Validierungs-AUC zu sinken? Was sagt das über Bias-Varianz-Tradeoff?

## Aufgabe 2 – Random Forest: Einfluss von `n_estimators`
Vergleiche die kreuzvalidierte AUC für `n_estimators` ∈ {10, 50, 100, 300, 500}.
Ab wie vielen Bäumen stagniert die Güte? Wann lohnt sich der Rechenaufwand nicht mehr?

## Aufgabe 3 – Lernrate und Iterationen beim Boosting
Trainiere `HistGradientBoostingClassifier` mit je zwei Kombinationen:
(a) `learning_rate=0.30, max_iter=50` und
(b) `learning_rate=0.01, max_iter=1500`.
Vergleiche CV-AUC und Trainingszeit (`time.time()`). Erkläre, warum kleine
Lernraten trotz höherem Aufwand oft besser generalisieren.

## Aufgabe 4 – Klassenungleichheit
Trainiere `RandomForestClassifier` *ohne* `class_weight` und vergleiche Precision,
Recall und F1 für die positive Klasse mit der gewichteten Variante.
Warum ist Recall für klinische Ereignisse oft wichtiger als Precision?

## Bonus – Feature Importance vergleichen
Vergleiche `feature_importances_` (Gini-basiert) des Random Forest mit
`permutation_importance` aus `sklearn.inspection`. Bei welchen Merkmalen
weichen sie stark ab? Was schlussfolgerst du über Korrelationen im Datensatz?
