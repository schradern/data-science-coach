# Übungen · Neuronale Netze und Deep Learning

Code in Englisch, Prosa/Antworten Deutsch. Lösungen: [`loesungen.md`](loesungen.md).

## Aufgabe 1 – Pipeline mit MLP
Baue eine sklearn-Pipeline (Imputation + Skalierung + MLPClassifier) und
berechne die 5-fache CV-AUC für `verstorben_30d`. Warum ist der
`StandardScaler` hier wichtiger als bei einem Entscheidungsbaum?

## Aufgabe 2 – Lernkurve interpretieren
Trainiere das MLP mit `early_stopping=True` und plotte `loss_curve_` und
`validation_scores_`. Woran erkennst du Überanpassung? Ab welcher Epoche
stoppt Early Stopping in deinem Lauf?

## Aufgabe 3 – Regularisierung vergleichen
Teste `alpha` ∈ {0.0001, 0.001, 0.01, 0.1, 1.0} per 5-facher CV-AUC.
Welcher Wert schneidet am besten ab? Erkläre, was zu großes und zu kleines
alpha jeweils anrichtet.

## Aufgabe 4 – Ehrlicher Modellvergleich
Vergleiche MLP, logistische Regression und Gradient Boosting auf denselben
Daten per 5-facher CV-AUC. Was schlägt was? Erkläre das Ergebnis anhand der
Datenmenge (~500 Patient:innen) und Modellkomplexität.

## Bonus – Netzarchitektur variieren
Probiere drei verschiedene `hidden_layer_sizes`: (32,), (64, 32), (128, 64, 32).
Wie verändert sich die AUC? Ab wann überpasst das Netz? Diskutiere, ob mehr
Schichten auf diesen Daten sinnvoll ist.
