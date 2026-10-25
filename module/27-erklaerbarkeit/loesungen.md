# Musterlösungen · Erklärbarkeit von Machine-Learning-Modellen

**Vorbereitung.** Die Lösungen setzen auf den Bausteinen aus [`code/python.py`](code/python.py) auf. Führe diesen Block einmal im Projektordner aus, lokal oder in Colab nach dem Klonen ([in Colab öffnen](https://colab.research.google.com/github/schradern/data-science-coach/blob/main/notebooks/start.ipynb)). Danach laufen die Blöcke dieser Seite der Reihe nach. `runpy` lädt nur die Definitionen des Skripts, `main()` läuft dabei nicht.

```python
# Setup: run once from the project folder.
import runpy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

lesson = runpy.run_path("module/27-erklaerbarkeit/code/python.py")
SEED, FEATURES, build_data, train_model = (
    lesson[name] for name in ("SEED", "FEATURES", "build_data", "train_model"))
X, y = build_data()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=SEED)
model = train_model(X_train, y_train)
```

## Aufgabe 1 – Permutation vs. Impurity
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

rf = RandomForestClassifier(n_estimators=100, random_state=SEED, class_weight="balanced")
rf.fit(X_train, y_train)

# Impurity importance (train-based, potentially biased)
for name, imp in sorted(zip(FEATURES, rf.feature_importances_), key=lambda x: -x[1])[:5]:
    print(f"  {name:<30} {imp:.4f}  (impurity)")

# Permutation importance (test-based, honest)
result = permutation_importance(rf, X_test, y_test, n_repeats=20,
                                random_state=SEED, scoring="roc_auc")
for i in np.argsort(result.importances_mean)[::-1][:5]:
    print(f"  {FEATURES[i]:<30} {result.importances_mean[i]:+.4f}  (permutation)")
```
Bei der Impurity-Methode rangieren kontinuierliche Merkmale mit vielen
Ausprägungen künstlich hoch. Auf diesem Split landet `bmi` auf Platz 3, obwohl
BMI im datenerzeugenden Prozess gar nicht auf die Mortalität wirkt und seine
Permutation Importance negativ ist; auch `crp_mg_l` liegt weit vorn. Der
SOFA-Score mit nur 12 Ausprägungen steht bei Impurity auf Platz 5, bei
Permutation Importance auf Platz 1. Grund: Viele Ausprägungen bieten viele
mögliche Schnittpunkte, an denen sich die Bäume an Trainingsrauschen anpassen
können, und die Impurity-Werte stammen aus den Trainingsdaten. Die als Ganzzahlen
kodierten Kategorien (`aufnahmegrund`, `raucherstatus`) liegen dagegen weit
hinten. Permutation Importance bewertet den tatsächlichen Leistungsabfall auf
ungesehenen Daten.

## Aufgabe 2 – PDP lesen
```python
from sklearn.inspection import PartialDependenceDisplay
fig, ax = plt.subplots(figsize=(6, 4))
PartialDependenceDisplay.from_estimator(model, X_test, features=["crp_mg_l"],
                                         kind="both", ax=ax, subsample=80)
```
Einen klaren Schwellenwert gibt es auf diesem Modell nicht: Sehr niedrige
CRP-Werte (unter etwa 30 mg/l) gehen mit deutlich niedrigerem vorhergesagtem
Risiko einher, zwischen etwa 80 und 105 mg/l hat die Kurve eine Delle, oberhalb
von etwa 110 mg/l steigt sie wieder. Ein solches Zickzack ist bei 375
Trainingszeilen eher Rauschen als Biologie; passend dazu liegt die Permutation
Importance von CRP auf dem Testset bei null. Im datenerzeugenden Prozess wirkt
CRP nur schwach und linear (Log-Hazard +0,0035 pro mg/l). Wegen
`class_weight="balanced"` sind die absoluten Werte auf der y-Achse zudem
überhöht (Modul 25). Die ICE-Kurven zeigen, dass der Effekt zwischen
Patient:innen sehr unterschiedlich stark ausfällt. Nicht parallele Kurven heißen:
Das Modell hat eine Interaktion gelernt (z. B. mit dem SOFA-Score), nicht, dass
es sie in der Realität gibt.

## Aufgabe 3 – SHAP lokal
```python
import numpy as np
import shap

explainer = shap.TreeExplainer(model)   # path-dependent TreeSHAP, exact for trees
shap_values = explainer(X_test)
# Highest-risk patients
top5_idx = model.predict_proba(X_test)[:, 1].argsort()[-5:]
for idx in top5_idx:
    sv = shap_values[idx].values
    top_feat = FEATURES[np.argmax(np.abs(sv))]
    print(f"  Patient {idx}: stärkster Treiber = {top_feat}")
# Patient   7: stärkster Treiber = sofa_score
# Patient  16: stärkster Treiber = sofa_score
# Patient 119: stärkster Treiber = sofa_score
# Patient 112: stärkster Treiber = sofa_score
# Patient  71: stärkster Treiber = sofa_score
```
Auf unserem Lauf ist `sofa_score` bei allen 5 Hochrisikopatient:innen der
stärkste Einzeltreiber. Global (`assets/shap_beeswarm.png`) liegen Alter, CRP
und SOFA-Score vorn; bei den Hochrisikofällen dominiert der SOFA-Score, weil
hohe Werte dort den Score am stärksten nach oben schieben. Lokale und globale
Rangfolge müssen nicht übereinstimmen.

Ohne SHAP: Für jedes Merkmal liest man an der ICE-Kurve der Person (sonst an der
PDP-Kurve) ab, wie weit die Vorhersage bei ihrem eigenen Merkmalswert über oder
unter dem Mittel der Vorhersagen liegt. Das Merkmal mit der größten Abweichung
ist der wahrscheinlichste Haupttreiber. Wie steil die Kurve ist, zählt dagegen
nicht: Ein Merkmal kann an anderer Stelle steil sein und für diese Person
trotzdem nichts beitragen.

## Aufgabe 4 – Korrelierte Merkmale
```python
from sklearn.inspection import permutation_importance

# X_train, X_test, y_train, y_test and train_model() as in code/python.py
for dropped in [None, "leukozyten_g_l"]:
    cols = [c for c in X_train.columns if c != dropped]
    model = train_model(X_train[cols], y_train)
    pi = permutation_importance(model, X_test[cols], y_test, n_repeats=10,
                                random_state=SEED, scoring="roc_auc", n_jobs=1)
    print(f"without {dropped}: PI(sofa_score) = "
          f"{pi.importances_mean[cols.index('sofa_score')]:.3f}")
```
```text
without None: PI(sofa_score) = 0.075
without leukozyten_g_l: PI(sofa_score) = 0.109
```
Die Permutation Importance des SOFA-Scores steigt von 0,075 auf 0,109. Solange
`leukozyten_g_l` im Modell ist, kann das Modell beim Vertauschen von
`sofa_score` teilweise auf die Leukozyten ausweichen, die einen Teil derselben
Information tragen; der gemessene AUC-Verlust und damit die Wichtigkeit fallen
zu klein aus. Korrelierte Merkmale teilen sich die Wichtigkeit, und keines von
beiden zeigt für sich, wie viel die gemeinsame Information wert ist.

Mit `bmi` und `gewicht_kg` (r = 0,90) funktioniert dasselbe Experiment in diesem
Datensatz nicht: BMI wirkt im datenerzeugenden Prozess gar nicht auf die
Mortalität, seine Permutation Importance liegt schon ohne `gewicht_kg` bei null.

## Bonus – Globale Sicht vs. klinische Realität
Der SOFA-Score ist in Beobachtungsstudien mit Mortalität assoziiert. Er wurde
per Expertenkonsens entwickelt, um das Ausmaß von Organdysfunktion zu
beschreiben; er ist ein Schweregrad-Marker und Prognose-Proxy, kein Hebel, an dem
man direkt ansetzen könnte. Ein hoher Wichtigkeitswert
im Modell bedeutet: „Das Modell nutzt dieses Signal“, nicht: „Eine SOFA-Reduktion
senkt kausal die Mortalität.“ Klinisch folgt daraus: Erklärbarkeit kann
Hypothesen generieren; für kausale Aussagen braucht es ein kausales Design
(randomisierte Studie oder eine sorgfältige kausale Analyse wie in Modul 15 und 19).
