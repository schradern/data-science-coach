# Musterlösungen · Workflow für Prädiktionsmodelle und Data Leakage

**Vorbereitung.** Die Lösungen setzen auf den Bausteinen aus [`code/python.py`](code/python.py) auf. Führe diesen Block einmal im Projektordner aus, lokal oder in Colab nach dem Klonen ([in Colab öffnen](https://colab.research.google.com/github/schradern/data-science-coach/blob/main/notebooks/start.ipynb)). Danach laufen die Blöcke dieser Seite der Reihe nach; der Bonus-Block läuft auch für sich allein. `runpy` lädt nur die Definitionen des Skripts, `main()` läuft dabei nicht.

```python
# Setup: run once from the project folder.
import runpy
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

lesson = runpy.run_path("module/24-praediktion-workflow/code/python.py")
SEED, NUMERIC, build_data, build_pipeline = (
    lesson[name] for name in ("SEED", "NUMERIC", "build_data", "build_pipeline"))
X, y = build_data()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=SEED)
```

## Aufgabe 1 – Ehrliche Kreuzvalidierung
```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
# build_pipeline() from code/python.py
auc = cross_val_score(build_pipeline(), X, y,
                      cv=StratifiedKFold(5, shuffle=True, random_state=42),
                      scoring="roc_auc").mean()
```
Ergebnis: CV-AUC ≈ 0,78. Die Pipeline passt Imputation und Skalierung in **jeder
Falte neu** nur auf den Trainingsteil an, so sieht kein Validierungswert das Modell vorab.

## Aufgabe 2 – Leakage sichtbar machen
```python
from sklearn.pipeline import make_pipeline
prep = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

X_all = prep.fit_transform(X[NUMERIC])            # on all data -> leakage
leaky = cross_val_score(LogisticRegression(max_iter=1000), X_all, y,
                        cv=cv, scoring="roc_auc").mean()
honest = cross_val_score(make_pipeline(prep, LogisticRegression(max_iter=1000)),
                         X[NUMERIC], y, cv=cv, scoring="roc_auc").mean()
print(f"leaky {leaky:.3f}   honest {honest:.3f}")   # leaky 0.787   honest 0.787
```
Bei reiner Skalierung (und Median-Imputation) ist der Effekt hier praktisch null
(Mittelwert/SD/Median ändern sich kaum durch wenige Validierungszeilen). Bei **Merkmalsauswahl** ist er groß: Die Auswahl
nutzt das Outcome aller Zeilen und greift gezielt Zufallskorrelationen ab
(siehe `code/python.py`, Abschnitt 1: 0,86 vs. 0,73).

## Aufgabe 3 – Tuning ohne Selbstbetrug
```python
grid = GridSearchCV(build_pipeline(), {"model__C": [0.01, 0.1, 1, 10]}, cv=cv, scoring="roc_auc")
grid.fit(X_train, y_train)
print(grid.best_score_)                                   # inner CV-AUC
print(roc_auc_score(y_test, grid.predict_proba(X_test)[:, 1]))   # Test-AUC
```
Ergebnis: innere CV-AUC 0,743 (bestes `C` = 0,01), Test-AUC 0,886. Erwartet hätte
man das Gegenteil: Die innere CV-AUC ist tendenziell optimistisch, weil `C` auf ihr
ausgewählt wurde. Dass der Testwert hier deutlich höher liegt, ist Zufall des
Splits: Das Testset enthält nur 19 Todesfälle; das 95-%-KI seiner AUC reicht grob
±0,1. Der Testwert ist für dieses eine Modell in derselben Population annähernd
unverzerrt, weil nichts an ihm auf dem Testset gewählt wurde, aber er ist unsicher; berichte ihn deshalb mit
95-%-KI (z. B. per Bootstrap) und lies ihn zusammen mit der CV-AUC.

## Aufgabe 4 – Schwelle nach Nutzen
```python
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_predict

# Choose the threshold on the training data (out-of-fold predictions), not on the test set
oof = cross_val_predict(grid.best_estimator_, X_train, y_train, cv=cv,
                        method="predict_proba")[:, 1]
fpr, tpr, thr = roc_curve(y_train, oof)
threshold = thr[np.where(tpr >= 0.80)[0][0]]

# evaluate on the test set exactly once
pred = grid.predict_proba(X_test)[:, 1] >= threshold
sens = pred[y_test == 1].mean()
spec = 1 - pred[y_test == 0].mean()
print(f"Schwelle {threshold:.2f}: Sensitivität {sens:.2f}, Spezifität {spec:.2f}")
# Schwelle 0.44: Sensitivität 0.89, Spezifität 0.54
```
Die Schwelle wird wie im README verlangt auf den Trainingsdaten festgelegt; auf
dem Testset erreicht sie hier 89 % Sensitivität bei nur 54 % Spezifität.
Höhere Sensitivität kostet Spezifität (mehr Fehlalarme). Gerechtfertigt, wenn ein
übersehener Fall schwer wiegt und Folgediagnostik billig/risikoarm ist (z. B.
Frühwarnung vor klinischer Verschlechterung).

## Bonus – Gruppen-Leakage
```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

vitals = pd.read_csv("{{DATEN}}vitalwerte.csv")
cohort = pd.read_csv("{{DATEN}}kohorte.csv")
df = vitals.merge(cohort[["patient_id", "alter", "crp_mg_l", "verstorben_30d"]], on="patient_id")
X = df[["herzfrequenz", "map_mmhg", "temperatur_c", "spo2_prozent", "alter", "crp_mg_l"]]
y = df["verstorben_30d"]
groups = df["patient_id"]             # up to four rows (days 0-3) per person

pipe = make_pipeline(SimpleImputer(strategy="median"),
                     RandomForestClassifier(n_estimators=300, random_state=42))
random_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
group_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
auc_random = cross_val_score(pipe, X, y, cv=random_cv, scoring="roc_auc").mean()
auc_group = cross_val_score(pipe, X, y, cv=group_cv, groups=groups, scoring="roc_auc").mean()
print(f"zufällige Falten: {auc_random:.2f}   gruppierte Falten: {auc_group:.2f}")
# zufällige Falten: 0.83   gruppierte Falten: 0.65
```
Ohne Gruppierung landen Zeilen derselben Person in Train **und** Test. Der Random
Forest erkennt die Person an ihren je Patient:in konstanten Werten (`alter`,
`crp_mg_l`) wieder und „weiß“ ihr Outcome schon: 0,83 statt ehrlicher 0,65. Mit
`StratifiedGroupKFold` bleibt jede Person in genau einer Falte. Mit einer
logistischen Regression ist der Unterschied auf denselben Daten kaum sichtbar
(0,70 vs. 0,70), weil sich ein lineares Modell einzelne Personen schlecht merken
kann. Gruppen-Leakage ist also gerade bei flexiblen Modellen gefährlich.
