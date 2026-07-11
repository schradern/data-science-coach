# Musterlösungen · Workflow für Prädiktionsmodelle und Data Leakage

## Aufgabe 1 – Ehrliche Kreuzvalidierung
```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
# build_pipeline() aus code/python.py
auc = cross_val_score(build_pipeline(), X, y,
                      cv=StratifiedKFold(5, shuffle=True, random_state=42),
                      scoring="roc_auc").mean()
```
Die Pipeline passt Imputation und Skalierung in **jeder Falte neu** nur auf den
Trainingsteil an, so sieht kein Validierungswert das Modell vorab.

## Aufgabe 2 – Leakage sichtbar machen
```python
from sklearn.preprocessing import StandardScaler
X_all = StandardScaler().fit_transform(X_num)     # auf allen Daten -> Leakage
leaky = cross_val_score(LogisticRegression(max_iter=1000), X_all, y, cv=cv, scoring="roc_auc").mean()
```
Bei reiner Skalierung ist der Effekt klein (Mittelwert/SD ändern sich kaum durch
wenige Validierungszeilen). Bei **Merkmalsauswahl** ist er groß: Die Auswahl
nutzt das Outcome aller Zeilen und greift gezielt Zufallskorrelationen ab
(siehe `code/python.py`, Abschnitt 1: 0,86 vs. 0,73).

## Aufgabe 3 – Tuning ohne Selbstbetrug
```python
grid = GridSearchCV(build_pipeline(), {"model__C": [0.01, 0.1, 1, 10]}, cv=cv, scoring="roc_auc")
grid.fit(X_train, y_train)
print(grid.best_score_)                                   # innere CV-AUC
print(roc_auc_score(y_test, grid.predict_proba(X_test)[:, 1]))   # Test-AUC
```
Die innere CV-AUC ist meist leicht optimistischer als die Test-AUC, weil `C` auf
ihr optimiert wurde. Der Testwert ist der ehrlichere Schätzer.

## Aufgabe 4 – Schwelle nach Nutzen
```python
import numpy as np
fpr, tpr, thr = roc_curve(y_test, proba)
idx = np.where(tpr >= 0.80)[0][0]
print(thr[idx], 1 - fpr[idx])     # Schwelle, zugehörige Spezifität
```
Höhere Sensitivität kostet Spezifität (mehr Fehlalarme). Gerechtfertigt, wenn ein
übersehener Fall schwer wiegt und Folgediagnostik billig/risikoarm ist (z. B.
Frühwarnung vor klinischer Verschlechterung).

## Bonus – Gruppen-Leakage
```python
from sklearn.model_selection import StratifiedGroupKFold
groups = df["patient_id"]
gcv = StratifiedGroupKFold(n_splits=5)
auc = cross_val_score(pipe, X, y, cv=gcv.split(X, y, groups), scoring="roc_auc")
```
Ohne Gruppierung landen Zeilen derselben Person in Train **und** Test; das Modell
„merkt sich" Personen und die AUC ist zu hoch. Mit `StratifiedGroupKFold` bleibt
jede Person in genau einer Falte, der ehrlichere Wert.
```
