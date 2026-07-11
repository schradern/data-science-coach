# Musterlösungen · Erklärbarkeit von Machine-Learning-Modellen

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
Kategorische Variablen (`aufnahmegrund`, `raucherstatus`) erscheinen bei der
Impurity-Methode oft künstlich hoch, weil One-Hot-Encoding viele Splitpunkte
erzeugt. Permutation Importance bewertet dagegen den tatsächlichen Leistungsabfall.

## Aufgabe 2 – PDP lesen
```python
from sklearn.inspection import PartialDependenceDisplay
fig, ax = plt.subplots(figsize=(6, 4))
PartialDependenceDisplay.from_estimator(model, X_test, features=["crp_mg_l"],
                                         kind="both", ax=ax, subsample=80)
```
Ab etwa CRP > 100 mg/l steigt das modellierte Sterberisiko erkennbar. Die
ICE-Kurven zeigen, dass bei einigen Patient:innen der Effekt früher einsetzt
(niedriger Schwellenwert) und bei anderen flach bleibt, ein Hinweis auf eine
Interaktion mit einem anderen Merkmal (z. B. SOFA-Score).

## Aufgabe 3 – SHAP lokal
```python
import numpy as np
import shap

explainer = shap.Explainer(model, X_train)
# check_additivity=False: needed for HistGradientBoostingClassifier +
# class_weight="balanced" — see code/python.py for why.
shap_values = explainer(X_test, check_additivity=False)
# Highest-risk patients
top5_idx = model.predict_proba(X_test)[:, 1].argsort()[-5:]
for idx in top5_idx:
    sv = shap_values[idx].values
    top_feat = FEATURES[np.argmax(np.abs(sv))]
    print(f"  Patient {idx}: stärkster Treiber = {top_feat}")
# Patient   7: stärkster Treiber = sofa_score
# Patient 112: stärkster Treiber = sofa_score
# Patient 119: stärkster Treiber = sofa_score
# Patient  16: stärkster Treiber = leukozyten_g_l
# Patient  71: stärkster Treiber = sofa_score
```
Auf unserem Lauf ist `sofa_score` bei 4 von 5 Hochrisikopatient:innen der
stärkste Einzeltreiber, konsistent mit der globalen SHAP-Zusammenfassung
(`assets/shap_beeswarm.png`) und der Permutation Importance (Abschnitt 2), wo
`sofa_score` ebenfalls vorne liegt.

Ohne SHAP: Man vergleicht den Merkmalswert der/des Patient:in gegen die PDP-Kurve und
identifiziert, bei welchem Merkmal der individuelle Wert im steilsten Teil der
Kurve liegt, das ist der wahrscheinlichste Haupttreiber der Vorhersage.

## Aufgabe 4 – Korrelierte Merkmale
```python
import pandas as pd

cohort = pd.read_csv("{{DATEN}}kohorte.csv")
X_ext = X.copy()
X_ext["gewicht_kg"] = cohort["gewicht_kg"]  # strongly correlated with bmi
# Re-train and recompute PI ...
```
Wenn `bmi` und `gewicht_kg` stark korreliert sind, kann das Modell bei gemischtem
`bmi` auf `gewicht_kg` ausweichen, der AUC-Abfall für `bmi` sinkt künstlich.
Beide Variablen teilen sich die Wichtigkeit, obwohl sie gemeinsam viel tragen:
ein klassischer Effekt korrelativer Redundanz bei Permutation Importance.

## Bonus – Globale Sicht vs. klinische Realität
Der SOFA-Score ist in Beobachtungsstudien mit Mortalität assoziiert und wurde
*auf Basis von Outcome-Daten* entwickelt, er ist ein Prognose-Proxy, kein
kausal verifizierbarer Treiber im mechanistischen Sinne. Ein hoher Wichtigkeitswert
im Modell bedeutet: „Das Modell nutzt dieses Signal", nicht: „Eine SOFA-Reduktion
senkt kausal die Mortalität." Klinisch folgt daraus: Erklärbarkeit kann
Hypothesen generieren, aber nur interventionelle Studien können Kausalität belegen.
