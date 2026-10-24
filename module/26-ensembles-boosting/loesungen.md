# Musterlösungen · Baum-Ensembles und Gradient Boosting

> Alle Lösungen nutzen `X, y = load_data()`, `pre = build_preprocessor()` und
> `build_pipeline(model)` aus `code/python.py`: Imputation/Encoding werden
> darin NIE vorab auf dem Gesamtdatensatz gefittet, sondern pro Split/Fold neu,
> genau wie in Modul 24. Zahlen unten aus einem Lauf auf der aktuellen Kohorte
> (N=500, 78 Ereignisse, 15,6 %).

## Aufgabe 1 – Overfitting beim einzelnen Baum
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)
X_train_imp = pre.fit_transform(X_train)   # fit on TRAIN
X_test_imp  = pre.transform(X_test)        # TEST is only transformed

train_aucs, val_aucs = [], []
for d in range(1, 16):
    tree = DecisionTreeClassifier(max_depth=d, random_state=42)
    tree.fit(X_train_imp, y_train)
    train_aucs.append(roc_auc_score(y_train, tree.predict_proba(X_train_imp)[:, 1]))
    val_aucs.append(roc_auc_score(y_test,   tree.predict_proba(X_test_imp)[:, 1]))
```
Die Trainings-AUC steigt monoton gegen 1,0. Auf unserem Lauf erreicht die
Validierungs-AUC ihr Maximum bereits bei Tiefe 3 (0,817) und fällt danach
wieder (Tiefe 8: 0,650), siehe `assets/baum_tiefe.png`: hoher Bias bei
flachen Bäumen, hohe Varianz bei tiefen. Kein einzelner Baum dominiert beide
Seiten, genau deshalb helfen Ensembles.

## Aufgabe 2 – Random Forest: Einfluss von `n_estimators`
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(5, shuffle=True, random_state=42)
for n in [10, 50, 100, 300, 500]:
    rf = RandomForestClassifier(n_estimators=n, class_weight="balanced",
                                random_state=42, n_jobs=-1)
    auc = cross_val_score(build_pipeline(rf), X, y, cv=cv, scoring="roc_auc").mean()
    print(f"n={n:>4}: AUC = {auc:.3f}")
```
Typischerweise stagniert die Güte ab ~200–300 Bäumen (bei n=300: AUC 0,791 in
unserem Lauf). Mehr Bäume reduzieren Varianz weiter, aber der Grenznutzen ist
gering. Rechenaufwand skaliert linear; ab 300 lohnt sich die Investition kaum
noch.

## Aufgabe 3 – Lernrate und Iterationen beim Boosting
```python
import time
from sklearn.ensemble import HistGradientBoostingClassifier

for lr, n_iter in [(0.30, 50), (0.01, 1500)]:
    t0 = time.time()
    hgb = HistGradientBoostingClassifier(learning_rate=lr, max_iter=n_iter,
                                          class_weight="balanced", random_state=42)
    auc = cross_val_score(build_native_missing_pipeline(hgb), X, y,
                          cv=cv, scoring="roc_auc").mean()
    dt = time.time() - t0
    print(f"lr={lr:.2f}, n_iter={n_iter}: AUC={auc:.3f}, Zeit={dt:.1f}s")
# lr=0.30, n_iter=50:   AUC=0.728
# lr=0.01, n_iter=1500: AUC=0.719
```
Kleine Lernrate macht jeden Schritt vorsichtiger, das Modell passt sich weniger
stark an Trainingsrauschen an und generalisiert oft besser, braucht aber mehr
Bäume (und Rechenzeit). Auf dieser kleinen Kohorte ist der Unterschied klein
und liegt im Rauschen der CV-Streuung (~±0,05-0,06), das Äquivalenzprinzip
`lr × n_iter ≈ const` bleibt trotzdem eine brauchbare Faustregel für größere
Datensätze.

## Aufgabe 4 – Klassenungleichheit
```python
from sklearn.metrics import classification_report

for cw in [None, "balanced"]:
    rf = RandomForestClassifier(n_estimators=300, class_weight=cw, random_state=42)
    pipe = build_pipeline(rf).fit(X_train, y_train)
    print(f"\nclass_weight={cw}")
    print(classification_report(y_test, pipe.predict(X_test), digits=3))
```
Ohne Gewichtung ist Recall für die positive Klasse (Verstorbene) an der
Standardschwelle 0,5 niedrig (auf diesem Split 0,32): Das Modell ignoriert
seltene Ereignisse zugunsten von Accuracy. Mit `class_weight="balanced"` steigt
Recall deutlich (0,58), Precision fällt etwas. Eine ähnliche Verschiebung
erreicht man ohne Gewichtung über eine niedrigere Schwelle (Modul 24). Im
klinischen Kontext (Hochrisikoerkennung) ist ein übersehener Todesfall teurer
als ein Fehlalarm, Recall hat Vorrang, siehe auch Modul 23/25 zur
Kalibrierungskehrseite dieser Gewichtung.

## Bonus – Feature Importance vergleichen
```python
from sklearn.inspection import permutation_importance

rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)
pipe = build_pipeline(rf).fit(X_train, y_train)

# Gini-based (fast, but computed on training data and biased towards features with many levels)
gini_imp = pipe.named_steps["model"].feature_importances_

# Permutation-based (more robust, but more expensive) – on the already transformed test data
X_test_imp = pipe.named_steps["pre"].transform(X_test)
perm_res = permutation_importance(pipe.named_steps["model"], X_test_imp, y_test,
                                  n_repeats=20, scoring="roc_auc", random_state=42)
perm_imp = perm_res.importances_mean
```
Gini-Importance gibt kontinuierlichen Merkmalen mit vielen Ausprägungen (BMI,
Kreatinin, Laktat, CRP) deutliche Anteile, obwohl ihre Permutation Importance
auf dem Testset um null liegt (BMI sogar leicht negativ): Sie bieten viele
mögliche Schnittpunkte, an die sich Trainingsrauschen anpassen lässt, und die
Gini-Werte stammen aus den Trainingsdaten. One-Hot-Dummies wie der
Aufnahmegrund bekommen dagegen kleine Gini-Werte. Bei korrelierten Merkmalen
(die Leukozyten hängen im Datensatz direkt vom SOFA-Score ab) unterschätzt
Permutation Importance eher beide, weil das Modell beim Durchmischen des einen
auf das andere ausweichen kann.
Siehe Modul 27 für den vollständigen Permutation-Importance-Workflow inklusive
Korrelations-Caveat.
