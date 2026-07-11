# Musterlösungen · Neuronale Netze und Deep Learning

## Aufgabe 1 – Pipeline mit MLP

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# pre and column lists from code/python.py
auc = cross_val_score(mlp_pipe, X, y,
                      cv=StratifiedKFold(5, shuffle=True, random_state=42),
                      scoring="roc_auc").mean()
print(f"MLP CV-AUC: {auc:.3f}")
```

`StandardScaler` ist beim MLP unverzichtbar, weil Gradient Descent auf der
euklidischen Verlustlandschaft arbeitet: Merkmale mit großen Absolutwerten
(CRP: 0–300 mg/l) dominieren die Gradienten, während kleine Merkmale (z. B.
BMI: 15–45) kaum Einfluss haben. Entscheidungsbäume splitten dagegen nur auf
Schwellenwerten, die Einheit spielt keine Rolle.

## Aufgabe 2 – Lernkurve interpretieren

```python
import matplotlib.pyplot as plt

mlp_pipe.fit(X_train, y_train)
mlp = mlp_pipe.named_steps["mlp"]

fig, ax1 = plt.subplots(figsize=(8, 4))
ax1.plot(mlp.loss_curve_, color="steelblue", label="Trainingsverlust")
ax2 = ax1.twinx()
ax2.plot(mlp.validation_scores_, color="firebrick", label="Validierungs-AUC")
ax1.set_xlabel("Epoche")
ax1.set_ylabel("Verlust")
ax2.set_ylabel("Validierungs-AUC")
print(f"Early stopping nach {mlp.n_iter_} Epochen")
plt.show()
```

Überanpassung erkennt man daran, dass der Trainingsverlust weiter sinkt,
während die Validierungsgüte stagniert oder fällt. Early Stopping hält an
dem Punkt an, an dem die Validierungsgüte über mehrere Epochen nicht mehr
verbessert wird (`n_iter_no_change` Epochen, Standard = 10).

## Aufgabe 3 – Regularisierung vergleichen

```python
for alpha in [0.0001, 0.001, 0.01, 0.1, 1.0]:
    pipe = Pipeline([("pre", pre),
                     ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32),
                                           alpha=alpha, max_iter=300,
                                           random_state=42))])
    auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean()
    print(f"  alpha={alpha:.4f}  AUC={auc:.3f}")
```

Zu kleines alpha (0.0001) führt zu Überanpassung, das Netz passt sich
den Trainingsdaten zu stark an. Zu großes alpha (1.0) erzwingt sehr kleine
Gewichte (Unteranpassung), das Modell ist zu stark regularisiert und lernt
das Signal nicht. Der beste Wert liegt meist im mittleren Bereich (0.001–0.1)
und ist datensatzabhängig.

## Aufgabe 4 – Ehrlicher Modellvergleich

```python
from scipy import stats
import numpy as np

def summarise_folds(scores):
    k, mean, sd = len(scores), scores.mean(), scores.std(ddof=1)
    half_width = stats.t.ppf(0.975, df=k - 1) * sd / np.sqrt(k)
    return mean, sd, mean - half_width, mean + half_width

fold_scores = {
    "Logistische Regression": cross_val_score(make_lr(), X, y, cv=cv, scoring="roc_auc"),
    "Gradient Boosting":      cross_val_score(make_gb(), X, y, cv=cv, scoring="roc_auc"),
    "MLP":                    cross_val_score(mlp_pipe,  X, y, cv=cv, scoring="roc_auc"),
}
for name, scores in fold_scores.items():
    mean, sd, lo, hi = summarise_folds(scores)
    print(f"  {name:<28} Mean={mean:.3f} SD={sd:.3f} 95%-CI=[{lo:.3f}, {hi:.3f}]")

# Paired comparison — same folds for every model.
for other in ("Gradient Boosting", "MLP"):
    diff = fold_scores["Logistische Regression"] - fold_scores[other]
    mean, sd, lo, hi = summarise_folds(diff)
    print(f"  LR - {other}: Mean diff={mean:+.3f}  95%-CI=[{lo:+.3f}, {hi:+.3f}]")
```

Ausgeführt auf diesem Datensatz (n=500, 78 Ereignisse) liefert das:

| Modell | Mean | SD | 95-%-CI |
|---|---|---|---|
| Logistische Regression | 0,775 | 0,037 | [0,729; 0,821] |
| Gradient Boosting | 0,735 | 0,036 | [0,690; 0,780] |
| MLP | 0,656 | 0,047 | [0,598; 0,714] |

Und die gepaarten Fold-Differenzen (LR minus jeweils anderes Modell):

- LR − Gradient Boosting: Mean-Diff +0,040, 95-%-CI [+0,010; +0,069] — CI schließt 0 nur **knapp** aus (k=5 Folds), ein fragiles, kein robustes Ergebnis.
- LR − MLP: Mean-Diff +0,119, 95-%-CI [+0,072; +0,166] — CI schließt 0 deutlich aus, ein vergleichsweise klarer Unterschied.

Führe den Code selbst aus, deine genauen Zahlen können je nach installierter
sklearn-Version leicht abweichen. Wichtig ist die ehrliche Einordnung,
nicht ein bestimmtes Modell als „Sieger" zu verkünden: Ohne Streuungsmaß
sähe die Mittelwert-Rangfolge (0,775 > 0,735 > 0,656) nach einem klaren,
dreistufigen Sieg von LR über GB über MLP aus. Die gepaarte CI-Betrachtung
zeigt aber, dass nur der Abstand zum MLP robust belegt ist. Der Abstand
zwischen den beiden Klassikern ist zwar statistisch von 0 verschieden, aber
so knapp (CI-Untergrenze 0,010, nur 5 Fold-Differenzen), dass er nicht als
verlässliche Rangfolge zwischen Gradient Boosting und logistischer
Regression zählt. Die ehrliche Aussage: **das MLP ist auf diesem Datensatz
eindeutig schwächer; welcher der beiden Klassiker besser ist, lässt sich bei
dieser Stichprobengröße nicht robust entscheiden.** Beide Klassiker sind
dateneffizient (Gradient Boosting braucht keine Skalierung und nutzt
Nichtlinearitäten implizit über Splits; logistische Regression hat wenige
Parameter und überpasst kaum), während das MLP Tausende von Beispielen
bräuchte, um seine Parameterzahl sinnvoll zu schätzen. Das zentrale
Ergebnis, mehr Modellkomplexität hilft auf kleinen klinischen Tabellendaten
nicht automatisch, gilt unabhängig davon, welcher der beiden einfacheren
Klassiker im Einzelfall knapp vorn liegt, nur eben mit der tatsächlich
belegten Unsicherheit statt einer scheinbar sauberen Rangfolge.

## Bonus – Netzarchitektur variieren

```python
for arch in [(32,), (64, 32), (128, 64, 32)]:
    pipe = Pipeline([("pre", pre),
                     ("mlp", MLPClassifier(hidden_layer_sizes=arch,
                                           alpha=0.01, max_iter=300,
                                           random_state=42))])
    auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean()
    print(f"  {str(arch):<18} AUC={auc:.3f}")
```

Auf 500 Patient:innen bringt eine tiefere Architektur (128, 64, 32) oft keinen
Vorteil, der Gewinn durch mehr Kapazität wird durch Überanpassung kompensiert.
Mehr Schichten machen nur Sinn, wenn genug Trainingsdaten vorhanden sind,
um die zusätzlichen Parameter zu schätzen. Bei n < 2 000 und einer kleinen
Merkmalszahl ist eine flache Architektur ((32,) oder (64, 32)) robuster.
