# Musterlösungen · Neuronale Netze und Deep Learning

**Vorbereitung.** Die Lösungen setzen auf den Bausteinen aus [`code/python.py`](code/python.py) auf. Führe diesen Block einmal im Projektordner aus, lokal oder in Colab nach dem Klonen ([in Colab öffnen](https://colab.research.google.com/github/schradern/data-science-coach/blob/main/notebooks/start.ipynb)). Danach laufen die Blöcke dieser Seite der Reihe nach. `runpy` lädt nur die Definitionen des Skripts, `main()` läuft dabei nicht.

```python
# Setup: run once from the project folder.
import runpy
from sklearn.model_selection import StratifiedKFold, train_test_split

lesson = runpy.run_path("module/30-deep-learning/code/python.py")
SEED, build_data, build_preprocessor, make_lr, make_gb, make_mlp, make_mlp_es = (
    lesson[name] for name in ("SEED", "build_data", "build_preprocessor",
                              "make_lr", "make_gb", "make_mlp", "make_mlp_es"))
X, y = build_data()
pre = build_preprocessor()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=SEED)
mlp_pipe = make_mlp_es(build_preprocessor())   # early-stopping variant from section 2
```

## Aufgabe 1 – Pipeline mit MLP

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# pre and column lists from code/python.py
mlp_cv = Pipeline([("pre", pre),
                   ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), alpha=0.01,
                                         max_iter=500, random_state=42))])
auc = cross_val_score(mlp_cv, X, y,
                      cv=StratifiedKFold(5, shuffle=True, random_state=42),
                      scoring="roc_auc").mean()
print(f"MLP CV-AUC: {auc:.3f}")
# MLP CV-AUC: 0.656
```

Nimm für die CV bewusst ein Netz **ohne** `early_stopping`: `mlp_pipe` aus
Abschnitt 2 bricht nach der Validierungsgenauigkeit ab und kommt in derselben
CV nur auf eine AUC um 0,50 (siehe Aufgabe 2).

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
ax2.plot(mlp.validation_scores_, color="firebrick", label="Validierungsgenauigkeit")
ax1.set_xlabel("Epoche")
ax1.set_ylabel("Verlust")
ax2.set_ylabel("Validierungsgenauigkeit (Accuracy)")
print(f"Early stopping nach {mlp.n_iter_} Epochen")
plt.show()
```

Überanpassung erkennt man daran, dass der Trainingsverlust weiter sinkt,
während die Validierungsgüte stagniert oder fällt. Early Stopping hält an
dem Punkt an, an dem die Validierungsgüte über mehrere Epochen nicht mehr
verbessert wird (`n_iter_no_change` Epochen, Standard = 10). Achtung:
`validation_scores_` ist die Genauigkeit (Accuracy), keine AUC. In unserem Lauf
stoppt das Training nach 15 Epochen bei 0,842, also etwa beim Anteil der
Überlebenden; das Netz sagt zu diesem Zeitpunkt praktisch immer „überlebt“
vorher (Test-AUC 0,407). Mit einem seltenen Ereignis ist Accuracy als
Abbruchkriterium ungeeignet.

## Aufgabe 3 – Regularisierung vergleichen

```python
for alpha in [0.0001, 0.001, 0.01, 0.1, 1.0]:
    pipe = Pipeline([("pre", pre),
                     ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32),
                                           alpha=alpha, max_iter=500,
                                           random_state=42))])
    auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean()
    print(f"  alpha={alpha:.4f}  AUC={auc:.3f}")
```

In unserem Lauf liegen alle fünf Werte zwischen 0,641 und 0,656 (knapp vorn:
alpha = 0,01), also weit innerhalb der Fold-Streuung; die Regularisierungsstärke
ist auf diesen Daten nicht der begrenzende Faktor.

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
    """Mean, SD, naive and Nadeau-Bengio-corrected t-interval (approximations)."""
    k, mean, sd = len(scores), scores.mean(), scores.std(ddof=1)
    t_crit = stats.t.ppf(0.975, df=k - 1)
    naive = t_crit * sd * np.sqrt(1 / k)                     # assumes independent folds
    corrected = t_crit * sd * np.sqrt(1 / k + 1 / (k - 1))   # n_test/n_train = 1/(k-1)
    return mean, sd, (mean - naive, mean + naive), (mean - corrected, mean + corrected)

fold_scores = {
    "Logistische Regression": cross_val_score(make_lr(build_preprocessor()), X, y, cv=cv, scoring="roc_auc"),
    "Gradient Boosting":      cross_val_score(make_gb(build_preprocessor()), X, y, cv=cv, scoring="roc_auc"),
    "MLP":                    cross_val_score(make_mlp(build_preprocessor()), X, y, cv=cv, scoring="roc_auc"),
}
for name, scores in fold_scores.items():
    mean, sd, naive, corrected = summarise_folds(scores)
    print(f"  {name:<24} Mean={mean:.3f} SD={sd:.3f} "
          f"naiv=[{naive[0]:.3f}, {naive[1]:.3f}] korrigiert=[{corrected[0]:.3f}, {corrected[1]:.3f}]")

# Paired comparison – same folds for every model.
for other in ("Gradient Boosting", "MLP"):
    diff = fold_scores["Logistische Regression"] - fold_scores[other]
    mean, sd, naive, corrected = summarise_folds(diff)
    print(f"  LR - {other}: Mean diff={mean:+.3f}  naiv=[{naive[0]:+.3f}, {naive[1]:+.3f}] "
          f"korrigiert=[{corrected[0]:+.3f}, {corrected[1]:+.3f}]")
```

Ausgeführt auf diesem Datensatz (n=500, 78 Ereignisse) liefert das:

| Modell | Mean | SD | naives 95-%-Intervall | korrigiert (Nadeau-Bengio) |
|---|---|---|---|---|
| Logistische Regression | 0,775 | 0,037 | [0,729; 0,821] | [0,706; 0,844] |
| Gradient Boosting | 0,735 | 0,036 | [0,691; 0,780] | [0,669; 0,802] |
| MLP | 0,656 | 0,047 | [0,598; 0,714] | [0,569; 0,743] |

Und die gepaarten Fold-Differenzen (LR minus jeweils anderes Modell):

- LR − Gradient Boosting: Mean-Diff +0,040, naiv [+0,012; +0,067], korrigiert [−0,002; +0,081] – nur das naive Intervall schließt 0 aus, das reicht bei korrelierten Folds nicht.
- LR − MLP: Mean-Diff +0,119, naiv [+0,072; +0,166], korrigiert [+0,048; +0,189] – auch das korrigierte Intervall schließt 0 aus, ein deutlicher Hinweis auf einen echten Unterschied.

Führe den Code selbst aus, deine genauen Zahlen können je nach installierter
sklearn-Version leicht abweichen. Wichtig ist die vorsichtige Einordnung,
nicht ein bestimmtes Modell als „Sieger“ zu verkünden: Ohne Streuungsmaß
sähe die Mittelwert-Rangfolge (0,775 > 0,735 > 0,656) nach einem klaren,
dreistufigen Sieg von LR über GB über MLP aus. Beide Intervalle sind aber
nur Näherungen: Die fünf Fold-Werte sind nicht unabhängig, und die Varianz
einer k-fachen Kreuzvalidierung lässt sich nicht unverzerrt schätzen (Bengio
und Grandvalet 2004). Das naive t-Intervall ist deshalb zu schmal, das
korrigierte nach Nadeau und Bengio vorsichtiger. Gestützt ist damit nur der
Abstand zum MLP. Der Abstand zwischen den beiden Klassikern verschwindet im
korrigierten Intervall und zählt nicht als Rangfolge. Die vorsichtige Aussage:
**Das MLP schneidet auf diesem Datensatz deutlich schlechter ab; welcher der
beiden Klassiker besser ist, lässt sich bei dieser Stichprobengröße nicht
entscheiden.** Beide Klassiker sind dateneffizient (Gradient Boosting braucht
keine Skalierung und nutzt Nichtlinearitäten implizit über Splits; logistische
Regression hat wenige Parameter und überpasst kaum), während das MLP Tausende
von Beispielen bräuchte, um seine Parameterzahl sinnvoll zu schätzen. Das
zentrale Ergebnis, mehr Modellkomplexität hilft auf kleinen klinischen
Tabellendaten nicht automatisch, gilt unabhängig davon, welcher der beiden
einfacheren Klassiker im Einzelfall knapp vorn liegt.

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

Auf 500 Patient:innen bringt eine tiefere Architektur (128, 64, 32) keinen
Vorteil (in unserem Lauf: (32,) 0,710, (64, 32) 0,662, (128, 64, 32) 0,652). Ob
die flache Variante wirklich besser ist, lässt sich aus diesen Mittelwerten
nicht ablesen: Die Unterschiede liegen in der Größenordnung der Fold-Streuung
(Aufgabe 4 zeigt, wie man das mit gepaarten Intervallen prüft). Klar ist nur,
dass mehr Kapazität hier nichts gewinnt. (Mit `max_iter=300` statt 500 kommt
(64, 32) auf 0,662 statt 0,656 aus Aufgabe 1, auch das ist Rauschen.)
Mehr Schichten machen nur Sinn, wenn genug Trainingsdaten vorhanden sind,
um die zusätzlichen Parameter zu schätzen. Bei n < 2 000 und einer kleinen
Merkmalszahl ist eine flache Architektur ((32,) oder (64, 32)) robuster.
