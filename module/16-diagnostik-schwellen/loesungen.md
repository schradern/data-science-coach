# Musterlösungen · Diagnostische Genauigkeit und Schwellenwerte

## Aufgabe 1 und 2 – 2x2 und Kennzahlen

```python
import pandas as pd
from sklearn.metrics import confusion_matrix

labs   = pd.read_csv("{{DATEN}}labor.csv")

df = pd.read_csv("{{DATEN}}kohorte.csv").merge(labs, on="patient_id").dropna(subset=["laktat_mmol_l"])
y = df["verstorben_30d"]
pred = df["laktat_mmol_l"] >= 2.0
tn, fp, fn, tp = confusion_matrix(y, pred).ravel()

sens = tp / (tp + fn)
spec = tn / (tn + fp)
ppv = tp / (tp + fp)
npv = tn / (tn + fn)
lr_pos = sens / (1 - spec)
lr_neg = (1 - sens) / spec
print(tp, fp, fn, tn, sens, spec, ppv, npv, lr_pos, lr_neg)
```

Auf den Kursdaten (Schwelle 2,0 mmol/l): TP=35, FP=164, FN=34, TN=181,
Sensitivität ≈ 0,51, Spezifität ≈ 0,53, PPV ≈ 0,18, NPV ≈ 0,84, LR+ ≈ 1,07,
LR- ≈ 0,94.

**Ehrliche Einordnung:** Sensitivität ≈ 0,51 und Spezifität ≈ 0,53 sind **nicht
akzeptabel**, sondern kaum besser als eine Münze. Die Likelihood Ratios machen
das deutlich: LR+ ≈ 1,07 und LR- ≈ 0,94 liegen beide dicht an 1, das heißt ein
positiver oder negativer Test verschiebt die Vortestwahrscheinlichkeit praktisch
nicht. Zusammen mit der AUC ≈ 0,58 (Aufgabe 3) ist Laktat bei dieser Schwelle ein
**im Wesentlichen nicht-diskriminierender** Marker. Der niedrige PPV spiegelt
zusätzlich die niedrige Ereignisrate (≈ 17 % unter den Patient:innen mit
Laktatwert) wider. Berichte PPV/NPV immer zusammen mit der Ereignisrate, weil
beide stark von der Prävalenz abhängen.

## Aufgabe 3 – Youden

```python
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

score = df["laktat_mmol_l"]
fpr, tpr, thr = roc_curve(y, score)
j = tpr - fpr
best = int(np.argmax(j))
print("AUC", roc_auc_score(y, score))
print("Youden threshold", thr[best], "sensitivity", tpr[best], "specificity", 1 - fpr[best])
```

```r
library(tidyverse); library(pROC)

cohort <- read_csv("{{DATEN}}kohorte.csv", show_col_types = FALSE)
labs   <- read_csv("{{DATEN}}labor.csv", show_col_types = FALSE)
df <- left_join(cohort, labs, by = "patient_id") |> drop_na(laktat_mmol_l)
y <- df$verstorben_30d; score <- df$laktat_mmol_l

roc_obj <- roc(y, score, quiet = TRUE, direction = "<")
cat("AUC", as.numeric(auc(roc_obj)), "\n")

# Youden at the observed cutoffs (convention pred = score >= t, as in sklearn)
cutoffs <- sort(unique(score))
j <- vapply(cutoffs, function(t) {
  pred <- score >= t
  sum(pred & y == 1) / sum(y == 1) + sum(!pred & y == 0) / sum(y == 0) - 1
}, numeric(1))
best <- cutoffs[which.max(j)]
pred <- score >= best
cat("Youden threshold", best,
    "sensitivity", sum(pred & y == 1) / sum(y == 1),
    "specificity", sum(!pred & y == 0) / sum(y == 0), "\n")
```

Auf den Kursdaten (identisch in Python und R): AUC ≈ **0,58**, ein sehr schwacher
Marker (Laktat ist im Datengenerierungsprozess nicht direkt in die Mortalitäts-
Wahrscheinlichkeit eingebaut, sondern nur indirekt über SOFA/Sepsis korreliert).
Die Youden-Schwelle liegt bei **1,70 mmol/l** (Sensitivität ≈ 0,70, Spezifität
≈ 0,49), niedriger als die klinisch einfache Schwelle 2,0 mmol/l und mit höherer
Sensitivität.

**Stolperstein – In-sample-Optimismus:** Die Youden-Schwelle wird hier an
*denselben* Daten gewählt, an denen wir sie anschließend bewerten. Diese
Sensitivität/Spezifität ist deshalb **optimistisch verzerrt**: Gewählt wird der
Cutoff, der auf *diesen* 414 Patient:innen am besten trennt, und damit auch
derjenige, der ihr Zufallsrauschen am besten ausnutzt. Auf unabhängigen
Daten fällt die Leistung typischerweise schlechter aus. Eine ehrliche Schätzung
braucht Kreuzvalidierung oder einen Testdatensatz (siehe Module 24/25). Youden
ist ein statistischer Kompromiss; klinisch kann eine niedrigere Schwelle
sinnvoller sein, wenn falsch Negative schwer wiegen.

## Aufgabe 4 – Prävalenz

Bei niedrigerer Prävalenz sinkt der PPV und der NPV steigt. Darum kann ein Test,
der auf Intensivstation brauchbar wirkt, im ambulanten Screening viele falsch
Positive erzeugen.

## Aufgabe 5 – Bericht

Beispiel:

> Laktat wurde als kontinuierlicher Marker für 30-Tage-Mortalität bewertet. Die
> AUC beschreibt die Diskrimination über alle Schwellen. Für die klinisch einfache
> Schwelle von 2,0 mmol/l berichten wir Sensitivität, Spezifität, PPV und NPV
> bezogen auf die beobachtete Ereignisrate. Die Schwelle dient als
> Risikosignal, nicht als alleinige Entscheidung.

## Aufgabe 6 – Net Benefit

```python
import numpy as np
import statsmodels.formula.api as smf

model = smf.logit("verstorben_30d ~ laktat_mmol_l", data=df).fit(disp=False)
prob = model.predict(df).to_numpy()
yv = y.to_numpy(); N = len(yv); prev = yv.mean()

def nb(pt):
    p = prob >= pt
    return (np.sum(p & (yv == 1)) - np.sum(p & (yv == 0)) * pt / (1 - pt)) / N
def nb_all(pt):
    return prev - (1 - prev) * pt / (1 - pt)

for pt in [0.10, 0.15, 0.20, 0.25]:
    print(pt, round(nb(pt), 4), round(nb_all(pt), 4))
```

```r
library(tidyverse)

cohort <- read_csv("{{DATEN}}kohorte.csv", show_col_types = FALSE)
labs   <- read_csv("{{DATEN}}labor.csv", show_col_types = FALSE)
df <- left_join(cohort, labs, by = "patient_id") |> drop_na(laktat_mmol_l)
y <- df$verstorben_30d; prev <- mean(y); N <- length(y)
model <- glm(verstorben_30d ~ laktat_mmol_l, data = df, family = binomial)
prob  <- predict(model, type = "response")
nb     <- function(pt) (sum(prob >= pt & y == 1) - sum(prob >= pt & y == 0) * pt / (1 - pt)) / N
nb_all <- function(pt) prev - (1 - prev) * pt / (1 - pt)
for (pt in c(0.10, 0.15, 0.20, 0.25)) cat(pt, nb(pt), nb_all(pt), "\n")
```

Auf den Kursdaten (identisch in beiden Sprachen):

| $p_t$ | NB Modell | NB alle behandeln | NB niemanden |
|------:|----------:|------------------:|-------------:|
| 0,10  | 0,0741    | 0,0741            | 0 |
| 0,15  | 0,0362    | 0,0196            | 0 |
| 0,20  | 0,0072    | −0,0417           | 0 |
| 0,25  | 0,0008    | −0,1111           | 0 |

Bei $p_t = 0{,}10$ ist das Modell nicht von „alle behandeln“ zu unterscheiden.
Nur zwischen $p_t = 0{,}10$ und knapp über $0{,}25$ liegt das Modell über **beiden**
Referenzstrategien, und auch dort nur knapp (bei 0,25 noch um 0,0008), passend
zur schwachen AUC ≈ 0,58.

**Warum Youden das nicht leistet:** Youden liefert eine einzige Schwelle auf dem
Rohscore und gewichtet FP und FN gleich. Der Net Benefit dagegen variiert die
Schwellenwahrscheinlichkeit $p_t$ und macht damit explizit, wie viele falsch
Positive ein vermiedener falsch Negativer wert ist ($(1-p_t)/p_t$), und bezieht
die Prävalenz über die „alle behandeln“-Linie mit ein. Modul 25 vertieft das.
