# Musterlösungen · Einführung in das maschinelle Lernen

> Vergleiche zuerst mit deinem eigenen Versuch. Es gibt oft mehrere richtige Wege.

## Aufgabe 1 – Genauigkeitsparadox

**Python**
```python
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

df = pd.read_csv("{{DATEN}}kohorte.csv")
y  = df["verstorben_30d"]

# Naive baseline: always predict the majority class (lebt = 0)
y_pred_baseline = np.zeros(len(y), dtype=int)
print("Accuracy (Baseline):", round(accuracy_score(y, y_pred_baseline), 3))  # 0.844
print("AUC (Baseline):     ", roc_auc_score(y, y_pred_baseline))              # 0.5
```

**R**
```r
library(readr)
df <- read_csv("{{DATEN}}kohorte.csv", show_col_types = FALSE)
y  <- df$verstorben_30d

# Accuracy: always predict 0
cat("Accuracy (Baseline):", round(mean(y == 0), 3), "\n")   # 0.844

# AUC for a constant zero predictor = 0.5
u  <- wilcox.test(rep(0, sum(y == 1)), rep(0, sum(y == 0)))$statistic
n1 <- sum(y == 1); n0 <- sum(y == 0)
cat("AUC (Baseline):     ", as.numeric(u) / (n1 * n0), "\n")   # 0.5
```

**Warum Accuracy ungeeignet ist:** Bei ~16 % Mortalität erreicht ein Modell, das
niemals einen Sterbefall erkennt, 84 % Accuracy. Der AUC = 0,5 entspricht einem
Zufallsklassifikator und macht den Informationswert korrekt sichtbar: null. AUC
misst die Wahrscheinlichkeit, dass das Modell einen zufälligen Sterbefall höher
einstuft als einen Überlebenden, unabhängig vom gewählten Schwellenwert.

## Aufgabe 2 – Split-Varianz

**Python**
```python
SEED = 42
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("{{DATEN}}kohorte.csv")
df["is_sepsis"] = (df["aufnahmegrund"] == "Sepsis").astype(int)
df["is_smoker"] = (df["raucherstatus"] == "aktiv").astype(int)
X = df[["alter", "sofa_score", "crp_mg_l", "diabetes", "is_sepsis", "is_smoker"]]
y = df["verstorben_30d"]

pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)),
])

for seed in [0, 42, 99]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    pipeline.fit(X_train, y_train)
    auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
    print(f"Seed {seed:>3}: AUC = {auc:.3f}")
```

**R**
```r
suppressPackageStartupMessages(library(tidyverse))

cohort <- read_csv("{{DATEN}}kohorte.csv", show_col_types = FALSE) |>
  mutate(is_sepsis = as.integer(aufnahmegrund == "Sepsis"),
         is_smoker = as.integer(raucherstatus == "aktiv"))

for (seed in c(0, 42, 99)) {
  set.seed(seed)
  idx_pos   <- which(cohort$verstorben_30d == 1)
  idx_neg   <- which(cohort$verstorben_30d == 0)
  train_idx <- c(sample(idx_pos, floor(0.8 * length(idx_pos))),
                 sample(idx_neg, floor(0.8 * length(idx_neg))))
  tr <- cohort[train_idx, ]; te <- cohort[-train_idx, ]
  fit    <- glm(verstorben_30d ~ alter + sofa_score + crp_mg_l + diabetes +
                  is_sepsis + is_smoker, data = tr, family = binomial)
  probs  <- predict(fit, newdata = te, type = "response")
  n1 <- sum(te$verstorben_30d == 1); n0 <- sum(te$verstorben_30d == 0)
  u  <- wilcox.test(probs[te$verstorben_30d == 1], probs[te$verstorben_30d == 0])$statistic
  cat("Seed", seed, ": AUC =", round(as.numeric(u) / (n1 * n0), 3), "\n")
}
```

**Befund:** Seed 0: AUC = 0,787; Seed 42: AUC = 0,880; Seed 99: AUC = 0,874,
eine Schwankung von ~0,09 zwischen den Extremen, rein durch Zufall, weil jedes
Testset nur ~16 Ereignisse enthält (16 % von 100). Ein einzelner Split ist kein
verlässliches Gütemaß. Kreuzvalidierung mittelt über mehrere Folds und liefert
eine stabilere Schätzung mit messbarer Standardabweichung.

## Aufgabe 3 – Merkmalssätze vergleichen

**Python**
```python
SEED = 42
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("{{DATEN}}kohorte.csv")
df["is_sepsis"] = (df["aufnahmegrund"] == "Sepsis").astype(int)
df["is_smoker"] = (df["raucherstatus"] == "aktiv").astype(int)
X = df[["alter", "sofa_score", "crp_mg_l", "diabetes", "is_sepsis", "is_smoker"]]
y = df["verstorben_30d"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)),
])

feature_sets = [
    ["sofa_score"],
    ["sofa_score", "alter"],
    ["sofa_score", "alter", "crp_mg_l", "diabetes", "is_sepsis", "is_smoker"],
]
for features in feature_sets:
    scores = cross_val_score(pipeline, X_train[features], y_train, cv=cv, scoring="roc_auc")
    print(f"{features}: CV-AUC = {scores.mean():.3f} ± {scores.std():.3f}")
# ['sofa_score']:                                         CV-AUC = 0.731 ± 0.072
# ['sofa_score', 'alter']:                                CV-AUC = 0.761 ± 0.062
# ['sofa_score', 'alter', 'crp_mg_l', 'diabetes', ...]:   CV-AUC = 0.728 ± 0.060
```

**Prinzip:** Mit mehr relevanter Information steigt der AUC *typischerweise*
an (Underfitting sinkt), das gilt aber nicht garantiert für jeden einzelnen
Lauf. Auf unseren Daten gewinnt tatsächlich **nicht** das vollständige Modell:
`sofa_score + alter` (CV-AUC 0,761) schlägt sogar den vollen 6-Merkmal-Satz
(0,728), sofa_score allein liegt dazwischen (0,731). Erwarte hier keinen
sauberen, monoton steigenden Trend, sondern lies die Standardabweichungen
(±0,06–0,07) mit: Die drei Werte überlappen sich stark, bei N=400 Trainings-
zeilen und ~62 Ereignissen ist die CV-Schätzung selbst unsicher. Der ehrliche
Befund ist: Mehr (schwach informative) Merkmale bringen hier keinen klaren
Zugewinn, und können durch zusätzliches Rauschen sogar leicht schaden, genau
der Effekt, den Kreuzvalidierung aufdecken soll, statt ihn wegzumitteln. In
größeren, realen Kohorten mit stärkeren Effekten sähe der Trend oft klarer aus.

## Aufgabe 4 – Schwellenwert und klinische Abwägung

**Python**
```python
SEED = 42
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("{{DATEN}}kohorte.csv")
df["is_sepsis"] = (df["aufnahmegrund"] == "Sepsis").astype(int)
df["is_smoker"] = (df["raucherstatus"] == "aktiv").astype(int)
X = df[["alter", "sofa_score", "crp_mg_l", "diabetes", "is_sepsis", "is_smoker"]]
y = df["verstorben_30d"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)),
])
pipeline.fit(X_train, y_train)
probs = pipeline.predict_proba(X_test)[:, 1]

for threshold in [0.10, 0.15, 0.20, 0.30, 0.50]:
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_test, preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        print(f"Threshold {threshold:.2f}: Sensitivity = {sens:.2f}  Specificity = {spec:.2f}")
# Threshold 0.10: Sensitivity = 1.00  Specificity = 0.01
# Threshold 0.15: Sensitivity = 1.00  Specificity = 0.07
# Threshold 0.20: Sensitivity = 1.00  Specificity = 0.14
# Threshold 0.30: Sensitivity = 1.00  Specificity = 0.36
# Threshold 0.50: Sensitivity = 0.81  Specificity = 0.69
```

**Befund:** Weil `class_weight="balanced"` die Wahrscheinlichkeiten stark nach
oben verzerrt (Abschnitt 5/Modul 25), reicht schon Schwelle 0,30, um *jeden*
Sterbefall im Testset zu erkennen (Sensitivität 1,00), allerdings auf Kosten
fast aller Spezifität (0,36 und darunter bei Schwellen ≤0,30, das bedeutet
sehr viele Fehlalarme). Erst bei Schwelle 0,50 sinkt die Sensitivität auf 0,81
und die Spezifität steigt auf 0,69, ein ausgewogenerer Punkt.

**Klinische Abwägung:**

- **Knappe ICU-Kapazitäten:** Schwelle nahe 0,50, sonst wird praktisch jede
 Person als Hochrisiko markiert (siehe Spezifität 0,01–0,36 bei Schwellen
 ≤0,30), das sprengt jede reale Kapazität.

- **Fehlalarme sehr kostspielig:** Schwelle bei oder über 0,50, höhere
 Spezifität, weniger unnötige Eskalationen, auch wenn dafür Sensitivität
 sinkt.

- **Kein verpasster Hochrisikofall tolerierbar:** Schwelle ≤0,30 garantiert auf
 diesen Daten Sensitivität 1,00, aber praktisch nur mit Fehlalarmquoten, die in
 der Realität kaum tragbar wären, ein Beispiel dafür, warum die Schwelle nicht
 isoliert von der Kapazität gewählt werden darf.

Die Schwelle ist eine klinische Entscheidung, kein statistischer Standard, und
sie hängt hier zusätzlich stark davon ab, dass die Wahrscheinlichkeiten selbst
verzerrt sind (`class_weight="balanced"`), nicht nur von Sensitivität/
Spezifität allein.

## Aufgabe 5 – Kalibrierungskurve interpretieren

**Befund:** `check_calibration()` zeigt eine mittlere vorhergesagte Mortalität
von **46,0 %**, während im Testset nur **16,0 %** tatsächlich sterben, das
Modell **überschätzt das Risiko erheblich**. Ursache ist `class_weight=
"balanced"` (Abschnitt 3/6): Es gewichtet die Verlustfunktion so um, als wären
Tod und Überleben gleich häufig. Die Rangordnung bleibt dabei praktisch gleich
(AUC 0,880 mit, 0,878 ohne Gewichtung), aber jede absolute Wahrscheinlichkeit ist verzerrt.

**Klinische Konsequenz:** In der Praxis würde eine solche Überschätzung zu
unnötiger Therapieeskalation führen, z. B. ICU-Aufnahme statt Normalstation,
mit Ressourcenverschwendung und potenziellen iatrogenen Schäden. Ein gut
kalibriertes Modell ist Voraussetzung für absolute Risikoangaben an
Patient:innen oder für Ressourcenplanung. Ein hoher AUC (gute Rangreihenfolge)
und schlechte Kalibrierung (falsche absolute Werte) können gleichzeitig
auftreten, genau das sehen wir hier.

**Der Effekt heißt:** class-weight-induzierte Verzerrung der Vorhersage-
wahrscheinlichkeiten (nicht „Overfitting“, die Rangordnung ist ja intakt).

**Post-hoc-Korrektur:**

```python
# continues the code from task 4 (pipeline, X_train, X_test, probs)
from sklearn.calibration import CalibratedClassifierCV

calibrated = CalibratedClassifierCV(pipeline, method="sigmoid", cv=5)
calibrated.fit(X_train, y_train)
probs_cal = calibrated.predict_proba(X_test)[:, 1]
print(f"Mittlere Vorhersage vorher: {probs.mean():.1%}  nachher: {probs_cal.mean():.1%}")
# vorher: 46.0%  nachher: 17.9% (close to the observed 16%)
```

`method="isotonic"` ist flexibler, braucht aber mehr Daten. Bei kleinem
Testset (~16 Ereignisse) bleibt die Beurteilung der Kalibrierung unsicher,
externe Validierungsdaten sind unerlässlich. Modul 25 baut diesen Schritt zu
einem vollständigen Kalibrierungs-Workflow aus.

## Bonus – Bias nach Subgruppe

**Python**
```python
SEED = 42
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("{{DATEN}}kohorte.csv")
df["is_sepsis"] = (df["aufnahmegrund"] == "Sepsis").astype(int)
df["is_smoker"] = (df["raucherstatus"] == "aktiv").astype(int)
X = df[["alter", "sofa_score", "crp_mg_l", "diabetes", "is_sepsis", "is_smoker"]]
y = df["verstorben_30d"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)),
])
pipeline.fit(X_train, y_train)
probs = pipeline.predict_proba(X_test)[:, 1]

for group_val, label in [(0, "ohne Diabetes"), (1, "mit Diabetes")]:
    mask = X_test["diabetes"] == group_val
    n_events = y_test[mask].sum()
    if mask.sum() > 5 and n_events > 0:
        auc_sub = roc_auc_score(y_test[mask], probs[mask])
        print(f"{label}: n = {mask.sum()}, Ereignisse = {n_events}, AUC = {auc_sub:.3f}")
    else:
        print(f"{label}: zu wenige Ereignisse für stabilen AUC ({n_events})")
# ohne Diabetes: n = 79, Ereignisse = 11, AUC = 0.902
# mit Diabetes:  n = 21, Ereignisse = 5,  AUC = 0.800
```

**Interpretation:** Auf diesem Split liegt die Subgruppen-AUC bei 0,902 (ohne
Diabetes) vs. 0,800 (mit Diabetes), eine Differenz von rund 0,10. Bei nur 5
Ereignissen in der Diabetes-Gruppe ist das kein belastbarer Befund (die
Konfidenzintervalle beider Werte überlappen sich vermutlich stark), aber es
illustriert das Prinzip: **Ein globaler AUC (hier 0,880) kann eine schlechtere
Diskriminierung in einer Subgruppe komplett verdecken.** Größere reale
Diskrepanzen (z. B. 0,85 vs. 0,60) haben typischerweise dieselben Ursachen:
weniger Trainingsbeispiele für diese Gruppe, systematisch andere klinische
Präsentation, oder weniger informative Prädiktoren in dieser Subgruppe.
Subgruppen-AUC ist deshalb eine **ethische Mindestprüfung** vor jedem
klinischen Einsatz, nicht ein optionales Extra, auch wenn (wie hier) die
Stichprobe für eine sichere Aussage zu klein ist.
