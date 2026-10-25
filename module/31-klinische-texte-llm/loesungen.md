# Musterlösungen · Verarbeitung klinischer Freitexte mit LLMs

## Aufgabe 1 – Pipeline-Leakage demonstrieren
```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

SEED = 42
df = pd.read_csv("{{DATEN}}notizen.csv")
texts, y = df["notiz"], df["verschlechterung"]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# WRONG: vectorizer fitted on the full dataset -> IDF leakage
X_leaky = TfidfVectorizer(ngram_range=(1, 2), min_df=2).fit_transform(texts)
auc_leaky = cross_val_score(
    LogisticRegression(max_iter=1000, class_weight="balanced"),
    X_leaky, y, cv=cv, scoring="roc_auc"
).mean()

# CORRECT: vectorizer inside the pipeline
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
auc_honest = cross_val_score(pipe, texts, y, cv=cv, scoring="roc_auc").mean()

print(f"Leaky AUC:  {auc_leaky:.3f}")
print(f"Honest AUC: {auc_honest:.3f}")
```
Auf diesem Datensatz ist der Unterschied praktisch null: Beide Varianten liegen
bei ≈ 0,82, die leckende sogar minimal darunter. Das ist kleiner als bei der
Merkmalsauswahl (Modul 24), weil IDF-Gewichte relativ stabil sind. Dennoch: Bei
kleinen Korpora oder seltenen Termen kann der Vectorizer Testdokumente lernen,
das ist konzeptuell Leakage, auch wenn es hier zufällig nichts kostet.

## Aufgabe 2 – N-Gramme und Stoppwörter
```python
pipe_uni = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 1), min_df=2)),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
auc_uni = cross_val_score(pipe_uni, texts, y, cv=cv, scoring="roc_auc").mean()
print(f"Unigramm-AUC: {auc_uni:.3f}")
```
Auf diesem (synthetischen) Datensatz bleibt die CV-AUC mit reinen Unigrammen
überraschenderweise mindestens gleich gut, teils sogar leicht höher, als mit
Bigrammen, ein Zeichen dafür, dass die einzelnen Wörter hier bereits stark
prädiktiv sind. Das ist kein Widerspruch zur allgemeinen Regel: Bigramme wie
„Laktat steigend“ oder „zunehmende Dyspnoe“ kodieren Wortkombinationen, die
Unigramme grundsätzlich nicht abbilden können, bei größeren, unaufgeräumteren
Korpora mit mehr Kontextabhängigkeit überwiegt dieser Vorteil meist. Vergleiche
immer die tatsächliche CV-AUC statt eine Faustregel blind zu übernehmen.

**b) Stoppwörter**

```python
GERMAN_STOP = ["der", "die", "das", "den", "des", "dem", "und", "oder", "mit",
               "bei", "im", "in", "am", "an", "auf", "zu", "zur", "zum", "von",
               "vom", "ein", "eine", "einer", "ist", "seit", "nach", "über", "durch"]
pipe_stop = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, stop_words=GERMAN_STOP)),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
auc_stop = cross_val_score(pipe_stop, texts, y, cv=cv, scoring="roc_auc").mean()
print(f"AUC mit Stoppwortliste: {auc_stop:.3f}")
```
Die CV-AUC ändert sich kaum (0,822 statt 0,819). TF-IDF gewichtet Wörter, die in
fast jeder Notiz stehen, ohnehin herunter; Artikel tragen hier wenig Signal.
Eine Stoppwortliste macht das Vokabular kleiner und die Top-Tokens lesbarer,
nicht das Modell besser. Negationen gehören nie auf die Liste: Aus „keine
Dyspnoe“ würde „Dyspnoe“, und das Modell läse einen Normalbefund als
Warnzeichen. Fertige Stoppwortlisten anderer Bibliotheken enthalten `kein` und
`nicht` oft, deshalb vor dem Einsatz immer durchsehen.

## Aufgabe 3 – Tokens interpretieren
```python
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

X_tr, X_te, y_tr, y_te = train_test_split(texts, y, test_size=0.25, stratify=y, random_state=42)
pipe.fit(X_tr, y_tr)
names = pipe.named_steps["tfidf"].get_feature_names_out()
coef  = pipe.named_steps["model"].coef_[0]

top_pos = pd.DataFrame({"token": names, "k": coef}).nlargest(10, "k")
top_neg = pd.DataFrame({"token": names, "k": coef}).nsmallest(10, "k")
print(top_pos); print(top_neg)
```
Klinisch plausibel: „zunehmend“, „katecholaminpflichtig“ und „dyspnoe“ auf der
Positiv-Seite; „stabil“, „rekonvaleszenz“ und „afebril“ auf der Negativ-Seite.
Überraschend: „zunehmend“ und „zunehmende“ tauchen als getrennte Tokens auf,
Stemming (README, Abschnitt 1) würde beide zusammenführen. Bei echten,
unbereinigten Freitexten wäre außerdem mit Confoundern zu rechnen, etwa
Wörtern, die gehäuft bei bestimmten Ärzt:innen oder Diagnosekategorien
auftauchen, ohne ein echtes Verschlechterungs-Signal zu sein.

## Aufgabe 4 – Schwelle klinisch wählen
```python
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve

# Choose the threshold on the training set only (cross-validated predictions),
# then evaluate it once on the untouched test set.
proba_cv = cross_val_predict(pipe, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
fpr, tpr, thr = roc_curve(y_tr, proba_cv)
threshold = thr[np.where(tpr >= 0.80)[0][0]]

pipe.fit(X_tr, y_tr)
pred = pipe.predict_proba(X_te)[:, 1] >= threshold
y_true = np.asarray(y_te)
sens = (pred & (y_true == 1)).sum() / (y_true == 1).sum()
spec = (~pred & (y_true == 0)).sum() / (y_true == 0).sum()
print(f"Schwelle (aus Training): {threshold:.2f}  Test: Sensitivität {sens:.2f}  Spezifität {spec:.2f}")
```
Die Schwelle wird nur auf den Trainingsdaten gewählt (kreuzvalidierte
Vorhersagen), dann genau einmal am Testset geprüft. Wer sie direkt auf dem
Testset sucht, wählt auf denselben Daten, mit denen er bewertet, und überschätzt
die Güte (Modul 24).

Auf diesen Daten liefert das eine Schwelle von 0,43. Auf dem Testset erreicht sie
eine Sensitivität von 0,91 und eine Spezifität von 0,72: Etwa jeder vierte stabile
Verlauf löst einen Fehlalarm aus. Bei 43 Verschlechterungen im Testset sind
beide Werte noch unsicher (grob ±0,1).

Gerechtfertigt ist das, wenn ein übersehener Verschlechterungsfall schwere
Konsequenzen hat (z. B. eine verspätete Intensivverlegung) und ein
falsch-positiver Alarm nur eine kurze klinische Überprüfung kostet.

## Bonus – Regularisierung
```python
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(
    Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ]),
    {"model__C": [0.01, 0.1, 1, 10]},
    cv=cv, scoring="roc_auc"
)
grid.fit(texts, y)
print(grid.best_params_, grid.best_score_)
```
Auf diesem Datensatz liefert `C=1` die beste CV-AUC (≈ 0,82); `C=0,01` und
`C=0,1` schneiden spürbar schlechter ab. Starke Regularisierung (kleines C)
schrumpft alle Koeffizienten gegen null, seltene Tokens mit hohem
Einzelkoeffizient werden stärker bestraft als häufige. Das kann Überanpassung
auf seltene Artefakte verhindern, kostet aber auch echtes Signal, hier zeigt
sich, dass das Signal in den Tokens robust genug ist, dass wenig Regularisierung
optimal bleibt.
