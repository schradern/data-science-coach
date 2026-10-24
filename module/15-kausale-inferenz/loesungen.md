# Musterlösungen · Kausale Inferenz und Directed Acyclic Graphs

## Aufgabe 1 – DAG

Plausibler DAG:

```text
Alter → Diabetes
Alter → Mortalität
Alter → SOFA
Aufnahmegrund → SOFA
Aufnahmegrund → Mortalität
Diabetes → SOFA → Mortalität
Diabetes → Mortalität
```

Alter ist der Confounder: Es verursacht Diabetes und Mortalität. Der
Aufnahmegrund ist in diesem DAG **kein** Confounder des Diabetes-Effekts, weil
kein Pfeil von ihm zu Diabetes führt; er wirkt auf SOFA und Mortalität. SOFA ist
ein Mediator (Diabetes → SOFA → Mortalität). Ob eine Variable Confounder oder
Mediator ist, entscheidet ihre zeitliche und kausale Stellung, nicht die
Fragestellung: Ein vor dem Diabetes gemessener Schweregrad könnte Confounder
sein, der SOFA bei Aufnahme ist es hier nicht.

Nicht blind adjustieren sollte man für SOFA, und zwar aus zwei Gründen: Es
blockiert den vermittelten Teil des Effekts, und es ist zugleich ein Kollider
auf dem Pfad Diabetes → SOFA ← Aufnahmegrund → Mortalität. Wer für SOFA
adjustiert, öffnet diesen Pfad, solange der Aufnahmegrund nicht mit im Modell
steht.

## Aufgabe 2 – Crude vs. adjustiert

```python
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

df = pd.read_csv("{{DATEN}}kohorte.csv")
crude = smf.logit("verstorben_30d ~ diabetes", data=df).fit(disp=False)
adj = smf.logit("verstorben_30d ~ diabetes + alter", data=df).fit(disp=False)
adj_sofa = smf.logit("verstorben_30d ~ diabetes + alter + sofa_score", data=df).fit(disp=False)

for name, fit in [("crude", crude), ("adjusted (alter)", adj), ("adjusted (alter+sofa)", adj_sofa)]:
    ci = fit.conf_int().loc["diabetes"]
    print(name, np.exp(fit.params["diabetes"]), np.exp(ci).tolist())
```

Ergebnis auf den Kursdaten: crude OR ≈ 2,07, adjustiert nur für den Confounder
Alter ≈ 1,59, zusätzlich für SOFA adjustiert ≈ 0,99 (praktisch kein Effekt
mehr). Die ersten beiden Zahlen schätzen den **totalen** Effekt von Diabetes
auf die Mortalität (Alter ist laut DAG ein Confounder, SOFA nicht im Modell).
Die dritte Zahl schätzt stattdessen den **direkten** Effekt, der nicht über
SOFA vermittelt wird, denn SOFA liegt laut DAG (Aufgabe 1) auf dem kausalen
Pfad Diabetes → SOFA → Mortalität. Auch als direkter Effekt ist die 0,99 nur
belastbar, wenn der Aufnahmegrund mit im Modell steht (Kollider-Pfad aus
Aufgabe 1); das 95-%-KI [0,52; 1,87] ist ohnehin zu breit für eine Aussage. Für
die Frage „Wie stark beeinflusst Diabetes insgesamt die Mortalität?“ ist die
alter-adjustierte Zahl (≈ 1,59) die richtige Antwort, nicht die SOFA-adjustierte.

## Aufgabe 3 – Propensity Score

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

# Only confounders go into the PS model: alter. Per the DAG, sofa_score is a mediator
# on the diabetes path and does NOT belong in it (task 1).
X = df[["alter"]]
y = df["diabetes"].astype(int)
ps_model = LogisticRegression(max_iter=1000).fit(X, y)
df["ps"] = ps_model.predict_proba(X)[:, 1]
df["iptw"] = np.where(y.eq(1), 1 / df["ps"], 1 / (1 - df["ps"]))
print(df[["diabetes", "ps", "iptw"]].describe())
```

Sehr große Gewichte sind Warnzeichen für fehlende Positivität.

## Aufgabe 4 – Balance

```python
import numpy as np

def smd(x, group, weights=None):
    x = np.asarray(x)
    g = np.asarray(group).astype(bool)
    if weights is None:
        m1, m0 = x[g].mean(), x[~g].mean()
        v1, v0 = x[g].var(ddof=1), x[~g].var(ddof=1)
    else:
        w = np.asarray(weights)
        m1 = np.average(x[g], weights=w[g])
        m0 = np.average(x[~g], weights=w[~g])
        v1 = np.average((x[g] - m1) ** 2, weights=w[g])
        v0 = np.average((x[~g] - m0) ** 2, weights=w[~g])
    return (m1 - m0) / np.sqrt((v1 + v0) / 2)

for col in ["alter", "sofa_score"]:
    print(col, "raw", smd(df[col], df["diabetes"]),
          "weighted", smd(df[col], df["diabetes"], df["iptw"]))
```

Auf den Kursdaten: `alter` raw SMD ≈ 0,58 → nach IPTW ≈ 0,13, also deutlich
näher an Balance, wie erwartet, da `alter` das PS-Modell bestimmt hat. Die
Zielmarke 0,1 ist damit aber noch nicht erreicht; ein flexibleres PS-Modell
(Alter als Spline) wäre der nächste Schritt.
`sofa_score` bleibt bei roher SMD ≈ 0,54 nahezu unverändert (es fließt nicht
ins PS-Modell ein). Das ist **kein Fehler**: `sofa_score` ist ein Mediator,
kein Confounder, Balance dort ist kein sinnvolles Ziel für diese PS-Analyse,
und ein Versuch, es zu balancieren, würde den Diabetes-Effekt verzerren (vgl.
Aufgabe 2). Ein Zielwert SMD < 0,1 ist ein häufiger pragmatischer Standard,
aber nur für tatsächliche Confounder.

## Aufgabe 5 – Interpretation

Beispiel:

> Die Analyse schätzt den Diabetes-Mortalitäts-Zusammenhang unter Adjustierung
> für gemessene Confounder. Da es sich um Beobachtungsdaten handelt und
> ungemessene Confounder möglich bleiben, ist die Schätzung als annahmenabhängig
> und nicht als kausaler Beweis zu interpretieren.
