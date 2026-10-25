# Musterlösungen · Maschinelles Lernen für Überlebenszeiten

**Vorbereitung.** Die Lösungen setzen auf den Bausteinen aus [`code/python.py`](code/python.py) auf. Führe diesen Block einmal im Projektordner aus, lokal oder in Colab nach dem Klonen ([in Colab öffnen](https://colab.research.google.com/github/schradern/data-science-coach/blob/main/notebooks/start.ipynb)). Danach laufen die Blöcke dieser Seite der Reihe nach. `runpy` lädt nur die Definitionen des Skripts, `main()` läuft dabei nicht.

```python
# Setup: run once from the project folder.
import runpy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

lesson = runpy.run_path("module/28-survival-ml/code/python.py")
SEED, FEATURES, build_data, impute_and_scale, make_survival_array = (
    lesson[name] for name in ("SEED", "FEATURES", "build_data", "impute_and_scale",
                              "make_survival_array"))
X, events, times = build_data()
(X_train, X_test, events_train, events_test,
 times_train, times_test) = train_test_split(
    X, events, times, test_size=0.25, stratify=events, random_state=SEED)
X_tr_np, X_te_np = impute_and_scale(X_train, X_test)
surv_train = make_survival_array(events_train, times_train)
surv_test = make_survival_array(events_test, times_test)
```

## Aufgabe 1 – Zensierung verstehen
Ein **zensierter** Patient ist eine Person, die das Beobachtungsfenster (30 Tage)
verlassen hat, **ohne** das Zielereignis (Tod) erlebt zu haben, in diesem
Datensatz entweder weil sie vorher aus der Nachbeobachtung gefallen ist oder
weil Tag 30 erreicht war. Wir wissen nur: „Bis zum
letzten Kontakt war diese Person noch am Leben.“ Ein Klassifikator würde das
Label `0` vergeben und annehmen, die Person sei **nie** gestorben, das ist korrekt für den Beobachtungszeitraum,
aber systematisch falsch, wenn die Beobachtungszeit kürzer als das Vorhersagefenster
ist, da das Modell dann falsche Nullen als Grundgesamtheit lernt.

## Aufgabe 2 – Cox als Baseline
```python
from lifelines import CoxPHFitter
import pandas as pd

train_df = pd.DataFrame(X_tr_np, columns=FEATURES)
train_df["fu_zeit_tage"] = times_train
train_df["status"] = events_train.astype(int)

cph = CoxPHFitter(penalizer=0.1)
cph.fit(train_df, duration_col="fu_zeit_tage", event_col="status")

test_df = pd.DataFrame(X_te_np, columns=FEATURES)
scores = cph.predict_partial_hazard(test_df).values

# Concordance = C-statistic (lifelines built-in)
from lifelines.utils import concordance_index
c = concordance_index(times_test, -scores, events_test)
print(f"Cox C-Statistik: {c:.3f}")
```
Auf dem Testset ergibt dieser Lauf eine Cox-Konkordanz von ≈0,86. Die
Konkordanz (C-Statistik) misst, wie oft das Modell bei zwei zufällig
gewählten Patient:innen die richtige Reihenfolge (wer stirbt früher?) vorhersagt.
Die Standard-AUC bei binärer Klassifikation misst Rang-Diskriminierung zu einem
**fixen Zeitpunkt**, sie ignoriert die Eventzeit.

## Aufgabe 3 – Risikogruppen und KM
```python
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

tertile_low  = np.percentile(scores, 33)
tertile_high = np.percentile(scores, 67)

groups = {
    "Niedrig": scores < tertile_low,
    "Mittel":  (scores >= tertile_low) & (scores < tertile_high),
    "Hoch":    scores >= tertile_high,
}
fig, ax = plt.subplots(figsize=(7, 4))
for label, mask in groups.items():
    kmf = KaplanMeierFitter()
    kmf.fit(times_test[mask], events_test[mask], label=label)
    kmf.plot_survival_function(ax=ax)

# Log-rank test (Niedrig vs. Hoch)
lr = logrank_test(times_test[groups["Niedrig"]], times_test[groups["Hoch"]],
                  event_observed_A=events_test[groups["Niedrig"]],
                  event_observed_B=events_test[groups["Hoch"]])
print(f"Log-Rank p-Wert (Niedrig vs. Hoch): {lr.p_value:.4f}")
```
Gut getrennte Kurven bei niedrigem p-Wert zeigen, dass der Score Patient:innen
nach Risiko ordnet (auf diesem Split: 0 Todesfälle im unteren, 16 im oberen
Tertil, Log-Rank p < 0,001). Ob die Gruppen klinisch relevant sind, hängt von den
absoluten Risiken und der Entscheidung ab, die daraus folgen soll.

## Aufgabe 4 – RSF (sksurv)
```python
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import cumulative_dynamic_auc

rsf = RandomSurvivalForest(n_estimators=200, min_samples_leaf=10, random_state=SEED)
rsf.fit(X_tr_np, surv_train)
chf = rsf.predict_cumulative_hazard_function(X_te_np, return_array=False)
rsf_scores = np.array([fn(28) for fn in chf])

# Pass the FULL test array + full score array in one call – cumulative_dynamic_auc
# builds the per-time risk set internally. Masking surv_test/rsf_scores by
# `time >= t` first (a common mistake) silently corrupts the AUC because the
# function needs to see who is still at risk, not just who survives past t.
auc, _ = cumulative_dynamic_auc(surv_train, surv_test, rsf_scores, [7, 21])
for t, a in zip([7, 21], auc):
    print(f"RSF AUC @ t={t}d: {a:.3f}")
```
Auf diesem Datensatz (500 Patient:innen, 78 Ereignisse; Testsatz n = 125 mit 19
Ereignissen) liegt die RSF-AUC bei Tag 7 bei ≈0,84 und bei Tag 21 bei ≈0,85, die
Cox-Baseline mit ≈0,92 bzw. ≈0,88 als **Punktschätzer** leicht darüber (siehe
`assets/rsf_vs_cox.png`). Das reicht aber nicht für die Aussage „Cox ist besser“:
`code/python.py` bootstrappt den Testsatz und berichtet ein 95%-KI auf die
**Differenz** Cox − RSF. Bei Tag 14, 21 und 28 enthält dieses KI die Null (z. B.
Tag 28: +0,02 [−0,02, +0,06]) – der Unterschied ist dort nicht von Rauschen zu
trennen. Nur bei Tag 7 (die wenigsten Ereignisse) liegt das KI knapp über null,
und dieser Bootstrap unterschätzt die Unsicherheit sogar noch, weil er die
Modelle festhält. Belastbare Aussage: **kein Modell ist hier klar überlegen.**
RSFs Vorteil gegenüber Cox entfaltet sich vor allem bei **nichtlinearen** oder
**nicht-proportionalen** Zusammenhängen und großen Stichproben; bei kleinen,
eher linear-proportionalen Kohorten wie dieser ist ein gut regularisiertes Cox-
Modell oft mindestens ebenbürtig. Der Lerneffekt: RSF ist kein Selbstläufer, und
eine AUC-Differenz ohne Konfidenzintervall ist keine Rangfolge.

## Bonus – Verlegung: Zensierung oder konkurrierendes Ereignis?
Wenn verlegte Patient:innen als Nicht-Ereignis kodiert werden (als hätten sie
bis Tag 30 sicher überlebt), überschätzt die KM-Kurve das Überleben, also
unterschätzt sie das Sterberisiko: Die Todesfälle nach der Verlegung fehlen. Auch
das Zensieren am Verlegungstag hilft nur bedingt, denn diese Zensierung ist
informativ (Verlegte sind oft kränker), KM setzt aber nicht-informative
Zensierung voraus. Die korrekte Strategie hängt vom Estimand ab: Zählt Tod
überall, braucht man die fehlende Nachbeobachtung (z. B. Abgleich mit
Melderegister oder aufnehmender Klinik) oder Methoden für informative Zensierung
(z. B. inverse Zensierungsgewichte). Zählt nur der Tod in dieser Klinik, ist die
Verlegung ein **konkurrierendes Ereignis** (Status 2); dann schätzt man die
kumulative Inzidenz mit dem **Aalen-Johansen-Schätzer**
(`lifelines.AalenJohansenFitter`) oder modelliert sie mit Fine-Gray
(subdistribution hazard).
