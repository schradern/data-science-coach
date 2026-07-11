# Musterlösungen · Maschinelles Lernen für Überlebenszeiten

## Aufgabe 1 – Zensierung verstehen
Ein **zensierter** Patient ist eine Person, die das Beobachtungsfenster (30 Tage)
verlassen hat, **ohne** das Zielereignis (Tod) erlebt zu haben, zum Beispiel
weil sie vorher entlassen wurde. Wir wissen nur: „Bis zum Tag der Entlassung war
diese Person noch am Leben." Ein Klassifikator würde den Label `0` vergeben und
annehmen, die Person sei **nie** gestorben, das ist korrekt für den Beobachtungszeitraum,
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
Gut getrennte Kurven bei niedrigem p-Wert bestätigen, dass das Modell klinisch
relevante Risikogruppen identifiziert, nicht nur statistische Artefakte.

## Aufgabe 4 – RSF (sksurv)
```python
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import cumulative_dynamic_auc

rsf = RandomSurvivalForest(n_estimators=200, min_samples_leaf=10, random_state=SEED)
rsf.fit(X_tr_np, surv_train)
chf = rsf.predict_cumulative_hazard_function(X_te_np, return_array=False)
rsf_scores = np.array([fn(28) for fn in chf])

# Pass the FULL test array + full score array in one call — cumulative_dynamic_auc
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
`assets/rsf_vs_cox.png`). Das reicht aber nicht für die Aussage „Cox ist besser":
`code/python.py` bootstrappt den Testsatz und berichtet ein 95%-KI auf die
**Differenz** Cox − RSF. Bei Tag 14, 21 und 28 enthält dieses KI die Null (z. B.
Tag 28: +0,02 [−0,02, +0,06]) — der Unterschied ist dort nicht von Rauschen zu
trennen. Nur bei Tag 7 (die wenigsten Ereignisse) liegt das KI knapp über null,
und dieser Bootstrap unterschätzt die Unsicherheit sogar noch, weil er die
Modelle festhält. Belastbare Aussage: **kein Modell ist hier klar überlegen.**
RSFs Vorteil gegenüber Cox entfaltet sich vor allem bei **nichtlinearen** oder
**nicht-proportionalen** Zusammenhängen und großen Stichproben; bei kleinen,
eher linear-proportionalen Kohorten wie dieser ist ein gut regularisiertes Cox-
Modell oft mindestens ebenbürtig. Der Lerneffekt: RSF ist kein Selbstläufer, und
eine AUC-Differenz ohne Konfidenzintervall ist keine Rangfolge.

## Bonus – Konkurrierende Risiken konzeptuell
Wenn verlegte Patient:innen als `verstorben_30d = 0` (Nicht-Ereignis) codiert
werden, unterschätzt das Modell das wahre Risiko, diese Personen sind de facto
in Beobachtungsabbruch (informative Zensierung), und der Klassifikator lernt
„alle Verlegten überleben", was falsch ist. Die korrekte Strategie: Verlegung als
**konkurrierendes Ereignis** (Status 2) definieren und den
**Aalen-Johansen-Schätzer** (`lifelines.AalenJohansenFitter`) oder das
Fine-Gray-Modell verwenden, das die subdistribution hazard modelliert.
