# Musterlösungen · Bewertung der Modellgüte und klinische Validierung

## Aufgabe 1 – AUC-Vergleich bei Klassenungleichheit
```python
from sklearn.metrics import roc_auc_score, average_precision_score

auc_roc = roc_auc_score(y_test, proba)
auc_pr  = average_precision_score(y_test, proba)
print(f"ROC-AUC: {auc_roc:.3f}  PR-AUC: {auc_pr:.3f}  Baseline PR: {y_test.mean():.3f}")
# ROC-AUC: 0.844  PR-AUC: 0.624  Baseline PR: 0.152
```
Die PR-AUC-Baseline ist der Ereignisanteil (~0,15). Ein Zufallsmodell erreicht bei
ROC-AUC dagegen immer 0,50. Die ROC-AUC hängt nicht von der Ereignisrate ab und
kann bei 15 % Ereignissen hoch wirken, obwohl viele Alarme Fehlalarme sind; die
PR-AUC zeigt, ob das Modell Hochrisikopatient:innen mit brauchbarer Präzision
findet. Hier bestätigt PR-AUC 0,624 (deutlich über
der Baseline 0,152), dass das Modell tatsächlich informativ ist.

Beide Werte gehören mit Konfidenzintervall berichtet. Auf n = 125 mit 19
Ereignissen liefert der Bootstrap ROC-AUC 0,844 (95 %-KI 0,707–0,946) und
PR-AUC 0,624 (95 %-KI 0,398–0,847) – die Intervalle sind so breit, dass eine
Punktschätzung allein Präzision vortäuscht.

## Aufgabe 2 – Erst rekalibrieren, dann Kalibrierung prüfen
```python
from sklearn.calibration import CalibratedClassifierCV

print(f"proba (roh):          mean={proba.mean():.3f}  vs. beobachtet={y_test.mean():.3f}")
calibrated = CalibratedClassifierCV(build_pipeline(), method="sigmoid", cv=5)
calibrated.fit(X_train, y_train)
proba_cal = calibrated.predict_proba(X_test)[:, 1]
print(f"proba_cal (kalibriert): mean={proba_cal.mean():.3f}  vs. beobachtet={y_test.mean():.3f}")
# proba (roh):            mean=0.409  vs. beobachtet=0.152
# proba_cal (kalibriert): mean=0.162  vs. beobachtet=0.152
```
`proba` liegt im Mittel bei 41 %, mehr als das Doppelte der beobachteten 15 %,
`class_weight="balanced"` verzerrt die absoluten Werte systematisch nach oben.
`proba_cal` liegt nahe an der beobachteten Rate. Für jede Kalibrierungskurve,
jeden Brier Score und jede DCA verwenden wir deshalb `proba_cal`, nicht `proba`.

## Aufgabe 3 – Kalibrierungskurve interpretieren
```python
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

frac_pos, mean_pred = calibration_curve(y_test, proba_cal, n_bins=10)
brier = brier_score_loss(y_test, proba_cal)
print(f"Brier Score: {brier:.4f}")
# Brier Score: 0.0955
```
Punkte oberhalb der Diagonalen: Modell unterschätzt das Risiko (beobachtet mehr
Ereignisse als vorhergesagt). Punkte unterhalb: Überschätzung. Das Null-Modell
(immer mittlere Ereignisrate ≈ 0,152) hat Brier Score ≈ p*(1-p) ≈ 0,129. Der
rekalibrierte Wert (0,0955) liegt klar darunter, besser als das Nullmodell. Bei nur
~19 Testevents in 10 Bins schwankt die Kurve zwischen den Bins deutlich, das ist
Stichprobenrauschen, kein Modellfehler.

## Aufgabe 4 – Net Benefit manuell
```python
import numpy as np

def net_benefit(y_true, proba, threshold):
    n = len(y_true)
    pos = proba >= threshold
    tp = int(((pos == 1) & (y_true == 1)).sum())
    fp = int(((pos == 1) & (y_true == 0)).sum())
    return tp / n - (threshold / (1 - threshold)) * fp / n

base = y_test.mean()
for t in [0.05, 0.10, 0.20, 0.30]:
    nb_m   = net_benefit(y_test.values, proba_cal, t)
    nb_all = base - (t / (1 - t)) * (1 - base)
    print(f"t={t:.2f}: Modell={nb_m:.4f} Alle={nb_all:.4f}")
# t=0.05: Modell=0.1120 Alle=0.1074
# t=0.10: Modell=0.0791 Alle=0.0578
# t=0.20: Modell=0.0780 Alle=-0.0600
# t=0.30: Modell=0.0594 Alle=-0.2114
```
Das Modell ist nützlich, solange sein Net Benefit *über* beiden Referenzlinien
liegt. Hier gilt das für alle vier gezeigten Schwellen, „alle behandeln“ fällt
bereits ab Schwelle ≈0,15 unter „niemanden behandeln“ (negativer Net Benefit),
während das Modell durchgehend positiv bleibt.

## Aufgabe 5 – Bootstrap-Optimismus
```python
auc_app, optimism, auc_corr = bootstrap_optimism(X_train, y_train,
                                                  build_pipeline(), n_boot=100)
print(f"Apparente AUC: {auc_app:.3f}  Optimismus: {optimism:.3f}  Korrigiert: {auc_corr:.3f}")
# Apparente AUC: 0.814  Optimismus: 0.051  Korrigiert: 0.762
```
Der Optimismus ist groß, wenn das Modell viele Freiheitsgrade hat (viele
Prädiktoren relativ zur Fallzahl) oder keine / schwache Regularisierung.
Logistische Regression mit Ridge-Penalty (C klein) hat geringen Optimismus.
Bootstrap-Optimismus betrifft die AUC (Ranking) und ist unabhängig davon, ob
später noch rekalibriert wird.

## Bonus – Sigmoid vs. isotonische Rekalibrierung
```python
from sklearn.calibration import CalibratedClassifierCV

cal_sigmoid   = CalibratedClassifierCV(build_pipeline(), method="sigmoid", cv=5).fit(X_train, y_train)
cal_isotonic  = CalibratedClassifierCV(build_pipeline(), method="isotonic", cv=5).fit(X_train, y_train)
p_sig = cal_sigmoid.predict_proba(X_test)[:, 1]
p_iso = cal_isotonic.predict_proba(X_test)[:, 1]
print(f"Brier sigmoid:   {brier_score_loss(y_test, p_sig):.4f}")
print(f"Brier isotonic:  {brier_score_loss(y_test, p_iso):.4f}")
# Brier sigmoid:   0.0955
# Brier isotonic:  0.0906
```
Auf diesem einzelnen Split schneidet isotonic sogar minimal besser ab, das ist
aber kein verlässliches Muster: Das Trainingsset hat nur 59 Ereignisse von 375
Zeilen, aufgeteilt auf 5 CV-Folds sind das ~12 Ereignisse pro Fold, zu wenig,
um einer flexiblen Treppenfunktion (isotonic) stabil zu vertrauen. Platt-Scaling
(sigmoid, logistisch) hat nur zwei Parameter und schwankt bei einem anderen
Split/Seed typischerweise weniger. Bei so kleinen Kohorten ist sigmoid deshalb
die robustere Standardwahl, auch wenn isotonic in einer einzelnen Stichprobe
zufällig vorn liegen kann; isotonic lohnt sich eher ab deutlich mehr Ereignissen.
