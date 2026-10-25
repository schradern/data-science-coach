# Musterlösungen · Modelleinsatz, Monitoring und Governance

**Vorbereitung.** Die Lösungen setzen auf den Bausteinen aus [`code/python.py`](code/python.py) auf. Führe diesen Block einmal im Projektordner aus, lokal oder in Colab nach dem Klonen ([in Colab öffnen](https://colab.research.google.com/github/schradern/data-science-coach/blob/main/notebooks/start.ipynb)). Danach laufen die Blöcke dieser Seite der Reihe nach. `runpy` lädt nur die Definitionen des Skripts, `main()` läuft dabei nicht.

```python
# Setup: run once from the project folder.
import runpy
from sklearn.model_selection import train_test_split

lesson = runpy.run_path("module/32-einsatz-governance/code/python.py")
SEED = lesson["SEED"]
X_train, X_test, y_train, y_test = lesson["prepare_data"]()
pipe = lesson["build_text_pipeline"]().fit(X_train, y_train)
```

## Aufgabe 1 – Modell speichern und laden
```python
import joblib, sklearn, tempfile
from pathlib import Path
from sklearn.metrics import roc_auc_score

# pipe, X_train, X_test, y_train, y_test as in code/python.py
# (build_text_pipeline() and prepare_data())
pipe.fit(X_train, y_train)

# Save
artefact = {
    "pipeline":        pipe,
    "sklearn_version": sklearn.__version__,
    "target":          "verschlechterung",
    "train_size":      len(X_train),
    "test_auc":        roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1]),
}
with tempfile.TemporaryDirectory() as d:
    p = Path(d) / "modell.joblib"
    joblib.dump(artefact, p)
    loaded = joblib.load(p)

# Verify identical predictions
import numpy as np
assert np.allclose(
    pipe.predict_proba(X_test),
    loaded["pipeline"].predict_proba(X_test)
), "Predictions differ after reload!"
print("OK – identische Vorhersagen nach Reload.")
```
Zusätzlich zum Pipeline-Objekt empfiehlt sich: sklearn-Version, Datum,
Trainingsgröße, Test-AUC, SHA-256-Prüfsumme der Trainingsdaten.

## Aufgabe 2 – Prävalenz-Shift: ROC-AUC vs. PR-AUC
```python
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

rng = np.random.default_rng(42)
for sf in [0.0, 0.5, 1.0]:
    n_extra = int(sf * (y_test == 1).sum())
    pos_idx = y_test[y_test == 1].index
    extra_idx = rng.choice(pos_idx, size=n_extra, replace=True) if n_extra > 0 else []
    Xs = pd.concat([X_test] + ([X_test.loc[extra_idx]] if n_extra > 0 else []))
    ys = pd.concat([y_test] + ([y_test.loc[extra_idx]] if n_extra > 0 else []))
    proba = pipe.predict_proba(Xs)[:, 1]
    roc = roc_auc_score(ys, proba)
    pr = average_precision_score(ys, proba)
    print(f"Shift {sf:.0%}: n={len(ys)}  Positivrate={ys.mean():.1%}  "
          f"ROC-AUC={roc:.3f}  PR-AUC={pr:.3f}")
```
Tatsächlicher Lauf: Shift 0 % → ROC-AUC 0,849 / PR-AUC 0,672 (Positivrate 34,4 %);
Shift 50 % → ROC-AUC 0,835 / PR-AUC 0,734 (43,8 %); Shift 100 % → ROC-AUC 0,850 /
PR-AUC 0,820 (51,2 %).

**ROC-AUC bleibt näherungsweise konstant** (0,849 → 0,835 → 0,850, Schwankung ist
Rauschen, kein Trend), weil ROC-AUC nur von der **Rangordnung** der Scores
innerhalb jeder Klasse abhängt. Duplizieren bereits vorhandener positiver Fälle
fügt keine neuen Score-Werte hinzu, jedes Duplikat hat exakt denselben Score wie
sein Original, also verschiebt es die Rangordnung zwischen Positiv- und
Negativ-Scores nicht. **PR-AUC steigt dagegen deutlich** (0,672 → 0,820), weil
Precision (`TP / (TP + FP)`) direkt von der Prävalenz abhängt: bei mehr Positiven
im Nenner-relevanten Bereich sinkt der Anteil falsch-positiver Fälle relativ zu
den (jetzt häufigeren) richtig-positiven, die Precision-Recall-Kurve verbessert
sich rein durch die veränderte Klassenbalance. Das ist die Kernlektion: **ROC-AUC
ist prävalenzunabhängig, PR-AUC nicht.** PR-AUC beschreibt, wie gut die Alarme
bei *genau dieser* Prävalenz zutreffen; sie ist bei seltenen Ereignissen
informativ, lässt sich aber nur zusammen mit der Prävalenz lesen und muss bei
einem Prävalenz-Shift in der Zielpopulation neu bestimmt werden.

## Aufgabe 3 – Subgruppenanalyse mit Konfidenzintervall
```python
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

def bootstrap_auc_ci(y, proba, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    y, proba = np.asarray(y), np.asarray(proba)
    boot_aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(y[idx], proba[idx]))
    return np.percentile(boot_aucs, [2.5, 97.5])

cohort = pd.read_csv("{{DATEN}}kohorte.csv")
cohort["geschlecht_clean"] = cohort["geschlecht"].replace({"w": "weiblich"})
merged = pd.read_csv("{{DATEN}}notizen.csv").merge(cohort[["patient_id", "geschlecht_clean"]], on="patient_id")
_, test_idx = train_test_split(merged.index, test_size=0.25,
                               stratify=merged["verschlechterung"], random_state=42)
test = merged.loc[test_idx]
proba = pipe.predict_proba(test["notiz"])[:, 1]

for g in ["weiblich", "maennlich"]:
    mask = (test["geschlecht_clean"] == g).values
    y_g, p_g = test.loc[mask, "verschlechterung"].to_numpy(), proba[mask]
    auc = roc_auc_score(y_g, p_g)
    lo, hi = bootstrap_auc_ci(y_g, p_g)
    print(f"{g}: AUC = {auc:.3f}  [95%-CI {lo:.3f}-{hi:.3f}]  (n={mask.sum()})")
```
Tatsächlicher Lauf: Weiblich AUC 0,880 [95%-CI 0,787–0,952] (n=61); Männlich
AUC 0,807 [95%-CI 0,689–0,909] (n=64). Maßgeblich ist das Konfidenzintervall
der Differenz (`code/python.py`): +0,072 [−0,064; +0,212]. Es schließt 0 ein,
die Lücke ist also bei diesen Stichprobengrößen **nicht von Zufall zu
unterscheiden**. Umgekehrt ist auch eine Lücke von 0,2 AUC-Punkten mit den Daten
vereinbar, Gleichheit ist also ebenso wenig belegt. Ob eine Lücke klinisch
relevant ist, ist eine zweite, eigene Frage: Sie hängt von ihrer Größe und den
Folgen für die Betroffenen ab, nicht vom p-Wert. Adressieren würde man eine
*bestätigte* Lücke (Differenz-KI schließt 0 aus) durch: (1) größere, unabhängige Kohorte zur Bestätigung,
(2) Datenanalyse nach Ursache (Dokumentationsunterschiede, Ereignisraten),
(3) gruppenspezifische Schwellen oder Kalibrierung.

## Aufgabe 4 – Model Card ausfüllen
Felder ohne prospektive Studie nicht befüllbar:

- Leistungsmetriken auf externen Kohorten (anderer Klinik, anderer Zeitraum)
- Subgruppenleistung auf Populationen, die im Testsplit gar nicht oder zu
  klein vertreten sind (z. B. Ethnizität, seltene Altersgruppen) – Geschlecht
  und ein Altersgruppen-Split (Schwelle 65 Jahre) lassen sich mit dem
  vorhandenen Testsplit inkl. Bootstrap-CI berechnen (siehe README Abschnitt 4).
- Kalibrierung in der Zielklinik
- Post-Market Surveillance Ergebnisse
- Ergebnis der klinischen Evaluierungsstudie (MDR-Anforderung)

## Bonus – Kalibrierungs-Drift
```python
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# Re-create the 100%-shift test set from Aufgabe 2 (Xs_max/ys_max): duplicate
# every positive test case once, same construction as sf=1.0 above.
rng = np.random.default_rng(42)
pos_idx = y_test[y_test == 1].index
extra_idx = rng.choice(pos_idx, size=(y_test == 1).sum(), replace=True)
Xs_max = pd.concat([X_test, X_test.loc[extra_idx]])
ys_max = pd.concat([y_test, y_test.loc[extra_idx]])

fig, ax = plt.subplots(figsize=(5.5, 5))
for label, Xs, ys in [("kein Shift", X_test, y_test),
                       ("100% Shift", Xs_max, ys_max)]:
    proba = pipe.predict_proba(Xs)[:, 1]
    frac_pos, mean_pred = calibration_curve(ys, proba, n_bins=5)
    ece = np.mean(np.abs(frac_pos - mean_pred))
    ax.plot(mean_pred, frac_pos, marker="o", label=f"{label} (ECE={ece:.3f})")

ax.plot([0,1],[0,1], ls="--", color="gray", label="Perfekt kalibriert")
ax.set_xlabel("Mittlere Vorhersagewahrscheinlichkeit")
ax.set_ylabel("Beobachtete Häufigkeit")
ax.set_title("Kalibrierungskurve: Shift-Effekt")
ax.legend(); plt.tight_layout(); plt.show()
```
Tatsächlicher Lauf: ECE steigt von 0,127 (kein Shift, n=125, Positivrate 34,4 %)
auf 0,173 (100 % Shift, n=168, Positivrate 51,2 %). Schon ohne Shift ist das
Modell nicht gut kalibriert: `class_weight="balanced"` schiebt die
Wahrscheinlichkeiten nach oben (mittlere Vorhersage ≈ 0,46 bei 34 % beobachteter
Rate). Mit dem Shift steigt der ECE weiter, obwohl (wie in Aufgabe 2 gezeigt) die
**ROC-AUC nahezu unverändert bleibt**: In den oberen Risikobereichen liegen die
Vorhersagen jetzt deutlich unter der beobachteten Häufigkeit, weil das Modell die
höhere Ereignisrate nicht kennt. Das ist die komplementäre Lektion zu Aufgabe 2:
Ranking (AUC) kann intakt bleiben, während die Kalibrierung unter Prävalenz-Shift
leidet; eine Rekalibrierung (z. B. Platt Scaling) auf der neuen Grundrate wäre
dann nötig. Der ECE hier ist der ungewichtete Mittelwert über die Bins; die
übliche Definition gewichtet jeden Bin mit seinem Anteil an den Fällen.
