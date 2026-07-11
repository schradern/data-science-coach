# Musterlösungen · Mixed-Effects-Modelle für Longitudinaldaten

## Übung 1: LMM-Spezifikation

### Python-Lösung
```python
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np

# Simulierter Datensatz
np.random.seed(42)
n_patients = 20
records = []
for pid in range(1, n_patients + 1):
    alter = np.random.randint(45, 80)
    baseline_rr = np.random.normal(16, 2)  # Individuelle Baseline
    for stunde in [1, 2, 3, 4]:
        # Atemfrequenz sinkt leicht über die Zeit
        rr = baseline_rr - 0.5 * stunde + np.random.normal(0, 0.5)
        records.append({"patient_id": pid, "alter": alter, "zeit_stunden": stunde, "respiratory_rate": rr})

df_rr = pd.DataFrame(records)

# 1. Random-Intercept-Modell fitten
model = smf.mixedlm("respiratory_rate ~ zeit_stunden + alter", df_rr, groups=df_rr["patient_id"])
result = model.fit()

print(result.summary())
```

**Interpretation des Koeffizienten für `zeit_stunden`:**
Der Koeffizient beschreibt die durchschnittliche Änderung der Atemfrequenz pro Stunde. Ein negativer Wert (z. B. `-0,5`) bedeutet, dass die Atemfrequenz pro Stunde postoperativ im Schnitt um 0,5 Atemzüge pro Minute sinkt – adjustiert für das Alter und unter Berücksichtigung der individuellen Patient:innen-Baseline.

**Praxis:** Wie im README (Abschnitt 4) gilt auch hier: Der interessante Vergleich ist nicht nur der
Koeffizient, sondern `result.bse` (Python) bzw. `summary(model)$coefficients[, "Std. Error"]` (R) gegen
das äquivalente OLS-Modell – erst der Standardfehler-Vergleich zeigt, ob die wiederholten Messungen etwas
verändern.

### R-Lösung
```r
library(lme4)

# 1. Modell fitten
model <- lmer(respiratory_rate ~ zeit_stunden + alter + (1 | patient_id), data = df_rr)

# 2. Summary
summary(model)
```

---

## Übung 2: OLS vs. LMM (Konzept)

1. **HDL-Vergleich (einzelner Zeitpunkt):** **OLS zulässig.** Da jede:r Patient:in nur genau einmal gemessen wurde, sind die Daten unabhängig. Eine Standard-OLS-Regression (oder ein t-Test) ist methodisch korrekt.
2. **Postoperativer Schmerzverlauf (stündlich über 12 Stunden):** **LMM erforderlich.** Die wiederholten Messungen innerhalb desselben/derselben Patient:in sind stark korreliert. Ein LMM mit `patient_id` als Random Effect ist zwingend nötig.
3. **Hausarztpraxen (Clustering):** **LMM erforderlich (bzw. Cluster-robuste Standardfehler).** Patient:innen in derselben Praxis teilen sich oft ähnliche Ärzt:innen, Geografie und Behandlungsstile. Hier sollte die `praxis_id` als Random Intercept aufgenommen werden (hierarchische Daten).

---

## Übung 3: Warum OLS die Standardfehler unterschätzt (Design-Effekt)

1. **Design-Effekt:** `DEFF = 1 + (m − 1) · ICC = 1 + (4 − 1) · 0,20 = 1 + 0,60 = 1,60`.
2. **Effektive Stichprobengröße:** `N_eff = (N · m) / DEFF = (50 · 4) / 1,60 = 200 / 1,60 = 125`. Statt 200 unabhängiger Beobachtungen trägt die Studie effektiv nur die Information von **125**.
3. **SE-Unterschätzung:** Der korrekte Standardfehler ist um den Faktor `√DEFF = √1,60 ≈ 1,265` größer als der naive OLS-SE. Anders gesagt: Der naive SE beträgt nur `1 / 1,265 ≈ 0,79` des korrekten – er ist also rund **21 % zu klein**. Konfidenzintervalle werden dadurch zu eng und p-Werte zu optimistisch.

   **Begründung:** Ein patientenkonstanter Prädiktor (Diabetes) hat in allen 4 Zeilen einer Person denselben Wert. OLS zählt jede der 4 Zeilen als neue, unabhängige Evidenz (Pseudo-Replikation), obwohl eine Person nur **eine** Information über den Diabetes-Effekt liefert. Bei einem zeitvarianten Prädiktor (z. B. `tag`, der sich zwischen den Messungen ändert) trägt dagegen jede Messung echte Zusatzinformation; dort greift die Faustregel „OLS-SE immer zu klein" nicht zuverlässig (vgl. README Abschnitt 4, wo das SE-Verhältnis für `tag` bei 0,94× lag).

---

## Übung 4: ICC aus den Varianzkomponenten

1. **Intraklassen-Korrelation:** `ICC = τ² / (τ² + σ²) = 9,0 / (9,0 + 66,0) = 9,0 / 75,0 = 0,12`. Rund **12 %** der Gesamtvarianz im MAP liegen zwischen Patient:innen, 88 % innerhalb einer Person über die Zeit – exakt der Wert aus der Kurskohorte (README Abschnitt 4).
2. **SD der Random Intercepts:** `√τ² = √9,0 = 3,0 mmHg`. Die individuellen Baselines der Patient:innen streuen also mit rund 3 mmHg um das Populationsmittel: Eine typische Person startet etwa 3 mmHg über oder unter dem gemeinsamen Achsenabschnitt.
3. **τ² ≈ 0:** Dann gibt es praktisch keine Variabilität zwischen den Patient:innen-Baselines (ICC ≈ 0), die wiederholten Messungen sind kaum korreliert, und OLS und Mixed Model liefern nahezu identische Standardfehler – das Mixed Model wird überflüssig. In der Praxis wirft `lme4` (bzw. `statsmodels`) dann eine „Singular Fit"-Warnung. Erst ein ICC deutlich über 0 macht die Korrektur überhaupt nötig.
