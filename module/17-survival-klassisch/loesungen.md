# Musterlösungen · Klassische Überlebenszeitanalyse

## Aufgabe 1 – Ereignis und Zeit

- Zeitvariable: `fu_zeit_tage` (Beobachtungs-/Follow-up-Zeit)
- Ereignisvariable: `status` (1 = Tod beobachtet, 0 = zensiert)
- Zensierung: Patient:innen ohne Tod im Beobachtungsfenster gelten bis zur
  beobachteten Zeit als ereignisfrei.

`verweildauer_tage` ist **nicht** geeignet: Es ist die Verweildauer im
Krankenhaus (Länge des Aufenthalts), unabhängig davon erhoben, ob und wann
ein Todesereignis eintrat. Es gibt keinen definierten Zusammenhang zwischen
Verweildauer und Sterbezeitpunkt, eine Person kann z. B. lange nach
Entlassung versterben oder am Todestag entlassen (verstorben) werden. Eine
Survival-Analyse auf `verstorben_30d` braucht eine echte Zeit-bis-Ereignis-
Spalte, das ist `fu_zeit_tage`.

## Aufgabe 2 – Kaplan-Meier

```python
import pandas as pd
from lifelines import KaplanMeierFitter

df = pd.read_csv("{{DATEN}}kohorte.csv")
df["sepsis"] = (df["aufnahmegrund"] == "Sepsis").astype(int)

for label, g in [("Sepsis", df[df["sepsis"] == 1]), ("Nicht-Sepsis", df[df["sepsis"] == 0])]:
    km = KaplanMeierFitter(label=label)
    km.fit(g["fu_zeit_tage"], event_observed=g["status"])
    print(label, km.survival_function_at_times([10, 20]))
```

Auf den Kursdaten: Sepsis ≈ 86,2 % (Tag 10) / 74,3 % (Tag 20) überlebend,
Nicht-Sepsis ≈ 95,7 % (Tag 10) / 92,2 % (Tag 20). Die Werte sind geschätzte
ereignisfreie Wahrscheinlichkeiten zu festen Zeitpunkten. Sie sind oft
leichter verständlich als eine HR.

## Aufgabe 3 – Log-rank

```python
from lifelines.statistics import logrank_test

a = df[df["sepsis"] == 1]
b = df[df["sepsis"] == 0]
res = logrank_test(a["fu_zeit_tage"], b["fu_zeit_tage"],
                   event_observed_A=a["status"],
                   event_observed_B=b["status"])
print(res)
```

Auf den Kursdaten: p < 0,0001, die Kurven unterscheiden sich klar. Ein
kleiner p-Wert spricht für unterschiedliche Überlebenskurven. Der Test sagt
nicht, wie groß der Unterschied klinisch ist; dafür braucht man Kurve, absolute
Risiken und ggf. Cox-HR.

## Aufgabe 4 – Cox

```python
from lifelines import CoxPHFitter

cph = CoxPHFitter()
cph.fit(df[["fu_zeit_tage", "status", "sepsis", "alter", "sofa_score"]],
        duration_col="fu_zeit_tage", event_col="status")
print(cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]])
```

Auf den Kursdaten: HR(Sepsis) ≈ 1,52 (95%-KI [0,86; 2,71], p ≈ 0,15, hier
nicht signifikant, weil Alter und SOFA einen Teil der rohen Assoziation
erklären), HR(Alter) ≈ 1,04/Jahr (p < 0,001), HR(SOFA) ≈ 1,42 pro Punkt
(p < 0,001). Die HR für `sepsis` ist adjustiert für Alter und SOFA. Sie
beschreibt das Verhältnis der momentanen Hazards, nicht direkt das
kumulative Risiko.

## Aufgabe 5 – Annahmen

```python
cph.check_assumptions(df[["fu_zeit_tage", "status", "sepsis", "alter", "sofa_score"]],
                      p_value_threshold=0.05, show_plots=False)
```

Auf den Kursdaten meldet `check_assumptions` keine Verletzung (alle
Schoenfeld-Residuen-Tests p > 0,05), die PH-Annahme ist plausibel. Bei
verletzter PH-Annahme: stratifizieren, zeitabhängige Koeffizienten modellieren
oder absolute Zeitpunkte/restricted mean survival time berichten.
