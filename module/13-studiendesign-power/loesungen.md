# Musterlösungen · Studiendesign und Fallzahlplanung

## Aufgabe 1 – PICO/PECO

- Population: stationäre/intensivmedizinische Patient:innen der Kohorte.
- Exposure: erhöhtes Laktat bei Aufnahme, z. B. >2 mmol/l oder kontinuierlich pro 1 mmol/l.
- Comparator: normales Laktat oder niedrigere Laktatwerte.
- Outcome: 30-Tage-Mortalität.

Wichtig: Vorher entscheiden, ob Laktat kontinuierlich oder als Schwelle analysiert
wird. Kontinuierlich erhält mehr Information; Schwellen sind klinisch oft leichter
zu kommunizieren.

## Aufgabe 2 – Fallzahl

```python
from statsmodels.stats.power import TTestIndPower

analysis = TTestIndPower()
for d in [0.3, 0.5]:
    n = analysis.solve_power(effect_size=d, alpha=0.05, power=0.8,
                             ratio=1.0, alternative="two-sided")
    print(d, n)
```

Ergebnis (aufgerundet): 176 Patient:innen je Gruppe für d = 0,3, aber nur 64 für
d = 0,5. Kleinere Effekte brauchen deutlich mehr Patient:innen, grob mit dem
Kehrwert des Quadrats der Effektgröße: (0,5 / 0,3)² ≈ 2,8. Das ist der Kern von
Powerplanung: klinisch relevante Mini-Effekte sind statistisch teuer.

## Aufgabe 3 – Präzision

```python
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

df = pd.read_csv("{{DATEN}}kohorte.csv")
events = int(df["verstorben_30d"].sum())
n = len(df)
lo, hi = proportion_confint(events, n, method="wilson")
print(events, n, events / n, lo, hi)
```

```r
library(tidyverse)
df <- read_csv("{{DATEN}}kohorte.csv", show_col_types = FALSE)
events <- sum(df$verstorben_30d)
n <- nrow(df)
# correct = FALSE gives the plain Wilson score interval, as in Python;
# the default correct = TRUE would give a wider, continuity-corrected interval.
ci <- prop.test(events, n, correct = FALSE)$conf.int
cat(sprintf("events=%d/%d rate=%.4f Wilson-KI [%.4f, %.4f]\n",
            events, n, events / n, ci[1], ci[2]))
```

Auf den Kursdaten: 78/500 = 15,6 %, Wilson-95-%-KI **[0,127; 0,190]** (12,7 %
bis 19,0 %) – in beiden Sprachen identisch. Das KI zeigt nicht nur den
Punktschätzer, sondern wie unsicher die Rate ist: eine Spanne von über sechs
Prozentpunkten trotz N = 500. Für Qualitätsmonitoring kann das zu breit sein;
für eine erste Übung reicht es.

**Hinweis:** Setzt man in R den Default `correct = TRUE`, erhält man ein
*stetigkeitskorrigiertes* Score-Intervall ([0,126; 0,191]) – etwas breiter und
**nicht** das reine Wilson-KI. Für Parität mit Python `correct = FALSE`.

## Aufgabe 4 – Events per Variable

```python
import pandas as pd

cohort = pd.read_csv("{{DATEN}}kohorte.csv")
events = int(cohort["verstorben_30d"].sum())
print("bei 10 EPV:", events // 10)
print("bei 20 EPV:", events // 20)
```

Auf den Kursdaten: 78 Todesfälle, also `78 // 10 = 7` Parameter bei 10 EPV
bzw. `78 // 20 = 3` Parameter bei 20 EPV. Bei wenigen Ereignissen sollte das
Modell klein bleiben. Alter und Diabetes sind plausibel; viele Kategorien und
Interaktionen wären instabil.

## Aufgabe 5 – Analyseplan

Laut DAG (Modul 12, Abschnitt 4, und `data/README.md`): `alter` ist ein **Confounder** von Diabetes und
Mortalität (verursacht beides) und gehört adjustiert. `sofa_score` liegt auf
dem kausalen Pfad `Diabetes → SOFA → Mortalität`, ist also ein **Mediator**.
Adjustiert man dafür, schätzt man nicht mehr den totalen Effekt von Diabetes
auf die Mortalität, sondern nur den Anteil, der nicht über SOFA vermittelt
wird, eine andere Frage. Für den totalen Effekt gehört SOFA **nicht** in die
Hauptanalyse.

Beispiel-Analyseplan:

> Primärer Endpunkt ist 30-Tage-Mortalität. Hauptanalyse ist eine logistische
> Regression mit Diabetes als Exposition, adjustiert für den Confounder Alter
> (nicht für SOFA, das laut DAG ein Mediator ist). Wir berichten Odds Ratio,
> 95-%-KI und p-Wert. Fehlende Kovariaten werden geprüft; bei relevanter
> Missingness erfolgt Multiple Imputation, Complete Case dient als
> Sensitivitätsanalyse.
