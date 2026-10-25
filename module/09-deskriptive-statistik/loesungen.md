# Musterlösungen · Deskriptive Statistik und die Table 1

> Vergleiche zuerst mit deinem eigenen Versuch. Es gibt oft mehrere richtige Wege.

## Aufgabe 1 – Mittelwert oder Median?

```python
import pandas as pd
k = pd.read_csv("{{DATEN}}kohorte.csv")

for col in ["crp_mg_l", "sofa_score", "verweildauer_tage"]:
    mw = k[col].mean()
    med = k[col].median()
    print(f"{col:25s}  Mittelwert={mw:.1f}  Median={med:.1f}  Diff={abs(mw-med):.1f}")
```
```r
library(readr)
k <- read_csv("{{DATEN}}kohorte.csv", show_col_types = FALSE)
for (col in c("crp_mg_l", "sofa_score", "verweildauer_tage")) {
  cat(sprintf("%-25s  Mittelwert=%.1f  Median=%.1f\n",
              col, mean(k[[col]], na.rm=TRUE), median(k[[col]], na.rm=TRUE)))
}
```
**Befund:** Deutlich weicht nur `crp_mg_l` ab (Mittelwert 75,3 vs. Median
61,2 mg/l): rechtsschief durch wenige sehr hohe Werte. Bei `verweildauer_tage`
(6,5 vs. 6) und `sofa_score` (3,7 vs. 3) ist die Differenz klein, obwohl beide
ebenfalls rechtsschief sind (Schiefe 1,1 bzw. 0,6). Bei ganzzahligen Werten mit
kleinem Wertebereich verrät der Abstand Mittelwert–Median die Schiefe nur grob;
ein Histogramm zeigt sie zuverlässiger. 
**Empfehlung:** CRP und Verweildauer → Median [IQR]; SOFA-Score (ordinal) →
ebenfalls Median [IQR], Mittelwert ± SD höchstens zusätzlich, wenn das Journal es
verlangt.

## Aufgabe 2 – Häufigkeiten und Anteile

```python
import pandas as pd
k = pd.read_csv("{{DATEN}}kohorte.csv")

# a) Heart failure (aufnahmegrund "Herzinsuffizienz")
n_hi = (k["aufnahmegrund"] == "Herzinsuffizienz").sum()
print(f"Herzinsuffizienz: {n_hi} ({n_hi/len(k):.1%})")

# b) Hypertension
print(f"Hypertonie: {k['hypertonie'].mean():.1%}")

# c) Cross-tabulation
xt = pd.crosstab(k["raucherstatus"], k["diabetes"],
                 margins=True, margins_name="Gesamt")
print(xt)
print(pd.crosstab(k["raucherstatus"], k["diabetes"], normalize="index").round(3) * 100)
```
```r
library(readr)
k <- read_csv("{{DATEN}}kohorte.csv", show_col_types = FALSE)

# a)
n_hi <- sum(k$aufnahmegrund == "Herzinsuffizienz")
cat(sprintf("Herzinsuffizienz: %d (%.1f %%)\n", n_hi, n_hi/nrow(k)*100))

# b)
cat(sprintf("Hypertonie: %.1f %%\n", mean(k$hypertonie)*100))

# c)
print(table(k$raucherstatus, k$diabetes))
print(round(prop.table(table(k$raucherstatus, k$diabetes), margin = 1) * 100, 1))
```

## Aufgabe 3 – Gruppenvergleich manuell

```python
import pandas as pd
k = pd.read_csv("{{DATEN}}kohorte.csv")
for col in ["alter", "sofa_score", "crp_mg_l", "verweildauer_tage"]:
    g = k.groupby("verstorben_30d")[col]
    q = g.quantile([0.25, 0.50, 0.75]).unstack()
    print(f"\n{col}:")
    for group, row in q.iterrows():
        print(f"  Gruppe {group}: Median={row[0.50]:.1f} [{row[0.25]:.1f}; {row[0.75]:.1f}]")
```
```r
library(readr)
k <- read_csv("{{DATEN}}kohorte.csv", show_col_types = FALSE)
for (col in c("alter", "sofa_score", "crp_mg_l", "verweildauer_tage")) {
  cat(sprintf("\n%s:\n", col))
  k |>
    dplyr::group_by(verstorben_30d) |>
    dplyr::summarise(
      zusammenfassung = sprintf("%.1f [%.1f; %.1f]",
                                median(.data[[col]], na.rm = TRUE),
                                quantile(.data[[col]], 0.25, na.rm = TRUE),
                                quantile(.data[[col]], 0.75, na.rm = TRUE))
    ) |>
    print()
}
```
**Befund:** `alter` und `sofa_score` zeigen die auffälligsten Unterschiede
(Verstorbene sind median ~7 Jahre älter, 69 vs. 62 Jahre; SOFA-Score fast
doppelt so hoch, 5,5 vs. 3,0). `crp_mg_l` und `verweildauer_tage` unterscheiden
sich zwischen den Gruppen dagegen kaum. Das passt zur eingebauten Wahrheit des
Datensatzes (Modul 12 prüft das formal).

## Aufgabe 4 – Table 1 nach Aufnahmegrund

```python
import pandas as pd
from tableone import TableOne
k = pd.read_csv("{{DATEN}}kohorte.csv")
k["geschlecht"] = k["geschlecht"].replace({"w": "weiblich"})   # as in README §4

table1 = TableOne(
    k,
    columns=["alter", "geschlecht", "diabetes", "sofa_score", "crp_mg_l"],
    categorical=["geschlecht", "diabetes"],
    nonnormal=["sofa_score", "crp_mg_l"],   # skewed -> median [Q1,Q3], Kruskal-Wallis
    groupby="aufnahmegrund",
    pval=True,
)
print(table1.tabulate(tablefmt="simple"))
```
```r
library(readr)
library(dplyr)
library(gtsummary)
k <- read_csv("{{DATEN}}kohorte.csv", show_col_types = FALSE)

k |>
  mutate(geschlecht = if_else(geschlecht == "w", "weiblich", geschlecht),
         diabetes = factor(diabetes, levels = c(0, 1), labels = c("nein", "ja"))) |>
  select(alter, geschlecht, diabetes, sofa_score, crp_mg_l, aufnahmegrund) |>
  tbl_summary(by = aufnahmegrund,
              statistic = list(all_continuous() ~ "{median} [{p25}; {p75}]",
                               alter ~ "{mean} ({sd})",
                               all_categorical() ~ "{n} ({p}%)")) |>
  add_p(test = list(alter ~ "aov"))
```
Beide Sprachen zeigen jetzt dieselben Kennzahlen: Alter als Mittelwert (SD) mit
ANOVA, die schiefen Variablen SOFA-Score und CRP als Median [Q1; Q3] mit
Kruskal-Wallis-Test, und `w` ist wie in der Lektion zu `weiblich` vereinheitlicht.

**Beobachtung:** Septische Patient:innen haben mit Abstand den höchsten
SOFA-Score (Median 6 vs. 3 in allen anderen Gruppen, p < 0,001), sie sind bei
Aufnahme am schwersten krank. Beim CRP unterscheiden sich die Gruppen zwar
formal (p = 0,008), aber ohne klinisch klares Muster: Die Mediane liegen
zwischen 52 und 70 mg/l, und Sepsis gehört mit rund 54 mg/l zu den niedrigsten.
Alter (p = 0,42) und Geschlecht unterscheiden sich kaum. Der SOFA-Score ist der
klinisch aussagekräftigste Unterschied. Denk an die Fallstricke der Lektion: Die
p-Werte beschreiben hier nur, sie sind kein Ziel einer Table 1.

## Aufgabe 5 – Fehlende Werte in Table 1

```python
import pandas as pd
from tableone import TableOne

labs   = pd.read_csv("{{DATEN}}labor.csv")

df = pd.read_csv("{{DATEN}}kohorte.csv").merge(labs, on="patient_id", how="left")
table1 = TableOne(df, columns=["alter", "bmi", "laktat_mmol_l", "kreatinin_mg_dl"],
                  groupby="verstorben_30d", pval=False, missing=True)
print(table1.tabulate(tablefmt="simple"))
```
```r
library(readr)
library(gtsummary)

cohort <- read_csv("{{DATEN}}kohorte.csv", show_col_types = FALSE)
labs   <- read_csv("{{DATEN}}labor.csv", show_col_types = FALSE)
df <- dplyr::left_join(cohort, labs, by="patient_id")

df |>
  dplyr::select(alter, bmi, laktat_mmol_l, kreatinin_mg_dl, verstorben_30d) |>
  tbl_summary(by=verstorben_30d, missing="ifany") |>
  add_overall()
```
**Befund:** `bmi` fehlt bei ~6 % (zufällig fehlend, MCAR), `laktat_mmol_l` bei
~17 % (**MAR**: die Fehlwahrscheinlichkeit hängt vom beobachteten
SOFA-Score ab, Laktat wird bei höherem SOFA-Score klinisch häufiger gemessen,
fehlt also seltener bei schwerer kranken Patient:innen, siehe `data/README.md`
und Modul 14 (Umgang mit fehlenden Werten)).

## Bonus – IQR des Laktats je Aufnahmegrund

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

labs   = pd.read_csv("{{DATEN}}labor.csv")

df = pd.read_csv("{{DATEN}}kohorte.csv").merge(labs, on="patient_id", how="left")

iqr_tbl = df.groupby("aufnahmegrund")["laktat_mmol_l"].agg(
    q25=lambda x: x.quantile(0.25),
    median="median",
    q75=lambda x: x.quantile(0.75),
)
iqr_tbl["iqr"] = iqr_tbl["q75"] - iqr_tbl["q25"]
print(iqr_tbl.sort_values("iqr", ascending=False).round(2))

# Boxplot for visual comparison
order = iqr_tbl.sort_values("iqr", ascending=False).index.tolist()
fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(data=df.dropna(subset=["laktat_mmol_l"]),
            x="aufnahmegrund", y="laktat_mmol_l", order=order,
            hue="aufnahmegrund", palette="Set2", legend=False, ax=ax)
ax.set_title("Laktat [mmol/l] nach Aufnahmegrund (nach IQR sortiert)")
ax.set_xlabel("Aufnahmegrund")
ax.set_ylabel("Laktat (mmol/l)")
fig.tight_layout()
fig.savefig("laktat_iqr_nach_aufnahmegrund.png", dpi=120)
```
```r
library(tidyverse)

cohort <- read_csv("{{DATEN}}kohorte.csv", show_col_types = FALSE)
labs   <- read_csv("{{DATEN}}labor.csv", show_col_types = FALSE)
df <- left_join(cohort, labs, by="patient_id")

df |>
  group_by(aufnahmegrund) |>
  summarise(
    q25    = quantile(laktat_mmol_l, 0.25, na.rm=TRUE),
    median = median(laktat_mmol_l, na.rm=TRUE),
    q75    = quantile(laktat_mmol_l, 0.75, na.rm=TRUE),
    iqr    = q75 - q25
  ) |>
  arrange(desc(iqr))
```
**Befund:** Die Streuung ist zwischen den Aufnahmegründen ähnlich groß (IQR
zwischen ~1,4 und ~2,0 mmol/l); in dieser Kohorte hat COPD-Exazerbation die
größte, Pneumonie die kleinste Streuung. Sepsis liegt im Mittel (nicht im
Median) am höchsten, aber die Unterschiede zwischen den Gruppen sind klein und
könnten in einer anderen Stichprobe anders ausfallen, das Muster ist mit
n = 55–105 je Gruppe nicht robust genug für eine starke Aussage.
