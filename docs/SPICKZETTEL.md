# Spickzettel

Die häufigsten Handgriffe nebeneinander in **Python (pandas)**, **R (tidyverse)**
und **SQL**. Zum Offenhalten neben der Arbeit.

## Daten laden

```python
from lib.helpers import load_cohort, load_labs, load_vitals
df = load_cohort()
```
```r
source(file.path("lib", "helpers.R")); library(tidyverse)
df <- load_cohort()
```

## Die fünf Kern-Verben

| Aufgabe | pandas | dplyr | SQL |
|---|---|---|---|
| Zeilen filtern | `df[df.alter > 65]` / `df.query("alter > 65")` | `filter(alter > 65)` | `WHERE alter > 65` |
| Spalten wählen | `df[["alter","sofa_score"]]` | `select(alter, sofa_score)` | `SELECT alter, sofa_score` |
| Spalte ableiten | `df.assign(alt70 = df.alter >= 70)` | `mutate(alt70 = alter >= 70)` | `... , alter >= 70 AS alt70` |
| sortieren | `df.sort_values("alter")` | `arrange(alter)` | `ORDER BY alter` |
| gruppieren + zusammenfassen | `df.groupby("aufnahmegrund")["crp_mg_l"].mean()` | `group_by(aufnahmegrund) \|> summarise(m = mean(crp_mg_l))` | `SELECT aufnahmegrund, AVG(crp_mg_l) GROUP BY aufnahmegrund` |
| verbinden (join) | `a.merge(b, on="patient_id", how="left")` | `left_join(a, b, by = "patient_id")` | `a LEFT JOIN b USING (patient_id)` |

## Erster Blick auf die Daten

| | pandas | dplyr |
|---|---|---|
| Form | `df.shape` | `dim(df)` |
| Spalten/Typen | `df.dtypes` | `glimpse(df)` |
| Überblick | `df.describe()` | `summary(df)` |
| Häufigkeiten | `df["raucherstatus"].value_counts()` | `count(df, raucherstatus)` |
| fehlende Werte | `df.isna().sum()` | `colSums(is.na(df))` |

## Fehlende Werte und Recodierung

```python
df["geschlecht"] = df["geschlecht"].replace({"w": "weiblich"})
df["laktat_mmol_l"].fillna(df["laktat_mmol_l"].median())   # bewusst, nicht reflexhaft
df.dropna(subset=["bmi"])
```
```r
df <- df |> mutate(geschlecht = if_else(geschlecht == "w", "weiblich", geschlecht))
df |> tidyr::replace_na(list(laktat_mmol_l = median(df$laktat_mmol_l, na.rm = TRUE)))
df |> drop_na(bmi)
```

## Umformen (long ↔ wide)

```python
v.pivot_table(index="patient_id", columns="tag", values="herzfrequenz")   # long -> wide
df.melt(id_vars="patient_id")                                              # wide -> long
```
```r
pivot_wider(v, names_from = tag, values_from = herzfrequenz)   # long -> wide
pivot_longer(df, -patient_id)                                  # wide -> long
```

## Deskriptiv und „Table 1“

```python
from tableone import TableOne
TableOne(df, groupby="verstorben_30d", pval=True)
```
```r
library(gtsummary)
tbl_summary(df, by = verstorben_30d) |> add_p()
```

## Tests (immer mit Effektmaß + KI berichten, nicht nur p)

| | Python (scipy) | R |
|---|---|---|
| t-Test | `stats.ttest_ind(a, b, equal_var=False)` | `t.test(a, b)` |
| gepaarter t-Test | `stats.ttest_rel(vor, nach)` | `t.test(vor, nach, paired = TRUE)` |
| Mann-Whitney | `stats.mannwhitneyu(a, b)` | `wilcox.test(a, b)` |
| Wilcoxon gepaart | `stats.wilcoxon(vor, nach)` | `wilcox.test(vor, nach, paired = TRUE)` |
| χ²-Test | `stats.chi2_contingency(tab)` | `chisq.test(tab)` |
| Fisher exakt | `stats.fisher_exact(tab)` | `fisher.test(tab)` |
| Kruskal-Wallis | `stats.kruskal(g1, g2, g3)` | `kruskal.test(list(g1, g2, g3))` |

## Welcher Test?

| Frage | Methode | Merke |
|---|---|---|
| zwei unabhängige stetige Gruppen | Welch-t oder Mann-Whitney | Welch für Mittelwerte, Mann-Whitney bei starker Schiefe |
| vorher/nachher | gepaarter t-Test oder Wilcoxon | dieselben Patient:innen, also gepaart |
| mehr als zwei stetige Gruppen | ANOVA oder Kruskal-Wallis | Post-hoc vorher planen |
| kategorial × kategorial | χ² oder Fisher | Fisher bei kleinen erwarteten Zellen |
| binärer Outcome + Adjustierung | logistische Regression | OR + 95-%-KI berichten |
| Zeit bis Ereignis | Kaplan-Meier, Log-rank, Cox | Zensierung nutzen, PH prüfen |

## Modelle

```python
import statsmodels.formula.api as smf
m = smf.logit("verstorben_30d ~ alter + sofa_score + crp_mg_l + diabetes", df).fit()
m.summary(); np.exp(m.params)        # Odds Ratios

from lifelines import CoxPHFitter
CoxPHFitter().fit(df[["fu_zeit_tage", "status", "alter"]],
                   duration_col="fu_zeit_tage", event_col="status")
```
```r
m <- glm(verstorben_30d ~ alter + sofa_score + crp_mg_l + diabetes, df, family = binomial)
exp(coef(m))                          # Odds Ratios
survival::coxph(survival::Surv(fu_zeit_tage, status) ~ alter, df)
```

## Diagnostik

```python
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
tn, fp, fn, tp = confusion_matrix(y, score >= schwelle).ravel()
sens = tp / (tp + fn); spec = tn / (tn + fp)
ppv = tp / (tp + fp); npv = tn / (tn + fn)
```
```r
tab <- table(test = score >= schwelle, ereignis = y)
sens <- tab["TRUE","1"] / sum(tab[,"1"])
spec <- tab["FALSE","0"] / sum(tab[,"0"])
```

## Grafik (Grundgerüst)

```python
import seaborn as sns
sns.histplot(df, x="alter"); sns.boxplot(df, x="aufnahmegrund", y="laktat_mmol_l")
```
```r
ggplot(df, aes(alter)) + geom_histogram()
ggplot(df, aes(aufnahmegrund, laktat_mmol_l)) + geom_boxplot()
```
