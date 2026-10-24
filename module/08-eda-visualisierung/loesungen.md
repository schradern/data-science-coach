# Musterlösungen · Explorative Datenanalyse und Datenvisualisierung

> Vergleiche zuerst mit deinem eigenen Versuch. Es gibt oft mehrere richtige Wege.

## Aufgabe 1 – Erste Erkundung

```python
import pandas as pd

labs   = pd.read_csv("{{DATEN}}labor.csv")

df = pd.read_csv("{{DATEN}}kohorte.csv").merge(labs, on="patient_id", how="left")

for col in ["bmi", "kreatinin_mg_dl", "haemoglobin_g_dl"]:
    s = df[col]
    print(f"{col}: Mittel={s.mean():.2f}  Median={s.median():.2f}  "
          f"SD={s.std():.2f}  Schiefe={s.skew():.2f}")
```
```r
library(tidyverse)

cohort <- read_csv("{{DATEN}}kohorte.csv", show_col_types = FALSE)
labs   <- read_csv("{{DATEN}}labor.csv", show_col_types = FALSE)
df <- left_join(cohort, labs, by = "patient_id")

for (col in c("bmi", "kreatinin_mg_dl", "haemoglobin_g_dl")) {
  cat(sprintf("%-20s  Mittel=%.2f  Median=%.2f  SD=%.2f\n",
              col, mean(df[[col]], na.rm=TRUE),
              median(df[[col]], na.rm=TRUE),
              sd(df[[col]], na.rm=TRUE)))
}
```
**Befund:** `kreatinin_mg_dl` ist mit einer Schiefe von ~1,27 am stärksten
rechtsschief (Mittel ≈ 1,23, Median ≈ 1,02 mg/dl). Das ist klinisch plausibel:
die meisten Patient:innen haben ein normales Kreatinin, aber schwere Niereninsuffizienz
treibt die Verteilung nach rechts.

## Aufgabe 2 – Häufigkeiten und fehlende Werte

```python
# Smoking status – absolute and relative
print(df["raucherstatus"].value_counts())
print(df["raucherstatus"].value_counts(normalize=True).round(3))

# Missing values – all columns
print(df.isna().sum()[df.isna().sum() > 0])

# Share of missing lactate values
anteil = df["laktat_mmol_l"].isna().mean()
print(f"laktat_mmol_l fehlend: {anteil:.1%}")
```
```r
df |> count(raucherstatus, sort = TRUE) |>
  mutate(anteil = n / sum(n))

colSums(is.na(df))[colSums(is.na(df)) > 0]

mean(is.na(df$laktat_mmol_l))
```
**Befund:** „nie“ (~47 %), „ehemalig“ (~31 %), „aktiv“ (~22 %). Fehlende Werte
gibt es bei `gewicht_kg`/`bmi` (~6 %, zufällig, MCAR) und `laktat_mmol_l`
(~17 %). Das Laktat-Fehlen ist **informativ**, aber **MAR** (nicht MNAR): die
Fehlwahrscheinlichkeit hängt vom beobachteten `sofa_score` ab, Laktat wird bei
höherem SOFA-Score (Verdacht auf Schock/Sepsis) klinisch häufiger gemessen,
fehlt also *seltener* bei schwer kranken Patient:innen (siehe `data/README.md`
und Modul 14 (Umgang mit fehlenden Werten)).

## Aufgabe 3 – Ausreißer

```python
series = df["kreatinin_mg_dl"].dropna()
q1, q3 = series.quantile(0.25), series.quantile(0.75)
iqr = q3 - q1
upper = q3 + 1.5 * iqr
n_high = (series > upper).sum()
print(f"IQR={iqr:.2f}  Obergrenze={upper:.2f}  Ausreißer oben: {n_high}")
print(series[series > upper].sort_values(ascending=False).head(5))
```
```r
vals <- df$kreatinin_mg_dl
q1 <- quantile(vals, 0.25, na.rm = TRUE)
q3 <- quantile(vals, 0.75, na.rm = TRUE)
upper <- q3 + 1.5 * (q3 - q1)
sum(vals > upper, na.rm = TRUE)
sort(vals[vals > upper], decreasing = TRUE) |> head(5)
```
**Befund:** ~15 Werte liegen oberhalb der Obergrenze (~3,37 mg/dl), der höchste
Wert bei ~5,6 mg/dl. Bei stationären Patient:innen klinisch plausibel: eher akutes
oder chronisches Nierenversagen als ein Datenfehler. Nicht ohne weiteres entfernen.

## Aufgabe 4 – Histogramm

```python
import matplotlib.pyplot as plt

order = (df.groupby("aufnahmegrund")["verweildauer_tage"]
           .median().sort_values(ascending=False).index.tolist())
fig, axs = plt.subplots(1, len(order), figsize=(15, 4), sharey=True)
for ax, grund in zip(axs, order):
    sub = df.loc[df["aufnahmegrund"] == grund, "verweildauer_tage"]
    ax.hist(sub, bins=15, color="#4C72B0", edgecolor="white", alpha=0.8)
    ax.set_title(grund, fontsize=8)
    ax.set_xlabel("Tage")
axs[0].set_ylabel("Anzahl Patient:innen")
fig.suptitle("Verweildauer je Aufnahmegrund")
plt.tight_layout()
fig.savefig("verweildauer_nach_aufnahmegrund.png", dpi=120)
```
```r
library(ggplot2)
order_lvl <- df |> group_by(aufnahmegrund) |>
  summarise(m = median(verweildauer_tage)) |>
  arrange(desc(m)) |> pull(aufnahmegrund)

p_los <- ggplot(df, aes(x = verweildauer_tage)) +
  geom_histogram(bins = 15, fill = "#4C72B0", colour = "white", alpha = 0.8) +
  facet_wrap(~ factor(aufnahmegrund, levels = order_lvl), nrow = 1) +
  labs(x = "Tage", y = "Anzahl", title = "Verweildauer je Aufnahmegrund") +
  theme_minimal(base_size = 9)
ggsave("verweildauer_nach_aufnahmegrund_r.png", p_los, width = 15, height = 4, dpi = 120)
```
**Befund:** Sepsis zeigt den höchsten Median (~6,5 Tage) und die breiteste
Verteilung mit Werten bis 20 Tage, Ausdruck schwerer Verläufe mit
Multiorganversagen. Alle Gruppen zeigen leichte Rechtsschiefe (wenige Langlieger).

## Aufgabe 5 – Boxplot

```python
import matplotlib.pyplot as plt
import seaborn as sns

order = (df.groupby("raucherstatus")["sofa_score"]
           .median().sort_values(ascending=False).index.tolist())
fig, ax = plt.subplots(figsize=(7, 4))
sns.boxplot(data=df, x="raucherstatus", y="sofa_score",
            order=order, hue="raucherstatus", palette="Set2", legend=False, ax=ax)
ax.set_xlabel("Raucherstatus")
ax.set_ylabel("SOFA-Score bei Aufnahme")
ax.set_title("SOFA-Score nach Raucherstatus")
fig.tight_layout()
fig.savefig("sofa_nach_raucherstatus.png", dpi=120)
```
```r
order_lvl <- df |> group_by(raucherstatus) |>
  summarise(m = median(sofa_score)) |>
  arrange(desc(m)) |> pull(raucherstatus)

p_sofa <- ggplot(df, aes(x = factor(raucherstatus, levels = order_lvl),
               y = sofa_score, fill = raucherstatus)) +
  geom_boxplot(outlier.size = 1.5, show.legend = FALSE) +
  scale_fill_brewer(palette = "Set2") +
  labs(x = "Raucherstatus", y = "SOFA-Score",
       title = "SOFA-Score nach Raucherstatus") +
  theme_minimal()
ggsave("sofa_nach_raucherstatus_r.png", p_sofa, width = 7, height = 4, dpi = 120)
```
**Befund:** Die Gruppen überlappen fast vollständig (Median 4 bei „nie“, 3 bei
„ehemalig“ und „aktiv“). Auch ein deutlicher Unterschied wäre noch kein
Kausalbefund: ohne Adjustierung, etwa für Alter und Aufnahmegrund, bleibt
Confounding möglich. EDA beschreibt, sie erklärt nicht.

## Aufgabe 6 – Streudiagramm und Korrelation

```python
import matplotlib.pyplot as plt
from scipy import stats

sub = df[["crp_mg_l", "verweildauer_tage"]].dropna()
r, p = stats.pearsonr(sub["crp_mg_l"], sub["verweildauer_tage"])
print(f"Pearson r = {r:.3f}, p = {p:.4f}")

fig, ax = plt.subplots(figsize=(6, 5))
colors = {0: "#4C72B0", 1: "#DD8452"}
for outcome, label in [(0, "überlebt"), (1, "verstorben")]:
    s = df[df["verstorben_30d"] == outcome]
    ax.scatter(s["crp_mg_l"], s["verweildauer_tage"],
               c=colors[outcome], label=label, alpha=0.4, s=18, edgecolors="none")
ax.set_xlabel("CRP (mg/l)")
ax.set_ylabel("Verweildauer (Tage)")
ax.set_title(f"CRP vs. Verweildauer (r={r:.2f}, p={p:.3f})")
ax.legend()
fig.savefig("crp_verweildauer_streudiagramm.png", dpi=120)
```
```r
sub <- df |> filter(!is.na(crp_mg_l), !is.na(verweildauer_tage))
r_val <- cor(sub$crp_mg_l, sub$verweildauer_tage)
cat(sprintf("Pearson r = %.3f\n", r_val))

p_crp <- ggplot(df |> filter(!is.na(crp_mg_l)),
       aes(x = crp_mg_l, y = verweildauer_tage,
           colour = factor(verstorben_30d, labels = c("überlebt","verstorben")))) +
  geom_point(alpha = 0.4, size = 1.5) +
  scale_colour_manual(values = c("#4C72B0", "#DD8452")) +
  labs(x = "CRP (mg/l)", y = "Verweildauer (Tage)",
       colour = NULL, title = "CRP vs. Verweildauer nach Mortalität") +
  theme_minimal()
ggsave("crp_verweildauer_streudiagramm_r.png", p_crp, width = 6, height = 5, dpi = 120)
```
**Befund:** Der Pearson-Koeffizient liegt nahe 0 (r ≈ −0,09) und ist bei diesem
Stichprobenumfang (N = 500) formal signifikant (p ≈ 0,047), aber praktisch
bedeutungslos (r² < 1 % erklärte Varianz). Das Streudiagramm bestätigt dies:
eine diffuse Punktewolke ohne erkennbares Muster. Ein Paradebeispiel dafür,
dass ein signifikantes p bei großem N nicht automatisch eine klinisch
relevante Aussage liefert, immer Effektgröße und Streudiagramm mitlesen.

## Bonus – Korrelationsmatrix

```python
numeric_cols = ["alter", "sofa_score", "crp_mg_l", "laktat_mmol_l",
                "verweildauer_tage", "leukozyten_g_l", "haemoglobin_g_dl",
                "kreatinin_mg_dl", "natrium_mmol_l", "verstorben_30d"]
corr = df[numeric_cols].corr().round(2)
print(corr["verstorben_30d"].drop("verstorben_30d")
          .sort_values(key=abs, ascending=False))
```
```r
numeric_cols <- c("alter","sofa_score","crp_mg_l","laktat_mmol_l",
                  "verweildauer_tage","leukozyten_g_l","haemoglobin_g_dl",
                  "kreatinin_mg_dl","natrium_mmol_l","verstorben_30d")
korr <- round(cor(df[numeric_cols], use = "pairwise.complete.obs"), 2)
sort(korr["verstorben_30d", -which(colnames(korr)=="verstorben_30d")],
     decreasing = TRUE)
```
**Befund:** `sofa_score` (r ≈ 0,36) korreliert mit Abstand am stärksten mit
`verstorben_30d`, gefolgt von `leukozyten_g_l` (r ≈ 0,24) und `alter`
(r ≈ 0,22). Nicht überraschend: der SOFA-Score bildet das Organversagen ab und ist
ein etablierter Prognosemarker, höheres Alter ist ein bekannter Risikofaktor, und
erhöhte Leukozyten sind ein klassisches Entzündungs-/Infektzeichen. Die übrigen
Labormarker (u. a. CRP, Kreatinin) zeigen kaum linearen Zusammenhang,
nichtlineare oder kombinierte Effekte werden erst in Modul 12 sichtbar.
