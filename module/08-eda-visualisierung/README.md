# 08 · Explorative Datenanalyse und Datenvisualisierung

> **Teil 3 – Explorative Datenanalyse und deskriptive Statistik** · Dauer: ~60 min · Voraussetzung: Modul 06 (Datenbereinigung und Datentransformation)

## Lernziele
- Daten mit **`describe()`**, **`value_counts()`** und einer **Korrelationsmatrix** systematisch erkunden.
- **Ausreißer** mit der IQR-Methode erkennen und klinisch einordnen.
- Das Prinzip der **Grammatik der Grafik** verstehen: Daten → Mapping → Geometrie.
- **Histogramm**, **Boxplot nach Gruppe** und **Streudiagramm** in Python (matplotlib/seaborn) und R (ggplot2) erstellen und richtig lesen.
- Verstehen, warum EDA **vor** jedem statistischen Test stehen muss.

## Klinischer Aufhänger
Deine Kohorte ist bereinigt und zusammengeführt. Bevor du einen Test rechnest,
willst du wissen: *Wie alt sind unsere Patient:innen? Welche Aufnahmegruppe
hat das höchste Laktat? Hängen CRP und Verweildauer zusammen?* Wer direkt mit dem
Test beginnt, übersieht Datenfehler, verletzt Verteilungsannahmen und erhält
falsche Ergebnisse.

Die Grafiken in diesem Kapitel liegen als fertige Dateien unter `assets/`.
Führst du [`code/python.py`](code/python.py) selbst aus, reproduziert es genau
diese drei Kapitelgrafiken (`verteilung_alter.png`,
`verteilung_laktat_nach_grund.png`, `streu_crp_verweildauer.png`) mit denselben
Variablen und derselben Trendgeraden nach `assets/`; die verbindlichen Versionen
erzeugt der gemeinsame Generator `data/figures.py`. [`code/r.R`](code/r.R)
schreibt inhaltsgleiche `_r`-Varianten daneben.

## 1 Grammatik der Grafik

Jede statistische Grafik folgt demselben Bauplan (*Grammar of Graphics*, Wilkinson 1999,
umgesetzt in ggplot2 und als Vorbild in seaborn):

| Schicht | Bedeutung | Beispiel |
|---------|-----------|---------|
| **Daten** | der DataFrame | `df` |
| **Mapping** (aesthetics) | welche Variable → welche Achse/Farbe | `x=alter, colour=verstorben_30d` |
| **Geometrie** | welche Form | Histogramm, Boxplot, Punkt |
| **Skala** | Achsentransformation, Farben | logarithmisch, kategoriale Farben |
| **Koordinaten und Thema** | Layout, Beschriftungen | `theme_minimal()`, Achsenlabels |

```python
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

labs   = pd.read_csv("{{DATEN}}labor.csv")

df = pd.read_csv("{{DATEN}}kohorte.csv").merge(labs, on="patient_id", how="left")
```

```r
library(tidyverse)

cohort <- read_csv("{{DATEN}}kohorte.csv", show_col_types = FALSE)
labs   <- read_csv("{{DATEN}}labor.csv", show_col_types = FALSE)

df <- left_join(cohort, labs, by = "patient_id")
```

**Stolperstein:** `matplotlib.use("Agg")` muss **vor** dem ersten
`import matplotlib.pyplot` stehen, nicht danach. Bei vertauschter Reihenfolge
erhältst du auf Servern einen `cannot connect to X server`-Fehler, der schwer zu
debuggen ist.
**Darüber hinaus:** Die Grammatik-Schichten lassen sich beliebig kombinieren, ein Histogramm mit überlagertem Dichteplot (`geom_histogram() + geom_density()`)
oder ein Streudiagramm mit Regressionslinie (`geom_smooth()`) kostet in ggplot2
nur eine weitere Zeile.

## 2 Numerische Zusammenfassung

`describe()` (pandas) und `summary()` (R) sind der erste Blick. Wonach du suchst:

- **Minimum / Maximum**, plausibel? Ein Alter von 180 ist ein Datenfehler.
- **Median vs. Mittelwert**, weit auseinander? Hinweis auf Schiefe oder Ausreißer.
- **Fehlende Werte**, `describe()` zeigt `count`, nicht den NaN-Anteil; extra prüfen!

```python
numeric_cols = ["alter", "bmi", "sofa_score", "crp_mg_l",
                "verweildauer_tage", "laktat_mmol_l", "kreatinin_mg_dl"]
print(df[numeric_cols].describe().round(2))

# Missing values separately — describe() hides them
print(df[numeric_cols].isna().sum())

# Frequencies for categorical variables
print(df["aufnahmegrund"].value_counts())
```

```r
numeric_cols <- c("alter", "bmi", "sofa_score", "crp_mg_l",
                  "verweildauer_tage", "laktat_mmol_l", "kreatinin_mg_dl")
summary(df[numeric_cols])
colSums(is.na(df))
df |> count(aufnahmegrund, sort = TRUE)
```

**Stolperstein:** `describe()` berechnet den Mittelwert auch für `0/1`-Spalten wie
`diabetes` und `verstorben_30d`. Das ist technisch korrekt (ergibt den Anteil),
aber im Kontext irreführend. Prüfe immer explizit, welche Variablen kategorial
sind, und werte diese separat mit `value_counts()` aus.
**Darüber hinaus:** `df.describe(include="all")` schließt kategoriale Spalten ein
und zeigt `top` (häufigster Wert) und `freq`, ein schneller Überblick über den
gesamten DataFrame in einer Zeile.

## 3 Ausreißer erkennen

Die **IQR-Methode** ist robust gegenüber extremen Werten: Werte außerhalb von
`[Q1 − 1,5 · IQR, Q3 + 1,5 · IQR]` gelten als potenzielle Ausreißer. *Achtung:*
Ein statistischer Ausreißer ist nicht automatisch ein Fehler, ein CRP von 320 mg/l
kann bei Sepsis klinisch real sein.

```python
series = df["crp_mg_l"].dropna()
q1, q3 = series.quantile(0.25), series.quantile(0.75)
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
n_out = ((series < lower) | (series > upper)).sum()
print(f"IQR={iqr:.1f}  Grenzen=[{lower:.1f}, {upper:.1f}]  Ausreißer: {n_out}")
```

```r
vals <- df$crp_mg_l
q1 <- quantile(vals, 0.25, na.rm = TRUE)
q3 <- quantile(vals, 0.75, na.rm = TRUE)
iqr <- q3 - q1
sum(vals < q1 - 1.5*iqr | vals > q3 + 1.5*iqr, na.rm = TRUE)
```

**Stolperstein:** Die IQR-Methode markiert bei normalverteilten Daten ~0,7 % der
Werte als Ausreißer, bei rechtsschiefen Variablen wie CRP oder Laktat deutlich
mehr. Nie blind entfernen: Werte immer anschauen und klinisch einordnen.
**Darüber hinaus:** Für eine schnelle Übersicht über viele Variablen gleichzeitig:
`df[numeric_cols].boxplot()` (pandas) oder `facet_wrap` mit `geom_boxplot` in R, Ausreißerpunkte springen sofort ins Auge.

## 4 Zusammenhänge, Korrelationsmatrix

```python
relevant = ["alter", "sofa_score", "crp_mg_l", "laktat_mmol_l",
            "verweildauer_tage", "verstorben_30d"]
print(df[relevant].corr().round(2))
```

```r
relevant <- c("alter", "sofa_score", "crp_mg_l", "laktat_mmol_l",
              "verweildauer_tage", "verstorben_30d")
round(cor(df[relevant], use = "pairwise.complete.obs"), 2)
```

Pearson-Korrelation misst **lineare** Zusammenhänge (−1 bis +1). Für ordinale oder
schiefe Variablen ist Spearmans ρ robuster: `df.corr(method="spearman")` (Python)
bzw. `cor(..., method = "spearman")` (R).

**Stolperstein:** Eine hohe Pearson-Korrelation zwischen zwei Laborwerten kann
eine gemeinsame Ursache widerspiegeln (z. B. Laktat und Kreatinin bei
Organversagen) statt eines direkten Zusammenhangs. r = 0 schließt zudem
einen U-förmigen Zusammenhang nicht aus.
**Darüber hinaus:** Eine Heatmap mit `seaborn.heatmap(df.corr(), annot=True,
cmap="coolwarm", center=0)` macht Muster in der Korrelationsmatrix auf einen
Blick sichtbar, besonders hilfreich bei mehr als sechs Variablen.

## 5 Histogramm, Verteilung einer Variablen

Das Histogramm teilt den Wertebereich in gleich breite Klassen (*bins*) und zählt,
wie viele Beobachtungen hineinfallen. Die Anzahl der bins prägt das Bild:
zu wenige verstecken Struktur, zu viele erzeugen Rauschen.

```python
fig, ax = plt.subplots(figsize=(7, 4))
for outcome, label, color in [
    (0, "Überlebt", "#2A5C8A"),
    (1, "Verstorben", "#B5482E"),
]:
    subset = df.loc[df["verstorben_30d"] == outcome, "alter"]
    ax.hist(subset, bins=20, alpha=0.65, label=label, color=color, edgecolor="white")
ax.set_xlabel("Alter (Jahre)")
ax.set_ylabel("Anzahl Patient:innen")
ax.set_title("Altersverteilung nach 30-Tage-Mortalität")
ax.legend()
fig.tight_layout()
fig.savefig("verteilung_alter.png", dpi=120)   # ohne savefig()/show() bleibt die Grafik unsichtbar!
```

```r
library(ggplot2)
p_hist <- ggplot(df, aes(x = alter,
               fill = factor(verstorben_30d, labels = c("Überlebt", "Verstorben")))) +
  geom_histogram(bins = 20, alpha = 0.65, position = "identity", colour = "white") +
  scale_fill_manual(values = c("Überlebt" = "#2A5C8A", "Verstorben" = "#B5482E")) +
  labs(x = "Alter (Jahre)", y = "Anzahl Patient:innen",
       title = "Altersverteilung nach 30-Tage-Mortalität", fill = NULL) +
  theme_minimal()
ggsave("verteilung_alter_r.png", p_hist, width = 7, height = 4, dpi = 120)
```

Erzeugte Grafik:

![Abbildung: Altersverteilung nach 30-Tage-Mortalität. Überlebende (blau) und Verstorbene (orange) überlappen sich stark, aber der Median der Verstorbenen liegt höher, ein erster Hinweis auf den Alterseffekt, der erst in der multivariablen Analyse quantifiziert wird.](assets/verteilung_alter.png)

**Stolperstein:** `position="identity"` in ggplot2 (und `alpha < 1` in matplotlib)
ist nötig, damit beide Histogramme übereinander sichtbar bleiben. Sonst entsteht ein
gestapeltes Histogramm, das Gruppenunterschiede verzerrt darstellt.
**Darüber hinaus:** `position="dodge"` (ggplot2) oder zwei separate Subplots mit
`sharey=True` (matplotlib) sind leichter zu lesen, wenn sich die Gruppen stark
unterscheiden, kein Überlappungsproblem, dafür direkter Vergleich nebeneinander.

## 6 Boxplot, Gruppenvergleich

Der Boxplot zeigt kompakt Median (Linie), IQR (Box) und Ausreißer (Punkte), ideal für **Gruppenvergleiche**, z. B. Laktat nach Aufnahmegrund.

```python
import seaborn as sns

order = (
    df.groupby("aufnahmegrund")["laktat_mmol_l"]
      .median()
      .sort_values(ascending=False)
      .index.tolist()
)
fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(data=df, x="aufnahmegrund", y="laktat_mmol_l",
            order=order, palette="Set2", ax=ax)
ax.set_xlabel("Aufnahmegrund")
ax.set_ylabel("Laktat (mmol/l)")
ax.set_title("Laktat nach Aufnahmegrund")
fig.tight_layout()
fig.savefig("verteilung_laktat_nach_grund.png", dpi=120)
```

```r
order_lvl <- df |>
  group_by(aufnahmegrund) |>
  summarise(m = median(laktat_mmol_l, na.rm = TRUE)) |>
  arrange(desc(m)) |>
  pull(aufnahmegrund)

p_box <- ggplot(df |> filter(!is.na(laktat_mmol_l)),
       aes(x = factor(aufnahmegrund, levels = order_lvl),
           y = laktat_mmol_l, fill = aufnahmegrund)) +
  geom_boxplot(outlier.size = 1.5, outlier.alpha = 0.5, show.legend = FALSE) +
  scale_fill_brewer(palette = "Set2") +
  labs(x = "Aufnahmegrund", y = "Laktat (mmol/l)",
       title = "Laktat nach Aufnahmegrund") +
  theme_minimal()
ggsave("verteilung_laktat_nach_grund_r.png", p_box, width = 8, height = 4, dpi = 120)
```

Erzeugte Grafik:

![Abbildung: Laktat nach Aufnahmegrund (Boxplot). Septische und herzinsuffiziente Patient:innen haben die höchsten Laktatwerte (die beiden Gruppen liegen dicht beieinander), deutlich über Pneumonie, ein klinisch plausibler Befund, der sich in Modul 10 statistisch prüfen lässt. Die Gruppen sind nach dem Median absteigend sortiert.](assets/verteilung_laktat_nach_grund.png)

**Stolperstein:** Der Boxplot zeigt keine Stichprobengröße. Gruppen mit n = 10
sehen genauso kompakt aus wie solche mit n = 200. Ergänze `n` je Gruppe als
Beschriftung oder prüfe es vorab mit `value_counts()`.
**Darüber hinaus:** Ein **Violin-Plot** (`geom_violin()` / `sns.violinplot()`)
zeigt zusätzlich zur Box die vollständige Verteilungsform, bei bimodalen oder
stark schiefen Daten informativer als der Boxplot allein.

## 7 Streudiagramm, Zusammenhang zweier Variablen

Das Streudiagramm (*scatter plot*) zeigt jede Beobachtung als Punkt; eine dritte
Variable lässt sich als Farbe kodieren. Das folgende Beispiel: CRP vs. Verweildauer,
farbkodiert nach 30-Tage-Mortalität.

```python
import numpy as np

fig, ax = plt.subplots(figsize=(7, 5))
colors = {0: "#2A5C8A", 1: "#B5482E"}
for outcome, label in [(0, "Überlebt"), (1, "Verstorben")]:
    sub = df[df["verstorben_30d"] == outcome]
    ax.scatter(sub["crp_mg_l"], sub["verweildauer_tage"],
               c=colors[outcome], label=label,
               alpha=0.42, s=22, edgecolors="none")

# Gesamt-Trendgerade (lineare Anpassung über alle Punkte)
mask = df["crp_mg_l"].notna() & df["verweildauer_tage"].notna()
coef = np.polyfit(df.loc[mask, "crp_mg_l"], df.loc[mask, "verweildauer_tage"], 1)
x_range = np.linspace(df.loc[mask, "crp_mg_l"].min(), df.loc[mask, "crp_mg_l"].max(), 200)
ax.plot(x_range, np.poly1d(coef)(x_range), color="#555555",
        linestyle="--", linewidth=1.1, alpha=0.7, label="Trend (gesamt)")

ax.set_xlabel("CRP (mg/l)")
ax.set_ylabel("Verweildauer (Tage)")
ax.set_title("CRP vs. Verweildauer nach 30-Tage-Mortalität")
ax.legend()
fig.tight_layout()
fig.savefig("streu_crp_verweildauer.png", dpi=120)
```

```r
p_scatter <- ggplot(df, aes(x = crp_mg_l, y = verweildauer_tage)) +
  geom_point(aes(colour = factor(verstorben_30d, labels = c("Überlebt", "Verstorben"))),
             alpha = 0.42, size = 1.5) +
  geom_smooth(method = "lm", formula = y ~ x, se = FALSE,
              colour = "#555555", linetype = "dashed", linewidth = 0.6) +
  scale_colour_manual(values = c("Überlebt" = "#2A5C8A", "Verstorben" = "#B5482E")) +
  labs(x = "CRP (mg/l)", y = "Verweildauer (Tage)",
       title = "CRP vs. Verweildauer nach 30-Tage-Mortalität",
       colour = NULL) +
  theme_minimal()
ggsave("streu_crp_verweildauer_r.png", p_scatter, width = 7, height = 5, dpi = 120)
```

Erzeugte Grafik:

![Abbildung: CRP vs. Verweildauer nach 30-Tage-Mortalität. Jeder Punkt ist eine Patient:in; Verstorbene (rot) verteilen sich über den gesamten CRP-Bereich ohne klaren Unterschied zu den Überlebenden. Die eingezeichnete Trendgerade zeigt nur einen sehr schwachen, leicht negativen Zusammenhang, erst die Korrelationsanalyse quantifiziert ihn.](assets/streu_crp_verweildauer.png)

**Stolperstein:** Bei 500 Punkten überlagern sich viele (*overplotting*). `alpha`
auf 0,3–0,5 setzen und `edgecolors="none"` (matplotlib) bzw. `size=1` (ggplot2)
verwenden. Alternativ: `geom_hex()` oder `geom_bin2d()` für sehr dichte Wolken.
**Darüber hinaus:** `sns.regplot()` oder `geom_smooth(method="lm")` fügt
automatisch eine Regressionsgerade mit Konfidenzband hinzu, der lineare
Trend wird sichtbar, ohne selbst zu rechnen.

## Fallstricke und Merksätze
- **EDA vor dem Test.** Wer direkt mit p-Werten anfängt, übersieht Datenfehler,
 Ausreißer und Verteilungsannahmen.

- **Bin-Anzahl matters.** 10 vs. 50 bins können zu gegensätzlichen Interpretationen
 führen. Immer mehrere Varianten prüfen.

- **Korrelation ≠ Kausalität.** Pearson misst nur *lineare* Zusammenhänge.
- **`matplotlib.use("Agg")` vor dem pyplot-Import** auf Servern ohne Anzeige.
- **Ausreißer zuerst verstehen.** Ein CRP von 400 kann ein echter Befund sein.

## Selbstcheck
1. **Was verrät ein großer Abstand zwischen Median und Mittelwert in `describe()`?**, Hinweis auf eine schiefe Verteilung oder Ausreißer; der Mittelwert wird in Richtung der Extremwerte gezogen.
2. **Warum ist ein statistischer Ausreißer per IQR-Methode nicht automatisch ein Datenfehler?**, Klinisch extreme Werte (z. B. CRP 320 mg/l bei Sepsis) können biologisch real sein; immer klinisch einordnen, bevor du ihn entfernst.
3. **Was misst Pearson-r, und wann ist Spearmans ρ robuster?**, Pearson misst lineare Zusammenhänge; Spearman (Rangkorrelation) ist robuster bei schiefen Verteilungen, Ausreißern oder ordinalen Variablen.
