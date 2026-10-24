# 09 · Deskriptive Statistik und die Table 1

> **Teil 3 – Explorative Datenanalyse und deskriptive Statistik** · Dauer: ~60 min · Voraussetzung: Module 06, 08 (Transformation und EDA)

## Lernziele
- **Lage- und Streuungsmaße** (Mittelwert, Median, SD, IQR) erklären und situationsgerecht auswählen können.
- **Häufigkeiten und Anteile** für kategoriale Variablen berechnen.
- Eine klinische **„Table 1“** automatisiert erstellen und korrekt interpretieren.
- Wissen, **wann Mittelwert vs. Median** angemessen ist, und warum das klinisch einen Unterschied macht.
- Dasselbe Ergebnis in **`pandas` + `tableone`** (Python) und **`gtsummary`** (R) erzeugen.

## Klinischer Aufhänger

![Abbildung: Verteilungsvergleich von Alter, SOFA-Score und CRP nach 30-Tage-Mortalität. Verstorbene (rot) weisen bei Alter und SOFA-Score deutlich rechtsverschobene Verteilungen auf; bei CRP ist der Unterschied gering. Gepunktete Linien markieren die jeweiligen Mediane.](assets/baseline_nach_outcome.png)

Jedes klinische Paper beginnt mit einer „Table 1“: Alter, Geschlecht,
Komorbiditäten, Aufnahmestatus, getrennt nach Hauptgruppen (Überlebende vs.
Verstorbene). Sie zeigt, wen die Studie eingeschlossen hat und worin sich die
Gruppen unterscheiden; bei Gruppen nach Exposition (z. B. Therapie ja/nein) zeigt
sie außerdem, ob diese vergleichbar sind. Wir bauen sie einmal manuell, einmal
mit einem Paket.

## 1 Lagemaße: Mittelwert vs. Median

**Mittelwert** (arithmetisches Mittel): Summe aller Werte geteilt durch die
Anzahl. Reagiert empfindlich auf Ausreißer.

**Median**: Der mittlere Wert nach Sortierung. Robust gegenüber Ausreißern.

| Situation | Empfehlung |
|-----------|-----------|
| Normalverteilung, keine Ausreißer | Mittelwert ± SD |
| Schiefe Verteilung (z. B. Liegedauer, CRP) | Median [IQR] |
| Kategoriale Daten | n (%) |

**Klinisches Beispiel:** Krankenhausverweildauer. Wenige Patient:innen liegen
extrem lang, das zieht den Mittelwert nach oben, während der Median den typischen
Verlauf besser abbildet.

```python
import numpy as np
los = [3, 4, 5, 4, 3, 6, 4, 5, 28]   # 28 = long-stay outlier
print(f"Mittelwert: {np.mean(los):.1f}")    # 6.9  – misleadingly high
print(f"Median:     {np.median(los):.1f}")  # 4.0  – typical value
```

```r
los <- c(3, 4, 5, 4, 3, 6, 4, 5, 28)
cat(sprintf("Mittelwert: %.1f\n", mean(los)))    # 6.9
cat(sprintf("Median:     %.1f\n", median(los)))  # 4.0
```

**Stolperstein:** Viele Journals verlangen Mittelwert ± SD auch für schiefe
Variablen, reproduzierbar, aber klinisch irreführend. Prüfe die Author
Guidelines und ergänze, falls SD gefordert ist, zusätzlich Median [IQR].
**Darüber hinaus:** Bei sehr kleinen Gruppen (n < 10) sind weder Mittelwert
noch Median stabil, berichte dann besser alle Einzelwerte tabellarisch
und verzichte auf zusammenfassende Statistik.

## 2 Streuungsmaße: SD vs. IQR

**Standardabweichung (SD):** Wurzel aus der mittleren quadrierten Abweichung vom
Mittelwert, grob: die typische Abweichung vom Mittelwert.
Sinnvoll bei annähernder Normalverteilung. Berichtet als `Mittelwert ± SD`.

**Interquartilsabstand (IQR):** Abstand zwischen Q1 (25. Perzentil) und Q3
(75. Perzentil). Robust gegenüber Ausreißern. Berichtet als `Median [Q1; Q3]`.

```python
import pandas as pd

cohort = pd.read_csv("{{DATEN}}kohorte.csv")
age = cohort["alter"]
print(f"Alter: {age.mean():.1f} ± {age.std():.1f} Jahre (Mittelwert ± SD)")
print(f"Alter: {age.median():.0f} [{age.quantile(0.25):.0f}; {age.quantile(0.75):.0f}] Jahre (Median [IQR])")
```

```r
library(readr)
cohort <- read_csv("{{DATEN}}kohorte.csv", show_col_types = FALSE)
age <- cohort$alter
cat(sprintf("Alter: %.1f ± %.1f Jahre (Mittelwert ± SD)\n", mean(age), sd(age)))
cat(sprintf("Alter: %.0f [%.0f; %.0f] Jahre (Median [IQR])\n",
            median(age), quantile(age, 0.25), quantile(age, 0.75)))
```

**Stolperstein:** `sd()` in R (und `std()` in pandas) verwendet standardmäßig
die Bessel-Korrektur (Nenner n−1, nicht n), korrekt für Stichproben, aber für
die deskriptive Beschreibung einer Vollkohorte manchmal missverständlich.
`numpy.std()` rechnet dagegen standardmäßig mit Nenner n (`ddof=0`).
**Darüber hinaus:** Für ordinale Variablen wie `sofa_score` (klinisch 0–24,
in unserer Kohorte 0–11, ganzzahlig) ist Median [IQR] fast immer vorzuziehen,
beide vermeiden halbe Einheiten, während SD Nachkommastellen erzeugt, die eine
falsche Präzision suggerieren.

## 3 Häufigkeiten für kategoriale Variablen

Kategoriale Variablen werden als **absolute Häufigkeit n** und **relativer Anteil %**
berichtet.

```python
import pandas as pd

cohort = pd.read_csv("{{DATEN}}kohorte.csv")
# Absolute and relative frequencies
print(cohort["aufnahmegrund"].value_counts())
print(cohort["aufnahmegrund"].value_counts(normalize=True).mul(100).round(1))
```

```r
library(readr)
cohort <- read_csv("{{DATEN}}kohorte.csv", show_col_types = FALSE)
table(cohort$aufnahmegrund)
round(prop.table(table(cohort$aufnahmegrund)) * 100, 1)
```

**Stolperstein:** Ein `0/1`-Wert für `diabetes` sieht numerisch aus, pandas und R
mitteln ihn unbemerkt. Das Ergebnis (hier 0,19) entspricht zwar dem Anteil, wird
in einer Tabelle aber leicht als Mittelwert fehlinterpretiert. Deklariere ihn
immer explizit als Faktor/Kategorie oder nutze `value_counts()`.
**Darüber hinaus:** `pd.crosstab(cohort["raucherstatus"], cohort["diabetes"],
normalize="index")` liefert in einer Zeile eine zeilenweise normierte Kreuztabelle, ideal für Untergruppen-Anteile ohne eigene `groupby`-Logik.

## 4 „Table 1“ mit `tableone` (Python) / `gtsummary` (R)

Der vollständige Code steht in [`code/python.py`](code/python.py) und
[`code/r.R`](code/r.R). Hier die Kernidee:

```python
import pandas as pd
from tableone import TableOne

labs   = pd.read_csv("{{DATEN}}labor.csv")

df = pd.read_csv("{{DATEN}}kohorte.csv").merge(labs, on="patient_id", how="left")
df["geschlecht"] = df["geschlecht"].replace({"w": "weiblich"})

columns     = ["alter", "geschlecht", "aufnahmegrund", "diabetes",
               "hypertonie", "sofa_score", "crp_mg_l", "verweildauer_tage"]
categorical = ["geschlecht", "aufnahmegrund", "diabetes", "hypertonie"]

nonnormal   = ["sofa_score", "crp_mg_l", "verweildauer_tage"]   # Median [Q1,Q3]

table1 = TableOne(df, columns=columns, categorical=categorical, nonnormal=nonnormal,
                  groupby="verstorben_30d", pval=True, missing=True)
print(table1.tabulate(tablefmt="simple"))
```

```r
library(readr)
library(gtsummary)

cohort <- read_csv("{{DATEN}}kohorte.csv", show_col_types = FALSE)
labs   <- read_csv("{{DATEN}}labor.csv", show_col_types = FALSE)
df <- left_join(cohort, labs, by = "patient_id") |>
  mutate(
    verstorben_30d = factor(verstorben_30d, levels = c(0, 1),
                            labels = c("Überlebt", "Verstorben")),
    diabetes   = factor(diabetes,   levels = c(0, 1), labels = c("nein", "ja")),
    hypertonie = factor(hypertonie, levels = c(0, 1), labels = c("nein", "ja"))
  )

df |>
  select(alter, geschlecht, aufnahmegrund, diabetes,
         hypertonie, sofa_score, crp_mg_l, verweildauer_tage, verstorben_30d) |>
  tbl_summary(by = verstorben_30d,
              statistic = list(all_continuous()  ~ "{median} [{p25}; {p75}]",
                               all_categorical() ~ "{n} ({p}%)"),
              missing = "ifany") |>
  add_p() |>
  add_overall()
```

**Stolperstein:** `pval=True` in tableone (und `add_p()` in gtsummary) erzeugt
automatisch p-Werte, der Test wird jedoch still gewählt. tableone zeigt stetige
Variablen als Mittelwert (SD) und testet sie per t-Test/ANOVA, **außer** du
listest sie in `nonnormal`: dann Median [Q1,Q3] und Kruskal-Wallis. gtsummary
zeigt standardmäßig Median [Q1; Q3] und testet mit Wilcoxon/Kruskal-Wallis.
Kategoriale Variablen bekommen in beiden Chi-Quadrat (bei kleinen Zellen Fisher).
Prüfe die Testauswahl (`table1.tabulate()`, Fußnote in gtsummary), statt sie
blind zu übernehmen.
**Darüber hinaus:** `add_overall()` in gtsummary fügt in einer Zeile eine
Gesamtspalte hinzu, die die Kohortengröße auf einen Blick zeigt, Standard in
klinischen Publikationen.

**Stolperstein:** `laktat_mmol_l` steht in der Table 1 von `code/python.py` mit `pval=True`, obwohl
seine Missingness nicht zufällig ist, sie hängt vom `sofa_score` ab (Modul 14
zeigt das mit Zahlen). Der Gruppenvergleichs-p-Wert wird also nur über die
vollständigen Fälle berechnet, einer **selektierten** Teilstichprobe, nicht
über die gesamte Kohorte. Das ist hier vertretbar, weil `tableone`s
`missing=True` die Fehlquote (86 fehlend, 17,2 %) direkt daneben ausweist,
sodass Lesende die Selektion einschätzen können. Ein p-Wert auf einer Variable
mit informativer Missingness ist ohne die Fehlquote wertlos.
`bga_ph` steht nicht in dieser Table 1; nähme man es auf, läge der Fall
umgekehrt. Sein Fehlen hängt nur vom Outcome ab (`verstorben_30d`: fehlt bei
41 % der Verstorbenen, 16 % der Überlebenden, siehe Modul 14). *Innerhalb* jeder
Outcome-Spalte fehlt der pH dann rein zufällig, Median und Gruppenvergleich
bleiben unverzerrt. Verzerrt ist die **Gesamtspalte**, weil Verstorbene unter den
vollständigen Fällen unterrepräsentiert sind. Die Regel bleibt dieselbe: Bevor
eine Variable in eine Table 1 aufgenommen wird, prüfe, wovon ihre Missingness
abhängt (siehe Modul 14).

## 5 Qualitätsindikatoren und SPC (Statistical Process Control)

In der klinischen Versorgungssteuerung und im Qualitätsmanagement (QM) reichen einfache Tabellen oft nicht aus. Hier müssen Prozesse **über die Zeit** überwacht werden, um Veränderungen (z. B. nach der Einführung eines neuen Triage-Systems oder Hygienestandards) objektiv zu bewerten.

### Qualitätsindikatoren (QIs)
Klinische Qualitätsindikatoren messen die Qualität der Versorgung. Typische Beispiele sind:

- **Prozessindikatoren:** Mittlere Wartezeit in der Notaufnahme (in Minuten), Anteil der leitliniengerechten Antibiotika-Gaben binnen 1 Stunde bei Sepsis.
- **Ergebnisindikatoren:** 30-Tage-Wiederaufnahmerate, ZVK-assoziierte Infektionsraten pro 1000 Patiententage.

### Statistical Process Control (SPC)
SPC nutzt grafische Methoden, um zwischen zufälligem Rauschen (**Common Cause Variation**) und systematischen Änderungen (**Special Cause Variation**) zu unterscheiden. Das einfachste und nützlichste Werkzeug hierfür ist das **Run Chart** (Verlaufschart).

![Abbildung: SPC Run Chart (Verlaufschart) für die monatliche mittlere Wartezeit in einer Notaufnahme. Die gestrichelte graue Linie zeigt den Baseline-Median. Nach Monat 12 ist ein Shift von 12 aufeinanderfolgenden Punkten unter dem Median erkennbar, was auf eine erfolgreiche Prozessverbesserung hindeutet.](assets/spc_run_chart.png)

`code/python.py`/`code/r.R` (Abschnitt 6) reproduzieren dieses Run Chart aus
denselben Zahlen und speichern eine eigene Kopie unter `assets/`
(`spc_run_chart_demo.png`/`spc_run_chart_demo_r.png`).

### Sonderereignisse erkennen (Regeln für Run Charts)
Ein Prozess hat sich systematisch verändert, wenn eine der folgenden Regeln verletzt wird:

1. **Shift (Verschiebung):** ≥ 6 aufeinanderfolgende Punkte liegen alle oberhalb oder alle unterhalb des Medians.
2. **Trend:** ≥ 5 aufeinanderfolgende Punkte steigen kontinuierlich an oder fallen kontinuierlich ab.
3. **Astronomical Point (Ausreißer):** Ein einzelner Wert weicht extrem und offensichtlich ab (Sonderereignis, z. B. Systemausfall).

### In Python – Run Chart zeichnen

```python
import matplotlib.pyplot as plt
import numpy as np

# Waiting-time data (months 1-24)
months = np.arange(1, 25)
wait_times = [43.5, 47.2, 41.8, 46.0, 48.5, 42.1, 45.2, 49.0, 44.1, 46.5, 43.0, 45.5,
              34.5, 31.2, 33.8, 30.5, 29.1, 32.4, 35.0, 31.8, 30.0, 32.5, 33.1, 28.5]

median_baseline = np.median(wait_times[:12])

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(months, wait_times, "o-", color="#2A5C8A", label="Messwert")
ax.axhline(median_baseline, color="gray", linestyle="--", label="Baseline-Median")

# Highlight the shift (months 13-24 all lie below the median)
ax.plot(months[12:], wait_times[12:], "o-", color="#B5482E", label="Shift (Prozessverbesserung)")
ax.set_xlabel("Monat")
ax.set_ylabel("Wartezeit (Minuten)")
ax.legend()
plt.show()
```

### In R – Run Chart zeichnen

```r
library(tidyverse)

df_spc <- tibble(
  monat = 1:24,
  wartezeit = c(43.5, 47.2, 41.8, 46.0, 48.5, 42.1, 45.2, 49.0, 44.1, 46.5, 43.0, 45.5,
                34.5, 31.2, 33.8, 30.5, 29.1, 32.4, 35.0, 31.8, 30.0, 32.5, 33.1, 28.5)
)

median_baseline <- median(df_spc$wartezeit[1:12])

ggplot(df_spc, aes(x = monat, y = wartezeit)) +
  geom_line(color = "#2A5C8A") +
  geom_point(color = "#2A5C8A") +
  geom_hline(yintercept = median_baseline, linetype = "dashed", color = "gray") +
  # Colour the shift red
  geom_point(data = df_spc %>% filter(monat > 12), color = "#B5482E", size = 2.5) +
  labs(x = "Monat", y = "Wartezeit (Minuten)", title = "Wartezeit-Verlaufschart") +
  theme_minimal()
```

> **Wann du Hilfe holst.** Sobald aus der Table 1 formale Gruppenvergleiche mit p-Werten werden oder SPC-Kontrollgrenzen ein reales Qualitätssignal auslösen sollen (Intervention, Meldung), gehört die Wahl von Test, Korrektur und Regelbasis mit einer Statistikerin abgestimmt – eine deskriptive Tabelle ist noch kein Inferenzverfahren.

## Fallstricke und Merksätze
- **Mittelwert ± SD bei schiefen Daten ist irreführend.** CRP, Liegedauer,
 Laktat sind fast immer rechtsschief, stets Median [IQR] berichten.

- **p-Werte in Table 1 sind kein Ziel.** In RCTs sind Baseline-Unterschiede per
 Design zufällig, der Test ist sinnlos (CONSORT rät ab). In Beobachtungsstudien
 hängt p vor allem von n ab und sagt nicht, ob ein Unterschied als Confounder
 relevant ist; dafür taugen standardisierte Mittelwertdifferenzen (Modul 19).

- **Fehlende Werte immer ausweisen.** `missing=True` / `missing="ifany"` muss
 aktiviert sein, Lesende sollen wissen, worauf die Analyse beruht.

- **Kategoriale Variablen als solche deklarieren.** Ein gemittelter `0/1`-Wert
 für Diabetes ist keine klinische Aussage.

- Rundungsregel: Anteile auf **eine** Dezimalstelle, absolute Zahlen ganzzahlig.

## Selbstcheck
1. **Wann ist Median [IQR] dem Mittelwert ± SD vorzuziehen?** – Bei schiefer Verteilung oder Ausreißern (z. B. Liegedauer, CRP, Laktat); der Mittelwert wird durch Extremwerte verzerrt, der Median nicht.
2. **Was prüft der p-Wert in einer Table 1, und warum ist er dort oft irrelevant?** – Er misst, wie gut die beobachteten Unterschiede zu der Annahme „kein Unterschied in der Grundgesamtheit“ passen. In RCTs ist jeder Baseline-Unterschied per Design zufällig, der Test beantwortet also eine Frage, deren Antwort feststeht; in Beobachtungsstudien hängt p stark von n ab und sagt nicht, ob ein Unterschied das Ergebnis verzerren kann.
3. **Warum muss ein `0/1`-Wert für Diabetes explizit als kategoriale Variable deklariert werden?** – Sonst berechnen pandas/R einen Mittelwert (hier 0,19), der keinen klinischen Sinn ergibt; korrekt ist n (%).
