# Musterlösungen · Datenbereinigung und Datentransformation

> Vergleiche zuerst mit deinem eigenen Versuch. Es gibt oft mehrere richtige Wege.

## Aufgabe 1 – Fehlende Werte

```python
import pandas as pd
k = pd.read_csv("{{DATEN}}kohorte.csv")
n = k["bmi"].isna().sum()
print(n, f"{n/len(k):.1%}")
```
```r
library(readr)
k <- read_csv("{{DATEN}}kohorte.csv", show_col_types = FALSE)
sum(is.na(k$bmi)); mean(is.na(k$bmi))
```
**Ergebnis:** 30 von 500 (6,0 %) fehlen.
**Entscheidung:** ~6 % fehlen, zufällig (MCAR). Für deskriptive Statistik
behalten und Fehlen ausweisen; für ein Modell, das `bmi` braucht, entweder
imputieren oder die Fälle dokumentiert ausschließen, nie stillschweigend.

## Aufgabe 2 – Filtern und Ableiten
```python
import pandas as pd
k = pd.read_csv("{{DATEN}}kohorte.csv")
hoch = (k.query("sofa_score >= 6")
         .assign(risikogruppe=lambda d: (d["alter"] >= 70).map({True:"hoch", False:"mittel"})))
```
```r
library(dplyr)
library(readr)
k <- read_csv("{{DATEN}}kohorte.csv", show_col_types = FALSE)
hoch <- k |> filter(sofa_score >= 6) |>
  mutate(risikogruppe = if_else(alter >= 70, "hoch", "mittel"))
```

## Aufgabe 3 – Gruppieren
```python
import pandas as pd
k = pd.read_csv("{{DATEN}}kohorte.csv")
k.groupby("raucherstatus").agg(
    verweildauer=("verweildauer_tage","mean"),
    mortalitaet=("verstorben_30d","mean")).round(3)
```
```r
library(dplyr)
library(readr)
k <- read_csv("{{DATEN}}kohorte.csv", show_col_types = FALSE)
k |> group_by(raucherstatus) |>
  summarise(verweildauer = mean(verweildauer_tage),
            mortalitaet  = mean(verstorben_30d))
```
**Beobachtung:** In dieser Kohorte zeigen aktive Raucher:innen die höchste
30-Tage-Mortalität (17,4 % vs. 16,1 % ehemalig, 14,4 % nie), bei den kleinen
Gruppengrößen kein Beweis, aber ein plausibler Befund, der sich formal testen
ließe (Modul 10, Hypothesentests).

## Aufgabe 4 – Join
```python
import pandas as pd
k = pd.read_csv("{{DATEN}}kohorte.csv")
labs = pd.read_csv("{{DATEN}}labor.csv")
df = k.merge(labs, on="patient_id", how="left")
df.groupby("diabetes")["kreatinin_mg_dl"].median()
```
```r
library(dplyr)
library(readr)
k <- read_csv("{{DATEN}}kohorte.csv", show_col_types = FALSE)
labs <- read_csv("{{DATEN}}labor.csv", show_col_types = FALSE)
df <- left_join(k, labs, by="patient_id")
df |> group_by(diabetes) |> summarise(krea = median(kreatinin_mg_dl, na.rm=TRUE))
```
**Ergebnis:** medianes Kreatinin ohne Diabetes ~1,0 mg/dl, mit Diabetes
1,1 mg/dl, minimal höher, konsistent mit diabetischer Nephropathie als
Risikofaktor, bei diesen kleinen Gruppengrößen aber nicht überinterpretieren.

## Aufgabe 5 – Long ↔ Wide
```python
import pandas as pd
v = pd.read_csv("{{DATEN}}vitalwerte.csv")
map_wide = v.pivot_table(index="patient_id", columns="tag", values="map_mmhg")
map_wide[3].notna().sum()   # Patient:innen mit Wert an Tag 3
```
```r
library(dplyr)
library(tidyr)
library(readr)
v <- read_csv("{{DATEN}}vitalwerte.csv", show_col_types = FALSE)
map_wide <- v |> select(patient_id, tag, map_mmhg) |>
  pivot_wider(names_from = tag, values_from = map_mmhg, names_prefix = "tag")
sum(!is.na(map_wide$tag3))
```
**Ergebnis:** 408 von 500 Patient:innen haben einen MAP-Wert an Tag 3 (die
übrigen wurden vorher entlassen oder verstarben, `vitalwerte.csv` hat pro
Patient:in nicht zwingend vier Messtage).

## Bonus
```python
import pandas as pd
k = pd.read_csv("{{DATEN}}kohorte.csv")
v = pd.read_csv("{{DATEN}}vitalwerte.csv")
mean_hr = v.groupby("patient_id")["herzfrequenz"].mean().rename("hf_mittel")
k2 = k.merge(mean_hr, on="patient_id", how="left")
```
```r
library(dplyr)
library(readr)
k <- read_csv("{{DATEN}}kohorte.csv", show_col_types = FALSE)
v <- read_csv("{{DATEN}}vitalwerte.csv", show_col_types = FALSE)
mean_hr <- v |> group_by(patient_id) |> summarise(hf_mittel = mean(herzfrequenz))
k2 <- left_join(k, mean_hr, by = "patient_id")
```
