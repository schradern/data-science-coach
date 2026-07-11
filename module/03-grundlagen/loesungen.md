# Musterlösungen · Programmiergrundlagen in Python und R

> Vergleiche zuerst mit deinem eigenen Versuch. Es gibt oft mehrere richtige Wege.

## Aufgabe 1 – Datentypen identifizieren

```python
crp_value      = 22.4          # float
sofa_score     = 8             # int
admission      = "Pneumonie"   # str
ventilated     = True          # bool

for name, val in [("crp_value", crp_value), ("sofa_score", sofa_score),
                  ("admission", admission), ("ventilated", ventilated)]:
    print(f"{name}: {val}  ({type(val).__name__})")
```
```r
crp_value  <- 22.4          # numeric
sofa_score <- 8L            # integer
admission  <- "Pneumonie"   # character
ventilated <- TRUE          # logical

for (v in list(crp_value, sofa_score, admission, ventilated)) {
  cat(v, " (", class(v), ")\n", sep = "")
}
```

## Aufgabe 2 – Vektoren und Grundrechenarten

```python
creatinine = [0.9, 1.4, 2.1, 0.8, 3.7, 1.1, 1.6]

print("Minimum:", min(creatinine))          # 0.8
print("Maximum:", max(creatinine))          # 3.7
print(f"Mean: {sum(creatinine)/len(creatinine):.2f}")  # 1.66

high = [k for k in creatinine if k > 2.0]
print("Creatinine > 2.0:", high)            # [2.1, 3.7]
print(f"Fraction: {len(high)/len(creatinine):.0%}")    # 29%
```
```r
creatinine <- c(0.9, 1.4, 2.1, 0.8, 3.7, 1.1, 1.6)

cat("Minimum:", min(creatinine), "\n")
cat("Maximum:", max(creatinine), "\n")
cat(sprintf("Mean: %.2f\n", mean(creatinine)))

high <- creatinine[creatinine > 2.0]
cat("Creatinine > 2.0:", paste(high, collapse = ", "), "\n")
cat(sprintf("Fraction: %.0f%%\n", mean(creatinine > 2.0) * 100))
```
**Hinweis:** ~29 %, klinisch relevant; Kreatinin > 2,0 ist ein grober Marker
für behandlungspflichtige Niereninsuffizienz (abhängig von Baseline und Alter).

## Aufgabe 3 – Eigene Funktion schreiben

```python
def is_tachycardic(heart_rate: float) -> bool:
    """Return True when heart rate exceeds 100 bpm."""
    return heart_rate > 100

hr_values   = [72, 88, 115, 54, 103, 98, 122]
tachycardic = [hr for hr in hr_values if is_tachycardic(hr)]
print("Tachycardic:", tachycardic)          # [115, 103, 122]
print(f"Count: {len(tachycardic)}")         # 3
```
```r
is_tachycardic <- function(heart_rate) {
  heart_rate > 100
}

hr_values <- c(72, 88, 115, 54, 103, 98, 122)
cat("Tachycardic:", hr_values[is_tachycardic(hr_values)], "\n")  # 115 103 122
cat("Count:", sum(is_tachycardic(hr_values)), "\n")               # 3
```

## Aufgabe 4 – Kontrollfluss

```python
def assess_map(map_mmhg: float) -> str:
    """Classify mean arterial pressure by clinical thresholds."""
    if map_mmhg < 65:
        return "Schock-Grenzwert unterschritten"
    elif map_mmhg <= 90:
        return "Normbereich"
    else:
        return "Erhöht"

for val in [58, 72, 95, 63, 88]:
    print(f"  MAP {val:3d} mmHg -> {assess_map(val)}")
```
```r
library(dplyr)
assess_map <- function(map_mmhg) {
  case_when(
    map_mmhg < 65  ~ "Schock-Grenzwert unterschritten",
    map_mmhg <= 90 ~ "Normbereich",
    .default       = "Erhöht"
  )
}

values  <- c(58, 72, 95, 63, 88)
results <- assess_map(values)
for (i in seq_along(values)) {
  cat(sprintf("  MAP %3d mmHg -> %s\n", values[i], results[i]))
}
```
**Klinischer Bezug:** MAP < 65 mmHg ist der Schwellenwert der Sepsis-Leitlinien
für Flüssigkeits- und Vasopressortherapie (Surviving Sepsis Campaign).

## Aufgabe 5 – DataFrame-Basics

```python
import pandas as pd
cohort = pd.read_csv("{{DATEN}}kohorte.csv")

# a) Age > 70
print("Age > 70:", (cohort["alter"] > 70).sum())

# b) Mean stay length with hypertension
print("Mean stay (hypertension):",
      cohort[cohort["hypertonie"] == 1]["verweildauer_tage"].mean().round(1))

# c) High CRP
cohort["high_crp"] = cohort["crp_mg_l"] > 50
print("High CRP (> 50 mg/l):", cohort["high_crp"].sum())
```
```r
library(dplyr)
library(readr)
cohort <- read_csv("{{DATEN}}kohorte.csv", show_col_types = FALSE)

# a) Age > 70
cat("Age > 70:", sum(cohort$alter > 70), "\n")

# b) Mean stay with hypertension
cat("Mean stay (hypertension):",
    round(mean(cohort$verweildauer_tage[cohort$hypertonie == 1]), 1), "\n")

# c) High CRP
cohort <- cohort |> mutate(high_crp = crp_mg_l > 50)
cat("High CRP (> 50 mg/l):", sum(cohort$high_crp), "\n")
```

## Bonus – z-Score des Alters

```python
import pandas as pd
cohort = pd.read_csv("{{DATEN}}kohorte.csv")
mean_age = cohort["alter"].mean()
std_age  = cohort["alter"].std()
cohort["age_z"] = (cohort["alter"] - mean_age) / std_age

print(cohort["age_z"].describe().round(2))
# z-values are typically between -3 and +3.
# |z| > 2 are outlier candidates (~5 % of a normal distribution).
```
```r
library(dplyr)
library(readr)
cohort <- read_csv("{{DATEN}}kohorte.csv", show_col_types = FALSE)
cohort <- cohort |> mutate(age_z = (alter - mean(alter)) / sd(alter))
print(summary(cohort$age_z))
# mean ≈ 0, sd ≈ 1 — as expected.
```
**Erklärung:** Der z-Score misst, wie viele Standardabweichungen ein Wert vom
Mittelwert entfernt liegt. In einer Normalverteilung liegen ≈ 95 % aller Werte
zwischen –2 und +2. Das Werkzeug ist nützlich, um Ausreißer zu identifizieren
und Variablen verschiedener Einheiten vergleichbar zu machen.
