# Musterlösungen · Einführung und Lernpfad

> Vergleiche zuerst mit deinem eigenen Versuch. Es gibt oft mehrere richtige Wege.

## Aufgabe 1 – Datensatz erkunden

```python
import pandas as pd
cohort = pd.read_csv("{{DATEN}}kohorte.csv")
print(cohort.shape)          # (500, 16)
print(list(cohort.columns))
```
```r
library(readr)
cohort <- read_csv("{{DATEN}}kohorte.csv", show_col_types = FALSE)
cat(nrow(cohort), "x", ncol(cohort), "\n")
print(names(cohort))
```

## Aufgabe 2 – Mortalität nach Aufnahmegrund

```python
import pandas as pd
cohort = pd.read_csv("{{DATEN}}kohorte.csv")
cohort.groupby("aufnahmegrund")["verstorben_30d"].mean().sort_values(ascending=False).round(3)
```
```r
library(dplyr)
library(readr)
cohort <- read_csv("{{DATEN}}kohorte.csv", show_col_types = FALSE)
cohort |>
  group_by(aufnahmegrund) |>
  summarise(mortality = mean(verstorben_30d)) |>
  arrange(desc(mortality))
```
**Beobachtung:** Sepsis-Patient:innen haben erwartungsgemäß die höchste
30-Tage-Mortalität, eingebaut in die synthetische Wahrheit des Datensatzes
(vgl. `data/README.md`).

## Aufgabe 3 – Altersverteilung

```python
import pandas as pd
cohort = pd.read_csv("{{DATEN}}kohorte.csv")
print(f"Mittelwert: {cohort['alter'].mean():.1f} Jahre")
print(f"Median:     {cohort['alter'].median():.0f} Jahre")
```
```r
library(readr)
cohort <- read_csv("{{DATEN}}kohorte.csv", show_col_types = FALSE)
cat("Mittelwert:", round(mean(cohort$alter), 1), "Jahre\n")
cat("Median:    ", median(cohort$alter), "Jahre\n")
```
**Interpretation:** Liegen Mittelwert und Median nah beieinander, ist die
Verteilung annähernd symmetrisch. Ein deutlich höherer Mittelwert würde auf
Ausreißer nach oben (sehr alte Patient:innen) hinweisen.

## Aufgabe 4 – Lernpfad wählen
Keine Musterlösung, dies ist eine persönliche Reflexion. Orientierung:

- Hauptsächlich Auswertung eigener Studiendaten → „Schnell zur Auswertung"
- Daten aus KIS/REDCap extrahieren → „Fokus Datenextraktion"
- Paper schreiben / Table 1 erstellen → „Klinische Forschung / Publikation"

## Bonus

```python
from pathlib import Path

def check_datasets() -> None:
    """Check that all four core datasets are present locally."""
    data_dir = Path("data")
    for name in ["kohorte.csv", "vitalwerte.csv", "labor.csv", "notizen.csv"]:
        status = "OK" if (data_dir / name).exists() else "MISSING — run: python data/generate_data.py"
        print(f"  {name}: {status}")

check_datasets()
```
```r
check_datasets <- function() {
  for (name in c("kohorte.csv", "vitalwerte.csv", "labor.csv", "notizen.csv")) {
    path <- file.path("data", name)
    if (file.exists(path)) {
      cat("  OK:", name, "\n")
    } else {
      cat("  MISSING:", name, "— run: python data/generate_data.py\n")
    }
  }
}
check_datasets()
```
