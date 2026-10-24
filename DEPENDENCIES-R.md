# R-Pakete des Kurses

Das Gegenstück zu [`requirements.txt`](requirements.txt) für die R-Spur.

> **Reine R-Nutzung braucht kein Python.** Die Kurs-Datensätze liegen fertig
> unter `https://schradern.github.io/data-science-coach/data/`. Die Codeblöcke
> der Lektionen lesen sie direkt über diese Web-Adresse, und `lib/helpers.R`
> lädt sie bei Bedarf automatisch nach – du musst nichts erzeugen. Python wird
> **nur** gebraucht, wenn du das komplette Repository klonst und die Daten
> lokal neu erzeugen willst (`python data/generate_data.py`); zum Mitrechnen
> ist das nicht nötig.

> **Fehlt ein Paket, bricht das Skript laut ab.** `require_pkgs()` in
> `lib/helpers.R` nennt das fehlende Paket und beendet das Skript mit
> Exitcode 1, statt still ohne Ausgabe durchzulaufen.

## Installation

Kopiere den `install.packages(...)`-Block unten in deine R-Konsole – das
funktioniert immer, auch ohne das Repository. Wer den Kurs geklont hat, kann
stattdessen in einem Schritt dieselbe Liste installieren:

```r
source("tools/install_r_packages.R")
```

Oder von Hand – **Kernpfad** (Module 01–10, 12–25, 32, 34):

```r
install.packages(c(
  "tidyverse", "gtsummary", "broom", "survival",
  "DBI", "duckdb", "readxl", "writexl", "jsonlite", "httr2",
  "tidymodels", "pROC", "lme4", "sandwich", "lmtest", "mice", "cmprsk"
))
```

**Vertiefungen** (Module 26–31, 33 – der Kernpfad läuft ohne sie):

```r
install.packages(c(
  "xgboost", "vip", "pdp", "DALEX", "DALEXtra",
  "randomForestSRC", "timeROC", "umap", "factoextra", "cluster",
  "tidytext", "nnet", "recipes", "ranger"
))
```

**Nur für einzelne optionale Snippets** (nicht für die Skripte unter `code/`):
`fhircrackr` (Modul 07), `pwr` (Modul 10), `MatchIt` (Modul 19),
`rms`, `dcurves` und `PRROC` (Modul 25). Installiere sie, wenn du genau diesen
Abschnitt in R nachrechnen willst.

```r
install.packages(c("fhircrackr", "pwr", "MatchIt", "rms", "dcurves", "PRROC"))
```

## Pro Modul

| Modul | Pakete |
|-------|--------|
| 01 Einführung | `dplyr` |
| 02 Werkzeuge | `tidyverse` |
| 03 Grundlagen | `dplyr` |
| 04 Datenimport | `dplyr`, `readr`, `readxl`, `writexl`, `jsonlite`, `httr2` |
| 05 SQL | `DBI`, `duckdb`, `dplyr` |
| 06 Transformation | `tidyverse` |
| 07 FHIR / OMOP | `jsonlite` |
| 08 EDA | `tidyverse`, `ggplot2` |
| 09 Deskriptive Statistik | `tidyverse`, `gtsummary` |
| 10 Inferenzstatistik | `tidyverse` |
| 11 Bayes-Inferenz | Basis-R (`stats`) |
| 12 Regression | `tidyverse`, `survival` |
| 13 Studiendesign & Power | `tidyverse` |
| 14 Fehlende Werte | `tidyverse`, `mice` |
| 15 Kausale Inferenz | `tidyverse`, `sandwich`, `lmtest` |
| 16 Diagnostik & Schwellen | `tidyverse`, `pROC` |
| 17 Survival (klassisch) | `survival` |
| 18 Longitudinal / Mixed Models | `lme4` |
| 19 Propensity Score | `survival`, `sandwich`, `lmtest` |
| 20 Konkurrierende Risiken | `survival`, `cmprsk` |
| 21 Methodenwahl | `tidyverse` |
| 22 Reproduzierbare Berichte | `tidyverse` |
| 23 Machine Learning | `tidyverse`, `tidymodels` |
| 24 Prädiktions-Workflow | `tidyverse`, `tidymodels` |
| 25 Modellgüte & Validierung | `tidyverse`, `tidymodels`, `pROC` |
| 26 Ensembles & Boosting | `tidyverse`, `tidymodels`, `ranger`, `xgboost` |
| 27 Erklärbarkeit | `tidyverse`, `tidymodels`, `ranger`, `vip`, `pdp`, `DALEX`, `DALEXtra` |
| 28 Survival-ML | `tidyverse`, `survival`, `randomForestSRC`, `timeROC` |
| 29 Unüberwacht / Phänotypisierung | `tidyverse`, `recipes`, `cluster`, `factoextra`, `umap` |
| 30 Deep Learning | `tidyverse`, `tidymodels`, `nnet`, `xgboost` |
| 31 Klinische Texte / LLM | `tidyverse`, `tidymodels`, `tidytext` |
| 32 Einsatz & Governance | `tidyverse` |
| 33 RAG / LLM-Pipelines | – (reines Base R) |
| 34 Design von KI-Studien | – (reines Base R) |

## Versionen

Der Kurs pinnt R-Pakete bewusst **nicht** auf exakte Versionen (anders als
`requirements.txt` für Python). Wer exakte Reproduzierbarkeit braucht, legt ein
[`renv`](https://rstudio.github.io/renv/)-Projekt an:

```r
install.packages("renv")
renv::init()
renv::snapshot()   # erzeugt renv.lock
```

Der Kurscode ist gegen R ≥ 4.2 getestet (nutzt die native Pipe `|>`).
