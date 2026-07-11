# Musterlösungen · Datenimport aus Dateien und APIs

> Vergleiche zuerst mit deinem eigenen Versuch. Es gibt oft mehrere richtige Wege.

## Aufgabe 1 – CSV explizit einlesen

```python
import pandas as pd
cohort = pd.read_csv(
    "{{DATEN}}kohorte.csv",
    sep=",", encoding="utf-8", decimal=".",
    dtype={"patient_id": int, "diabetes": int,
           "hypertonie": int, "verstorben_30d": int},
    na_values=["", "NA", "N/A", "-"],
)
print(cohort.shape)                                    # (500, 16)
missing = cohort.isna().sum()
print(missing[missing > 0])                           # bmi, gewicht_kg have missing values
```
```r
library(readr)
cohort <- read_csv("{{DATEN}}kohorte.csv",
  col_types = cols(patient_id=col_integer(), diabetes=col_integer(),
                   hypertonie=col_integer(), verstorben_30d=col_integer()),
  na = c("", "NA", "N/A", "-"),
  show_col_types = FALSE)
dim(cohort)  # 500 16
missing <- colSums(is.na(cohort))
print(missing[missing > 0])
```

## Aufgabe 2 – Nur bestimmte Spalten laden

```python
labs = pd.read_csv("{{DATEN}}labor.csv",
                   usecols=["patient_id", "kreatinin_mg_dl", "laktat_mmol_l"])
print(labs.shape)  # (500, 3)
```
```r
library(readr)
labs <- read_csv("{{DATEN}}labor.csv",
  col_select = c(patient_id, kreatinin_mg_dl, laktat_mmol_l),
  show_col_types = FALSE)
dim(labs)  # 500 3
```

## Aufgabe 3 – JSON erzeugen und wieder einlesen

```python
import io, json, pandas as pd
cohort = pd.read_csv("{{DATEN}}kohorte.csv",
                     usecols=["patient_id", "alter", "aufnahmegrund"]).head(10)
json_str = cohort.to_json(orient="records", force_ascii=False)
df = pd.read_json(io.StringIO(json_str))
print(df.shape)  # (10, 3)
```
```r
library(jsonlite); library(readr); library(dplyr)
cohort <- read_csv("{{DATEN}}kohorte.csv", show_col_types=FALSE) |>
  select(patient_id, alter, aufnahmegrund) |> head(10)
json_str <- toJSON(cohort, pretty=FALSE)
df <- fromJSON(json_str) |> as_tibble()
dim(df)  # 10 3
```

## Aufgabe 4 – Verschachtelte Struktur normalisieren

```python
import json, pandas as pd
raw = '[{"id":1,"vitals":{"hr":82,"map":75},"dx":"Sepsis"},{"id":2,"vitals":{"hr":91,"map":68},"dx":"Pneumonie"},{"id":3,"vitals":{"hr":74,"map":82},"dx":"COPD-Exazerbation"}]'
df = pd.json_normalize(json.loads(raw), sep="_")
# Columns: id, vitals_hr, vitals_map, dx
print(df)
```
```r
library(jsonlite); library(dplyr)
raw <- '[{"id":1,"vitals":{"hr":82,"map":75},"dx":"Sepsis"},{"id":2,"vitals":{"hr":91,"map":68},"dx":"Pneumonie"},{"id":3,"vitals":{"hr":74,"map":82},"dx":"COPD-Exazerbation"}]'
df <- fromJSON(raw, flatten = TRUE) |> as_tibble()
# Columns: id, dx, vitals.hr, vitals.map
print(df)
```

## Aufgabe 5 – API-Fallback-Funktion

```python
import json
import urllib.request
import pandas as pd

def load_api_or_local(url: str, local_file: str) -> pd.DataFrame:
    """Fetch from URL; fall back to local file on any error."""
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return pd.DataFrame(json.load(response))
    except Exception as exc:
        print(f"API unreachable ({exc}). Using: {local_file}")
        return pd.read_csv(local_file)

df = load_api_or_local(
    "https://disease.sh/v3/covid-19/countries?allowNull=false&limit=5",
    "{{DATEN}}kohorte.csv"
)
print(df.shape)
```
```r
library(dplyr)
load_api_or_local <- function(url, local_file) {
  tryCatch({
    if (!requireNamespace("httr2", quietly=TRUE)) stop("httr2 missing")
    library(httr2); library(jsonlite)
    fromJSON(resp_body_string(request(url) |> req_timeout(5) |> req_perform())) |>
      as.data.frame()
  }, error = function(e) {
    message("API unreachable: ", e$message, ". Using: ", local_file)
    readr::read_csv(local_file, show_col_types=FALSE)
  })
}
df <- load_api_or_local(
  "https://disease.sh/v3/covid-19/countries?allowNull=false&limit=5",
  "{{DATEN}}kohorte.csv"
)
dim(df)
```

## Bonus – IDs vergleichen

```python
import pandas as pd
cohort  = pd.read_csv("{{DATEN}}kohorte.csv")
vitals  = pd.read_csv("{{DATEN}}vitalwerte.csv")
ids_cohort = set(cohort["patient_id"])
ids_vitals = set(vitals["patient_id"])
print("Patients without vitals:", ids_cohort - ids_vitals)
```
```r
library(readr); library(dplyr)
cohort <- read_csv("{{DATEN}}kohorte.csv", show_col_types=FALSE)
vitals <- read_csv("{{DATEN}}vitalwerte.csv", show_col_types=FALSE)
missing_vitals <- anti_join(cohort, vitals, by="patient_id")
cat("Patients without vitals:", nrow(missing_vitals), "\n")
```
**Beobachtung:** `vitalwerte.csv` enthält 1873 statt 2000 Messungen (4 × 500),
d. h. bei einigen Patient:innen fehlen Tage, klinisch realistisch (z. B.
Patient:in vor Tag 3 entlassen oder verstorben).
