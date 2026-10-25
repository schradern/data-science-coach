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
print(cohort.shape)                                    # (500, 17)
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
dim(cohort)  # 500 17
missing <- colSums(is.na(cohort))
print(missing[missing > 0])
```

## Aufgabe 2 – Nur bestimmte Spalten laden

```python
import pandas as pd
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
# Columns: id, dx, vitals_hr, vitals_map
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
from pathlib import Path
import pandas as pd

def load_api_or_cache(url: str, cache_file: str) -> pd.DataFrame:
    """Fetch from URL and cache the answer locally; on any error read the cache."""
    cache = Path(cache_file)
    # A custom User-Agent: some APIs answer urllib's default one with HTTP 403.
    request = urllib.request.Request(url, headers={"User-Agent": "data-science-course"})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            df = pd.json_normalize(json.load(response))
        df.to_csv(cache, index=False)              # keep a copy for offline re-runs
        return df
    except Exception as exc:
        if cache.exists():
            print(f"API unreachable ({exc}). Using local cache {cache}.")
            return pd.read_csv(cache)
        raise RuntimeError(f"API unreachable and no local cache {cache}: {exc}") from exc

url = "https://disease.sh/v3/covid-19/countries?allowNull=false"
df = load_api_or_cache(url, "api_cache.csv")
print(df.shape)

# Second call with a deliberately broken address: must fall back to the cache.
df_offline = load_api_or_cache(url.replace("disease.sh", "disease.invalid"), "api_cache.csv")
print(df_offline.shape)
```
```r
library(httr2)
library(jsonlite)

load_api_or_cache <- function(url, cache_file) {
  tryCatch({
    response <- request(url) |> req_timeout(5) |> req_perform()
    # flatten = TRUE unpacks nested fields, so write_csv() can save them.
    df <- fromJSON(resp_body_string(response), flatten = TRUE) |> tibble::as_tibble()
    readr::write_csv(df, cache_file)          # keep a copy for offline re-runs
    df
  }, error = function(e) {
    if (!file.exists(cache_file)) {
      stop("API unreachable and no local cache ", cache_file, ": ", conditionMessage(e))
    }
    message("API unreachable: ", conditionMessage(e), ". Using local cache ", cache_file, ".")
    readr::read_csv(cache_file, show_col_types = FALSE)
  })
}

url <- "https://disease.sh/v3/covid-19/countries?allowNull=false"
df <- load_api_or_cache(url, "api_cache.csv")
dim(df)

# Second call with a deliberately broken address: must fall back to the cache.
df_offline <- load_api_or_cache(sub("disease.sh", "disease.invalid", url, fixed = TRUE),
                                "api_cache.csv")
dim(df_offline)
```

Der Fallback liest absichtlich keine zweite Web-Adresse: Im Klinik-Netz ohne
externe Verbindung wäre auch die nicht erreichbar. Nur eine lokale Kopie der
letzten erfolgreichen Antwort macht das Skript wirklich offline-fähig.

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
**Beobachtung:** Alle 500 IDs kommen in `vitalwerte.csv` vor (die Differenzmenge
ist leer, `anti_join` liefert 0 Zeilen). Auffällig ist etwas anderes: Die Datei
enthält nur 1873 statt 2000 Messungen (4 × 500). Bei Patient:innen mit einer
Verweildauer unter vier Tagen fehlen die späteren Tage, sie waren dann schon
entlassen. Die ID-Prüfung allein hätte das nicht gezeigt; zähle auch die Zeilen
je Patient:in (`vitals.groupby("patient_id").size()` bzw. `count(vitals, patient_id)`).
