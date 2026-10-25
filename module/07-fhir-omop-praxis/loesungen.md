# Musterlösungen · Patientendaten-Extraktion via FHIR und OMOP

## Übung 1: FHIR-Abfrage für Vitalparameter

### Python-Lösung
```python
import requests
import pandas as pd

FHIR_URL = "https://hapi.fhir.org/baseR4"
LOINC_BP = "85354-9"

# 1. Send the API query for blood pressure observations. timeout and
#    raise_for_status() as in Module 04: a public test server can be slow or down.
response = requests.get(
    f"{FHIR_URL}/Observation",
    params={"code": LOINC_BP, "_count": 5},
    headers={"Accept": "application/json"},
    timeout=10,
)
response.raise_for_status()
bundle = response.json()

# 2. Extract the relevant fields
records = []
for entry in bundle.get("entry", []):
    resource = entry["resource"]
    obs_id = resource["id"]
    patient_ref = resource.get("subject", {}).get("reference", "unknown")
    date_str = resource.get("effectiveDateTime", "unknown")
    records.append({"obs_id": obs_id, "patient_ref": patient_ref, "zeitpunkt": date_str})

df_bp = pd.DataFrame(records)
print(df_bp)
```

### R-Lösung
```r
library(httr2)

# 1. Same query with httr2 (in the course package list); req_timeout as in Python
bundle <- request("https://hapi.fhir.org/baseR4") |>
  req_url_path_append("Observation") |>
  req_url_query(code = "85354-9", `_count` = 5) |>
  req_headers(Accept = "application/json") |>
  req_timeout(10) |>
  req_perform() |>
  resp_body_json()

# 2. Extract the relevant fields
field_or_unknown <- function(x) if (is.null(x)) "unknown" else x
df_bp <- do.call(rbind, lapply(bundle$entry, function(entry) {
  res <- entry$resource
  data.frame(obs_id = res$id,
             patient_ref = field_or_unknown(res$subject$reference),
             zeitpunkt = field_or_unknown(res$effectiveDateTime))
}))
print(df_bp)
```

Ist der Testserver nicht erreichbar, bricht der Aufruf nach zehn Sekunden mit
einer klaren Fehlermeldung ab. Zum Üben des Parsings taugt dann das
Beispiel-Bundle aus der Lektion (`{{DATEN}}fhir_bundle.json`), das genauso
aufgebaut ist. Für sehr verschachtelte Bundles nimmt dir das optionale Paket
`fhircrackr` die Schleife ab (`fhir_crack()`).

---

## Übung 2: OMOP-SQL-Abfrage für Geburtskohorten und Outcome

```sql
SELECT 
    p.person_id,
    p.year_of_birth,
    d.death_date
FROM person p
LEFT JOIN death d ON p.person_id = d.person_id
WHERE p.year_of_birth < 1960
ORDER BY p.year_of_birth ASC;
```
