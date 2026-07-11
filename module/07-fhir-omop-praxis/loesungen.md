# Musterlösungen · Patientendaten-Extraktion via FHIR und OMOP

## Übung 1: FHIR-Abfrage für Vitalparameter

### Python-Lösung
```python
import requests
import pandas as pd

FHIR_URL = "https://hapi.fhir.org/baseR4"
LOINC_BP = "85354-9"

# 1. API-Abfrage für Blutdruck-Observations absenden
response = requests.get(
    f"{FHIR_URL}/Observation?code={LOINC_BP}&_count=5",
    headers={"Accept": "application/json"}
)
bundle = response.json()

# 2. Relevante Felder extrahieren
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
library(fhircrackr)

search_request <- fhir_url(
  url = "https://hapi.fhir.org/baseR4",
  resource = "Observation",
  parameters = c("code" = "85354-9", "_count" = "5")
)

bundle <- fhir_search(search_request)

obs_desc <- fhir_table_description(
  resource = "Observation",
  cols = list(
    obs_id = "id",
    patient_ref = "subject/reference",
    zeitpunkt = "effectiveDateTime"
  )
)

df_bp <- fhir_crack(bundle, obs_desc)
print(df_bp)
```

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
