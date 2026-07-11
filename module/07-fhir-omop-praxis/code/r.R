# Module 07 - parse a FHIR Bundle (Patient/Observation) into an analysis table.
#   Rscript module/07-fhir-omop-praxis/code/r.R

script <- normalizePath(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1]))
root <- dirname(dirname(dirname(dirname(script))))
source(file.path(root, "lib", "helpers.R"))

suppressPackageStartupMessages(library(jsonlite))

# Local-first, URL-fallback loader for the FHIR Bundle JSON -- mirrors the
# .load() pattern in lib/helpers.R (which only covers CSVs), so this script
# runs offline in the repo and online without a local clone.
load_fhir_bundle <- function() {
  path <- file.path(root, "data", "fhir_bundle.json")
  if (file.exists(path)) {
    return(fromJSON(path, simplifyVector = FALSE))
  }
  fromJSON(paste0(DATA_BASE_URL, "fhir_bundle.json"), simplifyVector = FALSE)
}

bundle <- load_fhir_bundle()

# 1. Ressourcen aus dem Bundle in zwei Listen trennen (Patient / Observation).
patient_rows <- list()
obs_rows <- list()
for (entry in bundle$entry) {
  res <- entry$resource
  if (res$resourceType == "Patient") {
    patient_rows[[length(patient_rows) + 1]] <- data.frame(
      patient_id = res$id,
      gender = if (!is.null(res$gender)) res$gender else NA,
      birth_date = if (!is.null(res$birthDate)) res$birthDate else NA,
      stringsAsFactors = FALSE
    )
  } else if (res$resourceType == "Observation") {
    obs_rows[[length(obs_rows) + 1]] <- data.frame(
      patient_id = sub("^Patient/", "", res$subject$reference),
      concept = res$code$text,
      value = res$valueQuantity$value,
      stringsAsFactors = FALSE
    )
  }
}
patients <- do.call(rbind, patient_rows)
observations <- do.call(rbind, obs_rows)

# 2. Observations von long (eine Zeile pro Messung) nach wide (eine Spalte pro
#    Konzept) drehen -- der Schritt, der aus verschachteltem FHIR-JSON eine
#    analysierbare Tabelle macht.
obs_wide <- reshape(observations, idvar = "patient_id", timevar = "concept",
                     direction = "wide")
names(obs_wide) <- sub("^value\\.", "", names(obs_wide))
names(obs_wide)[names(obs_wide) == "SOFA score"] <- "sofa_score"
names(obs_wide)[names(obs_wide) == "C-reactive protein"] <- "crp_mg_l"

analysis <- merge(patients, obs_wide, by = "patient_id", all.x = TRUE)

cat("=== Analysetabelle aus dem FHIR-Bundle (Patient + Observation) ===\n")
print(analysis, row.names = FALSE)

cat("\nOMOP-Gedanke: 'SOFA score' und 'C-reactive protein' sind hier Freitext\n")
cat("(code.text). Fuer eine multizentrische Analyse muesste jede Observation vor\n")
cat("dem Zusammenfuehren auf eine standardisierte Concept-ID (LOINC -> OMOP\n")
cat("measurement_concept_id) gemappt werden, nicht auf den Rohtext gefiltert.\n")
