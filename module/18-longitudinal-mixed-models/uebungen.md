# Übungen · Mixed-Effects-Modelle für Longitudinaldaten

## Übung 1: LMM-Spezifikation (Python oder R)
Simuliere oder lade einen longitudinalen Datensatz zur Atemfrequenz (`respiratory_rate`) von Patient:innen nach einer Operation über 4 Stunden (Stunden 1 bis 4).

1. Formuliere ein Random-Intercept-Modell: Die Atemfrequenz soll durch die Zeit (`zeit_stunden`) und das Alter (`alter`) vorhergesagt werden.
2. Jede:r Patient:in (`patient_id`) soll einen eigenen Random Intercept erhalten.
3. Interpretiere den Koeffizienten für `zeit_stunden`.

## Übung 2: OLS vs. LMM (Konzept)
Entscheide für die folgenden klinischen Szenarien, ob eine klassische lineare Regression (OLS) zulässig ist, oder ob ein gemischtes Modell (LMM) benötigt wird:

1. Vergleich der HDL-Cholesterinwerte zwischen einer Interventionsgruppe (Medikament) und einer Kontrollgruppe (Placebo) zu einem einzigen Messzeitpunkt (nach 12 Wochen).
2. Untersuchung des Schmerzscores (VAS, gemessen stündlich über 12 Stunden postoperativ) bei 30 Patient:innen nach Knie-TEP.
3. Analyse des Zusammenhangs von HbA1c und Nierenfunktion (eGFR) bei Patient:innen aus 10 verschiedenen Hausarztpraxen (wobei Patient:innen innerhalb einer Praxis ähnlicher sein könnten).

## Übung 3: Warum OLS die Standardfehler unterschätzt (Design-Effekt)
Ein Studienteam misst den mittleren arteriellen Druck (MAP) an 4 aufeinanderfolgenden Tagen bei 50 Intensivpatient:innen (jede:r vollständig 4-mal gemessen, also 200 Zeilen). Untersucht wird der Effekt von `diabetes` – einem **patientenkonstanten** Prädiktor (gleicher Wert in allen 4 Zeilen einer Person). Die Intraklassen-Korrelation (ICC) der wiederholten MAP-Messungen beträgt 0,20.

1. Berechne den Design-Effekt (nach Kish) `DEFF = 1 + (m − 1) · ICC` für m = 4 Messungen pro Person.
2. Wie groß ist die *effektive* Zahl unabhängiger Beobachtungen `N_eff = (N · m) / DEFF`?
3. Eine naive OLS behandelt alle 200 Zeilen als unabhängig. Um welchen Faktor ist ihr Standardfehler für den Diabetes-Koeffizienten zu klein? Begründe, warum ausgerechnet ein patientenkonstanter Prädiktor besonders betroffen ist.

## Übung 4: ICC aus den Varianzkomponenten
Der `summary()`-Ausdruck eines Random-Intercept-Modells auf MAP-Daten zeigt zwei Varianzkomponenten:

- Varianz des Random Intercept (Group Var, zwischen Patient:innen): τ² = 9,0 mmHg²
- Residualvarianz (innerhalb Patient:in, über die Zeit): σ² = 66,0 mmHg²

1. Berechne die Intraklassen-Korrelation `ICC = τ² / (τ² + σ²)`.
2. Wie groß ist die Standardabweichung der Random Intercepts, und was bedeutet sie klinisch?
3. Angenommen, ein anderes Modell liefert τ² ≈ 0. Was folgt daraus für die Wahl zwischen OLS und Mixed Model?
