# Lehrplan – Data Science für Ärztinnen und Ärzte

> Kurs-Architektur, Lernziele und Aufbau. Dies ist das zentrale Planungsdokument.
> Begleitend: [`KONVENTIONEN.md`](KONVENTIONEN.md) (Aufbau und Mitarbeit), [`../README.md`](../README.md) (Einstieg).

## Zielgruppe und Haltung

Ärztinnen und Ärzte sowie klinisch Forschende ohne Programmiervorerfahrung, die
eigene Daten extrahieren, aufbereiten und auswerten möchten. Der Kurs ist
**lernorientiert**, nicht referenzorientiert: jedes Modul folgt dem Bogen
*Motivation → Konzept → Code → eigene Übung → Lösung*.

Leitwerte (gelten für jedes Modul): **Wahrheit** (keine Statistik-Mythen),
**Transparenz** (jeder Schritt sichtbar), **Reproduzierbarkeit** (gleicher Code →
gleiches Ergebnis), **Zugänglichkeit** (klinische Beispiele, deutsche Sprache).

## Sprachstrategie

- **Python zuerst** – durchgehende Leitsprache, jedes Konzept zuerst in Python.
- **R parallel** – dasselbe Ergebnis idiomatisch in R (`tidyverse`), für die
 klinisch-statistische Community. Die Lektionen zeigen beide Wege nebeneinander,
 so weit die R-Spur reicht; wo sie kürzer ist, steht das in der Tabelle
 „R-Abdeckung“ unten.

- **SQL als Werkzeug** – fokussiert auf Datenextraktion (Modul 05: Datenbank-Abfragen mit SQL), später
 punktuell eingebunden.

## Aufbau in fünf Teilen

| Teil | Titel | Module | Frage, die der Teil beantwortet |
|------|-------|--------|----------------------------------|
| 1 | **Grundlagen und Setup** | 01–03 | *Wie richte ich eine reproduzierbare Arbeitsumgebung ein und schreibe ersten Code?* |
| 2 | **Datenimport, Datenbereinigung und Datenmanagement** | 04–07 | *Wie komme ich von Rohquellen zu einem sauberen, analysefertigen Datensatz?* |
| 3 | **Explorative Datenanalyse und deskriptive Statistik** | 08–09 | *Wie verstehe und beschreibe ich meine Daten, bevor ich teste?* |
| 4 | **Statistische Inferenz und medizinische Statistik** | 10–22 | *Wie ziehe ich belastbare Schlüsse, plane Studien und berichte publikationstauglich?* |
| 5 | **Machine Learning und KI in der Medizin** | 23–34 | *Wie baue, prüfe, erkläre und überwache ich klinische Modelle verantwortungsvoll?* |

Die klassische, „traditionelle“ Statistik (Inferenz, Regression, Studiendesign,
fehlende Werte, Kausalität, Diagnostik, Survival, Berichte) ist in **Teil 4**
zusammengeführt; **Machine Learning und KI in der Medizin** bilden als **Teil 5** den Abschluss.
So folgt der Kurs einer durchgehenden Kurve von den Grundlagen über die etablierte
Auswertung bis zu modernen, datengetriebenen Verfahren.

## Interaktionsprinzip

Die Website unterstützt Lernen statt Nachschlagen:

- **Lernkompass pro Modul:** Frage, Werkzeug, Ergebnis in drei kurzen Zeilen.
- **Sprachwahl:** Python, R oder beide; SQL bleibt sichtbar, wenn es relevant ist.
- **Zwei Spuren:** Grundlagen für Einsteiger:innen, Vertiefungen für Fortgeschrittene.
- **Aktive Erinnerung:** Selbstchecks sind aufdeckbare Fragekarten.
- **Fortschritt:** Module können abgehakt werden; die Startseite bietet den nächsten sinnvollen Einstieg.
- **Code-Sicherheit:** Jeder Codeblock erklärt auf Wunsch, wo er eingegeben wird.

## Modulübersicht

### Teil 1 – Grundlagen und Setup

**01 · Einführung und Lernpfad**
Warum Data Science in der Medizin? Der Datenanalyse-Zyklus, Kursüberblick,
Erwartungen, wie man mit dem Material arbeitet. *Ziel:* Orientierung und
realistische Erwartung. Kein Code-Zwang, erste Mini-Demo.

**02 · Werkzeuge und reproduzierbare Umgebung**
Installation Python (via `uv`), R + RStudio, VS Code, Git/GitHub. Projektordner,
virtuelle Umgebungen, erstes lauffähiges Skript, Beispieldaten erzeugen.
*Ziel:* lauffähige, isolierte, versionierte Umgebung.

**03 · Programmiergrundlagen in Python und R für Daten**
Variablen, Datentypen (bis Objekte/Klassen, CSV/JSON), Vektoren/Listen/Dictionaries,
Funktionen, Kontrollfluss, Tabellen (`DataFrame` / `data.frame`). Python und R
nebeneinander. *Ziel:* Code lesen und schreiben können.

### Teil 2 – Datenimport, Datenbereinigung und Datenmanagement

**04 · Datenimport aus Dateien und APIs**
CSV, Excel, JSON; Encodings/Trennzeichen; Web-APIs; Einstieg FHIR/REDCap.
*Ziel:* beliebige Quelldateien zuverlässig einlesen.

**05 · Datenbank-Abfragen mit SQL**
Relationales Denken, `SELECT/WHERE/JOIN/GROUP BY`, Aggregation, aus Python/R
heraus abfragen (SQLite/DuckDB). *Ziel:* Daten gezielt aus Datenbanken ziehen.

**06 · Datenbereinigung und Transformation**
Tidy-Data-Prinzip, fehlende Werte, Typen, Filtern/Mutieren/Gruppieren, Joins,
long/wide. `pandas` ↔ `dplyr`/`tidyr`. *Ziel:* analysefertige Datensätze.

**07 · Patientendaten-Extraktion via FHIR und OMOP**
FHIR-Ressourcen, OMOP-Denken, Patient:innen- und Laborwerte aus Schnittstellen
extrahieren, Forschungsdatenmodelle einordnen. *Ziel:* Daten nicht nur aus
Dateien, sondern aus klinischen Systemen strukturiert denken.

### Teil 3 – Explorative Datenanalyse und deskriptive Statistik

**08 · Explorative Datenanalyse und Visualisierung**
Verteilungen, Ausreißer, Zusammenhänge; Grammatik der Grafik. `matplotlib`/
`seaborn` ↔ `ggplot2`. *Ziel:* Daten verstehen, bevor man testet.

**09 · Deskriptive Statistik und die Table 1**
Lage- und Streumaße, Häufigkeiten, Baseline-Tabelle wie in klinischen Papern
(`tableone` / `gtsummary`). *Ziel:* die erste Tabelle jeder Publikation.

### Teil 4 – Statistische Inferenz und medizinische Statistik

Der komplette „traditionelle“ Auswertungs- und Publikationsweg: von
Hypothesentests über Regression und Studienplanung bis zum reproduzierbaren
Bericht – bevor in Teil 5 datengetriebene Verfahren dazukommen.

**10 · Inferenzstatistik und Hypothesentests**

**11 · Bayesianische Inferenz**
Stichprobe vs. Population, p-Wert und Konfidenzintervall *richtig* gelesen,
t-Test, χ², Mann-Whitney, multiples Testen. *Ziel:* Tests bewusst und korrekt
einsetzen.

**12 · Regressionsmodelle: lineare, logistische und Cox-Regression**
Lineare, logistische und Cox-Regression; Konfounder, Effektmaße (OR/HR),
Modelldiagnostik. `statsmodels`/`lifelines` ↔ `stats`/`survival`. *Ziel:*
adjustierte Zusammenhänge schätzen.

**13 · Studiendesign und Fallzahlplanung**
Klinische Frage, Analyseplan, primärer Endpunkt, Alpha, Power, minimale relevante
Differenz, Präzision und Events-per-Variable. *Ziel:* Studien planen, bevor Code
das Ergebnis schönrechnet.

**14 · Fehlende Werte und Imputation**
MCAR/MAR/MNAR, Missingness-Profile, wann Complete Case verzerrt und wann nicht,
einfache Imputation als Fehlerquelle, Multiple Imputation mit Rubins Regeln,
Sensitivitätsanalyse. *Ziel:* fehlende Daten nicht wegklicken, sondern begründet
behandeln – und wissen, welcher Schätzer dabei bricht.

**15 · Kausale Inferenz und Directed Acyclic Graphs**
DAGs, Confounder, Mediator, Kollider, crude vs. adjustiert, Propensity Score,
Standardized Mean Differences. *Ziel:* kausale Effektfragen von Vorhersage und
Assoziation trennen.

**16 · Diagnostische Genauigkeit und Schwellenwerte**
Sensitivität, Spezifität, PPV/NPV, Prävalenz, Likelihood Ratios, ROC, Youden,
klinische Schwellen und Net Benefit. *Ziel:* diagnostische Tests für reale
Entscheidungen bewerten.

**17 · Klassische Überlebenszeitanalyse**
Kaplan-Meier, Zensierung, Log-rank-Test, Cox-Modell, proportionale Hazards,
absolute Risiken und mediane Überlebenszeit. *Ziel:* Zeit-bis-Ereignis-Daten
publikationstauglich analysieren.

**18 · Mixed-Effects-Modelle für Longitudinaldaten**
Wiederholte Messungen, Random Intercept/Slope, GEE vs. Mixed Model. *Ziel:*
Patient:innenverläufe analysieren, ohne Unabhängigkeit vorzutäuschen.

**19 · Propensity Score Matching und Weighting**
Propensity Scores, Matching, IPW, Overlap und Balance via SMD. *Ziel:*
Selektionsbias in Beobachtungsstudien sichtbar reduzieren und ehrlich berichten.

**20 · Konkurrierende Risiken und zeitabhängige Cox-Modelle**
Cumulative Incidence, Fine-Gray, cause-specific Hazards, Start-Stop-Daten und
Immortal Time Bias. *Ziel:* Zeit-bis-Ereignis-Fragen ohne falsche Zensierung und
rückwirkende Expositionsfehler bearbeiten.

**21 · Auswahl der passenden statistischen Methode**
Messniveau, unabhängige vs. gepaarte Daten, zwei vs. mehrere Gruppen,
parametrisch vs. robust, Effektgröße und Reportingsatz. *Ziel:* den passenden
Test oder das passende Modell begründet wählen – der konsolidierende Kompass des
Statistik-Teils.

**22 · Reproduzierbare Berichte mit Quarto und Workflows**
Quarto / R Markdown, Projektstruktur, Seeds, Versionierung, Datenschutz-Hygiene.
*Ziel:* Analyse + Bericht aus einer Quelle, jederzeit nachvollziehbar – der
Abschluss des klassischen Auswertungswegs.

### Teil 5 – Machine Learning und KI in der Medizin

Python ist hier Leitsprache; R (`tidymodels`) läuft mit, wo es idiomatisch ist.
Der durchgehende rote Faden: ein **klinisches Risikomodell** sauber aufbauen,
ehrlich validieren, erklären und verantwortungsvoll einsetzen.

**23 · Einführung in das maschinelle Lernen**
Prädiktion vs. Erklärung, Train/Test, Über-/Unteranpassung, Validierung &
Kalibrierung, Bias/Fairness, Grenzen. `scikit-learn`. *Ziel:* ML einordnen und
seriös anwenden. Brücke aus dem Statistik-Teil in die KI-Methoden.

**24 · Workflow für Prädiktionsmodelle und Data Leakage**
Train/Validation/Test, Kreuzvalidierung, Data Leakage, `Pipeline`,
Hyperparameter-Tuning (Grid/Random, nested CV), Entscheidungsschwellen.
*Ziel:* ein lecksicheres, fair evaluiertes Grundgerüst.

**25 · Bewertung der Modellgüte und klinische Validierung**
Diskrimination (ROC/PR-AUC), Kalibrierung (Kalibrationskurve, Brier),
Decision-Curve-Analyse (Net Benefit), interne vs. externe Validierung,
Optimismus/Bootstrap, TRIPOD. *Ziel:* belegen, ob ein Modell klinisch nützt.

**26 · Baum-Ensembles und Gradient Boosting**
Bäume, Random Forest, XGBoost/LightGBM, Regularisierung, Umgang mit Fehlwerten.
*Ziel:* starke Tabellen-Modelle beherrschen.

**27 · Erklärbarkeit (Explainable AI)**
Permutation Importance, PDP/ICE, SHAP, global vs. lokal, Fallstricke. *Ziel:*
Modelle prüf- und kommunizierbar machen.

**28 · Maschinelles Lernen für Überlebenszeiten**
Random Survival Forests, zeitabhängige AUC, konkurrierende Risiken. *Ziel:*
Risiko über die Zeit lernen.

**29 · Unüberwachtes Lernen und Phänotypisierung von Patient:innen**
k-Means, hierarchisch, PCA, UMAP, Clustergüte. *Ziel:* Subgruppen ohne Labels
finden.

**30 · Deep Learning, Einstieg**
Neuronale Netze für Tabellen (MLP), CNN-Konzept für Bildgebung, wann DL hilft und
wann nicht. *Ziel:* DL realistisch einordnen.

**31 · Verarbeitung klinischer Freitexte mit LLMs**
NLP-Grundlagen, Embeddings, Klassifikation von Notizen, verantwortungsvolle
LLM-Nutzung in der Medizin. *Ziel:* mit unstrukturiertem Text arbeiten.

**32 · Modelleinsatz, Monitoring und Governance**
Modell speichern/bereitstellen, Model Cards, Kalibrierungs-Drift, Dataset Shift,
MDR und EU AI Act, Fairness. *Ziel:* vom Modell zum verantworteten klinischen
Werkzeug.

**33 · Retrieval-Augmented Generation und Leitlinien-Q&A**
Dokumente chunking-fähig machen, semantische Suche verstehen, Kontext in Prompts
geben, Halluzinationen begrenzen. *Ziel:* LLMs als quellengebundene Leser statt
als freie Ratgeber einsetzen.

**34 · Studiendesign zur Validierung klinischer KI-Systeme**
Retrospektive vs. prospektive Validierung, Silent Deployment, Automation Bias,
Cluster-Randomisierung, TRIPOD+AI/DECIDE-AI/CONSORT-AI. *Ziel:* prüfen, ob ein
KI-System Patient:innenversorgung verbessert, nicht nur Metriken.

## Lernpfade

Dies ist die kanonische Liste (sieben Pfade); Modul 01 zeigt dieselben Pfade
als Einstiegsorientierung.

- **Mein erstes Paper (statistischer Shortcut)** · 9 Module: 02–04 → 06 → 08–10 → 12 → 22
- **Schnell zur Auswertung** · 7 Module: 02–04 → 06 → 08–10
- **Datenextraktion im Fokus** · 6 Module: 02–07
- **Klinische Forschung / Publikation** · 16 Module: 06 → 08–22
  - Setzt Module 02–04 voraus (Umgebung, Grundlagen, Daten einlesen).
- **Statistik-Vertiefung** · 13 Module: 09–21
  - Setzt Module 02–04, 06 und 08 voraus, also den Pfad „Schnell zur Auswertung“ bis Modul 08.
- **Machine Learning / KI** · 12 Module: 23–34
  - Setzt Module 10 und 12 voraus (Inferenz, Regression), für Modul 28 zusätzlich Modul 17 (Survival).
- **Vollständig** · 34 Module: 01–34 in didaktischer Reihenfolge

Jeder Pfad enthält die Voraussetzungen seiner Module, die Vertiefungspfade nennen
ihr Vorwissen darunter. Welche Module ein Modul voraussetzt, steht oben in jeder Lektion.

## R-Abdeckung

Jedes Modul hat ein R-Skript (`code/r.R`). In den Modulen 01–11, 15–20, 22, 28,
32–34 rechnet es dieselben Analysen wie `code/python.py`. In den übrigen ist es
**teilweise** – der Kern läuft in R, einzelne Abschnitte nur in Python:

| Modul | fehlt in `code/r.R` |
|---|---|
| 12 Regression | Confounder-/Mediator-Vergleich (Abschnitt 2b) |
| 13 Studiendesign | Power bei beobachtetem n |
| 14 Fehlende Werte | Outcome im Imputationsmodell weglassen (5b) |
| 21 Testwahl | Cramérs V |
| 23 Machine Learning | Kalibrierungstabelle |
| 24 Prädiktions-Workflow | Leakage-Demo, Schwellenwahl |
| 25 Modellgüte | Rekalibrierung, Bootstrap-Optimismus |
| 26 Ensembles | Tiefen-Sweep, Lernraten-Tabelle, LightGBM |
| 27 Erklärbarkeit | Prädiktivitäts-Check (CV-AUC) |
| 29 Phänotypisierung | Benennung der Phänotypen |
| 30 Deep Learning | Lernkurve, Alpha-Sweep |
| 31 Klinische Texte | Klassifikationsbericht (Präzision/Recall/F1) |

**Musterlösungen nur in Python** gibt es in den Modulen 02, 05, 14, 15, 17, 21
und 24–32. Wer dort in R arbeitet, findet die passenden Schritte im R-Skript
des Moduls; die Aufgaben selbst lassen sich mit den Funktionen aus der Lektion
lösen.

## Durchgängiger Datensatz

Alle Module nutzen denselben synthetischen Kohortendatensatz (fiktive
Intensiv-/Stationskohorte, N = 500), erzeugt von
[`../data/generate_data.py`](../data/README.md) und geladen direkt über eine
Web-Adresse (Platzhalter `{{DATEN}}` in Codebeispielen) – ein lokaler Klon des
`data/`-Ordners ist zum Mitrechnen nicht nötig.

Vier Kern-Dateien bilden den Hauptdatensatz: `kohorte.csv` (eine Zeile je
Patient:in, inkl. der Ereigniszeit-Spalten `fu_zeit_tage`/`status` für
Survival-Module sowie der rein deskriptiven `verweildauer_tage`),
`vitalwerte.csv` (Verlaufsmessungen, long format), `labor.csv` (Laborwerte mit
**zwei verschiedenen MAR-Mechanismen**: `laktat_mmol_l` fehlt abhängig vom
`sofa_score`, `bga_ph` abhängig vom Outcome) und `notizen.csv` (klinischer
Freitext für NLP). Vier Begleitdateien für Modul 04 (Datenimport-Formate)
ergänzen den Datensatz: `kohorte_de.csv` (deutsches Klinik-Export-Format),
`einschluss.xlsx`, `beispiel.json` und `fhir_bundle.json`. Zwei Spezialdateien
kommen hinzu: `bga_ph_wahrheit.csv` (Ground Truth für Modul 14, nur möglich weil
die Kohorte synthetisch ist) und `konkurrenz_risiken.csv` (kohärenter
Konkurrenzrisiko-Prozess Tod vs. Entlassung plus zeitabhängige Exposition, Modul
20) – macht insgesamt **zehn** generierte Dateien.

Die 30-Tage-Mortalität liegt bei Seed 42 bei **≈ 15,6 %**. `alter` ist in der
zugrunde liegenden DAG ein **Confounder**, `sofa_score` ein **Mediator**
(siehe [`../data/README.md`](../data/README.md) für die vollständige
Kausalstruktur und die Ground-Truth-Effekte).

Fortgeschrittene Module verweisen zusätzlich auf reale offene Datensätze
(NHANES, MIMIC-IV Demo). Details: [`../data/README.md`](../data/README.md).

## Didaktische Standards (jedes Modul)

1. **Lernziele** explizit zu Beginn (3–5 Stichpunkte).
2. **Klinischer Aufhänger** – eine echte ärztliche Frage.
3. **Konzept vor Code** – kurze Erklärung, dann lauffähiges Beispiel.
4. **Python und R nebeneinander**: gleiches Ergebnis.
5. **Übungen** mit getrennten **Lösungen**.
6. **Fallstricke und Merksätze** am Ende.
7. **Reproduzierbar** – fester Seed, relativer Datenpfad, läuft ohne Internet
 (außer ausgewiesene API-Beispiele).
