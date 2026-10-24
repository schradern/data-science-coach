# Projektkompass – von der klinischen Frage zur belastbaren Analyse

Nutze diese Seite vor jedem eigenen Projekt. Sie verhindert die häufigsten
Fehlentscheidungen: unklare Kohorten, nachträglich gewählte Outcomes, falsche
Zeilenlogik, Leakage und überzogene Interpretation.

## 1 Klinische Frage schärfen

Formuliere eine Frage, die eine Entscheidung vorbereitet:

> Bei **welchen Patient:innen** interessiert mich **welcher Endpunkt**, in
> welchem **Zeitraum**, und soll ich **erklären**, **vorhersagen** oder
> **kausal vergleichen**?

| Ziel | Gute Frage | Passender Kursweg |
|------|------------|-------------------|
| Beschreiben | Wie sieht die Kohorte bei Aufnahme aus? | 06 → 08 → 09 |
| Testen | Unterscheidet sich Laktat zwischen Gruppen? | 08 → 09 → 10 → 21 |
| Erklären | Welche Faktoren sind adjustiert mit Mortalität assoziiert? | 10 → 12 → 15 |
| Kausal vergleichen | Wirkt Therapie A besser als B in Beobachtungsdaten? | 12 → 15 → 19 |
| Vorhersagen | Wer hat ein hohes Risiko für Ereignis X? | 23 → 24 → 25 |
| Einsetzen | Ist das Modell im Alltag nutzbar und sicher? | 25 → 32 → 34 |

## 2 Kohorte und Indexzeitpunkt festlegen

- **Einschluss:** Wer kann überhaupt in die Analyse?
- **Ausschluss:** Wer muss raus, bevor du das Outcome kennst?
- **Indexzeitpunkt:** Wann beginnt Beobachtung oder Vorhersage?
- **Zeilenlogik:** Eine Zeile pro Patient:in, pro Aufenthalt, pro Messung oder
  pro Zeitintervall?
- **Follow-up:** Wann endet Beobachtung: Entlassung, Tod, 30 Tage, Studienende?

> **Merksatz:** Wenn der Indexzeitpunkt schwimmt, schwimmt die ganze Analyse.

## 3 Datenwörterbuch anlegen

Vor dem ersten Modell braucht jede wichtige Variable eine kurze Spezifikation:

| Feld | Frage |
|------|-------|
| Name | Wie heißt die Variable im Datensatz? |
| Bedeutung | Was meint sie klinisch genau? |
| Einheit | mg/dl, mmol/l, Tage, Score, Kategorie? |
| Quelle | KIS, Labor, Medikation, Freitext, Register? |
| Zeitpunkt | Vor Index, bei Aufnahme, während Follow-up, nach Outcome? |
| Transformation | Binär, Kategorie, zentriert, log-transformiert? |
| Fehlend-Regel | Warum fehlt der Wert, und wie wird damit umgegangen? |

## 4 Leakage-Prüfung

Streiche oder verschiebe jede Information, die zum Vorhersage- oder
Analysezeitpunkt noch nicht verfügbar gewesen wäre.

- Laborwert nach Therapieentscheidung?
- Entlassdiagnose für ein Aufnahmemodell?
- Textnotiz nach Outcome?
- Imputation oder Skalierung vor Train/Test-Split?
- Feature-Selektion auf dem Gesamtdatensatz?

## 5 Methode wählen

Die Methode folgt aus Ziel, Outcome und Design:

| Situation | Startpunkt |
|-----------|------------|
| Stetiger Outcome, unabhängige Zeilen | Lineare Regression, robuste Diagnostik |
| Binärer Outcome | Logistische Regression, OR vorsichtig interpretieren |
| Zeit bis Ereignis | Kaplan-Meier, Cox, konkurrierende Risiken prüfen |
| Wiederholte Messungen | Mixed Model oder GEE |
| Beobachtungsdaten mit Therapieentscheidung | DAG, Propensity Score, Sensitivität |
| Vorhersagemodell | Split, Pipeline, AUC, Kalibrierung, Decision Curve |

## 6 Ergebnis als Satz prüfen

Schreibe vor der finalen Tabelle einen nüchternen Satz:

> In dieser retrospektiven synthetischen Kohorte war **X** mit **Y** assoziiert,
> adjustiert für **Z**; die Unsicherheit ist **...**. Daraus folgt **nicht**
> automatisch ein kausaler Effekt.

Für Vorhersage:

> Das Modell trennt Patient:innen mit und ohne Ereignis mit **AUC ...** und zeigt
> **Kalibrierung ...** im Testset. Klinischer Nutzen und Übertragbarkeit sind
> ohne externe/prospektive Validierung offen.

## 7 Minimaler Projektordner

```text
projekt/
├── data_raw/          # unveränderte Exporte, nicht überschreiben
├── data_processed/    # reproduzierbar erzeugte Analysedaten
├── code/              # 01_extract, 02_clean, 03_analyze
├── reports/           # Quarto/HTML/PDF
├── docs/              # Datenwörterbuch, Analyseplan
└── README.md          # Frage, Kohorte, Ausführung
```

## 8 Rote Flaggen

- Die primäre Fragestellung entsteht erst nach vielen Tests.
- Das Outcome ist nicht eindeutig datiert.
- Patient:innen erscheinen mehrfach, werden aber wie unabhängig behandelt.
- Fehlende Werte werden nur gelöscht, ohne Bias-Prüfung.
- Eine hohe AUC wird als klinischer Nutzen verkauft.
- Ein beobachteter Therapieeffekt wird kausal formuliert, ohne DAG oder Design.
