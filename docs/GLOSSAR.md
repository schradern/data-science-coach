# Glossar

Kurze Begriffe für den Kurs. Nicht zum Auswendiglernen, sondern zum schnellen
Nachsehen, wenn ein Wort stört.

## Arbeiten mit Code

| Begriff | Kurz erklärt |
|---|---|
| Terminal | Textfenster für Befehle. Du startest dort Skripte, installierst Pakete und wechselst Ordner. |
| Shell | Das Programm im Terminal, das Befehle versteht, z. B. zsh, bash oder PowerShell. |
| Pfad | Adresse einer Datei oder eines Ordners, z. B. `data/kohorte.csv`. |
| Projektordner | Der gemeinsame Arbeitsordner. Relative Pfade beziehen sich auf ihn. |
| Paket | Zusatzbibliothek, z. B. `pandas`, `scikit-learn` oder `tidyverse`. |
| Virtuelle Umgebung | Isolierter Python-Bereich pro Projekt, damit Paketversionen sich nicht mischen. |
| Skript | Datei mit Code, die von oben nach unten ausgeführt wird. |
| Notebook | Interaktives Dokument aus Text, Code und Ausgaben. Gut zum Erkunden, riskanter für finale Analysen. |
| Seed | Feste Startzahl für Zufall. Gleicher Seed, gleiche Zufallsstichprobe. |

## Daten

| Begriff | Kurz erklärt |
|---|---|
| DataFrame | Tabelle im Code: Zeilen sind Fälle, Spalten sind Variablen. |
| Variable | Merkmal oder benannter Wert, z. B. `alter` oder `crp_mg_l`. |
| Datentyp | Art eines Werts: Zahl, Text, Wahrheitswert, Datum, fehlend. |
| Fehlender Wert | Nicht erhoben oder unbekannt. Nicht dasselbe wie `0`. |
| Tidy Data | Jede Variable eine Spalte, jede Beobachtung eine Zeile, jede Beobachtungseinheit eine eigene Tabelle (Wickham 2014; spätere Fassungen formulieren die dritte Regel als „jeder Wert eine Zelle“). Modul 06. |
| Join | Tabellen über Schlüsselspalten verbinden, z. B. über `patient_id`. |
| Long Format | Mehr Zeilen, weniger Messwertspalten; gut für wiederholte Messungen. |
| Wide Format | Mehr Spalten, weniger Zeilen; gut für Übersichten und manche Modelle. |
| Kohorte | Definierte Gruppe von Patient:innen, die gemeinsam analysiert wird. |
| FHIR | Standard für den Austausch klinischer Daten, z. B. Patient, Encounter, Observation. |
| OMOP-CDM | Gemeinsames Datenmodell für Forschungsanalysen über verschiedene Datenquellen hinweg. |
| Pseudonymisierung | Personenbezug wird durch Schlüssel getrennt, bleibt aber theoretisch wiederherstellbar. |

## Statistik

| Begriff | Kurz erklärt |
|---|---|
| PICO/PECO | Struktur für klinische Fragen: Population, Intervention/Exposure, Comparison, Outcome. |
| Effektmaß | Größe eines Unterschieds oder Zusammenhangs, z. B. Odds Ratio oder Hazard Ratio. |
| Power | Wahrscheinlichkeit, einen echten Effekt einer festgelegten Größe statistisch zu entdecken. |
| EPV (Ereignisse pro Parameter) | Faustregel für Regressionsmodelle: mindestens 10–15 Ereignisse pro geschätztem Parameter, um Overfitting und instabile Schätzer zu vermeiden. Sinkt die EPV darunter, werden Koeffizienten und Konfidenzintervalle unzuverlässig (Modul 12). |
| Firth-Korrektur | Penalisiert die Likelihood der logistischen Regression mit einem Jeffreys-Prior und behebt so den systematischen Bias der Maximum-Likelihood-Schätzung bei kleinen Stichproben oder wenigen Ereignissen (Firth 1993). Ohne sie werden Koeffizienten bei knapper EPV tendenziell von der Null weggezogen (Modul 14). Der Hauptauslöser ist **Separation**: Trennt ein Prädiktor das Outcome perfekt oder fast perfekt, konvergiert die gewöhnliche logistische Regression nicht und liefert unendliche oder absurd große Koeffizienten mit unbrauchbaren Konfidenzintervallen (Modul 12). |
| Shrinkage | Absichtliche Stauchung von Regressionskoeffizienten Richtung Null, um Overfitting zu vermeiden (z. B. Ridge, Lasso, penalisierte Modelle). Ein angestrebter Shrinkage-Faktor kann die Fallzahlplanung leiten als Alternative zur reinen EPV-Regel (Modul 13). |
| Konfidenzintervall | Bereich plausibler Werte. Zeigt Unsicherheit, nicht nur einen Punktwert. |
| p-Wert | Wie überraschend die Daten unter einer Nullhypothese wären. Kein Maß für klinische Relevanz. |
| MCAR/MAR/MNAR | Mechanismen fehlender Werte: zufällig, durch beobachtete Variablen erklärbar oder vom fehlenden Wert selbst abhängig. |
| Prentice–Pyke | Statistisches Ergebnis (Prentice & Pyke 1979), wonach Selektion nach dem Outcome bei einer logistischen Regression nur den Intercept verschiebt, nicht die Steigungen. Erklärt, warum Complete-Case-Analysen bei Fehlen nach dem Outcome unverzerrte Odds Ratios liefern können, obwohl absolute Risiken verzerrt bleiben (Modul 14). |
| Confounder | Drittvariable, die Exposition und Outcome beeinflusst und einen Zusammenhang verzerren kann. |
| Mediator | Variable auf dem kausalen Pfad zwischen Exposition und Outcome. |
| Kollider | Variable, die gemeinsame Folge zweier Ursachen ist; Adjustierung kann Scheinzusammenhänge erzeugen. |
| Adjustierung | Ein Modell berücksichtigt zusätzliche Variablen, um Verzerrung zu reduzieren. |
| Overadjustment | Adjustierung für einen Mediator oder Kollider; kontrolliert einen Teil des echten Effekts fälschlich weg oder öffnet einen Scheinpfad. |
| DAG | Gerichteter Graph für Annahmen über Ursachen, Confounder, Mediatoren und Kollider. |
| Propensity Score | Geschätzte Wahrscheinlichkeit, eine Behandlung/Exposition zu erhalten, gegeben Kovariaten. |
| Standardized Mean Difference | Maß für Gruppenbalance; häufiges Ziel nach Gewichtung oder Matching: Betrag < 0,1. |
| MICE | Multiple Imputation by Chained Equations; Verfahren für mehrere plausible Ergänzungen fehlender Werte. |
| Regressionsverdünnung | Klassisch (englisch *regression dilution*, auch *regression attenuation*): nicht-differenzieller Messfehler in einem Prädiktor schwächt dessen geschätzten Koeffizienten Richtung Null. Der Kurs verwendet denselben Begriff für einen strukturell gleichartigen Effekt der Median-Imputation (Modul 14: OR 2,76 → 1,78) – hier ersetzt der imputierte Wert den unbekannten wahren Wert und wirkt wie ein Messfehler. Multiple Imputation vermeidet diese Verzerrung, weil sie plausible Streuung statt eines einzelnen Werts einsetzt. |
| Rubins Regeln (engl. *Rubin's rules*) | Regeln, um Schätzer und Standardfehler aus mehreren imputierten Datensätzen zu einem Gesamtergebnis mit korrekter Unsicherheit zusammenzuführen. |
| λ (Lambda) | Anteil der Gesamtvarianz `T`, der bei Rubins Regeln auf das Fehlen von Werten zurückgeht: `(1+1/m)·B / T`. Nicht dasselbe wie `fmi` (Modul 14). |
| FMI | Fraktion fehlender Information; `mice::pool()` berichtet die endlich-`m`-korrigierte Form `(r + 2/(ν+3))/(r+1)`. Zahlenmäßig verschieden von λ, auch wenn beide dieselbe Idee messen (Modul 14). |
| Barnard–Rubin | Korrektur der Freiheitsgrade von Rubins (1987) `ν_alt`, die für endliches `n` realistischere, kleinere Werte liefert; von `mice::pool()` standardmäßig verwendet (Modul 14). |
| SE-Ratio | Verhältnis aus berichtetem Standardfehler und tatsächlicher Streuung der Schätzer über Wiederholungen. Werte unter 1 zeigen, dass eine Methode ihre eigene Unsicherheit unterschätzt (Modul 14). |
| Überdeckung | Anteil nominaler 95-%-Konfidenzintervalle, die über viele Wiederholungen tatsächlich den wahren Wert enthalten; Sollwert 95 % (Modul 14). |
| PMM | Predictive Mean Matching; Imputationsverfahren, das fehlende Werte durch beobachtete „Spenderwerte“ ähnlicher Fälle ersetzt und dadurch garantiert im plausiblen Wertebereich bleibt (Modul 14). |
| Passive Imputation | Eine abgeleitete Variable (z. B. BMI aus Größe und Gewicht) wird bei jeder Imputation aus den imputierten Ausgangsvariablen neu berechnet statt unabhängig imputiert, damit die Definitionsgleichung erhalten bleibt (Modul 14). |
| Gelman–Rubin (R-hat) | Konvergenzdiagnostik für iterative Sampling-Verfahren wie MICE: vergleicht die Streuung zwischen mehreren parallelen Ketten mit der Streuung innerhalb der Ketten. Werte nahe 1 (Faustregel < 1,1) zeigen, dass sich die Ketten durchmischt haben und kein systematischer Trend mehr besteht (Modul 14). |
| Delta-Adjustment | Sensitivitätsanalyse für MNAR: imputierte Werte werden systematisch um einen Betrag δ verschoben, um zu prüfen, wie stark eine unbeobachtbare Abweichung von MAR das Ergebnis verändert (Modul 14). |
| Kipppunkt | Der δ-Wert einer Delta-Adjustment-Analyse, ab dem sich die Schlussfolgerung ändert, z. B. das Konfidenzintervall erstmals die 1 einschließt (Modul 14). |
| Kongenialität | Meng (1994): Übereinstimmung zwischen Imputations- und Analysemodell. Ein reicheres Imputationsmodell mit Hilfsvariablen, die im Analysemodell fehlen dürfen, senkt oft die Fraktion fehlender Information; ein zu armes verzerrt die gepoolten Schätzer (Modul 14). |
| IPW | Inverse Probability Weighting; gewichtet Beobachtungen mit dem Kehrwert ihrer Behandlungswahrscheinlichkeit, um Gruppen vergleichbar zu machen. |
| Stabilisierte Gewichte | IPW-Variante mit um die marginale Behandlungshäufigkeit skalierten Gewichten; dämpft extreme Gewichte und stabilisiert die Schätzung. |
| Sandwich-/robuste Varianz | Standardfehler-Schätzer, der auch dann konsistent bleibt, wenn die Varianzannahme des Modells falsch ist. Bei IPW zwingend nötig, weil die Gewichte den modellbasierten SE sonst verzerren (Module 15, 19). |
| ATT/ATE | Average Treatment Effect on the Treated / Average Treatment Effect: ATT ist der Effekt in der Gruppe der tatsächlich Behandelten, Ziel von 1:1-Matching. ATE ist der Effekt in der Gesamtpopulation, Ziel von IPW. Beide stimmen nur überein, wenn der Effekt in allen Subgruppen gleich ist (Modul 19). |
| Sensitivität | Anteil der Fälle mit Ereignis/Krankheit, die der Test korrekt positiv erkennt. |
| Spezifität | Anteil der Fälle ohne Ereignis/Krankheit, die der Test korrekt negativ erkennt. |
| PPV/NPV | Positiver/negativer Vorhersagewert; stark abhängig von der Prävalenz. |
| Likelihood Ratio | Wie stark ein Testergebnis die Vorwahrscheinlichkeit verändert; LR+ für positive, LR- für negative Tests. |
| Kalibrierung | Prüft, ob vorhergesagte Risiken als Wahrscheinlichkeiten stimmen. |
| Diskrimination | Prüft, ob ein Modell Fälle mit Ereignis höher einstuft als Fälle ohne Ereignis. |
| Zensierung | In Survival-Daten ist bis zum letzten Beobachtungszeitpunkt kein Ereignis aufgetreten. |
| Konkurrierende Risiken | Ereignisse, die das interessierende Ereignis verhindern (z. B. Tod vor Genesung); erfordern Cumulative Incidence statt Kaplan-Meier. |
| Fine-Gray-Modell | Regressionsmodell für konkurrierende Risiken; modelliert die Subdistribution-Hazard und damit direkt die kumulative Inzidenz. |
| Aalen-Johansen | Nichtparametrischer Schätzer der kumulativen Inzidenzfunktion bei konkurrierenden Risiken; korrekt, im Unterschied zu `1 − Kaplan-Meier`, das konkurrierende Ereignisse fälschlich als Zensierung behandelt und das Risiko überschätzt (Modul 20). |
| Immortal Time Bias | Verzerrung, wenn eine Zeitspanne, in der das Ereignis per Definition nicht eintreten konnte, fälschlich der exponierten Gruppe zugerechnet wird. |
| Net Benefit | Kennzahl der Decision-Curve-Analyse; wägt richtig-positive gegen falsch-positive Entscheidungen bei einer klinischen Risikoschwelle ab. |
| Bootstrap | Wiederholtes Ziehen mit Zurücklegen aus den Daten, um Unsicherheit oder Optimismus einer Schätzung ohne Formelannahmen abzuschätzen. |
| Optimismus | Überschätzung der Modellgüte, weil dieselben Daten Training und Bewertung dienen; per Bootstrap korrigierbar. |
| TRIPOD | Reporting-Leitlinie für Entwicklung und Validierung klinischer Vorhersagemodelle; die aktuelle Fassung TRIPOD+AI gilt für Regressions- und Machine-Learning-Modelle gleichermaßen. |
| Statistical Process Control | Zeitreihenmethode, um Prozessänderungen von Zufallsschwankung zu trennen. |

## Machine Learning und AI

| Begriff | Kurz erklärt |
|---|---|
| Train/Test-Split | Aufteilung in Lern- und Prüfdatensatz. Das Testset bleibt bis zum Schluss unberührt. |
| Cross-Validation | Mehrfaches Trainieren und Prüfen auf verschiedenen Folds, um stabilere Güteschätzungen zu bekommen. |
| Leakage | Information aus Testdaten oder Zukunft gelangt ins Training. Ergebnisse werden zu optimistisch. |
| Pipeline | Verkettete Verarbeitungsschritte, z. B. Skalierung plus Modell, die korrekt zusammen trainiert werden. |
| AUC | Schwellenunabhängiges Maß für Trennschärfe. 0,5 ist Zufall, 1,0 perfekt. |
| C-Index / Konkordanz | Wahrscheinlichkeit, dass das Modell bei zwei zufälligen Fällen den mit dem früheren Ereignis höher einstuft; das AUC-Pendant für Survival-Modelle. |
| Schwellenwert | Grenze, ab der ein Risiko als positiv klassifiziert wird. Klinische Entscheidung, nicht nur Mathematik. |
| Overfitting | Modell lernt Details des Trainingsdatensatzes statt robuste Muster. |
| SHAP | Methode, um Modellvorhersagen einzelnen Merkmalen zuzuschreiben. Hilfreich, aber erklärungsbedürftig. |
| TF-IDF | Gewichtung von Wörtern in Texten: häufig im Dokument, aber selten im gesamten Korpus ergibt ein hohes Gewicht. Grundlage einfacher Textklassifikation und Suche (Module 31, 33). |
| Embedding | Darstellung eines Wortes, Satzes oder Dokuments als Zahlenvektor, in dem bedeutungsähnliche Texte nahe beieinander liegen (Modul 31). |
| Halluzination | Plausibel klingende, aber falsche oder nicht belegte Ausgabe eines Sprachmodells (Module 31, 33). |
| RAG | Retrieval-Augmented Generation: Sprachmodell antwortet mit Hilfe gefundener Dokumentstellen. |
| Drift | Daten oder Zusammenhang ändern sich nach Deployment; Modellgüte kann fallen. |
| Kalibrierungsdrift | Vorhergesagte Risiken passen nach Deployment schrittweise immer schlechter zur beobachteten Ereignisrate, weil sich das Grundrisiko oder nicht erfasste Einflussgrößen innerhalb derselben Einsatzumgebung über die Zeit verändern. Eine stabile AUC schließt Kalibrierungsdrift nicht aus – beides muss getrennt geprüft werden (Module 32, 34). |
| Spektrum-Bias | Ein Modell wird an einer Population trainiert (z. B. Intensivstation), aber in einer anderen mit anderem Schweregrad-Spektrum eingesetzt (z. B. Normalstation). Sensitivität und Spezifität können sich dabei verschieben, und schon die andere Prävalenz ändert den Anteil der Fehlalarme (PPV) – anders als bei Kalibrierungsdrift bleibt die Zielpopulation dabei nicht dieselbe (Modul 34). |
| Model Card | Kurzes Dokument zu Zweck, Daten, Leistung, Grenzen, Fairness und Monitoring eines Modells. |
| Schattenbetrieb (Silent Deployment) | Das Modell läuft auf Live-Daten mit, seine Ausgaben sieht aber niemand; so lassen sich Technik, Diskriminierung und Kalibrierung prospektiv prüfen, ohne Patient:innen zu beeinflussen (Modul 34). |
| Automation Bias | Neigung, Modellausgaben unkritisch zu übernehmen und eigene klinische Warnsignale zu übergehen (Module 32, 34). |
