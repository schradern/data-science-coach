# Übungen · Modelleinsatz, Monitoring und Governance

Code in Englisch, Prosa/Antworten Deutsch. Lösungen: [`loesungen.md`](loesungen.md).

## Aufgabe 1 – Modell speichern und laden
Trainiere die Textpipeline aus Modul 31 und speichere sie mit `joblib.dump`.
Lade sie wieder und überprüfe, ob sie identische Vorhersagen liefert.
Welche Informationen sollte ein reales Artefakt zusätzlich zum Pipeline-Objekt enthalten?

## Aufgabe 2 – Prävalenz-Shift: ROC-AUC vs. PR-AUC
Simuliere drei Prävalenz-Stufen, indem du 0 %, 50 % und 100 % zusätzliche positive
Fälle durch **Duplizieren bereits vorhandener positiver Testfälle** hinzufügst.
Berechne für jede Stufe sowohl **ROC-AUC** als auch **PR-AUC**
(`average_precision_score`). Was fällt auf, wenn du die beiden Metriken
vergleichst? Erkläre, warum ROC-AUC dabei näherungsweise konstant bleibt,
PR-AUC aber deutlich steigt, obwohl das Modell selbst unverändert ist.

## Aufgabe 3 – Subgruppenanalyse mit Konfidenzintervall
Berechne die AUC separat für `weiblich` und `maennlich` (nach Bereinigung von `w`),
**inklusive 95%-Bootstrap-Konfidenzintervall** pro Gruppe (README Abschnitt 4 zeigt
die Funktion `bootstrap_auc_ci`). Ist der Unterschied klinisch bedeutsam? Überlappen
sich die Konfidenzintervalle? Wie würdest du eine Lücke adressieren, bei der sich
die Intervalle *nicht* überlappen?

## Aufgabe 4 – Model Card ausfüllen
Fülle die Model-Card-Vorlage aus README.md Abschnitt 2 für das trainierte Modell
mit echten Werten (AUC, Sensitivität, Spezifität, Vokabulargröße). Welche
Abschnitte kannst du ohne prospektive Studie nicht befüllen?

## Bonus – Kalibrierungs-Drift
Berechne mit `calibration_curve` die mittlere Vorhersagewahrscheinlichkeit vs.
beobachtete Häufigkeit für das unverschobene und das am stärksten verschobene
Testset. Zeige, wie der Kalibrierfehler (Expected Calibration Error) steigt.
