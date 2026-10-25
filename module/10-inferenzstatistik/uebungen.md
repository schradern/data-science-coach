# Übungen · Inferenzstatistik und Hypothesentests

Nutze den Kohorten- und Labordatensatz (`kohorte.csv` und `labor.csv` wie in der Lektion, im Projektordner auch `load_cohort()` und `load_labs()`).
Versuche jede Aufgabe zuerst in **Python**, dann in **R**. Lösungen: [`loesungen.md`](loesungen.md).

Für jede Aufgabe gilt: **p + Effektgröße + Konfidenzintervall** berichten, nie p allein.

---

## Aufgabe 1 – p-Wert interpretieren

Führe einen Welch-t-Test durch: Unterscheidet sich der **SOFA-Score** zwischen
Patient:innen mit und ohne Diabetes? Berechne auch Cohens d und das 95-%-KI
der Mittelwertdifferenz.

Formuliere anschließend **in einem Satz** in eigenen Worten:

1. Was der p-Wert in diesem konkreten Fall bedeutet.
2. Benenne eine häufige Fehldeutung, die du aus der Literatur kennst.

*Tipp:* Cohens d = (M₁ − M₂) / gepoolte SD. Welcher Effektstärken-Bereich wäre klinisch auffällig?

---

## Aufgabe 2 – Normalverteilung prüfen

Prüfe `laktat_mmol_l` auf Normalverteilung (Shapiro-Wilk-Test). Berechne dazu zusätzlich
die Schiefe und erstelle ein Histogramm.

Beantworte:

- Warum taugt der Shapiro-Wilk-Test bei großen (und bei sehr kleinen) Stichproben wenig als Entscheidungshilfe?
- Wenn du Laktat zwischen zwei Gruppen vergleichen willst: Welcher Test passt zu welcher Frage (Mittelwertdifferenz vs. „Wer hat eher höhere Werte?“), und was ändert die Schiefe daran?

*Tipp:* Achte auf den Unterschied zwischen statistischer Signifikanz des Shapiro-Tests
und praktischer Relevanz der Abweichung.

---

## Aufgabe 3 – Chi-Quadrat-Test

Gibt es einen Zusammenhang zwischen **Raucherstatus** (aktiv vs. nicht-aktiv) und
30-Tage-Mortalität? Berechne χ², p und Cramérs V.

Schritte:

1. Binarisiere `raucherstatus` zu zwei Gruppen: `"aktiv"` vs. `"ehemalig"/"nie"`.
2. Erstelle eine Kontingenztabelle.
3. Berechne χ² und Cramérs V.
4. Interpretiere den Befund klinisch, auch dann, wenn p > 0,05.

*Tipp:* Cramérs V nahe 0 heißt nicht, dass kein Effekt existiert, es könnte auch an
mangelnder Power liegen. Was bräuchtest du, um das zu klären?

---

## Aufgabe 4 – Konfidenzintervalle vergleichen

Berechne das 95-%-KI für das **mittlere Laktat** in den drei Gruppen:
`"Sepsis"`, `"Pneumonie"`, `"Herzinsuffizienz"`.

Besprich anschließend:

- Überschneiden sich die KI zweier Gruppen, schließt das statistische Signifikanz aus?
- Wie würdest du prüfen, ob die Sepsis-Gruppe ein signifikant höheres Laktat hat als
 die Pneumonie-Gruppe?

*Tipp:* Das KI der *Differenz* (nicht der Einzelgruppen) ist die korrekte Grundlage für
einen formalen Vergleich.

---

## Aufgabe 5 – Multiples Testen

Teste für alle fünf Labormarker (`leukozyten_g_l`, `haemoglobin_g_dl`,
`kreatinin_mg_dl`, `laktat_mmol_l`, `natrium_mmol_l`), ob sie sich zwischen
Überlebenden und Verstorbenen unterscheiden (Welch-t-Test).

Wende danach **Bonferroni** und **Benjamini-Hochberg (FDR)** an.

Beantworte:

1. Welche Marker bleiben nach Bonferroni signifikant?
2. Wo unterscheiden sich Bonferroni und BH-Ergebnis?
3. Wann würdest du BH einem Reviewer gegenüber bevorzugen?

*Tipp:* Verwende `statsmodels.stats.multitest.multipletests` (Python) bzw. `p.adjust` (R).

---

## Bonus – Power-Analyse

Berechne die **Power** des Welch-t-Tests (α = 0,05, zweiseitig) für den
Laktat-Unterschied Sepsis vs. Nicht-Sepsis bei den beobachteten Stichprobengrößen und
dem beobachteten Cohens d.

Frage außerdem: Wie viele Patient:innen je Gruppe wären für Power = 0,8 nötig?

- Python: `statsmodels.stats.power.TTestIndPower`
- R: Paket `pwr`, Funktion `pwr.t.test`

Was bedeutet Power = 0,8 konkret, und was folgt daraus, wenn die Power deutlich darunter liegt?

**Kritische Zusatzfrage** (Vorgriff auf Modul 13 (Studiendesign und Power)):
Ist die hier berechnete „beobachtete Power“ (aus dem tatsächlich gemessenen
Effekt, im *selben* Test, den du gerade nicht-signifikant gefunden hast) eine
sinnvolle nachträgliche Rechtfertigung für das nicht-signifikante Ergebnis?
Was wäre der korrekte, *prospektive* Verwendungszweck einer Power-Berechnung?
