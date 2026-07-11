# Übungen · Fehlende Werte und Imputation

> **Grundpfad (Aufgabe 1–6, ~45 min).** Handwerk (1–2), Entscheidung (3–4),
> Verfahren und Imputationsmodell (5–6). Wer nur §1–§6 der Lektion gelesen hat,
> hört hier auf — das reicht, um fehlende Werte in einer Publikation korrekt zu
> behandeln.
>
> **Vertiefung (Aufgabe 7–11, ~75 min).** Rubin's rules von Hand (7), Wahl von
> `m` (8), eine Methode überführen (9), MNAR beziffern (10), Bericht (11).
> Rechne mit 15–20 Minuten je Aufgabe; sie sind keine Fingerübungen.

---

## Grundpfad

## Aufgabe 1 – Missingness-Tabelle

Berechne für alle Laborvariablen Anzahl und Anteil fehlender Werte. Welche zwei
Variablen fehlen nennenswert?

## Aufgabe 2 – Wer fehlt?

Erzeuge `laktat_fehlend` und `bga_fehlt`. Regressiere **beide** auf
`sofa_score + verstorben_30d`.

Welche Variable fehlt nach Schweregrad, welche nach Outcome? Belege es mit den
Koeffizienten, nicht mit dem Bauchgefühl.

## Aufgabe 3 – Die Entscheidung

Für das Modell `verstorben_30d ~ laktat_mmol_l + alter + sofa_score`:

1. Wovon hängt das Fehlen von `laktat_mmol_l` ab?
2. Steht diese Variable im Modell?
3. Ist die Complete-Case-Schätzung des Laktat-Koeffizienten damit verzerrt oder unverzerrt? Begründe.
4. Und der Mittelwert von `laktat_mmol_l` in einer Table 1 — verzerrt oder nicht?

## Aufgabe 4 – Was Complete Case beim pH kostet

Schätze `verstorben_30d ~ bga_ph + alter + sofa_score`

1. als Complete Case,
2. auf den wahren Werten (`data/bga_ph_wahrheit.csv`, nur weil die Kohorte synthetisch ist).

Vergleiche **drei** Größen: das Odds Ratio je 0,1 pH-Abfall, den Intercept und
das vorhergesagte Risiko für eine 64-jährige Patientin mit SOFA 4 und pH 7,38.

Welche Größe überlebt die Complete-Case-Analyse, welche nicht? Warum?

## Aufgabe 5 – Median-Imputation gegen multiple Imputation

Ergänze zwei weitere Schätzungen für `bga_ph`: Median-Imputation und multiple
Imputation (m = 20, Outcome im Imputationsmodell).

Vergleiche Odds Ratio **und** Standardfehler. Die Median-Imputation liefert
einen kleineren Standardfehler als die Complete-Case-Analyse, obwohl sie weniger
Information hat. Erkläre den Widerspruch.

## Aufgabe 6 – Transfer: das Imputationsmodell sabotieren

Wiederhole die multiple Imputation, aber **lasse `verstorben_30d` aus dem
Imputationsmodell weg** (imputiere `bga_ph ~ alter + sofa_score`).

Vergleiche das Odds Ratio mit dem der korrekten Imputation. Was bricht — das
Odds Ratio oder das absolute Risiko? Erkläre den Mechanismus und formuliere in
einem Satz, warum das Imputationsmodell reicher sein muss als das Analysemodell.

---

## Vertiefung

## Aufgabe 7 – Rubin's rules von Hand

Erzeuge m = 20 vervollständigte Datensätze, fitte auf jedem
`verstorben_30d ~ bga_ph + alter + sofa_score` und sammle `Q̂ₗ` (den
pH-Koeffizienten) und `Uₗ` (dessen quadrierten Standardfehler).

1. Berechne `Q̄`, `Ū`, `B` und `T = Ū + (1 + 1/m)·B` **ohne** eine
   Pooling-Funktion zu benutzen. Vergleiche `SE = √T` mit dem, was
   `pool()` bzw. `MICE.combine()` ausgibt.
2. Berechne `r`, `λ` und `RE = 1/(1 + λ/m)`. Was bedeutet λ inhaltlich?
3. Berechne die Freiheitsgrade nach Rubin (1987), `ν_alt = (m−1)/λ²`. Vergleiche
   sie mit `n − k = 496`. Was fällt auf? Berechne dann `ν_BR` nach
   Barnard & Rubin (1999).
4. Setze `B = 0` und rechne `T` erneut. Welchen Standardfehler hättest du dann
   berichtet — und welchem Verfahren aus §4 entspricht das?

## Aufgabe 8 – Wie viele Imputationen?

1. Wende die Faustregel `m ≳ 100·λ` (White, Royston & Wood 2011) auf dein λ an.
   Reicht `m = 20`?
2. Wiederhole die Imputation für `m ∈ {2, 5, 20}` mit **fünf verschiedenen
   Seeds** und notiere jeweils den gepoolten Standardfehler.
3. Wie stark schwankt der berichtete SE bei `m = 2` gegenüber `m = 20`?
   Formuliere in einem Satz, warum `m = 5` heute keine gute Wahl mehr ist.

## Aufgabe 9 – Transfer: eine Methode überführen

`code/benchmark.py` misst neun Verfahren an 500 simulierten Kohorten. Sieh dir
die Tabelle für `beta_bga_ph` an.

1. Die **deterministische Regressions-Imputation** hat den kleinsten RMSE aller
   Einfachimputationen (0,0501) und die **schlechteste Überdeckung** (73,2 %).
   Erkläre den Widerspruch.
2. Mittelwert-Imputation hat eine SE-Ratio von 1,06 — ihr Standardfehler ist also
   *ehrlich*. Warum überdeckt sie trotzdem nur 83,8 %?
3. Complete Case überdeckt die Steigung zu 93,4 %, das absolute Risiko aber nur
   zu 57,0 %. Welche der beiden Größen würdest du in einem Prognosemodell
   berichten — und was folgt daraus für die Methodenwahl?

## Aufgabe 10 – Klinikfall: MNAR beziffern

Eine Gutachterin schreibt: *„Die Autoren nehmen MAR an. Was, wenn die
Patient:innen ohne Blutgasanalyse in Wahrheit unauffälligere pH-Werte hatten?"*

1. Führe eine Delta-Adjustment-Analyse durch: verschiebe **nur die imputierten**
   pH-Werte um δ ∈ [−0,05; +0,05] und poole jeweils neu.
2. Bestimme den Kipppunkt: das kleinste δ, bei dem das 95-%-KI die 1 einschließt.
3. Setze δ ins Verhältnis zur **bedingten Streuung** des pH. Warum ist der
   Vergleich mit der Messungenauigkeit des Blutgasanalysators ein Denkfehler?
4. Beantworte die Gutachterin in drei Sätzen.

## Aufgabe 11 – Klinikfall: Bericht

Schreibe einen Abschnitt „Umgang mit fehlenden Werten“ für eine Publikation.
Er muss enthalten: Fehlquote pro Variable, den **gemessenen** Zusammenhang des
Fehlens mit anderen Variablen, die Hauptanalyse samt Imputationsmodell, `m`,
Seed und λ, die Sensitivitätsanalyse mit Kipppunkt und ein Urteil zur Stabilität.
