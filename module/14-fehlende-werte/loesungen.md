# Musterlösungen · Fehlende Werte und Imputation

Die Zahlen zu Aufgabe 1–6 stammen aus [`code/python.py`](code/python.py)
(Seed 42), die zu Aufgabe 7–10 aus den dort genannten Vertiefungsskripten. Die
R-Skripte liefern dieselben deterministischen Werte; nur die multiple Imputation
weicht ab, weil sie stochastisch ist und `mice` predictive mean matching statt
Regressions-Imputation nutzt. Beim Punktschätzer ist das wenig, bei der
Between-Varianz deutlich: [`code/rubin.R`](code/rubin.R) misst λ ≈ 0,41 statt
0,16.

## Aufgabe 1 – Missingness-Tabelle

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.imputation import mice

df = pd.read_csv("{{DATEN}}kohorte.csv").merge(
    pd.read_csv("{{DATEN}}labor.csv"), on="patient_id", how="left")
print(df.isna().mean().loc[lambda s: s > 0].sort_values(ascending=False))
```

| Variable | fehlend |
|---|---|
| `bga_ph` | 19,8 % |
| `laktat_mmol_l` | 17,2 % |
| `bmi`, `gewicht_kg` | ~6 % |

Nennenswert fehlen `bga_ph` und `laktat_mmol_l`. Die ~6 % beim `bmi` sind MCAR
und kosten nur Power.

## Aufgabe 2 – Wer fehlt?

```python
work = df.assign(laktat_fehlt=df["laktat_mmol_l"].isna().astype(int),
                 bga_fehlt=df["bga_ph"].isna().astype(int))
for col in ["laktat_fehlt", "bga_fehlt"]:
    print(smf.logit(f"{col} ~ sofa_score + verstorben_30d", data=work).fit(disp=0).summary())
```

| fehlende Variable | β(sofa_score) | β(verstorben_30d) |
|---|---|---|
| `laktat_mmol_l` | **−0,268** (p = 0,0001) | −0,065 (p = 0,87) |
| `bga_ph` | −0,071 (p = 0,22) | **+1,455** (p < 0,0001) |

**Laktat fehlt nach Schweregrad:** je höher der SOFA-Score, desto seltener fehlt
es – Schwerkranke bekommen häufiger eine BGA. Das Outcome erklärt nichts.

**Der pH fehlt nach Outcome:** Verstorbene haben ein `exp(1,455) ≈ 4,3`-fach
höheres Odds, keinen pH-Wert zu haben. Konkret fehlt der pH bei **41,0 %** der
Verstorbenen und bei **15,9 %** der Überlebenden.

## Aufgabe 3 – Die Entscheidung

1. Das Fehlen von `laktat_mmol_l` hängt vom **`sofa_score`** ab.
2. **Ja**, `sofa_score` steht im Modell.
3. **Unverzerrt.** Hängt die Missingness nur von Variablen ab, auf die das
   Modell ohnehin konditioniert, dann ist `P(Y | X)` unter den vollständigen
   Fällen dieselbe Verteilung wie in der Gesamtkohorte. Complete Case schätzt
   hier alle Koeffizienten korrekt und kostet nur Präzision:

   ```text
   complete-case β(laktat) = +0,084   (95 % KI −0,113 bis +0,281)
   ```

4. **Verzerrt.** Der Mittelwert ist eine *marginale* Kennzahl, kein bedingter
   Schätzer. Weil Schwerkranke (hohes Laktat) überrepräsentiert sind, liegt der
   beobachtete Mittelwert von 2,16 mmol/l über dem wahren Kohortenmittel. Eine
   Table 1 aus vollständigen Fällen beschreibt die falsche Kohorte.

> **Merksatz:** Dieselbe Spalte, dasselbe „MAR“ – und trotzdem ist der
> Regressionskoeffizient in Ordnung, während die Table 1 falsch ist.

## Aufgabe 4 – Was Complete Case beim pH kostet

| | Intercept | OR je −0,1 pH | Risiko (64 J., SOFA 4, pH 7,38) |
|---|---|---|---|
| volle Daten (Wahrheit) | 68,05 | 2,76 | 0,106 |
| Complete Case | 65,72 | 2,67 | 0,076 |

- **Das Odds Ratio überlebt** (2,67 vs. 2,76 – der Unterschied ist Rauschen).
- **Intercept und absolutes Risiko brechen:** 0,076 statt 0,106, also **28 % zu
  niedrig**.

**Warum:** Die Selektion erfolgt nach dem Outcome `Y`, nicht nach den
Prädiktoren. In einem logistischen Modell verschiebt eine Selektion nach `Y`
ausschließlich den Intercept – die Steigungen bleiben unverzerrt. Das ist exakt
derselbe Grund, aus dem man aus einer Fall-Kontroll-Studie Odds Ratios schätzen
darf, aber keine Prävalenz (Prentice & Pyke 1979).

Klinisch: Das Modell erkennt korrekt, *dass* Azidose das Risiko erhöht, und sagt
jeder Patient:in trotzdem eine zu gute Prognose voraus. Wer nur Odds Ratios
berichtet, bemerkt den Fehler nie.

Sichtbar schon an der Kohorte selbst:

```text
volle Kohorte        N=500   30-Tage-Mortalität 15,6 %
vollständige Fälle   N=401   30-Tage-Mortalität 11,5 %
```

## Aufgabe 5 – Median- gegen multiple Imputation

| | β(bga_ph) | SE | OR je −0,1 pH | Risiko |
|---|---|---|---|---|
| volle Daten (Wahrheit) | −10,14 | 3,00 | 2,76 | 0,106 |
| Complete Case | −9,82 | 3,77 | 2,67 | 0,076 |
| Median-Imputation | −5,76 | **3,21** | **1,78** | 0,115 |
| Multiple Imputation (m = 20) | −9,57 | 3,60 | 2,60 | 0,107 |

**Der Widerspruch:** Die Median-Imputation setzt 99 Patient:innen auf exakt
denselben pH-Wert. Damit

1. **schrumpft der Effekt.** Die künstliche Punktwolke am Median trägt keinen
   Zusammenhang mit dem Outcome bei; das verdünnt die Steigung gegen Null
   (regression dilution): OR 2,76 → 1,78.
2. **schrumpft der Standardfehler.** Das Modell sieht 500 statt 401
   Beobachtungen und behandelt die 99 erfundenen Werte wie gemessene.

Der SE ist also nicht kleiner, weil mehr Information vorläge, sondern weil die
Methode so tut. Rubins Regeln addieren die **Between-Imputation-Varianz** wieder
hinzu: der MI-Standardfehler (3,60) ist ehrlich größer als der der
Median-Imputation und kleiner als der der Complete-Case-Analyse (3,77), die
ganze Fälle verwirft.

## Aufgabe 6 – Das Imputationsmodell sabotieren

```python
modell = "verstorben_30d ~ bga_ph + alter + sofa_score"
np.random.seed(42)
work = df[["verstorben_30d", "bga_ph", "alter", "sofa_score"]].copy()
imputer = mice.MICEData(work)
imputer.set_imputer("bga_ph", "alter + sofa_score")   # without verstorben_30d
mi = mice.MICE(modell, sm.Logit, imputer, fit_kwds={"disp": 0}).fit(
    n_imputations=20, n_burnin=10)
b_ph = np.asarray(mi.params)[1]                        # order: Intercept, bga_ph, ...
print(f"OR je -0,1 pH, Outcome weggelassen: {np.exp(-0.1 * b_ph):.2f}")
```

| | OR je −0,1 pH | Risiko (Referenz) |
|---|---|---|
| volle Daten (Wahrheit) | 2,76 | 0,106 |
| MI, Outcome **im** Imputationsmodell | 2,60 | 0,107 |
| MI, Outcome **weggelassen** | **1,75** | 0,114 |

**Es bricht das Odds Ratio, nicht das absolute Risiko** – das Spiegelbild von
Aufgabe 4.

**Mechanismus:** Ohne `verstorben_30d` werden die fehlenden pH-Werte allein aus
`alter` und `sofa_score` vorhergesagt, also *unabhängig vom Outcome*. Die 99
imputierten Patient:innen tragen per Konstruktion keinen pH-Outcome-Zusammenhang
bei, und die gepoolte Steigung wird gegen Null verdünnt – genau wie bei der
Median-Imputation. Die Missingness war MAR **gegeben das Outcome**; lässt man das
Outcome weg, ist sie es nicht mehr.

> **Merksatz:** Das Imputationsmodell muss jede Variable enthalten, von der die
> Missingness abhängt – auch dann, wenn sie im Analysemodell die abhängige
> Variable ist. Ein Imputationsmodell ist kein Analysemodell.

## Aufgabe 7 – Rubins Regeln von Hand

Vollständig in [`code/rubin.py`](code/rubin.py) / [`code/rubin.R`](code/rubin.R).

**1.** Aus den m = 20 Fits (Python, `statsmodels`):

```text
Q̄  = -9,9426      Ū = 9,1799      B = 1,6273
T  = Ū + (1 + 1/20)·B = 9,1799 + 1,7087 = 10,8886
SE = √T = 3,2998
```

`MICE.combine()` auf **denselben** vervollständigten Datensätzen liefert exakt
`Q̄ = -9,942579`, `SE = 3,299795` – bis auf Gleitkommarauschen identisch. Rubins
Regeln sind keine Blackbox, sondern vier Zeilen Arithmetik.

**2.**

```text
r  = (1 + 1/20)·B / Ū = 0,1861
λ  = (1 + 1/20)·B / T = 0,1569
RE = 1 / (1 + λ/m)    = 0,9922
```

**λ = 0,157** heißt: **15,7 % der Unsicherheit** über den pH-Koeffizienten
entstehen daraus, dass 99 Werte fehlen; die übrigen 84,3 % hätte man auch bei
vollständigen Daten. `RE = 0,992` heißt: `m = 20` schöpft 99,2 % der Effizienz
von `m = ∞` aus – für den **Punktschätzer**. Für den **Standardfehler** gilt das
nicht (siehe Aufgabe 8).

**3.**

```text
ν_alt = (m − 1)/λ² = 19 / 0,0246 = 771,6
n − k                            = 496
```

`ν_alt` ist **größer als die Freiheitsgrade bei vollständigen Daten**. Das ist
offensichtlich absurd: fehlende Werte können keine Information hinzufügen. Rubins
Formel von 1987 nimmt implizit `n → ∞` an. Barnard & Rubin (1999) korrigieren:

```text
ν_com = 496
ν_obs = ((496+1)/(496+3)) · 496 · (1 − 0,1569) = 416,5
ν_BR  = (771,6 · 416,5)/(771,6 + 416,5)        = 270,5
```

Ergebnis: `t = −3,013`, `p = 0,0028`, 95-%-KI `[−16,44; −3,45]`.

**Stolperstein:** `statsmodels` rechnet für MI gar keine Freiheitsgrade, sondern
einen Wald-**z**-Test (df = ∞). Auf denselben 20 Datensätzen ergäbe das
`p = 0,0026` statt 0,0028, also etwas zu liberal. Ein frischer
`mice.MICE(...).fit()`-Lauf meldet `p = 0,0041`, aber nur, weil er andere
Datensätze zieht (`code/rubin.py`, Abschnitt 4b). `mice::pool()` implementiert
Barnard–Rubin korrekt.

**4.** Mit `B = 0`:

```text
T  = Ū = 9,1799      SE = 3,0298      (statt 3,2998)
```

Das ist genau das, was eine **Einfachimputation** berichtet: sie behandelt die
99 geratenen Werte als gemessen und lässt die Between-Varianz weg. Der
Standardfehler fällt um 8,2 %, ohne dass ein einziges Datum hinzugekommen wäre.
`B` **ist** die Unsicherheit der Imputation. Sie wegzulassen ist der gesamte
Fehler der Einfachimputation.

## Aufgabe 8 – Wie viele Imputationen?

**1.** `m ≳ 100 · λ = 100 · 0,1569 = 15,7`. Mit `m = 20` ist die Faustregel
(White, Royston & Wood 2011) **gerade erfüllt** – kein Spielraum.

**2./3.** `code/rubin.py` wiederholt die gesamte Pipeline über 10 Seeds je `m`:

| m | mittlerer SE | sd(SE) | cv(SE) |
|---|---|---|---|
| 2 | 3,50 | 0,723 | **20,7 %** |
| 5 | 3,63 | 0,289 | 8,0 % |
| 10 | 3,68 | 0,268 | 7,3 % |
| 20 | 3,49 | 0,136 | 3,9 % |
| 50 | 3,68 | 0,120 | 3,3 % |

Bei `m = 2` schwankt der **berichtete Standardfehler** um 21 % – allein je nach
Zufallsseed. Bei `m = 5` sind es noch ~8 %.

> **Merksatz:** Die alte Regel „m = 5“ zielt auf die Effizienz des
> *Punktschätzers* (`RE` > 0,96 schon bei m = 5). Der *Standardfehler* – und
> damit dein p-Wert und dein Konfidenzintervall – ist dann noch deutlich
> seed-abhängig. Rechenzeit ist heute kein Argument mehr: nimm m = 20–50.

## Aufgabe 9 – Eine Methode überführen

**1. Der Widerspruch (RMSE gegen Überdeckung).**
Die deterministische Regressions-Imputation setzt jeden fehlenden pH auf seinen
**bedingten Erwartungswert** `E[pH | alter, sofa, outcome]`. Das ist per
Definition der Wert, der den mittleren quadratischen Fehler minimiert – deshalb
der beste RMSE (0,0501).

Genau dadurch entfernt sie aber die **Streuung** aus den imputierten Werten. Die
99 imputierten Patient:innen liegen alle exakt auf der Regressionsfläche, ohne
Residualvarianz. Das Analysemodell hält diese künstliche Enge für echte
Information und meldet einen viel zu kleinen Standardfehler: **SE-Ratio 0,66**,
d. h. es berichtet nur 66 % der Streuung, die es tatsächlich hat. Dazu kommt ein
Bias von −3,37: Weil die eingesetzten Werte exakt dem Outcome folgen (es steht im
Imputationsmodell), übertreibt die Methode den Zusammenhang. Überdeckung:
**73,2 %** statt 95 %.

> **Merksatz:** Imputation ist keine Vorhersageaufgabe. Wer die imputierten Werte
> nach RMSE optimiert, optimiert die falsche Größe. Die stochastische Variante
> derselben Methode hat den **schlechtesten** RMSE (0,0709 – sie addiert Rauschen)
> und zugleich zehn Prozentpunkte bessere Überdeckung.

**2. Mittelwert-Imputation: ehrlicher SE, trotzdem kaputt.**
Ihre SE-Ratio ist 1,06 – der Standardfehler stimmt also. Sie scheitert nicht an
der Varianz, sondern am **Bias**: +3,08 auf einem wahren Wert von −9,52, also
knapp ein Drittel des Effekts weg. Ein korrekt breites Intervall um einen falschen
Punkt trifft die Wahrheit trotzdem nicht: **83,8 %**.

> Ein Konfidenzintervall kann auf zwei Weisen versagen: es liegt an der falschen
> Stelle (Bias) oder es hat die falsche Breite (Varianz). Die Tabelle enthält je
> ein Lehrbuchbeispiel.

**3. Complete Case: Steigung 93,4 %, absolutes Risiko 57,0 %.**
In einem **Prognosemodell** ist das absolute Risiko die Zahl, die man einer
Patientin nennt. Ein nominal 95-prozentiges Intervall, das die Wahrheit in 43 %
der Fälle verfehlt, ist unbrauchbar – unabhängig davon, wie korrekt das Odds
Ratio ist.

Daraus folgt: **die Methodenwahl hängt vom Estimand ab.** Willst du nur den
Effekt (OR) berichten, ist Complete Case hier vertretbar. Willst du Risiken
vorhersagen oder kalibrieren, brauchst du MI (93,4 % Überdeckung auf dem Risiko).
Der Missing-Indikator ist für die Vorhersage attraktiv, weil „nicht gemessen“
zur Anwendungszeit bekannt und selbst prädiktiv ist – für die kausale
Effektschätzung ist er ungeeignet, weil der Indikator zum Kollider wird, sobald
das Fehlen vom Outcome abhängt.

## Aufgabe 10 – MNAR beziffern

Vollständig in [`code/mnar_delta.py`](code/mnar_delta.py).

**1./2.** Nur die imputierten Werte werden verschoben, die beobachteten nie (das
Skript prüft das per Assertion):

| δ | OR je −0,1 pH | 95-%-KI | Schluss hält? |
|---|---|---|---|
| −0,02 | 3,31 | (1,56; 6,99) | ja |
| 0,00 (MAR) | 2,40 | (1,12; 5,14) | ja |
| +0,01 | 2,21 | (1,05; 4,65) | ja, knapp |
| **+0,02** | 1,83 | (0,93; 3,59) | **nein – Kipppunkt** |
| +0,05 | 1,32 | (0,69; 2,50) | nein |

Der Kipppunkt liegt bei **δ = +0,02**.

**3.** Die bedingte Streuung des pH (Residual-SD nach Adjustierung für Alter,
SOFA und Outcome) beträgt **0,050**. Der Kipppunkt entspricht also
`0,02 / 0,050 = 0,40` **bedingten Standardabweichungen**.

Der Vergleich mit der Messungenauigkeit des Blutgasanalysators wäre ein
**Kategorienfehler**: Analysenrauschen ist *zufällig* und mittelt sich über 99
Patient:innen praktisch weg. `δ` ist ein *systematischer* Unterschied im wahren
Mittelwert zwischen gemessenen und nicht gemessenen Patient:innen – er mittelt
sich nicht weg. Das richtige Maß ist die Streuung, die das Imputationsmodell
ohnehin nicht erklären kann.

**4. Antwort an die Gutachterin:**

> Die Schlussfolgerung hält, solange die nicht gemessenen Patient:innen im wahren
> pH um weniger als 0,02 Einheiten – 0,4 bedingte Standardabweichungen – von dem
> abweichen, was das Imputationsmodell für sie annimmt. Sie müssten zudem in die
> Richtung abweichen, die ihrer *höheren* Sterblichkeit widerspricht: die
> unbeobachteten Patient:innen müssten trotz ihrer Übersterblichkeit
> **unauffälligere** pH-Werte gehabt haben. Wir halten das für unplausibel,
> geben die Annahme aber offen an und berichten den Kipppunkt, statt MAR
> unbelegt zu behaupten.

## Aufgabe 11 – Bericht

> **Umgang mit fehlenden Werten.** Der arterielle pH fehlte bei 99 von 500
> Patient:innen (19,8 %), Laktat bei 86 (17,2 %), der BMI bei 30 (6,0 %). Das
> Fehlen des pH-Werts war mit der 30-Tage-Mortalität assoziiert (41,0 % der
> Verstorbenen gegenüber 15,9 % der Überlebenden), nicht jedoch mit Alter oder
> SOFA-Score. Das Fehlen von Laktat war umgekehrt mit dem SOFA-Score assoziiert
> (β = −0,27; p < 0,001) und unabhängig vom Outcome.
>
> Die Hauptanalyse verwendet daher multiple Imputation (m = 20, Seed 42, Rubins
> Regeln mit Barnard–Rubin-Freiheitsgraden); das Imputationsmodell enthielt Alter,
> SOFA-Score und das Outcome. Die Fraktion fehlender Information für den
> pH-Koeffizienten betrug λ = 0,16, womit m = 20 die Empfehlung m ≳ 100·λ erfüllt.
> Eine Complete-Case-Analyse als Sensitivitätsanalyse lieferte vergleichbare Odds
> Ratios (2,67 vs. 2,60 je 0,1 pH-Abfall), aber systematisch zu niedrige absolute
> Risiken (−28 % am Referenzpatienten), da vollständige Fälle die Verstorbenen
> unterrepräsentieren. Eine Delta-Adjustment-Analyse zeigte, dass die
> Schlussfolgerung bestehen bleibt, solange die angenommene systematische
> pH-Differenz zwischen nicht gemessenen und gemessenen Patient:innen unter
> +0,02 Einheiten (0,4 bedingte Standardabweichungen) liegt. Auf Complete-Case-basierte
> absolute Risiken wird verzichtet.

**Was der Absatz leistet:** Fehlquoten *pro Variable*; der Mechanismus mit Zahlen
statt Vermutungen belegt; Imputationsmodell, `m`, Seed und λ explizit benannt;
die Sensitivitätsanalyse **beziffert** statt behauptet – und die Aussage, welche
Schlussfolgerung stabil ist und welche Größe deshalb *nicht* berichtet wird.
