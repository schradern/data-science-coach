# Musterlösungen · Regressionsmodelle: Lineare, logistische und Cox-Regression

> Vergleiche zuerst mit deinem eigenen Versuch. Es gibt oft mehrere richtige Wege.

## Aufgabe 1 – Einfache logistische Regression

```python
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

df = pd.read_csv("{{DATEN}}kohorte.csv")
result = smf.logit("verstorben_30d ~ sofa_score", data=df).fit(disp=False)
ci = result.conf_int()
print("OR sofa_score:   ", round(np.exp(result.params["sofa_score"]), 3))
print("95% CI lower:    ", round(np.exp(ci.loc["sofa_score", 0]), 3))
print("95% CI upper:    ", round(np.exp(ci.loc["sofa_score", 1]), 3))
```

```r
suppressPackageStartupMessages(library(tidyverse))

df <- read_csv("{{DATEN}}kohorte.csv", show_col_types = FALSE)
m  <- glm(verstorben_30d ~ sofa_score, data = df, family = binomial)
round(exp(coef(m)), 3)
round(exp(suppressMessages(confint(m))), 3)
```

**Interpretation:** Eine OR von ~1,60 (95-%-KI ≈ [1,41; 1,82]) bedeutet: jeder
zusätzliche SOFA-Punkt geht mit ~60 % höheren Chancen (Odds) auf
30-Tage-Mortalität einher. Das ist ein **unadjustierter** Schätzer; alle
anderen Risikofaktoren, die mit dem SOFA-Score zusammenhängen (Diabetes,
Aufnahmegrund Sepsis), sind noch nicht herausgerechnet.

Der wahre **unadjustierte** OR beträgt 1,477, der wahre **adjustierte** OR
1,387 (beide aus `lib/ground_truth.py`). Das einfache Modell zielt also von
vornherein auf eine größere Zahl, und die Stichprobe streut zusätzlich darum.

**Stolperstein:** Vergleiche hier **nicht** mit `exp(0,28) ≈ 1,32`. Die 0,28
ist ein **Log-Hazard**-Koeffizient der Weibull-Datengenerierung; `exp(0,28)` ist
eine Hazard Ratio, keine Odds Ratio. Wer eine geschätzte OR gegen eine wahre HR
hält, sieht eine „Verzerrung“, die keine ist.

## Aufgabe 2 – Multivariables Modell und Wahrheitsvergleich

```python
df["sepsis"] = (df["aufnahmegrund"] == "Sepsis").astype(int)
df["age_z"]  = df["alter"] - 64
formula = "verstorben_30d ~ age_z + sofa_score + crp_mg_l + diabetes + sepsis"
result2 = smf.logit(formula, data=df).fit(disp=False)
print(np.exp(result2.params).round(3))
```

Die wahren ORs liefert `lib/ground_truth.py`, und zwar für genau dieses
Fünf-Prädiktoren-Modell. Das Modell der Lektion enthält zusätzlich aktives
Rauchen und hat deshalb leicht andere wahre ORs (Diabetes 1,710 statt 1,703):
Odds Ratios sind nicht kollabierbar, ihr wahrer Wert hängt davon ab, wofür
adjustiert wird. Das geht nur im Projektordner
(lokal oder in Colab nach dem Klonen); im Browser-Weg nimmst du die Zahlen
aus der Tabelle unten.

```python
# Only from the project folder: lib/ is part of the repository.
from lib.ground_truth import true_odds_ratios_for
print(true_odds_ratios_for(("age_centred", "sofa_score", "crp_mg_l", "diabetes", "sepsis")))
```

```r
df <- df |> mutate(sepsis = as.integer(aufnahmegrund == "Sepsis"),
                   age_z  = alter - 64)
m2 <- glm(verstorben_30d ~ age_z + sofa_score + crp_mg_l + diabetes + sepsis,
          data = df, family = binomial)
round(exp(coef(m2)), 3)
```

| Prädiktor | OR geschätzt | 95-%-KI | **wahre OR** | überdeckt? |
|---|---|---|---|---|
| `age_z` | 1,048 | [1,03; 1,07] | 1,053 | ja |
| `sofa_score` | 1,486 | [1,27; 1,75] | 1,384 | ja |
| `crp_mg_l` | 1,001 | [1,00; 1,01] | 1,004 | ja |
| `diabetes` | 1,110 | [0,58; 2,14] | 1,703 | ja |
| `sepsis` | 2,013 | [0,99; 4,08] | 2,036 | ja |

**Beobachtung:** Das 95-%-KI überdeckt die wahre OR bei **allen fünf**
Prädiktoren. Das ist kein Hinweis auf eine systematische Abweichung; die
Unterschiede sind mit Stichprobenvarianz vereinbar. Beweisen lässt sich
Unverzerrtheit mit einer einzigen Stichprobe nicht, dafür braucht es viele
Wiederholungen wie im Benchmark von Modul 14 (der dort auch eine kleine
Verzerrung der logistischen Regression bei 78 Ereignissen zeigt).

Relativ am stärksten weicht `diabetes` ab (1,11 vs. 1,70). Der Grund ist
Präzision, nicht Bias: bei 78 Ereignissen und ~19 % Diabetes-Prävalenz reicht
das KI von 0,58 bis 2,14 – es enthält sowohl „schützend“ als auch „verdoppelt
das Risiko“. `crp_mg_l` sieht relativ ebenfalls schlecht aus, weil sein wahrer
Effekt winzig ist; jeder absolute Fehler wirkt dort relativ groß.

**Der wichtigste Punkt der Aufgabe:** Die wahren Werte sind **1,053 / 1,384 /
1,004 / 1,703 / 2,036** – nicht `exp(0,045) / exp(0,28) / …`. Letztere sind
Hazard Ratios (1,046 / 1,320 / 1,004 / 1,549 / 1,835) und liegen systematisch
**näher an 1**. Bei einer Ereignisrate von 15,6 % ist die Näherung „OR ≈ HR“
nicht mehr tragfähig. Wer sie trotzdem benutzt, produziert bei jedem Prädiktor eine scheinbare
Verzerrung in dieselbe Richtung, bei Diabetes und Sepsis rund 10 %.

## Aufgabe 3 – Confounder und Mediator unterscheiden

```python
for exposure in ["sepsis", "diabetes"]:
    for label, extra in [("crude", ""), ("+ alter", " + age_z"),
                         ("+ alter + sofa", " + age_z + sofa_score")]:
        m = smf.logit(f"verstorben_30d ~ {exposure}{extra}", data=df).fit(disp=False)
        ci = np.exp(m.conf_int().loc[exposure])
        print(f"{exposure:9s} {label:16s} OR={np.exp(m.params[exposure]):.3f}"
              f"  [{ci[0]:.2f}, {ci[1]:.2f}]")
```

```r
for (exposure in c("sepsis", "diabetes")) {
  for (extra in c("", " + age_z", " + age_z + sofa_score")) {
    m <- glm(as.formula(paste0("verstorben_30d ~ ", exposure, extra)),
             data = df, family = binomial)
    cat(sprintf("%-9s %-20s OR=%.3f\n", exposure, extra, exp(coef(m)[[exposure]])))
  }
}
```

### Sepsis – hier ist der Bias sichtbar

| Modell | OR geschätzt | 95-%-KI | wahre OR |
|---|---|---|---|
| roh | 5,01 | [2,96; 8,46] | 4,02 |
| + `alter` | **5,75** | [3,30; 10,04] | **4,68** ← **totaler Effekt** |
| + `alter` + `sofa_score` | **1,92** | [0,97; 3,81] | **1,82** ← direkter Effekt |

**`alter` ist für Sepsis weder Confounder noch Mediator.** In der
DAG (`data/README.md`) führt kein Pfeil von `alter` zum Aufnahmegrund: Ältere
bekommen nicht häufiger eine Sepsis. `alter` wirkt nur auf das Outcome, ist also
ein reiner Risikofaktor. Adjustieren darf man trotzdem, es verzerrt nichts. Dass
die OR dabei *steigt* (wahr 4,02 → 4,68), ist kein Confounding, sondern die
Nicht-Kollabierbarkeit der Odds Ratio: Kommt ein starker Risikofaktor ins Modell,
rückt die bedingte OR weiter von 1 weg, auch ohne Confounder. Die Begründung
kommt aus der DAG, nicht daraus, dass sich die Zahl ändert.

**`sofa_score` ist ein Mediator.** Laut DAG (`data/README.md`) liegt er auf dem
Pfad `sepsis → sofa_score → Tod`: eine Sepsis erhöht den SOFA-Score um ~3
Punkte, jeder Punkt erhöht die Hazard. Adjustiert man dafür, kontrolliert man
genau den Teil des Effekts weg, nach dem man gefragt hat. Die OR fällt von 5,75
auf 1,92. Belegt ist der Overadjustment-Bias durch den Vergleich mit der
Wahrheit (totaler Effekt wahr 4,68), nicht dadurch, dass sich die beiden
Konfidenzintervalle kaum überlappen: Beide Modelle rechnen auf denselben Daten,
ihre Intervalle sind deshalb nicht unabhängig, und fehlende Überlappung ist hier
kein formaler Test.

**Was du der Klinikleitung berichtest:** den **totalen** Effekt (OR ≈ 5,8;
wahr 4,68), also das Modell **ohne** `sofa_score`. Die Frage „Wie viel
gefährlicher ist eine Sepsis?“ schließt den Weg über den Organversagensscore
ausdrücklich mit ein. Für eine **Mediationsanalyse** („Wie viel des Effekts läuft
über SOFA?“) ist umgekehrt das SOFA-adjustierte Modell das richtige Werkzeug
(→ Modul 15).

### Diabetes – dasselbe Muster, aber nicht belegbar

| Modell | OR geschätzt | 95-%-KI | wahre OR |
|---|---|---|---|
| roh | 2,07 | [1,19; 3,58] | 2,90 |
| + `alter` | 1,59 | [0,90; 2,82] | 2,12 (totaler Effekt) |
| + `alter` + `sofa_score` | 0,99 | [0,52; 1,87] | 1,57 (direkter Effekt) |

Hier ist `alter` ein echter Confounder: In der DAG führt ein Pfeil von `alter`
zu `diabetes` (Ältere haben häufiger Diabetes), entsprechend fällt die wahre OR
mit der Altersadjustierung von 2,90 auf 2,12.

Die Punktschätzer fallen genauso – 2,07 → 1,59 → 0,99. Aber die ersten beiden
Konfidenzintervalle enthalten sowohl den wahren totalen (2,12) als auch den
wahren direkten Effekt (1,57); das dritte enthält nur den direkten. Bei N = 500 und 78 Ereignissen lässt sich der
Overadjustment-Bias für Diabetes **herleiten** (aus der DAG), aber nicht
**zeigen** (aus den Daten). Der Sprung auf 0,99 ist zum großen Teil Rauschen.

> **Merksatz:** Ob eine Variable Confounder oder Mediator ist, entscheidet die
> DAG – nicht die Korrelationsmatrix, nicht der p-Wert, und nicht der Grundsatz
> „mehr Adjustierung ist besser“. Und ob du einen Bias *demonstrieren* kannst,
> entscheidet dein Konfidenzintervall, nicht dein Punktschätzer.

## Aufgabe 4 – Lineare Regression und Residuenplot

```python
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

m_ols = smf.ols("verweildauer_tage ~ sofa_score + age_z", data=df).fit()
fitted    = m_ols.fittedvalues
residuals = m_ols.resid

plt.figure(figsize=(7, 4))
plt.scatter(fitted, residuals, alpha=0.4, s=20)
plt.axhline(0, color="red", linewidth=1)
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title("Residual plot – OLS for Verweildauer")
plt.tight_layout()
plt.savefig("residualplot.png", dpi=120)
print(f"R²: {m_ols.rsquared:.4f}")
```

```r
m_ols <- lm(verweildauer_tage ~ sofa_score + age_z, data = df)
plot(fitted(m_ols), resid(m_ols),
     main = "Residualplot – OLS Verweildauer",
     xlab = "Fitted values", ylab = "Residuals")
abline(h = 0, col = "red")
```

**Befund:** Der Residuenplot zeigt typischerweise eine trichterförmige Streuung
(Heteroskedastizität) und rechtschiefe Residuen, Zeichen, dass OLS die
Verweildauer-Verteilung schlecht abbildet. Ein **Gamma-GLM** modelliert den
strikt positiven, rechtsschiefen Outcome direkt:

```python
import statsmodels.api as sm          # the families live in statsmodels.api,
import statsmodels.formula.api as smf  # not in statsmodels.formula.api

m_gamma = smf.glm("verweildauer_tage ~ sofa_score + age_z", data=df,
                  family=sm.families.Gamma(link=sm.families.links.Log())).fit()
print(m_gamma.summary())
```

**Stolperstein:** `smf.families` gibt es **nicht**. `statsmodels.formula.api`
liefert nur die Formel-Schnittstelle (`ols`, `glm`, `logit`); die
Verteilungsfamilien und Linkfunktionen wohnen in `statsmodels.api`. Der Tippfehler
`smf.families.Gamma(...)` wirft `AttributeError` – und zwar erst zur Laufzeit,
nachdem der Rest des Skripts schon gelaufen ist.

## Aufgabe 5 – Cox-Regression

```python
from lifelines import CoxPHFitter

# fu_zeit_tage/status are the real time-to-event pair – NOT verweildauer_tage
# (that column is descriptive length of stay, see data/README.md).
cox_df = df[["fu_zeit_tage", "status", "sofa_score", "sepsis"]].copy()
cph = CoxPHFitter()
cph.fit(cox_df, duration_col="fu_zeit_tage", event_col="status")
cols = ["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]
print(cph.summary[cols])
```

```r
suppressPackageStartupMessages(library(survival))
m_cox <- coxph(Surv(fu_zeit_tage, status) ~ sofa_score + sepsis,
               data = df)
summary(m_cox)$conf.int
```

**Welche Frage beantwortet das Modell?** Weil `sofa_score` mit im Modell steht,
ist die Sepsis-HR der **direkte** Effekt bei gleichem SOFA-Score. Den Teil, den
die Sepsis über einen höheren SOFA-Score verursacht, rechnet das Modell heraus.
Für „Wie viel gefährlicher ist eine Sepsis insgesamt?“ gehört der Mediator SOFA
nicht ins Modell (Lektion §4).

**Interpretation:** Sepsis: HR ≈ 1,51 (95-%-KI ≈ [0,88; 2,59], p ≈ 0,14);
SOFA-Score: HR ≈ 1,43 (95-%-KI ≈ [1,27; 1,61], p < 0,0001). Eine HR von ~1,5
für Sepsis bedeutet: septische Patient:innen haben zu **jedem** Zeitpunkt ein
~50 % höheres momentanes Sterberisiko als Patient:innen mit anderem
Aufnahmegrund, unter der Voraussetzung, dass die proportionale-Hazards-Annahme
gilt (d. h. dieses Verhältnis bleibt über die gesamte Beobachtungszeit
konstant). Bei diesem einfachen Modell (nur SOFA + Sepsis, N = 500, 78
Ereignisse) ist der Sepsis-Effekt für sich genommen nicht signifikant, das KI
ist breit und schließt 1 ein, das KI überdeckt aber die wahre direkte
Hazard Ratio von 1,84 (bei festem SOFA-Score, siehe Lektion §4).
Prüfung der PH-Annahme mit `cox.zph()` (R) bzw. `cph.check_assumptions()`
(lifelines).

## Bonus – EPV und Modellkomplexität

```python
n_events = int(df["verstorben_30d"].sum())   # 78 at N=500 (15.6 % mortality)
n_pred   = 8
epv      = n_events / n_pred
print(f"EPV: {epv:.2f}")
# EPV just under 10 → model with 8 predictors is borderline for this cohort
```

**Diskussion:** Mit 78 Ereignissen und 8 Prädiktoren liegt EPV = 9,75, knapp
**unterhalb** der klassischen Faustregel (≥ 10). Mit den 6 Prädiktoren des
Hauptmodells in diesem Modul liegt EPV dagegen bei ≈ 13 (solide vertretbar),
das Bonus-Szenario mit 8 Prädiktoren zeigt also konkret, wie schnell zusätzliche
Variablen die Faustregel unterschreiten. Die Schätzungen sind bei EPV ≈ 9,75
noch möglich, aber mit breiteren Konfidenzintervallen und erhöhtem
Overfitting-Risiko verbunden als beim Hauptmodell.

Die Faustregel ist allerdings grob. Riley et al. (2019, *Statistics in Medicine*)
zeigen, dass die benötigte Stichprobengröße auch von der Zahl der
Modellparameter, dem erwarteten R² des Gesamtmodells und der Ereignisrate abhängt.
Das Paket `pmsampsize` (R)
berechnet die tatsächlich benötigte N für ein gegebenes Modell, eine präzisere
Alternative zur EPV-Faustregel.
