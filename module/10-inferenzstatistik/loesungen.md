# Musterlösungen · Inferenzstatistik und Hypothesentests

> Vergleiche zuerst mit deinem eigenen Versuch. Statistische Ehrlichkeit bedeutet:
> immer **p + Effektgröße + Konfidenzintervall** berichten, nie p allein.

---

## Aufgabe 1 – p-Wert interpretieren

```python
import pandas as pd
import numpy as np
from scipy import stats

cohort = pd.read_csv("{{DATEN}}kohorte.csv")
labs   = pd.read_csv("{{DATEN}}labor.csv")
df     = cohort.merge(labs, on="patient_id", how="left")

with_diabetes    = df.loc[df["diabetes"] == 1, "sofa_score"].dropna()
without_diabetes = df.loc[df["diabetes"] == 0, "sofa_score"].dropna()

t_stat, p = stats.ttest_ind(with_diabetes, without_diabetes, equal_var=False)

# Cohen's d (pooled SD)
pooled_sd = np.sqrt(
    ((len(with_diabetes) - 1) * with_diabetes.std()**2 +
     (len(without_diabetes) - 1) * without_diabetes.std()**2) /
    (len(with_diabetes) + len(without_diabetes) - 2)
)
d = (with_diabetes.mean() - without_diabetes.mean()) / pooled_sd

# 95-%-CI of the difference (Welch approximation)
se_diff = np.sqrt(with_diabetes.var(ddof=1) / len(with_diabetes) +
                  without_diabetes.var(ddof=1) / len(without_diabetes))
df_welch = (with_diabetes.var(ddof=1) / len(with_diabetes) +
            without_diabetes.var(ddof=1) / len(without_diabetes))**2 / (
    (with_diabetes.var(ddof=1) / len(with_diabetes))**2 / (len(with_diabetes) - 1) +
    (without_diabetes.var(ddof=1) / len(without_diabetes))**2 / (len(without_diabetes) - 1))
t_crit   = stats.t.ppf(0.975, df=df_welch)
diff     = with_diabetes.mean() - without_diabetes.mean()
ci_lo, ci_hi = diff - t_crit * se_diff, diff + t_crit * se_diff

print(f"Diabetes:         n={len(with_diabetes)}, SOFA={with_diabetes.mean():.2f}")
print(f"kein Diabetes:    n={len(without_diabetes)}, SOFA={without_diabetes.mean():.2f}")
print(f"Welch-t={t_stat:.3f}, p={p:.4f}, Cohen's d={d:.3f}")
print(f"Differenz={diff:.2f}, 95%-KI=[{ci_lo:.2f}, {ci_hi:.2f}]")
```

```r
suppressPackageStartupMessages(library(tidyverse))

cohort <- read_csv("{{DATEN}}kohorte.csv", show_col_types = FALSE)
labs   <- read_csv("{{DATEN}}labor.csv", show_col_types = FALSE)
df     <- left_join(cohort, labs, by = "patient_id")

with_d    <- df |> filter(diabetes == 1) |> pull(sofa_score) |> na.omit()
without_d <- df |> filter(diabetes == 0) |> pull(sofa_score) |> na.omit()

t_res <- t.test(with_d, without_d, var.equal = FALSE)
print(t_res)

# Cohen's d
pooled_sd <- sqrt(((length(with_d)-1)*var(with_d) + (length(without_d)-1)*var(without_d)) /
                  (length(with_d) + length(without_d) - 2))
d <- (mean(with_d) - mean(without_d)) / pooled_sd
cat(sprintf("Cohen's d = %.3f\n", d))
```

**Befund:** Patient:innen mit Diabetes haben einen höheren SOFA-Score (Mittel ≈ 4,59) als Patient:innen ohne Diabetes (≈ 3,48). Die Differenz beträgt ~1,11 Punkte (95-%-KI ≈ [0,64; 1,57]), Welch-t ≈ 4,69, p < 0,001, Cohens d ≈ 0,54 (mittlerer Effekt).

**p-Wert in eigenen Worten:** Wenn es in der Population keinen Unterschied im SOFA-Score zwischen Diabetiker:innen und Nicht-Diabetiker:innen gäbe, würden wir eine mindestens so große Teststatistik mit einer Wahrscheinlichkeit von p < 0,001 beobachten, der p-Wert sagt nichts darüber aus, wie wahrscheinlich diese Nullhypothese selbst ist.

**Häufige Fehldeutung:** „p < 0,001 bedeutet, dass H₀ mit 99,9 %-Wahrscheinlichkeit falsch ist.“ Falsch: Der p-Wert ist P(Daten | H₀), nicht P(H₀ | Daten). Letzteres wäre eine Bayesianische Aussage, die einen Prior erfordert.

---

## Aufgabe 2 – Normalverteilung prüfen

```python
laktat = df["laktat_mmol_l"].dropna()
w_stat, p_sw = stats.shapiro(laktat)
print(f"Shapiro-Wilk: W={w_stat:.4f}, p={p_sw:.2e}")
print(f"n={len(laktat)}, Schiefe={laktat.skew():.2f}")
print(f"Normalverteilung abgelehnt (p<0.05): {p_sw < 0.05}")
```

```r
laktat <- df$laktat_mmol_l |> na.omit()
sw_res <- shapiro.test(laktat)
print(sw_res)
n_l <- length(laktat)   # same skewness estimator as pandas .skew()
cat(sprintf("Schiefe = %.2f\n",
            n_l / ((n_l - 1) * (n_l - 2)) * sum(((laktat - mean(laktat)) / sd(laktat))^3)))
```

**Befund:** W ≈ 0,891, p < 0,001, die Normalverteilungshypothese wird klar abgelehnt. Laktat ist mit Schiefe ≈ 1,39 deutlich rechtsschief: die meisten Patient:innen haben niedrige Werte (~1–2 mmol/l), wenige haben sehr hohe Werte (septischer Schock).

**Trennschärfe:** Das Problem ist nicht zu wenig, sondern zu viel Power. Ab N > 200–300 ist Shapiro-Wilk so mächtig, dass er klinisch unwichtige Abweichungen als signifikant ausweist. Immer visualisieren (Histogramm, QQ-Plot) und den Shapiro-Test als ergänzendes Hilfsmittel sehen, nicht als alleinige Entscheidungsgrundlage. Bei sehr kleinem N übersieht er umgekehrt auch deutliche Abweichungen.

**Robustere Alternative:** Der Mann-Whitney-U-Test (Wilcoxon-Rangsummentest) benötigt keine Normalverteilungsannahme und ist bei schiefen Daten wie Laktat oder CRP die sicherere Wahl.

---

## Aufgabe 3 – Chi-Quadrat-Test

```python
import pandas as pd

# Binarize smoking status: active vs. former/never
df2 = df.copy()
df2["smoker_active"] = (df2["raucherstatus"] == "aktiv").astype(int)

ct = pd.crosstab(df2["smoker_active"], df2["verstorben_30d"],
                 rownames=["smoker_active"], colnames=["verstorben_30d"])
print(ct)

chi2, p_chi2, dof, _ = stats.chi2_contingency(ct, correction=False)
cramers_v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))

print(f"χ²={chi2:.3f}, df={dof}, p={p_chi2:.4f}, Cramér's V={cramers_v:.3f}")
```

```r
df2 <- df |> mutate(smoker_active = if_else(raucherstatus == "aktiv", 1L, 0L))
ct  <- table(smoker_active = df2$smoker_active, verstorben_30d = df2$verstorben_30d)
print(ct)

chi_res   <- chisq.test(ct, correct = FALSE)
print(chi_res)
cramers_v <- sqrt(chi_res$statistic / (sum(ct) * (min(dim(ct)) - 1)))
cat(sprintf("Cramér's V = %.3f\n", cramers_v))
```

**Befund:** χ² ≈ 0,355, df = 1, p ≈ 0,551, Cramérs V ≈ 0,027. In dieser Stichprobe zeigt sich kaum ein Zusammenhang zwischen aktivem Rauchen und 30-Tage-Mortalität (Cramérs V nahe 0). Das ist eine Beschreibung der Stichprobe, kein Beleg für „kein Effekt“.

**Klinische Interpretation:** Abwesenheit von Evidenz ist keine Evidenz für Abwesenheit eines Effekts. Um einen kleinen, aber klinisch relevanten Effekt (z. B. Cramérs V ≈ 0,1) mit 80 % Power nachzuweisen, wäre eine deutlich größere Kohorte nötig. Tatsächlich enthält der Datensatz einen wahren Effekt von aktivem Rauchen (Hazard Ratio ≈ 1,5, siehe `data/README.md`): Der Test hier ist ein falsch-negativer Befund.

---

## Aufgabe 4 – Konfidenzintervalle vergleichen

```python
groups = ["Sepsis", "Pneumonie", "Herzinsuffizienz"]

for group in groups:
    g  = df.loc[df["aufnahmegrund"] == group, "laktat_mmol_l"].dropna()
    n  = len(g)
    se = g.std() / np.sqrt(n)
    ci = stats.t.interval(0.95, df=n - 1, loc=g.mean(), scale=se)
    print(f"{group}: n={n}, Mittel={g.mean():.2f}, 95%-KI=[{ci[0]:.2f}, {ci[1]:.2f}]")
```

```r
groups <- c("Sepsis", "Pneumonie", "Herzinsuffizienz")
for (gr in groups) {
  g  <- df |> filter(aufnahmegrund == gr) |> pull(laktat_mmol_l) |> na.omit()
  ci <- t.test(g, conf.level = 0.95)$conf.int
  cat(sprintf("%s: n=%d, Mittel=%.2f, 95%%-KI=[%.2f, %.2f]\n",
              gr, length(g), mean(g), ci[1], ci[2]))
}
```

**Befund:**

| Gruppe | n | Mittel (mmol/l) | 95-%-KI |
|------------------|-----|-----------------|----------------|
| Sepsis | 87 | 2,38 | [2,08; 2,69] |
| Herzinsuffizienz | 90 | 2,33 | [2,02; 2,65] |
| Pneumonie | 105 | 2,00 | [1,71; 2,28] |

Die KI von Sepsis und Herzinsuffizienz überschneiden sich stark, ein formaler Test würde keinen signifikanten Unterschied ergeben. Überlappende KI der Einzelgruppen schließen Signifikanz jedoch nicht automatisch aus: Das KI der *Differenz* ist die korrekte Grundlage. Für die Frage Sepsis vs. Pneumonie heißt das: Welch-t-Test bzw. 95-%-KI der Differenz (`stats.ttest_ind(..., equal_var=False).confidence_interval()` bzw. `t.test()`). Er ergibt hier p ≈ 0,066, knapp nicht signifikant; die Überlappung der Einzel-KI allein hätte das nicht entscheiden können. Das niedrigste Laktat bei Pneumonie ist klinisch plausibel.

---

## Aufgabe 5 – Multiples Testen

```python
from statsmodels.stats.multitest import multipletests

markers = ["leukozyten_g_l", "haemoglobin_g_dl", "kreatinin_mg_dl",
           "laktat_mmol_l", "natrium_mmol_l"]

results = []
for m in markers:
    alive = df.loc[df["verstorben_30d"] == 0, m].dropna()
    dead  = df.loc[df["verstorben_30d"] == 1, m].dropna()
    t_s, p_s = stats.ttest_ind(alive, dead, equal_var=False)
    pooled_sd = np.sqrt(((len(alive)-1)*alive.std()**2 + (len(dead)-1)*dead.std()**2) /
                        (len(alive) + len(dead) - 2))
    d_val = (alive.mean() - dead.mean()) / pooled_sd
    results.append({"marker": m, "t": t_s, "p_raw": p_s, "cohens_d": d_val})

import pandas as pd
res = pd.DataFrame(results)

_, p_bonf, _, _ = multipletests(res["p_raw"], method="bonferroni")
_, p_bh,   _, _ = multipletests(res["p_raw"], method="fdr_bh")
res["p_bonferroni"] = p_bonf
res["p_bh"]         = p_bh
res["sig_bonf"]     = p_bonf < 0.05
res["sig_bh"]       = p_bh   < 0.05

print(res[["marker", "t", "p_raw", "cohens_d", "p_bonferroni", "p_bh",
           "sig_bonf", "sig_bh"]].round(4).to_string(index=False))
```

```r
suppressPackageStartupMessages(library(tidyverse))
markers <- c("leukozyten_g_l", "haemoglobin_g_dl", "kreatinin_mg_dl",
             "laktat_mmol_l", "natrium_mmol_l")

results <- map_dfr(markers, function(m) {
  alive <- df |> filter(verstorben_30d == 0) |> pull(!!sym(m)) |> na.omit()
  dead  <- df |> filter(verstorben_30d == 1) |> pull(!!sym(m)) |> na.omit()
  test  <- t.test(alive, dead, var.equal = FALSE)
  sp    <- sqrt(((length(alive)-1)*var(alive) + (length(dead)-1)*var(dead)) /
               (length(alive) + length(dead) - 2))
  tibble(marker = m, t = test$statistic, p_raw = test$p.value,
         cohens_d = (mean(alive) - mean(dead)) / sp)
})

results <- results |>
  mutate(p_bonferroni = p.adjust(p_raw, "bonferroni"),
         p_bh         = p.adjust(p_raw, "BH"),
         sig_bonf     = p_bonferroni < 0.05,
         sig_bh       = p_bh < 0.05)
print(results)
```

**Befund:** `leukozyten_g_l` zeigt einen deutlichen, hoch signifikanten Unterschied zwischen Überlebenden und Verstorbenen (t ≈ −5,77, p < 0,0001, Cohens d ≈ −0,69, mittlerer bis großer Effekt), der auch nach Bonferroni- **und** Benjamini-Hochberg-Korrektur signifikant bleibt. Die übrigen vier Marker (Hämoglobin, Kreatinin, Laktat, Natrium) zeigen keinen signifikanten Unterschied (alle p_raw > 0,05, |d| < 0,25); `laktat_mmol_l` liegt mit p_raw ≈ 0,07 am nächsten an der Grenze, verliert die (ohnehin nicht erreichte) Signifikanz nach Korrektur aber vollständig.

**Was auffällt:** Anders als man vielleicht erwartet, trennen die Labormarker Überlebende von Verstorbenen nicht gleichmäßig schlecht, Leukozyten sind hier ein starker, robuster Prädiktor, während Laktat trotz seiner klinischen Prominenz in dieser Kohorte nur einen kleinen, nicht signifikanten Unterschied zeigt. `sofa_score` und `alter` bleiben zusätzlich relevante Prädiktoren (→ Modul 08 (Explorative Datenanalyse und Datenvisualisierung)). Bei 5 Tests ist der Unterschied zwischen Bonferroni und BH gering; bei 50 oder 500 Tests (Omics-Daten) wäre BH deutlich mächtiger.

**Wann BH bevorzugen?** Wenn viele explorative Hypothesen geprüft werden und man den Anteil falsch-positiver Befunde kontrollieren will statt der Wahrscheinlichkeit für auch nur einen einzigen falsch-positiven Befund (familienweise Fehlerrate, Bonferroni). In konfirmatorischen Studien mit wenigen vorab festgelegten Tests ist Bonferroni angemessener.

---

## Bonus – Power-Analyse

```python
from statsmodels.stats.power import TTestIndPower

sepsis    = df.loc[df["aufnahmegrund"] == "Sepsis",  "laktat_mmol_l"].dropna()
no_sepsis = df.loc[df["aufnahmegrund"] != "Sepsis",  "laktat_mmol_l"].dropna()

_, p_t = stats.ttest_ind(sepsis, no_sepsis, equal_var=False)
pooled_sd = np.sqrt(((len(sepsis)-1)*sepsis.std()**2 +
                     (len(no_sepsis)-1)*no_sepsis.std()**2) /
                    (len(sepsis) + len(no_sepsis) - 2))
d_lac = (sepsis.mean() - no_sepsis.mean()) / pooled_sd

analysis = TTestIndPower()
# Observed power given actual sample sizes
power_obs = analysis.power(effect_size=abs(d_lac),
                            nobs1=len(sepsis),
                            ratio=len(no_sepsis) / len(sepsis),
                            alpha=0.05, alternative="two-sided")
# Required n per group for power = 0.8
n_needed = analysis.solve_power(effect_size=abs(d_lac), power=0.8,
                                 ratio=1.0, alpha=0.05)

print(f"Sepsis:    n={len(sepsis)}, mean={sepsis.mean():.2f}")
print(f"kein Sep.: n={len(no_sepsis)}, mean={no_sepsis.mean():.2f}")
print(f"Cohen's d={d_lac:.3f}, t-p={p_t:.4f}")
print(f"Observed power = {power_obs:.3f}")
print(f"n per group for power=0.8: ~{int(np.ceil(n_needed))}")
```

```r
# install.packages("pwr") # once
library(pwr)

# Parameters from observed data (approximate)
d_lac    <- 0.197   # observed Cohen's d
n_sepsis <- 87
n_other  <- 327

# Unequal group sizes: pwr.t2n.test takes n1 and n2 directly
power_res <- pwr.t2n.test(n1 = n_sepsis, n2 = n_other, d = d_lac,
                          sig.level = 0.05, alternative = "two.sided")
print(power_res)

# Required n per group for power = 0.8
pwr.t.test(d = d_lac, power = 0.8, sig.level = 0.05,
           type = "two.sample", alternative = "two.sided")
```

**Befund:** Der beobachtete Effekt ist klein (d ≈ 0,197, t-Test p ≈ 0,10, nicht
signifikant). Die *beobachtete* Power bei den vorliegenden Stichprobengrößen
(n_Sepsis ≈ 87, n_nicht-Sepsis ≈ 327) beträgt ~37 %. Für Power = 0,8 wären je
Gruppe etwa **n ≈ 405** erforderlich (prospektive Fallzahlplanung für eine
*künftige* Studie mit diesem Effekt als Planungsannahme).

**Was Power = 0,8 bedeutet:** In 8 von 10 Wiederholungsstudien, unter der Annahme, dass der wahre Effekt d = 0,197 besteht, würde der Test korrekt die Nullhypothese ablehnen. Diese Aussage ist nur für die **Planung** einer neuen Studie sinnvoll.

**Stolperstein – die „beobachtete Power“ ist (fast) wertlos:** Die oben berechnete
Power aus dem *tatsächlich gemessenen* Effekt der *bereits durchgeführten* Studie
als nachträgliche Erklärung für das nicht-signifikante Ergebnis zu verwenden
(„die Studie war halt underpowered, deshalb kein p < 0,05“) ist ein bekannter
Zirkelschluss (Hoenig & Heisey 2001): Beobachtete Power ist eine
**deterministische, monoton fallende Funktion des p-Werts** und liefert damit
**keine Information**, die der p-Wert nicht schon enthält, ein nicht
signifikantes Ergebnis erzeugt *immer* eine niedrig wirkende beobachtete Power,
unabhängig davon, ob der wahre Effekt existiert oder nicht. Genau davor warnt
Modul 13 (Studiendesign und Power) explizit: „Power nach dem Ergebnis ist meist
wertlos. Nutze beobachtete Effekte nicht als nachträgliche Rechtfertigung.“
Richtig eingesetzt wird eine Power-Berechnung **vor** der Datenerhebung, mit
einem *a priori* geplanten (nicht dem beobachteten) Effekt, zur Fallzahlplanung,
so wie das `n ≈ 405`-Ergebnis oben für eine künftige Studie sinnvoll ist. Eine
kleine Studie, die keinen Effekt findet, ist ein Designproblem
(zu wenig Power, um einen echten kleinen Effekt zuverlässig zu erkennen), kein
Beweis für „kein Effekt“.
