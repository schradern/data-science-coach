# 11 · Bayesianische Inferenz

> **Teil 4 – Statistische Inferenz und medizinische Statistik** · Dauer: ~90 min · Voraussetzung: Modul 10 (Inferenzstatistik)

## Lernziele
- Prior, Likelihood und Posterior in klinischen Worten erklären.
- Für eine Rate eine Beta-Binomial-Posterior und ihr **glaubwürdiges Intervall (Credible Interval)** berechnen.
- Erklären, warum ein Credible Interval etwas anderes aussagt als ein Konfidenzintervall.
- Zwei Gruppen bayesianisch vergleichen: Posterior der Differenz, P(Differenz > 0) und ROPE.
- Beurteilen, wann der Prior das Ergebnis trägt und wann er verschwindet.

## Klinischer Aufhänger

In Modul 10 hast du gelernt: `p > 0,05` heißt **nicht** „kein Effekt". Aber was
sollst du der Oberärztin dann sagen? Sie will keine Aussage über hypothetische
Wiederholungsstudien, sie will wissen: *Wie wahrscheinlich ist es, dass die
30-Tage-Mortalität unserer Sepsis-Patient:innen höher liegt als die der übrigen?*

Genau diese Frage — eine Wahrscheinlichkeitsaussage über den unbekannten Parameter
selbst — beantwortet der frequentistische p-Wert prinzipiell nicht. Die
bayesianische Inferenz beantwortet sie direkt. Wir bleiben bei der Kurskohorte
(500 Intensivpatient:innen, 78 Todesfälle) und brauchen dafür kein MCMC, nur
konjugierte Beta-Binomial-Rechnung.

## 1 Prior × Likelihood ∝ Posterior

Bayesianisches Denken hat drei Zutaten:

- **Prior** — was du über die Rate glaubst, *bevor* du diese Daten siehst (z. B. „Intensivmortalität liegt meist zwischen 10 % und 25 %").
- **Likelihood** — was die Daten sagen (78 von 500 verstorben).
- **Posterior** — dein aktualisierter Glaube, *nachdem* du die Daten gesehen hast.

Die Rechenregel ist der Satz von Bayes, hier in seiner proportionalen Kurzform:

```text
Posterior ∝ Likelihood × Prior
```

Der Posterior ist keine Punktschätzung, sondern eine ganze Verteilung: für jede
denkbare Mortalitätsrate sagt er, wie plausibel sie nach den Daten ist. Alles
Weitere — Mittelwert, Intervall, Vergleich — liest man aus dieser Verteilung ab.

## 2 Beta-Binomial: die Mortalitätsrate

Für eine Rate (Anteil zwischen 0 und 1) ist die **Beta-Verteilung** der
natürliche Prior. Sie ist *konjugiert* zur Binomialverteilung: Prior Beta(a, b)
plus k Ereignisse in n Fällen ergeben wieder eine Beta-Verteilung, ganz ohne
Simulation:

```text
Prior Beta(a, b)  +  k Tote / n Patient:innen  →  Posterior Beta(a + k, b + n − k)
```

Mit dem uniformen Prior Beta(1, 1) („jede Rate zunächst gleich plausibel") und
78 Toten unter 500 wird der Posterior zu **Beta(79, 423)**:

- Posterior-Mittel **0,157**
- 95-%-Credible-Interval **[0,127; 0,190]**

Zum Vergleich das frequentistische Wilson-95-%-KI aus Modul 10/13:
**[0,127; 0,190]** — bei n = 500 praktisch **identisch**. Das ist kein Zufall:
Bei viel Datenmaterial dominiert die Likelihood, und beide Welten treffen sich.

Der entscheidende Unterschied liegt in der **Interpretation**. Das Credible
Interval sagt genau das, was man beim Konfidenzintervall fälschlich glaubt:

> **Merksatz:** Das 95-%-Credible-Interval bedeutet „mit 95 % Wahrscheinlichkeit liegt die wahre Rate zwischen 0,127 und 0,190". Das 95-%-Konfidenzintervall bedeutet das gerade *nicht* (siehe Modul 10) — dort ist der Parameter fix und das Intervall die zufällige Größe.

## 3 Wie stark zieht der Prior?

Ein häufiger Einwand: „Dann bestimmt doch der Prior das Ergebnis." Meistens
nicht — es kommt auf die Datenmenge an. Wir vergleichen einen **schwachen** Prior
Beta(1, 1) mit einem **starken** Prior Beta(20, 80) (Mittel 0,20, so
informativ wie 100 fiktive Patient:innen), einmal an der vollen Kohorte und
einmal an einer frühen Zwischenanalyse (die ersten 30 Aufnahmen, nur 2 Tote):

| Prior | Vollkohorte (k=78/n=500) | Zwischenanalyse (k=2/n=30) |
|---|---|---|
| schwach Beta(1,1) | 0,157 | 0,094 |
| stark Beta(20,80) | 0,163 | 0,169 |

An der vollen Kohorte bewegt der Prior das Posterior-Mittel kaum (0,157 vs.
0,163). An der Zwischenanalyse dagegen zieht der starke Prior das Ergebnis von
0,094 auf 0,169 — hier trägt der Prior, nicht die Daten.

> **Merksatz:** Der Prior zählt, wenn die Daten knapp sind, und verschwindet, wenn sie reichlich sind. Deshalb muss ein informativer Prior begründet und im Bericht offengelegt werden.

## 4 Zwei Gruppen vergleichen

Jetzt die Aufhänger-Frage. Wir bilden für Sepsis- und Nicht-Sepsis-Patient:innen
je einen Beta-Posterior (uniformer Prior) und betrachten die **Posterior-Verteilung
der Differenz** der beiden Raten:

| Gruppe | k/n | Posterior-Mittel | 95-%-CrI |
|---|---|---|---|
| Sepsis | 35/94 | 0,375 | [0,281; 0,474] |
| Nicht-Sepsis | 43/406 | 0,108 | [0,080; 0,140] |

Die Differenz hat **Mittel 0,267** mit 95-%-Credible-Interval **[0,168; 0,370]**.
Und die Zahl, die die Oberärztin eigentlich wollte:

```text
P(Sepsis-Mortalität > Nicht-Sepsis-Mortalität) = 1,00
```

Das ist keine „Ablehnung von H₀", sondern eine direkte Wahrscheinlichkeitsaussage
über den klinischen Sachverhalt — genau die Aussage, die der p-Wert nicht liefern
kann.

**ROPE — praktische Äquivalenz.** Ein statistisch klarer Unterschied kann
klinisch belanglos sein. Die *Region of Practical Equivalence (ROPE)* ist ein
vorab festgelegtes Band um null, innerhalb dessen ein Unterschied als
klinisch vernachlässigbar gilt — hier ±0,05 (5 Prozentpunkte). Wir lesen ab,
wie viel Posterior-Masse der Differenz **innerhalb** dieses Bandes liegt:

- **Sepsis vs. Nicht-Sepsis:** 0 % der Masse in der ROPE → ein klar realer, relevanter Unterschied.
- **Hypertonie vs. keine Hypertonie:** Differenz-Mittel 0,032, 95-%-CrI [−0,033; 0,100], rund **70 %** der Masse in der ROPE.

Die 70 % sind die ehrliche bayesianische Formulierung für „kein klinisch
relevanter Unterschied" — die Aussage, die ein nicht-signifikanter p-Wert
gerade *nicht* treffen darf (Abwesenheit von Evidenz ≠ Evidenz für Abwesenheit,
Modul 10). Sicher wäre praktische Äquivalenz erst, wenn das gesamte
Credible-Interval in der ROPE läge; 70 % Masse ist ein starker, aber noch kein
vollständiger Beleg.

## 5 In Python

```python
import numpy as np
import pandas as pd
from scipy import stats

df = pd.read_csv("{{DATEN}}kohorte.csv")
n = len(df)
k = int(df["verstorben_30d"].sum())

# 1) Posterior der Gesamtmortalität: uniformer Prior Beta(1, 1)
a_post, b_post = 1 + k, 1 + n - k          # Beta(79, 423)
mean = a_post / (a_post + b_post)
lo, hi = stats.beta.ppf([0.025, 0.975], a_post, b_post)
print(f"Posterior mean={mean:.4f}, 95% CrI [{lo:.4f}, {hi:.4f}]")

# 2) Prior-Sensitivitaet: schwach vs. stark, voll vs. Zwischenanalyse
for a0, b0 in [(1, 1), (20, 80)]:
    m_full = (a0 + k) / (a0 + b0 + n)
    print(f"Prior Beta({a0},{b0}): Vollkohorte mean={m_full:.4f}")

# 3) Zwei Gruppen: Posterior der Differenz per Monte-Carlo
rng = np.random.default_rng(42)
sep = df["aufnahmegrund"] == "Sepsis"
k1, n1 = int(df.loc[sep, "verstorben_30d"].sum()), int(sep.sum())
k0, n0 = int(df.loc[~sep, "verstorben_30d"].sum()), int((~sep).sum())
p1 = rng.beta(1 + k1, 1 + n1 - k1, 200_000)
p0 = rng.beta(1 + k0, 1 + n0 - k0, 200_000)
diff = p1 - p0
print(f"Differenz mean={diff.mean():.4f}, "
      f"CrI [{np.quantile(diff, 0.025):.4f}, {np.quantile(diff, 0.975):.4f}]")
print(f"P(Sepsis > Nicht-Sepsis) = {np.mean(diff > 0):.4f}")

# 4) ROPE +/- 0.05 auf der Differenz
print(f"Masse in ROPE = {np.mean((diff > -0.05) & (diff < 0.05)):.4f}")
```

Das vollständige, ausführbare Skript liegt unter [`code/python.py`](code/python.py).

## 6 In R

```r
df <- read_csv("{{DATEN}}kohorte.csv", show_col_types = FALSE)
n <- nrow(df); k <- sum(df$verstorben_30d)

# 1) Posterior der Gesamtmortalitaet: uniformer Prior Beta(1, 1)
a_post <- 1 + k; b_post <- 1 + n - k          # Beta(79, 423)
ci <- qbeta(c(0.025, 0.975), a_post, b_post)
cat(sprintf("Posterior mean=%.4f, 95%% CrI [%.4f, %.4f]\n",
            a_post / (a_post + b_post), ci[1], ci[2]))

# 2) Prior-Sensitivitaet: schwach vs. stark
for (p in list(c(1, 1), c(20, 80))) {
  cat(sprintf("Prior Beta(%d,%d): Vollkohorte mean=%.4f\n",
              p[1], p[2], (p[1] + k) / (p[1] + p[2] + n)))
}

# 3) Zwei Gruppen: Posterior der Differenz per Monte-Carlo
set.seed(42)
sep <- df$aufnahmegrund == "Sepsis"
k1 <- sum(df$verstorben_30d[sep]);  n1 <- sum(sep)
k0 <- sum(df$verstorben_30d[!sep]); n0 <- sum(!sep)
diff <- rbeta(2e5, 1 + k1, 1 + n1 - k1) - rbeta(2e5, 1 + k0, 1 + n0 - k0)
cat(sprintf("Differenz mean=%.4f, CrI [%.4f, %.4f]\n",
            mean(diff), quantile(diff, 0.025), quantile(diff, 0.975)))
cat(sprintf("P(Sepsis > Nicht-Sepsis) = %.4f\n", mean(diff > 0)))

# 4) ROPE +/- 0.05 auf der Differenz
cat(sprintf("Masse in ROPE = %.4f\n", mean(diff > -0.05 & diff < 0.05)))
```

Das vollständige Skript liegt unter [`code/r.R`](code/r.R).

<!-- vertiefung: Darüber hinaus — jenseits konjugierter Modelle (~10 min) -->

**Darüber hinaus:** Die Beta-Binomial-Konjugiertheit funktioniert, weil eine
einzige Rate geschätzt wird. Sobald das Modell komplexer wird — mehrere
Kovariaten, hierarchische Struktur (Patient:innen in Stationen), nicht-konjugierte
Priors — gibt es keine geschlossene Posterior-Form mehr. Dann approximiert man
sie numerisch mit **MCMC** (Markov Chain Monte Carlo), etwa in PyMC, Stan oder
`brms`. Das Denkgerüst bleibt identisch (Prior × Likelihood ∝ Posterior); nur
die Berechnung wird zur Simulation. Für dieses Modul brauchst du das nicht — die
konjugierte Rechnung reicht für Raten vollständig aus.

<!-- /vertiefung -->

> **Wann du Hilfe holst.** Sobald ein **informativer Prior** ein reales Ergebnis mitträgt — und erst recht, wenn das Ergebnis sich mit dem Prior spürbar ändert (wie in der Zwischenanalyse oben) — gehört die Wahl und Begründung des Priors vorab mit einer Biostatistikerin abgestimmt und im Analyseplan dokumentiert, nicht nachträglich passend gewählt.

## Fallstricke und Merksätze

- **Der Prior ist eine Entscheidung, kein Naturgesetz.** Er muss begründet und im Bericht offengelegt werden — sonst ist die Analyse nicht nachvollziehbar.
- **Garbage Prior → Garbage Posterior.** Ein stark informativer, aber falscher Prior verzerrt das Ergebnis besonders dann, wenn die Daten knapp sind.
- **Credible Interval ≠ Konfidenzintervall.** Nur das Credible Interval ist eine Wahrscheinlichkeitsaussage über den Parameter; verwechsle die beiden nie im Bericht.
- **ROPE braucht eine klinisch gesetzte Grenze.** Das Band (hier ±0,05) muss *vorab* und fachlich begründet festgelegt werden, nicht so, dass das gewünschte Ergebnis herauskommt.
- **Bei viel Daten konvergieren beide Welten.** Bei n = 500 stimmen Credible Interval und Wilson-KI numerisch überein — der Unterschied ist die Interpretation, nicht die Zahl.

## Selbstcheck

1. **Woraus entsteht der Posterior?**, Aus Prior × Likelihood (proportional). Der Prior kodiert das Vorwissen, die Likelihood die Daten, der Posterior den aktualisierten Glauben.
2. **Was sagt ein 95-%-Credible-Interval, was ein Konfidenzintervall nicht sagt?**, Dass die wahre Rate mit 95 % Wahrscheinlichkeit im Intervall liegt. Beim Konfidenzintervall ist der Parameter fix und das Intervall die zufällige Größe (Modul 10).
3. **Warum bewegt der Prior die Vollkohorte kaum, die Zwischenanalyse aber stark?**, Weil der Prior gegen die Datenmenge gewichtet wird: 500 Beobachtungen überstimmen ihn, 30 nicht.
4. **Was bedeutet „70 % der Differenz-Masse in der ROPE"?**, Dass der Unterschied mit hoher Wahrscheinlichkeit klinisch vernachlässigbar (±0,05) ist — die ehrliche Formulierung für „kein relevanter Effekt", nicht nur ein nicht-signifikanter p-Wert.

## Abschluss
Übungen: [`uebungen.md`](uebungen.md) · Lösungen: [`loesungen.md`](loesungen.md)
· Zurück zur [Kursübersicht](../../README.md)
