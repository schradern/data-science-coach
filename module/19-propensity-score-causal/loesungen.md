# Musterlösungen · Propensity Score Matching und Weighting

## Übung 1: SMD manuell berechnen

1. **Gepoolte Standardabweichung (Pooled SD):**

```text
Pooled SD = sqrt((8,2² + 7,8²) / 2)
          = sqrt((67,24 + 60,84) / 2)
          = sqrt(64,04)
          ≈ 8,00
```

2. **Standardized Mean Difference (SMD):**

```text
SMD = |68,5 − 69,1| / 8,00 = 0,6 / 8,00 = 0,075   (also 7,5 %)
```

3. **Beurteilung:**
   Die SMD liegt mit `0,075` unter der gebräuchlichen Faustschwelle von `0,10`. Das Alter gilt damit als
   ausreichend balanciert. **Stolperstein:** `0,10` ist eine Konvention, kein Testergebnis – sie stammt aus
   der Simulationsliteratur (Austin 2011) und ist kein Beweis für Unverzerrtheit. Eine SMD knapp unter 0,10
   in einer kleinen Kohorte schließt relevantes Restconfounding nicht aus.

**Praxis:** Im README-Beispiel (Abschnitt 4) siehst du dieselbe Rechnung an den Kohortendaten – dort
liegt eine Kovariate (Geschlecht) ohne Caliper nur knapp unter der Schwelle (SMD ≈ −0,09), und nach IPW
verfehlt Alter sie (≈ 0,13). Rechne die SMD also nicht nur einmal aus, sondern für jede adjustierte
Kovariate einzeln.

---

## Übung 2: Overlap-Verlust interpretieren

1. **Konsequenz für das Matching:** Ein 1:1 Matching wird weitgehend scheitern. Da die Propensity Scores der beiden Gruppen kaum überlappen (kein *Common Support*), gibt es für fast keinen behandelten Fall eine passende Kontrolle mit ähnlichem Score.
2. **Verbleibende Patient:innen:** Nur die sehr wenigen Patient:innen, deren Scores im Überlappungsbereich liegen (z. B. zwischen 0,3 und 0,7), können erfolgreich gematcht und analysiert werden. Die Fallzahl wird drastisch schrumpfen.
3. **Generalisierbarkeit:** Die Generalisierbarkeit (externe Validität) geht verloren. Das Ergebnis der Analyse gilt nicht mehr für die typische Intensivkohorte, sondern nur für eine hochspezifische, atypische Subgruppe (z. B. Behandelte mit ungewöhnlich geringer Krankheitslast und Unbehandelte mit ungewöhnlich hoher Krankheitslast).

---

## Übung 3: IPW-Gewichte von Hand berechnen

1. **Unstabilisierte Gewichte:**

    - A (behandelt): `1 / 0,80 = 1,25`
    - B (behandelt): `1 / 0,10 = 10,0`
    - C (Kontrolle): `1 / (1 − 0,95) = 1 / 0,05 = 20,0`

2. **Stabilisierte Gewichte** (`P(A=a) / P(A=a | X)`):

    - A: `0,188 / 0,80 = 0,235`
    - B: `0,188 / 0,10 = 1,88`
    - C: `0,812 / 0,05 = 16,24`

3. **Interpretation:** Patient:in **C** dominiert die Schätzung – unstabilisiert mit Gewicht `20,0`, stabilisiert immer noch `16,24`. C ist ein:e Nicht-Diabetiker:in mit `PS = 0,95`: Laut Confoundern hätte diese Person fast sicher Diabetes, ist es aber nicht. Genau das ist das Zeichen einer (nahen) **Positivity-Verletzung** – in dieser Confounder-Region gibt es kaum echte Kontrollen, das Modell extrapoliert. Die Stabilisierung skaliert den Gewichtsbereich (von `1,25–20,0` auf `0,235–16,24`), multipliziert aber alle Gewichte einer Gruppe mit derselben Konstanten (`0,188` bzw. `0,812`). Sie **behebt die Positivity-Verletzung nicht**: Innerhalb der Kontrollen bleibt C genauso dominant wie vorher. Dagegen helfen nur Trimming/Trunkierung oder die Einschränkung auf den Common-Support-Bereich (README Abschnitt 5).

---

## Übung 4: Effektive Stichprobengröße bei extremen Gewichten

1. **Gewichte {1, 1, 1, 1, 12}:** `Σ wᵢ = 16` → `(Σ wᵢ)² = 256`. `Σ wᵢ² = 1 + 1 + 1 + 1 + 144 = 148`. `N_eff = 256 / 148 ≈ 1,73`.
2. **Gewichte {1, 1, 1, 1, 1}:** `Σ wᵢ = 5` → `(Σ wᵢ)² = 25`. `Σ wᵢ² = 5`. `N_eff = 25 / 5 = 5,0`.
3. **Schlussfolgerung:** Obwohl in beiden Fällen fünf reale Patient:innen vorliegen, drückt das eine Extremgewicht (12) die effektive Fallzahl auf **~1,73** – die Analyse trägt kaum mehr Information als knapp zwei unabhängige Beobachtungen. Wenige extreme Gewichte blähen die Varianz auf und vergrößern den Standardfehler drastisch. Genau deshalb prüft das README (Abschnitt 5) **Trimming**: Es kappt die extremen Gewichte, hebt die effektive Stichprobengröße und verhindert, dass einzelne Patient:innen die Schätzung dominieren. **Stabilisierte Gewichte** allein leisten das nicht: Sie multiplizieren alle Gewichte einer Gruppe mit derselben Konstanten, und `N_eff` innerhalb der Gruppe bleibt gleich (rechne nach: {1, 1, 1, 1, 12} × 0,188 ergibt wieder ≈ 1,73).
