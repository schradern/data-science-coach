# Musterlösungen · Bayesianische Inferenz

## Lösung 1: Posterior von Hand

1. **Posterior.** Beta-Binomial-Update: aus Prior Beta(a, b) und k Ereignissen in n Fällen wird Beta(a + k, b + n − k).

    ```text
    Prior Beta(2, 8),  k = 5,  n = 20
    Posterior = Beta(2 + 5, 8 + 20 − 5) = Beta(7, 23)
    ```

2. **Posterior-Mittel.** Der Mittelwert einer Beta(a, b) ist a / (a + b).

    ```text
    Mittel = 7 / (7 + 23) = 7 / 30 = 0,2333  ≈  0,233
    ```

3. **Vergleich mit dem Datenanteil.** Der reine Datenanteil ist 5/20 = 0,250, das Prior-Mittel ist 2/(2+8) = 0,200. Der Posterior (0,233) liegt **zwischen** beiden und wird vom Prior nach unten gezogen: Der Prior fügt gleichsam 2 Ereignisse in 10 fiktiven Fällen hinzu, und weil er in Richtung 0,20 zeigt, sinkt die Schätzung von 0,250 auf 0,233.

---

## Lösung 2: Warum der Prior an der großen Kohorte kaum zählt

1. **Posterior und Mittel.**

    ```text
    Prior Beta(2, 8),  k = 100,  n = 400
    Posterior = Beta(2 + 100, 8 + 400 − 100) = Beta(102, 308)
    Mittel = 102 / (102 + 308) = 102 / 410 = 0,24878  ≈  0,249
    ```

2. **Verschiebung gegenüber dem Datenanteil.** Der reine Datenanteil ist 100/400 = 0,250.

    ```text
    Verschiebung = 0,250 − 0,24878 = 0,00122  ≈  0,12 Prozentpunkte
    ```

    Der Prior bewegt das Ergebnis also um gut einen Zehntel-Prozentpunkt – praktisch nichts.

3. **Erklärung.** Der Prior Beta(2, 8) ist nur 10 fiktive Fälle wert; sein Gewicht ist 10 / (10 + 400) ≈ 2,4 %. Bei nur 20 realen Fällen (Übung 1) betrug sein Gewicht dagegen 10 / (10 + 20) ≈ 33 %. Je mehr echte Daten, desto stärker überstimmt die Likelihood den Prior – bei knappen Daten trägt er, bei reichlichen verschwindet er.

---

## Lösung 3: Praktische Äquivalenz beurteilen

1. **Null im Intervall?** Das 95-%-Credible-Interval [−0,03; 0,05] schließt die Null ein. Es gibt also keinen klar gerichteten Unterschied zwischen den Armen: Sowohl ein kleiner Vorteil für A als auch für B ist mit den Daten vereinbar.

2. **Ganz in der ROPE?** Das Intervall [−0,03; 0,05] liegt vollständig innerhalb der ROPE ±0,05 (die obere Grenze berührt die Bandkante gerade). Damit ist die Bedingung für praktische Äquivalenz erfüllt: Der Unterschied ist mit hoher Wahrscheinlichkeit (rund 97,5 %, weil nur die obere Intervallgrenze die Bandkante erreicht) klinisch vernachlässigbar (< 5 Prozentpunkte). Das ist die starke Form des Befunds – anders als „70 % der Masse in der ROPE“ (README, Abschnitt 4), wo ein Teil des Intervalls noch außerhalb lag.

3. **Formulierung für die Klinik.** Beispiel:

    > Die Komplikationsraten der beiden Arme unterscheiden sich mit hoher Wahrscheinlichkeit um weniger als 5 Prozentpunkte (Differenz 0,01, 95-%-Credible-Interval [−0,03; 0,05], vollständig innerhalb der vorab festgelegten Äquivalenzgrenze). Die Arme sind damit als praktisch gleichwertig anzusehen.
