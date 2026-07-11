# Übungen · Regressionsmodelle: Lineare, logistische und Cox-Regression

Nutze die Kohortendaten (`load_cohort()`). Versuche jede Aufgabe zuerst in
**Python**, dann in **R**. Lösungen: [`loesungen.md`](loesungen.md).

## Aufgabe 1 – Einfache logistische Regression

Passe ein logistisches Modell an, das `verstorben_30d` ausschließlich durch
`sofa_score` erklärt. Berechne die Odds Ratio mit 95-%-Konfidenzintervall.
Was bedeutet der Wert klinisch, und warum ist das ein unadjustierter Schätzer?

## Aufgabe 2 – Multivariables Modell und Wahrheitsvergleich

Erweitere das Modell um `alter` (zentriert auf 64), `diabetes` und `sepsis`
als weitere Prädiktoren. Vergleiche die geschätzten **Odds Ratios** mit den
wahren ORs aus `lib/ground_truth.py`.

**Achtung:** Vergleiche nicht mit `exp(0,45)` & Co. — das sind die wahren
**Hazard** Ratios der Datengenerierung. Warum ist der Unterschied hier relevant,
und in welche Richtung liegt die wahre OR immer?

## Aufgabe 3 – Confounder und Mediator unterscheiden

Nimm als Exposition zunächst **`sepsis`** (nicht Diabetes — warum, siehe unten).

1. Berechne den **rohen** Effekt von `sepsis` auf `verstorben_30d` (nur `sepsis`).
2. Adjustiere für `alter` (zentriert). Wie verändert sich die OR? Ist `alter`
   ein Confounder, ein Mediator oder keines von beidem — und wie begründest du
   das aus der DAG, nicht aus der Zahl?
3. Adjustiere **zusätzlich** für `sofa_score`. Was passiert mit der OR, und
   warum? Welche Rolle spielt `sofa_score` laut DAG in `data/README.md`?
4. Welche der drei Schätzungen ist der **totale** Effekt von Sepsis, welche der
   **direkte**? Welche würdest du berichten, wenn eine Klinikleitung fragt: „Wie
   viel gefährlicher ist eine Sepsis?"

**Und nun zu Diabetes:** Wiederhole Schritt 1–3 mit `diabetes`. Die
Punktschätzer zeigen dasselbe Muster. Sieh dir aber die 95-%-Konfidenzintervalle
an. Was kannst du bei N = 500 und 78 Ereignissen über den Overadjustment-Bias
bei Diabetes **belegen** — und was nur *herleiten*?

## Aufgabe 4 – Lineare Regression und Residuenplot

Passe ein OLS-Modell an: `verweildauer_tage ~ sofa_score + alter` (zentriert).
Erstelle einen Residuenplot (Residuen gegen Fitted Values). Erkennst du Muster,
die auf Heteroskedastizität oder Nichtlinearität hinweisen? Was würde ein
Gamma-GLM an dieser Stelle besser machen?

## Aufgabe 5 – Cox-Regression

Passe eine Cox-Regression an:
`Surv(fu_zeit_tage, status) ~ sofa_score + sepsis`
(`fu_zeit_tage`/`status` sind die echte Nachbeobachtungszeit/Ereignis, *nicht*
`verweildauer_tage`, siehe `data/README.md`). Interpretiere die Hazard Ratios
inklusive 95-%-KI. Wie viel höher ist das momentane Sterberisiko bei Sepsis im
Vergleich zu anderen Aufnahmegründen, und was setzt diese Aussage voraus?

## Bonus – EPV und Modellkomplexität

Berechne die Events per Variable (EPV) für ein Modell mit 8 Prädiktoren auf
unserer Kohorte. Reichen die Daten aus? Recherchiere, welche Einschränkungen
die klassische EPV-Faustregel (≥ 10) hat und was neuere Simulationsstudien
(z. B. Riley et al. 2019, *Statistics in Medicine*) empfehlen.
