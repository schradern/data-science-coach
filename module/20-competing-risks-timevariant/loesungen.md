# Musterlösungen · Konkurrierende Risiken und zeitabhängige Cox-Modelle

## Übung 1: Immortal Time Bias erkennen (Methoden-Kritik)

1. **Methodischer Fehler:** Es liegt ein **Immortal Time Bias** vor. Die Zuweisung zur ECMO-Gruppe erfolgt post-hoc („im Verlauf“). Wer an Tag 1 oder 2 verstirbt, bevor eine ECMO initiiert werden konnte, landet automatisch in der Kontrollgruppe. Die ECMO-Gruppe erhält somit eine künstlich unsterbliche Phase vom Aufnahmezeitpunkt bis zum Start der ECMO.
2. **Auswirkung:** Die ECMO-Gruppe wird im Vergleich zur Kontrolle scheinbar deutlich besser abschneiden. Da Patient:innen in der ECMO-Gruppe definitionsgemäß lang genug überleben mussten, um die Intervention überhaupt zu erhalten, ist ihre frühe Sterblichkeit künstlich auf Null gesetzt. Dies täuscht einen starken Überlebensvorteil vor.
3. **Korrektur:** 
    - Entweder wird die ECMO-Gabe als **zeitabhängige Kovariate** in einer zeitabhängigen Cox-Regression modelliert.
    - Oder man nutzt eine **Landmark-Analyse**: Es wird ein fester Zeitpunkt festgelegt (z. B. Tag 3). Nur Patient:innen, die Tag 3 überleben, werden analysiert und nach ihrem Status an Tag 3 klassifiziert. Alle früheren Todesfälle werden ausgeschlossen.

**Praxis:** Das README (Abschnitt 4) zeigt genau diese Korrektur an einer simulierten Vasopressor-Exposition
auf der geteilten Kohorte – inklusive eines Modells, das (anders als eine naive Mini-Kohorte) tatsächlich
konvergiert.

---

## Übung 2: Daten ins Start-Stop-Format überführen

Für Patient:in 42 sieht das Start-Stop-Format wie folgt aus:

| patient_id | start_tag | stop_tag | pneumonie | event (tod) |
|------------|-----------|----------|-----------|-------------|
| 42         | 0         | 4        | 0         | 0           |
| 42         | 4         | 10       | 1         | 1           |

### Erklärung
* **Zeile 1:** Vom Tag der Aufnahme (0) bis zur Diagnose (Tag 4) war der/die Patient:in noch nicht an Pneumonie erkrankt (`pneumonie = 0`). In diesem Intervall trat das Event (Tod) nicht auf (`event = 0`).
* **Zeile 2:** Ab Tag 4 bis zum Tod an Tag 10 war der/die Patient:in an Pneumonie erkrankt (`pneumonie = 1`). Am Ende dieses Intervalls trat das Event auf (`event = 1`).

**Zwingend erforderliche Spalten:** eine ID (die Zeilen derselben Person gehören
zusammen), `start` und `stop` (das Intervall gilt als (start, stop], also ohne
den Starttag, mit dem Stopptag), der Ereignisindikator (nur im letzten Intervall
einer Person kann er 1 sein) und die zeitabhängige Kovariate mit ihrem Wert im
jeweiligen Intervall. Ohne die ID würde das Modell die beiden Zeilen wie zwei
verschiedene Personen behandeln, und die Standardfehler wären bei mehreren
Intervallen je Person zu klein.

---

## Übung 3: 1−KM vs. CIF von Hand berechnen

**Teil 1 – naives 1−KM (Entlassung = Zensierung):**

Nur die Todes-Zeitpunkte tragen einen Faktor bei; Entlassungen verkleinern still das Risikoset:

- Tag 1: `(1 − 2/20) = 0,90`
- Tag 3: `(1 − 2/14) = 12/14 = 6/7 ≈ 0,857`
- Tag 5: `(1 − 1/6) = 5/6 ≈ 0,833`

`S_KM(5) = 0,90 · 6/7 · 5/6 = 0,90 · 5/7 = 0,643`. Naives Sterberisiko `= 1 − 0,643 = 0,357` (**35,7 %**).

**Teil 2 – korrekte CIF (Aalen-Johansen):**

Zuerst die Gesamtüberlebensfunktion `S` (jedes Ereignis – Tod *oder* Entlassung – zählt):

- Tag 1: `2/20` → Faktor `0,90` → `S(1) = 0,90`
- Tag 2: `4/18` → Faktor `7/9` → `S(2) = 0,90 · 7/9 = 0,70`
- Tag 3: `2/14` → Faktor `6/7` → `S(3) = 0,70 · 6/7 = 0,60`
- Tag 4: `6/12` → Faktor `0,50` → `S(4) = 0,60 · 0,50 = 0,30`
- Tag 5: `1/6` → Faktor `5/6` → `S(5) = 0,30 · 5/6 = 0,25`

Die CIF-Beiträge fallen nur an Todes-Tagen an, gewichtet mit `S` des *Vortags* (`S(tⱼ₋₁)`):

- Tag 1: `S(0) · 2/20 = 1 · 0,10 = 0,10`
- Tag 3: `S(2) · 2/14 = 0,70 · 1/7 = 0,10`
- Tag 5: `S(4) · 1/6 = 0,30 · 1/6 = 0,05`

`CIF_Tod(5) = 0,10 + 0,10 + 0,05 = 0,25` (**25,0 %**).

**Teil 3 – Vergleich:** Naiv **35,7 %** vs. korrekt **25,0 %** – das naive 1−KM überschätzt um rund das 1,4-Fache. Grund: Kaplan-Meier behandelt die 10 Entlassenen als zensiert, also so, als blieben sie weiter todesgefährdet. Dadurch schrumpft der Nenner (das Risikoset), und die späten Todesfälle (z. B. Tag 5 mit nur noch `n = 6`) werden auf die gesamte Ausgangspopulation hochgerechnet. Die CIF verschließt die „Tür Entlassung“ endgültig und rechnet die späten Todesfälle nur den tatsächlich noch Gefährdeten zu. Genau dieser Effekt tritt auch in der Kurskohorte auf (27,7 % naiv vs. 16,0 % CIF, README Abschnitt 2).

---

## Übung 4: Warum sich konkurrierende CIFs korrekt zu 1 addieren

1. **Die drei Anteile:**

    - `CIF_Tod(30) = 78/500 = 0,156` (15,6 %)
    - `CIF_Entlassung(30) = 315/500 = 0,630` (63,0 %)
    - noch im Krankenhaus, `S(30) = 107/500 = 0,214` (21,4 %)

    Summe: `0,156 + 0,630 + 0,214 = 1,000`. Die drei Zustände partitionieren die Kohorte vollständig.

    Mit der echten Zensierung (18 Patient:innen vor Tag 30 verloren) liefert Aalen-Johansen etwas andere
    Werte – `0,160 + 0,647 + 0,193 = 1,000` –, die Summe bleibt aber exakt 1.

2. **Warum 1−KM die Grenze sprengen kann:** Die CIFs teilen sich per Konstruktion **dieselbe** Überlebensfunktion `S(t)` – jeder Beitrag wird mit `S(tⱼ₋₁)` gewichtet, sodass `CIF_Tod + CIF_Entlassung + S(t) = 1` immer exakt gilt. Das naive 1−KM für ein Ereignis behandelt dagegen das *andere* Ereignis als Zensierung und benutzt so eine eigene, überhöhte Überlebenskurve: Die Todes-Kurve rechnet die Entlassenen fälschlich als weiter todesgefährdet mit, die Entlassungs-Kurve die Verstorbenen als weiter entlassungsfähig. Jede Kurve überschätzt für sich, und ihre Summe unterliegt keiner gemeinsamen Nebenbedingung – sie kann 100 % übersteigen.
3. **Antwort an die/den Ärzt:in:** `CIF_Tod ≈ 16 %` (15,6 % unter der vereinfachenden Annahme, 16,0 % mit Aalen-Johansen auf den echten Daten). Das ist die tatsächliche Wahrscheinlichkeit, binnen 30 Tagen im Krankenhaus zu versterben, unter korrekter Berücksichtigung der konkurrierenden Entlassung. Die 27,7 % aus dem naiven 1−KM überschätzen dieses Risiko und beantworten in Wahrheit eine hypothetische Frage („Sterberisiko, wenn eine Entlassung unmöglich wäre“), die klinisch nicht gestellt ist.
