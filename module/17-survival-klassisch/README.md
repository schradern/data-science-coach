# 17 · Klassische Überlebenszeitanalyse

> **Teil 4 – Statistische Inferenz und medizinische Statistik** · Dauer: ~100 min · Voraussetzung: Modul 12 (Regression)

## Lernziele
- Zeit-bis-Ereignis-Daten und Zensierung korrekt lesen.
- Kaplan-Meier-Kurven erstellen und interpretieren.
- Log-rank-Test sinnvoll einsetzen.
- Cox-Regression mit Hazard Ratios berichten.
- Proportional-Hazards-Annahme prüfen und Alternativen kennen.

## Klinischer Aufhänger

Mortalität nach 30 Tagen reduziert Zeit auf ja/nein. Das ist oft zu grob.
Survival-Analyse nutzt zusätzlich, *wann* ein Ereignis eintritt und wie lange
Patient:innen ereignisfrei beobachtet wurden.

![Abbildung: Prinzip der Rechtszensierung in einer Verlaufsbeobachtung. Einige Patient:innen erleben das Ereignis (Tod, rotes Kreuz), andere brechen die Beobachtung vorzeitig ab oder erreichen das Ende des Beobachtungszeitraums ohne Ereignis (zensiert, grauer Kreis).](assets/survival_zensierung.png)

## 1 Warum nicht einfach logistische Regression?

Logistische Regression ignoriert Zeit. Sie behandelt Tod an Tag 2 und Tag 29
gleich. Survival-Analyse nutzt die Zeitinformation und kann Zensierung
verarbeiten: Patient:innen ohne Ereignis bis Studienende tragen Information bei.

> **Merksatz:** Zensiert heißt nicht „fehlend“. Es heißt: bis zu diesem Zeitpunkt wurde kein Ereignis beobachtet.

## 2 Kaplan-Meier

Die Kaplan-Meier-Kurve schätzt die Überlebenswahrscheinlichkeit über die Zeit.
Sie fällt nur bei Ereignissen. Zensierte Beobachtungen reduzieren die Zahl der
noch beobachteten Patient:innen, erzeugen aber keinen Sprung.

Das ist nur unter einer Annahme korrekt: Die Zensierung ist **nicht
informativ**. Wer zensiert wird, muss dasselbe Ereignisrisiko haben wie die, die
weiter beobachtet werden. Wird jemand aus der Beobachtung genommen, *weil* es
ihm besser oder schlechter geht (Entlassung nach Hause, Verlegung ins Hospiz),
ist die Kurve verzerrt.

Gute KM-Abbildungen zeigen:

- Gruppen klar beschriftet,
- 95-%-KI,
- Zahl unter Risiko,
- Zeitraum,
- Ereignisdefinition.

## 3 Log-rank-Test

Der Log-rank-Test vergleicht Überlebenskurven über die gesamte Zeit. Er ist gut,
wenn die Hazard-Raten ungefähr proportional sind.

**Stolperstein:** Kreuzende Kurven können einen Log-rank-Test schwer
interpretierbar machen. Dann sind zeitabhängige Effekte oder restricted mean
survival time oft hilfreicher.

> [!CAUTION]
> **Der Immortal Time Bias in der Überlebensanalyse**  
> Ein schwerwiegender Fehler in medizinischen Kohortenstudien: Angenommen, du vergleichst Patient:innen, die im Verlauf der Intensivstation ein Medikament erhielten, mit solchen, die es nicht erhielten. Wenn du die Zuweisung zur Gruppe „Behandelt“ ab Tag 0 festlegst, erzeugst du einen Immortal Time Bias.  
> Patient:innen in der Behandlungsgruppe **mussten** per Definition lang genug überleben, um die Behandlung zu erhalten. Wer an Tag 2 verstirbt, bevor das Medikament angesetzt werden konnte, landet automatisch in der Kontrollgruppe. Die Behandelten erhalten so eine künstliche „unsterbliche Phase“ (Immortal Time) von Tag 0 bis zum Behandlungsbeginn.  
> *Abhilfe:* Behandle die Exposition als **zeitabhängige Kovariate** (time-varying covariate) im Cox-Modell oder nutze eine **Landmark-Analyse**.

## 4 Cox-Regression

Cox schätzt Hazard Ratios adjustiert für Kovariaten. Die zentrale Annahme:
proportionale Hazards. Eine HR von 1,5 bedeutet: zu jedem Zeitpunkt ist die
momentane Ereignisrate 50 % höher, wenn die PH-Annahme gilt.

**Praxis:** HR nicht als relatives Risiko über den gesamten Zeitraum verkaufen.
Sie ist ein Verhältnis momentaner Hazards.

**Darüber hinaus:** Absolute Risiken zu festen Zeitpunkten sind für klinische
Kommunikation oft verständlicher als HR allein.

## 5 In Python

`fu_zeit_tage` (Beobachtungszeit) und `status` (1 = Tod beobachtet, 0 =
zensiert) sind die echten Zeit-bis-Ereignis-Spalten der Kohorte.
`verweildauer_tage` (Verweildauer) ist eine rein deskriptive Spalte
(Krankenhausaufenthalt) und darf **nicht** als Überlebenszeit verwendet
werden, sie hat keinen definierten Bezug zum Ereignis „Tod“.

```python
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

df = pd.read_csv("{{DATEN}}kohorte.csv")
df["sepsis"] = (df["aufnahmegrund"] == "Sepsis").astype(int)

km = KaplanMeierFitter()
km.fit(df["fu_zeit_tage"], event_observed=df["status"])

a = df[df["sepsis"] == 1]
b = df[df["sepsis"] == 0]
print(logrank_test(a["fu_zeit_tage"], b["fu_zeit_tage"],
                   a["status"], b["status"]))

cph = CoxPHFitter()
cph.fit(df[["fu_zeit_tage", "status", "sepsis", "alter", "sofa_score"]],
        duration_col="fu_zeit_tage", event_col="status")
print(cph.summary)
cph.check_assumptions(df[["fu_zeit_tage", "status", "sepsis", "alter", "sofa_score"]])
```

Das vollständige Skript liegt unter [`code/python.py`](code/python.py).

## 6 In R

```r
library(readr)
library(survival)

df <- read_csv("{{DATEN}}kohorte.csv", show_col_types = FALSE)
df$sepsis <- as.integer(df$aufnahmegrund == "Sepsis")

fit <- survfit(Surv(fu_zeit_tage, status) ~ sepsis, data = df)
survdiff(Surv(fu_zeit_tage, status) ~ sepsis, data = df)

cox <- coxph(Surv(fu_zeit_tage, status) ~ sepsis + alter + sofa_score, data = df)
summary(cox)
cox.zph(cox)
```

`cox.zph()` prüft die PH-Annahme über Schoenfeld-Residuen; in dieser Kohorte
liefert sie für alle drei Kovariaten p > 0,05 (kein Hinweis auf eine
Verletzung der PH-Annahme).

## 7 Das Ergebnis

Beide Wege liefern dieselben Zahlen – deshalb stehen sie hier und nicht in
einem der beiden Sprachabschnitte.

Bei Tag 30 überleben Sepsis-Patient:innen mit ≈ 62 % deutlich seltener als
Nicht-Sepsis-Patient:innen mit ≈ 89 % (Log-rank p < 0,0001). Das adjustierte
Cox-Modell zeigt HR(Sepsis) ≈ 1,52 (95%-KI [0,86; 2,71], hier nicht
signifikant; SOFA liegt auf dem Pfad Sepsis → SOFA → Tod, das Modell schätzt
also nur den nicht über SOFA vermittelten Teil, vgl. Modul 15),
HR(Alter) ≈ 1,04/Jahr und HR(SOFA) ≈ 1,42 pro Punkt, alle in die erwartete
Richtung (> 1, höheres Risiko).

## 8 Bericht

Beispiel:

> Überleben wurde ab Aufnahme bis Tod oder Zensierung (`fu_zeit_tage`,
> `status`) analysiert. Kaplan-Meier-Kurven wurden nach Sepsisstatus
> dargestellt und mit Log-rank-Test verglichen (p < 0,0001). Ein Cox-Modell
> adjustierte für Alter und SOFA. Hazard Ratios werden mit 95-%-KI
> berichtet; die Proportional-Hazards-Annahme wurde mit `cox.zph()` geprüft
> und nicht verletzt.

> **Wann du Hilfe holst.** Wenn die Proportional-Hazards-Annahme verletzt ist, konkurrierende Ereignisse (z. B. Entlassung) das Zielereignis verhindern, oder sich eine Exposition erst im Verlauf ändert, reichen Kaplan-Meier und Standard-Cox nicht mehr aus – dann sollte eine Biostatistikerin das passende Modell mitwählen.

## Fallstricke und Merksätze

- **Zensierung ist Information.** Nicht einfach löschen.
- **HR ist nicht RR.** Hazard ist ein momentanes Risiko.
- **PH-Annahme prüfen.** Eine einzige HR ist bei verletzter PH-Annahme irreführend.
- **Absolute Risiken ergänzen.** Sie sind für Klinikgespräche oft klarer.

## Selbstcheck

1. **Warum ist zensiert nicht fehlend?** – Weil bekannt ist, dass bis zum letzten Beobachtungszeitpunkt kein Ereignis auftrat.
2. **Was prüft der Log-rank-Test?** – Ob sich die Überlebenskurven über die Zeit unterscheiden.
3. **Was bedeutet HR 1,5?** – Unter PH-Annahme ist die momentane Ereignisrate zu jedem Zeitpunkt 50 % höher.

## Abschluss
Übungen: [`uebungen.md`](uebungen.md) · Lösungen: [`loesungen.md`](loesungen.md)
· Zurück zur [Kursübersicht](../../README.md)
