# Datensätze

Alle Module nutzen denselben **synthetischen** Kohortendatensatz. So bleiben die
Beispiele klinisch nachvollziehbar, reproduzierbar und ohne Datenschutzproblem.

> **Wichtig:** Die Daten sind vollständig fiktiv. Sie enthalten **keine** realen
> Patientendaten. Erzeugt mit festem Seed (`42`) durch `generate_data.py`.

## Erzeugen

```bash
./.venv/bin/python data/generate_data.py
```

Erzeugt **zehn Dateien**: vier Kern-Datensätze (`kohorte.csv`, `vitalwerte.csv`,
`labor.csv`, `notizen.csv`), vier Begleitdateien für Modul 04
(`kohorte_de.csv`, `einschluss.xlsx`, `beispiel.json`, `fhir_bundle.json`),
`bga_ph_wahrheit.csv` (Ground Truth für Modul 14, s. u.) und
`konkurrenz_risiken.csv` (konkurrierende Risiken + zeitvariante Exposition für
Modul 20, s. u.).

### `kohorte.csv`, eine Zeile je Patient:in
| Spalte | Bedeutung | Typ |
|--------|-----------|-----|
| `patient_id` | eindeutige ID | int |
| `alter` | Alter in Jahren | int |
| `geschlecht` | `weiblich`/`maennlich` (bewusst teils `w` als „Schmutz") | str |
| `groesse_cm`, `gewicht_kg`, `bmi` | Körpermaße (BMI teils fehlend) | float |
| `aufnahmegrund` | Pneumonie/Sepsis/Herzinsuffizienz/COPD/Sonstiges | str |
| `diabetes`, `hypertonie` | Komorbidität (0/1) | int |
| `raucherstatus` | nie/ehemalig/aktiv | str |
| `sofa_score` | Schweregrad bei Aufnahme (**Mediator**, siehe DAG unten) | int |
| `crp_mg_l` | C-reaktives Protein | float |
| `verweildauer_tage` | Krankenhaus-Verweildauer (**deskriptiv**, keine Ereigniszeit) | int |
| `fu_zeit_tage` | Nachbeobachtungszeit bis Ereignis/Zensierung, gekappt bei 30 Tagen | int |
| `status` | Ereignisindikator: `1` = verstorben (Ereignis), `0` = zensiert | int |
| `verstorben_30d` | 30-Tage-Mortalität (Outcome, 0/1) — **abgeleitet** aus `status`/`fu_zeit_tage` | int |

`verstorben_30d` ist keine eigenständige Ziehung, sondern folgt
zwangsläufig aus `status` und `fu_zeit_tage` (`verstorben_30d = 1`, wenn
`status == 1` und `fu_zeit_tage <= 30`). Klassifikations- und
Survival-Module sind damit per Konstruktion konsistent: Module 12, 16, 23–27
nutzen `verstorben_30d`, Module 17/28/20 nutzen `fu_zeit_tage`/`status`.

### `vitalwerte.csv`, long format (mehrere Messungen je Patient:in)
`patient_id`, `tag` (0–3), `herzfrequenz`, `map_mmhg`, `temperatur_c`, `spo2_prozent`.
Für Joins (Modul 06) und Verlaufsgrafiken (Modul 08).

### `labor.csv`, wide format
`patient_id`, `leukozyten_g_l`, `haemoglobin_g_dl`, `kreatinin_mg_dl`,
`laktat_mmol_l` (**MAR nach Kovariate**, ~17 % fehlend), `natrium_mmol_l`,
`bga_ph` (arterieller pH, **MAR nach Outcome**, ~20 % fehlend). Die beiden
Missingness-Mechanismen sind unten ausführlich beschrieben.

### `notizen.csv`, klinischer Freitext (für NLP, Teil 5)
`patient_id`, `notiz` (kurze deutsche Aufnahme-Notiz, 2-3 Sätze), `verschlechterung` (0/1).
Das Label ist im Text *schwach* kodiert (z. B. „zunehmende Dyspnoe", „Laktat
steigend") und korreliert mit SOFA-Score und Outcome — echtes, aber nicht
triviales Signal für Textklassifikation (Modul 31).

### Begleitdateien für Modul 04 (Datenimport-Formate)
| Datei | Zweck |
|-------|-------|
| `kohorte_de.csv` | derselbe Kohorten-Datensatz im „deutschen Klinik-Export"-Format: `sep=";"`, `encoding="latin-1"`, Dezimalkomma. |
| `einschluss.xlsx` | Einschlussliste als Excel-Arbeitsmappe mit drei Blättern `Kohorte`, `Labor`, `Klinik` (je ~40 Zeilen). Spalten `A:F` im Blatt `Labor` sind die vollständigen Laborwerte. |
| `beispiel.json` | flaches JSON-Array (10 Patient:innen, REDCap-artiger Export) — lädt direkt als DataFrame. |
| `fhir_bundle.json` | minimales, fiktives FHIR-R4-`Bundle` mit `Patient`- und `Observation`-Ressourcen in `entry`, für `pd.json_normalize(bundle["entry"], sep="_")`. |

## Eingebaute Lernanlässe
- **Fehlende Werte:** `bmi`/`gewicht_kg` (~6 %, MCAR) → Modul 06.
  `laktat_mmol_l` (~17 %, MAR nach `sofa_score`) und `bga_ph` (~20 %, MAR nach
  `verstorben_30d`) → Module 06 und 14. Zwei Mechanismen, entgegengesetzte
  Konsequenzen für Complete-Case-Analysen (s. u.).
- **Uneinheitliche Kategorien:** `geschlecht` enthält teils `w` → Vereinheitlichen in Modul 06.
- **Bekannte „Wahrheit":** Mortalität und Überlebenszeit folgen einem einzigen
  Weibull-Ereigniszeitprozess (s. u.), sodass Regressions- (Modul 12) und
  Survival-Schätzungen (Modul 17/28) gegen die echten Effekte geprüft werden können.

## Die wahre DAG (kausale Struktur)
Für die Kausalitätsmodule (10, 13, 29) ist die zugrunde liegende DAG bewusst
so gebaut, dass `sofa_score` ein **Mediator**, kein Confounder ist:

```
alter ─────────────┬──────────────────────► Tod (fu_zeit_tage/status)
                    │                              ▲
diabetes ───────────┼──► sofa_score ────────────────┤
                    │        ▲                      │
aufnahmegrund=Sepsis┴────────┘                      │
                    │                                │
raucherstatus=aktiv ┘                                │
crp_mg_l ─────────────────────────────────────────────┘
```

- `alter` ist ein **Confounder**: Es beeinflusst sowohl das Risiko (über
  Komorbidität) als auch direkt den Tod — bei der Schätzung von
  Gesamteffekten **muss** dafür adjustiert werden.
- `sofa_score` liegt **auf dem kausalen Pfad** von `diabetes` und
  `aufnahmegrund == "Sepsis"` zu Tod (Diabetes/Sepsis erhöhen den SOFA-Score,
  der wiederum die Hazard erhöht). `sofa_score` ist damit ein **Mediator**:
  Wird bei der Schätzung des **totalen** Effekts von Diabetes oder Sepsis auf
  die Mortalität für `sofa_score` adjustiert, wird der (über SOFA
  vermittelte) Teil des Effekts fälschlich wegkontrolliert
  ("overadjustment"/Mediator-Bias). Für die **direkte** Effektzerlegung
  (Mediationsanalyse) ist die Adjustierung dagegen gewollt. Module 12, 15 und
  19 nutzen genau diese Unterscheidung.

## Der wahre Survival-/Mortalitätsprozess (ground truth)
`generate_cohort()` in `generate_data.py` simuliert **einen** kohärenten
Weibull-Ereigniszeitprozess (Shape `k = 1.3`), der sowohl `fu_zeit_tage`/`status`
als auch `verstorben_30d` erzeugt:

```
eta = -1.95 + 0.045*(alter-64) + 0.28*sofa_score + 0.0035*crp_mg_l
            + 0.45*diabetes + 0.6*(aufnahmegrund=="Sepsis") + 0.4*(raucherstatus=="aktiv")
scale_i     = 115.0 * exp(-eta/1.3)
T_event_i   = scale_i * (-ln(U))**(1/1.3),           U ~ Uniform(0,1)
C_lost_i    = Uniform(3, 40) für ~15 % der Patient:innen, sonst unzensiert
fu_zeit_tage = round(min(T_event_i, C_lost_i, 30))    (administrative Zensierung bei 30 Tagen)
status       = 1, wenn T_event_i <= min(C_lost_i, 30), sonst 0
verstorben_30d = 1, wenn status==1 und fu_zeit_tage<=30
```

Höhere `eta` → kürzere Skala → kürzere Zeit bis Ereignis → höhere Hazard.
Die Koeffizienten (`0.045` Alter, `0.28` SOFA, `0.0035` CRP, `0.45` Diabetes,
`0.6` Sepsis, `0.4` aktives Rauchen) sind die **Ground-Truth-Effekte**, gegen
die Modul 12 (Regression) und Modul 17/28 (Survival) Schätzungen prüfen
können. Erwartete Hazard-Ratio-Richtung: alle > 1 (höheres Risiko).

**Aktuelle Kennzahlen (Seed 42, N=500):** 30-Tage-Mortalität ≈ 15.6 % (78
Ereignisse), Median `fu_zeit_tage` = 30 Tage, Zensierung ≈ 84 % (die meisten
Patient:innen überleben die 30-Tage-Grenze und werden administrativ
zensiert — bei ~16 % Mortalität ist das die erwartete Mehrheit).
Cox-Hazard-Ratios auf `fu_zeit_tage`/`status`: Sepsis ≈ 1.6, Alter ≈ 1.04 (pro
Jahr), SOFA ≈ 1.4 — alle in der klinisch erwarteten Richtung (> 1). Diese
Zahlen sind reproduzierbar (Seed 42) und dienen als Referenz für
Modul-Übungen; kleine Abweichungen durch Paket-/Versionsupdates sind möglich,
die Größenordnung und Richtung bleiben stabil.

## Zwei Missingness-Mechanismen, zwei verschiedene Lehren

`labor.csv` trägt bewusst **zwei** unterschiedliche MAR-Mechanismen. Sie sehen
gleich aus („MAR“) und haben entgegengesetzte Konsequenzen. Modul 14 lebt von
diesem Kontrast.

### `laktat_mmol_l` — Fehlen hängt von einer **Kovariate** ab

Anders als `bmi`/`gewicht_kg` (rein zufällig fehlend, MCAR) hängt die
Missingness von `laktat_mmol_l` vom (beobachteten) `sofa_score` ab: schwerer
kranke Patient:innen bekommen klinisch häufiger eine Blutgasanalyse mit Laktat,
also **weniger** fehlende Werte bei hohem SOFA:

```
p_missing_i = sigmoid(-0.3 - 0.35 * sofa_score_i)
```

Eine logistische Regression `ist_fehlend ~ sofa_score` liefert einen klar
negativen, hochsignifikanten Koeffizienten (aktuell ≈ -0.27, p < 0.0001),
während `alter` unauffällig bleibt. Gesamt-Missingness ≈ 17 % (86 von 500).

**Konsequenz:** Weil
`sofa_score` in den Analysemodellen der Module 12/14/23 **enthalten** ist, sind
Complete-Case-**Regressionskoeffizienten** auf `laktat_mmol_l` **unverzerrt**.
Verzerrt ist nur der **marginale** Mittelwert (die Table 1), weil die
beobachteten Werte die Schwerkranken überrepräsentieren. „MAR“ allein sagt nicht,
ob Complete Case verzerrt — entscheidend ist, ob die Variable, von der das Fehlen
abhängt, im Modell steht.

### `bga_ph` — Fehlen hängt vom **Outcome** ab

```
bga_ph_i     = 7.40 - 0.006*sofa_score_i - 0.024*verstorben_30d_i + N(0, 0.05)
p_missing_i  = sigmoid(-1.7 + 1.4 * verstorben_30d_i)
```

Die Konstanten stehen in `lib/ground_truth.py` (`BGA_*`) und werden von
`generate_data.py` importiert — eine einzige Quelle der Wahrheit für den
Generator und den Monte-Carlo-Prüfstand aus Modul 14.

Klinisch: die Kontroll-BGA am Folgetag wurde bei früh Verstorbenen nie
abgenommen. Der pH fehlt bei **41,0 %** der Verstorbenen und **15,9 %** der
Überlebenden (gesamt ≈ 19,8 %, 99 von 500).

**Zur Wahl von `-0.024`:** Mit dem ursprünglichen `-0.014` lag der Effekt schon
auf den **vollständigen** Daten bei p = 0,046. Jede Imputationsmethode „verlor"
dann die Signifikanz, und Modul 14 hätte gelehrt, multiple Imputation versage —
obwohl der Effekt nur zu schwach war, um 99 fehlende Werte zu überleben. Bei
`-0.024` beträgt die wahre Odds Ratio 2,76 je 0,1 pH-Abfall, was Azidose bei
kritisch Kranken tatsächlich leistet, und der Kontrast wird lehrbar.

**Konsequenz:** Die Selektion erfolgt nach `Y`, nicht nach `X`. In einem
logistischen Modell verschiebt das **nur den Intercept**, nicht die Steigungen
(Prentice & Pyke 1979). Complete Case liefert deshalb ein praktisch unverzerrtes
Odds Ratio (2,67 gegen wahre 2,76 je 0,1 pH-Abfall) und ein um **28 % zu
niedriges absolutes Risiko**. Die scheinbare Mortalität der vollständigen Fälle
fällt von 15,6 % auf 11,5 %.

Multiple Imputation repariert beides — **aber nur, wenn das Outcome im
Imputationsmodell steht**. Lässt man es weg, bricht stattdessen das Odds Ratio
(1,80 statt 2,76). Der Mechanismus hängt ausschließlich von beobachteten Daten
ab, ist also MAR und nicht MNAR.

Beide Mechanismen sind seed-fest (`bga_ph` nutzt einen eigenen RNG-Strom,
`SEED + 1`, sodass die übrigen Spalten unverändert bleiben).

**Nebenwirkung, die Modul 06 nutzt:** Nach dem Join von `kohorte.csv` und
`labor.csv` löscht ein nacktes `dropna()` jetzt **192 von 500 Zeilen (38,4 %)** —
obwohl keine einzelne Spalte mehr als 20 % vermisst. Die Verluste addieren sich.

### `bga_ph_wahrheit.csv` — die Wahrheit hinter dem pH

Eine neunte Datei, die es in echten Daten nie gäbe: `patient_id`, `bga_ph_wahr`
— die pH-Werte **vor** dem Anwenden der Missingness. Nur Modul 14 lädt sie, um
Verzerrung zu *zeigen* statt zu behaupten. `labor.csv` bleibt bewusst ein
realistischer Laborexport ohne Lösungsschlüssel.

## Konkurrierende Risiken + zeitvariante Exposition (`konkurrenz_risiken.csv`, Modul 20)

Eine zehnte Datei mit einer **kohärent erzeugten** 3-Zustands-Ereignisvariable
plus einer zeitvarianten Vasopressor-Exposition. Spalten: `patient_id`,
`event_time`, `event_state` (`0` = zensiert an Tag 30, `1` = Tod, `2` =
Entlassung), `vasopressor_start_tag` (Tag, an dem die Exposition beginnt, sonst
leer).

Warum eine eigene Datei? Modul 20 braucht eine **echte** Entlassungs-Ereigniszeit
als Konkurrenzereignis zum Tod. `verweildauer_tage` in `kohorte.csv` ist rein
**deskriptiv** (aus einem nur locker gekoppelten Teil des Generators) und darf
**nicht** als Entlassungszeit missbraucht werden: Sie ist von der Weibull-Todes-/
Zensierungs-Uhr entkoppelt und würde ein degeneriertes Risikoset (naives 1-KM →
100 %) erzeugen. Die neue Datei löst das mit einem statistisch stimmigen Prozess.

### Der wahre Konkurrenzrisiko-Prozess (ground truth)

```
# Tod: exakt das beobachtete Kohortenereignis (status == 1) am Tag fu_zeit_tage.
#      Wer im Krankenhaus stirbt, wird nie entlassen -> die wahre CIF des Todes
#      entspricht exakt der 30-Tage-Mortalität (~15,6 %).

# Entlassung: für alle Überlebenden (status == 0) ein echtes Konkurrenzereignis
#             auf derselben Tagesachse:
mean_i      = 15.0 + 1.4 * sofa_score_i          # steigt mit Schweregrad
D_i         ~ Gamma(shape = 2.1, mean = mean_i)  # Entlassungs-Hazard FÄLLT mit SOFA
event_state = 2 (Entlassung), wenn D_i < min(fu_zeit_tage_i, 30), sonst
              0 (administrativ zensiert an Tag 30)
```

Die **Entlassungs-Hazard sinkt mit dem Schweregrad** (Kränkere bleiben länger).
Genau das macht im Fine-Gray-Modell die Subdistribution-HR für SOFA (≈ 1,51)
*größer* als die ursachenspezifische HR (≈ 1,40): höhere SOFA-Werte senken
zusätzlich die Entlassungschance und halten Patient:innen länger im Risikoset für
den Tod — ein Kanal, den Cause-Specific-Hazards ignorieren. Es gibt **keine harte
Obergrenze** bei Tag 20: `D_i` kann bis Tag 30 (und darüber) fallen, die
administrative Zensierung bei Tag 30 bleibt erhalten.

### Die zeitvariante Vasopressor-Exposition

```
p_treat_i   = clip(0.15 + 0.035 * sofa_score_i, 0.05, 0.85)   # P(je exponiert)
ever_i      ~ Bernoulli(p_treat_i)
start_tag_i = floor(Exp(mean = clip(6 - 0.25*sofa, 1, 6))) + 1 # früher bei hohem SOFA
```

Exposition und Startzeitpunkt hängen vom **Baseline-SOFA**, aber **nicht** vom
Outcome ab, sodass Tod in beiden Gruppen auftritt und das zeitabhängige
Cox-Modell ohne komplette Separation konvergiert. `vasopressor_start_tag` ist
gesetzt, wenn `ever_i == 1` und `start_tag_i < event_time_i`, sonst leer. Weil
Python und R **dieselbe** Spalte lesen, berechnen beide Sprachen identische HRs.

**Aktuelle Kennzahlen (Seed 42, N=500):** Zustände Tod/Entlassung/zensiert =
78/315/107. Wahre CIF des Todes an Tag 30 ≈ **15,6 %**, naives 1-KM (Entlassung
als Zensierung) ≈ **27,7 %** — sichtbar überschätzt, aber **begrenzt**
(Risikoset an Tag 30 ≈ 90, kein Kollaps). Cause-specific Cox: Alter 1,04, SOFA
1,40, Diabetes 0,99; Fine-Gray SHR: Alter 1,04, SOFA 1,51, Diabetes 0,94.
Zeitvarianter Vasopressor: naiv-zeitfix HR 0,73 (Immortal Time Bias), korrekt
zeitvariant HR 0,95.

Beide Prozesse sind seed-fest und nutzen **eigene RNG-Ströme** (`SEED + 2` für
die Entlassungszeit, `SEED + 3` für die Exposition), sodass die übrigen
Datensätze **unverändert** bleiben (`kohorte.csv`, `labor.csv`,
`vitalwerte.csv`, `notizen.csv`, `bga_ph_wahrheit.csv` bleiben byte-identisch).

## Klinische Notizen: synthetisch, mit fixem (aber angereichertem) Vokabular
`notizen.csv` wird aus einem festen Satz von Bausteinen zusammengesetzt:
6–8 Befund-Phrasen je `aufnahmegrund`, 8–10 Verschlechterungs- bzw.
Stabilitäts-Phrasen, ~22 % bewusst falsch-vorzeichenbehaftete oder
mehrdeutige Formulierungen als Rauschen, 2–3 Sätze mit variabler Eröffnung.
Das hält die Textklassifikation (Modul 31, TF-IDF + LogReg, 5-fach CV)
nicht-trivial (aktuelle AUC ≈ 0.82, Ziel-Korridor 0.72–0.82) und sorgt dafür,
dass die Top-Token nicht drei identische Fragmente derselben drei Sätze sind.
**Wichtiger Ehrlichkeits-Hinweis:** Weil das Vokabular synthetisch und fix
ist (Wörterbuch bekannter Phrasen statt echter klinischer Sprachvielfalt),
**überschätzt** diese demonstrierte AUC systematisch die tatsächlich zu
erwartende Performance auf echten, unstrukturierten klinischen Freitexten —
echte Notizen sind lexikalisch weit vielfältiger, fehlerbehafteter und
domänenspezifischer.

## Reale offene Datensätze (zum Weiterlernen, nicht Teil des Kurses)
Kein Modul verwendet diese Datensätze — der gesamte Kurs läuft mit den
synthetischen Daten oben. Für alle, die nach dem Kurs an echten Daten üben
wollen, als Ausblick:

- **NHANES** (US-Gesundheitssurvey, frei), `https://www.cdc.gov/nchs/nhanes/`,
 in R bequem über das CRAN-Paket `nhanesA`; in Python lädt man die SAS-Dateien
 direkt von der CDC-Seite (`pandas.read_sas`).
- **MIMIC-IV Clinical Database Demo** (100 Intensiv-Aufenthalte, frei nach
 kurzer Anmeldung), `https://physionet.org/content/mimic-iv-demo/`.
- **MIMIC-IV (Vollzugang)**, credentialed, erfordert CITI-Schulung; nur als Ausblick.
