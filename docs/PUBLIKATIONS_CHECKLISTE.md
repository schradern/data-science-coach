# Publikations-Checkliste für klinische Datenanalysen

Diese Checkliste orientiert sich an den internationalen Berichtsstandards (**EQUATOR Network**) und stellt sicher, dass deine statistischen Analysen und Machine-Learning-Modelle den Anforderungen hochrangiger medizinischer Fachzeitschriften (z. B. NEJM, Lancet, JAMA) entsprechen.

---

## 1. Studiendesign und Population (STROBE / CONSORT)
- [ ] **Patientenfluss (Flowchart):** Ist die Rekrutierung, der Ein- und Ausschluss von Patient:innen lückenlos dokumentiert? (z. B. Ausschluss wegen fehlender Laborwerte, Liegedauer < 24h).
- [ ] **Deskriptive Baseline (Table 1):** Werden stetige Variablen je nach Verteilung korrekt dargestellt? (Mittelwert ± Standardabweichung für normalverteilte, Median [IQR] für schiefe Daten).
- [ ] **Varianzberichterstattung:** Werden statt Standardfehlern (SEM) die Standardabweichungen (SD) zur Beschreibung der Streuung genutzt? (SEM misst die Präzision des Mittelwerts, nicht die Streuung der Population!).

---

## 2. Statistische Hypothesentests und Fehlerquellen
- [ ] **Effektstärken und Konfidenzintervalle:** Wird zu jedem signifikanten p-Wert die Effektstärke (z. B. Cohen's d, Odds Ratio, Hazard Ratio) inklusive des 95%-Konfidenzintervalls angegeben? (P-Werte allein beweisen keine klinische Relevanz!).
- [ ] **Multiple Testing:** Wurde bei der Prüfung mehrerer Hypothesen für Alpha-Fehler-Kumulierung korrigiert (z. B. Bonferroni für konfirmatorische, FDR/Benjamini-Hochberg für explorative Studien)?
- [ ] **Voraussetzungsprüfungen:** Wurden die Verteilungsannahmen der Tests (z. B. Normalität für t-Test, Varianzhomogenität) geprüft und ggf. nicht-parametrische Alternativen gewählt?
- [ ] **Der Median-Mythos:** Wurde beim Mann-Whitney-U-Test beachtet, dass er bei ungleichen Verteilungsformen stochastische Dominanz prüft und nicht zwingend den Median vergleicht?

---

## 3. Fehlende Werte (Missing Data)
- [ ] **Fehlenden-Profiling:** Ist für jede Variable der Prozentsatz fehlender Werte dokumentiert?
- [ ] **Mechanismus-Diskussion:** Wurde begründet, ob die Daten MCAR, MAR oder MNAR sind?
- [ ] **Imputationsverfahren:** Falls Multiple Imputation (z. B. MICE) genutzt wurde: Wurden die Variablen im Imputationsmodell benannt und die Anzahl der Datensätze angegeben?
- [ ] **Sensitivitätsanalyse:** Wurden die Ergebnisse der Komplettfall-Analyse (Complete Case) mit den imputierten Daten verglichen und Abweichungen diskutiert?

---

## 4. Regressionsmodelle und Confounding
- [ ] **Causal Directed Acyclic Graphs (DAGs):** Wurden Confounder auf Basis klinischen Vorwissens und kausaler Pfade (DAGs) ausgewählt, statt rein nach statistischer Signifikanz im univariablen Modell?
- [ ] **Events per Variable (EPV):** Ist die Anzahl der Ereignisse im Verhältnis zu den Modellparametern ausreichend? (Faustregel: Mindestens 10–15 Ereignisse pro Prädiktor im logistischen und Cox-Modell, um Overfitting zu verhindern).
- [ ] **Immortal Time Bias:** Falls Behandlungen im Verlauf der Nachbeobachtung initiiert wurden: Wurden diese als zeitabhängige Kovariaten modelliert oder Landmark-Analysen durchgeführt, statt sie ab Tag 0 der Behandlungsgruppe zuzuordnen?
- [ ] **Modellannahmen:** Wurde die Linearität stetiger Variablen (z. B. über spline-Modelle) und bei der Cox-Regression die Proportional-Hazards-Annahme (Schoenfeld-Residuen) geprüft?

---

## 5. Machine Learning und Validierung (TRIPOD)
- [ ] **Daten-Leakage:** Ist ausgeschlossen, dass Preprocessing-Schritte (Skalierung, Imputation) vor dem Train/Test-Split auf dem gesamten Datensatz durchgeführt wurden? (Pipelines verwenden!).
- [ ] **Diskrimination:** Werden neben der ROC-Kurve (AUC) auch Precision-Recall-Kurven (PR-AUC) berichtet, wenn das Zielereignis selten ist (Klassenungleichgewicht)?
- [ ] **Kalibrierung:** Wird die Übereinstimmung von vorhergesagten Wahrscheinlichkeiten und realen Ereignisraten (Kalibrierungskurve, Brier Score) dargestellt? (Nur eine gute Kalibrierung erlaubt den sicheren klinischen Einsatz).
- [ ] **Klinischer Nutzen (DCA):** Wurde eine Decision-Curve-Analyse (DCA) durchgeführt, um den Net Benefit des Modells gegenüber den Standardstrategien („alle behandeln“ vs. „niemanden behandeln“) zu zeigen?
- [ ] **Validierungsstrategie:** Wurde das Modell über Kreuzvalidierung oder Bootstrapping intern validiert? (Optimal ist eine zusätzliche externe Validierung an einer unabhängigen Kohorte).

---

## 6. Reproduzierbarkeit und Transparenz
- [ ] **Seed-Fixierung:** Wurden alle Zufallszahlengeneratoren für Splits und stochastische Algorithmen fixiert (`seed`), um exakte Reproduzierbarkeit zu garantieren?
- [ ] **Code und Daten-Freigabe:** Sind der Analyse-Code (z. B. als Quarto- oder R-Markdown-Dokument) und die anonymisierten Daten in einem öffentlichen Repository (z. B. GitHub, OSF, Zenodo) hinterlegt?
- [ ] **Umgebungskonsistenz:** Wurde die Software-Umgebung (z. B. über `requirements.txt` oder `renv.lock`) für Dritte nachvollziehbar dokumentiert?
