# 32 · Modelleinsatz, Monitoring und Governance

> **Teil 5 – Machine Learning und KI in der Medizin** · Dauer: ~65 min · Voraussetzung: Module 24–25 (Workflow und Validierung)

## Lernziele
- Ein Modell mit **joblib** reproduzierbar persistieren (Versionspinning) und als deutsche **Model Card** dokumentieren.
- **Kalibrierungsdrift** und **Dataset Shift** unterscheiden und an einer Konzept-Drift-Simulation zeigen, wie die AUC degradiert (Kalibrierungsdrift bei stabiler AUC: Modul 34).
- **Subgruppenleistung** (AUC nach `geschlecht` und Altersgruppe) mit **Bootstrap-Konfidenzintervallen** berechnen und Fairness-Implikationen ableiten.
- Die regulatorischen Anforderungen von **MDR 2017/745** und **EU AI Act** für klinische Prädiktionsmodelle in Grundzügen kennen.
- Praktische Governance-Fragen jenseits der Regulatorik einordnen: **Automation Bias**, **Haftung** und **informierte Einwilligung**.

## Klinischer Aufhänger
Das Modell läuft, die AUC im Validierungsset war gut. Jetzt beginnt der schwierigste Teil: der Einsatz in der Realität. Patient:innenpopulationen ändern sich, Jahreszeiteneffekte, neue Behandlungsprotokolle, veränderte Dokumentationspraxis. Ein Modell, das niemand überwacht, verliert still seine Leistung, und niemand merkt es, bis Schaden entsteht. Dieses Modul zeigt, wie du ein Modell deployst, beobachtest und verantwortlich einbettest.

## 1 Modelle persistieren mit joblib
Ein trainiertes scikit-learn-Objekt wird mit `joblib.dump` als komprimierte Binärdatei gespeichert. Beim Laden muss die **exakt gleiche scikit-learn-Version** vorhanden sein, sonst drohen stille Deserialisierungsfehler.

<!-- excerpt-note -->
> **Hinweis – die Codeblöcke unten sind Auszüge.** Sie zeigen die entscheidenden Schritte, nicht jede Import-Zeile. Das vollständige, am Stück lauffähige Skript ist [`code/python.py`](code/python.py) (im Browser über den **Python**-Reiter oben). In Colab führst du das ganze Modul mit einer Zeile aus: `!python module/32-einsatz-governance/code/python.py` – siehe das [in Colab öffnen](https://colab.research.google.com/github/schradern/data-science-coach/blob/main/notebooks/start.ipynb).

```python
# Excerpt from code/python.py: runs inside the full script, not on its own.
import joblib

# Save
joblib.dump({"pipeline": pipe, "sklearn_version": sklearn.__version__}, "modell.joblib")

# Load and check the version
artefakt = joblib.load("modell.joblib")
assert artefakt["sklearn_version"] == sklearn.__version__, "Versionsmismatch!"
pipe_loaded = artefakt["pipeline"]
```

Für den Produktiveinsatz gilt: das Modell zusammen mit einer `requirements.txt` (inkl. Versions-Pin) einfrieren und in einer Modell-Registry (z. B. MLflow oder DVC) versionieren.

**Stolperstein:** `pickle` (das `joblib` intern nutzt) ist **nicht sicher** für fremde Binärdaten, eine schädliche `.joblib`-Datei kann Code ausführen. Lade nur Modelle aus vertrauenswürdigen Quellen, idealerweise mit kryptographischer Prüfsumme.

**Darüber hinaus:** ONNX ermöglicht Framework-übergreifende Modell-Serialisierung (scikit-learn → ONNX → Java-Backend). Für regulierte Umgebungen bietet das zusätzliche Reproduzierbarkeit und Entkopplung vom Python-Ökosystem.

## 2 Model Card, der Steckbrief des Modells
Eine **Model Card** dokumentiert, was das Modell tut, für wen es gebaut wurde, wie es bewertet wurde und wo es versagt. Gesetzlich vorgeschrieben ist sie nicht, aber sie gehört zu verantwortungsvoller KI und liefert die Grundlage für die regulatorische Dokumentation.

```markdown
## Model Card – Klinisches Verschlechterungs-Prädiktionsmodell

**Modellname:** Klinische-Verschlechterung-v1  
**Version:** 1.0.0  
**Datum:** 2026-06  
**Verantwortlich:** [Klinik/Team, Kontakt]

### Aufgabe
Binäre Klassifikation: Vorhersage klinischer Verschlechterung innerhalb 24h
auf Basis der freitextlichen Pflege-/Arztnotizen (TF-IDF-Repräsentation der
dokumentierten Beobachtungen), nicht auf Basis strukturierter Vitalwerte oder
Laborwerte. Das Modell "liest" also Text, keine Zahlen-Tabelle.

### Trainingsdaten
Synthetischer Textdatensatz (n=500 Notizen), Aufnahmezeitraum simuliert.
Outcome: verschlechterung (0/1), ~35 % positiv (34,6 % im aktuellen Datenstand).
Kein Echtpatientendatensatz; klinischer Einsatz erfordert Validierung.

### Modellarchitektur
TF-IDF (Unigramme + Bigramme, min_df=2) + Logistische Regression (C=1.0,
class_weight=balanced). Pipeline, kein Pre-Leakage.

### Leistungsmetriken (Testset, 25 %)
- ROC-AUC: [Wert einsetzen]  
- Sensitivität @ Youden-Schwelle: [Wert]  
- Spezifität @ Youden-Schwelle: [Wert]

### Subgruppenleistung
AUC nach Geschlecht (jeweils mit 95%-Bootstrap-CI): weiblich [Wert, CI],
männlich [Wert, CI]. AUC nach Altersgruppe (Schwelle 65 Jahre): [Wert, CI].
Ein reiner Punktschätzer-Unterschied belegt keine Lücke; maßgeblich ist das
Konfidenzintervall der AUC-Differenz zwischen den Gruppen.

### Einschränkungen und Risiken
- Bag-of-Words ignoriert Negation (z. B. „kein Fieber“).
- Kalibrierung nicht validiert auf externen Kohorten.
- Sprache: Deutsch. Andere Dokumentationssprachen nicht unterstützt.
- Keine prospektive Validierung durchgeführt.

### Geplante Nutzung
Forschungszwecke / Pilotprojekt unter Aufsicht.  
NICHT für autonome klinische Entscheidungen ohne Human-in-the-Loop.

### Regulatorik
Klinische Entscheidungsunterstützung → MDR 2017/745: typischerweise Klasse IIa,
unter Regel 11 aber Klasse IIb bei möglicher schwerwiegender Verschlechterung
und Klasse III, wenn die unterstützte Entscheidung Tod oder irreversible
Verschlechterung verursachen kann. Ein 30-Tage-Mortalitäts-CDSS ist
ein IIb/III-Kandidat. EU AI Act: damit Hochrisiko, weil eine Benannte Stelle
beteiligt ist. (Keine Rechtsberatung; vgl. §5.)
Vor Produktiveinsatz: CE-Kennzeichnung, klinische Evaluierungsstudie, PMS-Plan.
```

**Stolperstein:** Eine Model Card ist kein Marketing-Dokument. Sie muss Grenzen und Versagensmodi ehrlich benennen, auch bei internen Projekten. Eine geschönte Model Card erhöht das Deployment-Risiko.

**Darüber hinaus:** Der ursprüngliche [Model-Card-Standard (Mitchell et al., 2019)](https://arxiv.org/abs/1810.03993) und die Model Cards offener Modell-Plattformen folgen ähnlichen Strukturen. EU AI Act Art. 11 fordert eine technische Dokumentation mit ähnlichem Umfang für Hochrisiko-KI.

## 3 Monitoring: Kalibrierungsdrift und Dataset Shift
Modelle altern. Zwei Hauptursachen:

**Kalibrierungsdrift** (*calibration drift*): Die ausgegebenen Wahrscheinlichkeiten stimmen nicht mehr mit der beobachteten Häufigkeit überein, weil sich die Ereignisrate in der Realität ändert. Die AUC kann dabei stabil bleiben; Modul 34 simuliert genau das und misst O:E.

**Dataset Shift:** Die Merkmalsverteilung verändert sich (Covariate Shift) oder der Zusammenhang zwischen Merkmalen und Outcome ändert sich (Concept Drift).

Wir simulieren **Konzept-Drift** durch gezieltes Label-Flipping: Ein wachsender Anteil der Testlabels wird invertiert. Das modelliert den Fall, in dem sich die Beziehung zwischen Textsignal und Outcome verändert, etwa weil geänderte Dokumentationspraktiken oder Therapieprotokolle die alten Signal-Tokens unzuverlässig machen. Das ist eine grobe
Vereinfachung: Zufälliges Vertauschen von Labels ist symmetrisches Rauschen, echte
Konzept-Drift betrifft meist bestimmte Muster. Für das Prinzip, dass nur
nacherfasste Labels einen AUC-Verlust sichtbar machen, reicht sie.

```python
# Excerpt from code/python.py: runs inside the full script, not on its own.
import numpy as np

rng = np.random.default_rng(SEED)

def simulate_concept_drift(y_test, flip_rate, rng):
    """Flip flip_rate fraction of labels to simulate concept drift."""
    y_noisy = y_test.copy()
    n_flip = int(flip_rate * len(y_test))
    if n_flip > 0:
        flip_idx = rng.choice(len(y_test), size=n_flip, replace=False)
        y_noisy.iloc[flip_idx] = 1 - y_noisy.iloc[flip_idx]
    return y_noisy

for flip_rate in [0.00, 0.05, 0.10, 0.15, 0.20]:
    y_drifted = simulate_concept_drift(y_test, flip_rate, rng)
    auc = roc_auc_score(y_drifted, pipe.predict_proba(X_test)[:, 1])
    print(f"Drift {flip_rate:.0%}: AUC = {auc:.3f}")
```

![Abbildung: AUC über fünf simulierte Monitoring-Zeitpunkte mit zunehmendem Konzept-Drift (Label-Flip-Rate 0–20 %). Die AUC fällt monoton von 0,85 (T1) auf ~0,64 (T5) und unterschreitet erst am letzten Zeitpunkt (T5) die Akzeptanzgrenze von 0,70. In der Praxis ist dieser Abfall nur erkennbar, wenn Outcome-Labels in der Produktion nacherfasst werden.](assets/drift_monitoring.png)

**Stolperstein:** Eine degradierte AUC ist nur erkennbar, wenn **Outcome-Labels** auch in der Produktion nacherfasst werden. Ohne Ground-Truth-Labels lässt sich nur Covariate Shift (Merkmalsverteilung) überwachen, nicht Concept Drift.

**Darüber hinaus:** Population Stability Index (PSI) ist ein verbreitetes Maß für Covariate Shift. Für unlabeled Production-Data eignen sich Maximum-Mean-Discrepancy (MMD) oder domänenadaptive Verfahren.

## 4 Fairness: Subgruppenleistung
Ein gutes mittleres AUC kann starke Leistungsunterschiede zwischen Subgruppen verdecken. Im klinischen Kontext ist Gleichheit nach `geschlecht`, Altersgruppe oder Ethnizität (soweit dokumentiert) eine ethische und zunehmend regulatorische Anforderung. Bei kleinen Subgruppen ist die AUC-Schätzung aber hochvariabel – **ein Bootstrap-Konfidenzintervall ist deshalb Pflicht**, nicht optional, bevor eine Lücke als real gilt.

```python
# Excerpt from code/python.py: runs inside the full script, not on its own.
import numpy as np
from sklearn.metrics import roc_auc_score

def bootstrap_auc_ci(y, proba, n_boot=1000, seed=42):
    """95%-Bootstrap-CI für AUC durch Resampling mit Zurücklegen."""
    rng = np.random.default_rng(seed)
    y, proba = np.asarray(y), np.asarray(proba)
    boot_aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(y[idx], proba[idx]))
    return np.percentile(boot_aucs, [2.5, 97.5])

for group in ["weiblich", "maennlich"]:
    mask = cohort_test["geschlecht_clean"] == group
    if mask.sum() < 10:
        continue
    y_g, p_g = y_test[mask], proba[mask]
    auc_g = roc_auc_score(y_g, p_g)
    lo, hi = bootstrap_auc_ci(y_g, p_g)
    print(f"  AUC {group}: {auc_g:.3f}  [95%-CI {lo:.3f}-{hi:.3f}]  (n={mask.sum()})")
```

![Abbildung: AUC nach Geschlechtsgruppe für den Textklassifikator mit 95%-Bootstrap-Fehlerbalken. Die Referenzlinie zeigt die Gesamt-AUC. Ein Balken wird nur dann rot markiert, wenn sein Konfidenzintervall die Gesamt-AUC nicht mehr einschließt – bei den hier gemessenen Werten (Weiblich 0,88 [0,79–0,95]; Männlich 0,81 [0,69–0,91]) schließen beide Intervalle die Gesamt-AUC von 0,85 ein, die Farbe bleibt daher neutral.](assets/subgruppen_auc.png)

Konkret gemessen auf dem Testsplit (n=125, Gesamt-AUC 0,849): **Weiblich** AUC 0,880, 95%-CI [0,787–0,952] (n=61); **Männlich** AUC 0,807, 95%-CI [0,689–0,909] (n=64). Der Punktschätzer-Unterschied (Δ = 0,072) sieht auf den ersten Blick nach einer Lücke aus. Entscheidend ist das Konfidenzintervall der **Differenz**, das das Skript per Bootstrap rechnet (beide Gruppen getrennt neu gezogen): [−0,064; +0,212]. Es schließt 0 ein. **Bei n≈60 pro Gruppe gibt es keinen Nachweis einer Lücke, aber ebenso wenig einen Nachweis von Gleichheit**: Auch ein Unterschied von 0,2 AUC-Punkten ist mit den Daten vereinbar. Eine ehrliche Model Card schreibt deshalb „keine belastbare Evidenz für eine Geschlechter-Disparität, Untersuchung an größerer Stichprobe nötig“, nicht „Modell diskriminiert nach Geschlecht“ und auch nicht „Modell ist fair“.

**Praxis – Worked Example Altersgruppe:** Dieselbe Analyse lässt sich auf jede andere klinisch relevante Subgruppe anwenden. Mit der Schwelle 65 Jahre (gängiger geriatrischer Cutoff) ergibt sich auf demselben Testsplit: **<65 Jahre** AUC 0,843, 95%-CI [0,733–0,940] (n=57); **≥65 Jahre** AUC 0,842, 95%-CI [0,734–0,930] (n=68). Hier liegen die Punktschätzer praktisch identisch (Differenz +0,002, 95-%-KI [−0,144; +0,145]): kein Hinweis auf eine Alters-Disparität, aber das breite Intervall zeigt, wie wenig zwei Gruppen von je etwa 60 Patient:innen über Fairness aussagen können.

**Stolperstein:** Zwei sich überlappende Konfidenzintervalle sind **kein** Beweis für Gleichheit – zwei Gruppen können sich signifikant unterscheiden, obwohl ihre einzelnen Intervalle überlappen (siehe Modul 10 §3). Sauber gilt nur die eine Richtung: überlappen die Intervalle **nicht**, ist die Lücke fast sicher real. Für eine belastbare Aussage über die *Differenz* gehört deshalb das Konfidenzintervall der Differenz berechnet, wie im Skript – und selbst ein deutlicher Unterschied bräuchte eine größere, unabhängige Kohorte zur Bestätigung.

**Darüber hinaus:** Über AUC-Gleichheit hinaus gibt es Fairness-Kriterien wie Equal Opportunity (gleiche Sensitivität) oder Calibration across groups. Diese können in Konflikt stehen, die Wahl des Fairness-Kriteriums ist eine normative Entscheidung.

## 5 Governance: MDR und EU AI Act
Klinische Prädiktionsmodelle sind in Europa reguliert. Zwei Rahmen sind relevant:

**MDR 2017/745 (Medical Device Regulation):** Ein Software-Produkt ist ein Medizinprodukt, wenn der Hersteller es zur Diagnose, Verhütung, Überwachung, Vorhersage, Prognose oder Behandlung von Krankheiten bestimmt (Zweckbestimmung, Art. 2 Nr. 1 MDR). Klinische Entscheidungsunterstützung fällt typischerweise unter **Klasse IIa** (mittleres Risiko) oder **Klasse IIb/III** (hohes Risiko). Anforderungen: Qualitätsmanagementsystem (ISO 13485), klinische Bewertung, Post-Market Surveillance Plan, CE-Kennzeichnung, Unique Device Identifier. Für Eigenentwicklungen, die nur innerhalb derselben Gesundheitseinrichtung eingesetzt werden, gibt es eine eng begrenzte Ausnahme von der CE-Kennzeichnung (Art. 5 Abs. 5 MDR); sie verlangt unter anderem ein Qualitätsmanagement, eine Dokumentation und die Begründung, warum kein gleichwertiges Produkt auf dem Markt den Bedarf deckt.

**EU AI Act (Regulation (EU) 2024/1689):** Die Verordnung ist seit
1. August 2024 in Kraft und wird stufenweise anwendbar. Klinische KI kann
insbesondere dann **Hochrisiko-KI** sein, wenn sie selbst ein Medizinprodukt ist
oder als Sicherheitskomponente eines Medizinprodukts dient **und** dieses nach der
MDR eine Konformitätsbewertung durch eine Benannte Stelle braucht (Art. 6 Abs. 1,
in der Praxis ab Klasse IIa), oder wenn sie in einem in Anhang III gelisteten
Kontext eingesetzt wird. Die konkrete Einordnung hängt vom
Verwendungszweck, vom Produktkontext und von aktueller EU-/MDCG-Leitlinie ab.
Für Lernzwecke genügt hier: patient:innenrelevante Entscheidungsunterstützung
niemals als „nur Software“ behandeln. Typische Pflichten sind:

- Transparenz: Der Anbieter liefert eine Gebrauchsanweisung mit Leistung und Grenzen des Systems (Art. 13). Davon getrennt gilt für alle KI-Systeme, nicht nur für Hochrisiko-KI: Wer direkt mit einem KI-System interagiert, muss das erkennen können (Art. 50).
- Menschliche Aufsicht: Entscheidungen müssen von qualifiziertem Personal überschaubar und übersteuerbar sein (Art. 14).
- Robustheit und Genauigkeit: Nachgewiesene Leistung über den gesamten Lebenszyklus (Art. 15).
- Konformitätsbewertung und technische Dokumentation vor Inbetriebnahme, wenn
  das System in den Hochrisiko-/Medizinproduktbereich fällt.

**Praktische Konsequenzen für Datenwissenschaftler:innen:**

- Kein Deployment ohne vollständige Dokumentation (Model Card + technische Akte).
- Prospektive klinische Studie vor Produktiveinsatz.
- Monitoring-Plan mit definierten Schwellenwerten für Intervention.
- Changelog und Audit-Trail für alle Modellversionen.

**Stolperstein:** „Research Use Only“ schützt nicht, wenn das Tool tatsächlich klinische Entscheidungen beeinflusst. Die Klassifikation richtet sich nach dem **Verwendungszweck**, nicht nach dem Entwicklungskontext.

**Darüber hinaus:** Prüfe vor realem Einsatz den aktuellen Stand von MDR, MDCG
2019-11, EU-AI-Act-Leitlinien und harmonisierten Normen. Die Fristen und
Auslegungshilfen ändern sich; binde Regulatory Affairs, klinische Informatik,
Datenschutz und Ethikkommission früh ein.

## 6 Weitere Praxisfragen: Automation Bias, Haftung, Aufklärung
Regulatorik allein macht ein KI-System nicht sicher. Drei weitere Fragen betreffen unmittelbar den klinischen Alltag:

**Automation Bias und Alert Fatigue:** Klinisches Personal neigt dazu, Modellvorhersagen unkritisch zu übernehmen (Automation Bias), besonders unter Zeitdruck. Umgekehrt führt eine hohe Rate falsch-positiver Alarme dazu, dass Warnungen zunehmend ignoriert werden (Alert Fatigue), ein Modell mit vielen Fehlalarmen „verbrennt“ das Vertrauen, das es eigentlich unterstützen soll. Beides untergräbt den Zweck der Entscheidungsunterstützung, in entgegengesetzte Richtungen.

**Praxis:** Gegenmaßnahmen: Schwellenwerte an die tatsächliche Fehlalarmtoleranz der Station anpassen (nicht nur an Youden-Index optimieren), regelmäßiges Feedback an das Team, wie oft ein Alarm zutraf, und Schulung, dass ein Modell-Score ein Hinweis ist, keine Diagnose.

**Haftung:** Wer haftet, wenn eine KI-gestützte Entscheidung einem Patienten schadet, die behandelnde Person, die Modellentwicklerin oder die Klinik als Betreiberin? Die Antwort ist **je nach Rechtsraum unterschiedlich und rechtlich nicht abschließend geklärt**; dieses Modul kann und will hier keine Rechtsberatung ersetzen. In der Praxis bleibt die behandelnde Person in aller Regel verantwortlich für die letztendliche Entscheidung (Human-in-the-Loop ist auch deshalb keine Formalität), während Hersteller- und Betreiberhaftung zusätzlich über Produkthaftungs- und Medizinprodukterecht greifen können. Vor Produktiveinsatz gehört diese Frage explizit zur Rechtsabteilung und Klinikleitung, nicht zur Data-Science-Entscheidung.

**Informierte Einwilligung:** Sollten Patient:innen erfahren, dass ein KI-System an ihrer Behandlungsentscheidung beteiligt war? Der EU AI Act verlangt eine Information der Betroffenen ausdrücklich nur für bestimmte Hochrisiko-Anwendungen aus Anhang III (Art. 26 Abs. 11); bei KI in Medizinprodukten stellt sich die Frage vor allem über die ärztliche Aufklärungspflicht (§ 630e BGB), deren Reichweite für KI-Unterstützung noch nicht abschließend geklärt ist. Praktisch heißt das: Aufklärungsbögen und Behandlungsgespräche sollten benennen, dass ein unterstützendes Vorhersagemodell eingesetzt wird, in verständlicher Sprache, ohne technische Details vorauszusetzen. Wie genau (mündlich, schriftlich, im Rahmen der allgemeinen Aufnahmeaufklärung) ist institutionell und je nach Risikoklasse zu klären.

> **Wann du Hilfe holst.** Sobald ein Modell reale Behandlungsentscheidungen beeinflusst, sind Rechtsabteilung, Datenschutz, Ethikkommission und gegebenenfalls Benannte Stelle keine Formalie, sondern Voraussetzung – spätestens bei nachgewiesenem Drift oder ungleicher Subgruppenleistung gehört die Weiternutzung gemeinsam mit diesen Stellen entschieden, nicht allein technisch.

## Fallstricke und Merksätze
- **Versionspinning ist Pflicht.** Eine joblib-Datei ohne Versionsinfo ist eine Zeitbombe.
- **Model Card = Ehrlichkeits-Dokument.** Grenzen und Versagen klar benennen.
- **AUC-Monitoring braucht Labels.** Ohne Outcomes sieht man nur Covariate Shift.
- **Subgruppen-AUC ist nicht optional.** Mittlere Güte verdeckt systematische Ungleichheit,
  aber erst ein Konfidenzintervall sagt dir, ob eine gemessene Lücke real oder Rauschen ist.
- **MDR + EU AI Act früh prüfen.** Ob ein System ein Medizinprodukt oder
  Hochrisiko-KI ist, folgt aus Verwendungszweck und Einsatzkontext.
- **Automation Bias, Haftung und Aufklärung mitdenken.** Governance endet nicht bei der Regulatorik-Checkliste, sie betrifft, wie Menschen dem Modell im Alltag begegnen.

> **Merksatz:** Keines der Modelle in diesem Kurs ist einsatzbereit, auch nicht das
> Modell dieses Moduls. Alle Modelle wurden auf **synthetischen Daten** (n=500,
> künstlich generiert) trainiert, ohne prospektive Validierung, ohne externe
> Validierung an einer echten Kohorte und ohne regulatorische Zulassung. Sie
> dienen ausschließlich der Vermittlung von Methodik. Jeder Schritt von hier zu
> einem realen klinischen Einsatz, echte Daten, externe Validierung, CE-Kennzeichnung,
> prospektive Studie, Monitoring-Infrastruktur, ist in diesem Kurs bewusst ausgeklammert
> und erfordert ein eigenes, mehrjähriges Projekt mit entsprechender Governance.

## Selbstcheck
1. **Was muss beim Laden eines joblib-Modells immer geprüft werden?** – Die scikit-learn-Version; ein Versionsmismatch kann stille Fehler oder falsche Vorhersagen produzieren.
2. **Warum reicht Covariate-Shift-Monitoring allein nicht?** – Weil Concept Drift (veränderte Outcome-Beziehung) nur mit echten Labels erkennbar ist. Merkmalsverteilung kann stabil sein, während das Modell trotzdem versagt.
3. **Was bedeutet gleiche AUC, aber unterschiedliche Sensitivität für zwei Subgruppen – und wann ist eine Subgruppen-Differenz überhaupt belegt?** – Gleiche AUC heißt gleich gute Trennung innerhalb jeder Gruppe, nicht gleiche Score-Verteilung oder Kalibrierung. Bei einer festen Schwelle können Sensitivität und Spezifität deshalb trotzdem auseinanderlaufen: ungleiche Identifikationsraten. Ein Punktschätzer-Unterschied allein belegt keine Fairness-Lücke – die AUC ist in kleinen Subgruppen hochvariabel; erst ein Konfidenzintervall der AUC-Differenz, das 0 ausschließt, spricht für mehr als Stichprobenrauschen (nicht überlappende Einzelintervalle sind ein strengeres, konservatives Kriterium).
4. **Ab welcher EU-AI-Act-Risikoklasse gelten die strengsten Pflichten?** – Bei Hochrisiko-KI; ob ein klinisches Prädiktionsmodell darunter fällt, hängt vom Verwendungszweck und Produktkontext ab.
5. **Was ist Alert Fatigue, und warum widerspricht sie Automation Bias?** – Alert Fatigue ist das Ignorieren von Warnungen nach zu vielen Fehlalarmen; Automation Bias ist das unkritische Übernehmen von Modellvorhersagen. Beide untergraben sinnvolle Entscheidungsunterstützung, aber in entgegengesetzte Richtungen, zu viel bzw. zu wenig Vertrauen ins Modell.
