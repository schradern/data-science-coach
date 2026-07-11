# 22 · Reproduzierbare Berichte mit Quarto

> **Teil 4 – Statistische Inferenz und medizinische Statistik** · Dauer: ~60 min · Voraussetzung: Module 06, 08–10, 12 (Aufbereitung bis Regression)

## Lernziele
- Verstehen, warum **Reproduzierbarkeit** in der klinischen Forschung keine Kür, sondern eine wissenschaftliche Pflicht ist.
- Eine bewährte **Projektstruktur** anlegen und mit Git versionieren.
- **Datenschutz-Hygiene** im Versionskontrollsystem einhalten: was nie in ein Repository darf.
- Analyse und Bericht **aus einer einzigen Quelle** erzeugen (Quarto).
- **Seeds, relative Pfade und Paketversionen** sichern, damit Ergebnisse langfristig nachvollziehbar bleiben.

## Klinischer Aufhänger

Deine Analyse ist fertig, und sechs Monate später fragt ein Gutachter: *„Kannst du zeigen, wie genau Tabelle 2 entstanden ist?"* Ohne reproduzierbaren Workflow lautet die ehrliche Antwort oft: *„Nicht mehr genau."* Nicht nachvollziehbare Ergebnisse gefährden Folgepublikationen, Peer-Review-Entscheidungen und letztlich klinische Schlussfolgerungen, die auf deiner Arbeit beruhen.

## 1 Reproduzierbarkeit: was, warum, wie

**Was** bedeutet reproduzierbar? Dieselbe Analyse, von einer anderen Person oder von dir selbst in zwei Jahren, auf demselben Datensatz, liefert **bitgenau dasselbe Ergebnis.**

**Warum** scheitert das so häufig?

| Ursache | Konsequenz |
|---------|------------|
| Zufallszahlen ohne Seed | Ergebnisse ändern sich bei jedem Lauf |
| Absolute Pfade (`C:\Users\…`) | Skript läuft nur auf einem Rechner |
| Paketversionen nicht dokumentiert | Update bricht die Analyse still |
| Manuelle Zwischenschritte (Excel-Kopieren) | Nicht rekonstruierbar |
| PHI im Repository | Datenschutzverletzung, rechtliche Konsequenzen |

**Wie** lösen wir es? Mit drei Prinzipien:

1. **Alles Code**, kein Klicken, kein Kopieren.
2. **Seed setzen**, Zufall muss deterministisch sein.
3. **Umgebung dokumentieren**, welche Pakete in welcher Version.

**Stolperstein:** Der häufigste Fehler ist nicht das vergessene Seed, sondern der *selektive* Workflow: Daten erst in Excel bereinigt, dann in Python geladen. Dieser Excel-Schritt ist unsichtbar und nicht reproduzierbar, er bricht die gesamte Kette. Sobald eine manuelle Editierung existiert, ist die Analyse nicht mehr vollständig nachvollziehbar.

**Darüber hinaus:** Containerisierung (Docker, Singularity) sichert die gesamte Softwareumgebung inklusive Betriebssystem-Bibliotheken. Für Publikationen in Hochrisiko-Journals ist das zunehmend gefordert; für die meisten klinischen Projekte reicht eine virtuelle Python-Umgebung mit dokumentierten Versionen.

## 2 Projektstruktur

Eine bewährte Struktur für ein klinisches Analyseprojekt:

```
mein-projekt/
├── README.md              # Kurzbeschreibung, Ansprechpersonen, Laufanleitung
├── .gitignore             # was Git ignorieren soll (s. u.)
├── data/
│   ├── roh/               # unveränderliche Rohdaten — nie committen!
│   └── aufbereitet/       # durch Skripte erzeugt, ggf. committen
├── code/
│   ├── 01_bereinigung.py
│   ├── 02_analyse.py
│   └── 03_modell.py
├── berichte/
│   └── bericht.qmd        # Quarto-Dokument (Text + Code)
└── ausgaben/
    ├── tabellen/
    └── figures/
```

Die nummerierten Skripte machen die Ausführungsreihenfolge sofort erkennbar. Wer vier Monate nach Abgabe zurückkommt, weiß sofort, wo er anfangen muss.

**Stolperstein:** Versionierung durch Dateinamen (`analyse_FINAL_v3_neu.py`) ist kein Ersatz für Git. Sobald zwei Personen am Projekt arbeiten, sind Namenskollisionen und verlorene Änderungen unvermeidlich. Git ist kein Luxus, sondern Grundlage kollaborativer Reproduzierbarkeit.

**Darüber hinaus:** Für komplexere Projekte lohnt ein `Makefile` oder `dvc` (Data Version Control), das die Abhängigkeiten zwischen Skripten explizit macht und nur die tatsächlich veränderten Schritte neu ausführt, besonders nützlich bei langen Preprocessing-Läufen.

## 3 Datenschutz-Hygiene: was nie ins Repository darf

Die wichtigste Regel: **Keine Patientendaten in Git.**

Erstelle eine `.gitignore`-Datei im Projektstamm, **vor dem ersten Commit**:

```gitignore
# Rohdaten und Dateien mit potenziellen PHI (Protected Health Information) -
# nur der Rohdaten-Ordner, NICHT alle CSVs im Projekt (data/aufbereitet/
# darf laut Projektstruktur oben committet werden)
data/roh/**
data/roh/*.csv
data/roh/*.xlsx
data/roh/*.xls
data/roh/*.sas7bdat
data/roh/*.dta

# Jupyter-Notebook-Ausgaben können Datenfragmente enthalten
.ipynb_checkpoints/

# Betriebssystem-Metadaten
.DS_Store
Thumbs.db
```

**Stolperstein:** Eine pauschale Regel wie `*.csv` (statt `data/roh/*.csv`) würde **jede** CSV im Projekt ignorieren, auch die aufbereiteten Ergebnisse in `data/aufbereitet/`, die laut Projektstruktur oben bewusst committet werden sollen. Ein späteres `!data/aufbereitet/*.csv` hilft hier nicht zuverlässig: Ist ein übergeordneter Ordner bereits ignoriert, greifen Negationen in `.gitignore` nicht mehr. Am robustesten: von Anfang an gezielt nur `data/roh/` ausschließen, nicht pauschal nach Dateityp.

Zusätzliche Schutzmaßnahmen:

- Rohdaten nur auf genehmigten Servern oder verschlüsselten Laufwerken speichern.
- Pseudonymisierung vor jeder Weitergabe, auch intern.
- Aggregierte Ergebnisse (Tabellen, Grafiken ohne Einzelfälle) können in der Regel geteilt werden; prüfe aber deine Einrichtungsregeln und das Ethikvotum.

**Stolperstein:** Git verfolgt die *gesamte Geschichte* aller Dateien. Eine einmal committete Datei lässt sich aus dem aktuellen Stand löschen, bleibt aber in `git log --all` sichtbar. Vollständiges Entfernen erfordert `git filter-repo` und ein erzwungenes Re-Push auf allen Klonen, erheblicher Aufwand mit Risiken für alle Mitarbeitenden. Besser: von Anfang an konsequent mit `.gitignore` arbeiten.

**Darüber hinaus:** Automatisierte Pre-Commit-Hooks (z. B. `detect-secrets` oder `git-secrets`) scannen jeden Commit nach Mustern, die wie PHI oder API-Schlüssel aussehen, und blockieren ihn, bevor der Schaden entsteht. In Teams sind solche Hooks eine sinnvolle Pflicht.

## 4 Quarto: Analyse und Bericht aus einer Quelle

**Quarto** ist ein modernes Format (`.qmd`), das Fließtext (Markdown) und Code (Python oder R) in einem Dokument vereint. `quarto render` führt alle Code-Chunks aus und bettet die Ergebnisse direkt ein.

![Abbildung: Eine Quelle, viele Formate, aus einer .qmd-Datei (Code, Text und Ergebnis vereint) erzeugt quarto render zugleich HTML, PDF und Word.](assets/eine_quelle.svg)

Warum das besser ist als „Kopieren aus dem Skript":

- Tabelle und der Code, der sie erzeugt, stehen **am selben Ort**.
- Datensatz geändert? Einfach neu rendern, die Tabelle aktualisiert sich automatisch.
- Gutachter erhalten den Bericht inklusive Code; manuelle Übertragungsfehler entfallen.

Ein vollständiges Beispiel-Dokument findest du unter [`code/bericht_beispiel.qmd`](code/bericht_beispiel.qmd).

Einrichten: `pip install jupyter` (Python-Kernel), `quarto install` von [quarto.org](https://quarto.org). Rendern: `quarto render code/bericht_beispiel.qmd`.

**Stolperstein:** Enthält der YAML-Header `echo: false`, bleibt der Code im gerenderten Bericht unsichtbar, Gutachter können die Analyse nicht nachvollziehen. Für Publikationen immer `echo: true` setzen. `code-fold: true` ist ein guter Kompromiss: Der Code ist vorhanden, aber standardmäßig eingeklappt.

**Darüber hinaus:** Mit `freeze: true` cached Quarto die Chunk-Ausgaben zwischen Renders. Das spart Zeit bei langen Berechnungen, ist aber gefährlich, wenn die Daten sich geändert haben. Im Zweifel: `freeze: false` und den Render sauber von vorne starten.

## 5 In Python

Das Skript [`code/python.py`](code/python.py) zeigt: Seed setzen, Paketversionen festhalten, Analyse mit fester Zufallskomponente durchführen und ein Manifest schreiben.

```python
SEED = 42
import pandas as pd
import numpy as np
np.random.seed(SEED)          # Pflicht vor jeder numpy-Zufallsoperation

cohort = pd.read_csv("{{DATEN}}kohorte.csv")
sample = cohort.sample(n=50, random_state=SEED)   # pandas unabhaengig von np
print("Mittleres Alter:", sample["alter"].mean().round(1))

import importlib.metadata
for pkg in ["numpy", "pandas", "scikit-learn"]:
    print(pkg, importlib.metadata.version(pkg))
```

`random_state` in pandas und scikit-learn ist **unabhängig** von `np.random.seed()`, beide müssen gesetzt werden, wenn beide genutzt werden.

**Stolperstein:** `np.random.seed()` beeinflusst nur den globalen NumPy-Zufallsgenerator. Bei parallelen Ausführungen oder verschachtelten Zufallsgeneratoren kann das zu unerwarteten Interaktionen führen. Robuster ist der neuere Ansatz: `rng = np.random.default_rng(SEED)` erzeugen und explizit weitergeben.

**Darüber hinaus:** `pip freeze > anforderungen.txt` oder `pip-compile` (pip-tools) erzeugt eine vollständige, nachvollziehbare Liste aller installierten Versionen. Zusammen mit einer virtuellen Umgebung ist das die einfachste Form von Environments-as-Code, der erste Schritt zu vollständiger Reproduzierbarkeit.

## 6 In R

Das Skript [`code/r.R`](code/r.R) zeigt dieselben Prinzipien in R: `set.seed()`, reproduzierbare Stichprobe, `sessionInfo()` und gezielte Versionsabfrage einzelner Pakete.

```r
SEED <- 42
# Root-Erkennung und Helper laden — siehe code/r.R fuer Details
library(tidyverse)

set.seed(SEED)                  # vor jeder Zufallsoperation

cohort <- read_csv("{{DATEN}}kohorte.csv", show_col_types = FALSE)
sample <- cohort |> slice_sample(n = 50)
cat("Mittleres Alter:", mean(sample$alter), "\n")

# Paketversionen
si <- sessionInfo()
cat("R-Version:", si$R.version$version.string, "\n")
cat("tidyverse:", as.character(packageVersion("tidyverse")), "\n")
```

Für R Markdown (`.Rmd`) gilt dasselbe Prinzip wie für Quarto: Text und Code in einer Datei, `rmarkdown::render()` erzeugt das Dokument.

**Stolperstein:** `set.seed()` setzt den Zufallsgenerator global. Ein späterer `set.seed()`-Aufruf innerhalb einer Funktion kann den äußeren State überschreiben, das passiert leicht, wenn Hilfsfunktionen aus Paketen intern Seeds setzen. Robuster: `withr::with_seed(SEED, expr)` für lokal kontrollierte Reproduzierbarkeit.

**Darüber hinaus:** `renv` ist das R-Äquivalent zu Pythons venv mit pip-tools: Es hält alle Paketversionen in einer `renv.lock`-Datei fest, die committet wird. Wer das Projekt klont, ruft `renv::restore()` auf und erhält exakt dieselbe Paketumgebung, ohne manuelle Installation.

## Fallstricke und Merksätze
- **Seed vergessen** ist der häufigste Fehler. Setze ihn ganz oben, bevor irgendein Zufallsgenerator aufgerufen wird.
- **`random_state` in scikit-learn** und **`np.random.seed()`** sind unabhängig, beide müssen gesetzt werden, wenn beide genutzt werden.
- **Relative Pfade** über `lib.helpers` statt absoluter Pfade sind Grundvoraussetzung, dass Analysen auf anderen Rechnern laufen.
- **Paketversionen ändern sich still.** Dokumentiere sie mit `pip freeze` oder `sessionInfo()`; noch besser: virtuelle Umgebungen mit gesperrten Versionen.
- **Manuelle Schritte sind reproduzierbarer Tod.** Jede Excel-Editierung, die nicht im Code steht, kann nicht nachvollzogen werden.
- **`echo: true` im Quarto-YAML**, so sieht der Leser den Code. Für Publikationsanhänge ist das Gold wert.

## Selbstcheck
1. **Warum muss der Seed ganz oben im Skript stehen, vor allen Zufallsoperationen?**, Jede Zufallsoperation zieht aus dem laufenden Generator; ein nachträglich gesetzter Seed beeinflusst nur spätere, nicht frühere Aufrufe. Das Ergebnis hängt dann von der Reihenfolge der Aufrufe ab und ist nicht mehr reproduzierbar.
2. **Was passiert, wenn Patientendaten versehentlich in Git committet wurden?**, Git behält die Datei dauerhaft in der gesamten Historie; vollständiges Entfernen erfordert `git filter-repo` und ein erzwungenes Re-Push auf alle Klone, aufwändig, fehleranfällig und nicht garantiert vollständig bei bereits weitergezogenen Klonen.
3. **Warum erzeugt ein Quarto-Dokument reproduzierbarere Berichte als Copy-Paste aus dem Skript?**, Code und Text sind in derselben Datei; `quarto render` führt alle Chunks aus und bettet Ergebnisse direkt ein, manuelle Übertragungsfehler zwischen Skript und Bericht entfallen vollständig.

