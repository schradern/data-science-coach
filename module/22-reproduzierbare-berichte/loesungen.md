# Musterlösungen · Reproduzierbare Berichte mit Quarto

> Vergleiche zuerst mit deinem eigenen Versuch. Reproduzierbarkeit hat oft mehrere
> richtige Wege, entscheidend ist die Konsequenz.

## Aufgabe 1 – Seed in Aktion

```python
import pandas as pd
cohort = pd.read_csv("{{DATEN}}kohorte.csv")

# With a seed: always the same sample
sample_a = cohort.sample(n=30, random_state=42)
sample_b = cohort.sample(n=30, random_state=42)
print(sample_a["patient_id"].tolist() == sample_b["patient_id"].tolist())  # True

# Without a seed: different on every run
sample_c = cohort.sample(n=30)
sample_d = cohort.sample(n=30)
print(sample_c["patient_id"].tolist() == sample_d["patient_id"].tolist())  # probably False
```

```r
library(tidyverse)
cohort <- read_csv("{{DATEN}}kohorte.csv", show_col_types = FALSE)

set.seed(42); a <- cohort |> slice_sample(n = 30)
set.seed(42); b <- cohort |> slice_sample(n = 30)
identical(a$patient_id, b$patient_id)  # TRUE
```

**Antwort a)** Ohne Seed ist jede Ausführung potenziell eine andere Stichprobe. Für
eine publizierte Analyse bedeutet das: Die Zahlen im Paper lassen sich nie mehr exakt
reproduzieren, ein ernsthaftes wissenschaftliches Problem, das Peer Review und
Replikation untergräbt.

**Antwort b)** Mit `random_state=42` ändert `np.random.seed(42)` nichts: `.sample()`
nutzt dann einen eigenen, mit 42 initialisierten Generator, der globale
NumPy-Zustand spielt keine Rolle. Ohne `random_state` zieht `.sample()` dagegen aus
dem globalen NumPy-Generator. Dann macht `np.random.seed(42)` die Stichprobe zwar
reproduzierbar, aber nur, solange davor im Skript keine anderen Zufallsaufrufe
stehen oder hinzukommen. Deshalb `random_state` bei jedem Aufruf explizit setzen.

## Aufgabe 2 – Paketversionen dokumentieren

```python
import importlib.metadata

for pkg in ["numpy", "pandas", "scikit-learn"]:
    try:
        ver = importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        ver = "nicht installiert"
    print(f"{pkg}: {ver}")
```

```r
for (pkg in c("dplyr", "ggplot2", "broom")) {
  cat(pkg, ":", as.character(packageVersion(pkg)), "\n")
}
```

**Warum reicht `import numpy` nicht?** Weil Paket-Updates das Verhalten von
Funktionen still ändern können. NumPy 1.x und 2.x unterscheiden sich in Randfällen
der Zufallsgenerierung. Scikit-learn ändert gelegentlich Default-Parameter zwischen
Minor-Versionen. Ohne Versionsangabe weiß niemand, welche Implementierung das
Ergebnis erzeugt hat, und eine Replikation auf neuerer Software kann abweichende
Ergebnisse liefern, ohne dass ein Fehler vorliegt.

## Aufgabe 3 – .gitignore verstehen und den Blanket-Fehler finden

**a)** Mit dieser `.gitignore` ignoriert Git:

- `data/roh/patienten.csv`, Regel `*.csv` **und** `data/roh/` (korrekt, das ist Rohdaten)
- `data/aufbereitet/ergebnisse.csv`, Regel `*.csv` (**das ist der Fehler**)
- `berichte/tabelle1.xlsx`, Regel `*.xlsx`

Git erfasst dagegen `code/analyse.py`. Analyse-Code gehört ins Repository,
aber nur, wenn er selbst keine Patientendaten enthält: keine eingefügten
Datenauszüge, keine Patienten-IDs oder Namen in Kommentaren, keine Pfade oder
Zugangsdaten zu Klinikservern. Prüfe das vor jedem Commit, ein Skript ist nicht
automatisch frei von schützenswerten Gesundheitsdaten.

**Das Problem:** Die pauschale Regel `*.csv` ignoriert **jede** CSV, also auch die
aufbereiteten Ergebnisse in `data/aufbereitet/`, die laut Projektstruktur bewusst
versioniert werden sollen. Genau davor warnt das README (§3): Ein späteres
`!data/aufbereitet/*.csv` würde das hier zwar reparieren, aber solche
Ausnahmeketten werden schnell unübersichtlich; ist ein übergeordneter Ordner selbst
ignoriert (etwa `data/`), greifen Negationen gar nicht mehr.

**b)** Korrigierte `.gitignore` – gezielt nur den Rohdaten-Ordner ausschließen,
nicht pauschal nach Dateityp:

```gitignore
# Nur Rohdaten mit potenziellen PHI ausschließen
data/roh/**
data/roh/*.csv
data/roh/*.xlsx

# Betriebssystem-Metadaten / Notebook-Ausgaben
.DS_Store
.ipynb_checkpoints/
```

So bleibt `data/roh/patienten.csv` ignoriert, während
`data/aufbereitet/ergebnisse.csv` normal versioniert wird. Auch
`berichte/tabelle1.xlsx` wird jetzt erfasst (aggregierte Ergebnisse ohne
Einzelfälle dürfen in der Regel ins Repo, prüfe deine Einrichtungsregeln).

**c)** Git verfolgt die gesamte History. Eine einmal committete Datei ist zwar aus
dem aktuellen Stand entfernbar, bleibt aber in `git log --all` sichtbar. Bei
PHI-Daten kann das rechtlich und ethisch problematisch sein, selbst nach dem
Löschen aus dem aktuellen Commit. Vollständiges Entfernen erfordert `git filter-repo`
und ein erzwungenes Re-Push auf alle Klone. Das ist aufwändig, fehleranfällig und
funktioniert nicht, wenn andere den alten Stand bereits geklont haben.

## Aufgabe 4 – Projektstruktur beurteilen

**Probleme:**

- Keine Trennung von Code, Daten und Ausgaben, alles im selben Ordner
- Versionierung durch Dateinamen (`_FINAL`, `_neu`, `_alt`) statt Git-Commits
- Rohdaten (`rohdaten_alle_patienten.csv`) und Code im selben Verzeichnis, gefährlich
- Excel als primäres Analyse-Werkzeug: Schritte sind nicht reproduzierbar
- Kein `.gitignore`: Patientendaten könnten versehentlich committet werden

**Bessere Struktur:**

```
analyse-sept-2024/
├── .gitignore          # gezielt data/roh/ (NICHT pauschal *.csv)
├── README.md           # Beschreibung, Ansprechperson, Laufanleitung
├── data/
│   ├── roh/            # nicht in Git (via .gitignore)
│   └── aufbereitet/    # erzeugt durch Skript (kann in Git)
├── code/
│   ├── 01_bereinigung.py
│   ├── 02_analyse.py
│   └── 03_bericht.qmd
└── ausgaben/
    ├── tabellen/
    └── figures/
```

## Aufgabe 5 – Quarto-Dokument lesen

**a)** Der Seed wird im Chunk `setup` gesetzt: `np.random.seed(SEED)`, unmittelbar
nach dem Import der Hilfsfunktionen. `SEED` ist als Konstante `42` in
`lib/helpers.py` definiert.

**b)** `echo: true` im YAML-Header (hier unter `format: html:`) sorgt dafür, dass der Code im gerenderten
Bericht sichtbar ist. `code-fold: true` ist ein guter Kompromiss: Code ist vorhanden
und anzeigbar, aber standardmäßig eingeklappt.

**c)** Für einen anderen Rechner braucht man:

- Python mit denselben Paketversionen (ideal: virtuelle Umgebung + `anforderungen.txt`)
- Quarto installiert (`quarto.org`) und `jupyter` in der Python-Umgebung (README Abschnitt 6)
- Die Daten: im Repository unter `data/` enthalten (neu erzeugen nur bei Bedarf mit `python data/generate_data.py`)
- Den Bericht aus der Projektwurzel rendern:
 `quarto render module/22-reproduzierbare-berichte/code/bericht_beispiel.qmd`

## Bonus – Erweiterte Tabelle in Quarto

Füge diesen Chunk in `bericht_beispiel.qmd` ein. Im `.qmd` öffnet der Code-Zaun mit
`{python}` in geschweiften Klammern (wie die übrigen Chunks dort), sonst führt Quarto
den Code nicht aus:

```python
#| label: tbl-stay-duration
#| tbl-cap: "Mediane Verweildauer je Aufnahmegrund"
# Excerpt for bericht_beispiel.qmd: uses the `df` loaded there.

stay_summary = (
    df
    .groupby("aufnahmegrund")["verweildauer_tage"]
    .agg(["median", "mean"])
    .round(1)
    .rename(columns={"median": "Median (Tage)", "mean": "Mittelwert (Tage)"})
)
print(stay_summary.to_string())
```
