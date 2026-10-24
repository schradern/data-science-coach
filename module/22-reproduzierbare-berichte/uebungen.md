# Übungen · Reproduzierbare Berichte mit Quarto

Code in Englisch, Prosa/Antworten Deutsch. Lösungen: [`loesungen.md`](loesungen.md).

## Aufgabe 1 – Seed in Aktion

Führe folgenden Code zweimal aus, einmal mit `random_state=42`, einmal ohne:

```python
import pandas as pd
cohort = pd.read_csv("{{DATEN}}kohorte.csv")
sample = cohort.sample(n=30, random_state=42)   # or: cohort.sample(n=30)
print(sample["patient_id"].tolist())
```

a) Was beobachtest du? Warum ist das für publizierte Analysen relevant?
b) Setze zusätzlich `numpy.random.seed(42)` ganz am Anfang. Ändert das etwas an
 der Stichprobe, einmal mit und einmal ohne `random_state=42`? Begründe.

## Aufgabe 2 – Paketversionen dokumentieren

Schreibe ein kleines Python-Skript (oder führe es in der Konsole aus), das die
installierten Versionen von `numpy`, `pandas` und `scikit-learn` ausgibt.
Nutze dafür `importlib.metadata.version("paketname")`.

Frage: Warum reicht `import numpy` als Dokumentation nicht aus? Nenne ein
konkretes Beispiel, wie eine Versionsänderung ein Analyseergebnis beeinflussen kann.

## Aufgabe 3 – .gitignore verstehen und den Blanket-Fehler finden

Eine Kollegin legt diese `.gitignore` an, um Patientendaten zu schützen:

```
*.csv
*.xlsx
data/roh/
```

Das Projekt folgt der Struktur aus dem README (§2) und enthält u. a.:

 - `data/roh/patienten.csv` (Rohdaten, dürfen **nie** ins Repo)
 - `data/aufbereitet/ergebnisse.csv` (durch Skript erzeugt, **sollen** committet werden)
 - `code/analyse.py`
 - `berichte/tabelle1.xlsx`

a) Welche dieser Dateien ignoriert Git? Welches **Problem** entsteht dadurch
 speziell für `data/aufbereitet/ergebnisse.csv`?

b) Korrigiere die `.gitignore` so, dass nur die Rohdaten ignoriert werden, die
 aufbereiteten Ergebnisse aber versioniert werden. (Tipp: gezielt `data/roh/`
 ausschließen statt pauschal nach Dateityp, siehe README §3.)

c) Warum ist es wichtig, `.gitignore` **vor dem ersten Commit** anzulegen, und
 nicht erst, wenn man merkt, dass eine Datei versehentlich getrackt wird?

## Aufgabe 4 – Projektstruktur beurteilen

Gegeben ist folgende Ordnerstruktur:

```
analyse_sept_2024/
├── finale_version_3_WIRKLICHFINAL.xlsx
├── skript_neu.py
├── skript_alt.py
├── skript_alt2.py
└── rohdaten_alle_patienten.csv
```

a) Benenne mindestens vier Probleme dieser Struktur.
b) Entwerfe eine bessere Struktur mit sinnvollen Ordnern, nummerierten Skripten
 und einer `.gitignore`-Datei.

## Aufgabe 5 – Quarto-Dokument lesen

Öffne [`code/bericht_beispiel.qmd`](code/bericht_beispiel.qmd) und beantworte:

a) Wo und wie wird der Seed gesetzt?
b) Welche YAML-Einstellung stellt sicher, dass der Code im gerenderten Bericht
 sichtbar ist?
c) Was braucht jemand auf einem anderen Rechner, um den Bericht identisch zu
 rendern?

## Bonus

Erweitere `bericht_beispiel.qmd` um einen neuen Chunk, der die **mediane
Verweildauer je Aufnahmegrund** als Tabelle ausgibt. Render das Dokument,
falls Quarto installiert ist:

```
quarto render module/22-reproduzierbare-berichte/code/bericht_beispiel.qmd
```
