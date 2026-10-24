# Musterlösungen · Entwicklungsumgebung und Werkzeuge

> Vergleiche zuerst mit deinem eigenen Versuch. Es gibt oft mehrere richtige Wege.

## Aufgabe 1 – Umgebung prüfen
```bash
# From the project root:
python module/02-werkzeuge/code/python.py
```
Fehlende Pakete nachinstallieren, mit den gepinnten Versionen aus `requirements.txt`
(ein bloßes `uv pip install pandas` zöge die neueste, für den Kurs ungetestete Version):
```bash
uv pip install -r requirements.txt
# or without uv, after activating the virtual environment:
source .venv/bin/activate        # macOS/Linux; Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Stolperstein:** Installiere nie mit `pip install --break-system-packages`
außerhalb einer virtuellen Umgebung, das überschreibt die vom Betriebssystem
verwaltete Python-Installation und kann sie beschädigen, genau die
„Dependency Hell“, vor der Abschnitt 3 der Lektion warnt. Aktiviere stattdessen immer
zuerst `.venv/`.

## Aufgabe 2 – Eigenes Prüfskript
Speichere das folgende Skript als `module/02-werkzeuge/code/check_env.py`
(also direkt neben `python.py`). Von der Datei aus gezählt ist dann
`parents[0]` der Ordner `code/`, `parents[1]` der Modulordner, `parents[2]`
`module/` und `parents[3]` die Projektwurzel; `ROOT` ist damit
`Path(__file__).resolve().parents[3]` (wie bei allen anderen `code/*.py`
im Kurs). Legst du die Datei woanders ab, muss die Anzahl der `.parents[...]`
entsprechend angepasst werden.

```python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import pandas as pd
from lib.helpers import SEED, load_cohort

print(f"Python:  {sys.version.split()[0]}")
print(f"pandas:  {pd.__version__}")
print(f"Seed:    {SEED}")

cohort = load_cohort()
print(f"Dataset: {cohort.shape[0]} rows x {cohort.shape[1]} cols")
```

## Aufgabe 3 – Virtuelle Umgebung
**Warum:** Jedes Projekt hat eigene Paket-Abhängigkeiten. Ohne Isolation kann
eine Paket-Aktualisierung in Projekt B das Skript in Projekt A kaputt machen.

**Was passiert ohne:** Alle Pakete landen in der globalen Python-Installation.
Versionskonflikte entstehen, Reproduzierbarkeit geht verloren.

## Aufgabe 4 – Git initialisieren
```bash
# create the .gitignore from the bonus first, so .venv/ stays out of the commit
git init
git add .
git commit -m "Initial commit: environment set up"
git log --oneline
# Output e.g.: a1b2c3d Initial commit: environment set up
```

## Bonus – .gitignore
```
# .gitignore
.venv/
__pycache__/
*.pyc
.DS_Store
```
```bash
git status   # .venv/ und __pycache__/ sollten nicht mehr erscheinen
```
