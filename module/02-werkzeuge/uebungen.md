# Übungen · Entwicklungsumgebung und Werkzeuge

Code in Englisch, Prosa/Antworten Deutsch. Lösungen: [`loesungen.md`](loesungen.md).

## Aufgabe 1 – Umgebung prüfen
Führe das Skript `module/02-werkzeuge/code/python.py` aus der Projektwurzel aus.
Welche Pakete sind installiert? Fehlt eines? Installiere fehlende Pakete nach.

## Aufgabe 2 – Eigenes Prüfskript
Schreibe ein Skript `check_env.py`, das folgendes ausgibt:

- Python-Version
- pandas-Version
- Anzahl Zeilen und Spalten von `kohorte.csv`

Verwende die Standard-Hilfsfunktionen (`load_cohort`, `SEED`). Speichere die
Datei unter `module/02-werkzeuge/code/check_env.py`, damit die
`ROOT`-Ermittlung mit `.parents[3]` (wie in `code/python.py`) funktioniert.

## Aufgabe 3 – Virtuelle Umgebung
Erkläre in zwei Sätzen: Warum brauche ich für jedes Projekt eine eigene virtuelle
Umgebung? Was passiert, wenn ich keine verwende?

## Aufgabe 4 – Git initialisieren
Lege im Projektordner ein Git-Repository an (falls noch nicht geschehen) und
erstelle einen ersten Commit mit der Nachricht `"Initial commit: environment set up"`.
Was zeigt `git log --oneline` danach? Tipp: Erledige vorher den Bonus
(`.gitignore`), sonst landet `.venv/` im ersten Commit.

## Bonus
Erstelle eine `.gitignore`-Datei im Projektordner, die `.venv/`,
`__pycache__/` und `*.pyc` ausschließt. Prüfe mit `git status`, ob die
Einträge korrekt ignoriert werden.
