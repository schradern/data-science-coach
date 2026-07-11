# Mitwirken

Danke für dein Interesse. Korrekturen (fachlich, sprachlich, Code) sind sehr
willkommen — besonders von Ärztinnen und Ärzten und von Statistiker:innen.

## Fehler melden

Öffne ein [Issue](https://github.com/schradern/data-science-coach/issues) mit:
Modul und Abschnitt, was falsch ist, und (bei fachlichen Punkten) eine Quelle.

## Änderungen einreichen

1. Repository forken, Branch anlegen.
2. Änderung machen. Prosa ist **deutsch**, Code-Bezeichner und Kommentare **englisch**.
3. Vor dem Commit die Prüfungen laufen lassen:

   ```bash
   ./.venv/bin/python tools/check_repo.py     # schnell (~10 s): Struktur, Links, Zahlen-Kontrakt
   ./.venv/bin/python website/build.py        # baut die Website; bricht bei toten Links ab
   ```

   Wer eine Zahl in einer Lektion ändert, prüft zusätzlich, dass der Code sie
   noch erzeugt:

   ```bash
   ./.venv/bin/python tools/check_numbers.py 13   # nur Modul 14 (schnell)
   ```

4. Pull Request öffnen. CI (`pages.yml`, `numbers.yml`) muss grün sein.

## Grundregeln des Kurses

Diese sind maschinell geprüft — bitte einhalten:

- **Jede Zahl in einer Lektion wird vom Code des Moduls erzeugt.** Keine
  handgeschriebenen Ergebnisse (Ausnahmen mit Begründung in `tools/numbers_allow.json`).
- **Kopierbare Codeblöcke laufen ohne Repo-Klon**: Daten über `{{DATEN}}…`, keine
  `Path(__file__)`-, `sys.path`- oder `commandArgs()`-Zeilen.
- **Seed ist überall `42`.** Datenerzeugung ist byte-genau reproduzierbar.
- **Kapitelnummern folgen der Lesereihenfolge** (`TEILE` in `website/build.py`).
- **Keine echten Patientendaten.** Der Kurs nutzt ausschließlich synthetische Daten.

Details in [`docs/KONVENTIONEN.md`](docs/KONVENTIONEN.md).

## Lizenz

Beiträge zu Inhalten stehen unter CC BY-NC-SA 4.0, Beiträge zu Code unter der
MIT-Lizenz (siehe [`LICENSE`](LICENSE) und [`LICENSE-CODE`](LICENSE-CODE)).
