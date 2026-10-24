# Konventionen – Aufbau und Mitarbeit

Damit der Kurs konsistent, trocken (DRY) und wartbar bleibt.

## Verzeichnisstruktur

```
Data Science Coach/
├── README.md                  # Einstieg & Kursindex (Lernende starten hier)
├── config.json                # zentrale Quelle aller veröffentlichten URLs
├── requirements.txt           # Python-Kernpakete (Teil 1–4)
├── requirements-ml.txt        # Python-Zusatzpakete (Teil 5: ML/KI)
├── DEPENDENCIES-R.md          # zentrale R-Paketliste
├── LICENSE                    # Lizenzübersicht + CC BY-NC-SA (Kursinhalte)
├── LICENSE-CODE               # MIT-Lizenz (Quellcode)
├── docs/
│   ├── LEHRPLAN.md            # Curriculum-Architektur (Planungsdokument)
│   ├── KONVENTIONEN.md        # dieses Dokument
│   ├── GLOSSAR.md             # Kurzdefinitionen der Fachbegriffe
│   ├── PROJEKTKOMPASS.md      # Frage → Datenquelle → Design → Auswertung
│   ├── PUBLIKATIONS_CHECKLISTE.md  # Checkliste für publikationsreife Analysen
│   ├── SPICKZETTEL.md         # kompakte Code-/Methoden-Referenz
│   └── RECHTLICHES.md         # Impressum, Datenschutz, Lizenzen
├── data/
│   ├── generate_data.py       # erzeugt alle synthetischen Datensätze (Seed-fest)
│   ├── figures.py             # erzeugt geteilte Abbildungen (Modul-übergreifend)
│   ├── README.md              # Daten-Spezifikation + reale Datenquellen
│   └── *.csv, *.xlsx, *.json  # erzeugte Datensätze (durch Skript reproduzierbar)
├── lib/
│   ├── helpers.py             # gemeinsame Python-Hilfsfunktionen (DRY)
│   ├── helpers.R              # gemeinsame R-Hilfsfunktionen (DRY)
│   ├── diagrams.py            # gemeinsame Diagramm-Bausteine
│   ├── ground_truth.py        # Ground-Truth-Effekte des Datensatzes
│   └── plotstyle.py           # einheitlicher Plot-Stil
├── notebooks/
│   └── start.ipynb            # Colab-Start ohne lokale Installation
├── website/
│   ├── build.py               # baut die statische Seite (Artefakt: site/)
│   ├── figure_source.py       # ordnet jeder Abbildung den Code zu, der sie zeichnet
│   ├── app.js                 # Interaktionsschicht (Vanilla-JS)
│   ├── style.css              # Styling
│   └── README.md              # Website-Doku
├── tools/
│   ├── install_r_packages.R   # installiert die R-Pakete aus DEPENDENCIES-R.md
│   ├── check_repo.py          # Konsistenz-Check, CI-Gate (.github/workflows/pages.yml)
│   ├── check_numbers.py       # prüft lokal: jede Ergebniszahl der Prosa kommt aus dem Code
│   └── numbers_allow.json     # begründete Ausnahmen für check_numbers.py
└── module/
    ├── README.md              # Modulübersicht (Index aller 34 Module)
    └── NN-kurzname/
        ├── README.md          # die Lektion (deutsch)
        ├── uebungen.md        # Übungsaufgaben
        ├── loesungen.md       # Musterlösungen
        └── code/
            ├── python.py      # Einstiegsskript der Lektion (lauffähig)
            ├── r.R            # dasselbe in R
            ├── sql.sql        # nur wo relevant (v. a. Modul 05)
            └── figures.py     # regeneriert die Abbildungen des Moduls, wo vorhanden
```

`python.py`/`r.R` sind die **Einstiegsskripte** der Lektion; `figures.py`
regeneriert deren Abbildungen (siehe `data/figures.py`, das die Module 01, 06, 08–10, 12, 13, 15–17 und 21 in
einem Lauf abdeckt). Weitere, thematisch benannte Skripte sind für
Vertiefungsmaterial erlaubt, müssen aber in der Modul-`README.md` verlinkt
sein – z. B. Modul 14: `rubin.py`/`.R`, `benchmark.py`, `mice_praxis.py`/`.R`,
`mnar_delta.py`/`.R`.

## Namens- und Schreibregeln

- Modulordner: `NN-kurzname`, zweistellig (`01`…`34`), Kleinbuchstaben, Bindestriche.
- **`NN` ist die Position in der Lesereihenfolge**, nicht das Entstehungsdatum.
  Verbindlich ist `TEILE` in `website/build.py`: das erste Modul dort heißt `01`,
  das siebzehnte `17`. Auch die Überschrift der Lektion (`# NN · Titel`) trägt
  genau diese Nummer. `tools/check_repo.py` erzwingt beides.

  Das war nicht immer so: die zuletzt geschriebenen Module bekamen die höchsten
  Nummern und wurden dann in frühere Teile einsortiert – das Menü las sich
  `04, 05, 06, 32, 07`, und jedes „siehe Modul 13“ zeigte auf das Modul, dem
  diese Nummer gerade gehörte. Wer ein Modul einfügt oder verschiebt, nummeriert
  daher **alle** folgenden Ordner um und zieht die Verweise mit.
- Lektionssprache (Prosa, Übungen, Bildunterschriften): **Deutsch**. Fachbegriffe
  englisch nur, wenn etabliert (z. B. *DataFrame*, *Tidy Data*), beim ersten
  Auftreten kurz erklärt.

- **Code ist Englisch.** Bezeichner (Variablen, Funktionen, Klassen), Kommentare
  und Docstrings durchgängig in klarem, idiomatischem Englisch (`snake_case` in
  Python/R). So bleibt der Code lesbar und entspricht den üblichen Standards.

- **Ausnahme – Datenschema:** Die Spaltennamen und Kategorien des (deutschen,
  klinischen) Datensatzes bleiben deutsch (`alter`, `aufnahmegrund`,
  `verstorben_30d`, …). Sie sind Daten, kein Code – der Zugriff erfolgt per
  String-Schlüssel (`df["aufnahmegrund"]`).

- Einheitlicher Seed: **`42`** in allen Beispielen.
- Datenzugriff **relativ** über die Hilfsfunktionen (`load_cohort()`,
  `load_labs()`, `load_vitals()`, `load_notes()`), nie absolute Pfade.
- **Datenladen in zwei Varianten**, je nach Kontext:
  - Lauffähige `code/python.py`/`code/r.R`-Skripte nutzen die
    `lib/`-Hilfsfunktionen (lokal zuerst, Web-Adresse als Rückfallebene, kein
    lokaler `data/`-Ordner nötig).
  - Einzelne, kopierbare Codeblöcke in `README.md`/`uebungen.md`/`loesungen.md`
    (die auch ohne Repo-Klon funktionieren müssen) nutzen den Platzhalter
    `{{DATEN}}` (z. B. `pd.read_csv("{{DATEN}}kohorte.csv")`); `website/build.py`
    ersetzt ihn beim Bauen der Seite durch die tatsächliche Basis-Adresse.
    Niemals einen lokalen Pfad wie `data/kohorte.csv` in einen Copy-paste-Block
    schreiben.

## Dateiverweise und Abbildungen

**Jeder Dateiname im Fließtext wird zu einem Link ins Repository.** Schreibe den
Pfad einfach als Inline-Code – `lib/helpers.py`, `kohorte.csv`, `code/rubin.py` –,
`website/build.py` macht daraus beim Bauen einen Link auf genau diese Datei.
Drei Regeln, damit das nie ins Leere zeigt:

- Verlinkt wird nur, was es wirklich gibt. Ein erfundener Pfad
  (`data/roh/patienten.csv` in einer Übung) bleibt einfach Code.
- Ein Pfad relativ zum Modul (`code/python.py`) zeigt auf die Datei **dieses**
  Moduls. Ein bloßer Dateiname wird nur verlinkt, wenn er im ganzen Repository
  eindeutig ist – `figures.py` gibt es siebzehnmal und bleibt deshalb ungelinkt.
- Codeblöcke werden nie angefasst.

**Jede Abbildung verlinkt auf den Code, der genau sie zeichnet** – nicht auf die
ganze erzeugende Datei. `website/figure_source.py` liest die Generatoren und
findet den passenden Ausschnitt: die Funktion, den Abschnitt oder den
Zeichenteil von `main()`. Von Hand gezeichnete Diagramme stehen dort in der
Liste `HANDMADE` und bekommen bewusst keinen Link. `tools/check_repo.py` schlägt
fehl, sobald eine gezeigte Abbildung weder einen Generator noch einen Eintrag in
dieser Liste hat.

## Lernwert-Regel

Jedes sichtbare Element muss eine Lernhandlung unterstützen: Orientierung,
Entscheidung, Ausführung, Erinnerung oder Qualitätskontrolle. Wenn ein Absatz,
eine Farbe, ein Kasten, ein Diagramm oder ein UI-Element nur schmückt, wird es
entfernt oder in eine Lernfunktion überführt.

Für neue oder überarbeitete Module gilt deshalb zusätzlich:

- Der klinische Nutzen muss vor dem ersten technischen Detail klar sein.
- Lernziele bleiben kurz: maximal fünf, als überprüfbare Handlungen formuliert.
- Jede Übung erzeugt ein sichtbares Ergebnis: Zahl, Tabelle, Plot, Modell,
  Reportingsatz oder begründete Entscheidung.
- Code-Module haben lauffähige Python- und R-Skripte oder erklären ausdrücklich,
  warum das Modul eine methodische Entscheidung statt ein Skript trainiert.
- Jede Vertiefung beantwortet die Frage: *Welche Fehlentscheidung verhindert
  dieses Wissen in einer echten klinischen Analyse?*

## Lektions-Vorlage (`module/NN-kurzname/README.md`)

```markdown
# NN · Titel

> **Teil N – Teiltitel** · Dauer: ~XX min · Voraussetzung: Modul …

## Lernziele
- …

## Klinischer Aufhänger
Eine konkrete ärztliche Frage.

## 1 Konzept
Kurze Erklärung vor dem Code.

## 2 In Python
```python
# lauffähig
```

## 3 In R
```r
# dasselbe Ergebnis
```

## Fallstricke und Merksätze
- …

## Weiter
Übungen: `uebungen.md` · Lösungen: `loesungen.md` · Nächstes Modul: …
```

## Didaktik und Interaktion

- Jede Lektion bleibt **kurz pro Absatz**: ein Gedanke, dann Beispiel oder
  Konsequenz. Keine Referenzhandbuch-Prosa.

- Der erste konkrete Nutzen muss früh sichtbar sein: klinische Frage,
  Werkzeuge, erwartetes Ergebnis. Die Website erzeugt daraus den Lernkompass.

- Markiere bewusst:
  - `**Grundlage:**` für Einsteiger-Erklärungen, die Profis überspringen können.
  - `**Darüber hinaus:**` für Vertiefungen, die im Grundlagenmodus einklappen.
  - `**Stolperstein:**` für häufige Fehler.
  - `> **Merksatz:**` für eine Kernaussage.
- Selbstchecks sollten 3–5 Fragen haben und im Muster
  `**Frage?**, Antwort` geschrieben sein; daraus werden aufdeckbare Karten.

- Übungen starten leicht, enden mit Transfer. Jede Aufgabe soll ein sichtbares
  Ergebnis erzeugen: Zahl, Tabelle, Plot, Modellbewertung oder Bericht.

- Designhaltung: **focused reduction**. Keine dekorativen Elemente ohne Lernwert,
  keine langen Einleitungstexte, keine UI-Fläche, die nicht Orientierung,
  Entscheidung oder Handlung unterstützt.

## Code-Standards

- Jedes `code/python.py` und `code/r.R` läuft **eigenständig** durch
  (`python python.py` bzw. `Rscript r.R`) und nutzt `lib/`.

- Keine Wiederholung von Daten-Lade-Logik – immer `lib/`-Funktionen.
- Reproduzierbar: Seed gesetzt, keine versteckten Zustände, kein Internet nötig
  (Ausnahme: ausgewiesene API-Beispiele, mit Offline-Fallback).

- Pakete pro Sprache zentral dokumentiert (Python: `requirements.txt` für den
  Kernpfad, `requirements-ml.txt` für die ML-/KI-Zusatzpakete; R:
  `DEPENDENCIES-R.md`).

## Reproduzierbarkeit prüfen

```bash
python data/generate_data.py     # Datensätze neu erzeugen
python module/06-transformation/code/python.py   # Beispiel: ein Modul laufen lassen
```

Für die R-Spur braucht dieser erste Schritt einmalig Python – Details und
Alternative unter [„Python für die R-Spur – einmalig“](../DEPENDENCIES-R.md).

## CI und automatisierte Prüfungen

Zwei Werkzeuge prüfen den Kurs automatisiert; beide laufen lokal und in CI.

- **`tools/check_repo.py`** – schnelle Struktur-Checks (Links, Platzhalter,
  Seeds, Konventionen, Assets, `code/`-Ordner ohne Abbildungen). Läuft in ~8 s
  und bei **jedem Push** (`.github/workflows/pages.yml`, zusammen mit dem
  Website-Build).

- **`tools/check_numbers.py`** – führt die Python- **und** R-Skripte jedes
  Moduls aus und prüft, ob jede Zahl, die eine Lektion in einer
  **Markdown-Tabelle** oder einem ```` ```text ````-Block behauptet,
  tatsächlich vom Code produziert wird. Das dauert ~8 Minuten (Modul 14s
  `benchmark.py` dominiert die Laufzeit). Ein einzelnes Modul lässt sich
  gezielt prüfen:

  ```bash
  ./.venv/bin/python tools/check_numbers.py 12
  ```

  `DSC_SKIP_SLOW=1` überspringt den Benchmark.

  **Blinde Flecken – bewusst, damit niemand einem grünen Lauf zu viel
  vertraut:** Das Skript liest keine Zahlen in Prosa, in ```` ```python ````-Blöcken
  oder in Bildunterschriften; und es vergleicht **Beträge**, ein falsches
  Vorzeichen fällt also nicht auf.

- **`tools/numbers_allow.json`** – das Ausnahmeventil für Zahlen, die
  legitim nicht aus dem Skript des Moduls selbst stammen (z. B. aus
  `lib/ground_truth.py`, einem Geschwister-Skript oder der Literatur). Jeder
  Eintrag braucht ein `reason`-Feld; ohne das bricht das Skript mit Exitcode 2
  ab. Diese Datei bleibt bewusst klein.

- **Exitcodes von `check_numbers.py`:**
  - `0` – jede behauptete Zahl wird vom Code des Moduls selbst reproduziert.
  - `1` – Drift: eine Zahl in einer Tabelle oder einem ```` ```text ````-Block
    hat keine Entsprechung in der Skript-Ausgabe. Entweder die Prosa
    korrigieren oder einen begründeten Eintrag in `tools/numbers_allow.json`
    ergänzen.
  - `2` – ein Eintrag in `tools/numbers_allow.json` hat kein `reason`-Feld.
  - `3` – das Python- oder R-Skript eines Moduls konnte nicht laufen (fehlendes
    Python- oder R-Paket, fehlendes `Rscript`), seine Zahlen also gar nicht
    geprüft werden. **Das ist ein Umgebungsproblem, kein Prosa-Problem.** Das
    Werkzeug listet die übrig gebliebenen Zahlen unter `unaccounted:` – tauchen
    sie auch nach einem funktionierenden Lauf nicht auf, ist es echte Drift.

  Das Werkzeug nutzt `./.venv/bin/python`, wenn im Projektstamm ein `.venv`
  existiert, sonst den Interpreter, unter dem es selbst läuft (damit es auch in
  CI funktioniert). Module, deren Skript fehlschlug, deren Zahlen aber
  vollständig von den *anderen* Skripten produziert wurden, werden nur als
  Hinweis gemeldet und lassen den Lauf nicht scheitern.

- **CI:** `pages.yml` baut die Seite und führt `check_repo.py` bei jedem Push
  aus. `check_numbers.py` läuft **nur lokal** (siehe oben) und ist bewusst
  **kein** CI-Gate: Iterative numerische Verfahren (etwa die Imputation in
  Modul 14 oder die IPW-Gewichtung in Modul 19) liefern mit den aktuellen
  Bibliotheken in den letzten Stellen plattformabhängige Ergebnisse
  (macOS/Accelerate vs. Linux/OpenBLAS); ein strikter Zahlenabgleich über
  Läufer hinweg meldete das fälschlich als Drift. Vor größeren Zahlenänderungen
  lohnt der lokale Lauf trotzdem – er fängt grobe Abweichungen zuverlässig.

## Vertiefungs-Marker (`<!-- vertiefung: … -->`)

Für Module mit einer langen, klar abgrenzbaren Vertiefungsspur über mehrere
`##`-Abschnitte (z. B. Modul 14, §7–§11) markiert ein Kommentarpaar den
Abschnitt:

```markdown
<!-- vertiefung: Titel der Spur -->
… mehrere ##-Abschnitte …
<!-- /vertiefung -->
```

`website/build.py` fasst alles zwischen den Markern in ein einziges
aufklappbares `<details class="vertiefung">`, das im Grundlagen-Modus
zugeklappt ist und sich per Grundlagen/Alles-Umschalter öffnet.

Regeln: genau ein Start-Marker pro Datei, immer geschlossen (`<!-- /vertiefung
-->` nicht vergessen), nie innerhalb eines Codeblocks.
