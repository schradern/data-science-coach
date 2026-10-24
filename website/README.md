# Website

Statische Kurs-Website, **reproduzierbar** aus den Markdown-Quellen gebaut.
Einzige Quelle der Wahrheit bleiben die `.md`-Dateien – nichts wird doppelt gepflegt.

## Bauen

```bash
pip install markdown pygments
python website/build.py
```

Öffnen: `website/site/index.html` im Browser.

## Was passiert
- `README.md`, `docs/*.md`, `data/README.md` und alle `module/*/` (Lektion,
  Übungen, Lösungen) werden zu HTML gerendert.
- Jede Code-Datei (`python.py`, `r.R`, `sql.sql`, `*.qmd`) wird als
  syntaxhervorgehobene Seite eingebunden; Grafiken (PNG) werden übernommen.
- Erzeugt werden Seitenleisten-Navigation (nach Teilen gruppiert),
  Lektions-Untermenü und Weiter/Zurück.

## Gestaltung
`style.css` – fokussierter Minimalismus, klinisch, seriös: ruhige Palette,
ein klinisches Blau als sparsamer Akzent, großzügiger Weißraum, lesefreundliche
Spaltenbreite, responsiv (einklappbare Navigation), `prefers-reduced-motion`.
Design ändern = nur `style.css` anpassen und neu bauen.

## Interaktion & Didaktik (`app.js`)
Alles Folgende entsteht **aus den vorhandenen Markdown-Markern** – die Prosa
bleibt einzige Quelle der Wahrheit, nichts wird doppelt gepflegt:

- **Drei-Spuren-Callouts.** `**Stolperstein:**` → Fallstrick-Box, `**Praxis:**`
  → Praxis-Box, `> **Merksatz:**` → Kernaussage. Dazu zwei **einklappbare,
  spiegelbildliche** Vertiefungsspuren:
  `**Grundlage:**` → *Für Einsteiger* (offen im Grundlagen-Modus, zu im
  Alles-Modus) und `**Darüber hinaus:**` → *Für Fortgeschrittene* (umgekehrt).
- **Niveau-Schalter** (Grundlagen / Alles): im Grundlagen-Modus sind die
  Einsteiger-Boxen offen und die Fortgeschrittenen zugeklappt, im Alles-Modus
  umgekehrt – dieselbe Seite dient Einsteiger:innen und Profis. Einzelne Boxen
  lassen sich jederzeit manuell auf-/zuklappen.
- **Sprachfilter** (Python / R / Beide): blendet Codeblöcke der jeweils anderen
  Sprache aus. Seitenübergreifend gemerkt.
- **Selbstcheck → Erinnerungskarten:** Fragen aus `## Selbstcheck` werden zu
  aufdeckbaren Karten (aktives Erinnern statt Mitlesen).
- **Fortschritt:** „Modul als erledigt“ wird pro Modul gespeichert; die
  Übersicht zeigt Häkchen, einen Ring und eine Zähler-Pille (n / Gesamtzahl).
- **Lernpfad-Auswahl** in der Modulübersicht hebt die Module eines Pfades hervor
  und nummeriert die Schritte.
- **Tastatur:** ← / → springt zum vorigen/nächsten Modul.

- **Ruhige Oberfläche.** Navigation und Lern-Schalter liegen in einer Zeile;
  aktive Zustände sind dezent getönt statt vollflächig; das Inhaltsverzeichnis
  ist einklappbar; lange Codeblöcke werden gekürzt gezeigt („Ganzen Code zeigen“).

Alles rein clientseitig (Vanilla-JS, keine Abhängigkeiten), `localStorage` für
Einstellungen, `prefers-reduced-motion` respektiert. Verhalten ändern = nur
`app.js` anpassen und neu bauen.

## Diagramme (SVG)
Konzeptdiagramme entstehen **reproduzierbar aus einem Generator** mit einem
gemeinsamen, minimalistischen Stil – Farben und Typografie liegen zentral in
`THEME`, Stil global ändern = dort anpassen und neu erzeugen:

```bash
python lib/diagrams.py     # schreibt module/*/assets/*.svg
```

Aktuell: `03` DataFrame-Anatomie, `04` Einlese-Pipeline, `05` Datenmodell +
JOIN-Typen, `06` Long ↔ Wide, `11` Eine Quelle → viele Formate, `13` Split &
Kreuzvalidierung. Einbindung wie Plots per Markdown-Bild (der `alt`-Text wird
zur Bildunterschrift); der Build kopiert `*.png` **und** `*.svg`. Weitere Module
ergänzt man durch eine Diagrammfunktion + Eintrag in `TARGETS`.

## Ausgabe
`website/site/` ist vollständig statisch (kein Server nötig), nutzt nur relative
Pfade (läuft also auch unter einem Unterpfad) und wird unter
[datascience.medizin.dev](https://datascience.medizin.dev/) veröffentlicht,
zusätzlich für ältere Links per GitHub Pages.

## Automatisches Deployment
`.github/workflows/pages.yml` baut die Seite bei jedem Push auf `main` und
veröffentlicht sie zweimal: per SSH auf den Server von datascience.medizin.dev
und auf GitHub Pages. Der Build braucht nur `markdown` + `pygments`; Abbildungen
(`module/*/assets/*.png`) und Datensätze (`data/*.csv`) sind eingecheckt und
werden unverändert übernommen.

- **Server:** Der Job `deploy-server` schickt die Seite als tar.gz an den
  Benutzer `dsc-deploy`, mit dem Commit-SHA als einzigem Argument. Der Server
  entpackt sie in einen neuen Ordner unter `/srv/datascience-coach/releases/`,
  schaltet `current` um und behält die letzten fünf Releases. Die Zugangsdaten
  liegen als Secrets `DSC_DEPLOY_HOST`, `DSC_DEPLOY_KEY` und
  `DSC_DEPLOY_KNOWN_HOSTS` in der Umgebung `datascience-medizin-dev`, die nur
  `main` nutzen darf.
- **GitHub Pages:** Einmalig im Repo aktivieren:
  **Settings → Pages → Source: GitHub Actions**.

Abbildungen/Daten lokal neu erzeugen (optional, vor dem Commit):

```bash
pip install -r requirements.txt
python data/generate_data.py && python data/figures.py
# Modul-Abbildungen aus Teil 4:
for m in module/1[3-9]* module/2*; do python "$m"/code/figures.py 2>/dev/null; done
```
