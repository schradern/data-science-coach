# Data Science für Ärztinnen und Ärzte

Von der Rohdatei zur belastbaren Auswertung – mit Python, R und SQL, an echten
klinischen Fragestellungen, reproduzierbar angelegt.

<!-- nur-github -->
**➜ Kurs lesen: [schradern.github.io/data-science-coach](https://schradern.github.io/data-science-coach/)**
<!-- /nur-github -->

## Drei Wege, den Kurs zu nutzen

| Weg | Wofür | Was du brauchst |
|-----|-------|-----------------|
| **Im Browser lesen & rechnen** | der Standard | nichts zum Lesen; Python **oder** R zum Mitrechnen. Jeder Codeblock lädt die Daten über eine Web-Adresse. |
| **In Google Colab** | rechnen ohne lokale Installation | ein Google-Konto. Das [Start-Notebook](https://colab.research.google.com/github/schradern/data-science-coach/blob/main/notebooks/start.ipynb) klont den Kurs und installiert alles. |
| **Lokal (geklont)** | volle Reproduzierbarkeit, offline | `git clone` + Python/R — die Einrichtung erklärt [Modul 02](module/02-werkzeuge/README.md) Schritt für Schritt. |

<!-- nur-github -->
Lokal einrichten:

```bash
git clone https://github.com/schradern/data-science-coach
cd data-science-coach
uv venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt           # + requirements-ml.txt für Teil 5
```

Für R: siehe [`DEPENDENCIES-R.md`](DEPENDENCIES-R.md). Details und Fehlerbehebung in [Modul 02](module/02-werkzeuge/README.md).
<!-- /nur-github -->

## Direkt einsteigen

Zum **Lesen** musst du nichts installieren. Zum **Rechnen** brauchst du Python
oder R — Modul 02 richtet beides ein. Wähle einen kleinen Einstieg:

- **Orientieren:** [Modul 01](module/01-einfuehrung/README.md) erklärt den
  Analyseweg. Kein Setup, ~30 Minuten Lesezeit.
- **Projekt klären:** Der [Projektkompass](docs/PROJEKTKOMPASS.md) sortiert
  Frage, Datenquelle, Design und Auswertung.
- **Code ausführen:** [Modul 02](module/02-werkzeuge/README.md) richtet Python
  **oder** R und den Editor ein, sobald du Übungen rechnen möchtest. Rechne mit
  45–60 Minuten für die Einrichtung. Die Datensätze liest jeder Codeblock über
  eine Web-Adresse — du musst nichts herunterladen oder erzeugen, und reine
  R-Nutzung braucht kein Python.

Der vollständige Kurs umfasst 34 Module und rund **43 Stunden**. Die
[Modulübersicht](module/README.md) zeigt Dauer und Werkzeugfokus je Modul; über
die Lernpfade lässt sich der Kurs auf 5–14 Module verkürzen.

Schon eingerichtet? Springe über die [Modulübersicht](module/README.md) direkt zu dem Thema, das gerade klemmt.

## Was du danach kannst

- eine Kohorte aus **CSV, Excel und SQL** extrahieren und zusammenführen
- Daten **bereinigen, transformieren** und fehlende Werte begründet behandeln
- **Verteilungen, Zusammenhänge und Gruppenunterschiede** prüfen und visualisieren
- eine **„Table 1“**, **Regressionsmodelle** (logistisch, Cox) und **Überlebenskurven** rechnen
- p-Werte, Konfidenzintervalle, Effektgrößen und **die passende Testwahl** korrekt begründen
- **fortgeschrittene Statistik** anwenden: Power, fehlende Werte, kausale Inferenz,
  diagnostische Genauigkeit und Survival-Analyse
- eine Analyse **reproduzierbar** aufsetzen und als Bericht aus einer Quelle erzeugen
- **Prädiktionsmodelle** bauen, leckfrei validieren, kalibrieren, erklären (SHAP) und einsetzen
- moderne KI einordnen: **Gradient Boosting, Survival-ML, Deep Learning, klinisches NLP/LLMs**

## Rechtliches und Datenschutz

Dieser Kurs dient ausschließlich Lehr- und Fortbildungszwecken und stellt keine klinische oder statistische Beratung dar. Die Website bindet keine Drittanbieter ein: keine Cookies, kein Tracking, keine externen Skripte oder Schriftarten. Deine Einstellungen und dein Lesefortschritt verbleiben ausschließlich lokal in deinem Browser. Wie bei jedem Webseitenaufruf verarbeitet der ausliefernde Server (GitHub Pages) technisch notwendige Zugriffsdaten einschließlich der IP-Adresse.

Verantwortlich für Inhalt und Konzeption: **Dr. Nikolas B. Schrader** · [nikolasschrader.com](https://nikolasschrader.com) · [medizin.dev](https://medizin.dev).

Ausführliche Informationen findest du unter **[Rechtliches und Datenschutz](docs/RECHTLICHES.md)**.

---

## Lizenz

Dieses Projekt steht unter zwei Lizenzen:

- **Kursinhalte** (Markdown-Lektionen, Übungen, Lösungen, Abbildungen, Datensätze) unter **[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](LICENSE)**. Du darfst sie frei teilen und anpassen, solange du den Urheber nennst, die Inhalte nicht kommerziell nutzt und Modifikationen unter derselben Lizenz weitergibst.
- **Quellcode** (`website/build.py`, `website/app.js`, `website/style.css`, `lib/*`, `data/*.py`, `module/*/code/*`, `tools/*`) unter der permissiven **[MIT-Lizenz](LICENSE-CODE)**.

Die genaue Aufteilung mit Datei-Zuordnung steht in der Datei [`LICENSE`](LICENSE).
