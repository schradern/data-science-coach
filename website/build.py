"""Statischer Website-Generator für den Kurs.

Liest die Markdown-Quellen (README, docs/, module/*/) und erzeugt eine
navigierbare HTML-Website im Typeset-Stil unter website/site/.

Einzige Quelle der Wahrheit bleibt das Markdown – die Website wird daraus
reproduzierbar gebaut:

    pip install markdown pygments
    python data/generate_data.py && python data/figures.py   # Daten + Abbildungen
    python website/build.py
    # öffnen: website/site/index.html

Gestaltung: website/style.css (Typeset-Ästhetik, Hell-/Dunkelmodus).
"""
from __future__ import annotations

import json
import os
import re
import shutil
from html import escape, unescape
from pathlib import Path
from urllib.parse import unquote

import markdown
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name
from pygments.util import ClassNotFound

from figure_source import build_index, displayed_images

WURZEL = Path(__file__).resolve().parent.parent
ZIEL = WURZEL / "website" / "site"

# Abbildung -> der Code, der genau sie zeichnet. Einmal gebaut, überall benutzt:
# von fig_code_resolver() für den „Code ansehen"-Link und von baue_abbildungs_seite()
# für die Seite dahinter. Beschränkt auf Abbildungen, die eine Lektion auch zeigt —
# für ein Bild, das niemand sieht, braucht niemand eine Seite.
ABB_INDEX = {k: s for k, s in build_index(WURZEL).items()
             if k in displayed_images(WURZEL)}

# --- Datenquellen (Publikations-Konfiguration) --------------------------------
# Alle veröffentlichten URLs stehen in genau einer Datei: ../config.json.
# Modul-Markdown enthält die literalen Platzhalter {{DATEN}} und {{REPO}} (auch
# in Codeblöcken, z. B. pd.read_csv("{{DATEN}}kohorte.csv")); schreibe() ersetzt
# sie im fertigen HTML jeder Seite, bevor sie geschrieben wird.
#
# {{DATEN}} zeigt auf das data/-Verzeichnis dieser Website selbst — der Build
# kopiert die Datensätze neben die Lektionen. Kein zweiter Ablageort, keine
# Abhängigkeit von raw.githubusercontent.com.
PLATZHALTER = "__SET_ME__"

KONFIG = json.loads((WURZEL / "config.json").read_text(encoding="utf-8"))

DATA_BASE_URL = os.environ.get("DSC_DATA_BASE_URL") or KONFIG["data_base_url"]
REPO_URL = KONFIG["repo_url"]
COLAB_URL = KONFIG["colab_url"]
COLAB_ENABLED = PLATZHALTER not in COLAB_URL
BRANCH = KONFIG.get("branch", "main")
FILE_BASE_URL = KONFIG.get("file_base_url") or f"{REPO_URL}/blob/{BRANCH}/"
TREE_BASE_URL = f"{REPO_URL}/tree/{BRANCH}/"
PAGES_BASE_URL = KONFIG["pages_base_url"]


def pruefe_konfiguration() -> None:
    """Bricht ab, statt tote Links in die Website zu backen.

    Ohne diese Prüfung erzeugt ein nicht gesetztes config.json eine Website, in
    der jeder kopierbare `pd.read_csv("{{DATEN}}…")`-Block und jeder
    Download-Button auf eine 404-Adresse zeigt — ohne dass der Build meckert.
    """
    if PLATZHALTER in DATA_BASE_URL:
        raise SystemExit(
            f"config.json ist nicht konfiguriert ({PLATZHALTER} in data_base_url).\n"
            "Setze 'owner' und 'repo' in config.json und leite pages_base_url,\n"
            "data_base_url, repo_url und colab_url davon ab.\n\n"
            "Nur zum lokalen Prüfen ohne Veröffentlichung:\n"
            "  DSC_DATA_BASE_URL=http://localhost:8000/data/ python website/build.py"
        )
    if not DATA_BASE_URL.endswith("/"):
        raise SystemExit("config.json: data_base_url muss auf '/' enden.")

# Herunterladbare Datendateien: Dateiname → menschenlesbares Label. Nur Dateien,
# die tatsächlich in data/ existieren, werden im Download-Panel angezeigt —
# kein Hardcode-Fail, wenn eine (noch) fehlt.
DATEN_LABELS = {
    "kohorte.csv": "Kohorte (Hauptdatensatz)",
    "kohorte_de.csv": "Kohorte (deutsche Spaltennamen)",
    "vitalwerte.csv": "Vitalwerte (Long-Format)",
    "labor.csv": "Laborwerte",
    "notizen.csv": "Klinische Notizen (Freitext)",
    "bga_ph_wahrheit.csv": "Wahre pH-Werte (Ground Truth, Modul 14)",
    "konkurrenz_risiken.csv": "Konkurrierende Risiken + zeitvariante Exposition (Modul 20)",
    "einschluss.xlsx": "Ein-/Ausschlusskriterien (Excel)",
    "beispiel.json": "Beispiel-API-Antwort (JSON)",
    "fhir_bundle.json": "FHIR Bundle (JSON)",
}

# Cache-Buster: kurzer Inhalts-Hash von style.css + app.js. Ändert sich nur, wenn
# sich die Assets ändern — so lädt der Browser aktualisierte Dateien zuverlässig.
def _asset_version() -> str:
    import hashlib
    h = hashlib.sha1()
    for name in ("style.css", "app.js"):
        p = Path(__file__).resolve().parent / name
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:8]


ASSET_VER = _asset_version()

# --- Kursstruktur ------------------------------------------------------------
TITEL = {
    "01-einfuehrung": "Einführung und Lernpfad",
    "02-werkzeuge": "Entwicklungsumgebung und Werkzeuge",
    "03-grundlagen": "Programmiergrundlagen in Python und R",
    "04-daten-einlesen": "Datenimport aus Dateien und APIs",
    "05-sql": "Datenbank-Abfragen mit SQL",
    "06-transformation": "Datenbereinigung und Datentransformation",
    "08-eda-visualisierung": "Explorative Datenanalyse und Datenvisualisierung",
    "09-deskriptive-statistik": "Deskriptive Statistik und die Table 1",
    "10-inferenzstatistik": "Inferenzstatistik und Hypothesentests",
    "11-bayes-inferenz": "Bayesianische Inferenz",
    "12-regression": "Regressionsmodelle: Lineare, logistische und Cox-Regression",
    "13-studiendesign-power": "Studiendesign und Fallzahlplanung",
    "14-fehlende-werte": "Fehlende Werte und Imputation",
    "15-kausale-inferenz": "Kausale Inferenz und Directed Acyclic Graphs",
    "16-diagnostik-schwellen": "Diagnostische Genauigkeit und Schwellenwerte",
    "17-survival-klassisch": "Klassische Überlebenszeitanalyse",
    "21-statistik-entscheidung": "Auswahl der passenden statistischen Methode",
    "22-reproduzierbare-berichte": "Reproduzierbare Berichte mit Quarto",
    "23-machine-learning": "Einführung in das maschinelle Lernen",
    "24-praediktion-workflow": "Workflow für Prädiktionsmodelle und Data Leakage",
    "25-modellguete-validierung": "Bewertung der Modellgüte und klinische Validierung",
    "26-ensembles-boosting": "Baum-Ensembles und Gradient Boosting",
    "27-erklaerbarkeit": "Erklärbarkeit von Machine-Learning-Modellen",
    "28-survival-ml": "Maschinelles Lernen für Überlebenszeiten",
    "29-unueberwacht-phenotyping": "Unüberwachtes Lernen und Phänotypisierung von Patient:innen",
    "30-deep-learning": "Neuronale Netze und Deep Learning",
    "31-klinische-texte-llm": "Verarbeitung klinischer Freitexte mit LLMs",
    "32-einsatz-governance": "Modelleinsatz, Monitoring und Governance",
    "18-longitudinal-mixed-models": "Mixed-Effects-Modelle für Longitudinaldaten",
    "19-propensity-score-causal": "Propensity Score Matching und Weighting",
    "20-competing-risks-timevariant": "Konkurrierende Risiken und zeitabhängige Cox-Modelle",
    "33-rag-llm-pipelines": "Retrieval-Augmented Generation und Leitlinien-Q&A",
    "07-fhir-omop-praxis": "Patientendaten-Extraktion via FHIR und OMOP",
    "34-design-ki-studien": "Studiendesign zur Validierung klinischer KI-Systeme",
}
TEILE = [
    ("Teil 1 · Grundlagen und Setup",
        ["01-einfuehrung", "02-werkzeuge", "03-grundlagen"]),
    ("Teil 2 · Datenimport, Datenbereinigung und Datenmanagement",
        ["04-daten-einlesen", "05-sql", "06-transformation", "07-fhir-omop-praxis"]),
    ("Teil 3 · Explorative Datenanalyse und deskriptive Statistik",
        ["08-eda-visualisierung", "09-deskriptive-statistik"]),
    ("Teil 4 · Statistische Inferenz und medizinische Statistik",
        ["10-inferenzstatistik", "11-bayes-inferenz", "12-regression", "13-studiendesign-power",
         "14-fehlende-werte", "15-kausale-inferenz", "16-diagnostik-schwellen",
         "17-survival-klassisch", "18-longitudinal-mixed-models",
         "19-propensity-score-causal", "20-competing-risks-timevariant",
         "21-statistik-entscheidung", "22-reproduzierbare-berichte"]),
    ("Teil 5 · Machine Learning und KI in der Medizin",
        ["23-machine-learning", "24-praediktion-workflow", "25-modellguete-validierung",
         "26-ensembles-boosting", "27-erklaerbarkeit", "28-survival-ml",
         "29-unueberwacht-phenotyping", "30-deep-learning", "31-klinische-texte-llm",
         "32-einsatz-governance", "33-rag-llm-pipelines", "34-design-ki-studien"]),
]
REIHENFOLGE = [s for _, slugs in TEILE for s in slugs]
TEIL_VON = {s: t for t, slugs in TEILE for s in slugs}

BESCHREIBUNG = {
    "01-einfuehrung": "Warum Datenanalyse, der Analyse-Zyklus, wie du mit dem Kurs arbeitest.",
    "02-werkzeuge": "Python, R, VS Code, Git: eine reproduzierbare, isolierte Umgebung.",
    "03-grundlagen": "Variablen, Funktionen, Kontrollfluss, DataFrames: Code lesen und schreiben.",
    "04-daten-einlesen": "CSV, Excel, JSON, APIs, FHIR: beliebige Quellen zuverlässig einlesen.",
    "05-sql": "SELECT, JOIN, GROUP BY: Daten gezielt aus Datenbanken ziehen.",
    "06-transformation": "Tidy Data, fehlende Werte, die fünf Kern-Verben, Joins, long/wide.",
    "08-eda-visualisierung": "Verteilungen, Ausreißer, Zusammenhänge; Grammatik der Grafik.",
    "09-deskriptive-statistik": "Lage- und Streumaße, Häufigkeiten, die „Table 1“ jeder Publikation.",
    "10-inferenzstatistik": "p-Wert und Konfidenzintervall richtig lesen; t-Test, χ², Mann-Whitney.",
    "11-bayes-inferenz": "Prior, Likelihood, Posterior; Kredibilitätsintervall und ROPE als Bayes-Antwort auf den p-Wert.",
    "12-regression": "Lineare, logistische und Cox-Regression; OR/HR, Konfounder, Diagnostik.",
    "22-reproduzierbare-berichte": "Quarto, Projektstruktur, Seeds, Versionierung, Datenschutz.",
    "23-machine-learning": "Prädiktion vs. Erklärung, Validierung, Kalibrierung, Bias, Grenzen.",
    "24-praediktion-workflow": "Split, Kreuzvalidierung, Data Leakage, Pipelines, Tuning, Schwellen.",
    "25-modellguete-validierung": "Diskrimination, Kalibrierung, Decision Curves, TRIPOD, externe Validierung.",
    "26-ensembles-boosting": "Entscheidungsbäume, Random Forest, XGBoost/LightGBM, Regularisierung.",
    "27-erklaerbarkeit": "Permutation Importance, PDP/ICE, SHAP: global und lokal erklären.",
    "28-survival-ml": "Random Survival Forests, zeitabhängige AUC, konkurrierende Risiken.",
    "29-unueberwacht-phenotyping": "k-Means, hierarchisch, PCA, UMAP, Patienten-Phänotypisierung.",
    "30-deep-learning": "Neuronale Netze für Tabellen (MLP), CNN-Konzept für Bildgebung.",
    "31-klinische-texte-llm": "NLP, Embeddings, Klassifikation von Notizen, verantwortungsvolle LLM-Nutzung.",
    "32-einsatz-governance": "Modell speichern/überwachen, Drift, Model Cards, MDR und EU AI Act, Fairness.",
    "21-statistik-entscheidung": "Welche statistische Methode passt zu Frage, Outcome, Design und Annahmen?",
    "13-studiendesign-power": "Analyseplan, Fallzahl, Power, Präzision und Ereignisse pro Parameter.",
    "14-fehlende-werte": "Missingness erkennen, Complete Case, Imputation und Sensitivitätsanalyse.",
    "15-kausale-inferenz": "DAGs, Confounding, Propensity Scores und robuste Effektinterpretation.",
    "16-diagnostik-schwellen": "Sensitivität, Spezifität, PPV/NPV, Likelihood Ratios, ROC und Cut-offs.",
    "17-survival-klassisch": "Kaplan-Meier, Log-rank, Cox-Annahmen, absolute Risiken und Zensierung.",
    "18-longitudinal-mixed-models": "Längsschnittdaten analysieren: Random Intercept/Slope, Mixed-Effects Models und GEE.",
    "19-propensity-score-causal": "Selektionsbias reduzieren: Propensity Score Matching, Weighting (IPW) und SMD-Balance.",
    "20-competing-risks-timevariant": "Konkurrierende Risiken (Fine-Gray) und Cox-Regression mit zeitabhängigen Kovariaten.",
    "33-rag-llm-pipelines": "Lokale Leitlinien-Q&A: PDFs parsen, ChromaDB/FAISS Vektordatenbanken und LLM-RAG.",
    "07-fhir-omop-praxis": "Datenextraktion aus APIs: Patient/Observation FHIR-Ressourcen parsen und OMOP-Konzept.",
    "34-design-ki-studien": "Translationale Studien planen: Silent Deployment, Spectrum Bias und klinische KI-Validierung.",
}

ERGEBNIS = {
    "01-einfuehrung": "Du kennst den Kurs, den Analyse-Zyklus und hast die erste Mini-Demo gesehen.",
    "02-werkzeuge": "Du hast Python, R, Editor und Projektumgebung so eingerichtet, dass Code reproduzierbar läuft.",
    "03-grundlagen": "Du kannst einfache Datenobjekte lesen, filtern und in Python/R nachvollziehen.",
    "04-daten-einlesen": "Du lädst CSV, Excel, JSON und einfache Schnittstellen-Daten ohne Rätselraten.",
    "05-sql": "Du ziehst eine klinische Kohorte gezielt aus relationalen Tabellen.",
    "06-transformation": "Du machst aus Rohdaten einen analysierbaren, dokumentierten Datensatz.",
    "08-eda-visualisierung": "Du erkennst Verteilungen, Ausreißer und Muster, bevor du testest.",
    "09-deskriptive-statistik": "Du erzeugst eine saubere Table 1 und liest sie kritisch.",
    "10-inferenzstatistik": "Du wählst Tests passend zur Frage und berichtest p-Wert, Effekt und Unsicherheit.",
    "11-bayes-inferenz": "Du gibst direkte Wahrscheinlichkeitsaussagen über Effekte an und begründest den gewählten Prior.",
    "12-regression": "Du schätzt adjustierte Zusammenhänge und erklärst OR, HR und Konfounder.",
    "22-reproduzierbare-berichte": "Du erzeugst Analyse und Bericht aus einer Quelle.",
    "23-machine-learning": "Du trainierst einen ersten Klassifikator und prüfst AUC, Accuracy-Falle und Kalibrierung.",
    "24-praediktion-workflow": "Du baust einen lecksicheren Prädiktions-Workflow mit Split, Pipeline und Tuning.",
    "25-modellguete-validierung": "Du beurteilst, ob ein Modell klinisch nützt, nicht nur statistisch glänzt.",
    "26-ensembles-boosting": "Du vergleichst Bäume, Random Forest und Boosting als starke Tabellenmodelle.",
    "27-erklaerbarkeit": "Du erklärst globale und lokale Modellentscheidungen ohne Scheinsicherheit.",
    "28-survival-ml": "Du modellierst zeitabhängiges Risiko und prüfst Survival-ML kritisch.",
    "29-unueberwacht-phenotyping": "Du findest Subgruppen und erkennst, wann Cluster nur Artefakte sind.",
    "30-deep-learning": "Du ordnest neuronale Netze realistisch ein und trainierst ein erstes MLP.",
    "31-klinische-texte-llm": "Du arbeitest mit Notizen, Tokens, Embeddings und verantwortungsvoller LLM-Nutzung.",
    "32-einsatz-governance": "Du denkst Deployment, Monitoring, Fairness und Regulierung von Anfang an mit.",
    "21-statistik-entscheidung": "Du wählst Tests und Modelle begründet aus und berichtest sie publikationstauglich.",
    "13-studiendesign-power": "Du prüfst, ob eine Studie groß genug, präzise genug und analysierbar geplant ist.",
    "14-fehlende-werte": "Du erkennst Verzerrung durch fehlende Werte und prüfst, ob dein Befund stabil bleibt.",
    "15-kausale-inferenz": "Du trennst Vorhersage, Assoziation und kausale Effektfrage sauber.",
    "16-diagnostik-schwellen": "Du bewertest Tests und Schwellen klinisch, nicht nur über eine AUC.",
    "17-survival-klassisch": "Du berichtest Überlebenszeiten, Log-rank und Cox-Modelle mit Annahmenprüfung.",
    "18-longitudinal-mixed-models": "Du modellierst wiederholte Patient:innenmessungen mit Mixed Models und GEE.",
    "19-propensity-score-causal": "Du korrigierst Selektionsbias mittels Matching/Weighting und prüfst die Balance.",
    "20-competing-risks-timevariant": "Du analysierst konkurrierende Risiken und zeitabhängige Effekte ohne Bias.",
    "33-rag-llm-pipelines": "Du baust eine lokale Leitlinien-Q&A-Pipeline mittels RAG und Vektordatenbank.",
    "07-fhir-omop-praxis": "Du fragst FHIR-APIs ab, extrahierst Patient:innendaten und verstehst OMOP.",
    "34-design-ki-studien": "Du konzipierst prospektive KI-Validierungsstudien und antizipierst klinischen Bias.",
}

WERKZEUG = {
    "01-einfuehrung": "Orientierung",
    "02-werkzeuge": "Terminal · VS Code · RStudio · Git",
    "03-grundlagen": "Python · R",
    "04-daten-einlesen": "pandas · readr/readxl · JSON",
    "05-sql": "SQL · DuckDB · SQLite",
    "06-transformation": "pandas · dplyr/tidyr",
    "08-eda-visualisierung": "seaborn · ggplot2",
    "09-deskriptive-statistik": "tableone · gtsummary",
    "10-inferenzstatistik": "scipy · R stats",
    "11-bayes-inferenz": "scipy · R stats",
    "12-regression": "statsmodels · survival",
    "22-reproduzierbare-berichte": "Quarto · Git",
    "23-machine-learning": "scikit-learn",
    "24-praediktion-workflow": "Pipeline · Cross-Validation",
    "25-modellguete-validierung": "ROC · Kalibrierung · Decision Curve",
    "26-ensembles-boosting": "Random Forest · XGBoost/LightGBM",
    "27-erklaerbarkeit": "Permutation · PDP · SHAP",
    "28-survival-ml": "Cox · Random Survival Forest",
    "29-unueberwacht-phenotyping": "PCA · k-Means · UMAP",
    "30-deep-learning": "MLP · CNN-Konzept",
    "31-klinische-texte-llm": "NLP · Embeddings · LLMs",
    "32-einsatz-governance": "Model Card · Drift · Fairness",
    "21-statistik-entscheidung": "scipy/R stats · Entscheidungsbaum",
    "13-studiendesign-power": "statsmodels · Power",
    "14-fehlende-werte": "pandas · MICE · Sensitivität",
    "15-kausale-inferenz": "DAG · Propensity Score",
    "16-diagnostik-schwellen": "ROC · Konfusionsmatrix",
    "17-survival-klassisch": "lifelines · survival",
    "18-longitudinal-mixed-models": "statsmodels (MixedLM) · lme4",
    "19-propensity-score-causal": "scikit-learn · base R",
    "20-competing-risks-timevariant": "lifelines · survival (R)",
    "33-rag-llm-pipelines": "TF-IDF · Retrieval",
    "07-fhir-omop-praxis": "FHIR JSON · OMOP-Konzept",
    "34-design-ki-studien": "Studiendesign · Checkliste",
}

CODE_LEXER = {".py": "python", ".R": "r", ".sql": "sql", ".qmd": "markdown"}
CODE_LABEL = {".py": "Python", ".R": "R", ".sql": "SQL", ".qmd": "Quarto"}

# „Wo gebe ich das ein?" – ein knapper, ausklappbarer Hinweis je Codeart, damit
# Einsteiger:innen wissen, wohin die Zeilen gehören. Fence-Sprache → Kategorie.
WHERE_KEY = {"bash": "shell", "sh": "shell", "shell": "shell", "console": "shell",
             "powershell": "shell", "ps1": "shell",
             "python": "python", "py": "python", "r": "r", "sql": "sql"}
WHERE_HINT = {
    "shell": ('Diese Zeilen kommen ins <b>Terminal</b> (macOS: „Terminal", Windows: „PowerShell"). '
              'Vorher in den Projektordner wechseln und die Umgebung aktivieren – Schritt für Schritt '
              'in Modul&nbsp;02. Ein <code>#</code>-Kommentar wird nicht mitausgeführt.'),
    "python": ('Python-Code: in eine Datei mit Endung <code>.py</code> schreiben und mit dem '
               '▶-Knopf in VS&nbsp;Code ausführen – oder Zeile für Zeile in die Python-Konsole. '
               'Setzt die in Modul&nbsp;02 eingerichtete Umgebung voraus.'),
    "r": ('R-Code: in <b>RStudio</b> ins Skriptfenster schreiben und mit '
          '<kbd>Strg</kbd>/<kbd>Cmd</kbd>+<kbd>Enter</kbd> ausführen – oder in die R-Konsole.'),
    "sql": ('SQL-Abfrage: in einem <b>Datenbank-Client</b> ausführen oder aus Python/R heraus '
            'absetzen, wie in Modul&nbsp;05 gezeigt.'),
}

MD_EXT = ["extra", "tables", "sane_lists", "toc", "attr_list"]


# --- Code-Hervorhebung (eigene Behandlung für Sprach-Label) ------------------
def highlight_block(lang: str, code: str) -> str:
    try:
        lexer = get_lexer_by_name(lang or "text")
    except ClassNotFound:
        lexer = get_lexer_by_name("text")
    kern = highlight(code, lexer, HtmlFormatter(cssclass="codehilite", nowrap=False))
    label = {"python": "Python", "r": "R", "sql": "SQL", "bash": "Shell",
             "markdown": "Quarto / Markdown", "text": "Text"}.get(lang, lang.upper() or "Code")
    # data-lang steuert den Sprachfilter (Python/R/Beide); nur py und r werden
    # gefiltert, alles andere (SQL, Shell …) bleibt immer sichtbar.
    dlang = {"python": "py", "r": "r"}.get(lang, "x")
    wkey = WHERE_KEY.get(lang)
    wo_btn = (f'<button type="button" class="wo-btn" aria-expanded="false">Wo eingeben?</button>'
              if wkey else "")
    wo_hint = (f'<div class="wo-hint" hidden><span class="wo-ico" aria-hidden="true">{ICON_INFO}</span>'
               f'<span>{WHERE_HINT[wkey]}</span></div>' if wkey else "")
    return (
        f'<div class="code-wrap" data-lang="{dlang}">'
        f'<div class="code-kopf"><span>{escape(label)}</span>'
        f'<span class="ck-actions">{wo_btn}'
        '<button class="kopieren">Kopieren</button></span></div>'
        f'{wo_hint}{kern}</div>'
    )


_FENCE = re.compile(r"```([\w+-]*)\n(.*?)```", re.DOTALL)


def md_to_html(text: str) -> str:
    """Code-Blöcke separat hervorheben, Rest per Markdown rendern."""
    bloecke: list[str] = []

    def merke(m: re.Match) -> str:
        bloecke.append(highlight_block(m.group(1).strip().lower(), m.group(2)))
        return f"\n\nCODEBLOCKTOKEN{len(bloecke) - 1}END\n\n"

    ohne_code = _FENCE.sub(merke, text)
    html = markdown.markdown(ohne_code, extensions=MD_EXT, output_format="html5")
    for i, b in enumerate(bloecke):
        html = html.replace(f"<p>CODEBLOCKTOKEN{i}END</p>", b).replace(f"CODEBLOCKTOKEN{i}END", b)
    return html


def rewrite_links(html: str) -> str:
    def ersetze(m: re.Match) -> str:
        href = m.group(1)
        if href.startswith(("http://", "https://", "#", "mailto:")):
            return m.group(0)
        anker = ""
        if "#" in href:
            href, anker = href.split("#", 1)
            anker = "#" + anker
        if href.endswith("README.md"):
            href = href[: -len("README.md")] + "index.html"
        elif href.endswith(".md"):
            href = href[:-3] + ".html"
        elif Path(href).suffix in CODE_LEXER:
            href = href + ".html"
        return f'href="{href}{anker}"'

    return re.sub(r'href="([^"]+)"', ersetze, html)


_IMG_P = re.compile(r'<p>\s*(<img\b[^>]*>)\s*(.*?)</p>', re.DOTALL)


def _png_size(pfad: Path):
    """Liest Breite/Höhe aus dem IHDR-Chunk einer PNG-Datei — kein Zusatzpaket nötig."""
    try:
        kopf = pfad.read_bytes()[:33]
        if len(kopf) < 33 or kopf[:8] != b"\x89PNG\r\n\x1a\n":
            return None
        import struct
        w, h = struct.unpack(">II", kopf[16:24])
        return w, h
    except OSError:
        return None


def _svg_size(pfad: Path):
    """Liest Breite/Höhe aus viewBox oder width/height-Attributen eines SVGs."""
    try:
        text = pfad.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    m = re.search(r'viewBox="[\d.\-]+\s+[\d.\-]+\s+([\d.]+)\s+([\d.]+)"', text)
    if m:
        return float(m.group(1)), float(m.group(2))
    mw, mh = re.search(r'width="([\d.]+)"', text), re.search(r'height="([\d.]+)"', text)
    if mw and mh:
        return float(mw.group(1)), float(mh.group(1))
    return None


def wrap_figures(html: str, fig_code=None, base_dir: "Path | None" = None) -> str:
    """Alleinstehende Bilder zu <figure> mit Bildunterschrift (alt) machen.

    fig_code(src) -> URL liefert optional die Quellcode-Seite dieser Abbildung.
    base_dir, falls gesetzt, ist das Quellverzeichnis, in dem `src` aufgelöst
    werden kann — daraus liest wrap_figures die tatsächlichen Bildmaße, um
    width/height zu setzen (verhindert Layout-Sprünge/CLS beim Laden), und
    versieht jedes Bild mit loading="lazy" (Bilder liegen unterhalb des ersten
    Bildschirms, hinter Lernkompass/TOC/Vorspann).
    """
    zaehler = {"n": 0}

    def ersetze(m: re.Match) -> str:
        img, rest = m.group(1), m.group(2).strip()
        alt = re.search(r'alt="([^"]*)"', img)
        cap = unescape(alt.group(1)) if alt else ""
        cap = re.sub(r"^Abbildung:?\s*", "", cap).strip()
        src_m = re.search(r'src="([^"]*)"', img)
        code_url = fig_code(src_m.group(1)) if (fig_code and src_m) else None
        code_link = (f' &middot; <a class="fig-code" href="{code_url}">Code ansehen</a>'
                     if code_url else "")
        zaehler["n"] += 1
        bu = (f'<figcaption><span class="fig-num">Abb. {zaehler["n"]}</span> · '
              f'{escape(cap)}{code_link}</figcaption>') if cap else ""
        data_attr = f' data-code="{code_url}"' if code_url else ""

        dims = ""
        if base_dir is not None and src_m:
            bildpfad = (base_dir / src_m.group(1)).resolve()
            groesse = None
            if bildpfad.suffix.lower() == ".png":
                groesse = _png_size(bildpfad)
            elif bildpfad.suffix.lower() == ".svg":
                groesse = _svg_size(bildpfad)
            if groesse:
                dims = f' width="{round(groesse[0])}" height="{round(groesse[1])}"'
        img = re.sub(r"^<img\b", f'<img loading="lazy" decoding="async"{dims}', img)

        fig = f"<figure{data_attr}>{img}{bu}</figure>"
        if rest:
            return fig + f"<p>{rest}</p>"
        return fig

    return _IMG_P.sub(ersetze, html)


def baue_toc(body_html: str) -> str:
    eintraege = re.findall(r'<(h2|h3) id="([^"]+)">(.*?)</\1>', body_html, re.DOTALL)
    if len([e for e in eintraege if e[0] == "h2"]) < 3:
        return ""
    li = []
    for tag, hid, inner in eintraege:
        text = re.sub(r"<[^>]+>", "", inner).strip()
        cls = "lvl-1" if tag == "h2" else "lvl-2"
        li.append(f'<li class="{cls}"><a href="#{hid}">{text}</a></li>')
    return ('<details class="toc-inline"><summary class="toc-title">Auf dieser Seite</summary>'
            f'<ul id="tocList">{"".join(li)}</ul></details>')


# --- Vorlage -----------------------------------------------------------------
def rel_root(tiefe: int) -> str:
    return "../" * tiefe if tiefe else "./"


ICON_GRID = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">'
             '<rect x="3.5" y="3.5" width="7" height="7" rx="1.6"/>'
             '<rect x="13.5" y="3.5" width="7" height="7" rx="1.6"/>'
             '<rect x="3.5" y="13.5" width="7" height="7" rx="1.6"/>'
             '<rect x="13.5" y="13.5" width="7" height="7" rx="1.6"/></svg>')
ICON_TOP =('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">'
            '<path d="M12 19V5M6 11l6-6 6 6" stroke-linecap="round" stroke-linejoin="round"/></svg>')
ICON_CHECK = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4">'
              '<path d="m5 12 4.5 4.5L19 7" stroke-linecap="round" stroke-linejoin="round"/></svg>')
ICON_MENU = ('<svg class="i-open" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9">'
             '<path d="M4 7h16M4 12h16M4 17h16" stroke-linecap="round"/></svg>'
             '<svg class="i-close" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9">'
             '<path d="M6 6l12 12M18 6L6 18" stroke-linecap="round"/></svg>')
ICON_INFO = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">'
             '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01" stroke-linecap="round"/></svg>')
ICON_CHEV = ('<svg class="chev" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">'
             '<path d="m6 9 6 6 6-6" stroke-linecap="round" stroke-linejoin="round"/></svg>')
ICON_SETTINGS = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">'
                 '<path d="M4 7h10M18 7h2M4 17h2M10 17h10M8 4v6M16 14v6" '
                 'stroke-linecap="round" stroke-linejoin="round"/></svg>')

ICON_RESOURCES = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">'
                  '<path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1-2.5-2.5Z"/>'
                  '<path d="M6 6h10M6 10h10" stroke-linecap="round"/></svg>')

ICON_DATA_SM = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">'
                '<ellipse cx="12" cy="5" rx="9" ry="3"/>'
                '<path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>'
                '<path d="M3 12c0 1.66 4 3 9 3s9-1.34 9-3"/></svg>')

ICON_CODE_SM = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">'
                '<polyline points="16 18 22 12 16 6"/>'
                '<polyline points="8 6 2 12 8 18"/></svg>')

ICON_CHECKLIST_SM = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">'
                     '<rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>'
                     '<polyline points="9 11 11 13 15 9"/></svg>')

ICON_GLOSSARY_SM = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">'
                    '<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/>'
                    '<path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>')

_CHECK_SVG = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" '
              'stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>')


def modul_menu(tiefe: int, aktiv_slug: str = "") -> str:
    """„Module ▾" – Sprung-Menü über den gesamten Kurs, auf jeder Seite gleich.

    Listet alle vorhandenen Module nach Teilen gruppiert; das aktuelle Modul ist
    hervorgehoben. Auf dem Desktop ein aufklappbares Dropdown, auf schmalen
    Viewports Teil der Burger-Schublade (CSS). Eine Quelle der Wahrheit: TEILE.
    """
    r = rel_root(tiefe)
    gruppen = []
    for teil_titel, slugs in TEILE:
        eintraege = []
        gruppe_aktiv = aktiv_slug in slugs
        for slug in slugs:
            if not (WURZEL / "module" / slug / "README.md").exists():
                continue
            num = slug.split("-")[0]
            aktiv = slug == aktiv_slug
            cls = ' class="aktiv"' if aktiv else ""
            cur = ' aria-current="page"' if aktiv else ""
            eintraege.append(
                f'<a{cls}{cur} data-num="{num}" data-slug="{slug}" href="{r}module/{slug}/index.html">'
                f'<span class="nm-num">{num}</span><span class="nm-t">{escape(TITEL[slug])}</span>'
                f'<span class="nm-status" aria-label="Erledigt">{_CHECK_SVG}</span></a>'
            )
        if eintraege:
            cur = ' data-current="true"' if gruppe_aktiv else ""
            gruppen.append(f'<details class="nm-group"{cur} open><summary class="nm-teil">{escape(teil_titel)}</summary>'
                           + "".join(eintraege) + "</details>")
    # Kopf des Menüs: die zwei Ziele, die kein Modul sind. Vorher stand „Zur
    # Modulübersicht" als Link mit leerer Nummer zwischen den Modulen — es sah
    # aus wie ein Modul ohne Nummer. Beides sind Sprungziele, also gehören sie
    # zusammen, abgesetzt und mit Symbol statt Nummernspalte.
    overview_html = (
        '<div class="nm-head">'
        f'<a class="nm-action" href="{r}module/index.html">{ICON_GRID}'
        f'<span>Zur Modulübersicht</span></a>'
        f'<button type="button" class="nm-action" id="resourcesBtn">{ICON_RESOURCES}'
        f'<span>Material und Referenz</span></button>'
        '</div>'
    )
    return (
        '<div class="nav-modules" id="navModules">'
        '<button type="button" class="nm-btn" id="navModulesBtn" '
        'aria-expanded="false" aria-controls="navModulesPanel">'
        f'Module <span class="nm-btn-progress" id="nmBtnProgress"></span> {ICON_CHEV}</button>'
        f'<div class="nm-panel" id="navModulesPanel">'
        + overview_html
        + "".join(gruppen) + "</div></div>"
    )


def topbar(tiefe: int, aktiv_slug: str = "") -> str:
    r = rel_root(tiefe)
    return f"""<header class="topbar">
  <a class="brand" href="{r}index.html"><span>Data Science <span class="dot">·</span> Klinik</span>
    <span class="sub">Klinische Datenanalyse &amp; Machine Learning</span></a>
  <nav class="topnav" id="topnav" aria-label="Hauptnavigation">{modul_menu(tiefe, aktiv_slug)}</nav>
  <div class="topbar-actions">
    <div class="settings" id="settingsBox">
      <button class="iconbtn" id="settingsBtn" title="Ansicht anpassen" aria-label="Ansicht anpassen"
        aria-expanded="false" aria-controls="settingsPanel">{ICON_SETTINGS}</button>
      <div class="settings-panel" id="settingsPanel" aria-label="Kurseinstellungen">
        <div class="settings-title">Ansicht</div>
        {studybar()}
      </div>
    </div>
    <button class="iconbtn burger" id="navBurger" title="Menü" aria-label="Menü öffnen"
      aria-expanded="false" aria-controls="topnav">{ICON_MENU}</button>
  </div>
</header>"""


_ICON_CHECK_SM = ('<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
                  'stroke-width="3"><path d="m5 12 4.5 4.5L19 7" stroke-linecap="round" stroke-linejoin="round"/></svg>')


def done_toggle(slug: str) -> str:
    """Kompakter, ruhiger „erledigt"-Umschalter (oben rechts in der Tab-Zeile)."""
    return (f'<button type="button" class="done-toggle" data-slug="{slug}" aria-pressed="false" '
            f'title="Modul als erledigt markieren">'
            f'<span class="dt-ico" aria-hidden="true">{_ICON_CHECK_SM}</span>'
            f'<span class="dt-label">Als erledigt markieren</span></button>')


def studybar() -> str:
    """Die drei Einstellungen, die bestimmen, wie die Seite aussieht.

    Alle drei sind seitenübergreifend persistent (localStorage). Der Sprachfilter
    blendet Python- oder R-Blöcke aus; „Grundlagen" klappt die Vertiefungen ein;
    das Farbschema folgt standardmäßig dem Betriebssystem.

    Das Farbschema war früher ein Knopf unter einem Trennstrich, neben „Material
    und Referenz" — einem Sprungziel, keiner Einstellung. Es ist aber genau das,
    was die anderen beiden auch sind: eine Ansichtsoption. Also steht es hier.
    """
    # Beschriftung und Optionen liegen in getrennten Zellen, damit die Optionen
    # aller drei Zeilen an derselben Kante beginnen. Vorher stand die Auswahl
    # direkt hinter der Beschriftung, und weil „CODEANSICHT" länger ist als
    # „LERNTIEFE", begann jede Zeile woanders.
    return (
        '<div class="studybar" aria-label="Kursweite Ansicht">'
        '<div class="seg" id="segLevel" data-target="level" data-store="dsc-level" data-default="grund">'
        '<span class="seg-label" id="segLevelLabel">Lerntiefe</span>'
        '<span class="seg-opts" role="group" aria-labelledby="segLevelLabel">'
        '<button type="button" data-val="grund">Grundlagen</button>'
        '<button type="button" data-val="alles">Alles</button></span></div>'
        '<div class="seg" id="segLang" data-target="lang" data-store="dsc-lang" data-default="py">'
        '<span class="seg-label" id="segLangLabel">Codeansicht</span>'
        '<span class="seg-opts" role="group" aria-labelledby="segLangLabel">'
        '<button type="button" data-val="py">Python</button>'
        '<button type="button" data-val="r">R</button>'
        '<button type="button" data-val="beide">Beide</button></span></div>'
        '<div class="seg" id="segTheme" data-target="theme" data-store="dsc-theme" data-default="system">'
        '<span class="seg-label" id="segThemeLabel">Farbschema</span>'
        '<span class="seg-opts" role="group" aria-labelledby="segThemeLabel">'
        '<button type="button" data-val="light">Hell</button>'
        '<button type="button" data-val="dark">Dunkel</button>'
        '<button type="button" data-val="system">System</button></span></div>'
        '</div>'
    )


def format_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.0f} KB"
    return f"{n / 1024 / 1024:.1f} MB"


def data_download_panel(href_prefix: str) -> str:
    """Kompaktes Download-Panel: eine Zeile je vorhandene Datendatei.

    href_prefix zeigt auf das Verzeichnis, in dem die kopierten Dateien liegen
    (siehe main()); "{{DATEN}}<name>" wird beim Schreiben der Seite (schreibe())
    zur echten Roh-URL aufgelöst — dieselbe Datei funktioniert also sowohl als
    direkter Download als auch als copy-paste-fähige Lade-URL für Notebooks.
    """
    rows = []
    for name, label in DATEN_LABELS.items():
        pfad = WURZEL / "data" / name
        if not pfad.exists():
            continue
        groesse = format_size(pfad.stat().st_size)
        rows.append(
            '<div class="dl-row">'
            f'<div class="dl-info"><span class="dl-name">{escape(label)}</span>'
            f'<span class="dl-meta">{escape(name)} · {groesse}</span></div>'
            '<div class="dl-actions">'
            f'<a class="dl-btn" href="{href_prefix}{name}" download>Herunterladen</a>'
            f'<button type="button" class="dl-copy" data-url="{{{{DATEN}}}}{name}">URL kopieren</button>'
            '</div></div>'
        )
    if not rows:
        return ""
    colab = (f'<a class="dl-colab" href="{COLAB_URL}" target="_blank" rel="noopener">'
             'In Google Colab öffnen ↗</a>') if COLAB_ENABLED else ""
    return (
        '<div class="data-downloads" aria-label="Datendateien herunterladen">'
        + "".join(rows) + colab + "</div>"
    )


ICON_SHIELD_SM = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">'
                   '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" stroke-linecap="round" stroke-linejoin="round"/></svg>')


# Ziel-Standalone-Seite je Ressourcen-Kachel (Schlüssel → Pfad relativ zur
# Website-Wurzel). Das Overlay lädt den Seiteninhalt bei Bedarf per fetch()
# nach (kein per-Seite-Duplikat mehr) und fällt bei fehlendem JS/fetch auf
# eine normale Navigation zur Standalone-Seite zurück — dieselbe Seite dient
# also beiden Zwecken, einmal gebaut.
RESOURCE_PAGES = {
    "daten": "data/index.html",
    "projektkompass": "docs/PROJEKTKOMPASS.html",
    "spickzettel": "docs/SPICKZETTEL.html",
    "checkliste": "docs/PUBLIKATIONS_CHECKLISTE.html",
    "glossar": "docs/GLOSSAR.html",
    "rechtliches": "docs/RECHTLICHES.html",
}


def resources_overlay(r: str) -> str:
    """Erzeugt das ausfahrbare Ressourcen-Overlay (Sheet Drawer) auf jeder Seite.

    Enthält nur das Kachel-Menü (klein, identisch auf jeder Seite) — die
    referenzierten Dokumente selbst werden NICHT mehr pro Seite dupliziert;
    app.js lädt sie bei Klick per fetch() aus RESOURCE_PAGES nach.
    """
    def link(key: str) -> str:
        return r + RESOURCE_PAGES[key]

    return f"""<div class="resources-overlay" id="resourcesOverlay" aria-hidden="true">
  <div class="ro-backdrop" id="resourcesBackdrop"></div>
  <div class="ro-sheet" role="dialog" aria-modal="true" aria-label="Ressourcen und Referenzen" tabindex="-1">
    <div class="ro-header">
      <button type="button" class="ro-back" id="resourcesBack" aria-label="Zurück zur Auswahl" style="display:none;">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="19" y1="12" x2="5" y2="12"/><polyline points="12 19 5 12 12 5"/></svg>
      </button>
      <h2 id="roTitle">Kursmaterial &amp; Referenz</h2>
      <button type="button" class="ro-close" id="resourcesClose" aria-label="Ressourcen schließen">×</button>
    </div>
    <div class="ro-body" id="roMenu">
      <div class="ro-section">
        <h3>Material &amp; Referenz</h3>
        <div class="ro-grid">
          <a class="ro-link-btn" data-target="daten" href="{link('daten')}">
            <span class="ro-icon">{ICON_DATA_SM}</span>
            <span class="ro-text">
              <strong>Datenübersicht &amp; Kohorte</strong>
              <span>Beschreibung der 500 Patient:innen umfassenden Kohorte und CSV-Downloads.</span>
            </span>
          </a>
          <a class="ro-link-btn" data-target="projektkompass" href="{link('projektkompass')}">
            <span class="ro-icon">{ICON_CHECKLIST_SM}</span>
            <span class="ro-text">
              <strong>Projektkompass</strong>
              <span>Ein-Seiten-Workflow für Frage, Kohorte, Datenwörterbuch, Leakage und Methode.</span>
            </span>
          </a>
          <a class="ro-link-btn" data-target="spickzettel" href="{link('spickzettel')}">
            <span class="ro-icon">{ICON_CODE_SM}</span>
            <span class="ro-text">
              <strong>Python &amp; R Spickzettel</strong>
              <span>Gegenüberstellung der wichtigsten Befehle für Datenmanipulation und Statistik.</span>
            </span>
          </a>
          <a class="ro-link-btn" data-target="checkliste" href="{link('checkliste')}">
            <span class="ro-icon">{ICON_CHECKLIST_SM}</span>
            <span class="ro-text">
              <strong>Publikations-Checkliste</strong>
              <span>Interaktive Checkliste nach EQUATOR-Standards (STROBE, CONSORT, TRIPOD).</span>
            </span>
          </a>
          <a class="ro-link-btn" data-target="glossar" href="{link('glossar')}">
            <span class="ro-icon">{ICON_GLOSSARY_SM}</span>
            <span class="ro-text">
              <strong>Klinisches Glossar</strong>
              <span>Definitionen und Begriffserklärungen statistischer und medizinischer Konzepte.</span>
            </span>
          </a>
          <a class="ro-link-btn" data-target="rechtliches" href="{link('rechtliches')}">
            <span class="ro-icon">{ICON_SHIELD_SM}</span>
            <span class="ro-text">
              <strong>Rechtliches &amp; Datenschutz</strong>
              <span>Pädagogischer Haftungsausschluss und Informationen zur rein lokalen Datenspeicherung.</span>
            </span>
          </a>
        </div>
      </div>
    </div>
    <div class="ro-content" id="roContent" style="display:none;">
      <div class="ro-detail" id="roDetail"></div>
    </div>
  </div>
</div>"""


def seite(titel: str, titleblock: str, body: str, aktiv: str, tiefe: int,
          klass: str = "") -> str:
    r = rel_root(tiefe)
    # `aktiv` ist auf Modulseiten der Slug (z. B. „05-sql"), sonst leer –
    # damit hebt das „Module"-Menü das aktuelle Modul hervor.
    return f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(titel)} · Data Science Coach</title>
<link rel="stylesheet" href="{r}style.css?v={ASSET_VER}">
<link rel="stylesheet" href="{r}pygments.css?v={ASSET_VER}">
<script>(function(){{var r=document.documentElement;r.setAttribute("data-js","1");try{{var g=localStorage;var t=g.getItem("dsc-theme");if(t&&t!=="system")r.setAttribute("data-theme",t);r.setAttribute("data-lang",g.getItem("dsc-lang")||"py");r.setAttribute("data-level",g.getItem("dsc-level")||"grund");}}catch(e){{}}}})();</script>
<script defer src="{r}app.js?v={ASSET_VER}"></script>
</head>
<body>
<a class="skip-link" href="#article">Direkt zum Inhalt</a>
<div class="progress"><div class="progress-bar" id="progressBar"></div></div>
{topbar(tiefe, aktiv)}
<main class="page{(' ' + klass) if klass else ''}"><article id="article" tabindex="-1">
  {titleblock}
  <div class="body" id="body">{body}</div>
  <footer class="seitenfuss">
    <span>Data Science für Ärztinnen und Ärzte</span>
    <span><a href="{r}index.html">Kursübersicht</a> · <a href="https://nikolasschrader.com">Nikolas B. Schrader</a> · medizin.dev</span>
  </footer>
</article></main>
<button class="fab" id="topBtn" aria-label="Nach oben" title="Nach oben">{ICON_TOP}</button>
{resources_overlay(r)}
</body>
</html>
"""


# --- Inhaltsaufbereitung -----------------------------------------------------
def parse_lektion(text: str):
    """Trennt H1-Titel, Meta-Blockquote und Lernziele vom Fließtext-Markdown."""
    titel, kicker, meta_paare, abstract, rest = None, None, [], None, text
    # H1
    m = re.search(r"^# (.+)$", rest, re.MULTILINE)
    if m:
        titel = m.group(1).strip()
        rest = rest[: m.start()] + rest[m.end():]
    # Meta-Blockquote „> **Teil …** · Dauer: … · Voraussetzung: …"
    m = re.search(r"^> (\*\*Teil.+)$", rest, re.MULTILINE)
    if m:
        teile = [t.strip() for t in m.group(1).split("·")]
        kicker = teile[0].strip("* ")
        for seg in teile[1:]:
            if ":" in seg:
                k, v = seg.split(":", 1)
                meta_paare.append((k.strip(), v.strip()))
        rest = rest[: m.start()] + rest[m.end():]
    # Lernziele-Abschnitt
    m = re.search(r"^## Lernziele\s*\n(.*?)(?=^## )", rest, re.DOTALL | re.MULTILINE)
    if m:
        abstract = m.group(1).strip()
        rest = rest[: m.start()] + rest[m.end():]
    return titel, kicker, meta_paare, abstract, rest


def modul_meta(slug: str) -> dict[str, str]:
    """Liest die knappe Meta-Zeile eines Moduls für Karten und Kurzinfos."""
    pfad = WURZEL / "module" / slug / "README.md"
    if not pfad.exists():
        return {}
    _, _, paare, _, _ = parse_lektion(pfad.read_text(encoding="utf-8"))
    return {k: v for k, v in paare}


def titleblock(kicker, titel, subtitle, meta_paare, subnav, abstract_md):
    parts = ['<div class="titleblock">']
    if kicker:
        parts.append(f'<p class="kicker">{escape(kicker)}</p>')
    parts.append(f'<h1 class="title">{escape(titel)}</h1>')
    if subtitle:
        parts.append(f'<p class="subtitle">{escape(subtitle)}</p>')
    if meta_paare:
        zellen = "".join(
            f'<div class="meta"><span class="label">{escape(k)}</span>'
            f'<span class="value">{escape(v)}</span></div>' for k, v in meta_paare
        )
        parts.append(f'<div class="meta-row">{zellen}</div>')
    # Ruhige Kontrollzeile mit den Ansicht-Tabs (Lektion/Übungen/Lösungen/Quelle).
    if subnav:
        parts.append(f'<div class="controlbar"><div class="cb-row">{subnav}</div></div>')
    parts.append("</div>")
    if abstract_md:
        parts.append('<div class="abstract"><div class="block-title">Lernziele</div>'
                     + md_to_html(abstract_md) + "</div>")
    return "".join(parts)


def sub_nav(slug, aktiv_datei, code_dateien, von_code=False):
    lp = "../" if von_code else ""
    cp = "" if von_code else "code/"
    eintraege = [("index.html", "Lektion")]
    if (WURZEL / "module" / slug / "uebungen.md").exists():
        eintraege.append(("uebungen.html", "Übungen"))
    if (WURZEL / "module" / slug / "loesungen.md").exists():
        eintraege.append(("loesungen.html", "Lösungen"))
    html = ['<nav class="sub-nav">']
    for href, label in eintraege:
        cls = ' class="aktiv"' if href == aktiv_datei else ""
        cur = ' aria-current="page"' if href == aktiv_datei else ""
        html.append(f'<a{cls}{cur} href="{lp}{href}">{label}</a>')
    if code_dateien:
        html.append('<span class="sub-sep" aria-hidden="true"></span>')
        # Module 14 ships five Python and four R scripts. Labelling them all
        # "Python"/"R" gave nine indistinguishable links and no run order.
        # Whenever a language has more than one script, name it — and put the
        # module's entry script (python.py / r.R) first.
        pro_sprache: dict[str, int] = {}
        for cf in code_dateien:
            pro_sprache[cf.suffix] = pro_sprache.get(cf.suffix, 0) + 1

        def sortier(cf) -> tuple:
            haupt = cf.stem in ("python", "r")
            return (CODE_LABEL.get(cf.suffix, cf.suffix), not haupt, cf.name)

        for cf in sorted(code_dateien, key=sortier):
            sprache = CODE_LABEL.get(cf.suffix, cf.name)
            if pro_sprache[cf.suffix] > 1:
                label = sprache if cf.stem in ("python", "r") else f"{sprache}: {cf.stem}"
            else:
                label = sprache
            aktiv = cf.name + ".html" == aktiv_datei
            cls = "src aktiv" if aktiv else "src"
            cur = ' aria-current="page"' if aktiv else ""
            html.append(f'<a class="{cls}"{cur} href="{cp}{cf.name}.html" '
                        f'title="{escape(cf.name)}">{escape(label)}</a>')
    html.append("</nav>")
    return "".join(html)


def lesson_actions(slug: str) -> str:
    """Primäre Anschlussaktion nach der Lektion.

    Prev/next bleibt Navigation zwischen Modulen; diese Leiste macht die
    didaktisch naheliegende Aktion sichtbar: erst üben, dann Lösung prüfen.
    """
    links = []
    if (WURZEL / "module" / slug / "uebungen.md").exists():
        links.append('<a class="la-primary" href="uebungen.html">Übungen starten</a>')
    if (WURZEL / "module" / slug / "loesungen.md").exists():
        links.append('<a href="loesungen.html">Lösungen</a>')
    links.append(done_toggle(slug))
    if not links:
        return ""
    return '<nav class="lesson-actions" aria-label="Nächste Schritte">' + "".join(links) + "</nav>"


def weiter_nav(slug):
    i = REIHENFOLGE.index(slug)
    links = []
    if i > 0:
        p = REIHENFOLGE[i - 1]
        links.append(f'<a href="../{p}/index.html"><div class="richtung">← Zurück</div>'
                     f'<div class="ziel">{p.split("-")[0]} · {escape(TITEL[p])}</div></a>')
    else:
        links.append('<span style="flex:1"></span>')
    if i < len(REIHENFOLGE) - 1:
        n = REIHENFOLGE[i + 1]
        links.append(f'<a class="weiter-rechts" href="../{n}/index.html"><div class="richtung">Weiter →</div>'
                     f'<div class="ziel">{n.split("-")[0]} · {escape(TITEL[n])}</div></a>')
    else:
        links.append('<a class="weiter-rechts" href="../../index.html"><div class="richtung">Abschluss →</div>'
                     '<div class="ziel">Zurück zur Übersicht</div></a>')
    return f'<nav class="weiter-nav">{"".join(links)}</nav>'


def schreibe(pfad: Path, inhalt: str) -> None:
    """Schreibt eine Textdatei — einzige Stelle, an der {{DATEN}}/{{REPO}} im
    fertig zusammengesetzten HTML/Text jeder Seite durch die echten URLs
    ersetzt werden (siehe DATA_BASE_URL/REPO_URL oben). Wirkt auch innerhalb
    von Codeblöcken, da diese zu diesem Zeitpunkt bereits reiner Text sind."""
    inhalt = inhalt.replace("{{DATEN}}", DATA_BASE_URL).replace("{{REPO}}", REPO_URL)
    pfad.parent.mkdir(parents=True, exist_ok=True)
    pfad.write_text(inhalt, encoding="utf-8")


# --- Didaktische Callouts (Zwei-Spuren-Prinzip) ------------------------------
# Die Lektionen markieren Wissen bereits semantisch im Fließtext:
#   **Stolperstein:** …   → Fallstrick (Grundspur, immer sichtbar)
#   **Darüber hinaus:** …  → Für Fortgeschrittene (Vertiefungsspur, einklappbar)
#   **Praxis:** …          → Praxis-Hinweis
#   > **Merksatz:** …      → Kernaussage
# Der Renderer macht daraus visuelle, im „Grundlagen"-Modus reduzierbare Boxen –
# ohne eine Zeile Prosa zu ändern (einzige Quelle der Wahrheit bleibt Markdown).
CALLOUT_ICON = {
    "fallstrick": ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">'
                   '<path d="M12 9v4M12 17h.01M10.3 3.9 2.4 18a2 2 0 0 0 1.7 3h15.8a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"'
                   ' stroke-linecap="round" stroke-linejoin="round"/></svg>'),
    "praxis": ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">'
               '<path d="M9 18h6M10 22h4M12 2a7 7 0 0 0-4 12.7c.6.5 1 1.3 1 2.1h6c0-.8.4-1.6 1-2.1A7 7 0 0 0 12 2Z"'
               ' stroke-linecap="round" stroke-linejoin="round"/></svg>'),
    "merksatz": ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">'
                 '<path d="m9 12 2 2 4-4M12 3l7 4v5c0 4.4-3 7.6-7 9-4-1.4-7-4.6-7-9V7l7-4Z"'
                 ' stroke-linecap="round" stroke-linejoin="round"/></svg>'),
    "adv": ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">'
            '<path d="M12 3l1.9 5.8H20l-4.9 3.6 1.9 5.8L12 14.6 7 18.2l1.9-5.8L4 8.8h6.1L12 3Z"'
            ' stroke-linecap="round" stroke-linejoin="round"/></svg>'),
    "basis": ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">'
              '<path d="M4 19.5V6a2 2 0 0 1 2-2h13v16H6a2 2 0 0 0-2 2zM19 16H6a2 2 0 0 0-2 2M9 8h6"'
              ' stroke-linecap="round" stroke-linejoin="round"/></svg>'),
}

_CALL_MERK = re.compile(r'<blockquote>\s*<p><strong>Merksatz:</strong>\s*(.*?)</p>\s*</blockquote>', re.DOTALL)
# Marker innerhalb eines Absatzes (auch mehrere im selben <p>, ohne Leerzeile).
_MARK = re.compile(r'<strong>(Stolperstein|Darüber hinaus|Praxis|Grundlage):</strong>\s*')
_PARA = re.compile(r'<p>(.*?)</p>', re.DOTALL)


def _render_box(art: str, body: str) -> str:
    body = body.strip()
    if art == "Stolperstein":
        return (f'<div class="callout fallstrick"><div class="c-tag">{CALLOUT_ICON["fallstrick"]}'
                f'<span>Fallstrick</span></div><div class="callout-body"><p>{body}</p></div></div>')
    if art == "Praxis":
        return (f'<div class="callout praxis"><div class="c-tag">{CALLOUT_ICON["praxis"]}'
                f'<span>Praxis</span></div><div class="callout-body"><p>{body}</p></div></div>')
    if art == "Grundlage":
        # Einsteiger-Spur → natives <details>, Standardansicht „Grundlagen": offen
        # (im „Alles"-Modus per JS zugeklappt). Statisch (ohne JS) ebenfalls offen,
        # konsistent mit dem neuen Standard.
        return (f'<details class="callout basis" data-track="basis" open><summary>'
                f'<span class="c-tag">{CALLOUT_ICON["basis"]}<span>Grundlage</span></span>'
                f'<span class="c-hint">Für Einsteiger</span></summary>'
                f'<div class="callout-body"><p>{body}</p></div></details>')
    # „Darüber hinaus" = Vertiefungsspur → natives <details>, Standardansicht
    # „Grundlagen": zugeklappt (für Fortgeschrittene/„Alles"-Modus per JS offen).
    # Statisch geschlossen, damit Einsteiger:innen nicht erschlagen werden.
    return (f'<details class="callout adv" data-track="adv"><summary>'
            f'<span class="c-tag">{CALLOUT_ICON["adv"]}<span>Für Fortgeschrittene</span></span>'
            f'<span class="c-hint">Vertiefung</span></summary>'
            f'<div class="callout-body"><p>{body}</p></div></details>')


# --- Abschnittsweise Vertiefungsspur ----------------------------------------
# Der Tiefen-Umschalter konnte bisher nur einzelne Callouts einklappen. Ein
# Modul wie 12 hat aber 90 Minuten Vertiefung über fünf ganze `##`-Abschnitte;
# im Grundlagen-Modus blieben sie vollständig sichtbar. Markiert die Lektion
#
#     <!-- vertiefung: Titel der Spur -->
#
# im Markdown, so wird ALLES danach bis zum Ende (oder bis
# `<!-- /vertiefung -->`) in ein einziges <details class="vertiefung"> gefasst,
# das app.js im Grundlagen-Modus zuklappt.
_VERTIEFUNG_START = re.compile(r"<!--\s*vertiefung:\s*(.*?)\s*-->")
_VERTIEFUNG_END = re.compile(r"<!--\s*/vertiefung\s*-->")


def transform_vertiefung(html: str, quelle: str = "?") -> str:
    """Wrap a `<!-- vertiefung: … -->` … `<!-- /vertiefung -->` span in a <details>.

    Both failure modes are silent and destructive, so both abort the build:
    a missing end marker folds the rest of the page — footer included — into the
    collapsed box, and a second start marker is ignored outright, leaving a raw
    HTML comment in the body and a section that never becomes a track.
    """
    starts = list(_VERTIEFUNG_START.finditer(html))
    if not starts:
        return html
    if len(starts) > 1:
        raise SystemExit(
            f"{quelle}: {len(starts)} '<!-- vertiefung: -->' Startmarken. "
            "Genau eine pro Datei — die übrigen würden stillschweigend verschwinden."
        )
    start = starts[0]
    titel = escape(start.group(1)) or "Vertiefung"

    ende = _VERTIEFUNG_END.search(html, start.end())
    if ende is None:
        raise SystemExit(
            f"{quelle}: '<!-- vertiefung: -->' wird nie mit '<!-- /vertiefung -->' "
            "geschlossen. Der gesamte Rest der Seite läge im eingeklappten Bereich."
        )
    inhalt_start = start.end()
    inhalt_ende = ende.start()
    rest = html[ende.end():]

    inhalt = html[inhalt_start:inhalt_ende]
    box = (
        f'<details class="vertiefung" data-track="adv">'
        f'<summary><span class="c-tag">{CALLOUT_ICON["adv"]}<span>{titel}</span></span>'
        f'<span class="c-hint">Vertiefung — im Grundlagen-Modus eingeklappt</span></summary>'
        f'<div class="vertiefung-body">{inhalt}</div>'
        f'</details>'
    )
    return html[: start.start()] + box + rest


# ---------------------------------------------------------------------------
# Dateiverweise -> Links ins Repository
#
# Die Lektionen nennen ständig Dateien: `lib/helpers.py`, `kohorte.csv`,
# `tools/check_repo.py`. Gedruckt auf einer Website ist so ein Pfad eine
# Sackgasse — man weiß, dass die Datei existiert, kommt aber nicht hin. Jeder
# Verweis auf eine Datei, die es wirklich gibt, wird deshalb zu einem Link auf
# genau diese Datei im Repository.
#
# Zwei Regeln halten das ehrlich:
#   * Verlinkt wird nur, was auf der Festplatte existiert. Tote Links kann diese
#     Funktion nicht erzeugen.
#   * Ein bloßer Dateiname wie `figures.py` gibt es siebzehnmal. Er wird nur
#     verlinkt, wenn er im ganzen Repository eindeutig ist. Sonst bliebe die
#     Wahl des Ziels geraten.
# ---------------------------------------------------------------------------
_IGNORIERTE_ORDNER = {".venv", ".git", "__pycache__", "site", ".ipynb_checkpoints"}
# `data/` steht in jeder zweiten Zeile Prosa; ohne Ordnerliste würde jedes
# Vorkommen zu einem Link. Nur diese wenigen Ordner sind gemeint.
_VERLINKTE_ORDNER = {"data", "docs", "lib", "module", "tools", "notebooks", "website"}
# Dateien, die im Kurs als Begriff vorkommen, nicht als Verweis auf dieses Repo.
# Modul 22 lässt eine erfundene `.gitignore` einer Kollegin korrigieren — ein
# Link auf die echte Datei dieses Projekts wäre schlicht das falsche Ziel.
_NICHT_VERLINKEN = {".gitignore"}


def _datei_index(wurzel: Path) -> tuple[dict[str, str], set[str]]:
    """(Pfad oder eindeutiger Dateiname) -> Repo-Pfad, plus die Ordnermenge."""
    pfade: dict[str, str] = {}
    basisnamen: dict[str, list[str]] = {}
    for p in wurzel.rglob("*"):
        if any(teil in _IGNORIERTE_ORDNER for teil in p.parts):
            continue
        if not p.is_file():
            continue
        rel = p.relative_to(wurzel).as_posix()
        pfade[rel] = rel
        basisnamen.setdefault(p.name, []).append(rel)

    for name, treffer in basisnamen.items():
        if len(treffer) == 1 and name not in pfade:
            pfade[name] = treffer[0]
    return pfade, _VERLINKTE_ORDNER


DATEI_PFADE, DATEI_ORDNER = _datei_index(WURZEL)

# <pre>-Blöcke und bereits verlinkte Bereiche bleiben unangetastet.
_GESCHUETZT = re.compile(r"<pre\b.*?</pre>|<a\b.*?</a>", re.DOTALL)
_INLINE_CODE = re.compile(r"<code>([^<>]+)</code>")


def _aufloesen(ziel: str, kontext: str | None) -> str | None:
    """Repo-Pfad zu einem Dateiverweis, oder None.

    `code/python.py` steht in jeder Lektion und meint jedes Mal eine andere
    Datei — nämlich die des Moduls, das man gerade liest. Deshalb zählt der
    Kontext der Seite mehr als der globale Index.
    """
    if ziel in _NICHT_VERLINKEN:
        return None
    if kontext:
        for kandidat in (f"{kontext}/{ziel}", f"{kontext}/code/{ziel}"):
            if kandidat in DATEI_PFADE:
                return kandidat
    if ziel in DATEI_PFADE:
        return DATEI_PFADE[ziel]
    return None


def verlinke_dateireferenzen(html: str, kontext: str | None = None) -> str:
    def ersetze(m: re.Match) -> str:
        ziel = unescape(m.group(1)).strip().rstrip("/")
        pfad = _aufloesen(ziel, kontext)
        if pfad:
            url, titel = FILE_BASE_URL + pfad, pfad
        elif ziel in DATEI_ORDNER and (WURZEL / ziel).is_dir():
            url, titel = TREE_BASE_URL + ziel, f"{ziel}/"
        else:
            return m.group(0)
        return (f'<a class="dateilink" href="{url}" '
                f'title="{escape(titel)} im Repository">{m.group(0)}</a>')

    teile, letzte = [], 0
    for schutz in _GESCHUETZT.finditer(html):
        teile.append(_INLINE_CODE.sub(ersetze, html[letzte:schutz.start()]))
        teile.append(schutz.group(0))
        letzte = schutz.end()
    teile.append(_INLINE_CODE.sub(ersetze, html[letzte:]))
    return "".join(teile)


def _kontext_von(quelle: str) -> str | None:
    """Das Verzeichnis, relativ zu dem eine Seite ihre Dateiverweise meint."""
    if quelle in ("?", ""):
        return None
    ordner = Path(quelle).parent.as_posix()
    return None if ordner == "." else ordner


def transform_callouts(html: str) -> str:
    def merk(m: re.Match) -> str:
        return (f'<div class="callout merksatz"><div class="c-tag">{CALLOUT_ICON["merksatz"]}'
                f'<span>Merksatz</span></div><div class="callout-body"><p>{m.group(1)}</p></div></div>')
    html = _CALL_MERK.sub(merk, html)

    def para(m: re.Match) -> str:
        inner = m.group(1)
        marks = list(_MARK.finditer(inner))
        if not marks:
            return m.group(0)  # kein Marker → Absatz unverändert
        pieces = []
        # Text vor dem ersten Marker bleibt normaler Absatz (Marker mitten im Text).
        vor = inner[: marks[0].start()].strip()
        if vor:
            pieces.append(f"<p>{vor}</p>")
        for i, mm in enumerate(marks):
            start = mm.end()
            end = marks[i + 1].start() if i + 1 < len(marks) else len(inner)
            pieces.append(_render_box(mm.group(1), inner[start:end]))
        return "".join(pieces)
    return _absorb_trailing_blocks(_PARA.sub(para, html))


# Ein Marker erfasst nur seinen eigenen <p>-Absatz. Folgt darauf ein Codeblock
# oder eine Liste, die inhaltlich dazugehört, blieb sie AUSSERHALB der Box —
# bei `Darüber hinaus:`/`Grundlage:` (native <details>) hieß das: der Inhalt
# war im Grundlagen-Modus dauerhaft sichtbar, obwohl er eingeklappt sein
# sollte. Wir ziehen genau solche direkt folgenden Blöcke in die Box hinein.
_DETAILS_END = '</div></details>'
_ABSORBABLE = ('<div class="code-wrap"', '<ul>', '<ol>')


def _block_ende(html: str, start: int) -> int:
    """Index hinter dem balancierten Block, der bei `start` beginnt."""
    if html.startswith(("<ul>", "<ol>"), start):
        tag = html[start + 1:start + 3]
        schluss = html.find(f"</{tag}>", start)
        return schluss + len(tag) + 3 if schluss != -1 else -1
    # <div …> — verschachtelte <div> mitzählen.
    tiefe, i = 0, start
    while i < len(html):
        if html.startswith("<div", i):
            tiefe += 1
            i += 4
        elif html.startswith("</div>", i):
            tiefe -= 1
            i += 6
            if tiefe == 0:
                return i
        else:
            i += 1
    return -1


def _absorb_trailing_blocks(html: str) -> str:
    suche = 0
    while True:
        ende = html.find(_DETAILS_END, suche)
        if ende == -1:
            return html
        nach = ende + len(_DETAILS_END)
        rest = html[nach:]
        vorlauf = len(rest) - len(rest.lstrip())
        kandidat = nach + vorlauf
        if not html.startswith(_ABSORBABLE, kandidat):
            suche = nach
            continue
        block_ende = _block_ende(html, kandidat)
        if block_ende == -1:
            suche = nach
            continue
        block = html[kandidat:block_ende]
        html = html[:ende] + block + _DETAILS_END + html[block_ende:]
        suche = ende  # derselbe Kasten darf mehrere Folgeblöcke aufnehmen


# --- Selbstcheck → aktive Erinnerung (aufdeckbare Karten) --------------------
_SELF = re.compile(r'(<h2 id="selbstcheck">.*?</h2>)\s*<ol>(.*?)</ol>', re.DOTALL)
_LI = re.compile(r'<li>(.*?)</li>', re.DOTALL)
# Frage/Antwort-Muster: „<strong>Frage?</strong>, Antwort". Trenner ist das erste
# Satzzeichen (Komma, Doppelpunkt oder Gedankenstrich) nach dem </strong>.
_QA = re.compile(r'\s*<strong>(.*?)</strong>\s*[,:—–]\s*(.*)', re.DOTALL)


def transform_quiz(html: str) -> str:
    def rebuild(m: re.Match) -> str:
        head, inner = m.group(1), m.group(2)
        cards = []
        for i, li in enumerate(_LI.findall(inner), start=1):
            qa = _QA.match(li)
            if not qa:  # Kein Frage/Antwort-Muster → unverändert als Listenpunkt lassen
                cards.append(f'<li>{li}</li>')
                continue
            frage, antwort = qa.group(1).strip(), qa.group(2).strip()
            cards.append(
                f'<div class="quiz-card"><button class="quiz-q" type="button">'
                f'<span class="quiz-num">{i}</span><span class="quiz-text">{frage}</span>'
                f'<span class="quiz-toggle">Antwort zeigen</span></button>'
                f'<div class="quiz-a">{antwort}</div></div>'
            )
        # Falls gar nichts als Karte erkannt wurde: Original beibehalten.
        if all(c.startswith("<li>") for c in cards):
            return m.group(0)
        return head + '<div class="quiz">' + "".join(cards) + "</div>"
    return _SELF.sub(rebuild, html)


# --- Sprachabschnitte gruppieren --------------------------------------------
# Überschriften wie „## 5 In Python" / „## 6 In R" leiten sprachspezifische
# Abschnitte ein. Wir umschließen jeden Abschnitt (Überschrift bis zur nächsten
# gleich-/höherrangigen Überschrift) mit <section data-lang>, damit der
# Sprachfilter den ganzen Abschnitt inkl. Überschrift ein-/ausblendet – nicht
# nur den Codeblock. Standard ist Python; R und „Beide" per Schalter.
_HEAD = re.compile(r'<h([23]) id="[^"]*">(.*?)</h\1>', re.DOTALL)


def wrap_lang_sections(html: str) -> str:
    heads = list(_HEAD.finditer(html))
    # Eine waagerechte Linie (`---` im Markdown) beendet einen Sprachabschnitt
    # ebenfalls. Ohne sie reicht ein „## … In Python"-Abschnitt bis zur nächsten
    # Überschrift und verschluckt jede neutrale Prosa dazwischen — ein Ergebnis,
    # eine Abbildung, einen Wegweiser —, die im R-Modus dann verschwindet. Die
    # Linie gibt Autor:innen eine ausdrückliche Grenze: „ab hier wieder für alle".
    rules = [m.start() for m in re.finditer(r"<hr\s*/?>", html)]

    def naechste_grenze(ab: int, lvl: int) -> int:
        kandidaten = [h.start() for h in heads if h.start() > ab and int(h.group(1)) <= lvl]
        kandidaten += [r for r in rules if r > ab]
        return min(kandidaten, default=len(html))

    spans = []
    for h in heads:
        lvl = int(h.group(1))
        txt = re.sub(r"<[^>]+>", "", h.group(2)).lower()
        if re.search(r"\bin python\b", txt):
            lang = "py"
        elif re.search(r"\bin r\b", txt):
            lang = "r"
        else:
            continue
        spans.append((h.start(), naechste_grenze(h.start(), lvl), lang))
    _pruefe_sprachabschnitte(html, spans)
    for s, e, lang in sorted(spans, key=lambda x: -x[0]):  # von hinten, Indizes stabil
        html = (html[:s] + f'<section class="lang-section" data-lang="{lang}">'
                + html[s:e] + "</section>" + html[e:])
    return html


def _pruefe_sprachabschnitte(html: str, spans: list) -> None:
    """Bricht ab, wenn ein Sprachabschnitt neutrale Inhalte verschluckt.

    Genau dieser Fehler ließ Modul 17 die Hazard Ratios nur im Python-Modus
    zeigen — im R-Modus fehlten sie ersatzlos. Ein Ergebnis, eine Abbildung oder
    ein Wegweiser, der NACH dem letzten Codeblock eines Abschnitts steht, gehört
    hinter eine `---`-Linie, damit er für beide Sprachen sichtbar bleibt.
    """
    for s, e, lang in spans:
        koerper = html[s:e]
        letzter_code = koerper.rfind('<div class="code-wrap"')
        if letzter_code == -1:
            continue
        schwanz = koerper[letzter_code:]
        schwanz = re.sub(r'<div class="code-wrap".*?</div>\s*</div>', "", schwanz, flags=re.S)
        # Eine Abbildung oder Überschrift nach dem Code ist NIE sprachspezifisch —
        # sie gehört immer hinter eine `---`-Linie.
        if re.search(r"<figure\b|<h[23]\b", schwanz):
            raise SystemExit(
                f"Sprachabschnitt ({lang}) verschluckt eine Abbildung oder Überschrift "
                "nach dem Codeblock. Setze eine `---`-Linie davor, damit der Inhalt "
                "für beide Sprachen sichtbar bleibt."
            )
        # Prosa nach dem Code darf sprachspezifisch sein (ein R-Paket-Tipp gehört
        # in den R-Abschnitt). Verdächtig ist nur ein Absatz, der ein ERGEBNIS für
        # beide Sprachen trägt — erkennbar daran, dass er die je ANDERE Sprache
        # erwähnt oder eine Kennzahl mit KI/HR/p-Wert nennt. Genau so lag Modul 17.
        andere = r"\bin R\b|\bR-Konsole\b|\bRscript\b" if lang == "py" else r"\bin Python\b|\bpandas\b|smf\.|sklearn"
        for p in re.findall(r"<p>(.*?)</p>", schwanz, re.S):
            klar = re.sub(r"<[^>]+>", "", p)
            trifft_ergebnis = re.search(r"\bHR\s*\(|\bOR\s*\(|95%-KI|\bp\s*[<=]\s*0", klar)
            if len(klar.split()) > 25 and (re.search(andere, klar) or trifft_ergebnis):
                raise SystemExit(
                    f"Sprachabschnitt ({lang}) verschluckt gemeinsame Prosa nach dem "
                    f"Codeblock: {klar[:60]!r}. Steht ein Ergebnis fuer beide Sprachen, "
                    "gehoert es hinter eine `---`-Linie."
                )


# Kurze Einleitungs-Absätze („In R:", „…dasselbe in Python:") direkt vor einem
# sprachspezifischen Codeblock erben dessen Sprache, damit sie mit dem Block
# ein- und ausgeblendet werden – sonst „hängt" die Zeile, wenn der Sprachfilter
# den Block versteckt. Nur kurze, mit „:" endende Absätze gelten als Einleitung.
_LEAD = re.compile(
    r'<p>((?:(?!</p>).)*?:(?:</(?:strong|em|b|i|code)>)*)\s*</p>\s*'
    r'(<div class="code-wrap" data-lang="(py|r)">)', re.S)


def tag_lang_leads(html: str) -> str:
    def rep(m):
        inner, block, lang = m.group(1), m.group(2), m.group(3)
        plain = re.sub(r"<[^>]+>", "", inner).strip()
        if len(plain) > 80:               # nur echte Kurz-Einleitungen taggen
            return m.group(0)
        return f'<p class="lang-lead" data-lang="{lang}">{inner}</p>\n{block}'
    return _LEAD.sub(rep, html)


# Eine 🗣 Code-Verbalisierung erklärt GENAU den Codeblock über ihr. Blendet der
# Sprachfilter diesen Block aus, muss die Erklärung mitgehen — sonst steht in
# der R-Ansicht eine Erläuterung zu Python-Code, dessen Code nirgends zu sehen
# ist. Das Gegenstück zu tag_lang_leads(), nur hinter statt vor dem Block.
# Shell-/SQL-Blöcke tragen data-lang="x", werden nie gefiltert und bleiben
# darum absichtlich ungetaggt.
_CODE_WRAP = re.compile(r'<div class="code-wrap" data-lang="(py|r|x)">')
_VERBAL_OPEN = '<div class="code-verbalization">'


def tag_lang_verbalizations(html: str) -> str:
    # Kein Regex über verschachtelte <div>: einfach die zuletzt geöffnete
    # code-wrap vor jeder Verbalisierung nachschlagen.
    wraps = [(m.start(), m.group(1)) for m in _CODE_WRAP.finditer(html)]
    if not wraps:
        return html
    teile, pos = [], 0
    for treffer in re.finditer(re.escape(_VERBAL_OPEN), html):
        davor = [lang for start, lang in wraps if start < treffer.start()]
        lang = davor[-1] if davor else "x"
        teile.append(html[pos:treffer.start()])
        teile.append(_VERBAL_OPEN if lang == "x"
                     else f'<div class="code-verbalization" data-lang="{lang}">')
        pos = treffer.end()
    teile.append(html[pos:])
    return "".join(teile)


def transform_blockquote_alerts(html: str) -> str:
    def repl(m: re.Match) -> str:
        content = m.group(1)
        m_type = re.match(r'^\s*<p>\s*\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*(?:<br\s*/?>)?\s*(.*)', content, re.DOTALL | re.IGNORECASE)
        if not m_type:
            return m.group(0)

        alert_type = m_type.group(1).upper()
        rest = m_type.group(2).strip()

        if alert_type in ("WARNING", "CAUTION"):
            cls = "fallstrick"
            lbl = "Achtung" if alert_type == "WARNING" else "Gefahr"
            ico = CALLOUT_ICON["fallstrick"]
        elif alert_type in ("TIP",):
            cls = "merksatz"
            lbl = "Tipp"
            ico = CALLOUT_ICON["merksatz"]
        else:
            cls = "praxis"
            lbl = "Hinweis" if alert_type == "NOTE" else "Wichtig"
            ico = CALLOUT_ICON["praxis"]

        if rest.startswith("</p>"):
            rest = rest[4:].strip()
        else:
            if "</p>" in rest:
                p_text, after = rest.split("</p>", 1)
                rest = f"<p>{p_text}</p>{after}"
            else:
                rest = f"<p>{rest}</p>"

        return (f'<div class="callout {cls}">'
                f'<div class="c-tag">{ico}<span>{lbl}</span></div>'
                f'<div class="callout-body">{rest}</div>'
                f'</div>')

    return re.sub(r'<blockquote>(.*?)</blockquote>', repl, html, flags=re.DOTALL)


def transform_checkboxes(html: str) -> str:
    html = re.sub(
        r'<li>\s*\[\s*\]\s*',
        r'<li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox"> ',
        html
    )
    html = re.sub(
        r'<li>\s*\[\s*[xX]\s*\]\s*',
        r'<li class="task-list-item"><input type="checkbox" class="task-list-item-checkbox" checked> ',
        html
    )
    return html


def wrap_tables(html: str) -> str:
    """Wraps <table> elements in <div class="table-scroll"> to make them scrollable on mobile."""
    return re.sub(r'(<table\b[^>]*>.*?</table>)', r'<div class="table-scroll">\1</div>', html, flags=re.DOTALL)


def aufbereiten(md_text: str, fig_code=None, base_dir: "Path | None" = None,
                quelle: str = "?") -> str:
    html = wrap_figures(rewrite_links(md_to_html(md_text)), fig_code=fig_code, base_dir=base_dir)
    html = transform_blockquote_alerts(html)
    html = transform_checkboxes(html)
    html = wrap_tables(html)  # Behebt mobile Layout-Breakage bei Tabellen
    html = transform_quiz(transform_callouts(html))
    html = verlinke_dateireferenzen(html, _kontext_von(quelle))
    html = tag_lang_verbalizations(html)
    return tag_lang_leads(wrap_lang_sections(transform_vertiefung(html, quelle)))


def fig_code_resolver(slug: str):
    """Verlinkt jede Abbildung auf GENAU den Code, der sie zeichnet.

    Früher zeigte jeder Link auf die ganze erzeugende Datei — bei data/figures.py
    fast tausend Zeilen für sechzehn Module. Und jedes .svg zeigte auf
    lib/diagrams.py, auch die fünf von Hand gezeichneten Diagramme, die dort gar
    nicht vorkommen. Wer nichts erzeugt hat, bekommt jetzt keinen Link.
    """
    def resolve(src: str):
        key = f"module/{slug}/{src}"
        if key not in ABB_INDEX:
            return None
        return f"abb/{Path(src).stem}.html"
    return resolve


def quelle_url(quelle: str, tiefe: int) -> str:
    """Link auf die Seite mit der vollständigen erzeugenden Datei."""
    r = rel_root(tiefe)
    teile = Path(quelle).parts
    if teile[0] == "module":                      # module/NN-slug/code/figures.py
        return f"{r}module/{teile[1]}/code/{teile[-1]}.html"
    return f"{r}quellcode/{teile[-1]}.html"       # data/figures.py, lib/diagrams.py


def baue_abbildungs_seite(slug: str, key: str, snippet) -> None:
    """Eine Seite pro Abbildung: das Bild, dann der Code, der es erzeugt.

    Genau so viel Code wie nötig — die Funktion, der Abschnitt oder der
    Zeichenteil von main(), nicht die ganze Datei. Der Weg zur vollständigen
    Datei bleibt einen Klick entfernt.
    """
    stem = Path(key).stem
    bild = Path(key).name
    ganze_datei = quelle_url(snippet.source, 3)

    hinweis = ""
    if snippet.kind == "section":
        hinweis = ('<p class="fig-hint">Dieses Skript läuft von oben nach unten. Der gemeinsame '
                   'Vorspann — Bibliotheken, Kohorte laden, Plot-Stil — steht am Anfang der '
                   'vollständigen Datei und gilt für alle Abbildungen darin.</p>')

    if snippet.kind == "section":
        herkunft = f'Erzeugt im Abschnitt „{escape(snippet.symbol)}"'
    elif snippet.kind == "part":
        herkunft = (f'Erzeugt in <code>{escape(snippet.symbol)}</code> — hier steht nur der '
                    'Teil, der zeichnet')
    else:
        herkunft = f'Erzeugt von <code>{escape(snippet.symbol)}</code>'

    body = (
        f'<figure class="fig-solo"><img src="../assets/{escape(bild)}" alt="{escape(stem)}" '
        f'loading="lazy" decoding="async"></figure>'
        f'<p class="fig-src">{herkunft} in '
        f'<a href="{ganze_datei}">{escape(snippet.source)}</a>, '
        f'Zeile {escape(snippet.zeilen)}.</p>'
        + hinweis
        + md_to_html(f"```{snippet.lang}\n{snippet.code}\n```")
        + f'<p class="fig-back"><a href="../index.html">← zurück zu Modul {slug.split("-")[0]}</a>'
        f' · <a href="{ganze_datei}">vollständige Datei ansehen</a></p>'
    )
    tb = titleblock(f'{slug.split("-")[0]} · {TITEL[slug]}', bild, "Abbildung · Quellcode",
                    [], None, None)
    schreibe(ZIEL / "module" / slug / "abb" / f"{stem}.html",
             seite(bild, tb, body, slug, 3, klass="codepage"))


# --- Voraussetzungs-Hinweis --------------------------------------------------
# Fast jeder Codeblock setzt die in Modul 02 eingerichtete Umgebung und die
# erzeugten Beispieldaten voraus. Statt das in jede Markdown-Datei zu schreiben
# (Redundanz), blendet der Renderer einen einzigen, ausblendbaren Hinweis über
# Lektionen mit Code ein – außer in den Einstiegsmodulen 01 (Konzept, kein Code)
# und 02 (die Einrichtung selbst).
PREREQ_FREI = {"01-einfuehrung", "02-werkzeuge"}


def prereq_banner(slug: str) -> str:
    if slug in PREREQ_FREI:
        return ""
    m02 = "../02-werkzeuge/index.html"
    return (
        '<aside class="prereq" data-prereq>'
        f'<span class="pq-ico" aria-hidden="true">{ICON_INFO}</span>'
        '<div class="pq-text"><strong>Bevor du Code ausführst.</strong> '
        f'Die Beispiele setzen die in <a href="{m02}">Modul&nbsp;02</a> eingerichtete '
        'Umgebung voraus – aktivierte <code>.venv</code>, installierte Pakete – sowie die '
        'einmalig erzeugten Daten: <code>python data/generate_data.py</code>. '
        'Noch nichts installiert? '
        f'<a href="{m02}">Beim Setup starten&nbsp;→</a></div>'
        '<button type="button" class="pq-close" aria-label="Hinweis ausblenden" title="Ausblenden">×</button>'
        '</aside>'
    )


def lesson_brief(slug: str) -> str:
    """Knapper Lernkompass pro Lektion.

    Die Details bleiben in der Lektion; diese Box beantwortet vorab nur:
    Worum geht es, womit arbeite ich, was habe ich am Ende in der Hand?
    """
    return (
        '<aside class="lesson-brief" aria-label="Lernkompass">'
        '<div class="brief-cell"><span class="brief-label">Frage</span>'
        f'<span>{escape(BESCHREIBUNG.get(slug, TITEL[slug]))}</span></div>'
        '<div class="brief-cell"><span class="brief-label">Werkzeug</span>'
        f'<span>{escape(WERKZEUG.get(slug, "Python · R"))}</span></div>'
        '<div class="brief-cell"><span class="brief-label">Ergebnis</span>'
        f'<span>{escape(ERGEBNIS.get(slug, ""))}</span></div>'
        '</aside>'
    )


def practice_brief() -> str:
    """Ein ruhiger Erwartungsrahmen für Übungsseiten."""
    return (
        '<aside class="practice-brief" aria-label="Übungsformat">'
        '<div class="practice-step"><span class="practice-num">1</span>'
        '<span><b>Basis</b> · nachbauen, prüfen, Sicherheit gewinnen</span></div>'
        '<div class="practice-step"><span class="practice-num">2</span>'
        '<span><b>Transfer</b> · kleine Änderung, gleiche Idee</span></div>'
        '<div class="practice-step"><span class="practice-num">3</span>'
        '<span><b>Klinikfall</b> · Ergebnis kurz interpretieren</span></div>'
        '</aside>'
    )


# --- Seitenarten -------------------------------------------------------------
def baue_lektion(slug, code_dateien):
    src = (WURZEL / "module" / slug / "README.md").read_text(encoding="utf-8")
    titel, kicker, meta_paare, abstract, rest = parse_lektion(src)
    titel = titel or TITEL[slug]
    kicker = kicker or TEIL_VON[slug]
    body = (aufbereiten(rest, fig_code=fig_code_resolver(slug), base_dir=WURZEL / "module" / slug,
                        quelle=f"module/{slug}/README.md")
            + lesson_actions(slug) + weiter_nav(slug))
    tb = titleblock(kicker, titel, None, meta_paare,
                    sub_nav(slug, "index.html", code_dateien), abstract)
    # TOC vor den Body setzen; Voraussetzungs-Hinweis (nur Code-Module) ganz oben
    toc = baue_toc(body)
    vorspann = prereq_banner(slug) if code_dateien else ""
    schreibe(ZIEL / "module" / slug / "index.html",
             seite(f"{slug.split('-')[0]} · {TITEL[slug]}", tb,
                   vorspann + lesson_brief(slug) + toc + body, slug, 2))


def baue_unterseite(slug, datei, ausgabe, code_dateien):
    quelle = WURZEL / "module" / slug / datei
    if not quelle.exists():
        return
    src = quelle.read_text(encoding="utf-8")
    m = re.search(r"^# (.+)$", src, re.MULTILINE)
    h1 = m.group(1).strip() if m else ausgabe
    if m:
        src = src[: m.start()] + src[m.end():]
    art = "Übungen" if "uebung" in datei else "Lösungen"
    tb = titleblock(TEIL_VON[slug], art, f"{slug.split('-')[0]} · {TITEL[slug]}", [],
                    sub_nav(slug, ausgabe, code_dateien), None)
    body = aufbereiten(src, quelle=f"module/{slug}/{datei}")
    if "uebung" in datei:
        body = practice_brief() + body
    schreibe(ZIEL / "module" / slug / ausgabe,
             seite(f"{art} · {TITEL[slug]}", tb, body, slug, 2))


# Wozu ist diese Datei da? Eine Quelltext-Seite zeigte bislang nur den nackten
# Code. Wer über den „Python"-Reiter darauf stieß, sah 57 Zeilen ohne Anlass:
# kein Zweck, kein Startbefehl, keine Rückverbindung zur Lektion. Der Text pro
# Dateiname steht hier, damit jede Code-Seite dieselbe Einordnung bekommt.
CODE_ROLLE: dict[str, str] = {
    "python.py": "Das vollständige, lauffähige Python-Skript dieser Lektion. Es führt "
                 "dieselben Schritte aus wie der Abschnitt „In Python“ — am Stück, "
                 "ohne Kürzungen und mit allen Zwischenausgaben.",
    "r.R": "Dasselbe Programm in R. Es rechnet mit denselben Daten und liefert "
           "dieselben Zahlen wie die Python-Fassung.",
    "figures.py": "Erzeugt die Abbildungen dieses Moduls neu und schreibt sie nach "
                  "<code>assets/</code>. Nur nötig, wenn du eine Abbildung verändern willst.",
    "sql.sql": "Die SQL-Abfragen der Lektion zum Nachschlagen — ausgeführt werden sie "
               "von <code>python.py</code> oder direkt in der DuckDB-Konsole.",
    "benchmark.py": "Die Monte-Carlo-Simulation aus der Vertiefung: 500 Wiederholungen "
                    "messen, welche Imputationsmethode ihre eigene Unsicherheit korrekt "
                    "angibt. Läuft mehrere Minuten.",
    "rubin.py": "Rubins Regeln Schritt für Schritt von Hand nachgerechnet — ohne "
                "Imputationspaket, damit jede Zwischengröße sichtbar wird.",
    "rubin.R": "Dieselbe Handrechnung in R, am Ende gegen <code>mice::pool()</code> geprüft.",
    "mice_praxis.py": "MICE in der Praxis: passive Imputation, plausible Wertebereiche, "
                      "Hilfsvariablen und Konvergenzdiagnostik.",
    "mice_praxis.R": "Dasselbe in R mit dem Paket <code>mice</code>.",
    "mnar_delta.py": "MNAR-Sensitivitätsanalyse: Wie weit darf die Annahme „fehlt zufällig“ "
                     "verletzt sein, bis der Schluss kippt?",
    "mnar_delta.R": "Dieselbe Sensitivitätsanalyse in R.",
    "bericht_beispiel.qmd": "Ein vollständiger Quarto-Bericht: Text, Code und Ergebnis in "
                            "einer Datei, aus der Quarto ein PDF oder HTML rendert.",
    "manifest.md": "Das Paket-Manifest, das dieses Modul erzeugt — die Liste aller "
                   "verwendeten Versionen, damit die Auswertung in fünf Jahren noch läuft.",
}
_START = {".py": "python", ".R": "Rscript"}

# Die beiden projektweiten Skripte hinter „Code ansehen" unter den Abbildungen.
QUELLCODE_ROLLE: dict[str, str] = {
    "figures.py": "Erzeugt die gemeinsamen Abbildungen mehrerer Module in einem Lauf — "
                  "ein Skript, das von oben nach unten läuft und eine Abbildung nach der "
                  "anderen zeichnet.",
    "diagrams.py": "Zeichnet die Konzept-Diagramme des Kurses als SVG — je Diagramm eine "
                   "Funktion, ohne Daten, rein aus Rechtecken, Linien und Text.",
}


def code_intro(slug: str, quelle: Path) -> str:
    name = quelle.name
    if name not in CODE_ROLLE:
        # Lieber laut abbrechen als eine Code-Seite ohne Einordnung ausliefern —
        # genau das war der Zustand, den dieser Kasten behebt.
        raise SystemExit(
            f"module/{slug}/code/{name}: kein Eintrag in CODE_ROLLE. Jede "
            "Quelltext-Seite braucht einen Satz, der sagt, wozu die Datei da ist. "
            "Ergänze ihn in website/build.py."
        )
    rolle = CODE_ROLLE[name]
    pfad = f"module/{slug}/code/{name}"

    if quelle.suffix in _START:
        start = (f'<code>{_START[quelle.suffix]} {escape(pfad)}</code> — aus dem '
                 f'Projektordner heraus.')
    elif quelle.suffix == ".qmd":
        start = f'<code>quarto render {escape(pfad)}</code>'
    else:
        start = "Wird nicht direkt ausgeführt — Nachschlagewerk zur Lektion."

    ergebnis = ("Schreibt die PNG-Dateien nach <code>assets/</code>."
                if name == "figures.py"
                else "Gibt die Ergebnisse im Terminal aus.")

    zeilen = [
        ('<div class="ci-row"><span class="ci-label">Ausführen</span>'
         f'<span class="ci-val">{start}</span></div>'),
        ('<div class="ci-row"><span class="ci-label">Daten</span>'
         '<span class="ci-val">Liegen fertig im Projekt unter <code>data/</code>. '
         'Fehlen sie, lädt das Skript dieselben Dateien über die Web-Adresse nach — '
         'erzeugen musst du nichts.</span></div>'),
        ('<div class="ci-row"><span class="ci-label">Ergebnis</span>'
         f'<span class="ci-val">{ergebnis}</span></div>'),
    ]
    return (
        '<aside class="code-intro">'
        f'<p class="ci-lead"><strong>Wozu diese Datei?</strong> {rolle}</p>'
        + "".join(zeilen) +
        '<p class="ci-back">Diese Datei gehört zur Lektion '
        f'<a href="../index.html">{escape(TITEL[slug])}</a>. '
        'Zum Mitrechnen im Browser brauchst du sie nicht — dort genügen die '
        'Codeblöcke der Lektion.</p>'
        '</aside>'
    )


def baue_code_seite(quelle, slug, code_dateien):
    text = quelle.read_text(encoding="utf-8")
    lexer = CODE_LEXER.get(quelle.suffix, "text")
    body = code_intro(slug, quelle) + md_to_html(f"```{lexer}\n{text}\n```")
    label = CODE_LABEL.get(quelle.suffix, quelle.name)
    tb = titleblock(f"{slug.split('-')[0]} · {TITEL[slug]}", quelle.name, f"Quelltext · {label}", [],
                    sub_nav(slug, quelle.name + ".html", code_dateien, von_code=True), None)
    schreibe(ZIEL / "module" / slug / "code" / (quelle.name + ".html"),
             seite(quelle.name, tb, body, slug, 3, klass="codepage"))


def baue_quellcode_seite(quelle: Path, subtitle: str) -> str:
    """Projektweites Skript (data/figures.py, lib/diagrams.py) als Code-Seite.

    Liegt unter quellcode/NAME.html, damit Abbildungen darauf verlinken können.
    """
    text = quelle.read_text(encoding="utf-8")
    rolle = QUELLCODE_ROLLE.get(quelle.name, subtitle)
    intro = (
        '<aside class="code-intro">'
        f'<p class="ci-lead"><strong>Wozu diese Datei?</strong> {rolle}</p>'
        '<div class="ci-row"><span class="ci-label">Ausführen</span>'
        f'<span class="ci-val"><code>python {escape(quelle.parent.name)}/'
        f'{escape(quelle.name)}</code> — aus dem Projektordner heraus.</span></div>'
        '<div class="ci-row"><span class="ci-label">Ergebnis</span>'
        '<span class="ci-val">Schreibt die Bilddateien neben die Lektionen, '
        'nach <code>module/*/assets/</code>.</span></div>'
        '<p class="ci-back">Du erreichst diese Seite über „Code ansehen“ unter einer '
        'Abbildung. Willst du nur den Ausschnitt sehen, der genau diese eine Abbildung '
        'zeichnet, gehe über die Abbildungsseite zurück — dort steht er allein.</p>'
        '</aside>'
    )
    body = intro + md_to_html(f"```python\n{text}\n```")
    tb = titleblock("Quellcode", quelle.name, subtitle, [], None, None)
    schreibe(ZIEL / "quellcode" / (quelle.name + ".html"),
             seite(quelle.name, tb, body, "", 1, klass="codepage"))


def uebersicht_kopf(r: str = "") -> str:
    """Klarer Wiedereinstieg über der Modulübersicht mit interaktiver Pfadauswahl.

    „Weiterlernen" ist die mit Abstand wertvollste Aktion für wiederkehrende
    Besucher:innen und deshalb IMMER sichtbar — nicht hinter einem Klick
    versteckt. Der Pfad-Filter ist ein sekundäres Werkzeug und bleibt daher
    kompakt in einem eigenen, kleinen Ausklapper.
    """
    # Fortschrittsring: `app.js` (paintRing) füllt #ringFill/#ringLabel. Ohne
    # dieses Markup war paintRing toter Code — README und LEHRPLAN versprachen
    # einen Ring, den keine Seite je gerendert hat. r=26 muss zum Umfang
    # passen, den paintRing berechnet (C = 2·π·26).
    # Nutzt die bereits vorhandenen .ring-Regeln in style.css (svg wird dort um
    # -90° gedreht, r=26 passt zum Umfang, den paintRing berechnet).
    ring_html = (
        '<div class="ring" role="img" aria-label="Fortschritt im gewählten Lernpfad">'
        '<svg viewBox="0 0 60 60" aria-hidden="true">'
        '<circle class="ring-bg" cx="30" cy="30" r="26"/>'
        '<circle class="ring-fill" id="ringFill" cx="30" cy="30" r="26"/>'
        '</svg>'
        '<span class="ring-label" id="ringLabel">–</span>'
        '</div>'
    )
    resume_html = (
        '<div class="course-resume">'
        + ring_html
        + '<div class="resume-text">'
        '<span class="resume-label">Weiterlernen</span>'
        f'<a id="continueLink" href="{r}module/01-einfuehrung/index.html">Bei Modul 01 starten</a>'
        '</div></div>'
    )
    # Sieben Pfade, wie in docs/LEHRPLAN.md. `paper` ist der kurze Shortcut,
    # `forschung` der vollständige Publikationsweg — vorher fehlte der zweite
    # Knopf, und die Pfadwahl fiel stillschweigend auf den Shortcut zurück.
    picker_html = (
        '<div class="pfad-picker">'
        '<span class="resume-label">Fokus</span>'
        '<div class="pfad-buttons" id="pfadPicker" aria-label="Lernpfad auswählen">'
        # Beschriftungen und Reihenfolge exakt wie die kanonische Liste in
        # docs/LEHRPLAN.md. Vorher hießen die Knöpfe anders als der Lehrplan,
        # und „Erstes Paper" ließ sich nicht von „Klinische Forschung"
        # unterscheiden — das `title` nennt jetzt Länge und Zweck.
        '<button type="button" data-path="all" class="on" aria-pressed="true" '
        'title="Alle 33 Module in didaktischer Reihenfolge">Vollständig</button>'
        '<button type="button" data-path="basis" aria-pressed="false" '
        'title="6 Module: schnell zu einer ersten Auswertung">Schnell zur Auswertung</button>'
        '<button type="button" data-path="daten" aria-pressed="false" '
        'title="5 Module: CSV, SQL, FHIR — Daten beschaffen und zusammenführen">Datenextraktion</button>'
        '<button type="button" data-path="paper" aria-pressed="false" '
        'title="7 Module: der kürzeste Weg zu Table 1, Regression und Bericht">Mein erstes Paper</button>'
        '<button type="button" data-path="forschung" aria-pressed="false" '
        'title="14 Module: der vollständige Publikationsweg inkl. Kausalität und Survival">Klinische Forschung</button>'
        '<button type="button" data-path="statistik" aria-pressed="false" '
        'title="12 Module: medizinische Statistik in der Tiefe">Statistik-Vertiefung</button>'
        '<button type="button" data-path="ki" aria-pressed="false" '
        'title="12 Module: Machine Learning und KI in der Medizin">Machine Learning / KI</button>'
        '</div></div>'
    )
    tools_html = (
        '<details class="pfad-details" id="pfadDetails">'
        '<summary class="pfad-summary">Lernpfad filtern &amp; Referenz</summary>'
        '<div class="overview-tools">'
        + picker_html
        + '<div class="overview-secondary-box"><span class="resume-label">Referenz</span>'
          f'<a class="overview-secondary" href="{r}docs/PROJEKTKOMPASS.html">Projektkompass</a></div>'
        '</div>'
        '</details>'
    )
    return f'<div class="uebersicht-kopf">{resume_html}{tools_html}</div>'


def landing_cards(r: str = "") -> str:
    """Modulübersicht als Kartenraster für die Inhaltsseite."""
    out = [uebersicht_kopf(r)]
    for i, (teil_titel, slugs) in enumerate(TEILE, start=1):
        nummer, _, name = teil_titel.partition(" · ")
        out.append(f'<div class="teil-head"><div class="teil-num">{escape(nummer)}</div>'
                   f'<h2>{escape(name)}</h2></div>')
        out.append('<div class="karten">')
        for slug in slugs:
            num = slug.split("-")[0]
            meta = modul_meta(slug)
            dauer = meta.get("Dauer", "").replace("~", "")
            out.append(
                f'<a class="karte" href="{r}module/{slug}/index.html" data-slug="{slug}" data-num="{num}">'
                f'<div class="k-top"><span class="k-num">{num}</span>'
                f'<span class="k-step"></span>'
                f'<span class="k-done" aria-hidden="true">{ICON_CHECK}</span></div>'
                f'<div class="k-titel">{escape(TITEL[slug])}</div>'
                f'<div class="k-text">{escape(BESCHREIBUNG[slug])}</div>'
                f'<div class="k-meta"><span>{escape(dauer)}</span><span>{escape(WERKZEUG.get(slug, ""))}</span></div></a>'
            )
        out.append("</div>")
    return "".join(out)


def wrap_landing_sections(html: str) -> str:
    """Jeden ``## ``-Abschnitt der Startseite in eine eigene <section> fassen."""
    schluessel = [("direkt einsteigen", "start"), ("schnellstart", "start"), ("danach", "ergebnis"), ("rechtliches", "rechtliches"), ("lizenz", "lizenz")]
    out, parts = [], re.split(r'(?=<h2)', html)
    for p in parts:
        m = re.match(r'<h2[^>]*>(.*?)</h2>', p, re.S)
        if not m:
            if p.strip():
                out.append(f'<section class="ls ls-intro">{p}</section>')
            continue
        txt = re.sub(r"<[^>]+>", "", m.group(1)).lower()
        key = next((k for w, k in schluessel if w in txt), "allg")
        out.append(f'<section class="ls ls-{key}">{p}</section>')
    return "".join(out)


# Die README ist Doppelverwendung: Repo-Startseite auf GitHub UND Startseite der
# Website. Ein „Kurs lesen → <Live-URL>"-Aufruf gehört auf die GitHub-Seite, wo
# er Besuchende zur Website schickt — auf der Website selbst ist er sinnlos, weil
# man schon dort ist. Alles zwischen diesen Markern erscheint darum nur auf
# GitHub (die HTML-Kommentare sind dort unsichtbar) und wird für die Website
# entfernt.
_NUR_GITHUB = re.compile(r"[ \t]*<!-- nur-github -->.*?<!-- /nur-github -->[ \t]*\n?", re.DOTALL)


def baue_startseite():
    src = (WURZEL / "README.md").read_text(encoding="utf-8")
    src = _NUR_GITHUB.sub("", src)
    m = re.search(r"^# (.+)$", src, re.MULTILINE)
    h1 = m.group(1).strip() if m else "Übersicht"
    if m:
        src = src[: m.start()] + src[m.end():]

    tb = titleblock("Kurs", h1, "Datenanalyse für die Klinik: lernorientiert, reproduzierbar.",
                    [], None, None)

    html = aufbereiten(src, quelle="README.md")
    body = wrap_landing_sections(html)
    schreibe(ZIEL / "index.html", seite("Übersicht", tb, body, "", 0, klass="landing"))


def baue_404_seite():
    """A self-contained 404 page. GitHub Pages serves it for ANY missing path,
    so it cannot rely on relative asset links (those would resolve against the
    missing URL, not the site root). It inlines its styles and links absolutely
    to the course — the graceful landing for a bookmark to a renamed module.
    """
    home = PAGES_BASE_URL
    inhalt = f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Seite nicht gefunden · Data Science Coach</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ margin:0; min-height:100vh; display:grid; place-items:center;
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    background:#F5F7FA; color:#1F2328; padding:2rem; }}
  @media (prefers-color-scheme: dark) {{ body {{ background:#0B0D11; color:#ECEEF2; }} }}
  main {{ max-width:34rem; text-align:center; }}
  .code {{ font-size:3rem; font-weight:800; letter-spacing:-.02em; color:#2A5C8A; margin:0; }}
  @media (prefers-color-scheme: dark) {{ .code {{ color:#5C9BD8; }} }}
  h1 {{ font-size:1.3rem; margin:.6rem 0 1rem; }}
  p {{ line-height:1.6; color:#565C66; }}
  @media (prefers-color-scheme: dark) {{ p {{ color:#A9B0BA; }} }}
  a {{ color:#2A5C8A; font-weight:600; }}
  @media (prefers-color-scheme: dark) {{ a {{ color:#89BCEF; }} }}
  .links {{ margin-top:1.6rem; display:flex; gap:1.2rem; justify-content:center; flex-wrap:wrap; }}
</style>
</head>
<body>
<main>
  <p class="code">404</p>
  <h1>Diese Seite gibt es hier nicht (mehr).</h1>
  <p>Der Kurs wurde neu geordnet, und Modulnummern können sich verschoben haben —
     ein alter Link zeigt dann ins Leere. Nichts ist verloren; die Modulübersicht
     führt dich zum passenden Thema.</p>
  <div class="links">
    <a href="{home}">← Zur Startseite</a>
    <a href="{home}module/index.html">Alle Module</a>
  </div>
</main>
</body>
</html>
"""
    schreibe(ZIEL / "404.html", inhalt)


def baue_moduluebersicht_seite():
    tb = titleblock("Übersicht", "Modulübersicht", 
                    "33 Module in fünf Teilen – von Grundlagen über klinische Datenextraktion und klassische Auswertung bis zu KI.",
                    [], None, None)
    body = (
        '<section class="ls ls-module" id="module">'
        + landing_cards("../") +
        '</section>'
    )
    schreibe(ZIEL / "module" / "index.html", 
             seite("Modulübersicht", tb, body, "module", 1, klass="landing"))


def baue_meta_seite(titel, kicker, quelle: Path, ziel: Path, tiefe: int):
    src = quelle.read_text(encoding="utf-8")
    m = re.search(r"^# (.+)$", src, re.MULTILINE)
    h1 = m.group(1).strip() if m else titel
    if m:
        src = src[: m.start()] + src[m.end():]
    tb = titleblock(kicker, h1, None, [], None, None)
    body = aufbereiten(src, quelle=quelle.relative_to(WURZEL).as_posix())
    schreibe(ziel, seite(titel, tb, baue_toc(body) + body, "", tiefe))


def baue_daten_seite():
    """Datensätze-Seite: Beschreibung aus data/README.md + echtes Download-Panel.

    Das Panel steht bewusst oben, vor der ausführlichen Beschreibung — es ist
    die höchstwertige Aktion auf dieser Seite (siehe Task 2, Download-UX).
    """
    quelle = WURZEL / "data" / "README.md"
    src = quelle.read_text(encoding="utf-8")
    m = re.search(r"^# (.+)$", src, re.MULTILINE)
    h1 = m.group(1).strip() if m else "Datensätze"
    if m:
        src = src[: m.start()] + src[m.end():]
    tb = titleblock("Material", h1, None, [], None, None)
    downloads = data_download_panel("")
    body = aufbereiten(src, quelle="data/README.md")
    schreibe(ZIEL / "data" / "index.html",
             seite("Datensätze", tb, downloads + baue_toc(body) + body, "", 1))


def pruefe_ergebnis() -> None:
    """Prüft die fertige Website, bevor der Build als Erfolg gilt.

    Zwei Fehlerklassen sind hier schon aufgetreten und dürfen nie wieder
    stillschweigend ausgeliefert werden:
      1. ein nicht ersetzter {{DATEN}}/{{REPO}}-Platzhalter,
      2. ein interner Link, der ins Leere zeigt.
    """
    reste: list[str] = []
    tote_links: list[str] = []

    for seite in ZIEL.rglob("*.html"):
        text = seite.read_text(encoding="utf-8", errors="replace")
        rel = seite.relative_to(ZIEL)
        if "{{DATEN}}" in text or "{{REPO}}" in text:
            reste.append(str(rel))
        for treffer in re.finditer(r'(?:href|src)="([^"]+)"', text):
            ziel = treffer.group(1)
            if ziel.startswith(("http://", "https://", "#", "mailto:", "data:", "javascript:")):
                continue
            pfad = ziel.split("#")[0].split("?")[0]
            if not pfad:
                continue
            aufgeloest = (seite.parent / unquote(pfad)).resolve()
            if not aufgeloest.exists():
                tote_links.append(f"{rel} -> {ziel}")

    fehler = []
    if reste:
        fehler.append(f"Nicht ersetzte Platzhalter in {len(reste)} Seite(n): {reste[:5]}")
    if tote_links:
        eindeutig = sorted(set(tote_links))
        fehler.append(
            f"{len(eindeutig)} toter interner Link(s):\n  " + "\n  ".join(eindeutig[:10])
        )
    if fehler:
        raise SystemExit("Build fehlgeschlagen:\n" + "\n".join(fehler))


def main() -> None:
    pruefe_konfiguration()

    shutil.rmtree(ZIEL, ignore_errors=True)
    ZIEL.mkdir(parents=True, exist_ok=True)

    shutil.copy(WURZEL / "website" / "style.css", ZIEL / "style.css")
    shutil.copy(WURZEL / "website" / "app.js", ZIEL / "app.js")
    # Wurzel-Dateien, die aus Seiten heraus verlinkt werden: die beiden
    # Lizenzen (Inhalte CC BY-NC-SA, Quellcode MIT) und die Paket-Manifeste,
    # auf die Modul 02 und DEPENDENCIES-R.md zeigen.
    for datei in ("LICENSE", "LICENSE-CODE", "requirements.txt", "requirements-ml.txt"):
        quelle = WURZEL / datei
        if quelle.exists():
            shutil.copy(quelle, ZIEL / datei)
    pyg = HtmlFormatter(style="default").get_style_defs(".codehilite")
    pyg_dark = HtmlFormatter(style="monokai").get_style_defs('html[data-theme="dark"] .codehilite')
    # Pygments setzt einen eigenen Container-Hintergrund – neutralisieren,
    # damit die Papier-/Dunkelflächen aus style.css greifen.
    override = ('\n.codehilite{background:transparent!important}\n'
                'html[data-theme="dark"] .codehilite{background:transparent!important}\n')
    (ZIEL / "pygments.css").write_text(pyg + "\n" + pyg_dark + override, encoding="utf-8")

    # Startseite
    baue_startseite()
    baue_404_seite()
    baue_moduluebersicht_seite()
    # Abbildungs-Quellcode (für „Code ansehen"-Links an den Abbildungen)
    if (WURZEL / "data" / "figures.py").exists():
        baue_quellcode_seite(WURZEL / "data" / "figures.py",
                             "Erzeugt die Plots der Grundlagen-Module (Seed 42, reproduzierbar).")
    if (WURZEL / "lib" / "diagrams.py").exists():
        baue_quellcode_seite(WURZEL / "lib" / "diagrams.py",
                             "Erzeugt die SVG-Konzeptdiagramme (ein Stil, zentrale Palette).")
    # Doku & Referenz
    baue_meta_seite("Spickzettel", "Referenz", WURZEL / "docs" / "SPICKZETTEL.md",
                    ZIEL / "docs" / "SPICKZETTEL.html", 1)
    baue_meta_seite("Publikations-Checkliste", "Referenz", WURZEL / "docs" / "PUBLIKATIONS_CHECKLISTE.md",
                    ZIEL / "docs" / "PUBLIKATIONS_CHECKLISTE.html", 1)
    baue_meta_seite("Glossar", "Referenz", WURZEL / "docs" / "GLOSSAR.md",
                    ZIEL / "docs" / "GLOSSAR.html", 1)
    baue_meta_seite("Projektkompass", "Referenz", WURZEL / "docs" / "PROJEKTKOMPASS.md",
                    ZIEL / "docs" / "PROJEKTKOMPASS.html", 1)
    baue_meta_seite("Lehrplan", "Dokumentation", WURZEL / "docs" / "LEHRPLAN.md",
                    ZIEL / "docs" / "LEHRPLAN.html", 1)
    baue_meta_seite("Rechtliches", "Referenz", WURZEL / "docs" / "RECHTLICHES.md",
                    ZIEL / "docs" / "RECHTLICHES.html", 1)
    baue_meta_seite("Konventionen", "Dokumentation", WURZEL / "docs" / "KONVENTIONEN.md",
                    ZIEL / "docs" / "KONVENTIONEN.html", 1)
    # Aus dem Repo-Wurzelverzeichnis, aber von Modul 02 aus verlinkt: die
    # R-Paketliste muss als Seite existieren, sonst zeigt der Link ins Leere.
    baue_meta_seite("R-Pakete", "Referenz", WURZEL / "DEPENDENCIES-R.md",
                    ZIEL / "DEPENDENCIES-R.html", 0)
    # Daten (+ Download-UX: alle vorhandenen Dateien aus DATEN_LABELS bereitstellen)
    baue_daten_seite()
    for name in DATEN_LABELS:
        quelle_datei = WURZEL / "data" / name
        if not quelle_datei.exists():
            continue
        ziel_datei = ZIEL / "data" / name
        ziel_datei.parent.mkdir(parents=True, exist_ok=True)
        if quelle_datei.suffix.lower() in (".csv", ".json"):
            schreibe(ziel_datei, quelle_datei.read_text(encoding="utf-8"))
        else:  # z. B. .xlsx — binär kopieren, keine Textsubstitution möglich/nötig
            shutil.copy(quelle_datei, ziel_datei)

    n_code = 0
    n_abb = 0
    for slug in REIHENFOLGE:
        mdir = WURZEL / "module" / slug
        if not (mdir / "README.md").exists():
            continue  # noch nicht geschriebenes Modul
        code_dateien = sorted([p for p in (mdir / "code").glob("*") if p.suffix in CODE_LEXER],
                              key=lambda p: (p.suffix != ".py", p.name))
        # Bilder kopieren (assets/ und code/) – PNG-Plots und SVG-Diagramme
        for unter in ("assets", "code"):
            for muster in ("*.png", "*.svg"):
                for bild in (mdir / unter).glob(muster):
                    ziel = ZIEL / "module" / slug / unter / bild.name
                    ziel.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(bild, ziel)

        baue_lektion(slug, code_dateien)
        baue_unterseite(slug, "uebungen.md", "uebungen.html", code_dateien)
        baue_unterseite(slug, "loesungen.md", "loesungen.html", code_dateien)
        for cf in code_dateien:
            baue_code_seite(cf, slug, code_dateien)
            n_code += 1

        # Eine Seite je Abbildung dieses Moduls, mit genau ihrem Quellcode.
        for key, snippet in ABB_INDEX.items():
            if key.startswith(f"module/{slug}/"):
                baue_abbildungs_seite(slug, key, snippet)
                n_abb += 1

    pruefe_ergebnis()

    seiten = sum(1 for _ in ZIEL.rglob("*.html"))
    print(f"Website gebaut: {seiten} HTML-Seiten "
          f"({n_code} Code-Seiten, {n_abb} Abbildungs-Seiten) → {ZIEL}")
    print(f"Datenquelle: {DATA_BASE_URL}")
    print("Öffnen:", ZIEL / "index.html")


if __name__ == "__main__":
    main()
