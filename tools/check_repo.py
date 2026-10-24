"""Repo-wide consistency checks. Run before committing, and in CI.

    ./.venv/bin/python tools/check_repo.py

Every check here exists because the corresponding bug actually shipped once:

  links            README.md linked module/index.html, a build artifact that
                   does not exist in the source tree -> 404 on GitHub.
  placeholders     Copy-paste code blocks must use {{DATEN}}; runnable scripts
                   must not (build.py never rewrites them).
  config           config.json must parse and stay the single source of truth
                   for published URLs.
  seeds            The course promises seed 42 everywhere.
  conventions      Lernziele <= 5, Selbstcheck 3-5 questions, every module has
                   the required sections.
  assets           Every image referenced from markdown must exist.
  code_figures     Figures belong in assets/, never in code/.

Exit code 0 = clean. Non-zero = at least one check failed.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from html import unescape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODULES = sorted(p for p in (ROOT / "module").glob("*") if p.is_dir())

PLACEHOLDER = "__SET_ME__"
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
IMAGE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
FENCE = re.compile(r"```[a-zA-Z]*\n(.*?)```", re.DOTALL)


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []
        self.checks = 0

    def check(self, name: str, problems: list[str]) -> None:
        self.checks += 1
        if problems:
            self.failures.append(name)
            print(f"FAIL  {name}  ({len(problems)} problem(s))")
            for p in problems[:15]:
                print(f"        {p}")
            if len(problems) > 15:
                print(f"        … and {len(problems) - 15} more")
        else:
            print(f"ok    {name}")

    def warn(self, message: str) -> None:
        """Not a failure – a state the maintainer must know about."""
        self.warnings.append(message)
        print(f"WARN  {message}")


def markdown_files() -> list[Path]:
    files = [ROOT / "README.md"]
    files += sorted((ROOT / "docs").glob("*.md"))
    files += [ROOT / "data" / "README.md"]
    files += sorted((ROOT / "module").glob("*/*.md"))
    if (ROOT / "module" / "README.md").exists():
        files.append(ROOT / "module" / "README.md")
    return [f for f in files if f.exists()]


def check_links() -> list[str]:
    bad = []
    for md in markdown_files():
        for m in LINK.finditer(md.read_text(encoding="utf-8")):
            target = m.group(1).strip()
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            path = (md.parent / target.split("#")[0]).resolve()
            if not path.exists():
                bad.append(f"{md.relative_to(ROOT)} -> {target}")
    return bad


def check_images() -> list[str]:
    bad = []
    for md in markdown_files():
        for m in IMAGE.finditer(md.read_text(encoding="utf-8")):
            target = m.group(1).strip()
            if target.startswith("http"):
                continue
            if not (md.parent / target).exists():
                bad.append(f"{md.relative_to(ROOT)} -> {target}")
    return bad


def check_placeholders() -> list[str]:
    bad = []
    # {{DATEN}} must never appear in a runnable script: build.py does not
    # rewrite files under module/*/code/.
    for script in sorted((ROOT / "module").glob("*/code/*")):
        if script.suffix.lower() not in {".py", ".r", ".sql"}:
            continue
        if "{{DATEN}}" in script.read_text(encoding="utf-8", errors="replace"):
            bad.append(f"{script.relative_to(ROOT)} contains {{{{DATEN}}}} (scripts use lib/ helpers)")

    # Copy-paste blocks in markdown must not hard-code a local data path.
    local = re.compile(r'["\'](?:\.\./)*data/[a-z_]+\.(?:csv|json|xlsx)["\']')
    for md in sorted((ROOT / "module").glob("*/*.md")):
        for fence in FENCE.finditer(md.read_text(encoding="utf-8")):
            for hit in local.finditer(fence.group(1)):
                bad.append(f"{md.relative_to(ROOT)} code block hard-codes {hit.group(0)}")
    return bad


def check_config() -> list[str]:
    bad = []
    cfg_path = ROOT / "config.json"
    if not cfg_path.exists():
        return ["config.json is missing"]
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"config.json does not parse: {exc}"]

    for key in ("owner", "repo", "branch", "pages_base_url", "data_base_url",
                "repo_url", "file_base_url", "colab_url"):
        if key not in cfg:
            bad.append(f"config.json misses key '{key}'")
    if not str(cfg.get("data_base_url", "")).endswith("/"):
        bad.append("config.json: data_base_url must end with '/'")

    # Every URL in config.json restates owner/repo/branch. They are only a single
    # source of truth if they agree – otherwise the 200+ file links the website
    # bakes into the lessons point at somebody else's repository.
    owner, repo = cfg.get("owner", ""), cfg.get("repo", "")
    branch = cfg.get("branch", "main")
    if PLACEHOLDER not in f"{owner}{repo}":
        erwartet = {
            "repo_url": f"https://github.com/{owner}/{repo}",
            "file_base_url": f"https://github.com/{owner}/{repo}/blob/{branch}/",
            "colab_url": (f"https://colab.research.google.com/github/{owner}/{repo}"
                          f"/blob/{branch}/notebooks/start.ipynb"),
        }
        # The website has its own domain, so pages_base_url is not derived from
        # owner/repo. The datasets must still live in that site's data/ folder.
        pages = str(cfg.get("pages_base_url", ""))
        if not (pages.startswith("https://") and pages.endswith("/")):
            bad.append("config.json: pages_base_url must be an https URL ending with '/'")
        erwartet["data_base_url"] = f"{pages}data/"
        for key, soll in erwartet.items():
            if key in cfg and cfg[key] != soll:
                bad.append(f"config.json: {key} is {cfg[key]!r}, but owner/repo/branch imply {soll!r}")

        # notebooks/start.ipynb cannot read config.json – Colab has no repo clone.
        # It therefore carries a copy of data_base_url, and a copy can go stale.
        nb = ROOT / "notebooks" / "start.ipynb"
        if nb.exists() and cfg.get("data_base_url", "") not in nb.read_text(encoding="utf-8"):
            bad.append("notebooks/start.ipynb does not contain config.json's data_base_url")

    # The dead-URL bug: a hardcoded repo slug scattered across four files.
    hardcoded = re.compile(r"raw\.githubusercontent\.com/[\w.-]+/[\w.-]+")
    for f in [ROOT / "lib" / "helpers.py", ROOT / "lib" / "helpers.R", ROOT / "website" / "build.py"]:
        if f.exists() and hardcoded.search(f.read_text(encoding="utf-8")):
            bad.append(f"{f.relative_to(ROOT)} hardcodes a raw.githubusercontent URL; read config.json instead")
    return bad


def check_seeds() -> list[str]:
    bad = []
    seed = re.compile(r"(?:random_state|seed|set\.seed|default_rng)\s*[=(]\s*(\d+)")
    allowed = {"42"}
    for script in sorted((ROOT / "module").glob("*/code/*.py")) + sorted((ROOT / "module").glob("*/code/*.R")):
        text = script.read_text(encoding="utf-8", errors="replace")
        for m in seed.finditer(text):
            if m.group(1) not in allowed:
                line = text[: m.start()].count("\n") + 1
                bad.append(f"{script.relative_to(ROOT)}:{line} uses seed {m.group(1)}, expected 42")
    return bad


def check_conventions() -> list[str]:
    bad = []
    for mod in MODULES:
        readme = mod / "README.md"
        if not readme.exists():
            bad.append(f"{mod.name}: no README.md")
            continue
        text = readme.read_text(encoding="utf-8")

        for section in ("## Lernziele", "## Fallstricke und Merksätze"):
            if section not in text:
                bad.append(f"{mod.name}: missing '{section}'")

        if not re.search(r"^> \*\*Teil \d", text, re.M):
            bad.append(f"{mod.name}: missing '> **Teil N – …**' header")

        goals = re.search(r"## Lernziele\n(.*?)\n##", text, re.DOTALL)
        if goals:
            n = len([ln for ln in goals.group(1).split("\n") if ln.strip().startswith("- ")])
            if n > 5:
                bad.append(f"{mod.name}: {n} Lernziele (max 5)")

        check = re.search(r"## Selbstcheck\n(.*?)$", text, re.DOTALL)
        if check:
            n = len(re.findall(r"^\d+\.\s+\*\*", check.group(1), re.M))
            if n and not 3 <= n <= 5:
                bad.append(f"{mod.name}: {n} Selbstcheck-Fragen (want 3–5)")

        for needed in ("uebungen.md", "loesungen.md"):
            if not (mod / needed).exists():
                bad.append(f"{mod.name}: missing {needed}")
    return bad


def check_code_figures() -> list[str]:
    return [
        f"{p.relative_to(ROOT)} – figures belong in assets/, not code/"
        for p in sorted((ROOT / "module").glob("*/code/*"))
        if p.suffix.lower() in {".png", ".svg", ".jpg"}
    ]


def check_dataset_docs() -> list[str]:
    """Every prose file that counts the generated data files must count right.

    This exists because `generate_data.py` grew from eight files to ten while
    `docs/LEHRPLAN.md` and `module/02-werkzeuge/README.md` still said "acht" –
    and module 02 printed an "expected output" block the script no longer
    produced. A learner in the setup module saw output that contradicted the doc.
    """
    bad = []
    data_files = sorted(
        p.name for p in (ROOT / "data").iterdir()
        if p.suffix.lower() in {".csv", ".json", ".xlsx"}
    )
    n = len(data_files)
    zahlwort = {
        7: "sieben", 8: "acht", 9: "neun", 10: "zehn", 11: "elf", 12: "zwölf",
    }
    erwartet = zahlwort.get(n)

    # Any doc that names a *different* German count of generated files is stale.
    prose = [ROOT / "data" / "README.md", ROOT / "docs" / "LEHRPLAN.md",
             ROOT / "module" / "02-werkzeuge" / "README.md"]
    # The count may be bolded alone ("**zehn** generierte Dateien") or together
    # with the noun ("**zehn Dateien**"). LEHRPLAN uses the first form and used
    # to slip through, which is exactly the bug this check exists to catch.
    muster = re.compile(
        r"\*\*(" + "|".join(zahlwort.values()) + r")(?:\s+(?:generierte\s+)?Dateien)?\*\*"
        r"(?:\s+(?:generierte\s+)?Dateien)?"
    )
    for f in prose:
        if not f.exists():
            continue
        for m in muster.finditer(f.read_text(encoding="utf-8")):
            if erwartet and m.group(1) != erwartet:
                bad.append(f"{f.relative_to(ROOT)} says '{m.group(1)} Dateien', "
                           f"but data/ holds {n} ({erwartet})")

    # Every generated file must be downloadable from the site.
    build = (ROOT / "website" / "build.py").read_text(encoding="utf-8")
    block = re.search(r"DATEN_LABELS = \{(.*?)\n\}", build, re.DOTALL)
    if block:
        labelled = set(re.findall(r'"([^"]+\.(?:csv|json|xlsx))"', block.group(1)))
        for missing in sorted(set(data_files) - labelled):
            bad.append(f"data/{missing} is generated but absent from build.py's DATEN_LABELS")
        for phantom in sorted(labelled - set(data_files)):
            bad.append(f"build.py's DATEN_LABELS lists data/{phantom}, which does not exist")
    return bad


# Aliases the course uses without always importing them in a given snippet.
_KNOWN_ALIASES = {
    "np": "numpy",
    "pd": "pandas",
    "plt": "matplotlib.pyplot",
    "sns": "seaborn",
    "sm": "statsmodels.api",
    "smf": "statsmodels.formula.api",
}
# Fenced Python blocks, including those nested inside a blockquote ("> ```python").
_PY_BLOCK = re.compile(r"^(> ?)?```python\n(.*?)^(?:> ?)?```", re.DOTALL | re.MULTILINE)


# Library roots that check_solution_api() could not import, and therefore could
# not check any snippet against. Populated by the check, reported by main().
_UNRESOLVED: set[str] = set()


def check_solution_api() -> list[str]:
    """Static API check on the Python snippets in lessons and solutions.

    Exists because `module/12-regression/loesungen.md` shipped
    `smf.families.Gamma(...)` for who knows how long. `statsmodels.formula.api`
    has no `families` attribute – the block raised AttributeError at runtime and
    nobody noticed, because solution snippets are never executed.

    We do not execute them either (they depend on lesson context). We parse them
    and resolve every attribute chain rooted at a known library alias against the
    *installed* library. That is enough to catch the whole class.

    It is not enough to conclude the snippet runs. This check is blind to:
      * wrong keyword arguments – `pd.read_csv(..., encodings="utf-8")` resolves,
        because `read_csv` exists and we never inspect its signature;
      * methods on returned objects – `result.summary2()`, `df.group_by()`; the
        roots `result`/`df` are locals, not import aliases;
      * wrong positional arguments, wrong dtypes, wrong anything at runtime.
    A green result means "no snippet names an attribute the library lacks".
    """
    import ast
    import importlib

    seen: set[str] = set()
    cache: dict[str, object] = {}
    _UNRESOLVED.clear()

    def resolve(name: str):
        """Import `name`, or fetch it as an attribute of its parent module.

        `from sklearn.inspection import PartialDependenceDisplay` binds a class,
        not a module, so import_module("sklearn.inspection.PartialDependenceDisplay")
        raises. Falling back to getattr keeps the class checkable – and stops the
        failure from being misreported as "sklearn is not installed".
        """
        if name in cache:
            return cache[name]
        cache[name] = None                       # guards against import cycles
        try:
            cache[name] = importlib.import_module(name)
            return cache[name]
        except Exception:
            pass
        if "." in name:
            parent_name, attr = name.rsplit(".", 1)
            parent = resolve(parent_name)
            if parent is not None:
                cache[name] = getattr(parent, attr, None)
                return cache[name]
        else:
            # A genuinely uninstalled library resolves to nothing, so every chain
            # rooted at it is skipped and the check silently passes. That is
            # exactly what happens in pages.yml, which installs only markdown +
            # pygments. Record it so main() can say so out loud.
            _UNRESOLVED.add(name)
        return None

    for md in sorted((ROOT / "module").glob("*/*.md")):
        text = md.read_text(encoding="utf-8")
        for block in _PY_BLOCK.finditer(text):
            code = block.group(2)
            if block.group(1):  # blockquoted snippet: strip the "> " prefix
                code = "\n".join(re.sub(r"^> ?", "", ln) for ln in code.split("\n"))
            if "{{DATEN}}" in code:
                code = code.replace("{{DATEN}}", "https://example.invalid/data/")
            line0 = text[: block.start()].count("\n") + 2
            try:
                tree = ast.parse(code)
            except SyntaxError as exc:
                seen.add(f"{md.relative_to(ROOT)}:{line0 + (exc.lineno or 1) - 1} "
                         f"python block does not parse: {exc.msg}")
                continue

            aliases = dict(_KNOWN_ALIASES)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for a in node.names:
                        if a.asname:
                            aliases[a.asname] = a.name
                        else:
                            # `import urllib.request` binds `urllib`, but the
                            # submodule must be imported for getattr to see it.
                            top = a.name.split(".")[0]
                            aliases[top] = top
                            resolve(a.name)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    for a in node.names:
                        aliases[a.asname or a.name] = f"{node.module}.{a.name}"

            for node in ast.walk(tree):
                if not isinstance(node, ast.Attribute):
                    continue
                chain, cur = [], node
                while isinstance(cur, ast.Attribute):
                    chain.append(cur.attr)
                    cur = cur.value
                if not isinstance(cur, ast.Name) or cur.id not in aliases:
                    continue
                mod = resolve(aliases[cur.id])
                if mod is None:
                    continue
                obj, path = mod, cur.id
                for attr in reversed(chain):
                    path += f".{attr}"
                    try:
                        obj = getattr(obj, attr)
                    except AttributeError:
                        seen.add(f"{md.relative_to(ROOT)}:{line0 + node.lineno - 1} "
                                 f"`{path}` does not exist")
                        break
                    except Exception:
                        break
    return sorted(seen)


def check_r_ground_truth() -> list[str]:
    """Module 12's r.R hardcodes the true ORs and HRs. Keep them honest.

    R cannot import `lib/ground_truth.py`, so the numbers are copied. A copy is a
    second chance to be wrong – and this one already went wrong once, when the
    replayed data-generating process drifted from the generator. Diff the copy
    against `lib/ground_truth.json`, which `python lib/ground_truth.py` writes.
    """
    snapshot = ROOT / "lib" / "ground_truth.json"
    r_file = ROOT / "module" / "12-regression" / "code" / "r.R"
    if not r_file.exists():
        return []
    if not snapshot.exists():
        return [f"lib/ground_truth.json is missing – run: python lib/ground_truth.py"]

    truth = json.loads(snapshot.read_text(encoding="utf-8"))
    text = r_file.read_text(encoding="utf-8")
    problems = []

    # Match a full R numeric literal, sign and exponent included. `[\d.]+` would
    # read `1.387e9` as `1.387` and wave it through while R evaluates 1.387e9.
    literal = re.compile(r'"(\w+)"\s*=\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)')

    for r_name, key in (("true_hr", "true_hr_direct"), ("true_or", "true_or")):
        m = re.search(rf"^{r_name}\s*<-\s*c\((.*?)\)\s*$", text, re.MULTILINE | re.DOTALL)
        if not m:
            problems.append(f"module/12-regression/code/r.R: no `{r_name} <- c(...)` vector")
            continue

        pairs: dict[str, str] = {}
        for pred, value in literal.findall(m.group(1)):
            # R's `vec["x"]` returns the FIRST match, so a duplicate key with a
            # correct value appended would hide a wrong one. Keep the first, and
            # refuse the vector outright – a duplicate is never intentional.
            if pred in pairs:
                problems.append(f"r.R `{r_name}` names '{pred}' twice; R would use the first value")
                continue
            pairs[pred] = value

        for pred, expected in truth[key].items():
            if pred not in pairs:
                problems.append(f"r.R `{r_name}` is missing '{pred}'")
                continue
            # The R literals are rounded to 3 decimals, so agreement to within
            # half a unit in the last place is exactly right.
            if abs(float(pairs[pred]) - expected) > 0.0005:
                problems.append(
                    f"r.R `{r_name}['{pred}']` = {pairs[pred]}, but lib/ground_truth.json "
                    f"says {expected:.3f}. Re-run: python lib/ground_truth.py"
                )
    return problems


def check_copyable_blocks() -> list[str]:
    """Copy-paste lesson blocks must run as pasted – no repo scaffolding, no
    syntax errors.

    A block that reads its data via {{DATEN}} is meant for a browser learner who
    pastes it into a bare console. Lines like `Path(__file__)`, `sys.path.insert`,
    `commandArgs()` or `source(file.path(...))` do nothing there (or throw) and
    are pure confusion. The browserification once left such lines behind – and
    once left orphaned `))))))` debris that broke two R blocks outright.
    """
    import ast

    fence = re.compile(r"(> ?)?```(python|r)\n(.*?)```", re.DOTALL)
    scaffold = re.compile(
        r"Path\(__file__\)|sys\.path\.insert|\bcommandArgs\(|source\(file\.path|"
        r"normalizePath\(|from lib\.helpers|dirname\(dirname")
    problems = []
    for md in sorted((ROOT / "module").glob("*/*.md")):
        text = md.read_text(encoding="utf-8")
        for m in fence.finditer(text):
            lang, code = m.group(2), m.group(3)
            if m.group(1):                     # blockquoted fence: strip "> "
                code = "\n".join(re.sub(r"^> ?", "", ln) for ln in code.split("\n"))
            if "{{DATEN}}" not in code:
                continue                       # a repo-only block may scaffold
            line0 = text[: m.start()].count("\n") + 1
            for treffer in scaffold.finditer(code):
                zeile = code[: treffer.start()].count("\n")
                problems.append(
                    f"{md.relative_to(ROOT)}:{line0 + zeile + 1} copy-paste block "
                    f"contains repo scaffolding ({treffer.group(0)}); a browser "
                    "learner cannot run it")
            if lang == "python":
                try:
                    ast.parse(code.replace("{{DATEN}}", "https://x/"))
                except SyntaxError as exc:
                    problems.append(f"{md.relative_to(ROOT)}:{line0 + (exc.lineno or 1)} "
                                    f"Python block has a syntax error: {exc.msg}")
    return problems


def check_code_prerequisites() -> list[str]:
    """No lesson script may claim it needs the data generated first.

    43 scripts opened with "Prerequisite: python data/generate_data.py". That is
    false twice over: `data/*.csv` are committed (see .gitignore), and when the
    folder is absent `lib/helpers` fetches the same files from the published URL.
    The claim is what made readers believe the repository is mandatory.
    """
    luege = re.compile(r"Prerequisite:\s*python\s+data/generate_data\.py")
    bad = []
    for script in sorted((ROOT / "module").glob("*/code/*")):
        if script.suffix not in {".py", ".R"}:
            continue
        if luege.search(script.read_text(encoding="utf-8", errors="replace")):
            bad.append(f"{script.relative_to(ROOT)} claims generate_data.py is a "
                       "prerequisite; the data files are committed and the helpers "
                       "fall back to the published URL")
    return bad


def check_learning_paths() -> list[str]:
    """The learning paths in prose must be the ones the website actually filters by.

    `website/app.js` holds the authoritative module lists (PFADE); Modul 01 and
    docs/LEHRPLAN.md restate them for the reader. After the renumbering most of
    them collapsed into runs ("09–20"), so the prose writes ranges – which is
    only safe if a machine checks that the range still expands to the same set.
    The prose drifted once already: it kept the pre-renumbering chain.
    """
    app = ROOT / "website" / "app.js"
    if not app.exists():
        return []
    quelle = app.read_text(encoding="utf-8")
    start = quelle.index("var PFADE = {")
    block = quelle[start:quelle.index("};", start)]
    pfade = {name: [int(x) for x in re.findall(r'"(\d\d)"', arr)]
             for name, arr in re.findall(r"(\w+): \[([^\]]*)\]", block)}

    label = {"Mein erstes Paper (statistischer Shortcut)": "paper",
             "Schnell zur Auswertung": "basis",
             "Datenextraktion im Fokus": "daten",
             "Klinische Forschung / Publikation": "forschung",
             "Statistik-Vertiefung": "statistik",
             "Machine Learning / KI": "ki"}
    zeile = re.compile(r"^- \*\*(.+?)\*\* · (\d+) Module: (.+)$", re.M)

    def entfalte(seq: str) -> list[int]:
        aus: list[int] = []
        for tok in seq.split("→"):
            tok = tok.strip()
            m = re.fullmatch(r"(\d\d)–(\d\d)", tok)
            if m:
                aus += list(range(int(m.group(1)), int(m.group(2)) + 1))
            elif re.fullmatch(r"\d\d", tok):
                aus.append(int(tok))
        return aus

    problems = []
    for rel in ("module/01-einfuehrung/README.md", "docs/LEHRPLAN.md"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        gefunden = {m.group(1): m for m in zeile.finditer(text)}
        for name, key in label.items():
            if name not in gefunden:
                problems.append(f"{rel}: learning path '{name}' is missing")
                continue
            m = gefunden[name]
            module = entfalte(m.group(3))
            if module != pfade[key]:
                problems.append(
                    f"{rel}: path '{name}' reads {m.group(3)!r} -> {module}, "
                    f"but app.js PFADE.{key} is {pfade[key]}"
                )
            if int(m.group(2)) != len(pfade[key]):
                problems.append(
                    f"{rel}: path '{name}' claims {m.group(2)} Module, "
                    f"app.js lists {len(pfade[key])}"
                )
    return problems


def check_markdown_lists() -> list[str]:
    """Lists must render as lists.

    Two silent failures, both invisible in the source and both shipped:
      * a list that starts immediately after a paragraph line – Python-Markdown
        needs a blank line, otherwise the `*` land as literal characters inside
        the paragraph ("… (Abhängigkeits-Hölle): * Modell A benötigt …");
      * a sub-list indented three spaces under `1. `, which needs four.

    Neither is detectable by reading the markdown, so render it and look.
    """
    try:
        import markdown
    except ImportError:                       # pragma: no cover
        return []

    quelle = (ROOT / "website" / "build.py").read_text(encoding="utf-8")
    treffer = re.search(r"^MD_EXT = (\[[^\]]*\])", quelle, re.M)
    ext = ast.literal_eval(treffer.group(1)) if treffer else ["extra"]

    fence = re.compile(r"^\s*```")
    problems = []
    for md in sorted((ROOT / "module").glob("*/*.md")) + sorted((ROOT / "docs").glob("*.md")) \
            + [ROOT / "README.md", ROOT / "DEPENDENCIES-R.md", ROOT / "data" / "README.md"]:
        if not md.exists():
            continue
        text = md.read_text(encoding="utf-8")
        # Strip fenced code: a `*` in a code block is not a bullet.
        zeilen, drin = [], False
        for line in text.split("\n"):
            if fence.match(line):
                drin = not drin
            zeilen.append("" if drin or fence.match(line) else line)
        html = markdown.markdown("\n".join(zeilen), extensions=ext, output_format="html5")

        for m in re.finditer(r"<p>(.*?)</p>", html, re.DOTALL):
            klartext = unescape(re.sub(r"<[^>]+>", "", m.group(1)))
            if re.search(r"(?:^|\s)\*\s+\S", klartext) and klartext.count("*") >= 2:
                vorschau = " ".join(klartext.split())[:70]
                problems.append(
                    f"{md.relative_to(ROOT)}: a list did not render as a list – "
                    f"'{vorschau}…'. Add a blank line before it, or indent a "
                    "sub-list by four spaces."
                )
    return problems


def check_module_numbering() -> list[str]:
    """The chapter numbers must BE the reading order.

    They drifted once: modules 29–34 were written last, numbered last, and then
    slotted into earlier parts. The menu read 04, 05, 06, 32, 07 – and every
    "siehe Modul 13" pointed at whichever module happened to own that number.

    Reading order is `TEILE` in website/build.py. Directory number == position in
    it, and each lesson's own `# NN · …` heading == its directory number.
    """
    build = ROOT / "website" / "build.py"
    if not build.exists():
        return []
    quelle = build.read_text(encoding="utf-8")
    teile = re.search(r"^TEILE = \[(.*?)^\]", quelle, re.S | re.M)
    if not teile:
        return ["website/build.py: no TEILE table found"]
    reihenfolge = re.findall(r'"(\d\d-[a-z0-9-]+)"', teile.group(1))

    problems = []
    verzeichnisse = sorted(p.name for p in (ROOT / "module").iterdir() if p.is_dir())
    if verzeichnisse != sorted(reihenfolge):
        fehlt = set(reihenfolge) ^ set(verzeichnisse)
        problems.append(f"module/ and TEILE disagree about which modules exist: {sorted(fehlt)}")
        return problems

    for i, slug in enumerate(reihenfolge, start=1):
        if slug[:2] != f"{i:02d}":
            problems.append(
                f"{slug} is number {i} in the reading order but is called {slug[:2]}. "
                "Renumber the directory, or move it in TEILE."
            )
        heading = re.search(r"^# (\d\d) · ", (ROOT / "module" / slug / "README.md")
                            .read_text(encoding="utf-8"), re.M)
        if heading is None:
            problems.append(f"{slug}/README.md: no '# NN · Titel' heading")
        elif heading.group(1) != slug[:2]:
            problems.append(f"{slug}/README.md heading says {heading.group(1)}, "
                            f"directory says {slug[:2]}")
    return problems


def check_figure_sources() -> list[str]:
    """Every displayed figure links to the code that draws it – or to nothing.

    The site used to point each figure at its whole generating file, and every
    `.svg` at `lib/diagrams.py`, including five diagrams drawn by hand that it
    never produced. Now each figure links to its own function or section. That
    mapping is derived by parsing the generators, so it cannot drift – but a new
    image with no generator would silently lose its link, and a hand-drawn one
    that later gains a generator would keep a stale exemption. Catch both.
    """
    sys.path.insert(0, str(ROOT / "website"))
    try:
        from figure_source import HANDMADE, build_index, displayed_images
    except Exception as exc:               # pragma: no cover - import guard
        return [f"website/figure_source.py could not be imported: {exc}"]

    index = build_index(ROOT)
    gezeigt = displayed_images(ROOT)
    problems = []

    for key in sorted(gezeigt - set(index) - HANDMADE):
        problems.append(
            f"{key} is shown in a lesson but no generator writes it. Add the code "
            "that draws it, or list it in website/figure_source.py::HANDMADE."
        )
    for key in sorted(HANDMADE & set(index)):
        problems.append(
            f"{key} is listed as hand-drawn, but {index[key].source} generates it. "
            "Remove it from website/figure_source.py::HANDMADE."
        )
    for key in sorted(HANDMADE):
        if not (ROOT / key).exists():
            problems.append(f"website/figure_source.py::HANDMADE lists {key}, which does not exist.")
    return problems


def config_is_placeholder() -> bool:
    cfg_path = ROOT / "config.json"
    if not cfg_path.exists():
        return False
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return any(PLACEHOLDER in str(v) for k, v in cfg.items() if k != "_comment")


def main() -> int:
    print(f"Checking {len(MODULES)} modules under {ROOT}\n")
    rep = Report()
    rep.check("internal markdown links resolve", check_links())
    rep.check("referenced images exist", check_images())
    rep.check("{{DATEN}} placeholder contract", check_placeholders())
    rep.check("config.json is the single URL source", check_config())
    rep.check("dataset docs match data/", check_dataset_docs())
    rep.check("lesson snippets use real library APIs", check_solution_api())
    rep.check("R's hardcoded truths match lib/ground_truth.json", check_r_ground_truth())
    rep.check("every figure maps to the code that draws it", check_figure_sources())
    rep.check("chapter numbers follow the reading order", check_module_numbering())
    rep.check("markdown lists render as lists", check_markdown_lists())
    rep.check("learning paths match website/app.js", check_learning_paths())
    rep.check("no script claims a false data prerequisite", check_code_prerequisites())
    rep.check("copy-paste blocks run as pasted", check_copyable_blocks())
    rep.check("seed is 42 everywhere", check_seeds())
    rep.check("module conventions", check_conventions())
    rep.check("no figures committed under code/", check_code_figures())

    # A check that cannot run must never look like a check that passed. With only
    # markdown + pygments installed, every import fails and the API check
    # inspects nothing at all.
    if _UNRESOLVED:
        fehlend = ", ".join(sorted(_UNRESOLVED))
        rep.warn(
            "the lesson-snippet API check could not import: " + fehlend + ".\n"
            "        Snippets using them were NOT checked. Install requirements.txt\n"
            "        (+ requirements-ml.txt) for the full check; pages.yml\n"
            "        does exactly that."
        )

    # Deliberately a warning, not a failure: an unpublished course is a valid
    # state. But the maintainer must not be surprised when CI goes red – the
    # website build refuses to bake placeholder URLs into 200 pages.
    if config_is_placeholder():
        rep.warn(
            f"config.json still holds {PLACEHOLDER}. This is fine while unpublished, "
            "but `python website/build.py` will REFUSE to build (and CI will be red) "
            "until you set 'owner' and 'repo'. Local preview:\n"
            "        DSC_DATA_BASE_URL=http://localhost:8000/data/ python website/build.py"
        )

    print()
    if rep.failures:
        print(f"{len(rep.failures)} of {rep.checks} checks FAILED: {', '.join(rep.failures)}")
        return 1
    suffix = f" ({len(rep.warnings)} warning(s))" if rep.warnings else ""
    print(f"All {rep.checks} checks passed{suffix}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
