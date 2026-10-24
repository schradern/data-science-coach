"""Catch the course's most persistent bug: a number in prose that the code no
longer produces.

    ./.venv/bin/python tools/check_numbers.py            # all modules
    ./.venv/bin/python tools/check_numbers.py 12 30      # only these

Every adversarial review round found the same defect in a new place:

  * module 06 said two columns were missing; the figure showed three.
  * module 02 printed an "Erwartete Ausgabe" the generator no longer emits.
  * module 14 updated its main text after a re-run benchmark but not its
    self-check answers (55 % where the truth was 57 %).
  * module 31 claimed "800+ Spalten"; its own script prints 742.
  * module 12 called exp(beta) an odds ratio in one section and correctly a
    hazard ratio in the next.

The pattern is always the same, so stop finding it by hand.

## How it works

For each module we run `code/python.py`, capture stdout, and collect every
number it prints. Then we collect every number that the module's markdown
asserts *as a result* – that is, numbers inside markdown **tables** and inside
```text fences, which is where computed values live. Prose numbers are ignored:
they are usually thresholds, years, module numbers or formula constants.

A markdown number is reported when no printed number matches it within a
tolerance that respects the number of decimals shown. `1,79` matches `1.7891`;
`0,95` does not match `0.9`.

## What this tool does NOT see

A green run is not a proof that a lesson is correct. It reads numbers only in
markdown tables and ```text fences, so a stale number in prose, in a ```python
fence, or in an image caption passes unnoticed. It compares magnitudes, so a
flipped sign passes. It matches any printed number anywhere in the module's
output, so a value that happens to coincide with an unrelated one passes. And
it says nothing at all about whether the numbers are *right* – only that the
code and the prose still agree.

## Escape hatch

Numbers that legitimately do not come from the module's own script (values from
`lib/ground_truth.py`, from a sibling script such as `benchmark.py`, from the
literature, or from R) are declared per module in `tools/numbers_allow.json`.
Keep that file small and justified – every entry is a number nobody re-checks.

Exit code 0 = clean.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALLOW_FILE = Path(__file__).resolve().parent / "numbers_allow.json"

# Locally everyone runs ./.venv/bin/python. CI installs into the runner's own
# interpreter and never creates a .venv, so hardcoding that path made this tool
# die with FileNotFoundError before checking a single number.
# DSC_PYTHON overrides both, so a second environment (say, a candidate
# dependency stack) can be validated without moving ./.venv out of the way.
_VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
if os.environ.get("DSC_PYTHON"):
    PYTHON = Path(os.environ["DSC_PYTHON"])
else:
    PYTHON = _VENV_PYTHON if _VENV_PYTHON.exists() else Path(sys.executable)

HAVE_RSCRIPT = shutil.which("Rscript") is not None

# A number as the lessons write it: German decimal comma, optional sign,
# optional thousands separator. We deliberately do not match integers with
# fewer than 2 digits – "3 Mechanismen" is not a result.
_NUM = re.compile(r"(?<![\w.,])([+-]?\d{1,3}(?:[  ]\d{3})*(?:[.,]\d+)?)(?![\w])")
# Table rows, including rows nested inside a blockquote (`> | a | b |`), which
# the solutions use to quote expected output.
_TABLE_ROW = re.compile(r"^\s*>?\s*\|.*\|\s*$")
_TEXT_FENCE = re.compile(r"```text\n(.*?)```", re.DOTALL)
# `benchmark.py` takes ~5 minutes. Skipping it would leave its numbers – the
# whole of module 14 §9 – unverified, which is exactly the drift we are hunting.
# So run it, and let the caller opt out explicitly.
SKIP_SLOW = os.environ.get("DSC_SKIP_SLOW") == "1"
_SLOW_SCRIPTS = {"benchmark.py"}

# Numbers that are conventions, not results. A lesson writes "95-%-KI" or
# "p < 0,001" in a table cell; no script needs to print them. Every entry here
# is a number the tool will never question, so each one is a place a stale value
# can hide. "100" used to be on this list, which meant a wrong "100 Patient:innen"
# in any table passed silently; it is gone, and the scripts print it anyway.
_CONVENTIONS = {
    "95", "99",                # confidence levels
    "0,05", "0,01", "0,001",   # alpha / p-value thresholds
    "1,96",                    # normal quantile
    "0,80", "0,8", "0,20",     # power, type-II error
}


def to_float(token: str) -> float | None:
    t = token.replace(" ", "").replace(" ", "").replace(",", ".")
    try:
        return float(t)
    except ValueError:
        return None


def decimals(token: str) -> int:
    t = token.replace(",", ".")
    return len(t.split(".")[1]) if "." in t else 0


def printed_numbers(text: str) -> list[float]:
    out = []
    for m in re.finditer(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text):
        try:
            out.append(float(m.group(0)))
        except ValueError:
            pass
    return out


# A chapter number is a name, not a result. `(Modul 15)` and the part ranges
# `10–21` in a table cell are cross-references; no script prints them, and they
# only ever passed this checker because their old values happened to collide
# with some number a script did print. Renumbering the course exposed that.
_MODUL_REF = re.compile(r"\bModul(?:e|s)?[\s ]*\d{1,2}(?:\s*[–—/-]\s*\d{1,2})*")
_TEIL_RANGE = re.compile(r"\d{1,2}\s*[–—-]\s*\d{1,2}")


def _ohne_modulnummern(line: str) -> str:
    """Strip module cross-references from a table row before reading numbers."""
    line = _MODUL_REF.sub(" ", line)
    if "Teil" in line:                      # "| **Teil 4 – …** | 10–21 | … |"
        line = _TEIL_RANGE.sub(" ", line)
    return line


def asserted_numbers(md: Path) -> list[tuple[int, str]]:
    """Numbers a lesson asserts as results: table cells and ```text blocks."""
    text = md.read_text(encoding="utf-8")
    found: list[tuple[int, str]] = []

    for lineno, line in enumerate(text.split("\n"), start=1):
        if not _TABLE_ROW.match(line) or set(line.strip()) <= set("|-: "):
            continue
        for m in _NUM.finditer(_ohne_modulnummern(line)):
            found.append((lineno, m.group(1)))

    for fence in _TEXT_FENCE.finditer(text):
        base = text[: fence.start()].count("\n") + 2
        for i, line in enumerate(fence.group(1).split("\n")):
            for m in _NUM.finditer(line):
                found.append((base + i, m.group(1)))
    return found


def matches(value: float, token: str, printed: list[float]) -> bool:
    """A printed value matches if it agrees to the precision the doc shows.

    Compare MAGNITUDES: the lessons use the typographic minus (U+2212) and write
    "−28 %" where the script prints "-28", and a table cell often drops the sign
    that the surrounding prose carries. Sign errors are not the bug class this
    tool exists to catch; stale magnitudes are.
    """
    d = decimals(token)
    # A value rounded to d decimals may sit exactly half a unit away, and binary
    # floating point then lands a hair above the tolerance (1.32 - 1.315 is
    # 0.005000000000000115). Without the epsilon the checker flags correctly
    # rounded numbers as drift.
    tol = (0.5 * (10 ** -d) if d else 0.5) + 1e-9
    v = abs(value)
    return any(abs(abs(p) - v) <= tol for p in printed)


def run_module(slug: str) -> tuple[str, list[str]]:
    """stdout of every runnable script in the module, plus the ones that failed.

    R matters: module 20's Fine-Gray subdistribution hazard ratios exist only in
    `r.R` (lifelines has no Fine-Gray), so a Python-only corpus would flag every
    one of them as drift.

    That is the trap this function has to avoid. A script that aborts prints
    nothing, so every number it should have produced looks stale. Silence from a
    broken environment is not evidence of a broken lesson, so failures are
    returned to the caller instead of being folded into the corpus.
    """
    code_dir = ROOT / "module" / slug / "code"
    if not code_dir.is_dir():
        return "", []
    chunks: list[str] = []
    broken: list[str] = []
    # Inherit the real environment. A stripped env silently broke the scripts,
    # and a script that fails prints nothing – so every number it should have
    # produced looked like drift. The checker must not manufacture its own
    # false positives.
    env = dict(os.environ, MPLBACKEND="Agg")

    for script in sorted(code_dir.glob("*.py")):
        if SKIP_SLOW and script.name in _SLOW_SCRIPTS:
            continue
        proc = subprocess.run(
            [str(PYTHON), script.name], cwd=code_dir, capture_output=True,
            text=True, env=env,
        )
        if proc.returncode != 0:
            broken.append(f"{script.name} exited {proc.returncode}")
        chunks.append(proc.stdout)

    r_scripts = sorted(code_dir.glob("*.R"))
    if r_scripts and not HAVE_RSCRIPT:
        broken.append(f"Rscript not on PATH ({len(r_scripts)} R script(s) unrun)")
    else:
        for script in r_scripts:
            proc = subprocess.run(
                ["Rscript", script.name], cwd=code_dir, capture_output=True,
                text=True, env=env,
            )
            if proc.returncode != 0:
                # Almost always a missing R package. The lesson is fine; the
                # environment is not. See DEPENDENCIES-R.md.
                broken.append(f"{script.name} exited {proc.returncode}")
            chunks.append(proc.stdout)
    return "\n".join(chunks), broken


def shared_output() -> str:
    """Course-wide sources of truth that no single module re-derives.

    `lib/ground_truth.py` prints the true odds and hazard ratios; module 12 and
    the figures quote them. `data/generate_data.py` prints the dataset's shape
    and its ground-truth summary; module 02 shows its expected output verbatim
    and `data/README.md` documents it. Both belong in every module's corpus.

    Both scripts rewrite files (`data/*`, `lib/ground_truth.json`). Both are
    deterministic down to the byte – the workbook's timestamps are frozen – so
    running this checker leaves the working tree exactly as it found it. Two
    concurrent runs would still race on those writes; don't do that.
    """
    env = dict(os.environ, MPLBACKEND="Agg")
    chunks = []
    for rel in ("lib/ground_truth.py", "data/generate_data.py"):
        proc = subprocess.run([str(PYTHON), rel], cwd=ROOT,
                              capture_output=True, text=True, env=env)
        chunks.append(proc.stdout)
    return "\n".join(chunks)


def main(argv: list[str]) -> int:
    allow: dict[str, list[str]] = {}
    if ALLOW_FILE.exists():
        raw = json.loads(ALLOW_FILE.read_text(encoding="utf-8"))
        for slug, entry in raw.items():
            if slug.startswith("_"):
                continue
            if not entry.get("reason"):
                print(f"tools/numbers_allow.json: '{slug}' has no reason – refusing it.")
                return 2
            allow[slug] = entry["numbers"]

    wanted = set(argv)
    modules = sorted(p.name for p in (ROOT / "module").glob("*") if p.is_dir())
    if wanted:
        modules = [m for m in modules if m.split("-")[0] in wanted]

    shared = shared_output()

    total_bad = 0
    unverifiable: dict[str, list[str]] = {}
    degraded: list[str] = []
    for slug in modules:
        stdout, broken = run_module(slug)
        if not stdout.strip() and not broken:
            continue
        printed = printed_numbers(stdout + "\n" + shared)
        allowed = set(allow.get(slug, []))

        unmatched: list[str] = []
        for md in sorted((ROOT / "module" / slug).glob("*.md")):
            for lineno, token in asserted_numbers(md):
                if token in allowed or token in _CONVENTIONS:
                    continue
                value = to_float(token)
                if value is None or abs(value) < 1e-9:
                    continue
                if not matches(value, token, printed):
                    unmatched.append(f"{md.relative_to(ROOT)}:{lineno}  {token}")

        # A script that did not run cannot be the source of a number, so its
        # absence explains any leftovers – but only if there ARE leftovers. Most
        # modules assert nothing that only their R script prints, and those stay
        # fully verified even when R is missing. Only module 20 truly needs it.
        if unmatched and broken:
            unverifiable[slug] = broken + [f"unaccounted: {u}" for u in unmatched[:10]]
            continue
        if broken:
            degraded.append(f"{slug} ({'; '.join(broken)})")
        if unmatched:
            total_bad += len(unmatched)
            print(f"\n{slug}: {len(unmatched)} number(s) not produced by the module's code")
            for u in unmatched[:25]:
                print(f"    {u}")
            if len(unmatched) > 25:
                print(f"    … and {len(unmatched) - 25} more")

    print()
    if degraded:
        print(f"{len(degraded)} module(s) had a script that did not run, but every number")
        print("they assert was still produced by the scripts that did:")
        for d in degraded:
            print(f"    {d}")
        print()
    # An unrunnable script is an environment problem. Reporting its lesson's
    # numbers as "drift" would send a contributor hunting for a prose bug that
    # does not exist – so say what actually happened, and use a distinct exit
    # code so CI can tell a stale number from a missing R package.
    if unverifiable:
        print(f"{len(unverifiable)} module(s) could NOT be checked – a script did not run,")
        print("and the numbers it should have printed are exactly the ones left over:")
        for slug, reasons in sorted(unverifiable.items()):
            for reason in reasons:
                print(f"    {slug}: {reason}")
        print("\nProbably an environment problem, not a prose problem: install the missing")
        print("dependencies (requirements.txt, DEPENDENCIES-R.md) and run again. If the")
        print("numbers above still do not appear once the script runs, they are drift.")

    if total_bad:
        print(f"{total_bad} asserted number(s) have no counterpart in the code's output.")
        print("Either fix the prose, or add a justified entry to tools/numbers_allow.json.")
        return 1
    if unverifiable:
        return 3
    print("Every asserted number is reproduced by the module's own code.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
