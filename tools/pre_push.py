"""Local gate before `git push`. A push to main deploys the site at once.

Runs from the git hook in tools/hooks/pre-push (activate once per clone with
`git config core.hooksPath tools/hooks`), or by hand:

    ./.venv/bin/python tools/pre_push.py            # checks the commits not yet on origin/main

Three steps, cheapest first, stopping at the first failure:

  1. tools/check_repo.py     structure, links, placeholders (~10 s)
  2. website/build.py        the same build pages.yml deploys
  3. tools/check_numbers.py  only for the modules the pushed commits touch;
                             all modules when data/, lib/ or the checker
                             itself changed. The ~5-minute Modul-14 benchmark
                             is skipped (DSC_SKIP_SLOW=1); run it by hand when
                             module 14's benchmark table changes.

The number check runs locally because it needs R and several minutes, which a
CI runner would have to install on every push.

Exit code 0 = push may proceed. `git push --no-verify` skips the hook once.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NO_COMMIT = "0" * 40
# Changes here can move numbers in every module, so all modules get checked.
SHARED_PREFIXES = ("data/", "lib/", "tools/check_numbers.py", "tools/numbers_allow.json",
                   "requirements.txt", "requirements-ml.txt")
MODULE_FILE = re.compile(r"^module/(\d\d)-")
ENVIRONMENT_PROBLEM = 3       # check_numbers.py: a script could not run at all


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout


def changed_files(push_lines: list[str]) -> set[str] | None:
    """Files touched by the commits being pushed; None = unknown, check all.

    Git feeds the hook one line per ref: `<local ref> <local sha> <remote ref>
    <remote sha>`. Without that input (manual run) compare with origin/main.
    """
    ranges = []
    for line in push_lines:
        parts = line.split()
        if len(parts) != 4 or parts[1] == NO_COMMIT:     # malformed, or a deletion
            continue
        local_sha, remote_sha = parts[1], parts[3]
        ranges.append((remote_sha, local_sha) if remote_sha != NO_COMMIT
                      else ("origin/main", local_sha))
    if not push_lines:
        ranges.append(("origin/main", "HEAD"))
    files: set[str] = set()
    for base, tip in ranges:
        try:
            files |= set(git("diff", "--name-only", f"{base}...{tip}").split())
        except subprocess.CalledProcessError:
            return None                                  # base unknown here
    return files


def modules_to_check(files: set[str] | None) -> list[str] | None:
    """Module numbers for check_numbers.py; [] = nothing, None = all modules."""
    if files is None or any(f.startswith(SHARED_PREFIXES) for f in files):
        return None
    return sorted({m.group(1) for f in files if (m := MODULE_FILE.match(f))})


def run(step: str, args: list[str], env: dict[str, str] | None = None) -> int:
    print(f"\n== {step} ==", flush=True)
    return subprocess.run([sys.executable, *args], cwd=ROOT, env=env).returncode


def main() -> int:
    push_lines = [] if sys.stdin.isatty() else sys.stdin.read().splitlines()
    if run("Repo-Konsistenz", ["tools/check_repo.py"]) != 0:
        return 1
    if run("Website-Build", ["website/build.py"]) != 0:
        return 1

    modules = modules_to_check(changed_files(push_lines))
    if modules == []:
        print("\n== Zahlenabgleich: kein Modul geändert, übersprungen ==")
        return 0
    scope = "alle Module" if modules is None else "Module " + ", ".join(modules)
    env = dict(os.environ, DSC_SKIP_SLOW="1")
    code = run(f"Zahlenabgleich ({scope}, ohne Benchmark)",
               ["tools/check_numbers.py", *(modules or [])], env=env)
    if code == ENVIRONMENT_PROBLEM:
        print("\nWarnung: Ein Skript lief nicht (übersprungener Benchmark, fehlendes R oder\n"
              "Paket). Die oben gelisteten Zahlen sind ungeprüft; der Push wird nicht blockiert.\n"
              "Voller Lauf: ./.venv/bin/python tools/check_numbers.py")
        return 0
    return code


if __name__ == "__main__":
    sys.exit(main())
