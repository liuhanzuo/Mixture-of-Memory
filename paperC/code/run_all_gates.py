#!/usr/bin/env python3
"""Run every paperC GATE and report one rc per gate. Exit 1 if any gate fails.

WHY THIS EXISTS -- "9/10 gates pass, 1 is red" was a false statement about the repo.
------------------------------------------------------------------------------------
Every ad-hoc sweep I ran used `for g in paperC/code/gate*.py`, i.e. it selected files by
FILENAME PREFIX. That glob matches `gate2_crossfamily_nulls.py`, which is not a gate: it
is the ANALYSIS EMITTER for the cross-family letter/content extension (task #250). Its own
docstring calls itself "the analysis half"; it takes three required positionals
(`xf_root out_json [out_csv]`) and ends in `json.dump`. Called with no arguments it prints
an argparse usage message and exits 2 -- forever, by construction.

So for several rounds I reported a standing red that was really "I invoked an emitter as
though it were a gate". Its outputs have been on disk the whole time:

    paperC/evidence/second_mc_benchmark_crossfamily/gate2_crossfamily_nulls.json  1176542 B
    paperC/evidence/second_mc_benchmark_crossfamily/gate2_crossfamily_nulls.csv    203150 B

and paperC/README.md line 241 lists them as shipped artifacts on BOTH disks.

The fix is to select on the PROPERTY, not the name: a gate is a checker that runs with no
arguments. Measured 2026-08-17 across all ten `gate*.py` files -- `gate2_crossfamily_nulls`
is the ONLY one with required positionals, and all nine others are no-arg runnable. So the
discriminator is exact here, not a heuristic that happens to fit.

Rather than hardcode a skip-list (a hardcoded list in a checker silently defines what gets
checked -- cf. memory/a-hardcoded-list-in-an-emitter-silently-defines-a-headline.md), this
DERIVES the classification by parsing each file's argparse calls with `ast`. Add a gate and
it is picked up; add an emitter and it is correctly excluded, with its exclusion PRINTED so
the exclusion can never be silent.

Interpreter: paperC's gates need PyMuPDF and numpy, which live in the conda env, not in
.venv. Running them under .venv makes gate_build_record_matches_pdf print FAIL on a
ModuleNotFoundError -- an environment gap that reads as a content regression
(memory/wrong-interpreter-reads-as-a-content-regression.md). This script therefore checks
its own interpreter and says so instead of producing a misleading red.
"""
import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # paperC/
REPO = ROOT.parent
CONDA = "/opt/conda/envs/torch-base/bin/python"


def required_positionals(path):
    """Count argparse positionals with no default -- i.e. arguments the caller MUST pass.

    Parsed with ast rather than grepped: a grep for `add_argument("` would also match a
    positional inside a comment or a string, and would miss `nargs="?"` (which makes a
    positional optional and therefore does NOT disqualify a file from being a gate).
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return None                                     # unparseable: report, do not guess
    n = 0
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        name = node.args[0].value
        if not isinstance(name, str) or name.startswith("-"):
            continue                                    # an option, not a positional
        kw = {k.arg: k.value for k in node.keywords}
        # nargs="?" or "*", or an explicit default, makes it optional
        if "default" in kw:
            continue
        nargs = kw.get("nargs")
        if isinstance(nargs, ast.Constant) and nargs.value in ("?", "*"):
            continue
        n += 1
    return n


def main():
    if not os.path.exists(CONDA):
        print(f"WARNING: {CONDA} not found; falling back to {sys.executable}.")
        print("         paperC gates need PyMuPDF+numpy. Under an env lacking them, a gate")
        print("         prints FAIL on ModuleNotFoundError, which reads as a content bug.")
    py = CONDA if os.path.exists(CONDA) else sys.executable

    env = dict(os.environ)
    tl = REPO / ".texlive" / "2026" / "bin" / "x86_64-linux"
    if tl.is_dir():
        env["PATH"] = f"{tl}:{env.get('PATH','')}"

    gates, emitters, unparseable = [], [], []
    for p in sorted(ROOT.glob("code/gate*.py")) + sorted(ROOT.glob("code/check_*.py")):
        n = required_positionals(p)
        if n is None:
            unparseable.append(p)
        elif n > 0:
            emitters.append((p, n))
        else:
            gates.append(p)

    if emitters:
        print("EXCLUDED (emitters, not gates -- they require positional arguments and")
        print("          would print an argparse usage message and exit 2 with no args):")
        for p, n in emitters:
            print(f"  {p.relative_to(ROOT)}  ({n} required positional(s))")
        print()
    for p in unparseable:
        print(f"UNPARSEABLE, not classified: {p.relative_to(ROOT)}")

    print(f"GATES ({len(gates)}), interpreter={py}")
    failed = []
    for p in gates:
        r = subprocess.run([py, str(p)], cwd=REPO, env=env,
                           capture_output=True, text=True)
        rc = r.returncode                               # captured BEFORE anything else
        print(f"  rc={rc:<3} {p.stem}")
        if rc != 0:
            failed.append((p.stem, rc, (r.stdout + r.stderr).strip().splitlines()[-3:]))

    print()
    if failed:
        print(f"FAIL: {len(failed)}/{len(gates)} gate(s) non-zero")
        for stem, rc, tail in failed:
            print(f"  {stem} rc={rc}")
            for line in tail:
                print(f"      {line}")
        return 1
    print(f"PASS: {len(gates)}/{len(gates)} gates rc=0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
