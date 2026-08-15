#!/usr/bin/env python3
r"""paperC E3: STATIC structural validation of the .tex sources, with no TeX installed.

Why this exists
---------------
The E3 deliverable (`sections/tab_construct_nulls.tex`, emitted by
`emit_tab_construct_nulls.py`) would normally be validated by compiling the
paper. It cannot be: **TeX Live is not present on this machine**, and measured
2026-08-15 it is not present on `.212` either (`command -v pdflatex latexmk
xelatex tectonic` -> nothing; `/usr/local/texlive`, `/opt/texlive`,
`/root/texlive`, `/usr/share/texlive` all absent on both). It *was* present
earlier the same day --- `paperC/main.pdf` has mtime 03:35 and `paperC/main.log`
records `pdfTeX ... (TeX Live 2026)` --- so the toolchain was wiped by a node
restart, the same failure mode already recorded for conda envs, `sshpass` and
`tcodex` in `memory/persist-artifacts-on-wzc1-or-diskb.md`.

So this script substitutes a *structural* check for a compile. It is deliberately
described as what it is: *"no TeX ran"*. It cannot prove the document typesets;
it can only catch the mechanical breakages that a generated table realistically
introduces --- an unbalanced environment, a row whose `&` count disagrees with
the column spec, a missing `\\`, an unbalanced brace or `$`, or an `\input`
pointing at a file that does not exist.

Checks (per file, all reported with file:line)
  C1 environment nesting: every \begin{X} matched by \end{X}, correctly nested
  C2 brace balance, ignoring \{ \} escapes and comment tails
  C3 inline-math `$` parity, ignoring \$ escapes and comment tails
  C4 tabular column arithmetic: for each row between \midrule/\bottomrule and
     the header, the number of cells (accounting for \multicolumn{n}) must equal
     the column count parsed from the tabular's own column spec
  C5 every data row inside a tabular ends with `\\`
  C6 every \input{...} resolves to a file that exists

`--require-clean F` makes the run fail unless file F has zero findings; it is how
the E3 deliverable is gated without asserting anything about the other files,
which carry pre-existing prose this task did not author.

CPU only. No GPU, no model, no network, no TeX.

Usage:
  python paperC/code/validate_tex_static.py
  python paperC/code/validate_tex_static.py --require-clean sections/tab_construct_nulls.tex
  python paperC/code/validate_tex_static.py --out paperC/evidence/tex_static_validation.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
REPO = os.path.dirname(PAPER)
DEFAULT_OUT = os.path.join(PAPER, "evidence", "tex_static_validation.json")

# Environments whose bodies are not LaTeX-tokenised the usual way; their
# contents are skipped by the brace/math checks to avoid false positives.
VERBATIM_ENVS = {"verbatim", "lstlisting", "Verbatim", "minted"}


def strip_comment(line: str) -> str:
    r"""Drop the comment tail, honouring `\%` (and `\\%`, where the `\\` is a row
    break and the `%` therefore still starts a comment)."""
    out = []
    i = 0
    while i < len(line):
        c = line[i]
        if c == "\\" and i + 1 < len(line):
            out.append(line[i:i + 2])   # keep any escaped char verbatim
            i += 2
            continue
        if c == "%":
            break
        out.append(c)
        i += 1
    return "".join(out)


def parse_colspec(spec: str) -> int:
    r"""Number of columns in a tabular column spec.

    Handles the forms this paper uses: letters lrc, `p{..}`/`m{..}`/`b{..}`,
    `@{..}` and `!{..}` (which insert material, not columns), `|`, and
    `*{n}{...}` repetition.
    """
    spec = re.sub(r"[@!]\{(?:[^{}]|\{[^{}]*\})*\}", "", spec)   # drop inserts
    n = 0
    i = 0
    while i < len(spec):
        c = spec[i]
        if c == "*":
            m = re.match(r"\*\{\s*(\d+)\s*\}\{([^{}]*)\}", spec[i:])
            if m:
                n += int(m.group(1)) * parse_colspec(m.group(2))
                i += m.end()
                continue
            i += 1
            continue
        if c in "pmb" and i + 1 < len(spec) and spec[i + 1] == "{":
            depth = 0
            j = i + 1
            while j < len(spec):
                if spec[j] == "{":
                    depth += 1
                elif spec[j] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            n += 1
            i = j + 1
            continue
        if c in "lrcXY":
            n += 1
        i += 1
    return n


def split_cells(row: str) -> int:
    r"""Cell count of a tabular row: unescaped `&` at brace depth 0, plus one,
    with `\multicolumn{n}` contributing n columns instead of 1."""
    depth = 0
    cells = [[]]
    i = 0
    while i < len(row):
        c = row[i]
        if c == "\\" and i + 1 < len(row):
            cells[-1].append(row[i:i + 2])
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        if c == "&" and depth == 0:
            cells.append([])
            i += 1
            continue
        cells[-1].append(c)
        i += 1
    total = 0
    for cell in cells:
        text = "".join(cell)
        m = re.search(r"\\multicolumn\s*\{\s*(\d+)\s*\}", text)
        total += int(m.group(1)) if m else 1
    return total


BEGIN = re.compile(r"\\begin\s*\{([^}]*)\}")
END = re.compile(r"\\end\s*\{([^}]*)\}")
INPUT = re.compile(r"\\(?:input|include)\s*\{([^}]*)\}")


def check_file(path: str) -> list[dict]:
    rel = os.path.relpath(path, REPO)
    findings: list[dict] = []

    def add(line, check, msg):
        findings.append({"file": rel, "line": line, "check": check,
                         "message": msg})

    with open(path, encoding="utf-8") as f:
        raw = f.read().split("\n")

    env_stack: list[tuple[str, int]] = []
    # tabular state: (ncols, colspec, start_line, name)
    tab_stack: list[dict] = []
    brace = 0
    verbatim_depth = 0
    pending: list[str] = []      # a row may be split across physical lines
    pending_start = 0

    for lineno, raw_line in enumerate(raw, 1):
        line = strip_comment(raw_line)

        for m in BEGIN.finditer(line):
            name = m.group(1)
            env_stack.append((name, lineno))
            if name in VERBATIM_ENVS:
                verbatim_depth += 1
            if name in ("tabular", "tabularx", "tabulary", "array"):
                rest = line[m.end():]
                if name in ("tabularx", "tabulary"):   # skip the width argument
                    w = re.match(r"\s*\{(?:[^{}]|\{[^{}]*\})*\}", rest)
                    if w:
                        rest = rest[w.end():]
                sm = re.match(r"\s*(?:\[[^\]]*\])?\s*\{((?:[^{}]|\{[^{}]*\})*)\}",
                              rest)
                if sm:
                    tab_stack.append({"ncols": parse_colspec(sm.group(1)),
                                      "spec": sm.group(1), "line": lineno,
                                      "name": name})
                else:
                    add(lineno, "C4", f"\\begin{{{name}}}: could not parse a "
                                      f"column spec on this line")
                    tab_stack.append({"ncols": None, "spec": None,
                                      "line": lineno, "name": name})

        for m in END.finditer(line):
            name = m.group(1)
            if not env_stack:
                add(lineno, "C1", f"\\end{{{name}}} with no open environment")
            else:
                open_name, open_line = env_stack[-1]
                if open_name != name:
                    add(lineno, "C1", f"\\end{{{name}}} closes "
                                      f"\\begin{{{open_name}}} from line "
                                      f"{open_line} (mis-nested)")
                env_stack.pop()
            if name in VERBATIM_ENVS:
                verbatim_depth = max(0, verbatim_depth - 1)
            if name in ("tabular", "tabularx", "tabulary", "array"):
                if pending:
                    add(pending_start, "C5",
                        "row before \\end{%s} does not end with \\\\ : %r"
                        % (name, " ".join(pending)[:90]))
                    pending = []
                if tab_stack:
                    tab_stack.pop()

        if verbatim_depth:
            continue

        # C2 / C3 accumulate over the file
        body = re.sub(r"\\[{}$]", "", line)
        brace += body.count("{") - body.count("}")

        # C4 / C5 -- only inside a tabular, and only for material that is a row
        if tab_stack and tab_stack[-1]["ncols"]:
            content = line.strip()
            skel = re.sub(r"\\(?:toprule|midrule|bottomrule|cmidrule"
                          r"|hline|addlinespace|rowcolor|specialrule)"
                          r"(?:\s*(?:\[[^\]]*\]|\{[^{}]*\}))*", "", content)
            skel = skel.strip()
            if skel and not BEGIN.search(skel) and not END.search(skel):
                if not pending:
                    pending_start = lineno
                pending.append(skel)
                joined = " ".join(pending)
                if joined.rstrip().endswith("\\\\"):
                    row = joined.rstrip()[:-2]
                    n = split_cells(row)
                    want = tab_stack[-1]["ncols"]
                    if n != want:
                        add(pending_start, "C4",
                            "row has %d cells but the column spec %r declares "
                            "%d: %r" % (n, tab_stack[-1]["spec"], want,
                                        row.strip()[:90]))
                    pending = []

        for m in INPUT.finditer(line):
            target = m.group(1).strip()
            cands = [os.path.join(PAPER, target),
                     os.path.join(PAPER, target + ".tex"),
                     os.path.join(os.path.dirname(path), target),
                     os.path.join(os.path.dirname(path), target + ".tex")]
            if not any(os.path.isfile(c) for c in cands):
                add(lineno, "C6", f"\\input{{{target}}} resolves to no file "
                                  f"(tried {len(cands)} candidate paths)")

    for name, line in env_stack:
        add(line, "C1", f"\\begin{{{name}}} is never closed")
    if brace:
        add(0, "C2", "brace imbalance over the whole file: %+d "
                     "(unmatched '{' if positive)" % brace)

    # C3 file-level `$` parity, computed outside verbatim bodies
    dollars = 0
    vd = 0
    for raw_line in raw:
        line = strip_comment(raw_line)
        for m in BEGIN.finditer(line):
            if m.group(1) in VERBATIM_ENVS:
                vd += 1
        if not vd:
            body = re.sub(r"\\\$", "", line)
            body = re.sub(r"\$\$", "", body)
            dollars += body.count("$")
        for m in END.finditer(line):
            if m.group(1) in VERBATIM_ENVS:
                vd = max(0, vd - 1)
    if dollars % 2:
        add(0, "C3", "odd number of unescaped inline-math '$' in the file (%d)"
                     % dollars)
    return findings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--require-clean", action="append", default=[],
                    help="path relative to paperC/ that must have 0 findings")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    files = [os.path.join(PAPER, "main.tex")]
    files += sorted(glob.glob(os.path.join(PAPER, "sections", "*.tex")))
    files = [p for p in files if os.path.isfile(p)]

    per_file = {}
    all_find: list[dict] = []
    for p in files:
        f = check_file(p)
        per_file[os.path.relpath(p, PAPER)] = len(f)
        all_find += f

    required = {}
    for req in args.require_clean:
        hits = [f for f in all_find
                if f["file"] == os.path.relpath(os.path.join(PAPER, req), REPO)]
        required[req] = {"findings": len(hits), "clean": not hits}

    payload = {
        "what": "static structural validation of paperC .tex sources",
        "IMPORTANT_no_tex_ran": (
            "NO LaTeX COMPILE WAS PERFORMED. TeX Live is absent from this "
            "machine and, measured 2026-08-15, from .212 as well: "
            "`command -v pdflatex latexmk xelatex tectonic` returns nothing and "
            "/usr/local/texlive, /opt/texlive, /root/texlive, /usr/share/texlive "
            "are all absent on both. paperC/main.pdf (mtime 2026-08-15 03:35) "
            "and main.log ('pdfTeX ... TeX Live 2026') show the toolchain "
            "existed earlier the same day and was removed by a node restart. "
            "These checks are structural only and CANNOT establish that the "
            "document typesets."),
        "checks": {
            "C1": "every \\begin{X} matched by \\end{X}, correctly nested",
            "C2": "brace balance (escaped \\{ \\} and comment tails ignored)",
            "C3": "inline-math $ parity (escaped \\$ and $$ ignored)",
            "C4": "tabular row cell count == column count from the tabular's "
                  "own column spec, with \\multicolumn{n} counted as n",
            "C5": "every data row inside a tabular ends with \\\\",
            "C6": "every \\input{...} resolves to an existing file",
        },
        "verbatim_envs_skipped": sorted(VERBATIM_ENVS),
        "gpu_used": "NONE",
        "n_files": len(files),
        "n_findings": len(all_find),
        "findings_per_file": per_file,
        "findings": all_find,
        "require_clean": required,
        "verdict": ("PASS" if not all_find else
                    ("PASS_REQUIRED_ONLY"
                     if all(v["clean"] for v in required.values()) and required
                     else "FAIL")),
    }

    if not args.no_write:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1, sort_keys=True)
            f.write("\n")

    print("[static] NO TeX COMPILE -- structural checks only "
          "(no pdflatex/latexmk/xelatex/tectonic on LOCAL or .212)")
    print(f"[static] {len(files)} files, {len(all_find)} findings")
    for rel, n in sorted(per_file.items()):
        print(f"   {'OK  ' if n == 0 else 'FIND'} {rel}: {n}")
    for f in all_find:
        print(f"  [{f['check']}] {f['file']}:{f['line']} {f['message']}")
    for req, st in required.items():
        print(f"[require-clean] {req}: "
              f"{'CLEAN' if st['clean'] else 'NOT CLEAN'} "
              f"({st['findings']} findings)")
    if not args.no_write:
        print(f"[done] wrote {args.out}")

    if required and not all(v["clean"] for v in required.values()):
        return 1
    if not required and all_find:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
