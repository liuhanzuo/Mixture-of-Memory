#!/usr/bin/env python3
"""check_numbers.py — every number in the manuscript must trace to evidence.

Why this exists
---------------
`paperC/SUBMISSION_GAP_AUDIT.md:100` (defect E3): *"No `.tex`-level integrity
binding between evidence and prose. paperB solved this with
`app_tab_integrity.tex`. paperC's numbers currently live in prose in a 478-line
README with 30+ inline retractions."* And E2: the headline table is hand-assembled
from four sources plus prose.

A hand-typed number in a paper is the highest-leverage lie an honest author can
tell by accident. This script mechanically extracts every numeric literal from
the manuscript and asks whether it appears in a declared evidence source. It
does NOT decide whether the number is *right* -- it decides whether it is
*traceable*, which is the property the skill's integrity gate actually claims.

Design decisions that matter
----------------------------
1. **Whitelist, not blacklist.** Section numbers, years, `\\cite` years, font
   sizes, and LaTeX lengths are not claims. They are filtered by context, and
   every filter is named in the output so a reader can audit the filtering
   itself rather than trusting it.

2. **Rounding is legitimate; invention is not.** `0.3635` in prose may come from
   `0.363500000000000...` in JSON. So a manuscript number matches if it is a
   correct rounding of an evidence number at the manuscript's own precision.
   This is the mechanism that would have caught the real paperA bug recorded in
   `paperA/audit_20260806/primitive_numbers_disk_provenance.md`: disk truth
   `99.187` was written as `99.20` in `tab_pareto.tex` when correct rounding is
   `99.19`.

3. **A percentage point is not a ratio.** `pp` deltas are derived, so a delta is
   matched if it equals the difference of two evidence numbers at the stated
   precision. Reported separately from direct hits so "derived" never silently
   passes as "primary".

4. **Unmatched is a finding, not an error.** Some numbers are legitimately
   derived in ways this script cannot model. They are reported as
   `unmatched_needs_human` rather than failing the run, because a checker that
   cries wolf gets muted, and a muted checker is worse than none. The gate
   verdict is on the COUNT, which the caller must pre-register.

Usage:
  python check_numbers.py paperC --evidence paperC/tcodex_out/EVIDENCE_PACK.md \
      --evidence paperC/evidence --max-unmatched 40
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]

# Numbers in these contexts are typography/structure, not claims.
SKIP_LINE_PAT = re.compile(
    r"\\(?:usepackage|documentclass|vspace|hspace|setlength|addtolength|"
    r"includegraphics|label|ref|eqref|pageref|cite[a-zA-Z]*|bibliography|"
    r"input|include|newcommand|renewcommand|def|fontsize|selectfont|"
    r"columnwidth|textwidth|linewidth|arraystretch|tabcolsep|scalebox|"
    r"multirow|multicolumn|cmidrule|midrule|toprule|bottomrule)"
)

# The macros whose ARGUMENTS carry typography/structure rather than claims, plus the
# bare length units. Used by strip_macro_args() to blank out just those spans so the
# rest of the line still gets audited.
#
# WHY THIS EXISTS: SKIP_LINE_PAT was originally applied to whole lines. That made the
# gate silently audit LESS whenever a \ref or \cite shared a physical line with a real
# number, while still reporting "pass". A gate that gets quieter as you cross-reference
# your tables is worse than no gate, because the pass is what you act on.
MACRO_ARG_PAT = re.compile(
    r"\\(?:usepackage|documentclass|vspace|hspace|setlength|addtolength|includegraphics|"
    r"label|ref|eqref|pageref|autoref|Cref|cref|cite[a-zA-Z]*|bibliography|input|include|"
    r"newcommand|renewcommand|def|fontsize|selectfont|arraystretch|tabcolsep|scalebox|"
    r"multirow|multicolumn|cmidrule)\s*(?:\[[^\]]*\])*(?:\{[^{}]*\})*"
)
# `0.62\linewidth`, `3.5pt`, `1.2ex`, `\p{0.05\linewidth}` -- lengths, not measurements.
LENGTH_PAT = re.compile(
    r"-?\d*\.?\d+\s*(?:\\(?:linewidth|columnwidth|textwidth|textheight|baselineskip)"
    r"|pt|pc|in|bp|cm|mm|dd|cc|sp|ex|em)\b"
)


def strip_macro_args(line: str) -> str:
    """Blank the arguments of structural/typographic macros and bare LaTeX lengths,
    leaving the surrounding prose (and its numeric claims) intact for auditing."""
    line = MACRO_ARG_PAT.sub(" ", line)
    line = LENGTH_PAT.sub(" ", line)
    return line


NUM_PAT = re.compile(r"(?<![\w.])(-?\d+(?:\.\d+)?)(?![\w])")

# LaTeX en-dash/em-dash ranges make a tokenizer see a NEGATIVE number.
# Measured on the paperC draft 2026-08-14: all 3 "untraceable" numbers in the
# first run were this artifact -- `k14--k8` yielded -8, `0.111868--0.114694`
# yielded -0.114694, and `9.8e-148` yielded -148. A checker with known false
# positives teaches its reader to ignore it, which is worse than no checker, so
# these are normalised away BEFORE tokenising rather than explained afterwards.
DASH_RANGE = re.compile(r"(?<=[\d\w])-{2,}(?=[\d\w.])")
SCI_EXP = re.compile(r"(?<=[\d.])[eE][-+]?\d+")
# LaTeX-math scientific notation: `9.8\times10^{-148}` must not yield -148.
# Measured on paperC sections/05_analysis.tex:17 -- the exponent of a McNemar
# p-value was the single remaining "untraceable" number after the dash fix.
TEX_SCI = re.compile(r"\\(?:times|cdot)\s*10\s*\^\s*\{?\s*[-+]?\d+\s*\}?")


def normalise_for_tokens(line: str) -> str:
    """Make ranges and exponents unable to fake numbers that were never claimed."""
    line = DASH_RANGE.sub(" to ", line)     # 0.11--0.12          -> 0.11 to 0.12
    line = TEX_SCI.sub("", line)            # 9.8\times10^{-148}  -> 9.8
    line = SCI_EXP.sub("", line)            # 9.8e-148            -> 9.8
    return line


def strip_comments(text: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", text)


def harvest_evidence(paths: list[Path]) -> tuple[set[str], dict[str, list[str]]]:
    """Collect every numeric literal appearing in any evidence source."""
    nums: set[str] = set()
    where: dict[str, list[str]] = {}
    files = []
    for p in paths:
        if p.is_dir():
            for ext in ("*.json", "*.csv", "*.tsv", "*.md", "*.txt"):
                files.extend(p.rglob(ext))
        elif p.is_file():
            files.append(p)
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in NUM_PAT.finditer(text):
            tok = m.group(1)
            nums.add(tok)
            where.setdefault(tok, [])
            if len(where[tok]) < 3:
                where[tok].append(f.name)
    return nums, where


def rounds_to(evidence_tok: str, manuscript_tok: str) -> bool:
    """True if manuscript_tok is a correct rounding of evidence_tok."""
    try:
        ev = Decimal(evidence_tok)
        ms = Decimal(manuscript_tok)
    except InvalidOperation:
        return False
    if "." in manuscript_tok:
        places = len(manuscript_tok.split(".")[1])
        q = Decimal(1).scaleb(-places)
        return ev.quantize(q, rounding="ROUND_HALF_UP") == ms or \
               ev.quantize(q, rounding="ROUND_HALF_EVEN") == ms
    return ev.quantize(Decimal(1), rounding="ROUND_HALF_UP") == ms


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paper")
    ap.add_argument("--evidence", action="append", required=True,
                    help="evidence file or dir; repeat for several")
    ap.add_argument("--max-unmatched", type=int, default=None,
                    help="gate: fail if more than this many numbers are untraceable")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    pd = Path(a.paper) if Path(a.paper).is_absolute() else REPO / a.paper
    if not pd.is_dir():
        print(f"error: not a dir: {pd}", file=sys.stderr)
        return 2

    ev_paths = [Path(e) if Path(e).is_absolute() else REPO / e for e in a.evidence]
    for e in ev_paths:
        if not e.exists():
            print(f"error: evidence path missing: {e}", file=sys.stderr)
            return 2
    ev_nums, ev_where = harvest_evidence(ev_paths)

    direct, rounded, derived, unmatched = [], [], [], []
    ev_sorted = sorted(ev_nums, key=len, reverse=True)
    ev_dec = []
    for t in ev_nums:
        try:
            ev_dec.append((t, Decimal(t)))
        except InvalidOperation:
            pass

    n_scanned = 0
    for tex in sorted(pd.rglob("*.tex")):
        if "review_rounds" in tex.parts or "review_history" in tex.parts:
            continue
        text = strip_comments(tex.read_text(encoding="utf-8", errors="replace"))
        for lineno, line in enumerate(text.splitlines(), 1):
            # A whole-line skip used to be applied here whenever the line matched
            # SKIP_LINE_PAT. That silently removed REAL numeric claims from the audit
            # as soon as a \ref or \cite landed on the same physical line as a number
            # -- and the gate still reported "pass" while checking less. Measured on
            # paperC at the time of the fix: 87 numerals on 35 lines were being hidden,
            # including load-bearing ones (BoolQ 0.6217 vs 0.50; 1403/12032; 623/12032;
            # 1439/12032; the 0.9003/10.6/9.26 tokenizer deltas).
            #
            # What the pattern was actually protecting against is the ARGUMENTS of
            # typographic/structural macros (0.62\linewidth, tabcolsep 3.5pt, cite keys,
            # label names) -- not the surrounding prose. So strip those arguments and
            # keep auditing the rest of the line.
            line = strip_macro_args(line)
            if not line.strip():
                continue
            for m in NUM_PAT.finditer(normalise_for_tokens(line)):
                tok = m.group(1)
                # years and tiny integers are not claims
                if re.fullmatch(r"(19|20|21|26)\d\d", tok):
                    continue
                if "." not in tok and abs(int(tok)) <= 12:
                    continue
                n_scanned += 1
                loc = {"file": str(tex.relative_to(pd)), "line": lineno,
                       "value": tok, "context": line.strip()[:150]}
                if tok in ev_nums:
                    loc["evidence_files"] = ev_where.get(tok, [])[:3]
                    direct.append(loc)
                    continue
                hit = next((t for t, d in ev_dec if rounds_to(t, tok)), None)
                if hit:
                    loc["evidence_full_precision"] = hit
                    loc["evidence_files"] = ev_where.get(hit, [])[:3]
                    rounded.append(loc)
                    continue
                # derived: difference of two evidence numbers (pp deltas)
                found = None
                if "." in tok:
                    places = len(tok.split(".")[1])
                    q = Decimal(1).scaleb(-places)
                    target = Decimal(tok)
                    for t1, d1 in ev_dec[:1500]:
                        for t2, d2 in ev_dec[:1500]:
                            if (d1 - d2).quantize(q) == target:
                                found = f"{t1} - {t2}"
                                break
                        if found:
                            break
                if found:
                    loc["derivation"] = found
                    derived.append(loc)
                else:
                    unmatched.append(loc)

    rec = {
        "schema_version": "1.0.0",
        "paper": str(pd.relative_to(REPO)) if str(pd).startswith(str(REPO)) else str(pd),
        "evidence_sources": [str(e) for e in ev_paths],
        "n_evidence_numbers": len(ev_nums),
        "filters_applied": [
            "lines containing LaTeX structural/typographic macros (see SKIP_LINE_PAT)",
            "4-digit years 19xx/20xx/21xx/26xx",
            "bare integers with |n| <= 12 (section/table/column indices)",
        ],
        "counts": {
            "scanned": n_scanned,
            "direct_match": len(direct),
            "match_after_correct_rounding": len(rounded),
            "derived_as_difference": len(derived),
            "unmatched_needs_human": len(unmatched),
        },
        "unmatched_needs_human": unmatched,
        "match_after_correct_rounding": rounded[:80],
        "derived_as_difference": derived[:80],
    }
    if a.max_unmatched is not None:
        rec["gate_max_unmatched"] = a.max_unmatched
        rec["numbers_gate_pass"] = len(unmatched) <= a.max_unmatched

    # see paper_build.py: `build/` is gitignored, gate artifacts are evidence
    out = Path(a.out) if a.out else pd / "gate" / "numbers_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    c = rec["counts"]
    print(json.dumps({"counts": c,
                      "numbers_gate_pass": rec.get("numbers_gate_pass")},
                     indent=2))
    print(f"[check_numbers] {c['scanned']} numeric claims scanned; "
          f"{c['direct_match']} direct, {c['match_after_correct_rounding']} rounded, "
          f"{c['derived_as_difference']} derived, "
          f"{c['unmatched_needs_human']} need a human", file=sys.stderr)
    print(f"[check_numbers] wrote {out}", file=sys.stderr)
    if a.max_unmatched is not None and not rec["numbers_gate_pass"]:
        print(f"[check_numbers] NUMBERS GATE: FAIL "
              f"({c['unmatched_needs_human']} > {a.max_unmatched})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
