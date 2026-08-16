#!/usr/bin/env python3
"""Guard: integer COUNT claims in the prose must match the evidence that carries them.

WHY THIS EXISTS
---------------
On 2026-08-16 three prose-vs-evidence defects were found in this manuscript by hand:

  * the abstract and introduction still quoted the retracted denominator 14/15 while
    sections 5 and 6 already used 15/17;
  * `03b_nulls.tex` printed the credit/wrong ratio as 4.6x when 0.532164/0.125914 = 4.2264;
  * `tab_integrity` claimed "all nine flips on affected items" when the evidence JSON
    totals eight, counted three independent ways.

All three passed `check_prose_vs_evidence.py`, every time, with
`n_checked=91 n_ok=91 n_mismatch=0 n_uncovered=0` and `verdict: PASS`. That checker is
not broken. Its own schema says it checks "prose statements of each construct null"
against full-precision floor/chance values, with `min_decimals: 3`. Integer counts and
one-decimal ratios are structurally outside its target space, and its `n_uncovered=0`
means "the targets I enumerated are covered", not "every number in the prose is covered".

So the uncovered class is exactly: COUNTS. Denominators, flip totals, cell counts. They
are integers, they live in more than one file, and nothing was comparing them.

WHAT THIS GUARD CHECKS
----------------------
Each entry in CLAIMS below names a prose string, the evidence file that decides it, and a
callable that recomputes the number from that file. The gate fails if the prose asserts a
count the evidence does not support. Every recomputation is an integer operation on
on-disk records -- no re-derivation, no simulation, no tolerance.

Adding a claim here is cheap and is the intended way to grow coverage: one line per
count that appears in the manuscript.

Exit codes: 0 = every registered count matches. 2 = at least one mismatch.
3 = an evidence file or prose file could not be read.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SECTIONS = ROOT / "sections"
EVIDENCE = ROOT / "evidence"


def load(rel):
    p = EVIDENCE / rel
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- recomputations
def n_argmax_flips(doc):
    """Total argmax flips over cells that HAVE a before/after pair.

    llama2_7b_base has no 'before' -- it OOM'd on 5/8 shards and the merge guard
    refused a partial merge -- so it carries no delta block and cannot contribute a
    flip. The denominator is therefore 14 cells, not 15.
    """
    total = 0
    for cell in doc["cells"].values():
        delta = cell.get("delta")
        if not delta or "n_argmax_flips_total" not in delta:
            continue
        total += delta["n_argmax_flips_total"]
    return total


def n_flips_outside_affected(doc):
    return sum(c["delta"]["n_argmax_flips_outside_affected_items"]
               for c in doc["cells"].values()
               if c.get("delta", {}).get("n_argmax_flips_outside_affected_items") is not None)


def n_cells_with_pair(doc):
    """The denominator of the verdict-change ratio, as the emitter itself recorded it.

    DO NOT reimplement this as `sum('before' in c and 'after' in c)`. All 15 cells carry
    both keys -- llama2_7b_base has an `after` and a placeholder `before` -- so that
    predicate returns 15 and contradicts the paper's correct 14. The field that actually
    distinguishes the OOM'd cell is `integrity_before`, which it alone lacks. Rather than
    encode that subtlety a second time, read the number the producing script wrote:
    `summary.n_cells_with_before_and_after`.

    The first version of this function did reimplement the predicate, returned 15, and
    inverted both negative controls -- the correct prose failed and a wrong denominator
    passed. That is the signature of a gate whose recomputation disagrees with the
    evidence it is checking.
    """
    return doc["summary"]["n_cells_with_before_and_after"]


def n_verdicts_changed(doc):
    return sum(1 for c in doc["cells"].values()
               if c.get("delta", {}).get("verdict_changed") is True)


TRUNC = "mmlu_scale_power/trunc_fix_before_after.json"

CLAIMS = [
    # (label, section file, regex whose group(1) is the asserted count,
    #  evidence file, recomputation, human note)
    ("argmax flips, total", "tab_integrity.tex",
     r"all (\w+) argmax flips on affected items",
     TRUNC, n_argmax_flips,
     "the word is spelled out in the table, so the regex captures a word"),
    ("argmax flips outside affected items", "tab_integrity.tex",
     r"all \w+ argmax flips on affected items, (\d+) elsewhere",
     TRUNC, n_flips_outside_affected,
     "0 elsewhere is the load-bearing half: a flip outside the affected set would "
     "indicate nondeterminism rather than the truncation fix"),
    # ANCHORING MATTERS. tab_integrity has TWO rows that each say "N/M verdicts
    # changed": the bootstrap mid-p row (0/24) and the truncation row (0/14). An
    # unanchored r"(\d+)/(\d+) verdicts changed" matches the bootstrap row first and
    # compares it against truncation evidence -- which is exactly what the first
    # version of this gate did, reporting a 24-vs-15 mismatch that was a bug in the
    # gate, not in the paper. Each claim below therefore anchors on text unique to
    # its own row.
    ("verdicts changed (numerator)", "tab_integrity.tex",
     r"hard per-shard assertion & (\d+)/\d+ verdicts changed",
     TRUNC, n_verdicts_changed,
     "the numerator of the truncation row's verdict-change ratio"),
    ("verdicts changed (denominator)", "tab_integrity.tex",
     r"hard per-shard assertion & \d+/(\d+) verdicts changed",
     TRUNC, n_cells_with_pair,
     "the denominator must equal the number of cells that HAVE a before/after pair, "
     "which is 14 rather than 15 because llama2_7b_base OOM'd and has no 'before'"),
]

WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
         "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
         "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
         "fifteen": 15, "sixteen": 16, "seventeen": 17}


def as_int(token):
    token = token.strip().lower()
    if token.isdigit():
        return int(token)
    return WORDS.get(token)


def main():
    failures = []
    rows = []
    for label, sec, pattern, ev_rel, recompute, note in CLAIMS:
        sec_path = SECTIONS / sec
        if not sec_path.exists():
            print(f"CANNOT READ: {sec_path}")
            return 3
        text = sec_path.read_text(encoding="utf-8")
        m = re.search(pattern, text)
        try:
            doc = load(ev_rel)
        except FileNotFoundError as exc:
            print(f"CANNOT READ evidence: {exc}")
            return 3
        expected = recompute(doc)
        if not m:
            failures.append(f"{label}: pattern {pattern!r} found no match in {sec}. "
                            f"Either the sentence was reworded (update this gate) or the "
                            f"claim was dropped. Evidence says {expected}.")
            rows.append((label, "NOT FOUND", expected, "FAIL"))
            continue
        claimed = as_int(m.group(1))
        if claimed is None:
            failures.append(f"{label}: could not parse {m.group(1)!r} as an integer.")
            rows.append((label, m.group(1), expected, "FAIL"))
            continue
        ok = claimed == expected
        if not ok:
            failures.append(f"{label}: prose in {sec} says {claimed}, evidence "
                            f"{ev_rel} says {expected}. {note}")
        rows.append((label, claimed, expected, "ok" if ok else "FAIL"))

    print(f"{'count claim':40}{'prose':>10}{'evidence':>10}{'':>7}")
    for label, claimed, expected, verdict in rows:
        print(f"{label[:40]:40}{str(claimed):>10}{expected:>10}{verdict:>7}")
    print()

    if failures:
        print("FAIL:")
        for f in failures:
            print(f"  {f}")
        return 2

    print(f"PASS: all {len(rows)} registered count claims match their evidence.")
    print("      Coverage is the CLAIMS list above, nothing wider. Counts not listed")
    print("      here are unchecked -- add a line rather than assuming coverage.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
