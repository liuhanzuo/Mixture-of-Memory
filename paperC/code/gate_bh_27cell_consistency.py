#!/usr/bin/env python3
"""Guard: prose about multiplicity over the 27-cell v2 family must match the table.

WHY THIS EXISTS
---------------
Round_04's stats reviewer and the independent meta-reviewer both applied Benjamini--Hochberg
to the 27 p-values printed in `tab_v2_full.tex` and both concluded that `olmo2/keep14`
(p=0.0172) does not survive. They were right about that cell, and the paper's §5.3
bidirectionality claim rests on it alone, since it is the only cell moving upward.

But both reviewers ALSO wrote that *neither* trace-signal cell survives correction, and that
is false for BH. `qwen3/k14` at p=0.0066 sits at rank 6, where the BH threshold is
6*0.05/27 = 0.011111, so it rejects; its q-value is 0.029700. Only under Bonferroni
(alpha/27 = 0.001852) do both trace cells fail. Each reviewer's own rank-by-rank enumeration
says rank 6 rejects, contradicting its own summary sentence. Copying either summary into the
paper would have printed a false statement -- one that understates the paper's own surviving
evidence, but false either way.

So the fact needing a guard is not "BH kills the trace cells". It is the exact split:

    BH q=0.05 over m=27 rejects 6: the five anchors (p=0.0001) plus qwen3/k14 (q=0.0297).
    olmo2/keep14 is NOT among them (q=0.0663).
    Bonferroni rejects only the 5 anchors.

WHAT THIS GUARD DOES
--------------------
It recomputes tie-aware BH q-values from `tab_v2_full.tex` -- the table is the only input --
and asserts three things a later edit could silently break:

  1. the BH rejection count is 6 and the Bonferroni count is 5;
  2. `qwen3/k14` survives BH and `olmo2/keep14` does not (the asymmetry that decides which
     half of the bidirectionality claim is defensible);
  3. no prose file asserts that BOTH trace cells fail correction without naming Bonferroni.

Check 3 is the one that catches the reviewers' error. A sentence may say both cells fail
under Bonferroni; it may not say both fail under BH, or say it unqualified.

SCOPE, HONESTLY STATED
----------------------
This does not argue that BH is the correct correction for this dependence structure. The 27
cells share the MMLU-Pro item axis and nest arms within families, so positive regression
dependence is plausible but unverified. The guard enforces internal consistency between what
the table implies under the paper's own stated correction and what the prose claims about it.

Exit codes: 0 = all checks pass. 2 = a numeric or prose check failed. 3 = could not parse.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TABLE = ROOT / "sections" / "tab_v2_full.tex"
PROSE = ROOT / "sections"

EXPECT_M = 27
EXPECT_BH = 6
EXPECT_BONF = 5
ALPHA = 0.05
SURVIVES = "k14, unhealed"      # qwen3 row: expected to survive BH
FAILS = "keep14@200000"         # olmo2 row: expected to fail BH

# A sentence is exonerated when it states the asymmetry rather than a both/neither verdict:
# "one of the two", "exactly one", "one survives ... the other does not", "only qwen3/k14".
EXONERATED = re.compile(
    r"one of the two|exactly one|only one|\bone\b[^.]{0,60}\bthe other\b|only\s+\\?texttt\{?k14",
    re.I)


def sentences(text):
    """Split on sentence-final punctuation, flattening LaTeX line wrapping.

    Sentence scope matters here: an earlier version of this guard matched a fixed character
    window starting at 'neither', so whether 'Bonferroni' counted as present depended on which
    side of that word it happened to fall. Both of its negative controls came out inverted --
    the false sentence passed and the correct one failed. Scope to the sentence, not a window.
    """
    flat = " ".join(text.split())
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", flat) if s.strip()]


TEXTTT = re.compile(r"\\texttt\{([^}]*)\}")


def strip_markup(cell):
    """`\\texttt{keep14}@200000` -> `keep14@200000`, so labels read as the arm names do."""
    return TEXTTT.sub(r"\1", cell).strip()


def parse_rows(text):
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if "&" not in line or stripped.startswith("%") or stripped.startswith("\\"):
            continue
        cells = [c.strip() for c in line.split("&")]
        if len(cells) < 8:
            continue
        m_p = re.search(r"\d\.\d+", cells[6])
        if not m_p:
            continue
        label = f"{strip_markup(cells[0])} {strip_markup(cells[1])}"
        rows.append([label, float(m_p.group()), None, None])
    return rows


def bh_qvalues(rows):
    """Tie-aware BH step-up: q_(i) = min_{j>=i} m*p_(j)/j, enforced monotone."""
    m = len(rows)
    order = sorted(range(m), key=lambda i: rows[i][1])
    running = 1.0
    for pos in range(m - 1, -1, -1):
        i = order[pos]
        running = min(running, m * rows[i][1] / (pos + 1))
        rows[i][2] = min(running, 1.0)
        rows[i][3] = pos + 1
    return rows


def main():
    if not TABLE.exists():
        print(f"CANNOT PARSE: {TABLE} does not exist")
        return 3
    rows = parse_rows(TABLE.read_text(encoding="utf-8"))
    if not rows:
        print(f"CANNOT PARSE: no data rows recognised in {TABLE}")
        return 3

    rows = bh_qvalues(rows)
    m = len(rows)
    bonf_thr = ALPHA / m
    n_bh = sum(1 for r in rows if r[2] <= ALPHA)
    n_bonf = sum(1 for r in rows if r[1] < bonf_thr)

    print(f"{'rank':>4} {'cell':32}{'p':>9}{'q_BH':>10}{'BH':>5}{'Bonf':>6}")
    for r in sorted(rows, key=lambda r: r[3]):
        print(f"{r[3]:>4} {r[0][:32]:32}{r[1]:9.4f}{r[2]:10.4f}"
              f"{'YES' if r[2] <= ALPHA else '-':>5}{'YES' if r[1] < bonf_thr else '-':>6}")
    print()

    failures = []
    if m != EXPECT_M:
        failures.append(f"row count {m}, expected {EXPECT_M}")
    if n_bh != EXPECT_BH:
        failures.append(f"BH rejects {n_bh}, expected {EXPECT_BH}")
    if n_bonf != EXPECT_BONF:
        failures.append(f"Bonferroni rejects {n_bonf}, expected {EXPECT_BONF}")

    surv = [r for r in rows if SURVIVES in r[0]]
    fail = [r for r in rows if FAILS in r[0]]
    if not surv:
        failures.append(f"row matching {SURVIVES!r} not found")
    elif surv[0][2] > ALPHA:
        failures.append(f"{surv[0][0]} q_BH={surv[0][2]:.4f} > {ALPHA}; it is expected to "
                        f"SURVIVE BH (this is the half of the re-sort that is defensible)")
    if not fail:
        failures.append(f"row matching {FAILS!r} not found")
    elif fail[0][2] <= ALPHA:
        failures.append(f"{fail[0][0]} q_BH={fail[0][2]:.4f} <= {ALPHA}; it is expected to "
                        f"FAIL BH (the upward-moving cell). If this changed, §5.3's "
                        f"bidirectionality claim may now be supportable -- re-argue it.")

    # Check 3: prose must not assert a both/neither verdict on the trace cells under BH.
    bad_prose = []
    for tex in sorted(PROSE.glob("*.tex")):
        for sent in sentences(tex.read_text(encoding="utf-8")):
            if not re.search(r"\btrace\b", sent, re.I):
                continue
            if not re.search(r"\bneither\b|\bboth\b", sent, re.I):
                continue
            if not re.search(r"Benjamini|\bBH\b", sent):
                continue          # a Bonferroni-only sentence never names BH; it is correct
            if EXONERATED.search(sent):
                continue          # already says exactly one of the two survives
            bad_prose.append((tex.name, sent[:220]))
    if bad_prose:
        failures.append("prose asserts a both/neither verdict on the trace cells while naming "
                        "BH; under BH exactly ONE of the two survives (qwen3/k14, q=0.0297):")
        for name, sent in bad_prose:
            failures.append(f"    {name}: {sent}")

    if failures:
        print("FAIL:")
        for f in failures:
            print(f"  {f}")
        return 2

    print(f"PASS: m={m}, BH rejects {n_bh}, Bonferroni rejects {n_bonf}.")
    print(f"      {surv[0][0]} survives BH (q={surv[0][2]:.4f}); "
          f"{fail[0][0]} does not (q={fail[0][2]:.4f}).")
    print("      Internal consistency only -- see the module docstring on dependence.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
