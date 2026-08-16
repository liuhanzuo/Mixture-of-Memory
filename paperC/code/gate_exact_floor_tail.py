#!/usr/bin/env python3
"""Guard: a one-sided tail must INCLUDE the observed outcome.

THE HAZARD
----------
Every floor in this paper is an exact rational: a count of gold labels over the item count.
The table prints it rounded to six decimals. When the rounded value is LARGER than the exact
rational, the event `{max_L m_L >= printed_floor}` no longer contains the observed outcome,
because the observed count is one short of what the printed threshold demands. A one-sided
p-value P(X >= x_obs) must include x_obs; excluding it biases p DOWNWARD, i.e. toward
"this floor survives the null", i.e. in the authors' favour.

MEASURED, 2026-08-16
--------------------
Five of the nine rows have stored > exact and therefore carry the hazard:

    MMLU-Pro naive       1403/12032 = 0.116605718  vs stored 0.116606
    MMLU-Pro item-avg.   1403/12032 = 0.116605718  vs stored 0.116606
    MMLU                 3776/14042 = 0.268907563  vs stored 0.268908
    PIQA                  928/1838  = 0.504896627  vs stored 0.504897
    BoolQ                2033/3270  = 0.621712538  vs stored 0.621713

For MMLU-Pro the smallest integer count satisfying `c/12032 >= 0.116606` is 1404, while the
observed count is 1403 -- so the observed outcome sits outside its own tail event. The
correct p for that row is 0.083, not the 0.078 obtained against the rounded threshold.

PIQA is the instructive case for how to read this table. Its printed p is 0.691, which is
the INCLUSIVE tail computed against the exact rational (0.691218 in a 2e6-draw check);
the strict tail against the stored decimal is 0.657285. So PIQA's number is right, and the
presence of the hazard on a row does not by itself mean that row's p was computed wrongly.
That is exactly why this needs a guard rather than an assumption in either direction.

WHAT THIS GUARD DOES
--------------------
It does not recompute p. It reports, per row, whether the stored floor exceeds the exact
rational, and therefore which rows must have their p computed against the exact rational
rather than the printed decimal. Any emitter that thresholds on the printed value is wrong
on those rows.

Exit codes: 0 = no row carries the hazard. 1 = at least one row does (informational --
this is expected and is NOT a failure by itself; it flags rows needing exact-rational
handling). 3 = could not parse.
"""
import re
import sys
from fractions import Fraction
from pathlib import Path

SECTIONS = Path(__file__).resolve().parent.parent / "sections"
TABLE = SECTIONS / "tab_nulls.tex"
COL_N, COL_FLOOR = 1, 3


def parse_rows(text):
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if "&" not in line or stripped.startswith("%") or stripped.startswith("\\"):
            continue
        cells = [c.strip() for c in line.split("&")]
        if len(cells) < 10:
            continue
        m_n = re.search(r"\b(\d{3,6})\b", cells[COL_N])
        m_f = re.search(r"\d+\.\d+", cells[COL_FLOOR])
        if not (m_n and m_f):
            continue
        rows.append((cells[0], int(m_n.group(1)), m_f.group()))
    return rows


def main():
    if not TABLE.exists():
        print(f"CANNOT PARSE: {TABLE} does not exist")
        return 3
    rows = parse_rows(TABLE.read_text(encoding="utf-8"))
    if not rows:
        print(f"CANNOT PARSE: no data rows recognised in {TABLE}")
        return 3

    print(f"{'row':24}{'n':>7}{'count':>7}{'exact':>14}{'stored':>12}{'hazard':>9}")
    flagged = []
    for name, n, floor_str in rows:
        stored = Fraction(floor_str)
        count = round(float(floor_str) * n)
        exact = Fraction(count, n)
        hazard = stored > exact
        if hazard:
            flagged.append((name, n, count, exact, stored))
        print(f"{name[:24]:24}{n:7d}{count:7d}{float(exact):14.9f}"
              f"{float(stored):12.6f}{'YES' if hazard else '-':>9}")

    print()
    if flagged:
        print(f"{len(flagged)} of {len(rows)} rows have stored floor > exact rational.")
        print("On these rows, P(max >= stored) EXCLUDES the observed outcome and biases p")
        print("downward, toward 'the floor survives the null'. Their p MUST be computed")
        print("against the exact rational:")
        for name, n, count, exact, stored in flagged:
            need = count
            while Fraction(need, n) < stored:
                need += 1
            print(f"  {name}: observed count {count}, but c/{n} >= {float(stored):.6f} "
                  f"requires c >= {need}")

        # A row-list the prose contradicts is a defect, not an informational note.
        # rc=1 alone made this gate indistinguishable from a real failure in a sweep, and
        # that is exactly how the defect below survived: 03b_nulls.tex asserted "the other
        # seven rows are unaffected because their stored floors round down" while this gate
        # was printing FOUR hazard rows on every run. MMLU and BoolQ round UP too; they are
        # merely at p<1e-5, where a one-count shift cannot reach the verdict. So the prose
        # was wrong about the mechanism and right about the conclusion, which is the kind of
        # error a reviewer finds and an author cannot explain.
        rc = 1
        prose = SECTIONS / "03b_nulls.tex"
        if prose.exists():
            text = " ".join(prose.read_text(encoding="utf-8").split())
            claimed = re.search(r"the other (\w+) rows are unaffected", text)
            if claimed:
                print()
                print("PROSE CONTRADICTION: 03b_nulls.tex still says "
                      f"'the other {claimed.group(1)} rows are unaffected', which implies "
                      f"{len(rows) - len(flagged)} clean rows, but {len(flagged)} of "
                      f"{len(rows)} carry the hazard.")
                rc = 2
            named = [n for n, *_ in flagged if _prose_names(text, n)]
            missing = [n for n, *_ in flagged if n not in named]
            if missing:
                print()
                print("PROSE OMISSION: these hazard rows are not named in the rounding "
                      f"paragraph: {missing}")
                print("      Naming only the rows whose verdict moved understates the "
                      "scope of the convention.")
                rc = 2
        return rc

    print(f"All {len(rows)} rows have stored floor <= exact rational; the inclusive tail is safe.")
    return 0


def _prose_names(text, row_label):
    """Is this table row's construct named in the rounding paragraph?

    Row labels carry a chance-convention suffix ('MMLU-Pro, naive'); the prose names the
    construct once and covers both of its rows, so match on the construct only.
    """
    construct = row_label.split(",")[0].strip()
    return construct in text


if __name__ == "__main__":
    sys.exit(main())
