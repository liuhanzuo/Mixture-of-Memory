#!/usr/bin/env python3
"""Guard: E[max_L m_L] >= E[m_A] must hold for every row of the construct-null table.

WHY THIS EXISTS
---------------
Round_04's stats reviewer found the MMLU-Pro calibration defect without running a single
Monte Carlo draw, using only two numbers already printed side by side in the paper's own
table. The argument is a two-line identity:

  * The `Chance` column, under the item-average convention, is defined by the paper itself
    as mean(1/n_opt). Letter A is legal on EVERY item of every construct here, so
    mean(1/n_opt) is exactly E[m_A] -- the expected marginal of letter A under a balanced
    null that respects each item's legal option set.
  * `E[f_hat]` is E[max_L m_L], an expectation of a maximum over the letters.
  * A maximum cannot have expectation below one of its own arguments. So
    E[f_hat] >= Chance is FORCED, for any legal balanced process whatsoever.

On the frozen round_04 snapshot the MMLU-Pro item-average row printed
Chance = 0.110877 and E[f_hat] = 0.104460. That pair is internally impossible: it reports
the expectation of a maximum BELOW one of its arguments. It was the tell that the null had
been drawn uniform over all ten letters -- crediting J on the 2051 items where J is not
legal -- rather than over each item's legal letters. Under the corrected null the same row
reads E[f_hat] = 0.113873, and this test passes.

Eight of nine rows passed the test even before the fix. The single violation was the row
that carried the paper's flagship claim.

WHAT THIS GUARD IS FOR
----------------------
It costs milliseconds, needs no simulation, no seed and no per-item data, and it fires
exactly on the class of defect that took five independent Monte Carlo derivations to pin
down. Run it whenever the construct-null table is regenerated.

SCOPE, HONESTLY STATED
----------------------
Passing is NECESSARY, not sufficient. A null can respect this inequality and still be
misspecified in other ways. This guard detects one specific, high-consequence failure:
a null whose support excludes outcomes the data can produce, in the direction that
understates E[max] and so overstates significance.

Exit codes: 0 = all rows pass. 2 = at least one row violates. 3 = could not parse.
"""
import re
import sys
from pathlib import Path

TABLE = Path(__file__).resolve().parent.parent / "sections" / "tab_nulls.tex"
# column layout of tab_nulls.tex, 0-indexed after splitting on '&'
COL_FLOOR, COL_CHANCE, COL_EMAX = 3, 4, 7


def parse_rows(text):
    """Yield (construct, chance, e_max, floor) for each data row."""
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if "&" not in line or stripped.startswith("%") or stripped.startswith("\\"):
            continue
        cells = [c.strip() for c in line.split("&")]
        if len(cells) < 10:
            continue
        nums = []
        for cell in cells:
            cleaned = cell.replace("$", "").replace("\\times", "")
            m = re.search(r"-?\d+\.\d+", cleaned)
            nums.append(float(m.group()) if m else None)
        chance, e_max, floor = nums[COL_CHANCE], nums[COL_EMAX], nums[COL_FLOOR]
        if chance is None or e_max is None:
            continue
        rows.append((cells[0], chance, e_max, floor))
    return rows


def main():
    if not TABLE.exists():
        print(f"CANNOT PARSE: {TABLE} does not exist")
        return 3
    rows = parse_rows(TABLE.read_text(encoding="utf-8"))
    if not rows:
        print(f"CANNOT PARSE: no data rows recognised in {TABLE}")
        return 3

    print(f"{'construct':28}{'Chance=E[m_A]':>15}{'E[f_hat]':>12}{'verdict':>12}")
    violations = []
    for name, chance, e_max, _floor in rows:
        ok = e_max >= chance
        if not ok:
            violations.append((name, chance, e_max))
        print(f"{name[:28]:28}{chance:15.6f}{e_max:12.6f}{'OK' if ok else 'VIOLATION':>12}")

    print()
    if violations:
        print(f"FAIL: {len(violations)} of {len(rows)} rows report the expectation of a maximum")
        print("      BELOW one of its own arguments. That is impossible for any legal balanced")
        print("      process, so the null does not respect the construct's legal option sets.")
        for name, chance, e_max in violations:
            print(f"      {name}: E[f_hat]={e_max:.6f} < Chance={chance:.6f} "
                  f"(deficit {chance - e_max:.6f})")
        return 2

    print(f"PASS: all {len(rows)} rows satisfy E[max_L m_L] >= E[m_A].")
    print("      Necessary, not sufficient -- see the module docstring.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
