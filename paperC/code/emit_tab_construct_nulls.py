#!/usr/bin/env python3
r"""paperC E3: GENERATE sections/tab_construct_nulls.tex from the evidence JSON.

Why this exists
---------------
`paperC/SUBMISSION_GAP_AUDIT.md` gap **E3**: there is no `.tex`-level integrity
binding between evidence and prose. paperB has one --- `paperB/sections/
app_tab_integrity.tex` is written by `paperB/scripts/generate_appendix_tables.py:
write_integrity()`, so every number in it is a dict lookup into a summary JSON
and cannot be mistyped. paperC's numbers, by contrast, live in prose in a
478-line README with 30+ inline retractions, and its `.tex` tables
(`sections/tab_nulls.tex` and friends) are hand-typed. E3 notes why that is not
paranoia: this direction has already shipped 10 truncated cells *with summaries
already written* (README:401-420), i.e. a hand-authored artefact that read as
finished while being wrong.

E2 (closed 2026-08-15, commit `f5d52b2`,
`evidence/e2_single_source_assertion.json`) established that
`evidence/floor_winners_curse_calibration.json` **already is** the single
machine-readable source for the 9-row headline null table --- 9/9 rows reproduce
the README. E2 therefore supplies the SOURCE; this script supplies the BINDING.
No number is recomputed here and no evidence file is modified: floors, chance
lines, calibration moments and p-values are read, formatted, and cross-checked.

What is deliberately NOT in this file
-------------------------------------
Any floor or chance literal. `grep -nE '0\.[0-9]{3}' emit_tab_construct_nulls.py`
must return nothing but tolerances. The expected row count and the expected
per-row `construct`/`n`/`k`/`floor`/`chance` are taken from the *other* evidence
file (`e2_single_source_assertion.json`), so the row-identity self-test is a
two-file agreement rather than a hardcoded table.

Self-tests (fail-closed: raise, write nothing) --- pattern copied from
`paperC/code/construct_nulls_length_unit.py`, which raises rather than writing on
any mismatch:

  T1 provenance fields present and usable (schema_version, seed, n_draws, method)
  T2 row identity agrees with e2_single_source_assertion.json, row for row
  T3 stored gap_pp == round((floor - chance) * 100, 3)
  T4 n * floor is an integer count to within the JSON's own 6-dp storage error
  T5 floor >= 1/k  (a maximum over k marginals that sum to 1 cannot be smaller)
  T6 per-row `survives` agrees with p_one_sided, and with the top-level
     survives / inside_noise partition
  T7 every number in the ALREADY-SHIPPED hand-typed sections/tab_nulls.tex is a
     correct rounding of this JSON  <-- this is the actual E3 gap: it catches a
     hand-typing error in the paper's existing headline table
  T8 round-trip: the table this script just built is re-parsed and every number
     in it is re-verified against the JSON before anything is written to disk

CPU only. No GPU, no model, no network.

Usage:
  python paperC/code/emit_tab_construct_nulls.py
  python paperC/code/emit_tab_construct_nulls.py --out paperC/sections/tab_construct_nulls.tex
  python paperC/code/emit_tab_construct_nulls.py --check-only   # self-tests, no write
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)

EVIDENCE = os.path.join(PAPER, "evidence", "floor_winners_curse_calibration.json")
E2_ASSERT = os.path.join(PAPER, "evidence", "e2_single_source_assertion.json")
HANDTYPED = os.path.join(PAPER, "sections", "tab_nulls.tex")
DEFAULT_OUT = os.path.join(PAPER, "sections", "tab_construct_nulls.tex")

# Tolerances only. No floors, no chance lines.
TOL_GAP = 5.05e-4      # gap_pp is stored at 3 dp; a .5 tie may round either way
TOL_COUNT = 2e-2       # n * floor vs integer, with floor stored at 6 dp
TOL_EXACT = 1e-12

ROW_KEYS = ("construct", "n", "k", "floor", "chance")


def _fail(msgs: list[str]) -> None:
    for m in msgs:
        print("[FAIL] " + m)
    raise SystemExit("self-test FAILED; nothing written")


def _dp(literal: str) -> int:
    return len(literal.split(".")[1]) if "." in literal else 0


def is_rounding_of(literal: str, value: float) -> bool:
    """True iff `literal` is a correct fixed-point rounding of `value`.

    Compared as a half-ulp interval rather than by string equality, because the
    prose stores the same quantity at 2, 3, 4 and 6 dp in different places. E2's
    documented error class was exactly the reverse mistake: a naive grep for the
    README's ROUNDED form misses a value stored at full precision.
    """
    d = _dp(literal)
    half = 0.5 * (10.0 ** -d)
    return abs(float(literal) - value) <= half * (1.0 + 1e-9)


def tex_escape(s: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    return s


def p_bound_exponent(n_draws: int) -> int:
    """Largest m with 10^-m >= 1/n_draws, so `p < 10^-m` is a true statement.

    Derived from the JSON's own n_draws instead of writing `10^{-5}` by hand.
    """
    return int(math.floor(math.log10(n_draws)))


def fmt_p(p: float, n_draws: int) -> str:
    if p <= 1.0 / n_draws:
        return "$<10^{-%d}$" % p_bound_exponent(n_draws)
    return "%.3f" % p


def load() -> tuple[dict, dict]:
    with open(EVIDENCE, encoding="utf-8") as f:
        ev = json.load(f)
    with open(E2_ASSERT, encoding="utf-8") as f:
        a2 = json.load(f)
    return ev, a2


# --------------------------------------------------------------------------
# self-tests on the evidence itself
# --------------------------------------------------------------------------
def check_evidence(ev: dict, a2: dict) -> list[str]:
    bad: list[str] = []
    rows = ev.get("rows") or []

    # T1 provenance
    for key in ("schema_version", "seed", "n_draws", "method"):
        if not ev.get(key):
            bad.append(f"T1 {EVIDENCE}: missing/empty provenance field '{key}'")
    n_draws = ev.get("n_draws") or 0
    if not isinstance(n_draws, int) or n_draws <= 0:
        bad.append(f"T1 n_draws is not a positive int: {n_draws!r}")

    # T2 row identity vs the OTHER evidence file (no hardcoded expectations)
    a_rows = a2.get("rows") or []
    if a2.get("n_rows") != len(a_rows):
        bad.append(f"T2 {E2_ASSERT}: n_rows={a2.get('n_rows')} but len(rows)={len(a_rows)}")
    if len(rows) != len(a_rows):
        bad.append(f"T2 row count {len(rows)} != e2 assertion's {len(a_rows)}")
    else:
        for i, (r, a) in enumerate(zip(rows, a_rows)):
            for key in ROW_KEYS:
                if key not in r:
                    bad.append(f"T2 row {i}: evidence missing key '{key}'")
                    continue
                if key not in a:
                    bad.append(f"T2 row {i}: e2 assertion missing key '{key}'")
                    continue
                if isinstance(r[key], str):
                    if r[key] != a[key]:
                        bad.append(f"T2 row {i} '{key}': {r[key]!r} != e2 {a[key]!r}")
                elif abs(float(r[key]) - float(a[key])) > TOL_EXACT:
                    bad.append(f"T2 row {i} '{key}': {r[key]!r} != e2 {a[key]!r}")

    for i, r in enumerate(rows):
        tag = f"row {i} ({r.get('construct')})"
        floor, chance, n, k = r["floor"], r["chance"], r["n"], r["k"]

        # T3 gap_pp is derivable
        derived = (floor - chance) * 100.0
        if abs(derived - r["gap_pp"]) > TOL_GAP:
            bad.append(f"T3 {tag}: stored gap_pp={r['gap_pp']} but "
                       f"(floor-chance)*100={derived:.6f}")

        # T4 floor is a count / n
        cnt = n * floor
        if abs(cnt - round(cnt)) > TOL_COUNT:
            bad.append(f"T4 {tag}: n*floor={cnt:.6f} is not an integer count "
                       f"(n={n}); floor is not a label marginal on n items")

        # T5 max of k marginals summing to 1 is >= 1/k
        if floor < 1.0 / k - 1e-9:
            bad.append(f"T5 {tag}: floor={floor} < 1/k={1.0/k:.6f}; a maximum "
                       f"over {k} marginals that sum to 1 cannot be smaller")

        # T6 survives <-> p
        if n_draws:
            want = r["p_one_sided"] <= 1.0 / n_draws
            if bool(r["survives"]) != want:
                bad.append(f"T6 {tag}: survives={r['survives']} but "
                           f"p_one_sided={r['p_one_sided']} vs 1/n_draws="
                           f"{1.0/n_draws:.3e}")

    # T6b top-level partition
    surv = {r["construct"] for r in rows if r.get("survives")}
    noise = {r["construct"] for r in rows if not r.get("survives")}
    if set(ev.get("survives") or []) != surv:
        bad.append(f"T6b top-level 'survives' {sorted(ev.get('survives') or [])} "
                   f"!= per-row {sorted(surv)}")
    if set(ev.get("inside_noise") or []) != noise:
        bad.append(f"T6b top-level 'inside_noise' {sorted(ev.get('inside_noise') or [])} "
                   f"!= per-row {sorted(noise)}")
    return bad


# --------------------------------------------------------------------------
# T7: audit the already-shipped hand-typed table
# --------------------------------------------------------------------------
NUM = re.compile(r"(?<![\d.])\d+\.\d+(?![\d.])")


def parse_body_rows(text: str) -> list[list[str]]:
    """Data rows of the first tabular between \\midrule and \\bottomrule."""
    body = text.split(r"\midrule", 1)[-1].split(r"\bottomrule", 1)[0]
    out = []
    for line in body.splitlines():
        line = line.strip()
        if not line.endswith(r"\\") or line.startswith("%"):
            continue
        out.append([c.strip() for c in line[:-2].split("&")])
    return out


def check_handtyped(ev: dict) -> list[str]:
    """Every literal in sections/tab_nulls.tex must round-trip to the JSON.

    This is the concrete risk E3 names. The hand-typed table carries n, floor,
    chance, gap, ratio, E[f-hat], q95 and p; each is checked against the
    corresponding JSON field at the precision the table itself chose.
    """
    bad: list[str] = []
    if not os.path.exists(HANDTYPED):
        return [f"T7 {HANDTYPED} not found"]
    with open(HANDTYPED, encoding="utf-8") as f:
        rows = parse_body_rows(f.read())
    ev_rows = ev["rows"]
    n_draws = ev["n_draws"]
    if len(rows) != len(ev_rows):
        return [f"T7 {os.path.basename(HANDTYPED)} has {len(rows)} data rows, "
                f"evidence has {len(ev_rows)}"]

    for i, (cells, r) in enumerate(zip(rows, ev_rows)):
        tag = f"T7 tab_nulls.tex row {i+1} ({cells[0]})"
        # n is an integer cell
        try:
            n_tex = int(cells[1].replace(",", ""))
        except (ValueError, IndexError):
            bad.append(f"{tag}: cannot parse n from {cells[1]!r}")
            n_tex = None
        if n_tex is not None and n_tex != r["n"]:
            bad.append(f"{tag}: n={n_tex} but evidence n={r['n']}")

        # column -> the evidence quantities it may legitimately be showing
        want = {
            3: [r["floor"]],
            4: [r["chance"]],
            5: [(r["floor"] - r["chance"]) * 100.0, r["gap_pp"]],
            6: [r["floor"] / r["chance"]],
            7: [r["E_max_balanced"]],
            8: [r["q95_balanced"]],
        }
        for col, cands in want.items():
            if col >= len(cells):
                bad.append(f"{tag}: missing column {col}")
                continue
            lits = NUM.findall(cells[col])
            if not lits:
                bad.append(f"{tag} col{col}: no number in {cells[col]!r}")
                continue
            for lit in lits:
                if not any(is_rounding_of(lit, v) for v in cands):
                    bad.append(f"{tag} col{col}: {lit} is not a rounding of "
                               + " or ".join("%.10g" % v for v in cands))
        # p column: either the derived bound string or a rounding of p
        pcell = cells[9] if len(cells) > 9 else ""
        expect_bound = fmt_p(r["p_one_sided"], n_draws)
        if expect_bound.startswith("$<"):
            if pcell.replace(" ", "") != expect_bound.replace(" ", ""):
                bad.append(f"{tag} p: {pcell!r} != {expect_bound!r} "
                           f"(p_one_sided={r['p_one_sided']}, n_draws={n_draws})")
        else:
            lits = NUM.findall(pcell)
            if not lits or not all(is_rounding_of(l, r["p_one_sided"]) for l in lits):
                bad.append(f"{tag} p: {pcell!r} is not a rounding of "
                           f"{r['p_one_sided']}")
    return bad


# --------------------------------------------------------------------------
# the table
# --------------------------------------------------------------------------
def tt_breakable(s: str) -> str:
    r"""\texttt{} the string one semicolon-clause at a time.

    A single \texttt{} around the whole `method` string is one long rigid box:
    measured, it added 9 `Underfull \hbox (badness 10000)` warnings to the build.
    Splitting at `; ` gives TeX legal break points between clauses while keeping
    every clause verbatim.
    """
    parts = [p.strip() for p in str(s).split(";") if p.strip()]
    return r"; ".join(r"\texttt{" + tex_escape(p) + r"}" for p in parts)


def build(ev: dict, sha12: str) -> list[str]:
    n_draws = ev["n_draws"]
    lines = [
        r"% GENERATED FILE -- DO NOT EDIT BY HAND.",
        r"% Emitted by paperC/code/emit_tab_construct_nulls.py from",
        r"% paperC/evidence/floor_winners_curse_calibration.json (sha256 "
        + sha12 + r").",
        r"% Regenerate with:  python paperC/code/emit_tab_construct_nulls.py",
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.4pt}",
        r"\begin{tabular}{lrrrrrrl}",
        r"\toprule",
        r"Construct & $n$ & $k$ & Chance & Floor & Gap (pp) & $p$ & Verdict \\",
        r"\midrule",
    ]
    for r in ev["rows"]:
        verdict = ("above balanced null" if r["survives"]
                   else "inside estimator noise")
        lines.append(
            " & ".join([
                tex_escape(str(r["construct"])),
                str(r["n"]),
                str(r["k"]),
                "%.6f" % r["chance"],
                "%.6f" % r["floor"],
                "%+.3f" % ((r["floor"] - r["chance"]) * 100.0),
                fmt_p(r["p_one_sided"], n_draws),
                verdict,
            ]) + r" \\"
        )
    caption = (
        r"\textbf{Construct-null manifest (generated, not transcribed).} "
        r"Every value in this table is emitted programmatically from "
        r"\texttt{paperC/evidence/floor\_winners\_curse\_calibration.json} by "
        r"\texttt{paperC/code/emit\_tab\_construct\_nulls.py}; no number in it "
        r"is typed by hand. $k$ is the nominal option count and Chance is the "
        r"chance line used for that row, so the two MMLU-Pro rows differ only "
        r"in whether chance is naive $1/10$ or the item average "
        r"$\mathrm{mean}(1/\texttt{n\_opt})$. Gap is "
        r"$100\,(\text{Floor}-\text{Chance})$, recomputed here rather than "
        r"read. The $p$-value is $\Pr(\hat f \ge \text{Floor})$ under an "
        r"exactly balanced multinomial null at that construct's own $(n,k)$, "
        r"and Verdict states only whether that null could plausibly have "
        r"produced the observed floor; a floor inside the estimator's noise "
        r"still shows that chance is the wrong reference to report. "
        r"Provenance, from the same file: "
        r"\texttt{schema\_version}~" + tex_escape(str(ev["schema_version"]))
        + r", \texttt{seed}~" + str(ev["seed"])
        + r", \texttt{n\_draws}~$" + ("%d" % n_draws)
        + r"$, \texttt{sha256}~\texttt{" + sha12 + r"}. Method: "
        + tt_breakable(ev["method"]) + r"."
    )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        # The caption is verbatim-heavy (long \texttt paths plus a \texttt method
        # string), so it leaves 5 `Underfull \hbox` warnings. Measured: \sloppy,
        # \raggedright and \footnotesize inside the float all give the identical
        # total (36), so none of them is included -- an ineffective typesetting
        # command in a generated file is just noise. For scale, the paper's
        # existing hand-written tab_integrity.tex and tab_claims.tex carry 14 and
        # 8 such warnings. Underfull hboxes are not errors: the build reports
        # 0 errors, 0 undefined references and 0 overfull boxes.
        r"\caption{" + caption + r"}",
        r"\label{tab:app-construct-nulls}",
        r"\end{table}",
        "",
    ]
    return lines


def roundtrip(lines: list[str], ev: dict) -> list[str]:
    """T8: re-parse what we just built and re-verify it against the JSON."""
    bad: list[str] = []
    rows = parse_body_rows("\n".join(lines))
    ev_rows = ev["rows"]
    n_draws = ev["n_draws"]
    if len(rows) != len(ev_rows):
        return [f"T8 generated table has {len(rows)} rows, evidence has {len(ev_rows)}"]
    for i, (cells, r) in enumerate(zip(rows, ev_rows)):
        tag = f"T8 generated row {i+1}"
        if len(cells) != 8:
            bad.append(f"{tag}: {len(cells)} cells, expected 8")
            continue
        if cells[0] != tex_escape(str(r["construct"])):
            bad.append(f"{tag}: construct {cells[0]!r} != {r['construct']!r}")
        if cells[1] != str(r["n"]):
            bad.append(f"{tag}: n {cells[1]!r} != {r['n']}")
        if cells[2] != str(r["k"]):
            bad.append(f"{tag}: k {cells[2]!r} != {r['k']}")
        for col, val in ((3, r["chance"]), (4, r["floor"])):
            lits = NUM.findall(cells[col])
            if not lits or not is_rounding_of(lits[0], val):
                bad.append(f"{tag} col{col}: {cells[col]!r} not a rounding of {val}")
        lits = NUM.findall(cells[5])
        gap = (r["floor"] - r["chance"]) * 100.0
        if not lits or not is_rounding_of(lits[0], gap):
            bad.append(f"{tag} gap: {cells[5]!r} not a rounding of {gap:.6f}")
        if cells[6] != fmt_p(r["p_one_sided"], n_draws):
            bad.append(f"{tag} p: {cells[6]!r} != {fmt_p(r['p_one_sided'], n_draws)!r}")
        if bool(r["survives"]) != cells[7].startswith("above"):
            bad.append(f"{tag} verdict {cells[7]!r} vs survives={r['survives']}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    ev, a2 = load()
    with open(EVIDENCE, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()

    bad = check_evidence(ev, a2) + check_handtyped(ev)
    if bad:
        _fail(bad)

    lines = build(ev, sha[:12])
    bad = roundtrip(lines, ev)
    if bad:
        _fail(bad)

    n = len(ev["rows"])
    print(f"[selftest] OK: T1-T6 provenance/invariants on {n} rows; "
          f"T2 row identity agrees with e2_single_source_assertion.json; "
          f"T7 hand-typed sections/tab_nulls.tex round-trips "
          f"({n}/{n} rows); T8 generated table round-trips ({n}/{n} rows)")
    print(f"[source] {os.path.relpath(EVIDENCE, PAPER)} sha256={sha}")
    for r in ev["rows"]:
        print(f"  {r['construct']:30s} n={r['n']:6d} k={r['k']:2d} "
              f"chance={r['chance']:.6f} floor={r['floor']:.6f} "
              f"gap={100*(r['floor']-r['chance']):+.3f}pp "
              f"p={fmt_p(r['p_one_sided'], ev['n_draws'])}")
    if args.check_only:
        print("[check-only] nothing written")
        return 0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[done] wrote {args.out} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
