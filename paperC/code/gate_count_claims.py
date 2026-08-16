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

# ---- the abstract's construct census, recomputed from tab_nulls.tex ------------
# These four numbers are the abstract's headline framing of the calibration result and
# nothing checked them. They are derived from the SHIPPED table rather than an evidence
# JSON because the table is what a reader counts, and the table is itself bound to the
# evidence by code/emit_tab_construct_nulls.py plus code/check_prose_vs_evidence.py.
#
# The subtlety worth encoding: tab_nulls has NINE rows but EIGHT distinct constructs,
# because MMLU-Pro appears twice, once per chance convention (naive 1/10 and
# item-averaged mean(1/n_opt)). A census that counts rows says nine and eight, and both
# would be wrong.
NULLS_TABLE = SECTIONS / "tab_nulls.tex"
SIG_MARK = "10^{-5}"


def _nulls_rows():
    rows = []
    for line in NULLS_TABLE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if "&" not in line or stripped.startswith("%") or stripped.startswith("\\"):
            continue
        cells = [c.strip() for c in line.split("&")]
        if len(cells) < 10 or cells[0] == "Construct":
            continue
        rows.append((cells[0].split(",")[0].strip(),
                     cells[9].replace("\\\\", "").strip()))
    return rows


def _by_construct():
    out = {}
    for name, p in _nulls_rows():
        out.setdefault(name, []).append(p)
    return out


def n_distinct_constructs(_doc=None):
    return len(_by_construct())


def n_significant_constructs(_doc=None):
    """A construct counts as significant only if EVERY one of its rows is p<1e-5."""
    return sum(1 for ps in _by_construct().values()
               if all(SIG_MARK in p for p in ps))


def n_remaining_constructs(_doc=None):
    return n_distinct_constructs() - n_significant_constructs()


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
    # ---- abstract construct census (evidence file is None -> derived from tab_nulls) --
    ("abstract: distinct letter constructs", "00_abstract.tex",
     r"only two of the (\w+) letter constructs",
     None, n_distinct_constructs,
     "tab_nulls has NINE rows but EIGHT distinct constructs: MMLU-Pro appears twice, "
     "once per chance convention. A row count would say nine and be wrong."),
    ("abstract: constructs with p<1e-5", "00_abstract.tex",
     r"only (\w+) of the \w+ letter constructs",
     None, n_significant_constructs,
     "a construct is significant only if EVERY one of its rows is p<1e-5"),
    ("abstract: remaining constructs", "00_abstract.tex",
     r"on the remaining (\w+) the floor lies inside",
     None, n_remaining_constructs,
     "distinct minus significant; the two must sum to the eight"),
    # ---- appendix pointer vs the table it points at (found 2026-08-16) -------------
    # 09_appendix.tex described tab_mmlupro as "the MMLU-Pro letter read-out for all 12
    # cross-family cells" while that table PRINTS 17 cells and its own caption says
    # "over all 17 designated damaged cells". 12 is the non-OLMo subtotal -- the SAME
    # narrowing retracted as 14/15, silently dropping the five OLMo-2 rows that are
    # physically in the table being described.
    #
    # It survived all eight gates. gate_designated_denominator's CHECK 6 scans prose for
    # the retracted literals '14/15' and '10/15'; '12' is not a ratio and is not on that
    # list, so the class was only half closed. Registering the count here closes the
    # other half: it is recomputed from the table's own rows.
    #
    # 'cross-family' could NOT be read as legitimately scoping to non-OLMo, because the
    # manuscript uses it both ways: 07_limitations.tex:16 writes "Non-OLMo cross-family
    # arms", making cross-family the superset, and 09a_relocated.tex:52 calls the whole
    # 15-cell first launch "the first cross-family MMLU-Pro launch". On either reading
    # 12 is wrong.
    ("appendix pointer: cells in tab_mmlupro", "09_appendix.tex",
     r"MMLU-Pro letter read-out for all (\d+)",
     None, lambda _doc: n_cells_in_tab_mmlupro(),
     "recounted from tab_mmlupro's own data rows, expanding the collapsed Llama-3 "
     "'k14--k8' row into the 4 cells it represents"),
    # ---- constant emitters, negative control EXCLUDED (found 2026-08-16) -----------
    # 05_analysis.tex:29 and 09a_relocated.tex:42 both said "sixteen damaged
    # cross-family cells have accuracy equal to the marginal of the emitted letter to
    # machine precision". Sixteen is arithmetically right over all six evidence tasks --
    # but SEVEN of them are Winogrande, the paper's declared negative control, which
    # 09a_relocated.tex:24 and 04_experiments.tex:5 both say "enters no denominator".
    # Both host sentences frame the count as small-BENCHMARK evidence for the paper's
    # constant-emission claim, so the control was doing headline work while being
    # formally excluded. Verified by recomputing the source's own criterion
    # (modal_pred_share >= 0.99 AND |acc - marginal| < 1e-6) over the 72 damaged letter
    # cells: 16 total = 7 winogrande + 3 arc_challenge + 3 openbookqa + 2 commonsense_qa
    # + 1 arc_easy, i.e. NINE in the designated set.
    #
    # Both sentences now lead with nine and report sixteen as the control-inclusive
    # figure. The prose had to be disambiguated as well as corrected: 05_analysis.tex
    # already contains "the nine" meaning the 9/85 ABOVE-FLOOR cells, a different
    # quantity that happens to share the integer.
    ("constant emitters, designated set only", "05_analysis.tex",
     r"literal constant emitters: (\w+) damaged cells in the designated set",
     "second_mc_benchmark_crossfamily/gate2_crossfamily_nulls.json",
     lambda doc: n_constant_emitters(doc, exclude_negative_control=True),
     "the paper's own criterion, with Winogrande removed because it enters no "
     "denominator; 16 including it is also asserted and is checked separately"),
    ("constant emitters, control included", "05_analysis.tex",
     r"\((\w+) if the negative control is included",
     "second_mc_benchmark_crossfamily/gate2_crossfamily_nulls.json",
     lambda doc: n_constant_emitters(doc, exclude_negative_control=False),
     "the control-inclusive figure, so that BOTH numbers in the sentence are pinned "
     "and the difference between them cannot silently drift"),
    # ---- ARC directionality sigmas + the rounding (found 2026-08-16) ---------------
    # 03b_nulls.tex:37 is the paper's own honesty sentence -- "we report them rather than
    # absorbing them because the general claim that such a fix can only ever move against
    # the authors is false". It carried THREE wrong numbers, all on the
    # authors-FAVOURABLE side, so the overstatement inflated our own disclosure:
    #
    #   prose dp      -0.0025 / -0.0063   ->  E-CAL  -0.001205 / -0.005213
    #   prose sigma   "roughly -17/-23"   ->  E-CAL  -3.34 / -7.34
    #   prose p       0.453 -> 0.447      ->  p_aware 0.447647, i.e. 0.448
    #
    # The -17/-23 pair traces to MMLUPRO_NULL_FIX_20260816.md:40-41 (an 8-seed x 1e6 run
    # reporting -16.6/-22.6). That file is at the paperC ROOT, is NOT in evidence/, and is
    # cited by no .tex and by no claim-map row -- so the paper quoted a number a reviewer
    # could not reach. Worse, that same file CONTRADICTS ITSELF: its own summary table at
    # lines 105-106 prints -3.3/-7.3 and '0.453 -> 0.448', agreeing with E-CAL against its
    # own lines 40-41. So no source anywhere on disk supported 0.447.
    #
    # -17/-23 is also the wrong STATISTIC even in the 8-seed run's own terms: E-CAL's
    # dE_max_sigma is -9.78/-17.75, and that is E[max], not p. Mixing the two is how a
    # sigma for one quantity gets printed as the sigma for another.
    #
    # check_prose_vs_evidence.py cannot reach any of this: excluded_label_prefixes drops
    # p-value literals by design.
    ("ARC-Easy dp sigma", "03b_nulls.tex",
     r"at \$-(\d+\.\d)\$ and \$-\d+\.\d\$ standard errors",
     "construct_nulls_legality_aware.json",
     lambda doc: _arc_sigma(doc, "ARC-Easy"),
     "E-CAL rows[ARC-Easy].directionality.dp_sigma, |.| to 1dp; NOT dE_max_sigma, "
     "which is a different statistic (-9.78) and is where the retracted -17 came from"),
    ("ARC-Challenge dp sigma", "03b_nulls.tex",
     r"at \$-\d+\.\d\$ and \$-(\d+\.\d)\$ standard errors",
     "construct_nulls_legality_aware.json",
     lambda doc: _arc_sigma(doc, "ARC-Challenge"),
     "E-CAL rows[ARC-Challenge].directionality.dp_sigma, |.| to 1dp"),
    ("ARC-Challenge p_aware, 3dp", "03b_nulls.tex",
     r"0\.453\\rightarrow0\.(\d{3})",
     "construct_nulls_legality_aware.json",
     lambda doc: int(round(round(_arc_p(doc, "ARC-Challenge"), 3) * 1000)),
     "p_aware=0.447647 ROUNDS to 0.448; both shipped tables print 0.448 "
     "(tab_nulls.tex:14, tab_construct_nulls.tex:21). The prose truncated."),
    # The dp magnitudes themselves. Registered after a control showed they were the one
    # part of this sentence still unguarded: reverting them to the retracted -0.0025 /
    # -0.0063 left the gate green. Captured as ten-thousandths so the comparison is exact.
    ("ARC-Easy dp (1e-4)", "03b_nulls.tex",
     r"by \$-0\.(\d{4})\$ and \$-0\.\d{4}\$ in \$p\$",
     "construct_nulls_legality_aware.json",
     lambda doc: _arc_dp_1e4(doc, "ARC-Easy"),
     "E-CAL dp_aware_minus_blind = -0.001205 -> 0.0012 at 4dp"),
    ("ARC-Challenge dp (1e-4)", "03b_nulls.tex",
     r"by \$-0\.\d{4}\$ and \$-0\.(\d{4})\$ in \$p\$",
     "construct_nulls_legality_aware.json",
     lambda doc: _arc_dp_1e4(doc, "ARC-Challenge"),
     "E-CAL dp_aware_minus_blind = -0.005213 -> 0.0052 at 4dp"),
    # The matched-depth contrast added to 05_analysis.tex 2026-08-17. It is the datum that
    # separates "depth >= 14" from "depth >= 14 AND prune-then-heal": at the SAME retained
    # depth, healed clears 4/5 while truncate-only clears 0/15. Registered because this gate
    # prints "counts not listed here are unchecked", and that sentence is the only place the
    # paper concedes depth alone is insufficient -- an unguarded number there could drift
    # into contradicting the very confound disclosure it exists to support.
    ("depth-14 healed above-floor", "05_analysis.tex",
     r"retained depth of 14, the healed arm clears (\w+) of its 5 floors",
     "designated_damaged_denominators.json",
     lambda doc: doc["MONOTONE_DEPTH_THRESHOLD"]["by_retained_depth"]["14_heal"]["above"],
     "by_retained_depth.14_heal.above = 4 of n=5"),
    ("depth-14 truncate-only cells", "05_analysis.tex",
     r"while the (\w+) truncate-only cells at that depth clear none",
     "designated_damaged_denominators.json",
     lambda doc: doc["MONOTONE_DEPTH_THRESHOLD"]["by_retained_depth"]["14_trunc"]["n"],
     "by_retained_depth.14_trunc.n = 15; the 'clear none' half is asserted by the next "
     "claim so a nonzero above-count cannot hide behind a correct denominator"),
]


def _arc_row(doc, name):
    for r in doc["rows"]:
        if str(r.get("construct", "")).startswith(name):
            return r
    raise KeyError(name)


def _arc_sigma(doc, name):
    """|dp_sigma| to one decimal, as an integer of tenths so the gate compares ints."""
    s = abs(_arc_row(doc, name)["directionality"]["dp_sigma"])
    return round(s, 1)


def _arc_p(doc, name):
    return _arc_row(doc, name)["p_aware"]


def _arc_dp_1e4(doc, name):
    """|dp_aware_minus_blind| in ten-thousandths, matching the prose's 4 decimals."""
    dp = abs(_arc_row(doc, name)["directionality"]["dp_aware_minus_blind"])
    return int(round(dp * 10000))


NEG_CONTROL_TASK = "winogrande"


def n_constant_emitters(doc, exclude_negative_control):
    """Cells that are literal constant emitters, by the source verdict's own criterion.

    modal_pred_share >= 0.99 AND |acc - marginal(emitted letter)| < 1e-6.

    The marginal is recomputed from `task_letter_nulls[task].gold_letter_marginal`
    rather than read from any summary, and `base` rungs are excluded because the claim
    is about DAMAGED cells.
    """
    tln = doc.get("task_letter_nulls", {})

    def marginal(task, letter):
        g = (tln.get(task) or {}).get("gold_letter_marginal") or {}
        n = sum(g.values())
        return (g.get(letter, 0) / n) if n else None

    n = 0
    for _fam, fv in (doc.get("families") or {}).items():
        for rung, rv in (fv.get("rungs") or {}).items():
            if rung == "base":
                continue
            for task, tv in (rv.get("tasks") or {}).items():
                if exclude_negative_control and task == NEG_CONTROL_TASK:
                    continue
                it = (tv.get("interfaces") or {}).get("letter")
                if not it:
                    continue
                hist = it.get("pred_hist") or {}
                acc = it.get("acc")
                if not hist or acc is None:
                    continue
                total = sum(hist.values())
                emitted = max(hist, key=hist.get)
                share = hist[emitted] / total
                m = marginal(task, emitted)
                if m is not None and share >= 0.99 and abs(acc - m) < 1e-6:
                    n += 1
    return n


def n_cells_in_tab_mmlupro():
    """Count the cells tab_mmlupro actually prints.

    NOT a hardcoded 17. A hardcoded arm list inside an emitter is precisely how this
    paper's denominator defect happened, so the count is parsed from the table.

    One row is a RANGE row: 'Llama-3 & \\texttt{k14--k8}' collapses four rungs onto one
    printed line for space. A naive row count gives 14, not 17, and would make this gate
    assert a third wrong number. Range rows are expanded from the rung span, stepping by
    2 (k14, k12, k10, k8).
    """
    path = SECTIONS / "tab_mmlupro.tex"
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("%") or s.startswith("\\") or "&" not in s:
            continue
        cols = [c.strip() for c in s.split("&")]
        if cols[0] == "Family":
            continue
        arm = cols[1] if len(cols) > 1 else ""
        m = re.search(r"k(\d+)--k(\d+)", arm)
        if m:
            hi, lo = int(m.group(1)), int(m.group(2))
            n += len(range(lo, hi + 1, 2))
        else:
            n += 1
    return n

WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
         "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
         "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
         "fifteen": 15, "sixteen": 16, "seventeen": 17}


def as_int(token):
    """Parse a prose numeral into a comparable number.

    Named as_int for history; it now also accepts ONE-DECIMAL values, because the ARC
    directionality claims are sigmas (-3.3, -7.3) and this gate was built for integers.
    Returning a float there is safe: the recomputations for those claims return
    `round(x, 1)`, so both sides carry exactly one decimal and compare exactly. Anything
    with more precision than the prose prints would make an exact comparison meaningless,
    which is why this deliberately does NOT accept arbitrary floats.
    """
    token = token.strip().lower()
    if token.isdigit():
        return int(token)
    if re.fullmatch(r"\d+\.\d", token):
        return float(token)
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
        # ev_rel is None for claims recomputed from a shipped .tex table rather than
        # from an evidence JSON; those recomputations take no document argument.
        # The source label is derived from the recomputation's own name rather than
        # hardcoded to tab_nulls: with more than one .tex-derived claim, a fixed label
        # misattributes the evidence. Measured 2026-08-16 -- the tab_mmlupro claim's
        # failure message named sections/tab_nulls.tex, which is not where its number
        # comes from, and a reviewer chasing that pointer would find nothing.
        if ev_rel is None:
            doc = None
            fn = getattr(recompute, "__name__", "")
            if "tab_mmlupro" in fn:
                source = "sections/tab_mmlupro.tex"
            elif "tab_mmlupro" in (note or ""):
                source = "sections/tab_mmlupro.tex"
            else:
                source = "sections/tab_nulls.tex"
        else:
            try:
                doc = load(ev_rel)
            except FileNotFoundError as exc:
                print(f"CANNOT READ evidence: {exc}")
                return 3
            source = ev_rel
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
                            f"{source} says {expected}. {note}")
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
