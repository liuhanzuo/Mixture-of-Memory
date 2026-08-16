#!/usr/bin/env python3
r"""paperC E3: verify every prose/`.tex` statement of a construct null against the evidence.

Why this exists
---------------
`paperC/SUBMISSION_GAP_AUDIT.md` gap **E3**: no `.tex`-level integrity binding
between evidence and prose. `emit_tab_construct_nulls.py` supplies the *generated*
half of that binding. This script supplies the *auditing* half: it reads
`README.md` and every `sections/*.tex`, finds the places that state one of the
construct nulls, and checks each against the full-precision value in the evidence
JSON. It exits non-zero on any mismatch, so a mistyped number becomes a build
failure rather than something a reader has to notice.

E3 explains why this is not hypothetical: this direction already shipped 10
truncated cells *with the summaries already written* (README:401-420). A hand-typed
number that reads as finished is the failure mode, and prose is where paperC keeps
most of its numbers.

The comparison rule, and the mistake it is built to avoid
--------------------------------------------------------
E2 (closed 2026-08-15, commit `f5d52b2`, `evidence/e2_single_source_assertion.json`)
was wrong twice about "this number has no machine-readable home", both times
because a naive grep for the README's **rounded** form misses a value stored at
**full precision**: `0.2689` is `rows[2].floor = 0.268908`, `0.6217` is
`rows[8].floor = 0.621713`. So no comparison here is a string equality. A prose
literal is accepted iff it lies within a half-unit of its own last decimal place
of the stored value -- `0.2689`, `0.268908` and `0.269` are all correct statements
of the same number, and the check knows that.

What counts as a site (two tiers, and what is deliberately excluded)
-------------------------------------------------------------------
Every site needs at least `MIN_DP` decimals. Coarser literals are excluded and
**counted**, not dropped silently: `0.25` is a correct rounding of four different
stored chance lines (0.250000, 0.250156, 0.250161, and 0.25 itself), so accepting
it would manufacture agreement rather than test it.

  tier 1, `near_miss`
      A literal within `NEAR_ABS` of a stored value, on a line that names the
      construct and carries a floor/null marker (`floor`, `always-`,
      `best constant`, `null`, ...). Catches the single-digit-typo class.
      Distinct quantities that merely sit nearby are not flagged: the calibration
      moments `E[f-hat]` and `q95` live 0.003--0.02 from their row's floor, far
      outside `NEAR_ABS`.

  tier 2, `declaration`
      A literal preceded by an explicit declaration form -- ``always-D =``,
      ``floor is``, ``longest-option split =``. Here the text *asserts* the
      number IS that construct's null, so ANY disagreement is a mismatch at any
      magnitude. Tier 1 alone cannot do this: mistyping BoolQ's `0.6217` as
      `0.6127` lands outside `NEAR_ABS` and reads as "some other quantity"
      (measured -- that mutation escapes tier 1 and is caught by tier 2).
      The subject is the construct named nearest to the left of the literal, with
      one line of lookback for the README's hard wraps; both rules were chosen by
      measurement, see the comments at the tier-2 block.

Coverage is reported separately from correctness: `n_targets_uncovered` counts
floors that no prose site states at all. That is a gap in the paper's reporting,
not a contradiction of the evidence, so by default it does not fail the run;
`--strict` makes it fail.

Sources (read-only; this script never modifies an evidence file)
  evidence/floor_winners_curse_calibration.json    9 letter nulls + chance lines
  evidence/construct_nulls_length_unit.json        OpenBookQA content null, 2 units
  evidence/second_mc_benchmark/...json             Winogrande control floor
  evidence/second_mc_benchmark_crossfamily/...json pooled five-task floor
  ../proposal/active/A01-.../gate3_content_null_conventions.json
      ^ OUTSIDE paperC. The MMLU *content* null `0.2845`, quoted at README.md:23,
        has no paperC-side machine-readable home; see `provenance_gaps` in the
        emitted JSON and the note at that loader.

CPU only. No GPU, no model, no network.

Usage:
  python paperC/code/check_prose_vs_evidence.py
  python paperC/code/check_prose_vs_evidence.py --strict
  python paperC/code/check_prose_vs_evidence.py --out paperC/evidence/prose_vs_evidence_check.json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
REPO = os.path.dirname(PAPER)

CALIB = os.path.join(PAPER, "evidence", "floor_winners_curse_calibration.json")
LENUNIT = os.path.join(PAPER, "evidence", "construct_nulls_length_unit.json")
GATE2 = os.path.join(PAPER, "evidence", "second_mc_benchmark",
                     "gate2_letter_content_nulls.json")
GATE2X = os.path.join(PAPER, "evidence", "second_mc_benchmark_crossfamily",
                      "gate2_crossfamily_nulls.json")
# Outside paperC on purpose; see the provenance-gap note in build_targets().
A01_GATE3 = os.path.join(
    REPO, "proposal", "active", "A01-null-calibration-methodology", "evidence",
    "gate3_content_null_conventions.json")
DEFAULT_OUT = os.path.join(PAPER, "evidence", "prose_vs_evidence_check.json")

MIN_DP = 3         # 0.25 / 0.50 / 0.20 carry too little information to test
NEAR_ABS = 1e-3    # "plainly meant to be this number" band for typo detection

# A numeric literal, not part of a longer dotted token (so 1.0.0 and 10^{-5} are
# not mistaken for measurements).
NUM = re.compile(r"(?<![\w.])(\d+\.\d+)(?![\d.])")

# A literal immediately preceded by one of these labels is a different quantity
# that merely happens to land near a chance line, and is excluded before any
# marker test. Measured need: `sections/05_analysis.tex:41` writes a p-value
# `($p=0.1002$)` on a line that also says "floor" and "null", which the
# line-level marker test alone reports as a near-miss of MMLU-Pro's naive chance
# line 0.100000. It is a p-value, not a chance line.
EXCL_PREFIX = re.compile(
    r"(?:"
    r"\$?p\$?\s*[=<>]\s*|"                 # p=, $p$<, p >
    r"p_?(?:one_sided|value|val)\s*[=<>]\s*|"
    r"\\?alpha\s*[=<>]\s*|"
    r"(?:CI|ci)\s*9?5?\s*[=:]?\s*\[?|"
    r"[Ss]eed\s*[=:]?\s*|"
    r"(?:version|schema_version|sha256)\s*[=:~]?\s*"
    r")$"
)

FLOOR_MARKERS = (
    "floor", "always-", "always ", "best constant", "best-constant",
    "best_constant", "null", "constant emitter", "longest-option",
    "longest option",
)

# Construct -> regex that must match the line for a literal on it to be a site.
# MMLU must not swallow MMLU-Pro.
NAME_RE = {
    "MMLU-Pro letter, naive": r"MMLU[-\s]?Pro|mmlu[-_\\\s]?pro",
    "MMLU-Pro letter, item-avg.": r"MMLU[-\s]?Pro|mmlu[-_\\\s]?pro",
    "MMLU letter": r"MMLU(?![-\s]?Pro)|mmlu(?![-_\\\s]?pro)",
    "OpenBookQA letter": r"OpenBookQA|openbookqa|OBQA|obqa",
    "ARC-Easy letter": r"ARC[-\s]?Easy|arc[-_\\\s]?easy",
    "ARC-Challenge letter": r"ARC[-\s]?Challenge|arc[-_\\\s]?ch",
    "CommonsenseQA letter": r"CommonsenseQA|commonsense[-_\\\s]?qa|CSQA|csqa",
    "PIQA letter": r"PIQA|piqa",
    "BoolQ": r"BoolQ|boolq",
    "OpenBookQA content": r"OpenBookQA|openbookqa|OBQA|obqa",
    "MMLU content": r"MMLU(?![-\s]?Pro)|mmlu(?![-_\\\s]?pro)",
    "Winogrande letter, control": r"Winogrande|winogrande|WinoGrande",
    "Pooled five-task letter": r"pooled|Pooled",
}

# TIER 2 -- the declaration form.
#
# The near-miss tier above catches single-digit typos, which is the realistic
# failure mode, but by construction it cannot catch a LARGE error: mistyping
# BoolQ's `0.6217` as `0.6127` lands outside NEAR_ABS and reads as "some other
# quantity". Measured, not assumed: that mutation escapes tier 1.
#
# This tier closes it by matching the *syntax of a declaration* -- "always-D =",
# "floor is", "longest-option split =" -- immediately left of the literal. On
# such a line the number is definitionally the construct's null, so ANY
# disagreement is an error regardless of magnitude. The `always-X` form needs no
# copula ("always-B `0.265358`" is already a declaration); the generic words do,
# so that "above floor by $+1.538$" (a delta, not a floor) does not match.
DECL_FORM = re.compile(
    r"(?:always[-\s]?[A-J]\b(?:\s*(?:=|is|of|:))?|"
    r"(?:best[-\s]constant|best constant|floor|"
    r"longest[-\s]option(?:\s+split)?)\s*(?:=|is|of|:))"
    r"(?:\s*\\?(?:texttt|emph|textbf)?\{?)?\s*[`'\"$\\ ]{0,6}$"
)


def dp(literal: str) -> int:
    return len(literal.split(".")[1]) if "." in literal else 0


def is_rounding_of(literal: str, value: float) -> bool:
    """True iff `literal` is a correct fixed-point rounding of `value`."""
    half = 0.5 * (10.0 ** -dp(literal))
    return abs(float(literal) - value) <= half * (1.0 + 1e-9)


def sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def build_targets() -> tuple[list[dict], dict]:
    """Every (construct, quantity) pair whose value prose may state.

    Values are read from the evidence JSONs. Nothing here is a literal.
    """
    with open(CALIB, encoding="utf-8") as f:
        calib = json.load(f)
    targets: list[dict] = []
    seen: set[tuple] = set()

    def add(construct, quantity, value, source, must_cover):
        key = (construct, quantity, round(float(value), 12))
        if key in seen:
            return
        seen.add(key)
        targets.append({
            "construct": construct,
            "quantity": quantity,
            "value": float(value),
            "source": source,
            "must_cover": bool(must_cover),
        })

    for r in calib["rows"]:
        c = r["construct"]
        add(c, "floor", r["floor"], "floor_winners_curse_calibration.json", True)
        add(c, "chance", r["chance"], "floor_winners_curse_calibration.json", False)

    sources = {
        "floor_winners_curse_calibration.json": {
            "path": os.path.relpath(CALIB, REPO),
            "sha256": sha256(CALIB),
            "schema_version": calib.get("schema_version"),
            "seed": calib.get("seed"),
            "n_draws": calib.get("n_draws"),
        }
    }

    # Secondary: the OpenBookQA content null in both length units (E1's file).
    if os.path.exists(LENUNIT):
        with open(LENUNIT, encoding="utf-8") as f:
            lu = json.load(f)
        obqa = (lu.get("tasks") or {}).get("openbookqa") or {}
        char = (obqa.get("char") or {}).get("split")
        tok = ((obqa.get("token_from_per_item_records") or {}) or {}).get("split")
        if char is not None:
            add("OpenBookQA content", "longest_split_char", char,
                "construct_nulls_length_unit.json", True)
        if tok is not None:
            add("OpenBookQA content", "longest_split_token", tok,
                "construct_nulls_length_unit.json", True)
        sources["construct_nulls_length_unit.json"] = {
            "path": os.path.relpath(LENUNIT, REPO),
            "sha256": sha256(LENUNIT),
        }

    # Secondary: the Winogrande negative-control floor and the pooled five-task
    # floor. Both are quoted in the README and both are arm-invariant in their
    # source (measured: 1 unique value across 6 arms / 15 family-rungs), so a
    # single target per quantity is well defined.
    if os.path.exists(GATE2):
        with open(GATE2, encoding="utf-8") as f:
            g2 = json.load(f)
        wino = (g2.get("rollup_letter_floor") or {}).get("winogrande") or {}
        if wino.get("letter_null") is not None:
            add("Winogrande letter, control", "floor", wino["letter_null"],
                "second_mc_benchmark/gate2_letter_content_nulls.json", True)
        sources["second_mc_benchmark/gate2_letter_content_nulls.json"] = {
            "path": os.path.relpath(GATE2, REPO), "sha256": sha256(GATE2)}
    if os.path.exists(GATE2X):
        with open(GATE2X, encoding="utf-8") as f:
            g2x = json.load(f)
        pooled = {
            round(float(rung["pooled_floor"]), 12)
            for fam in ((g2x.get("pooled_across_tasks_letter_floor") or {})
                        .get("by_family") or {}).values()
            for rung in fam.values() if rung.get("pooled_floor") is not None
        }
        if len(pooled) == 1:
            add("Pooled five-task letter", "floor", pooled.pop(),
                "second_mc_benchmark_crossfamily/gate2_crossfamily_nulls.json",
                False)
        sources["second_mc_benchmark_crossfamily/gate2_crossfamily_nulls.json"] = {
            "path": os.path.relpath(GATE2X, REPO), "sha256": sha256(GATE2X),
            "pooled_floor_unique_values": 1 if not pooled else "NOT_UNIQUE"}

    # The MMLU *content* null (longest-option split, continuation-token unit).
    # ⚠ PROVENANCE GAP, measured 2026-08-15: this value has NO paperC-side
    # machine-readable home. It is quoted in README.md:23 and in
    # tcodex_out/EVIDENCE_PACK.md, but a full walk of every float in every
    # paperC/evidence/**/*.json finds nothing within 5e-6 of it. Its only
    # machine-readable home in the repository is A01's proposal evidence
    # (invariant across all 6 arms there), which is why it is loaded from
    # OUTSIDE paperC and flagged below rather than silently accepted. This is
    # exactly the fragmentation E2 was raised about, for a value E2 did not
    # cover; it is reported, not repaired, because relocating evidence is an
    # editorial decision.
    if os.path.exists(A01_GATE3):
        with open(A01_GATE3, encoding="utf-8") as f:
            g3 = json.load(f)
        vals = {
            round(float((arm.get("longest_option_floor_by_conv") or {})["split"]), 12)
            for arm in (g3.get("arms") or {}).values()
            if (arm.get("longest_option_floor_by_conv") or {}).get("split") is not None
        }
        if len(vals) == 1:
            add("MMLU content", "longest_split_token", vals.pop(),
                "A01/gate3_content_null_conventions.json (OUTSIDE paperC)", True)
            sources["A01/gate3_content_null_conventions.json"] = {
                "path": os.path.relpath(A01_GATE3, REPO),
                "sha256": sha256(A01_GATE3),
                "WARNING": "not a paperC evidence file. The MMLU content null "
                           "has no paperC-side machine-readable home; see "
                           "provenance_gaps.",
            }
    return targets, sources


def prose_files() -> list[str]:
    files = [os.path.join(PAPER, "README.md")]
    files += sorted(glob.glob(os.path.join(PAPER, "sections", "*.tex")))
    return [p for p in files if os.path.exists(p)]


def scan(targets: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Return (ok_sites, mismatch_sites, skipped_low_precision).

    Two tiers, both emitting into the same three lists and tagged by `tier`:
      near_miss    a literal within NEAR_ABS of a stored value on a line that
                   names the construct and carries a floor/null marker.
      declaration  a literal preceded by an explicit declaration form
                   ("always-D =", "floor is", "longest-option split ="). Here the
                   number IS the construct's null by syntax, so any disagreement
                   is a mismatch at any magnitude.
    """
    compiled = [(t, re.compile(NAME_RE[t["construct"]])) for t in targets]
    ok: list[dict] = []
    bad: list[dict] = []
    skipped: list[dict] = []

    for path in prose_files():
        rel = os.path.relpath(path, REPO)
        with open(path, encoding="utf-8") as f:
            all_lines = f.read().split("\n")
        for lineno, line in enumerate(all_lines, 1):
            if line.lstrip().startswith("%"):
                continue          # LaTeX comment (incl. the generated header)
            lits = [(m.group(1), m.start())
                    for m in NUM.finditer(line)
                    if not EXCL_PREFIX.search(line[:m.start()])]
            if not lits:
                continue
            low = line.lower()
            marker = any(m in low for m in FLOOR_MARKERS)

            def rec_for(t, lit, v, diff, tier):
                return {
                    "file": rel, "line": lineno, "tier": tier,
                    "construct": t["construct"], "quantity": t["quantity"],
                    "prose_literal": lit, "prose_value": float(lit),
                    "evidence_value": v, "evidence_source": t["source"],
                    "abs_diff": diff, "decimals": dp(lit),
                    "context": line.strip()[:220],
                }

            # ---- tier 2: declaration form ------------------------------
            # Done first so a declaration site is never also counted as a
            # near-miss site.
            #
            # Disambiguation is by the construct name that ends NEAREST to
            # the left of the literal, not by "the line names exactly one
            # construct". Measured need: README.md:285 reads
            #   ``(5-way, always-B `0.208845`), **piqa** ... **winogrande**''
            # -- three constructs on one line, each with its own declared
            # floor. A line-level uniqueness rule mis-pairs PIQA's 0.504897
            # with Winogrande's 0.504341 and reports a mismatch that is not
            # one. Nearest-name-to-the-left pairs every such site correctly
            # (measured: 15/15 resolved sites, 0 mismatches, on the current
            # text).
            decl_seen: set[tuple] = set()
            for lit, pos in lits:
                if dp(lit) < MIN_DP or not DECL_FORM.search(line[:pos]):
                    continue

                def nearest(hay: str, limit: int) -> list[dict]:
                    c: list[dict] = []
                    end = -1
                    for t, rx in compiled:
                        for mo in rx.finditer(hay):
                            if mo.end() > limit:
                                break
                            if mo.end() > end:
                                c, end = [t], mo.end()
                            elif mo.end() == end:
                                c.append(t)
                    return c

                cands = nearest(line, pos)
                subject = "same-line"
                if not cands and lineno >= 2:
                    # One line of lookback, and only one: the README hard-wraps
                    # its prose, so a declaration can be separated from its
                    # subject by a soft wrap (README.md:285-286 names
                    # winogrande at the end of 285 and declares its floor at
                    # the start of 286). Depth was swept: 0 leaves 5 sites
                    # unresolved, 1 resolves 3 of them with 0 mismatches, and
                    # 2 starts pairing a declaration with a heading 5 lines up.
                    prev = all_lines[lineno - 2]
                    if prev.strip() and not prev.lstrip().startswith("%"):
                        cands = nearest(prev, len(prev))
                        subject = "prev-line"
                decl_seen.add((lit, pos))
                if not cands:
                    # No construct named to the left or on the previous line:
                    # the declaration's subject is in a table cell or further
                    # up. Recorded so the count is visible, not dropped.
                    skipped.append({
                        "file": rel, "line": lineno, "tier": "declaration",
                        "construct": None, "quantity": None,
                        "prose_literal": lit, "prose_value": float(lit),
                        "evidence_value": None, "evidence_source": None,
                        "abs_diff": None, "decimals": dp(lit),
                        "context": line.strip()[:220],
                        "reason": "declaration form, but no construct name "
                                  "occurs to the left of the literal nor on "
                                  "the previous line, so the subject is "
                                  "unresolvable; not counted either way",
                    })
                    continue
                hit = next((t for t in cands
                            if is_rounding_of(lit, t["value"])), None)
                if hit is not None:
                    r = rec_for(hit, lit, hit["value"],
                                abs(float(lit) - hit["value"]), "declaration")
                    r["subject_from"] = subject
                    ok.append(r)
                else:
                    best = min(cands,
                               key=lambda t: abs(float(lit) - t["value"]))
                    r = rec_for(best, lit, best["value"],
                                abs(float(lit) - best["value"]),
                                "declaration")
                    r["subject_from"] = subject
                    r["reason"] = (
                        "the text declares this literal to BE the null of "
                        "the construct named nearest to its left, but it is "
                        "not a correct rounding of any evidence value for "
                        "that construct at %d dp" % dp(lit))
                    r["candidate_values"] = {
                        t["quantity"]: t["value"] for t in cands}
                    bad.append(r)
                decl_seen.add((lit, pos))

            # ---- tier 1: near-miss ------------------------------------
            for t, rx in compiled:
                if not rx.search(line):
                    continue
                v = t["value"]
                for lit, pos in lits:
                    if (lit, pos) in decl_seen:
                        continue
                    diff = abs(float(lit) - v)
                    if diff > max(NEAR_ABS, 0.5 * 10.0 ** -dp(lit)):
                        continue      # not about this quantity at all
                    rec = rec_for(t, lit, v, diff, "near_miss")
                    if dp(lit) < MIN_DP:
                        rec["reason"] = (
                            f"fewer than MIN_DP={MIN_DP} decimals; too "
                            f"coarse to test (it also rounds to other "
                            f"stored values)")
                        skipped.append(rec)
                        continue
                    if is_rounding_of(lit, v):
                        ok.append(rec)
                    elif marker:
                        rec["reason"] = (
                            "within NEAR_ABS of the stored value and on a "
                            "line carrying a floor/null marker, but not a "
                            "correct rounding of it at %d dp" % dp(lit))
                        bad.append(rec)
                    else:
                        rec["reason"] = (
                            "near the stored value but the line carries no "
                            "floor/null marker, so it is probably a "
                            "different quantity; not counted either way")
                        skipped.append(rec)
    return ok, bad, skipped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--strict", action="store_true",
                    help="also fail when a must-cover target appears nowhere")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    targets, sources = build_targets()
    ok, bad, skipped = scan(targets)

    covered = {(r["construct"], r["quantity"]) for r in ok}
    uncovered = [
        {"construct": t["construct"], "quantity": t["quantity"],
         "value": t["value"], "source": t["source"]}
        for t in targets
        if t["must_cover"] and (t["construct"], t["quantity"]) not in covered
    ]

    n_checked = len(ok) + len(bad)
    payload = {
        "what": "prose/.tex statements of each construct null, checked against "
                "the full-precision value in the evidence JSON",
        "closes": "paperC/SUBMISSION_GAP_AUDIT.md gap E3 (auditing half; the "
                  "generated half is paperC/code/emit_tab_construct_nulls.py "
                  "-> paperC/sections/tab_construct_nulls.tex)",
        "comparison_rule": "a prose literal is accepted iff |literal - stored| "
                           "<= 0.5 * 10^-decimals(literal); never string "
                           "equality. E2's documented error class was a naive "
                           "grep for the README's ROUNDED form missing a value "
                           "stored at FULL precision (0.2689 is 0.268908, "
                           "0.6217 is 0.621713).",
        "site_rule": "two tiers. TIER 1 'near_miss': a literal within "
                     "NEAR_ABS of a stored value, on a line naming the construct "
                     "and carrying a floor/null marker, at >= MIN_DP decimals, "
                     "not preceded by a p-value / CI / seed / version label. "
                     "TIER 2 'declaration': a literal preceded by an explicit "
                     "declaration form (always-D =, floor is, longest-option "
                     "split =), whose subject is the construct named nearest to "
                     "its left with one line of lookback for hard wraps; on such "
                     "a site ANY disagreement is a mismatch at any magnitude, "
                     "which is what catches errors too large for tier 1.",
        "declaration_form_regex": DECL_FORM.pattern,
        "excluded_label_prefixes": EXCL_PREFIX.pattern,
        "min_decimals": MIN_DP,
        "near_miss_abs_window": NEAR_ABS,
        "floor_markers": list(FLOOR_MARKERS),
        "gpu_used": "NONE",
        "sources": sources,
        "provenance_gaps": [
            {
                "value_as_quoted": "0.2845",
                "quantity": "MMLU content null, longest-option split, "
                            "continuation-token unit, OLMo-2 tokenizer",
                "quoted_at": ["paperC/README.md:23",
                              "paperC/tcodex_out/EVIDENCE_PACK.md:13"],
                "finding": "has NO paperC-side machine-readable home. Measured "
                           "2026-08-15 by walking every float in every "
                           "paperC/evidence/**/*.json: nothing within 5e-6. Its "
                           "only machine-readable home in the repository is "
                           "proposal/active/A01-null-calibration-methodology/"
                           "evidence/gate3_content_null_conventions.json, where "
                           "it is invariant across all 6 arms at "
                           "0.28445022076627263.",
                "action_taken": "REPORTED, NOT REPAIRED. The value itself is "
                                "correct -- README.md:23 states 0.2845 and the "
                                "A01 evidence says 0.28445022076627263, which "
                                "rounds to it. So this is a provenance "
                                "fragmentation finding, not a wrong number, and "
                                "it is not counted as a mismatch. Relocating or "
                                "copying evidence into paperC/evidence/ is an "
                                "editorial decision for the paper's owner.",
                "relation_to_E2": "same class as E2 (fragmentation across four "
                                  "sources plus prose), for a value E2's 9-row "
                                  "headline table does not cover. E2 is closed "
                                  "and is not reopened by this.",
            }
        ],
        "files_scanned": [os.path.relpath(p, REPO) for p in prose_files()],
        "n_targets": len(targets),
        "n_checked": n_checked,
        "n_ok": len(ok),
        "n_mismatch": len(bad),
        "n_skipped_low_precision_or_unmarked": len(skipped),
        "n_targets_uncovered": len(uncovered),
        "n_by_tier": {
            tier: {
                "ok": sum(1 for r in ok if r.get("tier") == tier),
                "mismatch": sum(1 for r in bad if r.get("tier") == tier),
                "skipped": sum(1 for r in skipped if r.get("tier") == tier),
            }
            for tier in ("near_miss", "declaration")
        },
        "targets_uncovered": uncovered,
        "mismatches": bad,
        "ok_sites": ok,
        "skipped_sites": skipped,
        "verdict": "PASS" if not bad else "FAIL",
    }

    if not args.no_write:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1, sort_keys=True)
            f.write("\n")

    print(f"[scan] {len(payload['files_scanned'])} files, "
          f"{len(targets)} targets")
    print(f"[result] n_checked={n_checked} n_ok={len(ok)} "
          f"n_mismatch={len(bad)} "
          f"n_skipped={len(skipped)} n_uncovered={len(uncovered)}")
    for r in bad:
        print(f"  [MISMATCH] {r['file']}:{r['line']} {r['construct']} "
              f"{r['quantity']}: prose {r['prose_literal']} vs evidence "
              f"{r['evidence_value']:.6f} (|d|={r['abs_diff']:.3e})")
        print(f"             {r['context']}")
    for u in uncovered:
        print(f"  [UNCOVERED] {u['construct']} {u['quantity']}="
              f"{u['value']:.6f} appears in no scanned file")
    if not args.no_write:
        print(f"[done] wrote {args.out}")

    if bad:
        return 1
    if args.strict and uncovered:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
