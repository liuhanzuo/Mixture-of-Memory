#!/usr/bin/env python3
"""SELF-CORRECTION to section 13.5, written the same session that wrote 13.5.

WHAT I GOT WRONG
================
Section 13.5, as written earlier this session, claimed the ``Delta_U`` PASS
threshold "is written three non-equivalent ways" and tabulated three rows:
``next_gate.decidable_outcome``, ``kill_gate`` K2, and ``RELATED_WORK.md`` section 8.

I then enumerated the written forms as actual predicates over ``(point, lo, hi)``
and measured how many DISTINCT predicates they are. Two errors:

1. **K2 is not a PASS spelling.** ``hi < 5.0`` is a KILL-side clause. Putting it in
   a table of PASS thresholds compares a kill condition against pass conditions --
   a units error of exactly the kind recorded in
   ``memory/an-informational-nonzero-rc-hides-real-defects.md`` ("my first version
   of the correction again mixed row/construct units").
2. **The count is two, not three.** Measured over a 108-point grid of
   ``(point, lo, width)``, the four PASS-side forms collapse into TWO predicate
   classes:
     * ``lo > 5.0`` -- ``next_gate.decidable_outcome``, prereg 5.6, section 3.1
       (THREE CONCORDANT SOURCES)
     * ``point > 5.0 and lo > 0`` -- ``RELATED_WORK.md`` section 8 blockquote (LONE OUTLIER)
   I had also failed to check prereg 5.6, which is the pre-registration of record
   and agrees with ``next_gate``. Checking it is what collapsed the count.

The underlying inconsistency IS real and survives the correction, but it is
smaller and better localised than I first wrote: section 8's blockquote is the
single outlier and it is strictly WEAKER than the pre-registered condition, so a
result can satisfy section 8's sentence and still fail the gate as pre-registered.
Worked case (verified by executing the predicates, not by reading them):
``Delta_U = +6.0 pp``, CI ``[+1.0, +11.0]`` -> section 8 says PASS; ``next_gate`` /
prereg 5.6 / section 3.1 all say FAIL; K2 does not fire. Since section 8 is the
"safe residual claim" sentence a reviewer actually reads, that matters.

I also retract a second sentence from 13.5: I called the perturbed case "the
proposal has no verdict" and implied it was itself a defect. It is not. A
two-sided gate with an inconclusive middle band is normal and is exactly what
``kill_gate`` describes (KILL iff ALL THREE; PASS on its own condition; neither ->
inconclusive). The defect is the section 8 / prereg disagreement, nothing more.

MECHANICS: insert-only, same machine-checked proof as the two prior scripts.
Section 13.5's original text is NOT deleted -- it is marked ``RETRACTED IN PART``
in place with a pointer to section 13.7, which carries the corrected version. A
wrong sentence that has been on disk is provenance
(``memory/append-only-records-outlive-their-own-truth.md``).

0 GPU, 0 ssh. Run once.
"""
import hashlib
import os
import sys

ROOT = "/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory"
DOC = os.path.join(ROOT, "proposal/backlog/B08-memory-applications/RELATED_WORK.md")

SHA_BEFORE = "274fa83c9cbfeb2f4781885fdb303a9729628fa5f3ea4f0825e6605dfecf9843"
BYTES_BEFORE = 74838

# Three in-place retraction markers, at the three places 13.5's wrong count is
# stated. Each anchor asserted unique before any write.
ANCHOR_1355_HDR = ("### 13.5 ⚠ NEW DEFECT FOUND WHILE PLACING EDIT 3: the `Δ_U` "
                   "PASS threshold has three non-equivalent spellings")
MARK_1355_HDR = (
    "\n\n⛔ **RETRACTED IN PART, SAME SESSION — see §13.7 for the corrected"
    " version.** The count below is WRONG: it is **two** distinct PASS predicates,"
    " not three, and `kill_gate` K2 is a **kill-side** clause that does not belong"
    " in a table of PASS thresholds. The underlying inconsistency is real but is"
    " narrower: §8's blockquote is a **lone outlier** against three concordant"
    " sources. Original text retained below unedited."
)

ANCHOR_S31 = ("⚠ §13.5: this line's `Δ_U` PASS threshold is ALSO one of three"
              " non-equivalent spellings of it in the proposal — do not score"
              " against this line.**")
MARK_S31 = (
    " ⛔ **§13.7 CORRECTS THE PRECEDING SENTENCE: this line is NOT an outlier."
    " It agrees with `next_gate.decidable_outcome` and prereg 5.6 (all three ="
    " `CI lower bound > +5.0 pp`). The lone outlier is §8's blockquote. Scoring"
    " against THIS line is fine; it is the majority reading.**"
)

ANCHOR_S8 = ("⚠ **§13.5 — the `Δ_U` PASS threshold is written three non-equivalent"
             " ways in this\nproposal. Do not score until the owner picks one.**"
             " See §13.5.")
MARK_S8 = (
    "\n\n⛔ **§13.7 CORRECTS THE LINE ABOVE, AND THE CORRECTION LANDS ON THIS VERY"
    " BLOCKQUOTE.** It is **two** spellings, not three — and **the blockquote"
    " directly above is the outlier**. It states PASS as *\"exceeds +5.0 pp with a"
    " 95% paired-bootstrap CI entirely above 0\"*, i.e. `point > +5.0 AND lo > 0`."
    " The pre-registration of record (`B08_LEG1_GATE_PREREG.md` §5.6), the"
    " machine-read `STATUS.json.next_gate.decidable_outcome`, and §3.1 all say"
    " `CI entirely above +5.0 pp`, i.e. `lo > +5.0` — **strictly stronger**. So this"
    " blockquote can call a result a PASS that the pre-registration calls a FAIL"
    " (worked case in §13.7). **Score against prereg 5.6, not against this"
    " blockquote, until the owner reconciles them.**"
)

SECTION_13_7 = """

### 13.7 SELF-CORRECTION to §13.5 (same session): it is TWO spellings, not three, and §8 is the outlier

§13.5 was written from *reading* the threshold sentences. I then *executed* them as
predicates over `(point, lo, hi)` on a 108-point grid, and the reading was wrong in
two ways. Both corrections make the defect **smaller and better localised**, not
larger — recorded because a self-correction that shrinks my own finding is exactly
the kind that does not get written down
(`memory/state-direction-only-for-rows-you-computed.md`).

**Error 1 — a units error.** `kill_gate` K2 (`Δ_U` 95% CI **upper** bound < +5.0 pp)
is a **KILL-side** clause. §13.5 tabulated it as a third PASS threshold. Comparing a
kill condition with pass conditions is a category error, and "K2 does not fire" is
**not** the same predicate as "the gate passes" — it only blocks the ALL-THREE KILL.

**Error 2 — the count, and the missing source.** §13.5 never checked
`B08_LEG1_GATE_PREREG.md` §5.6, which is the **pre-registration of record**. It reads
*"`ΔU` CI entirely **above +5.0 pp**"* — agreeing with `next_gate.decidable_outcome`
and with §3.1. Adding it collapses the count:

| written form | as a predicate | verdict |
|---|---|---|
| `B08_LEG1_GATE_PREREG.md` §5.6 — "CI entirely above +5.0 pp" | `lo > +5.0` | **pre-registration of record** |
| `STATUS.json.next_gate.decidable_outcome` — "CI entirely above +5.0 pp" | `lo > +5.0` | concordant (machine-read) |
| `RELATED_WORK.md` §3.1 — "Δ_U CI > +5.0 pp" | `lo > +5.0` | concordant |
| `RELATED_WORK.md` §8 blockquote — "exceeds +5.0 pp with a 95% CI entirely above 0" | `point > +5.0` **and** `lo > 0` | ⛔ **LONE OUTLIER, strictly weaker** |

**So: TWO distinct PASS predicates, 3-to-1, and the outlier is the sentence §8 calls
the "safe residual claim" — the one a reviewer reads.** It is strictly weaker than
the pre-registration, so it can declare a PASS the prereg calls a FAIL.

**Worked case (predicates executed, not read).** `Δ_U = +6.0 pp`, CI `[+1.0, +11.0]`:
§8 → **PASS** (6.0 > 5.0 and 1.0 > 0); prereg 5.6 / `next_gate` / §3.1 → **FAIL**
(lo = 1.0 is not > 5.0); K2 → does not fire (hi = 11.0 ≥ 5.0), so no KILL either.
Control: on prereg 5.7's own plausible-PASS figures (`Δ_U = +11.4`, CI `[+6.2, +16.9]`)
**all four forms agree → PASS**, and on its plausible-KILL figures (`+0.7`,
`[-2.6, +4.1]`) all four agree → not a pass, K2 fires. The divergence is confined to
the band where `lo ∈ (0, 5.0]` and `point > 5.0`.

**Also retracted from §13.5**: it called the divergent case *"the proposal has no
verdict"* and framed that as part of the defect. That framing is wrong. A two-sided
gate with an inconclusive middle band is **normal** and is what `kill_gate` already
specifies (KILL iff ALL THREE; PASS on its own clause; neither ⇒ inconclusive). An
inconclusive region is not a defect. **The only defect is the §8-vs-prereg
disagreement.**

**Repair, unchanged and still for the owner, PRE-DATA**: make §8's blockquote quote
prereg §5.6 by reference instead of restating the arithmetic, so exactly one
definition exists. Doing it before any number is scored keeps it a clarification
rather than an outcome-dependent choice. **Until then, prereg 5.6 governs.**"""


def _verify_insert_only(before, after, inserted):
    residue = after
    for text in inserted:
        idx = residue.find(text)
        if idx < 0:
            print(f"  FAIL: inserted text not found: {text[:70]!r}")
            return False
        residue = residue[:idx] + residue[idx + len(text):]
    if residue != before:
        print(f"  FAIL: residue {len(residue)} B != before {len(before)} B")
        return False
    return True


def main():
    with open(DOC, "rb") as f:
        raw_b = f.read()
    before = raw_b.decode("utf-8")
    sha = hashlib.sha256(raw_b).hexdigest()
    print(f"[pre]  {len(raw_b)} bytes, sha256 {sha}")
    if len(raw_b) != BYTES_BEFORE or sha != SHA_BEFORE:
        sys.exit(f"ABORT: unexpected input state.\n  want {BYTES_BEFORE} B / "
                 f"{SHA_BEFORE}\n  got  {len(raw_b)} B / {sha}")
    if "### 13.7" in before:
        sys.exit("ABORT: 13.7 already present. Run once only.")

    sites = [("13.5 header", ANCHOR_1355_HDR, MARK_1355_HDR),
             ("sec 3.1 marker", ANCHOR_S31, MARK_S31),
             ("sec 8 marker", ANCHOR_S8, MARK_S8)]

    print(f"\n--- anchor uniqueness ({len(sites)} sites) ---")
    bad = False
    for label, anchor, _m in sites:
        n = before.count(anchor)
        if n != 1:
            bad = True
        print(f"  [{'OK  ' if n == 1 else 'FAIL'}] count={n}  {label}")
    if bad:
        sys.exit("ABORT: an anchor is not unique; nothing written.")

    after = before
    inserted = []
    for label, anchor, marker in sites:
        idx = after.index(anchor) + len(anchor)
        after = after[:idx] + marker + after[idx:]
        inserted.append(marker)
        print(f"  [ins ] {len(marker):5d} B  {label}")

    add = SECTION_13_7.rstrip("\n") + "\n"
    # 13.7 goes at the very end of section 13, i.e. after 13.6. Located by header,
    # computed not hardcoded (memory/a-writers-hardcoded-destination...).
    hdr = "\n### 13.6 What this round did NOT do"
    if after.count(hdr) != 1:
        sys.exit("ABORT: cannot locate a unique 13.6 header.")
    after = after.rstrip("\n") + "\n" + add
    inserted.append(add)
    print(f"  [ins ] {len(add):5d} B  SECTION_13_7 (appended after 13.6)")

    print("\n--- insert-only proof ---")
    if not _verify_insert_only(before, after, list(inserted)):
        sys.exit("ABORT: insert-only FAILED; nothing written.")
    print("  [OK  ] residue byte-identical to input")

    with open(DOC, "w", encoding="utf-8") as f:
        f.write(after)
    with open(DOC, "rb") as f:
        raw_after = f.read()
    disk = raw_after.decode("utf-8")
    print(f"\n[post] {len(raw_b)} -> {len(raw_after)} bytes "
          f"(+{len(raw_after)-len(raw_b)}), sha256 "
          f"{hashlib.sha256(raw_after).hexdigest()}")

    ok = _verify_insert_only(before, disk, list(inserted))
    print(f"[post] insert-only re-verified from disk: {'PASS' if ok else 'FAIL'}")

    checks = [
        ("13.5 marked RETRACTED IN PART", "RETRACTED IN PART, SAME SESSION" in disk),
        ("13.5 original wrong text retained", ANCHOR_1355_HDR in disk),
        ("13.7 appended", "### 13.7 SELF-CORRECTION" in disk),
        ("13.7 states TWO predicates, not three",
         "TWO distinct PASS predicates, 3-to-1" in disk),
        ("13.7 names prereg 5.6 as the source 13.5 missed",
         "pre-registration of record" in disk),
        ("13.7 identifies sec 8 blockquote as the outlier",
         "LONE OUTLIER, strictly weaker" in disk),
        ("13.7 retracts the 'no verdict' framing",
         "An\ninconclusive region is not a defect." in disk),
        ("13.7 sits after 13.6",
         disk.index("### 13.7") > disk.index("### 13.6")),
        ("sec 3.1 marker corrected (line is majority, not outlier)",
         "Scoring against THIS line is fine" in disk),
        ("sec 8 marker says score against prereg 5.6",
         "Score against prereg 5.6, not against this" in disk),
    ]
    print("\n--- read-outs ---")
    for label, cond in checks:
        if not cond:
            ok = False
        print(f"  [{'OK  ' if cond else 'FAIL'}] {label}")

    print("\nRESULT:", "PASS" if ok else "FAIL - RESTORE FROM GIT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
