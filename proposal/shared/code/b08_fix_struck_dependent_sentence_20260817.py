#!/usr/bin/env python3
"""Follow-up marker: the ROC row's own conclusion still cites the STRUCK (a).

FOUND BY SWEEPING FOR CONSEQUENCES, NOT BY THE MANDATE
=====================================================
``b08_apply_required_narrowing_20260817.py`` struck differentiator (a) in the
``arXiv:2607.17545`` row (RELATED_WORK.md section 3.3). But that row's CLOSING
sentence reads:

    "**Leg 1's residual claim must be stated against this paper explicitly, and
    it shrinks to (a)+(b).**"

With (a) struck, "(a)+(b)" is a dangling reference to a falsified conjunct, and a
reader who trusts the summary sentence rather than the struck list walks away with
the size of the residual claim OVERSTATED -- exactly the error the strike existed
to prevent. It is arithmetically now (b) ALONE.

This was not in the four mandated edits. It surfaced from a regex sweep of
sections 1-11 for statements DEPENDENT on the struck text
(``\\(a\\)\\+\\(b\\)|shrinks to|differentiator|retrieval-closed isolation``), which
returned exactly two live hits: line 82 (already marked by the previous script)
and this one. Recorded per ``memory/fix-the-class-not-the-instance.md``: striking
a sentence is not finished until the sentences that CITE it are reconciled.

Same insert-only discipline and the same machine-checked proof as the previous
script: nothing is deleted or reworded, and the residue after stripping the
inserted marker is asserted byte-identical to the input.

0 GPU, 0 ssh. Run once.
"""
import hashlib
import os
import sys

ROOT = "/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory"
DOC = os.path.join(ROOT, "proposal/backlog/B08-memory-applications/RELATED_WORK.md")

# Output state of b08_apply_required_narrowing_20260817.py, measured.
SHA_BEFORE = "3219395b2bdcbb80bc20ae90b93d437da68e98407a650f0bc363a2a0737d1e48"
BYTES_BEFORE = 73348

ANCHOR = ("**Leg 1's residual claim must be stated against this paper "
          "explicitly, and it shrinks to (a)+(b).**")

MARKER = (
    " ⛔ **§13.1 — CORRECTED: it shrinks to (b) ALONE, not (a)+(b). (a) is"
    " STRUCK above as FALSE, so this sentence's own arithmetic no longer holds"
    " as written; leaving it would let a reader restore the falsified conjunct"
    " from the summary line. (c) is a true statement about ROC but is not a"
    " differentiator of leg 1 — it is the same notes-only contrast that (b)"
    " already names.**"
)

SECTION_13_1_ADDENDUM = """

**⛔ Addendum (same session, found by sweeping for CONSEQUENCES of the strike).**
The ROC row's own closing sentence — *"Leg 1's residual claim … shrinks to
(a)+(b)"* — still cited the struck (a). Striking a sentence is not finished until
the sentences that CITE it are reconciled, so that conclusion now carries a
correction marker in place: **the residual claim shrinks to (b) ALONE.**
(b) is "its harm signal is a *calibrated utility estimate*, not a measured
unsupported-claim rate on a notes-only arm" — the `Δ_U` differentiator, which is
exactly what §13.3 re-anchors the gate on. (c) — "it does not run notes-only-vs-raw
as a paired faithfulness contrast" — is true of ROC but is not an independent
differentiator: it is the same notes-only contrast (b) already names.

A regex sweep of §1-§11 for statements dependent on the struck text
(`\\(a\\)\\+\\(b\\)`, `shrinks to`, `differentiator`, `retrieval-closed isolation`)
returned exactly **two** live hits — §3.2's RECOMP row (already demoted, §13.1)
and this one. Both are now marked; no third dependent statement exists in §1-§11."""


def _verify_insert_only(before, after, inserted):
    residue = after
    for text in inserted:
        idx = residue.find(text)
        if idx < 0:
            print(f"  FAIL: inserted text not found: {text[:60]!r}")
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
        sys.exit(f"ABORT: not the expected post-narrowing state.\n"
                 f"  want {BYTES_BEFORE} B / {SHA_BEFORE}\n"
                 f"  got  {len(raw_b)} B / {sha}")

    if MARKER.strip() in before:
        sys.exit("ABORT: marker already present. Run once only.")

    n = before.count(ANCHOR)
    print(f"[pre]  anchor count = {n} (require exactly 1)")
    if n != 1:
        sys.exit("ABORT: anchor not unique.")

    # 13.1 addendum goes at the END of section 13.1, i.e. immediately before the
    # "### 13.2" header. Computed, not hardcoded to a line number.
    hdr = "\n### 13.2 EDIT 2 —"
    if before.count(hdr) != 1:
        sys.exit("ABORT: cannot locate a unique 13.2 header to insert before.")

    after = before
    idx = after.index(ANCHOR) + len(ANCHOR)
    after = after[:idx] + MARKER + after[idx:]

    add = SECTION_13_1_ADDENDUM.rstrip("\n") + "\n"
    j = after.index(hdr)
    after = after[:j] + "\n" + add + after[j:]

    inserted = [MARKER, "\n" + add]
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
        ("marker present at the ROC row conclusion",
         "it shrinks to (b) ALONE, not (a)+(b)" in disk),
        ("original '(a)+(b)' sentence still on disk (no deletion)",
         ANCHOR in disk),
        ("13.1 addendum appended inside section 13.1",
         "found by sweeping for CONSEQUENCES of the strike" in disk),
        ("addendum sits BEFORE 13.2",
         disk.index("found by sweeping for CONSEQUENCES")
         < disk.index("### 13.2 EDIT 2 —")),
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
