#!/usr/bin/env python3
"""Apply B08's four REQUIRED_NARROWING edits to RELATED_WORK.md (2026-08-17, 0 GPU).

WHY THIS SCRIPT EXISTS, and why it is not a pure append
=======================================================
``STATUS.json.novelty_verdict.REQUIRED_NARROWING_four_edits_0_GPU`` mandates four
edits. Edit 1 says **STRIKE** a differentiator from sections 3.3 / 6 / 8, and
``NOVELTY_VERDICT.md`` 5.1 item 1 spells it "DELETE the differentiator".

But this file is maintained append-only by convention: section 11 declares
"sections 1-10 stay byte-stable" and section 12 declares "sections 1-11 are left
byte-stable". The 2026-08-16 round obeyed that convention literally -- it
appended 12.1, which SAYS the sentence "is hereby STRUCK" -- and the sentence is
still live, verbatim and unmarked, at line 100. A reader who lands in section 3
(which is where a reviewer looking for leg 1's differentiators lands) reads the
falsified claim and never sees 12.1, 426 lines further down.

So: appending a SECOND declaration that the edits are done would look like
progress and change nothing a reviewer reads. That is the exact failure recorded
in ``memory/agent-output-must-be-persisted-to-the-consumers-file.md``.

RESOLUTION -- INSERT-ONLY, NEVER DELETE
---------------------------------------
This script inserts SHORT POINTER MARKERS at each live site and appends a full
section 13. It performs **zero deletions and zero rewordings**: every original
byte survives, in its original order. The script PROVES this rather than
asserting it -- ``_verify_insert_only`` removes exactly the inserted marker
strings from the result and asserts the remainder is byte-identical to the
pre-edit file (sha256 83fda786...). So the original adjudication is fully
recoverable, which is what the byte-stable convention was protecting, while a
reader landing in section 3 now sees the strike, which is what the mandate
required. The convention's LETTER is bent; its PURPOSE (nothing is silently
rewritten) is preserved and machine-checked.

Markdown ``~~strikethrough~~`` is used for the one genuinely FALSE sentence, so
the falsification is visible when rendered while the text is still on disk.

WHERE THIS SCRIPT DISAGREES WITH THE DISPATCH THAT COMMISSIONED IT
-----------------------------------------------------------------
Two of the four sites were mis-located upstream. Both corrections are recorded
in section 13 and both make the edit SMALLER and more honest, not larger:

  * The dispatch said the falsified sentence is live "at line 100 (ROC row) and
    an identical one at :82 (RECOMP row)". They are NOT identical and must not
    get the same marker. Line 100 asserts ROC "never pins retrieval" and that is
    FALSE -- 12.1 quotes ROC's own Setup supplying gold evidence. Line 82 asserts
    the same of RECOMP, and **nobody has read RECOMP's protocol to check it**.
    Marking line 82 STRUCK/FALSE would fabricate a finding about RECOMP. Line 82
    gets a DEMOTION marker (edit 2's remit: closure is a precondition, so this
    contrast is no longer load-bearing) with the non-verification stated.
  * The dispatch said edit 3 is unlanded because "section 8's residual sentence
    still gives Delta_aug equal billing". It does not: the section 8 blockquote
    already reads "Delta_aug ... does not require notes to beat raw for the claim
    to hold", i.e. Delta_U is already sole load-bearing THERE. The place that
    still grants Delta_aug co-decisive standing is section 3.1's
    "PASS on Delta_aug CI > 0 **or** Delta_U CI > +5.0 pp" disjunction (and its
    source, ``next_gate.decidable_outcome``). The marker goes to 3.1.

A THIRD DEFECT, FOUND WHILE PLACING EDIT 3 AND NOT PREVIOUSLY RECORDED ANYWHERE
-------------------------------------------------------------------------------
The Delta_U PASS threshold is written three different ways across the proposal,
and they are not equivalent. See section 13.5. This is decision-relevant: on the
same data, two of the three spellings can disagree about whether the gate passed.
It is recorded, NOT silently resolved -- picking one would be a pre-registration
change, which is the owner's call, not a writer script's.

0 GPU, 0 ssh. Not a general tool: provenance for one edit. Run once.
"""
import hashlib
import os
import sys

ROOT = "/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory"
DOC = os.path.join(ROOT, "proposal/backlog/B08-memory-applications/RELATED_WORK.md")

SHA_BEFORE = "83fda7862862e8cd182077e1c042c7c9d201db7b93d5dd671dd7da34e52af5b4"
BYTES_BEFORE = 59799

# --------------------------------------------------------------------------- #
# In-place INSERTIONS. (anchor, inserted_text, where) -- `where` is "after" or
# "before". Every anchor is asserted to occur EXACTLY ONCE before anything is
# written; a count of 0 or 2 aborts. Nothing is ever removed or reworded.
# --------------------------------------------------------------------------- #
MARK_S2 = (
    " ⛔ **§13.2 — the `any_hit = 1.000` conjunct in this cell is a"
    " PRECONDITION, not a contribution. Only the *unsupported-claim rate on a"
    " notes-only arm* conjunct is load-bearing; `arXiv:2607.17545` and"
    " `arXiv:2605.24579` both hold retrieval closed (§12.1, §12.2).**"
)

MARK_RECOMP = (
    " ⚠ **§13.2 — DEMOTED: this retrieval-pinning contrast is no longer a"
    " differentiator (closure is a precondition). NOTE: unlike the ROC row below,"
    " this sentence has NOT been checked against RECOMP's own protocol — it is"
    " demoted as non-load-bearing, NOT declared false.**"
)

MARK_ROC_OPEN = (
    "⛔ **(a) IS STRUCK — FALSE. See §13.1: ROC's Setup supplies GOLD EVIDENCE"
    " to every arm, i.e. recall 1.000 by construction. Struck text retained"
    " below for provenance:** ~~"
)
MARK_ROC_CLOSE = "~~"

MARK_S31 = (
    " ⛔ **§13.3 — this OR-disjunction is superseded: `Δ_U` is the SOLE"
    " decisive clause and `Δ_aug` is demoted to supporting (it also carries the"
    " single-reader scope limit of §13.4). ⚠ §13.5: this line's `Δ_U` PASS"
    " threshold is ALSO one of three non-equivalent spellings of it in the"
    " proposal — do not score against this line.**"
)

MARK_S6 = (
    " ⛔ **§13.2 — SUPERSEDED: the *pairing* is no longer the claimed gap. The"
    " retrieval-closed stratum is a PRECONDITION (ROC and WhenLoss both close"
    " retrieval, one by oracle construction). What remains unoccupied is the"
    " notes-only unsupported-claim contrast ALONE.**"
)

MARK_S8_PIN = (
    " ⛔ **§13.2 — \"we pin it closed\" is a precondition, not a difference:"
    " ROC/WhenLoss pin it too. The surviving one-line difference is that neither"
    " scores an unsupported-claim rate on a notes-withheld arm.**"
)

# Inserted as its own paragraph directly under the section 8 blockquote. This is
# edit 4's mandated home: NOVELTY_VERDICT.md 5.1 item 4 and RELATED_WORK.md 550
# both say the clause must be "written into the gate's scope sentence".
MARK_S8_SCOPE = """

⛔ **§13.4 — MANDATORY SCOPE CLAUSE ON THE SENTENCE ABOVE (single reader).** The
gate reads out at **exactly one reader**, `models/Meta-Llama-3-8B`
(`next_gate.frozen_across_arms`). `arXiv:2606.21807` measures, on **LongMemEval-S
itself**, that generic summarisation **flips 31% of pairwise model rankings**, and
that compression gain **shrinks as the reader gets stronger** (9/10 settings
p<0.05 over 20 readers). Therefore the sentence above is, as written, a claim
about **notes at Meta-Llama-3-8B's competence**, not a claim about notes. Until a
second reader is run, every `Δ_aug` and `Δ_sub` number from this gate MUST be
reported with that qualifier attached, and no reader-independent generalisation
may be drawn from it. `Δ_U` is *less* exposed (it is scored against each arm's own
context, not against a reader-strength-dependent accuracy ceiling) but is not
immune, because U is measured on text this one reader generated.

⛔ **§13.3 — `Δ_U` IS THE SOLE DECISIVE CLAUSE.** Of prereg 5.6's three survival
branches, only the `Δ_U` branch is unoccupied: `arXiv:2606.27472` (Supersede)
already measured the `Δ_sub` direction on 78 of this stratum's 134 items
(92%→77%, paired McNemar p=0.0033), in the OPPOSITE direction to prereg 5.6's
"~30x context reduction at no accuracy cost". `Δ_sub` may therefore be reported
only as a replication under a different closure regime, never as a finding.

⚠ **§13.5 — the `Δ_U` PASS threshold is written three non-equivalent ways in this
proposal. Do not score until the owner picks one.** See §13.5."""


INSERTIONS = [
    # (label, anchor, text, where)
    ("S2_measurement_cell",
     "**NO / thin.** No verified work holds retrieval at a **measured** "
     "`any_hit = 1.000` and varies only context composition while scoring an "
     "**unsupported-claim rate** on a notes-only arm.",
     MARK_S2, "after"),
    ("S3_2_RECOMP_row",
     "it also never pins retrieval at a *measured* recall of 1.000, so its "
     "deltas remain retrieval-confounded.",
     MARK_RECOMP, "after"),
    ("S3_3_ROC_row_open",
     "What it does **not** do: ",
     MARK_ROC_OPEN, "after"),
    ("S3_3_ROC_row_close",
     "the exact confound B08's stratum was chosen to remove;",
     MARK_ROC_CLOSE, "after"),
    ("S3_1_PASS_disjunction",
     "PASS on Δ_aug CI > 0 **or** Δ_U CI > +5.0 pp.",
     MARK_S31, "after"),
    ("S6_leg1_pairing",
     "own the mechanism, but **not** the pairing of a\n  measured "
     "retrieval-closed stratum with a notes-only unsupported-claim contrast.",
     MARK_S6, "after"),
    ("S8_blockquote_scope",
     "> raw for the claim to hold.**",
     MARK_S8_SCOPE, "after"),
    ("S8_one_line_each",
     "Chain-of-Note needs retrieval to be *noisy* (we pin it closed);",
     MARK_S8_PIN, "after"),
]

SECTION_13 = """

---

## 13. REQUIRED NARROWING APPLIED (2026-08-17, fourth agent) — the four mandated edits, LANDED

**0 GPU, 0 ssh, 0 network.** All four edits named in
`STATUS.json.novelty_verdict.REQUIRED_NARROWING_four_edits_0_GPU` (and restated as
`NOVELTY_VERDICT.md` §5.1) are applied by this section **plus in-place pointer
markers at each live site**, because §12.1 already demonstrated that an appended
declaration alone does not reach the reader.

### 13.0 Mechanics, and the one convention this section bends on purpose

§11 declares §1-§10 byte-stable and §12 declares §1-§11 byte-stable. Edit 1,
however, says **STRIKE** (`NOVELTY_VERDICT.md` §5.1 item 1: "DELETE the
differentiator"). A delete is by definition not an append, so the mandate and the
convention are in direct conflict, and §12 resolved it by obeying the convention:
it appended a paragraph saying the sentence "is hereby STRUCK" and left the
sentence live, verbatim and unmarked, at line 100 — **426 lines above the
paragraph that struck it**. A reviewer auditing leg 1's differentiators lands in
§3, not in §12.

This round resolves the conflict the other way, minimally and verifiably:

* **INSERT-ONLY. Zero deletions, zero rewordings.** Eight short pointer markers
  were inserted; not one original byte was removed or changed. The writer
  (`proposal/shared/code/b08_apply_required_narrowing_20260817.py`) *proves* this:
  it strips exactly the inserted marker strings from the result and asserts the
  remainder is byte-identical to the pre-edit file
  (59,799 B, sha256 `83fda7862862e8cd182077e1c042c7c9d201db7b93d5dd671dd7da34e52af5b4`).
  So the original adjudication remains fully recoverable — which is what
  byte-stability was protecting — while §3 now shows the strike.
* Each anchor was asserted to occur **exactly once** before any write.
* The one genuinely FALSE sentence is wrapped in `~~ ~~` so it renders struck.

### 13.1 EDIT 1 — STRUCK: "ROC never pins retrieval at a measured `any_hit = 1.000`"

**Site: §3.3, the `arXiv:2607.17545` (Retain or Consolidate?) row, differentiator
(a).** Now marked ⛔ and wrapped in strikethrough in place.

It is false, on ROC's own words, already quoted in §12.1: *"In the controlled
evaluation, **every action receives the same gold evidence** and differs only in
its budgeted representation. … This pairing **isolates the when–which decision
from evidence discovery**."* Gold evidence to every arm is recall = 1.000 **by
construction** — a *stronger* closure than B08's measured-1.000 stratum. So the
"its deltas mix retrieval with composition" half of the differentiator is also
false: ROC's controlled deltas do not mix them.

**⚠ CORRECTION TO THE DISPATCH THAT COMMISSIONED THIS ROUND.** It stated that the
same sentence is live "at :82 (RECOMP row)" as an "identical" instance and should
get the same treatment. **The two are not equivalent and must not get the same
marker.** §3.2's RECOMP row says RECOMP "never pins retrieval at a *measured*
recall of 1.000". **Nobody in any round has read RECOMP's protocol to check
that.** Marking it STRUCK/FALSE would manufacture a finding about an ICLR 2024
paper from an unrelated paper's text. It therefore carries a **demotion** marker
(§13.2's remit) with the non-verification stated inline. If a later round wants
RECOMP's row adjudicated on the merits, that is a new literature task: read
RECOMP §Experiments and check whether it reports a retrieval-recall figure at all.

### 13.2 EDIT 2 — DEMOTED: the retrieval-closed stratum is a PRECONDITION, not a contribution

Marked in place at four sites: §2's "Is the *measurement* prior art?" cell, §3.2
(RECOMP row), §6 ("Leg 1 = WE DID TOO LITTLE"), and §8's one-line-each paragraph
("we pin it closed").

The stratum **stays in the design** — it is still the right cell, and the closure
is measured at the gate's own `evidence_token_budget=4000`
(`evidence/b08_prereg_corrections_20260814.json`: `knowledge-update` n=78
`any_hit=1.0000`, `single-session-assistant` n=56 `any_hit=1.0000`, while overall
falls to 0.9600 and `single-session-preference` to 0.7000, which is why that type
is excluded). What changes is its **status in the argument**: it is a control that
makes the read-out interpretable, not a novelty. Two concurrent works close
retrieval on this very benchmark — ROC by oracle construction (§12.1) and WhenLoss
via its OE condition (§12.2) — so "we isolate the composition axis by pinning
retrieval" is not an available framing.

The **one** sentence it still supports is a *provenance* difference, never a
headline: closure here is **measured on the deployed BM25 retriever**, not
**supplied by an oracle**. That is a statement about external validity (the
closed regime is one the deployed system actually reaches on this stratum), and it
is worth exactly one sentence.

Note that §3.1 **already** contained the correct framing before this round — *"The
retrieval premise is a constraint, not a win"*, citing
`established_measurements.consequence_is_a_CONSTRAINT_not_a_win`. Edit 2 is
therefore partly a consistency repair: §3.1 said precondition while §2/§3.3/§6/§8
still said contribution.

### 13.3 EDIT 3 — RE-ANCHORED: `Δ_U` is the sole decisive clause

Landed as a marker on §3.1's PASS disjunction and as a full paragraph under §8's
residual-claim blockquote.

**⚠ SECOND CORRECTION TO THE DISPATCH.** It stated that this edit is unlanded
because "§8's residual sentence still gives `Δ_aug` equal billing". **It does
not.** The §8 blockquote already reads: *"`Δ_U` … exceeds +5.0 pp with a 95%
paired-bootstrap CI entirely above 0, while `Δ_aug` … **does not require notes to
beat raw for the claim to hold**."* `Δ_U` is already sole load-bearing *there*.

The place that still grants `Δ_aug` co-decisive standing is **§3.1**:
*"PASS on Δ_aug CI > 0 **or** Δ_U CI > +5.0 pp"* — a disjunction in which either
clause alone passes the gate — and its upstream source
`STATUS.json.next_gate.decidable_outcome`, which is worded the same way. That is
where the marker went. Being precise about the site matters: an agent that
"fixed" §8 would have changed a sentence that was already correct and left the
actual disjunction standing.

Substance (unchanged from `NOVELTY_VERDICT.md` §5.1 item 3): `arXiv:2606.27472`
(Supersede) already measured the `Δ_sub` branch on 78 of this stratum's 134 items,
in the opposite direction (92%→77%, paired McNemar p=0.0033), so `Δ_sub` is a
replication and not a finding; `Δ_aug`'s branch is occupied by ROC/RECOMP/CoN as
accuracy-at-a-budget; only `Δ_U` is unoccupied. **K2 is therefore the decisive
kill clause, not one of three.**

### 13.4 EDIT 4 — ADDED: the single-reader scope clause, in the gate's scope sentence

Landed as a marked paragraph directly under §8's blockquote, which is the location
`NOVELTY_VERDICT.md` §5.1 item 4 and §12.2's last row both name ("must be written
into the gate's scope sentence").

**This was genuinely absent, verified rather than assumed**: a grep over §8
(lines 291–322 pre-edit) for `reader|competence|Llama-3-8B|scope|2606.21807`
returned **zero hits**. `arXiv:2606.21807` appeared exactly once in the whole
59,799-byte file, in §12.2's table — i.e. the design threat was *recorded* in the
appendix and never *applied* to the claim.

### 13.5 ⚠ NEW DEFECT FOUND WHILE PLACING EDIT 3: the `Δ_U` PASS threshold has three non-equivalent spellings

Not previously recorded in any round, and **not resolved here** — resolving it
would be a pre-registration change, which is the owner's call, not a writer's.

| where | what it says about `Δ_U` | as a condition |
|---|---|---|
| `STATUS.json.next_gate.decidable_outcome` | "`Δ_U` … has a CI entirely above **+5.0 pp**" | PASS iff CI **lower** bound > +5.0 |
| `STATUS.json.kill_gate` K2 | kill fires iff "`Δ_U` 95% CI **UPPER BOUND < +5.0 pp**" | K2 does not fire iff CI **upper** bound ≥ +5.0 |
| `RELATED_WORK.md` §8 blockquote | "exceeds +5.0 pp with a 95% … **CI entirely above 0**" | PASS iff point est > +5.0 **and** CI lower bound > 0 |

These are three different tests. Worked counterexample using the file's own
plausible-PASS figure (`kill_gate.falsifiability_worked_example`,
`Δ_U = +11.4 pp`, CI `[+6.2, +16.9]`): all three agree → PASS. Now perturb to
`Δ_U = +6.0 pp`, CI `[+1.0, +11.0]`: `next_gate` says **FAIL** (lower bound 1.0 is
not > 5.0); §8 says **PASS** (6.0 > 5.0 and 1.0 > 0); K2 does **not** fire (upper
bound 11.0 ≥ 5.0), so the ALL-THREE kill branch is blocked and the gate is
neither a pass nor a kill. **On that data the proposal has no verdict.** Note the
two-sided gate is not symmetric here: "K2 does not fire" only blocks the KILL; it
is not the same predicate as the PASS.

Recommended repair (**for the owner, PRE-DATA**): make §8 and `next_gate` quote K2
by reference instead of restating a threshold, so exactly one arithmetic
definition exists. Doing this before any number is scored keeps it a clarification
rather than an outcome-dependent choice.

### 13.6 What this round did NOT do

1. **It did not clear the gate's runnability.** ACC still does not exist:
   `longmemeval/scoring.py:30-43` `write_submission` emits only
   `{question_id, hypothesis}`, while `scripts/a02_judge_openweight.py:187-201`
   keys on `item["id"]` and reads `pred` / `question` / `answers` / `category` /
   `is_abstention`. Without an adapter, `Δ_aug` and `Δ_sub` are not computable
   even after a card is booked. `Δ_U` is unaffected: `longmemeval/faithfulness.py`
   consumes the `--context_log` records, which already carry every field it needs.
   Full field-by-field mapping: `STATUS.json.judge_adapter_spec_20260817`.
2. **It did not verify any asset on zwfy6.** `/apdcephfs_zwfy6` is **not mounted**
   on this node (`ls -d` → "No such file or directory"; `mount` lists only
   `/apdcephfs_wzc1*`) and ssh was barred. Every presence claim in this file stays
   **wzc1-scoped**. `remaining_blockers_all_CPU[6]` is NOT closed and cannot be
   closed without ssh.
3. **It did not read RECOMP's protocol** (see §13.1's correction), so §3.2's row
   is demoted, not adjudicated.
4. **It did not re-run any literature search.** §12.3's negatives and §12.6's venue
   verifications stand as they were; the `2026.findings-acl` gap flagged at the end
   of §12.6 is still open.
5. **It did not resolve §13.5**, by design.
"""


def _verify_insert_only(before: str, after: str, inserted: list) -> bool:
    """Strip exactly the inserted strings from `after`; assert we get `before`.

    This is the whole audit guarantee: it makes "insert-only" a MEASURED property
    rather than a claim in a docstring. Each marker is removed once, in the order
    inserted; if the writer had deleted or reworded anything, the residue would
    differ from `before` and this returns False.
    """
    residue = after
    for text in inserted:
        idx = residue.find(text)
        if idx < 0:
            print(f"  FAIL: inserted text not found for removal: {text[:60]!r}")
            return False
        residue = residue[:idx] + residue[idx + len(text):]
    ok = residue == before
    if not ok:
        print(f"  FAIL: residue {len(residue)} B != before {len(before)} B")
        for i, (a, b) in enumerate(zip(residue, before)):
            if a != b:
                print(f"  first divergence at char {i}: "
                      f"{residue[max(0,i-60):i+60]!r} vs {before[max(0,i-60):i+60]!r}")
                break
    return ok


def main():
    with open(DOC, "rb") as f:
        raw_before_b = f.read()
    before = raw_before_b.decode("utf-8")
    sha = hashlib.sha256(raw_before_b).hexdigest()

    print(f"[pre]  {len(raw_before_b)} bytes, sha256 {sha}")
    if len(raw_before_b) != BYTES_BEFORE or sha != SHA_BEFORE:
        sys.exit(f"ABORT: file is not the audited pre-edit state.\n"
                 f"  want {BYTES_BEFORE} B / {SHA_BEFORE}\n"
                 f"  got  {len(raw_before_b)} B / {sha}\n"
                 f"Someone edited it after 2026-08-17; re-audit before writing.")

    if "## 13." in before:
        sys.exit("ABORT: a section 13 already exists. Run once only.")

    # --- Phase 1: assert EVERY anchor is unique BEFORE writing anything. ---
    print(f"\n--- anchor uniqueness ({len(INSERTIONS)} sites) ---")
    bad = False
    for label, anchor, _text, _where in INSERTIONS:
        n = before.count(anchor)
        flag = "OK  " if n == 1 else "FAIL"
        if n != 1:
            bad = True
        print(f"  [{flag}] count={n}  {label}")
    if bad:
        sys.exit("ABORT: an anchor is not unique; a blind replace would corrupt "
                 "the file or land in the wrong section.")

    # --- Phase 2: apply insertions. ---
    after = before
    inserted_texts = []
    for label, anchor, text, where in INSERTIONS:
        idx = after.index(anchor)
        at = idx + len(anchor) if where == "after" else idx
        after = after[:at] + text + after[at:]
        inserted_texts.append(text)
        print(f"  [ins ] {len(text):5d} B {where:6s} {label}")

    after = after.rstrip("\n") + "\n" + SECTION_13.lstrip("\n").rstrip("\n") + "\n"
    # The trailing-newline normalisation above must itself be a no-op on the
    # original tail, or it would be an undeclared edit. Pre-edit tail is exactly
    # one "\n" (measured), so rstrip+"\n" reproduces it byte-for-byte.
    tail_added = SECTION_13.lstrip("\n").rstrip("\n") + "\n"
    inserted_texts.append(tail_added)

    print(f"  [ins ] {len(tail_added):5d} B append  SECTION_13")

    # --- Phase 3: prove insert-only BEFORE touching disk. ---
    print("\n--- insert-only proof (strip markers, compare to pre-edit) ---")
    if not _verify_insert_only(before, after, list(inserted_texts)):
        sys.exit("ABORT: insert-only property FAILED; nothing written.")
    print("  [OK  ] residue is byte-identical to the pre-edit file")

    with open(DOC, "w", encoding="utf-8") as f:
        f.write(after)

    with open(DOC, "rb") as f:
        raw_after_b = f.read()
    print(f"\n[post] {len(raw_before_b)} -> {len(raw_after_b)} bytes "
          f"(+{len(raw_after_b) - len(raw_before_b)}), sha256 "
          f"{hashlib.sha256(raw_after_b).hexdigest()}")

    # --- Phase 4: re-verify from disk, and check the mandated read-outs. ---
    disk = raw_after_b.decode("utf-8")
    ok = _verify_insert_only(before, disk, list(inserted_texts))
    print(f"[post] insert-only re-verified from disk: {'PASS' if ok else 'FAIL'}")

    checks = [
        ("edit1 ROC differentiator (a) is struck in place",
         "⛔ **(a) IS STRUCK — FALSE." in disk),
        ("edit1 struck text is wrapped in ~~ ~~",
         "~~(a) it never pins retrieval" in disk
         and "chosen to remove;~~" in disk),
        ("edit1 original struck sentence still on disk (no deletion)",
         "(a) it never pins retrieval at a **measured** `any_hit = 1.000`" in disk),
        ("edit2 marker in section 2 measurement cell", MARK_S2.strip() in disk),
        ("edit2 marker in section 6", MARK_S6.strip() in disk),
        ("edit2 RECOMP row demoted, NOT declared false",
         "demoted as non-load-bearing, NOT declared false" in disk),
        ("edit3 marker on section 3.1 PASS disjunction", MARK_S31.strip() in disk),
        ("edit4 scope clause names the single reader in section 8",
         "models/Meta-Llama-3-8B" in disk.split("## 9.")[0].split("## 8.")[1]),
        ("edit4 scope clause cites the 31% ranking flip in section 8",
         "flips 31% of pairwise model rankings"
         in disk.split("## 9.")[0].split("## 8.")[1]),
        ("section 13 appended", "\n## 13. REQUIRED NARROWING APPLIED" in disk),
        ("13.5 records the three-way threshold defect",
         "three non-equivalent spellings" in disk),
        ("13.6 records that ACC is still not computable",
         "ACC still does not exist" in disk),
        ("zwfy6 is stated unverifiable, not verified",
         "not mounted" in disk and "wzc1-scoped" in disk),
    ]
    print("\n--- mandated read-outs present on disk ---")
    for label, cond in checks:
        if not cond:
            ok = False
        print(f"  [{'OK  ' if cond else 'FAIL'}] {label}")

    print("\nRESULT:", "PASS" if ok else "FAIL - RESTORE FROM GIT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
