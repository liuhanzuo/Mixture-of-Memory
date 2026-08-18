# paperC — decision log

Append-only. Each entry: what was decided, the alternatives considered, why, and what
would reverse it. Created 2026-08-15 when the `autonomous-paper-agent` skill was resumed
on paperC (the skill's `state/` had never been created; rounds 0–2 predate it).

---

## D-001 (2026-08-15) — Resume at round 3 rather than review the existing round_02 snapshot

**Decided:** freeze a new `round_03` snapshot and review that.

**Alternatives:**
1. Review the existing `round_02/submission` (frozen 08-14 23:41, never reviewed).
2. Review `round_01` (frozen 08-15 03:03, build gate PASS at 17 pp, never reviewed).
3. Freeze fresh as round 3. ← selected

**Why:** the working tree has changed since both freezes — the current build yields
**19 pages** where round_01 recorded **17**. Reviewing a stale snapshot would produce
findings against text that no longer exists, and any fix would be unverifiable against
the reviewed artifact. Round_01 additionally carries a known freeze defect (its
`snapshot_sha256_prefix` `96304267e30d79c3` was recorded without the recipe that made it;
7 candidate recipes were tried and none reproduced it, so that freeze documents intent
rather than verifiable state).

**Reverses if:** nothing — rounds 1 and 2 remain on disk as checkpoints and stay eligible
for `select_best_round.py`.

---

## D-002 (2026-08-15) — Carry forward round_01's prompt design, do not regress to round_00's

**Decided:** each reviewer receives **one defect with the numbers inlined** plus its
specialty lens, not "read the whole paper and find problems".

**Why:** round_00 used the broad form and **only 3 of 6 reviewers returned**; the `<4/6`
abort guard fired and the round produced no interpretable median (cost: 980,358 subagent
tokens, 8,042 s). Round_01 recorded the design change explicitly as the fix. Regressing
would risk repeating a five-figure-token abort.

**Reverses if:** a round returns 6/6 comfortably and the panel complains the framing was
too narrow to see cross-cutting problems.

---

## D-003 (2026-08-15) — Tell the panel that GPU is unavailable, and what that permits

**Decided:** state in the reviewer framing that zero GPUs are available, that reviewers
may still demand recomputation/reanalysis from on-disk per-item records, additional
statistical tests, scope narrowing, or clearer reporting — and that a claim which
genuinely needs new GPU work should be treated as a **scope** problem (narrow or drop it),
not a missing-experiment problem.

**Why:** withholding the constraint invites unactionable "train another model" requests
that would stall the round; but a blanket "don't ask for experiments" would let unsupported
claims survive. This framing keeps the pressure on the claims while pointing it at the
free, decisive actions. paperC's own gap audit states the headline needs 0 additional
GPU-hours, so the constraint does not bind the central contribution.

**Reverses if:** GPUs free up and a specific claim is blocked only by compute.

---

## D-004 (2026-08-15) — Venue compliance: relocate, never delete, to fix the page budget

**Decided:** fix the ICLR 2026 main-text overage by moving **unreferenced** main-text
tables into the appendix and adding explicit `\ref` for the tables that stay. Do not
delete evidence or drop claims to buy pages.

**Measured, not assumed** (`evidence/venue_page_budget_round03.json`):

| quantity | value |
|---|---|
| main-text pages (markers before `main.bbl`) | **11** |
| total pages incl. refs + appendix | 19 |
| ICLR 2026 main-text limit | 9 |
| **over by** | **2 pages** |
| tables never `\ref`'d in prose | **8 of 11** |

The unreferenced set: `tab:claims`, `tab:conventions`, `tab:integrity`, `tab:mmlupro`,
`tab:power`, `tab:two-nulls`, `tab:v2-alternatives`, `tab:v2-resort`.

**Why this is the right lever:** the two facts interact. Eight tables the prose never
points a reader to are already a presentation defect, and they are what pushes the main
text past the limit. Relocating them costs no evidence — it changes where the evidence
sits — so no claim weakens. This is the one page-saving move that is strictly an
improvement on both axes.

⚠️ **Two corrections and one open item are attached to this decision:**

1. **My first measurement was wrong and I corrected it.** I initially recorded 10 main-text
   pages / over by 1. The true figures are **11 / over by 2**. Root cause: my shipout-marker
   regex ran against the raw log, and TeX hard-wraps the log at ~79 columns, so markers whose
   brackets contain long embedded font paths straddle a newline and never matched. The
   discrepancy surfaced because the log states `Output written on main.pdf (19 pages` while my
   derivation said 18 — a self-inconsistency with the tool's own report. Sizing the fix from
   the wrong number would have left the paper non-compliant.
2. **The build gate cannot see this class of defect.** `latexmk` reports 0 undefined
   references because every `\label` exists; an *unreferenced* label is the opposite failure
   and no LaTeX warning covers it.
3. **OPEN:** whether ICLR 2026 excludes the Ethics Statement / Reproducibility Statement
   from the 9 pages. The `.sty` carries no such text (0 hits for `page limit`/`ethics`). If
   they are excluded, the overage is smaller than 2 pages. **This must be checked against the
   official CFP before the fix is finally sized** — I am not going to assume the stricter or
   the looser reading.

**Reverses if:** the CFP shows a different limit or different exclusions, or a reviewer
argues a specific relocated table is load-bearing for a main-text claim (in which case it
returns and something else moves).

---

## D-005 (2026-08-15) — Standing adjudication policy: MAIN re-derives every critical from raw data

**Decided:** no reviewer-raised critical or selected major is fixed on the reviewer's word.
MAIN re-derives it from the on-disk per-item records first, and the re-derivation is what
gets cited in the fix.

**Why:** this policy already paid for itself in round_00. All three criticals were
re-derived, all three were confirmed, and **two came back larger than the reviewer had
computed** — S2-01's winner's-curse bias was 2.315 pp at OBQA where the reviewer reported
1.017 pp, which changed the fix from a footnote into a rescoping of the abstract to the
three surviving constructs. A reviewer's arithmetic is a hypothesis, not a measurement.

**Reverses if:** never, for criticals. Minors may be taken on the reviewer's word when the
fix is purely editorial.
