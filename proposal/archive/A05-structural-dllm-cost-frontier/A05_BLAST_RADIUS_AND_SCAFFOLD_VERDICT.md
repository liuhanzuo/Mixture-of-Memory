# A05 closeout follow-up — blast radius, ownership, and the Scaffold provenance hole

**Ran** 2026-08-12, node `.73` (8×H20, zwfy6) + CPU on wzc1. **GPU cost: 0.** No model was loaded;
every number below is a re-grade of outputs already on disk, or a structural fact about code.
**Falsification conditions F1–F4 were registered in `A05_BLAST_RADIUS_PREREGISTRATION.md` and
committed (`bed7e43`) before the first grep.**

Grader: `evalplus.eval.untrusted_check`, vendored `evalplus 0.1.0.dev1`, with a **mandatory
per-dataset self-test on every invocation** — canonical solution PASSES, `pass` stub FAILS, on both
HumanEval+ (`HumanEval/0`) and MBPP+ (`Mbpp/100`). Recorded in
`evidence/a05_blast_radius.json:grader_self_tests`. Never a hand-rolled verifier (invariant 1).

---

## Headline: three answers

1. **§1 — the blast radius is NIL. F2 confirmed, F1 does NOT fire.** The premise in my brief that
   "published numbers in `DLLM_RESULTS_20260807.md` are affected too, including the AR control" is
   **wrong, and I am rejecting it on evidence.** 17 arms re-graded; **2** are affected (both are the
   already-known DreamOn HE+ arms), and even those move by **+0.61 pp = 1 item**, not 31 pp.
2. **§3 — the provenance hole is CLOSED, at 0 GPU and without moving the 29 GB checkpoint.**
   Scaffold Medium recomputed from its own stored per-item programs: **HE+ `.1768`, MBPP+ `.3545`**,
   versus READ `.177`/`.354`. **F3 does not fire.** Scaffold's numbers are *not* understated — so
   **A05 was killed on a correctly-measured comparison and must stay dead.**
3. **§2 — the finding does NOT belong to A01, and it is not big enough to be its own proposal
   either.** It belongs to **B10**, as a scoped sub-claim. Argued in §2 below; `STATUS.json`'s
   suggestion of A01 is **rejected**.

**The single most important correction to today's closeout narrative**: the "3.95× swing" is real
arithmetic but it is **not two independent artifacts of comparable size**. It is
**one artifact (canvas) plus a second defect whose severity is created by the first.** See §1.4.

---

## §1 — Blast radius

### 1.1 Structural enumeration (all three checkouts)

`combine_humaneval_prompt` — the only function containing the buggy `textwrap.indent` — is defined
and called in **exactly one driver**, `scripts/generate_evalplus_dreamon.py`, and only on the branch
`if args.dataset == "humaneval"`. Confirmed on all three checkouts (wzc1 `dllm_draft` @ `5735d72`,
zwfy6 `dllm_draft` @ `9651406`, zwfy6 `dllm_draft_104` @ `d214d37c`). The only other occurrences
anywhere are its own unit tests and A05's own copy in `a05_k1/a05_k1_dreamon_canvas.py`. Every
`textwrap.indent` hit outside those two files is inside `.venv_dream/` site-packages (torch, ray,
numpy) and never touches model output.

What the other arms do instead — **these are different code paths, not variants of the same one**:

| driver | arms it produced | post-processing | can the defect bite? |
|---|---|---|---|
| `generate_evalplus_dreamon.py` (`dataset=humaneval`) | `dreamon_heplus.r1`, `dreamon_heplus_r2`, A05 `he_c*` | `combine_humaneval_prompt` → extract, then **indent** | **YES — the only affected path** |
| `generate_evalplus_dreamon.py` (`dataset=mbpp`) | `dreamon_mbppplus.r1/_r2`, A05 `mbpp_c*` | `extract_python(raw)` only | no — `indent` unreachable |
| `generate_evalplus_dream.py` | all `dream_*`, `dream_coder_*` | `extract_python(raw)`, or `combine_base_continuation` = `extract_python(prompt + continuation)` | no — **concatenate-then-extract, never indents** |
| `generate_evalplus_ar.py` | AR control (Qwen2.5-Coder-7B) | imports `combine_base_continuation` from the dream driver | no — same concatenate-then-extract |
| `generate_evalplus_scaffold.py` | all `scaffold_*` tiers | **`solution = result.text`, no post-processing at all** | no |
| `generate_kspan*.py`, `refine_verifier_guided.py` | `kspan_*`, `refine_*` | own indent logic (`indent_of`) / `extract_python` | no — separate code, separate benchmark |

### 1.2 Empirical confirmation — 17 arms re-graded as-run vs corrected

`items_reaching_indent_branch` counts items where `extract_python(raw)` has no top-level `def`, i.e.
where the buggy branch is actually taken. `evidence/a05_blast_radius.json`.

| arm | benchmark | published (READ) | my as-run (RAN) | corrected (RAN) | Δ fix | items on buggy branch |
|---|---|---:|---:|---:|---:|---:|
| `dreamon_heplus.r1` | HE+ | — | .1098 | **.1159** | **+0.61 pp** | 164 |
| `dreamon_heplus_r2` | HE+ | .122 | .1220 | **.1280** | **+0.61 pp** | 164 |
| `dreamon_mbppplus.r1` | MBPP+ | — | .0661 | .0661 | +0.00 | 0 |
| `dreamon_mbppplus_r2` | MBPP+ | .085 | .0847 | .0847 | +0.00 | 0 |
| `dream_coder_instruct_heplus_r2` | HE+ | **.707** | **.7073** | .7073 | +0.00 | **0** |
| `dream_coder_instruct_mbppplus_r2` | MBPP+ | **.680** | .6772 | .6772 | +0.00 | **0** |
| `dream_coder_base_heplus` | HE+ | .079 | .0793 | .0793 | +0.00 | 0 |
| `dream_coder_base_mbppplus` | MBPP+ | .159 | .1587 | .1587 | +0.00 | 0 |
| `ar_qwen25coder7b_base` (AR control) | HE+ | (this run: .4939) | **.4939** | .4939 | +0.00 | **0** |
| `scaffold_tiny_heplus` | HE+ | — | .0000 | .0000 | +0.00 | 0 |
| `scaffold_small_heplus` | HE+ | — | .0427 | .0427 | +0.00 | 0 |
| **`scaffold_medium_heplus`** | HE+ | **.177** | **.1768** | .1768 | +0.00 | **0** |
| `scaffold_large_heplus` | HE+ | .177 | .1768 | .1768 | +0.00 | 0 |
| `scaffold_tiny_mbppplus` | MBPP+ | — | .0000 | .0000 | +0.00 | 0 |
| `scaffold_small_mbppplus` | MBPP+ | — | .1561 | .1561 | +0.00 | 0 |
| **`scaffold_medium_mbppplus`** | MBPP+ | **.354** | **.3545** | .3545 | +0.00 | **0** |
| `scaffold_large_mbppplus` | MBPP+ | .325 | .3254 | .3254 | +0.00 | 0 |

**Result: 2 of 17 arms move at all, and both are DreamOn HE+ — the path already known to be
affected.** Every non-DreamOn-HE+ arm has **exactly 0** items on the buggy branch, so its immunity is
structural, not a lucky cancellation. All 15 published values reproduce within **±0.28 pp** (worst
case `dream_coder_instruct_mbppplus_r2`, `.6772` vs `.680` = 1 item of 378; most are ±0.05 pp),
which independently validates the whole re-grading pipeline against the archive.

**F1 does not fire. F2 (nil blast radius) is confirmed. My brief's premise was wrong.**

### 1.3 The AR asymmetry is real but it is not a confound

My pre-registration predicted arms would split by **output shape** (bare indented body → affected;
emits its own `def` → immune), and that this asymmetry would be a confound penalising diffusion
outputs. **The first half is right, the second half is wrong**, and the distinction matters:

* AR is immune *not because* it emits `def`, but because it goes through a **different function**
  (`combine_base_continuation`, which concatenates prompt+continuation and *then* extracts — there is
  no `indent` call to get wrong). Even if AR had emitted a bare body, it would not have been damaged.
* So the correct statement is **not** "a shared post-processor mangles diffusion outputs and leaves AR
  intact". There is **no shared stitch**. It is: *one benchmark×driver combination out of six had a
  buggy bespoke stitch.* That is a narrower and less interesting claim than the one I was asked to test.
* The comparison A05 actually rested on (DreamOn vs Scaffold) is **not** contaminated by an asymmetry
  either, because Scaffold does no post-processing at all and its numbers are confirmed exact.

### 1.4 ★ The two artifacts are not independent — the second is conditional on the first

This is the finding that changes how the whole story should be told, and it is **not** in the closeout.

The archived, published `.122` moves by only **+0.61 pp (1 item: `HumanEval/13`, 0 items lost)** when
the stitch is fixed — not the +31 pp seen at canvas=128. Mechanism, measured on the archived r2 rows:

> At `initial_masks=8`, **only 1 of 164** raw outputs has more than one non-blank line
> (128/164 are empty, 36 non-empty after extraction). A double-indent can only corrupt a
> **multi-line** body. **The bug had almost nothing to damage until the canvas was enlarged.**

So the correct causal account is:

* `initial_masks` 8→32 is a **genuine, standalone** artifact (MBPP+ `.0899 → .3545`, +26.5 pp, on a
  path with **no** stitch at all — so it is uncontaminated by defect (c)).
* The stitch bug's **severity is created by fixing the first one**. It is +0.61 pp at c8, +4.27 pp at
  c32, +31.10 pp at c128 (the last two RAN by the closeout,
  `evidence/cells_corrected/a05_closeout_stitch_regrade.json`). It is an **interaction**, not a
  second independent 31 pp defect sitting in the published literature.
* Therefore the honest framing of the "3.95× swing" (`.122 → .4817`) is: **one config integer, whose
  correction then exposes a latent post-processing bug that the original mis-configuration had been
  masking.** Writing it as "two independent artifacts each worth more than typical method deltas"
  **overstates the second one at the published operating point by ~50×.**

### 1.5 F4 — defect (a) (the NFE mis-count) re-tested: also narrow

`len(output.history)` used as an `nfe` occurs in **one** driver (the dreamon one, now renamed to
`history_len_NOT_nfe`). `generate_evalplus_dream.py` / `_dream_alg.py` log `"nfe": args.steps` — a
declared step budget, not a history length. `smoke_dreamon.py` / `smoke_dream_coder.py` record it but
correctly name it `history_length`. **The closeout's narrow-scope claim for (a) is independently
confirmed.** F4 does not fire.

---

## §2 — Where the finding belongs: **B10**, not A01, not a new proposal

`STATUS.json:finding_that_outlives_a05:suggested_owner` says A01. **I am rejecting that**, having read
A01's `PROPOSAL.md`, its six `GATE*_VERDICT.md`, `NOVELTY_CHECK.md` and `TCODEX_AUDIT_RESPONSE.md`.

### Why not A01

A01's thesis is a **specific, falsifiable statistical protocol**: a measurement must be compared to
its own *input-blind null* (best-constant floor, permutation null, layer-order null) before being read
as capability, and it reports `reported / null / convention / calibrated residual / residual fraction`.
Its machinery is null construction, tie/tokenizer conventions, BH correction, residual fractions.

The A05 finding has **no null and no calibration in it**. "A config integer was set to a value that
crippled the baseline" is a **baseline-configuration / harness-validity** defect. There is no
input-blind reference distribution involved; the fix is *not* "compare to a floor", it is *"sweep the
knob"*. Filing it under A01 would dilute A01's one crisp methodological claim into a general grab-bag
of "ways evals go wrong" — and A01 is **already in MAJOR REVISION after an external audit retracted
two of its claims and downgraded a third**. Adding a fourth, differently-shaped claim to a proposal
that is currently narrowing its scope is the wrong move. A01's own lesson #5/#6 is that claims must be
*narrow and individually defensible*.

There is also a **novelty problem specific to A01's framing**: A01's `NOVELTY_CHECK.md` already
concedes it may not claim "benchmarks need stronger-than-chance baselines" (Balepur et al., ACL 2024)
or "construct validity is a problem" (Bean et al., NeurIPS 2025 D&B). "Baselines can be
under-configured" is adjacent to a large existing literature on baseline tuning
(e.g. the well-known result that properly-tuned baselines close claimed gains in recommender systems
and metric learning). **Not checked yet — see §4.**

### Why not a new proposal

The evidence base, after §1, is: **one integer on one model on two benchmarks, plus a one-driver
post-processing bug with a nil blast radius, whose effect at the published operating point is 1 item.**
That does not support a standalone paper-shaped claim, and standing up `A06/` for it would violate the
spirit of the promotion rule (a direction needs a real kill gate it could fail; this one has no live
hypothesis left to test — the measurement is already done and the answer is known).

### Why B10 is right

`B10-dllm-infilling-ar-dominance` already owns:
* the **same model family** (DreamOn-v0-7B, Dream-Coder, Qwen2.5-Coder AR control);
* the **same repo and run tree** (`dllm_draft`);
* the **same question shape** — B10's `NUMBER_AUDIT.md` is *literally* an audit finding that headline
  dLLM-vs-AR numbers do not survive proper controls (its C1/C2/C3 corrections: a claimed AR win is
  `p=0.635`, the ordering sign-flips against the benchmark's own gold ceiling, and the cost claim
  reverses depending on which unit you pick);
* a status of `backlog_headline_not_significant_motivation_false` — i.e. it is **already** the home
  for "this family's reported comparisons were produced by harness/protocol choices".

The A05 finding is the **full-program-generation counterpart** of B10's infilling audit, in the same
family, with the same conclusion shape. It strengthens B10's existing thesis instead of diluting A01's.

**Implemented**: added as sub-claim **`S4`** in `B10/PROPOSAL.md` + `B10/STATUS.json`, with its own
pre-registered kill gate (see §4). B10 stays in `backlog/` — it has **not** earned a node.

---

## §3 — The Scaffold provenance hole: CLOSED (option (c), 0 GPU)

`POSTMORTEM.md:123` recorded Scaffold's `.177`/`.354` as READ-only, single-round, from a wzc1-only
29 GB checkpoint, and called it "the weakest provenance link in the whole direction". Both A05
verdicts lean on it.

**The brief framed this as a choice between staging 29 GB cross-disk (~31 min at 16 MiB/s) or leaving
it open. Both were unnecessary.** The 29 GB checkpoint is only needed to *re-generate*. But
`generate_evalplus_scaffold.py` writes `solution = result.text` **verbatim, with no post-processing**,
and those final per-item programs are on disk in `runs/scaffold_*/solutions.jsonl` (29 KB / 54 KB).
Re-grading them is not an approximation of the original scoring — because there is no post-processing
step between generation and grading, **it is bit-for-bit the same scoring input the original run
used** (invariant 6). Total staged: **1.6 MB**, md5-verified after `scp -O`
(`2b1e289852d2ac5628f80674e90260e9`).

Recomputed on `.73` with the self-tested grader, all four tiers, both benchmarks:

| tier | HE+ (RAN) | HE+ (READ) | MBPP+ (RAN) | MBPP+ (READ) |
|---|---:|---:|---:|---:|
| tiny | .0000 | — | .0000 | — |
| small | .0427 | — | .1561 | — |
| **Medium** | **.1768** | **.177** ✓ | **.3545** | **.354** ✓ |
| Large | .1768 | .177 ✓ | .3254 | .325 ✓ |

**Every published Scaffold value reproduces (±0.05 pp = rounding).** `.177`/`.354` is now RAN, not
READ. The GT pickles used are the same hashes the original runs used (`fe585eb4…` for HE+,
`ee43ecab…` for MBPP+), both already resident on `.73`.

### F3: does Scaffold's own number go through the buggy stitch? **NO.**

* **Structurally**: `generate_evalplus_scaffold.py` imports nothing from the dreamon driver and
  contains no `indent`/`extract`/`combine` call. Its solution is `result.text`.
* **Empirically**: `items_reaching_indent_branch = 0` for all 8 Scaffold cells; 162/164 of Scaffold
  Medium's HE+ programs already contain a top-level `def`, so they would have short-circuited the
  buggy branch even if they had been routed through it.

**F3 does not fire.** Scaffold's `.177`/`.354` are **not** understated. The K1 margin does not move in
A05's favour.

### Verdict on un-archiving A05: **NO. A05 stays dead, and now on firmer ground than when it died.**

K1 required DreamOn to come within 5.0 pp on **both** benchmarks. With Scaffold now measured rather
than read:

| benchmark | Scaffold Medium (**now RAN**) | DreamOn best non-oracle | gap | K1 clause |
|---|---:|---:|---:|---|
| HE+ | **.1768** | .4817 (c128, corrected) | **−30.49 pp** | fires |
| MBPP+ | **.3545** | .3545 (c32) | **−0.00 pp** | fires |

Both clauses still fire, with the same margins. The one thing that *could* have revived A05 —
Scaffold being understated by its own harness — is now **measured and excluded**. I found no evidence
that A05 was killed on a mis-measured comparison, and I looked for it specifically (F3 was registered
in advance precisely so this could not be waved away).

**Residual provenance limitations (still open, honestly stated):**
* Still **single-round**. Re-grading fixes *provenance*, not *seed variance*; nobody has run Scaffold
  Medium twice. K2 (noise floor) was never run and is moot for a −0.00 pp / −30.49 pp pair.
* The 29 GB checkpoint remains wzc1-only. **What would close the remaining gap**: a second Scaffold
  Medium generation round with a different seed, on LOCAL or `.21` (wzc1, where the checkpoint lives).
  Est. ~1-2 GPU-h. **Not worth spending** — it defends a comparison already decided by 30 pp on one
  side and 0.00 pp on the other.
* `he_c512` / `mbpp_c128` / `mbpp_c512` still never completed, so DreamOn's HE+ ceiling is still only
  known to be **≥ .4817**. Cannot change K1 (which takes a max).

---

## §4 — What I did NOT do (owed work, do not record as done)

* **Novelty check for the B10 `S4` sub-claim: NOT DONE.** Semantic Scholar returned HTTP 429 on every
  query attempted through the proxy. This is a **hard blocker on promoting S4 beyond backlog** and is
  written into S4's kill gate. Must cover: baseline-tuning literature (tuned baselines erasing claimed
  gains), code-eval harness sensitivity / post-processing and prompt-format effects on HumanEval
  pass@1, and dLLM generation-length/canvas budget sensitivity. Per the repo's venue rules:
  OpenReview `venueid` + `Camera_Ready_Revision` for ICLR/NeurIPS/ICML, ACL Anthology + DBLP for the
  ACL family.
* **No re-generation of anything.** Every number is a re-grade of stored output (invariant 6). No
  cell here is labelled "replacement"; the two changed DreamOn cells are **corrected**.
* **`refine_*` and `kspan_*` arms were not re-graded.** They are structurally excluded (different
  drivers, different benchmark surfaces, no `combine_humaneval_prompt`), and 22 of them are already
  marked `.BROKEN_VERIFIER`. Reported as *structurally out of scope*, not as "verified unaffected".
* **`dreamon_heplus.r1`'s published value** is not in the results table under a separate row, so its
  `.1098` as-run has no published counterpart to check against — only r2's `.122` does.

---

## Files

* **Pre-registration** (committed before evidence): `A05_BLAST_RADIUS_PREREGISTRATION.md` (`bed7e43`)
* **Audit code** (0 GPU, re-grade only): `code/a05_blast_radius_audit.py`
* **Evidence**: `evidence/a05_blast_radius.json` — per-arm as-run vs corrected pass@1 base/plus,
  item-level gain/loss lists, parseability, `items_reaching_indent_branch`, grader self-tests
* Prior closeout evidence relied on: `evidence/cells_corrected/a05_closeout_stitch_regrade.json`,
  `evidence/a05_k1_he_stitch_bug.json`, `evidence/a05_closeout_cost_audit.json`
