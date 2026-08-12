# A05 — Structural dLLM: is there a cost regime where an explicit structural runtime beats its own family?

**Created**: 2026-08-12. **Source repo**: `/apdcephfs_wzc1/share_304376610/pighzliu_code/dllm_draft`
(wzc1) + `/apdcephfs_zwfy6/share_304376610/pighzliu_code/dllm_draft{,_104}` (zwfy6).
**Predecessor**: `proposal/backlog/B10-dllm-infilling-ar-dominance` — B10 owns the *infilling* surface
and its AR-dominance audit. A05 owns the *full-program generation* surface and the Scaffold model we
trained. They must not be merged: B10's headline died on a McNemar test, A05's has never been tested.
**GPU authorised by this document**: 0. Every gate below is scoped and costed; nothing launches on the
strength of this file alone.

---

## 0. Why this is a proposal and not an archive entry

`DLLM_SALVAGE_ROADMAP_20260808.md` §1.1 marks Scaffold full-program generation **STOP** and the
Pareto-frontier claim **RETRACTED**. Both dispositions are correct *as stated* — and both are about a
claim A05 does not make.

What was retracted: *"structural runtime owns the low-cost Pareto frontier."* That died because the
frontier omitted an AR baseline, and Qwen2.5-Coder-7B strictly dominates Scaffold on both cost axes.

What was **never tested**: whether Scaffold's advantage *over its own family* is real. The retraction
note itself says the surviving description is that "Scaffold can only be described as a diffusion-family
internal point." **An internal point with a measured 4.16× advantage is a finding about the family, not
a non-finding.** Nobody ran the control that would kill it.

So A05 is a *narrowing to what was measured*, not a revival of what was refuted. The AR ceiling is
carried forward as a **stated boundary condition in the abstract**, not hidden.

---

## 1. The observation A05 is built on

Same two benchmarks, same grader (`evalplus.evaluate`), same model scale (7B), same family
(Dream-Coder lineage):

| | HE+ | MBPP+ | NFE mean (HE+/MBPP+) |
|---|---|---|---|
| **Scaffold Medium** (ours) | **.177** | **.354** | 63.8 / 56.7 |
| DreamOn-v0-7B | .122 | .085 | 265.9 / 135.7 |
| **advantage** | **+5.5 pp (1.45×)** | **+26.9 pp (4.16×)** | **4.17× / 2.39× cheaper** |
| matched Plain-SFT @ nfe64 | .000 | — | (43,854 tok = 3.1× Scaffold's cost) |
| AR / flat-diffusion ceiling | .707 | .680 | — |

Provenance, all re-read by MAIN 2026-08-12 rather than copied from a summary table:

* Scaffold: `DLLM_RESULTS_20260807.md:447` (HE+ tier table), `:456` (MBPP+ tier table).
* DreamOn: `runs/dreamon_heplus_r2/evalplus.out`, `runs/dreamon_mbppplus_r2/evalplus.out` —
  read from evalplus's own stdout, not from a roll-up.
* DreamOn NFE: aggregated from `runs/dreamon_*.r1/metrics.rank*.jsonl` (`process.nfe`).
  **`r2` logs `nfe: null` for every item** — the NFE column is therefore r1-sourced while the
  pass@1 column is r2-sourced. This is a real mismatch and §6 D3 forbids reporting them as one row.
* Plain-SFT control: `BASELINE_STATS.md` (same base ckpt / split / corruption weighting / trainer,
  scaffold mechanism removed) — this is what rules out "Scaffold was just trained more."

**The claim A05 proposes to defend** (and to try to kill first):

> On full-program code generation under a *token-cost* budget, an explicit structural decoding
> runtime reaches quality that its own model family does not reach at any comparable cost — while
> remaining well below autoregressive models, which dominate both axes. The structural gain is
> attributable to the mechanism, not to extra training, because a matched plain-SFT control at 3.1×
> the token cost scores .000.

Note what is *not* claimed: no advantage over AR, no general dLLM advantage, no claim about infilling
(that is B10's surface).

---

## 2. What is wrong with the observation right now

Three defects, in descending severity. **All three are answerable with eval only — no training.**

### D-A (fatal if unfixed): the DreamOn baseline is harness-crippled, so `+26.9 pp` is uninterpretable

Per-item telemetry, aggregated by MAIN from `runs/dreamon_*/metrics.rank*.jsonl`:

```
generated_tokens : mean 1.90 (r1) / 2.29 (r2)   max 19-26
raw_output       : EMPTY for 133/164 (HE+ r1), 128/164 (r2), 346/378 (MBPP+ r1)
initial_masks    : 8 for every single item (no per-item headroom)
transfer_tokens  : 1 for every single item
```

DreamOn emitted a **mean of ~2 tokens** and produced literally nothing on ~80% of items. I checked
whether the 18 passing HE+ items were empty-output artifacts: they are not — all 18 generated 5–19
tokens (`solutions_eval_results.json` cross-referenced against per-item metrics, 0 of 18 with
`generated_tokens == 0`). So the score is real but the *canvas budget* is what bounds it, and
`initial_masks=8` fixed for all items is exactly the "no per-item headroom" condition that
`DLLM_SALVAGE_ROADMAP` §1.1 flags as making DreamOn's long-span behaviour **NOT ESTABLISHED**.

Consequence: `+26.9 pp` currently measures *Scaffold vs a length-controller that never engaged.*
Until the canvas is swept, this number cannot be published in either direction.

### D-B: the reverse cell is empty — Scaffold was never run on DreamOn's home turf

`grep -i scaffold KSPAN_INFILLING_RESULTS.md` → **0 hits** (verified). On infilling, DreamOn climbs
`.122 → .702`. We have six arms there (`qwen_fim`/`dreamon_oracle`/`dream_fim`/`dreamon_fim`/
`qwen_prefix`/`dream_prefix`) and **no Scaffold arm**. Claiming Scaffold > DreamOn from
full-program generation alone, while its home turf is untested, is the same one-sided-comparison
error the frontier retraction was about.

### D-C: the DreamOn baseline's own sampler noise is ~22% relative

r1 → r2: HE+ `.110 → .122`, MBPP+ `.066 → .085`. That is 1.9 pp of round-to-round movement on a
`.085` base. Any Scaffold-vs-DreamOn margin must be quoted against this, and n=2 rounds cannot
estimate it — a third round is needed just to have a df≥2 spread.

---

## 3. Pre-registered kill gate

**Register before running. A05 dies if ANY clause fires.** Thresholds fixed here, 2026-08-12,
before any A05 measurement exists.

### K1 — the advantage is a canvas artifact

Sweep DreamOn `initial_masks ∈ {8, 32, 128, 512}` plus a per-item-headroom arm
(`initial_masks = gold_tokens + 32`, an explicitly **oracle** arm, labelled as such and never used
for the headline) on both HE+ and MBPP+, same grader, same sampler.

**K1 fires if** DreamOn at its best *non-oracle* canvas setting reaches within **5.0 pp** of
Scaffold Medium on **both** benchmarks. Then the gap was budget, not structure, and the direction is
dead. Recorded either way: the emitted/gold length ratio and parseability at each setting.

### K2 — the advantage does not survive its own noise floor

Run a **3rd and 4th** DreamOn round at the best non-oracle canvas (fresh sampler seeds, same
protocol) to get `sd_round` with df≥2.

**K2 fires if** the Scaffold−DreamOn margin on MBPP+ is smaller than
`t_{0.05,df} · sd_round · sqrt(2)` — i.e. not distinguishable from round noise. HE+'s +5.5 pp is
already at risk here; MBPP+'s +26.9 pp is the load-bearing one, and if *that* fails, A05 is dead.

### K3 — the direction is one-sided

Run Scaffold on the infilling surface (D-B) with the recovered generator.

**K3 fires if** Scaffold falls below **half** of DreamOn's `.702` on SingleLine infilling *and*
K1/K2 leave the MBPP+ margin under 15 pp. Rationale: a mechanism that wins one surface by a lot and
loses the other by a lot is a task-surface result (already B10's territory, and already narrowed),
not a mechanism result. One clean win plus one clean loss is publishable; a marginal win plus a
collapse is not.

### Explicitly NOT a kill condition

* **Losing to AR.** Already known, already in the abstract, boundary not defect. The AR ceiling
  (`.707/.680`) is reported in the main table of every version of this paper.
* **A prior paper covering part of the surface.** Per user directive 2026-08-12 (「小的点被抢走了
  并不意味着我们需要 narrow」), a literature collision produces a citation obligation, not a
  scope cut. Only the gates above can kill A05.

---

## 4. Cost, and why the gates are cheap

Measured anchors from the archived DreamOn runs (`metrics.rank*.jsonl`, `elapsed_seconds` summed):

| leg | measured basis | est. GPU-h |
|---|---|---|
| K1 canvas sweep, 5 settings × 2 benchmarks | HE+ r1 = 659 s, MBPP+ r1 = 919 s wall on 8 shards; long canvases cost more, budget 4× | ~14 |
| K2 rounds 3+4 at best canvas | 2 × (659+919) s × 4× | ~7 |
| K3 Scaffold on SingleLine (n=1033) | `qwen_fim` arm's own runtime × diffusion overhead | ~6 |
| **total** | | **~27 GPU-h ≈ 3.4 h on one 8-card node** |

No training. All five model checkpoints already on disk. This is a **one-node, half-day** gate for a
direction currently marked STOP — which is the entire argument for running it.

---

## 5. Asset audit (verified on both disks, 2026-08-12)

Per the two-disk rule, "missing" is only asserted after checking wzc1 **and** zwfy6.

| asset | wzc1 | zwfy6 | note |
|---|---|---|---|
| `models/DreamOn-v0-7B` | ✅ | ✅ (both `dllm_draft` and `_104`) | |
| `models/Qwen2.5-Coder-7B` | ✅ | ✅ | AR control |
| `models/Dream-Coder-v0-{Base,Instruct}-7B` | ✅ | ✅ | family reference |
| **`models/Scaffold-v0-stage1-7B` (29 GB)** | ✅ | ❌ **NOT PRESENT** | **our model — wzc1-only** |
| `data/infilling/HumanEval-SingleLineInfilling.jsonl` | ✅ | ✅ | K3 input |
| `data/evalplus/` | ✅ | ✅ | K1/K2 input |
| `scripts/generate_infilling.py` | ❌ | ✅ **in `dllm_draft_104`** | see below |
| `scripts/score_infilling.py` | ❌ | ✅ in `dllm_draft_104` | |
| `scripts/generate_evalplus_dreamon.py` | ✅ | — | K1/K2 driver |

### ⚠️ The roadmap's "repository blocker" is stale — verified false

`DLLM_SALVAGE_ROADMAP_20260808.md` P1-C states: *"Repository blocker:
`scripts/generate_infilling.py` is missing."* **It is not missing.** It is 392 lines, on zwfy6, in
`dllm_draft_104/scripts/`, and it already implements all six arms
(`ARMS = ("dreamon_fim", "dreamon_oracle", "dream_fim", "dream_prefix", ...)`), `--initial_masks`,
the oracle/non-oracle split, and first-line grading for unidirectional arms. Its companion
`score_infilling.py` (197 lines) is there too.

The roadmap's author searched one disk. This is the failure mode recorded in
`memory/subagent-audit-must-specify-cross-disk.md` and
`memory/two-disk-rule-applies-to-main-too.md`, and it had **blocked P1-C since 2026-08-08**.
K3 is therefore *not* gated on writing a generator.

### Scheduling consequence

Scaffold's 29 GB checkpoint is wzc1-only, and the measured cross-disk rate on this cluster is
**16 MiB/s** (A04 agent measured 3 GiB in 183 s, 2026-08-12) → ~31 min to stage it. So:

* **K1 + K2** (DreamOn only) → run on **zwfy6** (`.73`/`.82`/`.104`), zero transfer.
* **K3** (needs Scaffold) → either run on **LOCAL/.21** when SparseForge frees them, or pay the
  ~31 min `scp -O` once. Both acceptable; prefer whichever node is free first.
* The generator must move the other way (zwfy6 → wzc1) if K3 runs on wzc1. `scp -O` + md5.

---

## 6. Protocol invariants (binding on every A05 run)

1. **Grader is `evalplus`, never a hand-rolled verifier.** `DLLM_RESULTS_20260807.md` Retraction 1:
   the old verifier discarded return values and scored empty stubs as 7/7 pass, inflating
   visible-pass to 77% against a true 20.7%. Every A05 number goes through
   `evalplus.eval.untrusted_check` with per-invocation self-test (canonical PASS, stub FAIL).
2. **Report `tokens_fed` AND NFE, never NFE alone.** Retraction 3: NFE-as-cost inflated the Large
   tier by 2.85×/3.69× and moved it off the frontier once fixed.
3. **Never merge an r1 NFE with an r2 pass@1.** They are different runs; `r2` logs `nfe: null`.
   Any row mixing them must say so inline (as §1's table does).
4. **Oracle arms are labelled `*_oracle` and excluded from headline comparisons.** Being told the
   gold length is worth ~5.7 pp on the infilling surface; it is not a capability.
5. **The AR ceiling appears in the main table.** Not an appendix, not a footnote.
6. **Sharded runs assert shard completeness before merge** — expected item count per benchmark
   (HE+ 164, MBPP+ 378, SingleLine 1033), 0 duplicate `task_id`, 0 nan. A silent 5/8 merge has
   already destroyed one comparison in this project.
7. **`mask_expansion` / `delete_eos_token` are NOT parameters** — `DLLM_RESULTS_20260807.md:342`:
   they are silently swallowed by `**kwargs` in DreamOn's code. Any doc claiming a number was
   obtained "with mask_expansion on" is void. A05 must not reintroduce that phrasing.

---

## 7. Relationship to B10, and why both exist

| | B10 | A05 |
|---|---|---|
| surface | HumanEval-SingleLineInfilling (n=1033) | HE+ / MBPP+ full-program generation |
| subject | do dLLMs beat a matched AR control at infilling? | does an explicit structural runtime beat its own family under a token budget? |
| verdict | **headline dead** (McNemar p=0.6353, 5/1033) | **untested** |
| surviving asset | suffix-visibility gain (AR +23.14 pp, diffusion +29.91 pp, difference +6.78 pp, CI [+4.07,+9.49] excludes zero) | Scaffold vs DreamOn +5.5/+26.9 pp, ungated |
| A05 uses from it | the 6-arm infilling harness + the `.702` DreamOn reference for K3 | — |

B10 stays in `backlog`. A05 goes to `active` because its gate is cheap, its assets are on disk, and
its blocker turned out to be a stale doc rather than missing code.

---

## 8. Novelty check status

**NOT DONE.** Required before any GPU under this proposal, per the promotion rules in
`CODEBUDDY.md`. Specifically to check: structural/hierarchical decoding runtimes for diffusion LMs,
length-controller work post-DreamOn, and any published Scaffold-like meta-token decoder. Per the
standing directive this check produces a **citation-obligation list**, and only §3's gates can kill
the direction. Venue verification must use the right registry per family — OpenReview `venueid` for
ICLR/NeurIPS/ICML, ACL Anthology + DBLP for the ACL family (including Findings).

---

## 9. Immediate next action

Run **K1** on `.73` (idle as of 2026-08-12, verified 0%/0 MiB on all 8 cards). It is the clause most
likely to fire, it needs no Scaffold checkpoint, and it costs ~14 GPU-h. If K1 fires, A05 is dead for
~1.8 h of one node and the `dllm_draft` repo gets a clean disposition instead of an open STOP.
