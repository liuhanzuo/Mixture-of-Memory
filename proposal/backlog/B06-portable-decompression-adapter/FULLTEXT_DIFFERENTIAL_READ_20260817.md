# B06 — Full-text differential read of SeDeM / KV Packet / RAC

**Date**: 2026-08-17
**Leg**: `lifecycle_why_20260817` item (2) — "full-text differential read of SeDeM 2608.00311,
RAC 2608.04991 and the newly found KV Packet 2604.13226, all three of which were adjudicated
from **ABSTRACTS ONLY**."
**Cost**: 0 GPU, 0 GPU-seconds, 0 ssh. PDFs fetched over `hy-proxy.woa.com:3128`, text extracted
with PyMuPDF under `/opt/conda/envs/torch-base/bin/python`.
**Supersedes**: nothing is deleted. This file *narrows* `surface_e_closed_20260817.revised_finding_on_surface_e`
and *challenges* `RELATED_WORK.md:103` differentiator (a) and (b). See §6 for the exact
sentences it puts at risk.

---

## 0. Provenance of the artefacts read (so a later agent can re-derive, not re-download blindly)

| arXiv id | title (from page 1 of the PDF itself, not from metadata) | file | sha256 | pages | extracted chars |
|---|---|---|---|---|---|
| 2608.00311 | *SEDEM: Selective Decompression of Hidden-State Memories for Long-Context Question Answering* (Haghifam, Cong, Sun; UCLA) | `/tmp/ft_2608.00311.pdf` | `ae10b2c682d57f78cad06bf18fb8172d3f27f1cc42322cdc08052e2efc523970` | 20 | 81264 |
| 2608.04991 | *RAC: Reference-Aware Activation Compression for Communication-Efficient Split LLM Inference* (Yang et al.; Tianjin Univ.) | `/tmp/ft_2608.04991.pdf` | `35e812dfc6e51608f22090fd7b4657e6ac4f1fa6ef24029b14ef6762f38b85fc` | 10 | 50450 |
| 2604.13226 | *KV Packet: Recomputation-Free Context-Independent KV Caching for LLMs* (Chen, Zhang, Yin, Zhuo, Li, Schlichtmann; TUM/TU-Darmstadt/ZJU) | `/tmp/ft_2604.13226.pdf` | `d0a03684db4b2f0c5c57367a6cbd6cedb4f5657d63459043b30b228350c9cba4` | 11 | 37132 |

⚠️ `/tmp` is wiped by node restart (`memory/persist-artifacts-on-wzc1-or-diskb.md`). The sha256
values above are the durable part; re-fetch with
`http_proxy=http://hy-proxy.woa.com:3128 curl -sL https://arxiv.org/pdf/<id> -o <file>`
and check the hash before trusting any quote here.

---

## 1. FIRST CORRECTION — the "42 occurrences of LoRA" figure handed to me is wrong

The task brief (and the scout that produced it) stated SeDeM's full text "contains 42 occurrences
of `LoRA`". Measured:

```
LoRA (exact, case-sensitive):                34
lines containing 'LoRA':                     33
case-insensitive /\w*lora\w*/ tokens:        42   <-- this is where 42 came from
  broken down:  LoRA  x34
                rlora x4     <- the rank hyperparameter symbol r_lora
                alora x3     <- the scaling hyperparameter symbol alpha_lora
                Rrlora x1    <- a subscripted matrix symbol
```

`42` is the **case-insensitive token count**, and 8 of those 42 are **math symbols**
(`r_lora`, `alpha_lora`, `R_{r_lora}`), not prose mentions of the method. The honest number is
**34 exact `LoRA` occurrences on 33 distinct lines**. This does not change any conclusion below —
but it is exactly the class of artefact the risk brief warned about (item 5), and the count was
in fact a *pointer*, not a result. Every quote below was read in its printed context.

**The same case-insensitive substring trap fires much worse on RAC**, and it nearly produced a
false claim in this very file: a `grep -i lora` on RAC returns **1** hit, which is the word
**"Explo·ra·tory"** in "D. Exploratory Joint Bandwidth Sensitivity". RAC contains **zero**
mentions of LoRA. Anyone quoting a case-insensitive count as evidence of a method's presence is
counting English words.

---

## 2. SeDeM (2608.00311) — the two sections that occupy B06's variable

### 2.1 §7.4 "Effect of LoRA on Decoder LLM" (PDF p.8-9, Table 6 on p.9)

Verbatim, read in printed context:

> **7.4 Effect of LoRA on Decoder LLM** — To isolate decoder adaptation, we compare a SE-DEM-only
> model with no decoder LoRA against the same model with decoder LoRA in the **selector-disabled
> setting**. As shown in Table 6, the SEDEM-only model remains strong on both datasets. These
> results indicate that the method's gain is not merely a decoder-LoRA effect; LoRA provides an
> additional adaptation benefit when useful, but the learned SEDEM itself carries the main signal.

Table 6 (p.9), verbatim including its caption:

| Dataset | SEDEM + frozen dec. | SEDEM + adapted dec. |
|---|---|---|
| HotpotQA-Dist. | 62.55 | 69.78 |
| QASPER | 23.36 | 24.37 |

> *Table 6: Decoder adaptation ablation in the **selector-disabled setting**. Both variants
> condition on all segment reconstructions, so these results are **not efficiency-comparable to
> the top-K setting**. We report F1 on HotpotQA-Distractor and QASPER.*

**What this is, precisely.** It is a **single-variable decoder-side-LoRA on/off toggle**, with the
selector disabled, on a decompressed-hidden-state readout. That is *structurally the same
experiment as B06's own headline* (HCache path, `no_retrieval=true` in both arms, ± distilled Read
LoRA, `j=12`, Qwen3-8B). B06's headline is `16.69 -> 39.81` on LoCoMo Judge_1:4; SeDeM's is
`62.55 -> 69.78` on HotpotQA-Distractor F1 and `23.36 -> 24.37` on QASPER F1.

**What it is NOT** — and this is the part a confirmation-shaped read would miss in the other
direction, i.e. it is the part that *saves* B06:
* SeDeM's LoRA is trained **jointly with the compressor in Stage 2** (see §2.3 below). It is not
  held fixed and moved to a different cache-production rule. Both Table 6 arms use SeDeM's **own**
  memories.
* Both arms are the **same compressor**. There is no second compressor anywhere in the paper.
* SeDeM's framing of the ablation is *defensive* — its purpose is to show the gain is **not** a
  LoRA effect ("the learned SEDEM itself carries the main signal"). B06's claim is the opposite in
  sign of interest: that a reader-side repair carries transferable signal.

### 2.2 §7.3 "Out-of-Distribution Transfer Test" (PDF p.8)

> Under matched two-stage training, SEDEM reaches **20.02** token-F1 on zero-shot QASPER, compared
> with **11.30** for ICAE. This suggests that decompressed hidden-state memories transfer better
> than direct memory-token conditioning when the test domain and document structure shift.

The transferred axis is **the evaluation DOMAIN** (SlimPajama compression-pretrain → multi-task
Wikipedia-style QA adaptation → zero-shot QASPER, scientific papers). It is **domain transfer of
one pipeline**, not transfer of a fixed reader module across **cache-production rules**. B06's
residual claim is the latter. §7.3 does **not** occupy it.

A bounded search of the whole SeDeM text for portability vocabulary returns only 7 hits
(`cross-backbone|different backbone|portab|transfer|interoperab|swap`), and all 7 are §7.3 /
its forward references / the conclusion. **The word "portable" and the phrase "different
backbone" do not appear in SeDeM at all.**

### 2.3 What SeDeM actually trains, from §4 "Training Objective" (PDF p.5)

> Stage 1 trains the compressor and decompressor using context-window next-token reconstruction,
> decoder distillation, and hidden-state reconstruction. **Stage 2 trains the full
> query-conditioned pipeline** … In Stage 2, we **train the selection module, continue updating
> the compressor and decompressor, and activate the decoder LLM LoRA adapters.**

Stage-1 losses are `L(1) = Lctx + λ_distill·L_distill + λ_rec·L_rec`, with a hidden-state
directional-cosine term `Ldir` and a pooled term `Lpool` against `H^(ℓ_extract)·W_in`.
LoRA rank is `r_lora = 64` (α = 32).

---

## 3. KV Packet (2604.13226) — this is the one that is closer than the abstract suggested

The abstract-level adjudication (`RELATED_WORK_ADJUDICATION_20260817.md` §3.2) listed three
differentiators. Full text verdict on each:

| # | abstract-level differentiator | survives full text? |
|---|---|---|
| 1 | "Cache level: KV Packet operates on the KV cache; B06 on mid-layer hidden states at `j=12`" | **SURVIVES.** §3 is entirely about concatenating **precomputed KV caches** with RoPE realignment. No hidden-state-at-depth-j object anywhere. |
| 2 | "What is trained: KV Packet trains soft tokens wrapping the cache (cache side); B06 trains rank-32 LoRA on upper layers (reader side)" | **SURVIVES, and the paper says so itself.** §2.3: "KV Packet similarly employs a small set of trainable tokens; however, instead of task adaptation from scratch, our goal is **cache composition**: training Header/Trailer tokens that absorb boundary artifacts". Setup: `Nh = Nt = 8` adapter tokens, float32, 30 epochs AdamW, 256–512 samples. The backbone is never adapted. |
| 3 | **"Portability is NOT tested. Its evaluation is Llama-3.1 + Qwen2.5 on its OWN packet format."** | **DOES NOT SURVIVE AS WRITTEN.** See §3.1. |

### 3.1 §4.4 "Compatibility with KV Compression" (PDF p.7-8, Fig. 4) — the real overlap

Verbatim:

> Fig. 4 evaluates Llama-3.1-8B-Instruct under **five SOTA compression methods** (CUR, KVzap,
> LeverageScore, TOVA, and random pruning) from the KVPress library, at compression rates of
> **10%–50%**, under three configurations: KVPacket Normal (compression over full wrapped cache),
> KVPacket Keep Filler (filler tokens excluded from pruning), and Single Cache (compression over
> full concatenated context). Notably, KV Packet proves significantly more robust than the
> baseline under random pruning, maintaining a much flatter performance profile. … indicates that
> **our trained fillers are inherently resilient to KV compression.**

And the motivating paragraph:

> KV Packet natively bypasses these bottlenecks entirely by treating each document cache as an
> **opaque unit** and never re-evaluating compressed hidden states; it **integrates seamlessly
> with off-the-shelf unstructured KV pruning.**

**This is a trained repair module held fixed while the cache-production rule is varied across
5 algorithms × 5 compression rates.** The training setup (§4.1) trains adapters per *dataset*
(Biography / NIAH / MusiQue / HotpotQA / Universal mixture) with no mention of retraining per
compression method, and §4.4's claim ("our trained fillers are **inherently** resilient") only
makes sense if the fillers are the same ones. So the differentiator "portability is not tested"
is **too strong**: KV Packet does test resilience of one trained module to a changed cache
production rule.

**Where B06 still differs, on the full text:**
1. **Type of variation.** KV Packet varies a **post-hoc pruning mask applied to a cache produced
   the same way**. B06 varies the **cache-production mechanism itself** (CoMem compression pack vs
   HCache mid-layer checkpoint — two different write policies, not two prunings of one write).
   Pruning-invariance is a weaker property than write-policy-invariance.
2. **Which side is held fixed.** KV Packet's fixed module is on the **cache side** (8 soft tokens
   wrapping the cache). Even under §4.4 the *reader* is untouched. B06's fixed module is the
   **reader-side LoRA**, which is the only thing that could constitute a "shared readout skill".
3. **Object level.** Still KV tensors vs. mid-layer hidden states (differentiator 1, which holds).
4. **§4.5.1 is a different axis again.** "Cross-Domain Generalization and Universal Alignment"
   (Table 1) trains adapters on one *dataset* and evaluates on the others — domain transfer, same
   format. Biography-trained → HotpotQA collapses 0.96 → 0.18; the Universal mixture is the only
   stable one. That is an argument about **training-data diversity**, not about codec portability.

### 3.2 Consequence for the MUST-NOT-CLAIM list — one item must be tightened further

`surface_e_closed_20260817` already recorded that KV Packet forecloses the
*adapter-instead-of-recompute* framing. The full text forecloses **one more thing**:

> ❌ B06 may **not** claim "we are the first to hold a trained cache-repair module fixed while the
> cache-production rule changes." KV Packet §4.4 does that across 5 pruning algorithms at 2026-04.
> B06's surviving, narrower claim is: **a fixed READER-SIDE adapter, trained under one WRITE
> policy, still repairs a structurally DIFFERENT write policy** (compression-pack → mid-layer
> checkpoint), which is a change of *what produced the cache*, not a mask over one cache.

---

## 4. RAC (2608.04991) — full text moves it FURTHER away, not closer

Read in full (10 pages, headings: I Introduction / II Background / III Motivation / IV RAC Design
/ V Evaluation / VI Related Work / VII Conclusion).

RAC is a **split-inference activation CODEC** for the local↔cloud boundary
(LOCAL head → CLOUD middle → LOCAL tail). Its object is the **activation tensor crossing a
network link**, and its mechanism is **grouped affine alignment + calibrated residual +
quantization** against a *reference* activation (previous-turn / intra-turn / cross-turn matched
reference). Its metrics are latency + accuracy under bandwidth (Tables I-II:
HellaSwag/commonsense/math on Qwen, Llama and GLM up to 70B-Instruct).

`RELATED_WORK.md` reads RAC as training-free. **It is not, quite — and I had to correct my own
first draft of this section, which is why the exact scope matters.** RAC trains exactly one
thing, in §IV-A-3 "Predicted References for Decode" (p.5):

> …each boundary uses a dedicated **one-layer causal Transformer trained on consecutive
> activations from the corresponding target layer.** Training records target-layer activations
> and optimizes **next-activation prediction** within a bounded history; at inference, the
> predictor retains this bounded causal state and uses a key-value cache. … Separate parameters
> φ_b account for the observed difference between uplink and downlink similarity patterns.

So RAC has a **trained per-boundary next-activation PREDICTOR** whose only job is to supply a
better *reference tensor* so the residual quantizes to fewer bits (Fig. 4: cosine-similarity gain
0.156–0.201 uplink, 0.059–0.118 downlink over reusing the previous activation). Its Related Work
states the design intent explicitly: "**RAC leaves the partitioned LLM unchanged**, calibrating
the boundary codec offline."

**What RAC still does not contain** (verified by exhaustive token search, not by skim):
* **`LoRA`: 0 occurrences.** `adapter`: 0. `distill`: 0. `low-rank`: 0. `fine-tun`: 1, and that
  one is a citation-context mention, not RAC's own method.
* No module inserted into the frozen LLM's forward path; no reader-side repair; no cached-context
  reuse across queries. RAC's "compression" is *communication* compression of a **transient**
  boundary tensor, not *context* compression of a **persisted** one — its own Related Work
  distinguishes itself from "state offloading, prompt reuse, and KV-cache transport" on precisely
  that axis.

**RAC is hereby downgraded from ADJACENT to DISTANT for B06's purposes.** It should stay cited
(it is the closest work on *hidden-state-as-transmittable-object*, and its trained predictor is a
genuine trained-module-in-the-hidden-state-path data point), but it constrains **nothing** on
B06's claim surface: it never asks a frozen reader to consume a differently-produced cache.

---

## 5. Net effect on B06's claim surface

| claim | before this read | after |
|---|---|---|
| "we introduce a decompression adapter for cached intermediate states" | already foreclosed (SeDeM + ICAE) | **still foreclosed** |
| "a trained adapter is a cheaper alternative to recomputation-based cache repair" | foreclosed by KV Packet (abstract) | **still foreclosed, confirmed in §3 + §2.3 of KV Packet** |
| "decoder-side LoRA on/off is our single-variable isolation of readout repair" | believed open | ⚠️ **OCCUPIED by SeDeM §7.4 as an experimental DESIGN** (selector-disabled, ±decoder LoRA). B06 must present its ±LoRA toggle as a *replication of a known ablation design on a new axis*, not as a novel isolation. |
| "one fixed trained repair module survives a change of cache production rule" | believed open | ⚠️ **PARTIALLY OCCUPIED by KV Packet §4.4** (5 pruning algorithms, "inherently resilient"). B06 must say **write-policy** change, not "cache-production-rule" change, and must contrast against pruning-invariance explicitly. |
| "one fixed **reader-side** adapter, trained under **one write policy**, repairs a **structurally different write policy**" | the residual claim | **STILL OPEN.** No paper read here holds a reader-side module fixed across two write policies. SeDeM has one compressor. KV Packet fixes a cache-side module. RAC has no trained module. |

**So `surface_e_closed_20260817.revised_finding_on_surface_e` ("the opening is the TRANSFER
EXPERIMENT, not the mechanism and not the framing") is CONFIRMED IN DIRECTION but must be
NARROWED in two places**: the *transfer experiment* is only open in the **reader-side ×
write-policy** cell, because (i) the ±LoRA isolation design is SeDeM's, and (ii)
fixed-module-across-changed-cache-rule is KV Packet's when the change is a pruning mask.

---

## 6. Sentences now at risk on disk (nothing edited — listed for the owner)

1. `RELATED_WORK.md:103`, SeDeM row: "SeDeM's decompressor is trained **jointly with its own
   compressor** and is a *component of one pipeline*". **This half is CORRECT** (§4, Stage 2
   updates compressor + decompressor + selector + LoRA jointly). But the row's differentiator
   **(b)** — "SeDeM's selector is *in* the path; B06's decisive property is `no_retrieval=true` in
   both arms" — **is FALSIFIED for §7.4**: Table 6 is explicitly the **selector-disabled**
   setting, i.e. SeDeM has a no-retrieval arm too. Differentiator (b) must be rewritten to say
   *SeDeM's selector-disabled arm exists but uses the same compressor in both arms*.
2. `RELATED_WORK_ADJUDICATION_20260817.md` §3.2 differentiator **3** ("Portability is not tested.
   Its evaluation is Llama-3.1 + Qwen2.5 on its own packet format") — **too strong**; §4.4 varies
   the cache production rule across 5 pruning methods. Needs the narrower wording in §3.2 above.
3. `RELATED_WORK_ADJUDICATION_20260817.md` §5.1 item 1 and
   `STATUS.json.drift_resolution_leg_20260815.highest_value_next_0_gpu_followup` — the ~1439
   record decision rule is **arithmetically wrong**; see `JUDGE_CACHE_GATE_RESTATED_20260817.md`
   in this directory (written by the same leg).
4. RAC's `relation: ADJACENT` in `RELATED_WORK.md` / the adjudication — downgrade to DISTANT per §4.

**I did not edit any of the four.** They are append-only-adjacent artefacts owned by MAIN; §6 is
the punch list.

---

## 7. What this leg does NOT establish

* It does **not** run B06's second-compressor experiment. Kill condition 3 remains untested and
  remains the only GPU-bound item.
* It read **three** papers' full text. It did **not** re-run the arXiv query sweep, so a fourth
  paper occupying the reader-side × write-policy cell would not have been found here.
* Table 6 / Fig. 4 numbers are transcribed from **extracted PDF text**, not from the authors'
  released artefacts. Two-column extraction interleaves columns; I verified each load-bearing
  quote appears on the printed page claimed (SeDeM §7.3/§7.4 → p.8, Table 6 → p.9;
  KV Packet §4.4 → p.7-8, Table 1 → p.8) but I did not cross-check against any released code.
* Fig. 4 of KV Packet is a **figure**; its per-point values are not in the text layer. My reading
  of §4.4 rests on the authors' prose claim ("inherently resilient", "flatter performance
  profile"), not on digitised curve values.
* Whether KV Packet retrains adapters per compression method is **inferred** from §4.1 (which
  lists per-dataset training only) plus the word "inherently" in §4.4. The paper does not state
  it in a single explicit sentence. If it did retrain, differentiator 3 would survive after all
  and §3.2's tightening would be unnecessary — this is the one load-bearing inference here.

---

## 8. What was written to STATUS.json, and two reader traps I hit doing it

Appended (nothing edited; verified by diffing every pre-existing key against a backup — 0 changed,
0 removed, prefix order preserved):
`fulltext_differential_read_20260817`, `judge_cache_gate_restated_20260817`,
`next_gate_gpu_20260817_{b,c,d}`, `next_gate_executable_20260817_a`,
`lifecycle_20260817_b`, `lifecycle_why_20260817_b`.

`lifecycle` stays **ready_cpu**. No clearance key was added — in particular **not**
`related_work_status`, which `DRIFT_RESOLUTION_VERDICT.md` §6.1 measured as flipping B06 to
`ready_gpu` because `"audited"` is in `ready_queue.py`'s `VERDICT_CLEARED` and B06 has
`live_blockers=[]`.

**Trap 1 — the free-marker hazard fired on my own key, then failed to fire on my own fix.**
`_next_gate_is_free` (ready_queue.py:309) substring-scans the dated cost value. My first cost key
wrote the GPU estimate as a digit range whose upper bound ends in zero, immediately before the
GPU-hour unit — that matched a marker and re-pinned B06 as free even though the key *said* the
next step needs a card. The documented hazard (docstring at :328-336) warns only about narrating a
past free leg; **a numeric range is a second, undocumented way in**. Then my *replacement* used
"needs no card" and "none", neither of which is in the marker list, so a genuinely free next step
would have read as needing GPU. **The hazard is two-sided.** Assert on the *expected* firing
outcome, not on the absence of markers. (A third variant: quoting the offending substring inside
the correction reproduces the defect — a pre-write assertion caught that one.)

**Trap 2 — a temp-copy test invented a defect that does not exist.** Testing the dated gate key on
a `/tmp` copy of the proposal dir printed the problem `RELATED_WORK.md absent (blocks PROMOTION)`.
That is an artefact of my copy: I copied `STATUS.json` but not `RELATED_WORK.md`. On the real path
`rq.read_one` returns `related_work_md=True`, `problems=[]`, `novelty_checked=True`. Had I reported
that line, I would have re-raised the exact false-absence claim this proposal has already been
burned by twice. **A sandbox that omits a file will report that file missing.**

**Reader limitation I could NOT fix from here, for the next agent.** The queue's printed `cost:`
line for B06 still shows the old `gpu_cost_estimate` dict ("0 GPU-h. Two of the three relevant
judge caches are on wzc1…  (NO BASIS)"), *not* my dated `next_gate_gpu_20260817_d`. Reason
(ready_queue.py:892): the printed line resolves `["gpu_cost_estimate", "cost",
"cost_to_first_result"]` with **no dated slot**, while `dated_cost_keys` is consulted only inside
`_next_gate_is_free`. So the dated cost keys steer the *scheduling decision* but not the *displayed
cost*. This is the same missing-dated-slot defect class already fixed four times in that file
(lifecycle, lifecycle_why, next_gate cost-for-free-check, novelty_verdict) — it simply has not been
fixed for the display path. Fixing it means editing `ready_queue.py`, which is outside this leg's
scope; recorded here so nobody reads the stale printed cost as B06's current estimate.

Also verified: `proposal/check_stale_absence_claims.py` (rc=1) flags exactly **one** stale absence
assertion repo-wide, and it is **B09**'s `RELATED_WORK.md` (33942 bytes on disk), not B06's. My
appends introduced none.


