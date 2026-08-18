# ADJUDICATION — N1–N5, the five proposals nobody had tried to kill

**Written**: 2026-08-16 · **GPU used: 0** · **GPU authorised: 0** (all 40 cards on all 5 nodes
occupied for ~15 h) · **Inputs**: 5 proposals × (1 firability precheck + 3 independent refutation
lenses: prior-art / confound / decisiveness) = 20 reports, plus my own re-derivation of every
load-bearing number from disk and from primary text.

**Why this document exists**: N1–N5 were proposed in `TCODEX_REVIEW2_20260815.md` §4 by the same
model that reviewed P1–P7. Nobody had tried to kill them. `AUDIT_VERDICT_20260815.md` killed 7/7
earlier proposals, **5 of them because the pre-registered kill gate could not fire**. This document
holds N1–N5 to that same standard.

Provenance convention: **RETRIEVED** = I fetched/opened it this pass. **COMPUTED** = I derived it
myself this pass from something RETRIEVED. **CARRIED** = taken from a lens report without
independent re-verification (always labelled). **UNVERIFIED** = no authority confirmed it; the
authority tried is named in §8.

**Network state, established BEFORE any absence claim in this document.** Proxy exported on two
separate lines; `curl -sL`. (a) `export.arxiv.org/api/query?id_list=2502.09992` → `http=200
size=3102`, `<title>arXiv Query: ...id_list=2502.09992...>`. (b)
`api2.openreview.net/notes/search?term=DAEDAL%20variable%20length%20denoising&limit=5` → `http=200
size=36071`, first hit **"Beyond Fixed: Training-Free Variable-Length Denoising for Diffusion Large
Language Models" | ICLR 2026 Poster | `ICLR.cc/2026/Conference`** — independently re-confirming the
ground-truth DAEDAL venue. Both controls green, so the NOT-FOUND results below are meaningful *as of
this pass*.

---

## 1. Bottom line

**Zero of five survive.** All five were refuted **3/3** by lenses with independent attack surfaces —
so none died of one reviewer's taste. The failure is the same one that killed the previous batch,
and it recurs at a **higher** rate:

| batch | proposals | gate cannot fire / fires against itself | survivors |
|---|---|---|---|
| P1–P7 | 7 | 7/7 (5 cannot fire, 1 fires against, 1 closed by target paper) | 0 |
| **N1–N5** | **5** | **5/5** (1 cannot fire, 1 UNDEFINED, 3 fire against themselves) | **0** |

The distribution shifted in an informative way. In the P-batch the dominant defect was gates that
were **unfalsifiable**. In the N-batch the dominant defect is gates that **were already answered,
against the proposal, by data the proposal itself points at** — N4 by a column TAD ships in its own
launch script, N5 by two independent published replications, N1 by a 47.7 pp lever published in
DreamOn's own Table 2. **That is a cheaper defect to catch than the P-batch's, and nobody caught it,
because nobody spent the 0-GPU hour of literature arithmetic before pre-registering.**

**Do not queue any of the 24–99 GPU-h these five ask for.** The honest first action is 0 GPU, and it
is not any of N1–N5: it is the **latency-irreproducibility note** that fell out of refuting N3, which
is the only genuinely new, non-preempted, decision-changing result the whole exercise produced (§5).

**One methodological result worth more than the five proposals combined**: three of the five gates
were decided by data already on disk or already published. The recurring root cause is that
**pre-registration was written before the 0-GPU literature-and-disk arithmetic, not after.** §7 makes
that a standing rule.

---

## 2. Verdict table

| id | claim (short) | GPU asked | gate fires? | refuted /3 | verdict |
|---|---|---|---|---|---|
| **N1** | AR→diffusion conversion changed how a matched lineage uses right context (Δ_surface ≥ 5 pp, ≥70 % retention) | 2–4 GPU-h | **NO** — all 3 clauses miss; the **zero-effect null of clause 1 is +10.49 pp**, already above its own 5.0 pp bar | **3/3** | **DEAD** |
| **N2** | ≥ half of DreamOn's c32→c128 HE+ gain is refinement compute, not space | 4–6 GPU-h (真 ~12–46) | **UNDEFINED** — treatment arithmetically impossible on ≥ 50.6 % of items; cap vector on no reachable disk; named mediator ≡ 0 | **3/3** | **DEAD** |
| **N3** | dLLM/AR latency ratio ≥ 20 % lower on B200 than H20 at matched quality | 8–12 GPU-h | **FIRES AGAINST ITSELF** — and worse, the statistic **cannot clear its own measured noise floor** (2.05× vs a 1.25× threshold) | **3/3** | **DEAD as written; strongest residue (§5)** |
| **N4** | ≥ half of a few-step distilled checkpoint's 8-step gain is sampler-specific | 6–10 GPU-h (真 ~12–46) | **FIRES AGAINST ITSELF at 0 GPU** — retention is **371.1 % / 148.3 %**, not < 50 % | **3/3** | **DEAD** |
| **N5** | ≥ half of LLaDA-1.5's HE+ gain is length policy, not semantics | 6–10 GPU-h | **FIRES AGAINST ITSELF at 0 GPU** — clause 1 fires **4/4** at N5's own canvases, in **two** independent published replications | **3/3** | **DEAD** |

**Survivors: 0 / 5.**

### 2b. Would a REPAIRED gate make it viable? (repair is 0 GPU, so this must be asked)

| id | gate repairable? | does the repaired gate make the DIRECTION viable? |
|---|---|---|
| **N1** | **Yes, trivially** — re-register on the log-odds or headroom-recovered DiD, whose zero-effect value is 0 by construction. | **No.** The repaired statistic reads **−0.0218** (log-odds) and **−5.99 pp** (headroom) today, i.e. **it fires against N1 on day one at 0 GPU**. Repair converts an unfalsifiable gate into a falsified one. |
| **N2** | Partly — restrict to the arithmetically feasible stratum (≤ 81/164) and disclose it pre-data. | **No.** At that n the 11.28 pp threshold has no power, and the estimand itself dissolves: **95.7 % of the 22.56 pp target quantity is the empty-output channel**, which the two-way space/compute partition has no cell for. |
| **N3** | **Yes** — drop matched-quality, drop the AR comparator, freeze both legs at torch 2.11, require ≥ 3 interleaved exclusive-tenancy replicates. | **Yes, but as a different claim.** The repaired version is a latency-**reproducibility** result, not a hardware-crossover one. This is the batch's one real residue (§5). |
| **N4** | Yes on paper — NFE-match both arms under one code path. | **No.** That design no longer contains TAD's release schedule, so it cannot answer "is the gain sampler-specific". And its own target paper's Table 3 says the compute axis is near-flat (2.1× forwards ↔ **0.6 pp**). |
| **N5** | Yes — but there is nothing left to gate. | **No.** At N5's canvases the raw gain is **negative or zero** in both replications, so there is no positive gain for a length mediator to decompose, and the mediator's between-arm variance is **0 tok** (c128) / **+4 tok** (c512). |

---

## 3. Per-proposal detail

### N1 — DEAD. The zero-effect value of the statistic already passes the gate, and the canvas is a 47.7 pp lever.

**Refuted 3/3** (prior-art, confound, decisiveness). All three agree on the arithmetic; I reproduced
every cell independently.

**COMPUTED by me** from the six on-disk per-item maps (`.../B10-dllm-infilling-ar-dominance/
evidence/gate1_base/score_base/*_score_base.json`, n=1033, identical `task_id` sets asserted):
`Q_FIM .935140 / Q_prefix .660213 / D_FIM .879961 / D_prefix .502420`; Qwen gain 27.4927 pp, Dream
gain 37.7541 pp, **Δ_surface = +10.2614 pp**; retention **242/296 = .817568**. This reproduces the
proposal's field-3 numbers exactly, so the dispute is about the statistic's *construction*, not the
arithmetic.

**Strongest surviving objection (the firability defect, and it is decisive).** Δ_surface is a
pp-scale difference-in-differences between four cells sitting at .935/.660/.880/.502 against a gold
ceiling of .9894. A pp-scale DiD on near-ceiling cells is **not scale-free**. Under the exact null
"conversion changed *nothing* about suffix use" (Dream carries Qwen's identical log-odds suffix
effect and is merely uniformly weaker), the predicted Δ_surface is **+10.4902 pp** — *above* the
5.0 pp kill line, and 0.23 pp above the observed value. **The zero-effect value of the statistic
already clears its own threshold.** The gate fires only if Dream retains < 77.7 % of Qwen's log-odds
suffix effect; the observed ratio is **0.9891**. This is the P1-c1 defect (arithmetic
pre-determination) in a subtler costume.

Three scale-free re-expressions of the identical four cells, **COMPUTED**: log-odds DiD
**−0.02184** (≈ zero, slightly negative); ceiling-normalised **+10.3713 pp**; headroom-recovered
**−5.9900 pp** (sign flip — Qwen recovers 83.5 % of available headroom from the suffix, Dream
77.5 %). Regrouping by model instead of by condition destroys the narrative outright:
`D_FIM−Q_FIM = −5.52 pp`, `D_prefix−Q_prefix = −15.78 pp` — **Qwen wins both cells**. The entire
+10.26 pp is Dream being *less bad* in the FIM cell.

**Independent second kill — the sign is a hyperparameter the proposal freezes by fiat.** N1 fixes
`initial_masks=32`. DreamOn (arXiv:2602.01326, **ICLR 2026 Poster**, venueid
`ICLR.cc/2026/Conference`, forum `EQTPmqukiU`, `Camera_Ready_Revision` present — **RETRIEVED by me**
this pass) Table 2 publishes Dream-Coder-7B single-line pass@1 across exactly that knob. Substituting
each published value into N1's own DiD, holding D_prefix and both Qwen cells at their measured
values (**COMPUTED by me**):

| `initial_masks` | 4 | 8 | 16 | 32 (N1's choice) | 64 | oracle |
|---|---|---|---|---|---|---|
| published D_FIM | 24.9 | 61.2 | 72.6 | 62.4 | 55.5 | 93.3 |
| implied Δ_surface | −52.8 | −16.5 | −5.1 | **−15.3** | −22.2 | +15.6 |
| gate verdict | FIRES | FIRES | FIRES | **FIRES** | FIRES | does not fire |

**The gate fires at every non-oracle canvas DreamOn validated, and survives only at oracle.** The
swing across DreamOn's own validated 4–64 range is **47.7 pp = 9.54× N1's entire 5.0 pp threshold**,
and 32 is not even the published optimum (16 is). This is the P4 defect, made worse by the row being
non-monotone: a reviewer can pick a canvas yielding any sign they want. I also **COMPUTED** from the
score JSONs that the pilot's Dream arms ran at `forward_passes_per_task = 8.5789`, i.e. **not** at
`initial_masks=32` — so the number N1 quotes was not produced at the canvas N1 wants to freeze.

**Third, and it is a governance problem, not just a science one.** N1 is an **undisclosed
re-proposal of an experiment this repo explicitly de-authorised**. `B10/STATUS.json` (**RETRIEVED**):
`lifecycle: "dead"`; `kill_gate.gate_2` = *"lineage repair on Dream-Coder-v0-Base-7B"*, `gpu:
"~2-4 GPU-h"` — **N1's arms and N1's cost**. `lifecycle_reason`: *"Gates 2 and 3 were conditioned on
PROCEED and are NOT authorised → zero GPU-costing work remains"*, because gate_1 **FIRED** (exact
McNemar p=1.0000, b=39 c=38, |Δ|=0.00096805). `next_gate`: *"REWRITE B10 AS A PROTOCOL NOTE, OR
ARCHIVE IT. 0 GPU."* Worse, N1's headline is the repo's already-**sealed** survivor with the seal's
interpretation reversed — `PROTOCOL_NOTE.md:90-96` states verbatim that it *"must **not** be read as
'diffusion benefits more from bidirectionality'"*, precisely because `dream_prefix` retains an oracle
length that `qwen_prefix` lacks.

**Where I overrule a lens.** The prior-art lens called N1 `refuted: true` partly because its three
named comparators are the wrong set and because DiffuLLaMA/InCoder already own the aggregate version.
The **factual** part is verified and I uphold it: DiffuLLaMA = arXiv:2410.17891, **ICLR 2025 Poster**,
venueid `ICLR.cc/2025/Conference`, forum `j1tSLYKwg8`, `Camera_Ready_Revision` present
(**RETRIEVED**), and its full text (**RETRIEVED**, 124,554 chars) states verbatim *"we use
humaneval-single-line infilling ... which contains 1033 test cases"* and *"we implement infilling
tasks for AR models by feeding the prefix and cutting off the generation length using the oracle
length"* — the **same benchmark, same n, same oracle-length device**, plus a Table 8 matched-lineage
control (CodeLLaMA FT FIM-SPM 0.80 / FIM-PSM 0.74 / Diffu-CodeLLaMA 0.76). **But this is NOT
preemption** under this repo's bar: DiffuLLaMA reports cell means, never a paired item-level
suffix-utilisation decomposition. N1 dies on **its own gate**, which is one of the two licensed kill
routes. It must not be recorded as preempted.

**Narrowed form that survives (0 GPU).** *"A canvas-frozen pp-scale DiD does not identify
suffix-utilisation transfer."* Four legs, all already payable: (1) report the scale-free statistic
(**−0.0218**) as a **negative result**, not a gate to be passed; (2) make canvas an axis and
pre-register against DreamOn's published 4–64 curve; (3) fix the comparator set to DiffuLLaMA
(ICLR 2025) + InCoder + FIM, stating the delta against each; (4) replace the retention clause, whose
mirror direction breaks it (242/296 = .8176 **passes** but 242/411 = .5888 **fires** — same
numerator, verdict decided by which model is nominated reference; **COMPUTED**). This is a
reporting-standards paragraph, not a paper, and it must carry the `PROTOCOL_NOTE.md` §2.1 seal.

---

### N2 — DEAD. The estimand is 95.7 % a third mechanism the two-way partition has no cell for.

**Refuted 3/3.** The confound lens found the decisive defect; I reproduced it exactly.

**Strongest surviving objection.** N2 proposes to split the c32→c128 HE+ gain into "space" and
"refinement compute". **COMPUTED by me** from `evidence/cells/he_c{32,128}.json` and
`cells_corrected/a05_closeout_stitch_regrade.json`: `empty_raw_output` is **75/164 at c32 and 0/164
at c128**. Decomposing pass@1 = P(nonempty) × P(pass | nonempty):

```
he_c32 -> he_c128:  total = +22.5600 pp
                    OCCUPANCY   = +21.5815 pp  (95.7 %)
                    CONDITIONAL = + 0.9785 pp  ( 4.3 %)      sum reproduces exactly
                    conditional rate .471915 -> .481700
```

**95.7 % of the target quantity is the elimination of empty output** — the sampler ceasing to delete
its entire canvas. That is neither "space" nor "refinement compute"; the partition has no cell for
it, and the conditional term (+0.98 pp) is **11.5× smaller than the gate's own 11.28 pp threshold**.
A positive result would mean "canvas 128 stops the model emitting nothing", which is not the claim.

**Independent second kill — the treatment is arithmetically impossible on the majority of the
sample.** From `generation_utils.py` (**RETRIEVED**, on disk at `models/DreamOn-v0-7B/`): line 435
`torch.topk(confidence, number_transfer_tokens)` with `number_transfer_tokens=1` resolves exactly one
mask per counted forward, and line 397's loop bound plus lines 383-384 (`expand_budget = max_gen_len
* 2`) give `NFE = canvas + 2·E`. **Verified against disk**: `nfe_median == canvas` **exactly in 5/5
cells** (8.0 / 32.0 / 128.0 / 8.0 / 32.0) and `max − canvas = 2052` in 5/5 (max 2060/2084/2180). With
n=164 even and floor 32, **≥ 83/164 = 50.61 % of items have `max_nfe_i = 32`**, so a 32-forward cap
on a 128-mask canvas leaves ≥ 96/128 = 75 % of masks unresolved — and the failure is **silent**
(`<|mask|>` 151666 is in `all_special_ids`; the driver decodes with `skip_special_tokens=True`).

**Third: the named mediator is identically zero.** N2 field 2 requires recording
"remasking/revision count" and its causal agent is "refinement compute". **COMPUTED**: word-boundary
counts in `generation_utils.py` are `remask=0, unmask=0, revision=0, refine=0`; line 442
`x[mask_index] = x0_` writes only where line 403's `(x == mask_token_id)` selector holds, so a
settled content token is never revisited. **DreamOn has no revision channel.** This is the P6 defect
(a statistic referencing a mechanism the target never implements).

**Fourth: closed at 0 GPU by its own target paper.** DreamOn §5.3 (**RETRIEVED verbatim** by me):
*"Disabling deletion broadcasting reduces performance by 0.6 % on average"* and *"also accelerates
generation by 2.1×"*; §5.4: *"reducing total inference steps from as high as 122.8 (w/o
broadcasting) to just 52.4 (w/ broadcasting)"*. I re-derived the Table 3 row: DreamOn 90.8 vs w/o
Deletion Broadcasting 90.2 = **0.6 pp**. So the published same-model **fixed-canvas** elasticity is
**2.1× forwards ↔ 0.6 pp**, whereas N2's gate demands that a **1.5073×** NFE change explain
**≥ 11.28 pp** — an **18.8× larger** quality response to a **1.39× smaller** compute change. This is
the P5 defect: closed at 0 GPU by tables in the proposal's own target paper.

**Fifth: prerequisites and cost.** The per-item cap vector does not exist on any disk reachable from
LOCAL — all five A05 cells record `run_dir = /apdcephfs_zwfy6/...`, and I re-confirmed
`ls -d /apdcephfs_zwfy6` → **"No such file or directory"** with `df` showing only `dop-fuse` on
`/apdcephfs_wzc1/share_304376610`. Per the two-disk rule that is "not readable from here", not "does
not exist" — but it means the arm cannot be *configured*, let alone run. Also `mbpp_c128` does not
exist as a scored cell, so the grid is **HE+ only, 5 cells not 6**. And the cost unit is already
adjudicated wrong: **COMPUTED** NFE ratio 1.5073× vs `tokens_fed_effective` 1.9334× ⇒ **22.0 % of
attended-work growth unmatched** by an NFE match, which `NFE_SEAM_VERDICT_20260816.md` §1/§4 already
ruled insufficient for canvas-expanding controllers.

**Narrowed form that survives (0 GPU), and it has a gate that genuinely fires.**
*"The apparent initial-canvas effect on EvalPlus pass@1 is dominated by an occupancy channel — the
fraction of items for which the sampler deletes its entire canvas — not by code quality conditional
on something being emitted."* Pre-register: **KILL if occupancy is not the majority contributor in
≥ 5 of the 6 available (benchmark × grader × canvas-step) cells, OR if the conditional-term 95 % CI
excludes zero in the HE+ c32→c128 corrected cell.** Both branches were attainable before looking.
**Measured: occupancy dominates 6/6** (MBPP+ c8→c32 replicates it at 123.4 %, with the conditional
term **negative**, −6.20 pp), and the conditional CI contains zero. Two paragraphs plus one table
inside an existing document. Must **not** be framed as a canvas finding (§6).

---

### N3 — DEAD as written. The statistic cannot clear its own noise floor, which is on disk.

**Refuted 3/3.** The confound lens found the decisive fact; I reproduced it exactly, 48/48 items.

**Strongest surviving objection — and it is the single most valuable measurement in this whole
exercise.** N3's gate turns on a ratio of ratios with a 20 % (|ln 0.80| = 0.2231) detection
threshold. There are **three** H20 observations of the identical thing on disk and they disagree by
**2×**. Same node (.73), same torch 2.5.1+cu124 / transformers 4.46.2, same 48 `task_id`s, same
driver, and constant 512-NFE work per item. `CROSSNODE_REPRODUCIBILITY.md:169` declares this pair
verbatim **"F vs C (H20, both torch 2.5.1) | what varies: nothing"**. **COMPUTED by me**:

```
C_73_H20_zwfy6_t251   48-item subset mean =  51.784 s
F_73_H20_t251_lim48                  mean = 106.064 s
  ratio F/C = 2.0482 (mean) | 2.0315 (median) | 2.0485 (geomean per-item)
  F > C on 48/48 items; per-item ratio range [2.009, 2.110]  -> uniform level shift, not a tail
  ln(2.0482) = 0.7170  vs  gate threshold |ln 0.80| = 0.2231   ->  noise = 3.21x the whole effect
```

And the noise is **asymmetric between the two arms being compared**: the sm_100 side has four
replicates across two hosts (A 11.415 / A2 11.279 / B 11.851 / B2 11.812 s, paired n=164) =
**1.0507× spread**, so **H20 carries 14.5× more log-noise than sm_100**. The cross-hardware ratio the
gate consumes therefore takes the value **6.173× (C), 8.003× (E), or 9.086× (F)** against the same
sm_100 arm, purely by replicate choice — implied kill bars 4.938× / 6.402× / 7.269×. Mechanism is
visible and is **not** thermal: **COMPUTED** Pearson(position-within-rank-stream, elapsed) =
**−0.5689** on C (first-half 84.73 s → last-half 53.94 s) versus **−0.0239** on A (ratio 1.011) — a
decaying co-tenant on a shared node. **A positive G is fully explained by which hour each leg ran.**

**Independent second kill — the precondition is destroyed by the effect under study.** Clause 2
demands the arms sit within 1 pp pass@1. On n=164 one item = 0.60976 pp. **COMPUTED** from the two
grader files: HE+ **116/164 = .707317 on sm_100** versus **112/164 = .682927 on H20** — identical
weights, one grader, **4 items = 2.4390 pp = 2.44× the tolerance**. Clause 2 must hold on *both*
legs, so a point validated within 1 pp on B200 is off-match by up to 2.44 pp on H20 **because of the
very effect N3 isolates**.

**Third: N3's stated delta is factually false.** N3 field 3 says existing studies *"primarily profile
A6000/A100/H100-class systems"* and cites arXiv:2510.18480. **RETRIEVED and COMPUTED by me**: a
case-insensitive scan of that paper's full text for
`A6000|A100|H100|H200|B200|H800|A800|V100|RTX|MI300` returns **exactly one hardware token: `A800`**,
in the verbatim sentence *"All experiments are conducted on a single NVIDIA A800 GPU (80 GB) using
FP16 precision."* Getting the closest work's hardware wrong is not a small error when hardware *is*
the proposal.

**Fourth: 2 of the 4 gate cells have no code path.** Confirmed independently: neither
`scripts/generate_evalplus_dream.py` nor `generate_evalplus_ar.py` takes a batch argument, so the
batch-8 cells do not exist and the "geometric mean across the four batch/length cells" is over 2.

**Where I overrule / correct the reports.** (a) The **precheck's clause-2 refutation used the wrong
axis.** It fired clause 2 from `runs/dream_instruct_heplus_nfe*`, calling it a sweep of N3's named
"output budgets". **I verified** `scripts/_run_nfe_sweep_wzc1.sh` line 4 reads *"Vary --steps ∈
{64,128,256,1024}"* and line 25 pins `--max-new-tokens 512`; a repo-wide grep gives 35 occurrences of
`--max-new-tokens 512` and none at 64. So that is the **parallelism** axis at fixed canvas, and no
output-budget ladder exists on disk. The directional conclusion survives (the steps curve jumps
41.5 pp between the only two cells bracketing Qwen) but "clause 2 fires at 0 GPU" is **not
established on N5's—sorry, N3's—own knob**, and must not be quoted as if it were. (b) The precheck
also asserts *"no latency-vs-software control exists anywhere"* and that the software axis is
"physically un-freezable". **Both are wrong**: E vs F on fixed H20 (torch 2.6.0 vs 2.5.1) is a
latency control and **COMPUTED** gives **E/F = 0.8807**, i.e. 11.9 %. That does not rescue N3 — it
indicts it further, because the **null** C-vs-F (nothing varies) is 2.0482×, i.e. **5.1× larger than
the measured software effect**. The precheck attributed the excess-over-roofline to software; the
data says software ≈ 1.14× and the residual is co-tenancy. (c) The precheck's roofline argument
("6.17× cannot be arithmetic intensity because SM ratio is 1.897× and HBM ratio 1.874×") is
**invalid**: neither SM count nor memory capacity is a roofline parameter. H20 peak FLOPs/bandwidth
remain **UNVERIFIED** (§8), so no roofline prediction can honestly be written.

**Narrowed form that survives — and this is the batch's best residue (§5).**

---

### N4 — DEAD. Its gate is answered, against it, by a column TAD ships in its own launch script.

**Refuted 3/3.** The precheck found the decisive fact; I reproduced it from the authors' own file.

**Strongest surviving objection.** N4's "foreign vanilla sampler" arm is not an unrun experiment — it
is a **published column**. I read `tad_gh/eval_llada.sh` (**RETRIEVED**): four blocks each headed
`# TAD-LLaDA-TPF1`, invoking `threshold=0` with no `multi_block`, on gsm8k/math/humaneval/mbpp. And
`threshold=0` provably degenerates to one-token-per-forward: `eval_llada.py:75` tests
`if entropy_threshold is not None` (true for 0, since `0 is not None`), Shannon entropy computed at
line 67 is ≥ 0 so line 76's `< 0` selects nothing, and the argmin fallback at lines 77-80 unmasks
exactly one position ⇒ TPF 1.000. From TAD's own `data_llada.yaml` (**COMPUTED by me**, baseline
LLaDA 38.28 HE / 41.72 MBPP):

| benchmark | arm | vanilla gain | official gain | retention | gate (KILL if > 50 %) |
|---|---|---|---|---|---|
| HumanEval | TAD-S | +5.01 | +1.35 | **371.1 %** | **FIRES** |
| HumanEval | TAD-Q | +5.62 | +3.79 | **148.3 %** | **FIRES** |
| MBPP | TAD-S | +1.68 | **−1.12** | −150.0 % | denominator negative → ill-posed |
| MBPP | TAD-Q | +1.08 | **−0.12** | −900.0 % | denominator negative → ill-posed |

**Retention exceeds 100 % everywhere it is defined**: the distilled checkpoint scores **higher** under
the **foreign** vanilla schedule (HE 43.29 vs 39.63; 43.90 vs 42.07; MBPP 43.40 vs 40.60; 42.80 vs
41.60). The official schedule **costs** accuracy and buys parallelism — the opposite of the claim.
And on MBPP (n=500, 1 item = 0.20 pp) the denominators are **5.6 and 0.6 items**: the unguarded
near-zero denominator that is the P2 defect.

**Second: "8-step" is not expressible under the official sampler, and sits 3.78× out of range.** I
re-derived rather than carrying: `generate_multi_block` spans `eval_llada.py:328-424`, and
`get_num_transfer_tokens` is called only at lines 139/199/278 — all in the *non*-multi_block paths.
So `steps` is dead in the official branch; NFE is a settable knob only in the vanilla branch.
Separately, **COMPUTED**: at TAD's `gen_length=256`, N4's NFE {8, 32, 128} maps to TPF
**{32.0, 8.0, 2.0}**, while TAD's largest measured TPF anywhere in its own yaml is **9.11** (largest
TAD row 8.47) ⇒ N4's **headline** cell is **3.78× beyond the furthest point the target paper ever
measured**. That is the P4 out-of-validated-range defect.

**Third: the "samplers" are two points on a compute axis.** **COMPUTED** from the yaml's own ρ (TPF)
values: the official arm runs at implied NFE ≈ 256/ρ = **41.9–75.7** while the vanilla arm runs at
**256**. So the two "gains" are measured **3.38×–6.11× apart in compute**. Any interaction term is a
budget effect with a sampler label on it — which also means the precheck's retention ratios divide a
~43-NFE gain into a 256-NFE gain.

**Fourth: benchmark identity.** N4's gate names HE+/MBPP+. **COMPUTED**: `grep -ci` over TAD's full
text for `evalplus|eval-plus|humaneval\+|mbpp\+|sanitized` = **0**; TAD evaluates original
HumanEval/MBPP via lm-evaluation-harness. The retention denominator has **no published referent** on
the benchmarks the gate names.

**Where I overrule a lens.** The confound lens reports a further finding — that 24/24 commented-out
operating points in TAD's yaml fall on competitor arms, and that restoring the 8 not forced by
`aup_utils.py`'s assertion flips the AUP winner on both code benchmarks. I **partially verified** the
input: the commented-out competitor points are real and on disk (I read them: Fast-dLLM
`[4.13,70.20] [4.24,69.00] [4.37,68.31]`, d3LLM `[9.54,70.58] [10.04,63.08] [10.99,60.27]`, dParallel
`[5.63,72.02] [7.32,71.37] [7.58,70.77]`, all behind `#`, and **zero** commented points on TAD's own
arms). I did **not** re-run `get_aup` to reproduce the claimed rank flips, so **the rank-flip numbers
are CARRIED, not verified**, and must not be quoted as measured. Treat as a lead requiring its own
0-GPU verification and, before any publication, contact with the AUP authors.

**Narrowed form: none as an experiment.** Do not queue the 6–10 GPU-h (true cost ~12–46 GPU-h by the
A05 anchor) and do not download the 32.1 GB. The only survivor is a one-sentence reporting note: in
one released repo the selector string `low_confidence` names two different statistics — the eval path
(`eval_llada.py:66-67`, Shannon entropy, `topk(largest=False)`) versus the generation path
(`generate.py:107,121`, max-probability, `topk(largest=True)`). Note the two lenses **disagree on the
magnitude** of that divergence (5/40 rank agreement on flat synthetic logits versus 99.0 % top-1
agreement on peaked realistic logits); the second is the better-evidenced regime, so the honest
framing is *"the string is overloaded; papers should say which statistic they used"*, **not** "the
samplers disagree".

---

### N5 — DEAD. Clause 1 fires 4/4 at its own canvases, in two independent published replications.

**Refuted 3/3.** I verified this against **two** primary texts and it is the cleanest kill in the
batch.

**Strongest surviving objection.** DAEDAL (**ICLR 2026 Poster**, venueid `ICLR.cc/2026/Conference`)
runs **both** of N5's checkpoints at **six matched canvases** with **N5's own length metric**, using
verbatim (line 500, **RETRIEVED**) *"the official generation code released with LLaDA, without any
acceleration or caching optimizations ... 8 NVIDIA A800 80G GPUs, with the batch size set to 8"*, and
prints `E_token` = *"the 'net' response length after removing trailing EOS padding"*. ρ-EOS
(arXiv:2601.22527, CoRR 2026, **RETRIEVED** full text on disk) replicates the identical grid and
states *"each experiment is repeated three times, and we report the averaged results"*. **COMPUTED by
me from both primary texts** (Δ = LLaDA-1.5 − LLaDA-Instruct, HumanEval Acc):

| canvas | 64 | **128** | 256 | **512** | 1024 | 2048 | fires (< 2 pp) |
|---|---|---|---|---|---|---|---|
| DAEDAL Δ | −0.6 | **−4.2** | +1.8 | **−1.9** | +1.8 | +3.0 | **5/6** |
| ρ-EOS Δ | +0.5 | **−5.5** | +1.8 | **+0.0** | +4.3 | +0.6 | **5/6** |

**Clause 1 fires 4/4 at N5's two pre-registered canvases {128, 512}, in both replications.** And the
length mediator is already equalised there: DAEDAL `E_token` **125 vs 125** (c128) and **471 vs 475**
(c512); ρ-EOS 125 vs 124.9 and 462 vs 473.8. **At matched output length, LLaDA-1.5 is *worse* on
HumanEval.** There is no positive gain for a length mediator to explain, so clause 2's "% of gain
removed" divides by a negative number — the P3 defect (a negative margin has no ">50 % shrink").

**Second: the mediator has near-zero between-arm variance where N5 measures, and N5 does not look
where it moves.** Mean |ΔE_token| is **2.0 tok** at {128, 512} versus **60.5 tok** at {1024, 2048} —
and {1024, 2048} are the only canvases with a positive gain. **COMPUTED**: the implied dose-response
flips sign and spans three orders of magnitude (−32.8 pp/token at c128, −0.41 at c512, +0.05 at
c1024), so any length-stratified adjustment is an auditor free parameter.

**Third, and I correct the precheck here.** It reports clause 2's AND-conjunct as **satisfied**
(interaction +2.3 pp < 3 pp), collapsing clause 2. **COMPUTED**: that holds on DAEDAL only. On ρ-EOS
the same interaction is **+5.50 pp** (Instruct 26.2→45.1 = +18.9; LLaDA-1.5 20.7→45.1 = +24.4) —
**not** satisfied. The 3 pp threshold sits **between two equally matched published measurements**
whose spread (3.2 pp) exceeds the threshold itself. The clause-2 verdict is decided by which lab you
cite. This does not rescue N5 (clause 1 is an OR-branch firing 4/4) but "clause 2 collapses" is not
established.

**Fourth: the EOS convention is circular.** VRPO states verbatim (**RETRIEVED**) *"setting the |EOS|
token's confidence score to zero improved HumanEval scores from 47.6 to 49.4. Consequently, we
adopted this setting for evaluation."* So **1.8 pp of N5's own denominator is a knob setting, and
that knob is the termination policy N5 wants to measure.**

**Fifth: the step-1 input has no referent.** N5 field 5 audits "response-length distributions
described by the training recipes". **CARRIED and consistent with my own reading**: word-boundary
counts in VRPO's full text give `response length` 0, `length distribution` 0, `average length` 0,
`output length` 0, `generation length` 0. Same defect class as P6 clause (c).

**A venue correction the programme must absorb.** VRPO/LLaDA-1.5 is **published**, not a preprint:
**ACL Anthology `2026.acl-long.524`**, *Proceedings of the 64th Annual Meeting of the ACL (Volume 1:
Long Papers)*, July 2026, San Diego, **pages 11425–11460**, doi `10.18653/v1/2026.acl-long.524`,
bibkey `zhu-etal-2026-llada` — **RETRIEVED by me** (`aclanthology.org/2026.acl-long.524.bib`,
http=200). `CLAUDE_FRONTIER_20260815.md:97` and the precheck both treat it as arXiv-only. This is a
textbook instance of both repo fallacies: OpenReview shows only `CoRR 2025` plus an ICLR 2026
`Withdrawn_Submission`, and **ACL-family venues require Anthology, not OpenReview**.

**Narrowed form: none as an experiment.** No narrowing of this checkpoint pair on HumanEval can
produce the positive gain the claim requires. What survives is a caveat paragraph: three papers
report Instruct HumanEval at length 512 as **49.4 / 47.0 / 45.1** — a **4.3 pp spread on one frozen
(checkpoint, benchmark, length) cell**, with a **2.5 pp maximum** disagreement across the 12 shared
cells of the two full replications (**COMPUTED**; mean 0.983 pp). Both exceed N5's own 2.0 pp
decision threshold.

---

## 4. Ranked survivor list

**GPU survivors: none.** Ranking the **0-GPU residues** by (expected information gain)/(cost), since
that is the only currency available for the next ~15 h:

| rank | residue | from | cost | why it ranks here |
|---:|---|---|---|---|
| **1** | **Cross-node dLLM latency is not reproducible to better than ~2×** | N3 | **0 GPU, ~2 h** | Only genuinely NEW, non-preempted, decision-changing result in the batch. ~80 % already measured. Directly supplies the variance floor that arXiv:2510.18480's own critique of dLLM efficiency evaluation lacks. Pairs with the existing `CROSSNODE_REPRODUCIBILITY.md` **quality** result to give a two-axis reproducibility finding on one protocol. Has a gate that can fire both ways. |
| **2** | **The canvas effect is an occupancy effect** (95.7 % of our own 22.56 pp) | N2 | **0 GPU, ~2 h** | Measurement-validity correction to **our own** headline number, replicated on a second benchmark with the conditional term flipping **negative**. Gate fires 6/6, verified. Risk: must be framed as decomposition, never as a canvas finding (§6). |
| **3** | **A canvas-frozen pp-scale DiD does not identify suffix transfer** | N1 | **0 GPU, ~1 h** | Fully computed already (−0.0218 log-odds; 47.7 pp canvas lever). Fixes a comparator set that omits both nearest works. It is a negative result, so lower ceiling. |
| **4** | **Between-lab spread on one frozen cell exceeds published effect sizes** (4.3 pp) | N5 | **0 GPU, ~1 h** | Strong caveat paragraph, composes with #1. Not standalone. |
| **5** | **AUP is affine in the author-chosen sweep endpoint** | N4 | **0 GPU, ~2 h** | Potentially the most interesting metric critique, but the rank-flip numbers are **CARRIED, not verified**, and it points at named third parties. Needs its own verification pass and author contact before anything is written. |

---

## 5. The single first action

**GPU: NONE.** Do not queue any of N1–N5.

**First action (0 GPU, startable immediately, does not wait for a card):** write the **N3-narrowed
latency-reproducibility note**.

> **Claim**: For a fully pinned mask-diffusion decoding protocol (fixed checkpoint, fixed canvas,
> fixed steps, fixed sampler, batch 1, constant 512 NFE per item), the **within-condition** replicate
> spread of end-to-end latency on a shared H20 node is **≥ 2×** — larger than the cross-architecture
> effects dLLM efficiency papers report to two significant figures. Therefore **single-run latency
> numbers in dLLM efficiency papers are not falsifiable claims.**

**Exact first step, and it needs NO GPU**: assemble the note from four files already on wzc1 —
`runs/xnode/{A_local_L20A_wzc1_t211, C_73_H20_zwfy6_t251, E_73_H20_t260_lim48,
F_73_H20_t251_lim48}/metrics.jsonl` plus their `stack_meta.json`. Three numbers, all **COMPUTED and
reproduced by me this pass**: (i) C = 51.784 s vs F = 106.064 s on the same 48 items under
conditions `CROSSNODE_REPRODUCIBILITY.md:169` declares *"what varies: nothing"* → **2.0482×, F > C on
48/48**, per-item ratio range [2.009, 2.110]; (ii) a within-run warm-up/co-tenancy transient present
on H20 (Pearson −0.5689, first/last 84.73→53.94 s) and **absent** on sm_100 (−0.0239, ratio 1.011);
(iii) a torch 2.5.1→2.6.0 upgrade on **fixed** hardware is worth **11.9 %** (E/F = 0.8807) even
though the same upgrade is worth **0** quality flips.

**Measured cost**: 0 GPU-h, ~2 CPU-hours. **Blocking prerequisite**: none.

**Pre-registered gate for the note itself (can fire both ways):** *KILL if, with n ≥ 3 temporally
interleaved exclusive-tenancy replicates per cell, the max/min within-cell latency ratio is < 1.20 on
**both** hardware types.* On current (non-exclusive) data this would **not** fire on H20 (2.0482×)
but **would** fire on sm_100 (1.0507×) — genuinely two-sided, outcome unknown in advance. That is
what all five N-proposals lacked.

**Five binding scope guards** (drop any one and the note dies): (1) **no cross-family Dream/Qwen
latency number** — the AR harness carries a per-forward device→host `.item()` sync
(`scripts/forward_cost.py:57`, registered at `generate_evalplus_ar.py:205`) that the Dream harness
does not, so every Dream/Qwen ratio is biased toward the dLLM by instrumentation; (2) **no claim that
hardware architecture causes the gap** — report the variance, not a cause; (3) **no canvas or
NFE-cost framing** (§6); (4) **no batch-size axis** — no code path exists; (5) **report the C-vs-F
2.05× as "unexplained, provenance-incomplete"** — both files were written 2026-08-07 (C 21:38, F
22:05), no driver for the E/F arms survives in the repo, so co-tenancy cannot be *excluded* as the
cause. That uncertainty **is** the point: the artefact is real and its cause is unrecorded, which is
exactly why a 20 % gate on this statistic cannot fire honestly. **Do not assert a cause.**

**If a card frees later**, the follow-up is ~2 additional H20 replicates × 164 items ≈ **6.4 GPU-h
card-time ≈ 0.8 h wall on 8 H20** (anchored on arm C's measured Σ elapsed = 11,555.8 s = 3.21
GPU-h) — but it **must** run on an idle, exclusively-held node, because exclusive tenancy is the
entire point. Running it on a shared node would reproduce the defect it measures.

---

## 6. What must NOT be claimed

1. **The canvas-budget finding is a REPRODUCTION, not a discovery.** Preempted by DAEDAL
   (arXiv:2508.00819, **ICLR 2026 Poster**, venueid `ICLR.cc/2026/Conference`, `Camera_Ready_Revision`
   present — re-verified by me this pass) and independently replicated by ρ-EOS (arXiv:2601.22527,
   2026-01, CoRR 2026). Both predate our statement by ≥ 6.5 months, so the 2–3-month concurrency
   clause does **not** apply. Our `initial_masks` 8→32 MBPP+ `.0899→.3545` (+26.45 pp) is a
   reproduction on a third model. **Banned verbatim, including disguised forms: "cost of canvas",
   "quality is bought with canvas."** Also barred: "we discovered dLLM code scores are a
   canvas-budget artefact." **Permitted**: "we independently reproduce the known canvas-budget
   sensitivity (DAEDAL, ρ-EOS)"; "our published DreamOn operating point sat in the crippled regime."
2. **The NFE cost-accounting seam is DEAD.** No universal about the field's reporting. "No
   adaptive-length paper reports a total forward count" is false (VoidPadding). "No adaptive-length
   paper reports an absolute total cost" is false (ρ-EOS `T_runtime`). "The specific flaw is
   per-token normalisation" is false (DAEDAL `N_token`). N2's residue must be framed as
   **occupancy decomposition**, never as cost accounting.
3. **VoidPadding (arXiv:2606.17999) may NOT serve as a preemption authority.** Published 2026-06-16,
   updated 2026-06-22 → **2.0 months = CONCURRENT** under this repo's own rule; and OpenReview
   returns only `CoRR 2026`, **`COLM 2026 ER Workshop`**, and **`ACL ARR 2026 August Submission`** —
   a workshop poster plus an under-review submission, **not a main conference**. Where a
   non-concurrent authority is needed, use ρ-EOS (arXiv:2601.22527).
4. **A `CoRR` DBLP record means NOT-FOUND, not IS-A-PREPRINT** (DAEDAL has both). **An OpenReview
   `Withdrawn_Submission` means NOT-FOUND, not IS-NOT-PUBLISHED** — VRPO is the live example: it is
   **ACL 2026 Long, pages 11425–11460**. **ACL-family venues (incl. Findings) require ACL Anthology
   + DBLP**, not OpenReview.
5. **N1's licensed sentence must not be written.** *"AR→diffusion conversion changed how a matched
   lineage exploits visible right context"* is contradicted by today's data on the scale-free
   statistic (log-odds DiD **−0.0218**, ratio 0.9891; headroom-recovered **−5.99 pp**). Also barred:
   any implication that the matched-lineage infilling comparison is new (DiffuLLaMA, ICLR 2025,
   Table 1 + Table 8, same 1033-item benchmark, same oracle-length device).
6. **`PROTOCOL_NOTE.md` §2.1's seal still binds**: bidirectional context is an affordance of the FIM
   task **framing**, not a property of the model class, and `dream_prefix` retains an oracle length
   `qwen_prefix` lacks. Qwen wins **both** cells (−5.52 pp FIM, −15.78 pp prefix).
7. **N2's "refinement compute" is a category error for DreamOn** — `remask/unmask/revision/refine`
   all have **0** word-boundary occurrences in its sampler. Do not report a revision count.
8. **N4's retention ratios must not be quoted in either direction** — they divide a ~43-NFE gain into
   a 256-NFE gain, so they are compute-ratio artefacts. And the **AUP rank-flip numbers are CARRIED,
   not verified**; do not publish them without an independent re-run and author contact.
9. **N3's "B200 helps dLLMs" is unidentified by ±2×** — the ratio is 6.173× / 8.003× / 9.086×
   depending on replicate choice. And **do not quote H20-vs-B200 roofline predictions**: H20 peak
   FLOPs/bandwidth are UNVERIFIED (§8), and neither SM count nor memory capacity is a roofline
   parameter.
10. **Every "DreamOn reaches X" figure in this repo remains a LOWER BOUND** (`he_c512`, `mbpp_c128`,
    `mbpp_c512` never ran; `mbpp_c128` was killed at 30/378 and never graded). The grid is **5 cells,
    not 6** — HE+ has 8/32/128, MBPP+ has 8/32 only.

---

## 7. The recurring defect, and the standing rule that would have caught all five

Across 12 adjudicated proposals (P1–P7, N1–N5), **12/12** died and **10/12** died on gate
construction. The N-batch sharpens the diagnosis: **3 of 5 gates were already answered, against the
proposal, by data the proposal itself cited.**

| id | what would have caught it | cost |
|---|---|---|
| N1 | Compute the statistic's value under its own null hypothesis. | minutes |
| N1 | Substitute the target paper's published hyperparameter row into your own statistic. | minutes |
| N2 | Decompose the target quantity by the obvious third channel (`empty_raw_output` was already in the cell JSON). | minutes |
| N2 | Read the sampler source for the mechanism you named as mediator. | minutes |
| N3 | Compute the within-condition replicate spread before setting a threshold. | minutes |
| N4 | Read the target repo's own launch script for the arm you propose to run. | minutes |
| N5 | Read the two published papers that already ran your exact 2×6 grid. | ~1 hour |

**STANDING RULE (proposed, 0 GPU, binding on the next proposal batch).** Before a kill gate is
pre-registered, four checks must be recorded in the proposal itself:

1. **Null-value check** — state the statistic's value under the hypothesis of *no effect*. If that
   value already passes the gate, the gate is void. (Kills N1 c1.)
2. **Published-lever check** — for every hyperparameter frozen by fiat, quote the target paper's own
   sweep over it and state the implied swing in your statistic. If the swing exceeds your threshold,
   the knob is an axis, not a constant. (Kills N1; would have caught P4.)
3. **Already-answered check** — grep the target paper's tables *and its released code/launch scripts*
   for the arms you propose to run. (Kills N4, N5; would have caught P5.)
4. **Floor check** — state the measured within-condition replicate spread of the statistic. If the
   threshold is inside the floor, the gate cannot fire honestly. (Kills N3; generalises
   `a-range-is-not-a-measurement-until-it-clears-its-floor`.)

Additionally, three gate-shape defects recur and should be barred outright: **unguarded ratio
denominators** (N2, N4, N5 — MBPP denominators of 0.6 and 5.6 items); **direction-arbitrary set
statistics** (N1's retention: 242/296 passes, 242/411 fires, same numerator); and **AND-conjunctions
that cannot resolve the proposal's own modal prediction** (N4, N5 — recurring from P2).

---

## 8. Unverified — with the exact authority tried and its exact error

| item | status | authority tried, and its exact response |
|---|---|---|
| **H20 peak FLOPs / peak bandwidth** | **UNVERIFIED** — blocks any roofline claim in N3 | arXiv `all:"H20" AND all:"bandwidth" AND all:"TFLOPS"` → **n=0**. arXiv:2607.13068 Table 1 prints F/B ridge points for H100 591, B200 584, H200 412, MI300X 493, TPU v7 626 but **omits H20 entirely**. No vendor datasheet retrieved. Consequence: the precheck's "6.17× cannot be arithmetic intensity" inference does not follow from the numbers it used (SM count and memory capacity are not roofline parameters). |
| **N4's AUP rank-flip numbers** (d3LLM 96.6→119.1 overtaking TAD-Q 117.4 on HumanEval; 88.4→99.5 vs 89.3 on MBPP) | **CARRIED, not verified by me** | I verified the *inputs* on disk (24 commented-out points, all on competitor arms, 0 on TAD's own; `aup_utils.py:38` assertion present) but did **not** re-execute `get_aup` to reproduce the flips. Must be independently re-run before use. |
| **TAD (arXiv:2605.09536) venue** | **UNVERIFIED — NOT-FOUND, not "is a preprint"** | `api2.openreview.net/notes/search` on the exact title → only an unrelated *"STA-TAD"* (CASA 2025). N4's own cited authority `openreview.net/forum?id=ue1zFeD275` → `/notes/search` returns **no matching titled note**. DBLP returned **HTTP 500** on longer queries (server-side; my positive controls in the same pass were http=200), so absence is not established there. |
| **N5's "response-length distributions in the training recipes"** | **CARRIED** (word counts consistent with my own reading of the on-disk text) | I did not re-run the full enumeration; the precheck and confound lens agree on 0 occurrences for all six phrasings across `primary_texts/t_2505.19223.txt`. |
| **`/apdcephfs_zwfy6`** | **Not mountable from LOCAL** — re-confirmed by me | `ls -d /apdcephfs_zwfy6` → `No such file or directory`; `df` shows only `dop-fuse` on `/apdcephfs_wzc1/share_304376610` (120T, 92 % used, 10T avail). Per the two-disk rule this is "not readable from here", **not** "does not exist". Consequence: N2's per-item cap vector and N1's `generate_infilling.py` / `score_infilling.py` / `_run_infilling_5arm_8gpu.sh` are unreachable; they require `scp -O` from .73/.82/.104. |
| **`scipy`** | **Absent from every LOCAL interpreter tried** | `/opt/conda/envs/torch-base/bin/python` → `ModuleNotFoundError: No module named 'scipy'`. All exact tests in the lens reports are hand-rolled implementations; quote **counts**, not p-values, until one is recomputed with a validated implementation. |
| **Semantic Scholar** | **REFUSED throughout (HTTP 429 per prior passes)** | Not attempted this pass; **not relied upon for any field in this document.** Its silence is not evidence of absence. |
| **N3 output-budget ladder** | **Does not exist on disk** (this is a finding, not a gap) | `scripts/_run_nfe_sweep_wzc1.sh:4` varies `--steps`, line 25 pins `--max-new-tokens 512`; repo-wide grep gives 35× `512`, 1× `256`, 1× `32`. So the precheck's clause-2 refutation was computed on the **parallelism** axis, not N3's named output-budget axis. |

---

## 9. Files this document relies on

**On disk (wzc1), all opened by me this pass:**
`Mixture-of-Memory/proposal/backlog/B10-dllm-infilling-ar-dominance/{STATUS.json, PROTOCOL_NOTE.md,
NUMBER_AUDIT.md, evidence/gate1_base/score_base/*_score_base.json}` ·
`Mixture-of-Memory/proposal/archive/A05-structural-dllm-cost-frontier/evidence/{cells/he_c{8,32,128}.json,
cells_corrected/a05_closeout_stitch_regrade.json, a05_closeout_cost_audit.json}` ·
`dllm_draft/models/DreamOn-v0-7B/generation_utils.py` ·
`dllm_draft/runs/xnode/{A,A2,B,B2,C,E,F}*/metrics.jsonl` + `stack_meta.json` ·
`dllm_draft/CROSSNODE_REPRODUCIBILITY.md` ·
`dllm_draft/proposal/primary_texts/{2508.00819v2.txt, t_2601.22527.txt, t_2505.19223.txt}` ·
`dllm_draft/proposal/n4_probe/tad_gh/{data_llada.yaml, eval_llada.py, eval_llada.sh, aup_utils.py}` ·
`dllm_draft/scripts/_run_nfe_sweep_wzc1.sh`

**Retrieved from network this pass:** DAEDAL venue (OpenReview) · DreamOn venue + full text
(OpenReview + arXiv HTML, 258,357 B) · DiffuLLaMA venue + full text (OpenReview + arXiv HTML,
442,931 B) · ANY-ORDER venue (OpenReview: ICLR 2026 Poster, forum `vtDUomlazQ`) · VRPO ACL Anthology
bib · arXiv:2510.18480 metadata + full text (242,092 B).
