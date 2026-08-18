# dLLM / Scaffold-Coder Salvage Roadmap

**Date:** 2026-08-08 (Asia/Shanghai)  
**Scope:** repository audit only; no new GPU experiment was launched.  
**Canonical role:** this document is the clean forward-looking decision record.
`DLLM_RESULTS_20260807.md` remains the chronological retraction log and should
not be read from bottom to top as a current proposal.

---

## 0. Executive decision

The original thesis — **Scaffold-Coder as a competitive full-program code
generator** — should be stopped in its present form.

What survived is not the current checkpoint, but three reusable assets:

1. a well-instrumented dLLM evaluation stack;
2. a typed mutable runtime with real subtree-collapse primitives;
3. a large collection of negative controls that sharply constrain the next
   hypothesis.

The best next **method** experiment is:

> **Use the typed runtime as a controller around a strong dLLM checkpoint, and
> test oracle-localized, preservation-aware code repair before training a
> localizer or another full-generation checkpoint.**

The best next **paper** direction is:

> **An evaluation/protocol paper on sampler sensitivity, fair cost accounting,
> and task-surface effects — but only after a second dLLM family and a small
> non-code replication.**

No result in this repository currently establishes a general dLLM advantage
over autoregressive code models.

---

## 1. Scientific status map

### 1.1 Dead as a paper headline

| Direction | Decision | Why |
|---|---|---|
| Scaffold full-program generation | **STOP** | HE+/MBPP+ peak at `0.177/0.354` for Medium, while the Qwen2.5-Coder-7B product-level AR control is about `0.494–0.518/0.648–0.651` under the currently corrected/local protocols and dramatically cheaper under `tokens_fed`. |
| “Structural runtime owns the low-cost Pareto frontier” | **RETRACTED** | The original frontier omitted the natural AR baseline. Scaffold can only be described as a diffusion-family internal point. |
| Depth-banded schedule | **STOP** | Schedule-only reaches only `0.0366/0.1138`, far below matched Plain SFT (`0.2195/0.2434`). |
| Shared LM-head meta-token training | **STOP IN CURRENT FORM** | Ordinary vocabulary rows and structural actions interfere; literal edit-word pollution, compound-token recursion, and empty-program deletion all occurred. |
| Token-level `[expand]/[delete]` as the length controller | **STOP** | Expansion remained inactive even after leaf-only training and bonus calibration; fixed larger holes worsened malformed output. |
| Restart-based verifier refinement | **INVALID** | The old verifier did not compare return values, and the corrected `restart` policy discards the draft. |
| HumanEval k-span “diffusion home turf” | **STOP** | Nested task sets, changing hole locations, grader-axis mismatches, and oracle-length asymmetry invalidate the family-level story. Correct controls do not establish a degradation-rate advantage for either family. |
| dLLM-specific pass@1 hardware sensitivity | **STOP AS HEADLINE** | AR and dLLM cross-architecture deltas are small, nonsignificant, and have no consistent direction. Only greater text/bit-level divergence remains observationally interesting. |
| Confidence-threshold C1/C2/C3 correction | **NOT A NEGATIVE RESULT; REDESIGN** | The tested thresholds triggered zero actions. This is an inactive grid, not evidence that correction cannot work. |
| DreamOn long-span advantage | **NOT ESTABLISHED** | Under the current `max_new=256, initial_masks=4, T=0` harness it under-generates badly, but the decisive per-item-headroom / initial-canvas control is missing. |

> **★ 2026-08-12 UPDATE — the missing canvas control has now been RUN, and it changes this table.**
> Source: A05 K1 gate + closeout, `Mixture-of-Memory/proposal/archive/A05-structural-dllm-cost-frontier/`
> (`A05_K1_CANVAS_SWEEP_VERDICT.md`, `A05_CLOSEOUT_VERDICT.md`, `evidence/`). 8 shards, exact item
> counts (164/378), 0 dups, evalplus with per-invocation self-test, ~21 GPU-h on `.73`.
>
> 1. **The under-generation IS the canvas artifact this row suspected.** Sweeping `initial_masks`
>    8→32 with every other sampler knob frozen: MBPP+ **.085 → .3545**, HE+ **.122 → .2134**
>    (→ **.2561** after fixing a stitch bug, and **.4817** at canvas=128). Empty outputs go
>    128/164 → 75/164 → **0/164** on HE+ and 332/378 → 165/378 on MBPP+. So *"DreamOn emits ~2
>    tokens and nothing on ~80% of items"* is a property of `initial_masks=8`, **not of the model**.
> 2. **But the row's own continuation criteria still FAIL**, so `NOT ESTABLISHED` stands as the
>    disposition, for a different reason than originally written: median emitted/gold ratio peaks at
>    0.46 (MBPP+ @c32) and 0.23 (HE+ @c128), both below the `0.8` bar in P1-C; MBPP+'s 65+ token
>    spans stay at ratio 0.000 even at c32. DreamOn under a bigger canvas is **no longer crippled
>    but still not a calibrated length controller** on full-program generation.
> 3. **Consequence for §1.1's first row (Scaffold full-program generation = STOP).** That STOP is
>    now *confirmed and can be closed with a reason* rather than left ambiguous: once DreamOn's
>    canvas is set sensibly it matches or beats Scaffold Medium (.177/.354) on both benchmarks, so
>    Scaffold has no advantage over its own family either. The internal-point framing was tested
>    and it failed.
> 4. Three defects in the archived DreamOn harness were found in the process and are corrected at
>    source: the logged `nfe` was `len(output.history)` (not a forward count; true 172.3/153.4, not
>    265.88/135.65), `mask_expansion`/`delete_eos_token` were always inert (confirmed by execution),
>    and the HE+ stitch double-indented bodies (understating every HE+ number). See
>    `DLLM_RESULTS_20260807.md` §"2026-08-12 CORRECTION BLOCK".

### 1.2 Findings that survive, with narrow wording

1. **Sampler/protocol sensitivity is large but benchmark-dependent.**
   - HumanEval+: plausible-grid spread `26.8` points.
   - MBPP+ current 10-cell mirror: the same conservative filter
     (`origin` and `alg_temp=0.5` removed) leaves seven cells with only a
     `4.5`-point spread. The `10.6`-point full-grid result reported by the
     remote audit is not independently reconstructible from this partial local
     mirror and should not be used until its raw cells are recovered.
   - The three confidence-based ranking rules at the reference setting are
     identical; the large contrast is confidence-ranked reveal versus random
     reveal.
   - At `T=0`, changing `top_p` changes reveal order without changing the
     tokenwise argmax and moves HumanEval+ by about `4.9` points.
   - Distinct-plausible cross-benchmark ranking transfer is only
     `rho=0.60, n=5`, not significant.

2. **NFE is not a fair cross-family cost unit.**
   An AR decode with KV caching, a flat full-canvas diffusion step, and a
   short mutable Scaffold serialization do different work. Always report:
   `tokens_fed`, context-length integral, wall time/GPU time, and a profiler
   proxy when available. These are still proxies, not literal FLOPs.

3. **Dream-Coder's fixed-protocol NFE curve saturates at 512 aggregate steps.**
   HE+ and MBPP+ both show zero aggregate gain from 512 to 1024. This is a
   checkpoint characterization, not a universal dLLM law.

4. **Low-NFE failure is initially syntactic, then semantic.**
   Syntax error rates fall sharply from 64 to 256 steps; after parseability is
   mostly recovered, functional correctness remains the bottleneck.

5. **Task surface changes the apparent result.**
   Line-aligned short infills are much friendlier to DreamOn than long spans or
   tokenizer-misaligned random spans. Infilling comparisons must report their
   own-gold ceiling and condition on it where the ceiling is non-flat.

6. **The typed runtime is a functioning engineering artifact.**
   Stable tree IDs, source maps, typed legal sets, capacity telemetry, C1 leaf
   remasking, C2 subtree collapse, and C3 deferral exist. The missing piece is
   execution-grounded routing with a semantically strong proposal model.

### 1.3 Important qualifications to the baseline story

- Qwen2.5-Coder-7B versus Dream-Coder-Instruct is a strong
  **matched-capacity/product baseline**, not a clean causal AR-versus-diffusion
  experiment: Base versus Instruct post-training is not matched.
- On HumanEval+, flat Dream diffusion is substantially more accurate than the
  local Qwen AR run, but far more expensive under both recorded cost proxies.
  This is a quality-cost trade-off, not universal AR dominance.
- On HumanEval+ `attended_context_sum`, AR is not cheaper than Scaffold Medium;
  it buys much higher quality with somewhat more context integral. Therefore
  do not repeat the old statement that AR strictly dominates Scaffold under
  every cost definition.
- On MBPP+, Qwen AR comes within roughly three points of Dream diffusion at a
  much smaller reported cost; this is the strongest current efficiency result.

---

## 2. Why the original architecture failed

The failure is not “one more hyperparameter away.” It is a stack of interacting
interface problems.

### 2.1 Structure and lexical semantics compete in one softmax

- Structural rows were initialized from semantically loaded ordinary tokens.
- Suppression was required to avoid literal `expand`, `delete`, and `mask`
  appearing as Python text.
- Restricted structural vocabularies magnified row-norm and prior differences.
- `[delete]` target imbalance once produced 16/16 empty programs.
- A `[STMT]` logit prior stopped recursive topology expansion but did not
  restore correct function headers or statements.

**Decision:** do not spend more runs tuning bonuses or merge scales for this
shared-head design.

### 2.2 Length control never became a dependable model capability

- Four masks truncate statements.
- Eight or sixteen masks create longer malformed continuations.
- Shallow-only length increases also fail.
- Leaf-only training still produced zero actual expansions.

**Decision:** future full-generation work, if attempted at all, needs an
explicit length head or length bucket, not a token action hidden in the lexical
softmax.

### 2.3 The training state distribution does not match full generation

Training mostly sees oracle-derived local corruptions. Inference repeatedly
conditions on its own wrong topology, headers, and lengths. A low masked-token
validation loss therefore does not predict executable program quality.

### 2.4 Segmented tokenization is a real distribution shift

The clean rendered text round-trips, but segment-wise BPE IDs differ from
normal whole-text BPE. A pretrained lexical model is therefore asked to reason
over an unfamiliar token segmentation throughout decoding.

### 2.5 Loss normalization can over-weight a few structural decisions

A small number of root/body labels can receive loss mass comparable with many
ordinary lexical targets. This offers a plausible mechanism for a few
structural updates overwhelming the pretrained language model.

---

## 3. What to keep

### 3.1 Typed mutable runtime

Keep and test as infrastructure:

- `scaffold_coder/decoder_runtime.py`
- `scaffold_coder/model_sampler.py`
- stable anchors and source-role maps;
- typed legal-vocabulary constraints;
- real `remask_leaf`;
- real `backtrack_structural_subtree`;
- capacity and termination telemetry.

Clarification: the C2 reverse primitive **is implemented**. What is missing is
the verifier-/analysis-guided Arm C pipeline that maps an observed program
failure to one of those typed anchors and then evaluates neural repair.

### 3.2 Evaluation and cost harness

Keep:

- official EvalPlus grading;
- full coverage assertions;
- base-and-plus conjunction;
- failure-inclusive cost summaries;
- NFE/token/context/wall-time sidecars;
- paired gained/lost accounting;
- own-gold ceiling checks for infilling.

### 3.3 Repair datasets already on disk

- `data/edit/humanevalpack_python.parquet`: 164 Python buggy/correct pairs;
  138/164 are one-line changes by a simple line-diff proxy, making this a good
  first oracle-localization gate.
- `data/edit/canitedit.parquet`: 105 instruction-driven code edits with tests;
  useful only after the simpler repair operator passes.

### 3.4 Agent trajectory pool

`runs/icc_track_c/pool.jsonl` contains 100,002 steps from 21,744 trajectories
over 19 environments. This is a potentially valuable follow-on asset for
trajectory repair or world-modeling, but it currently has no repair benchmark
or outcome-aligned experimental pipeline.

---

## 4. Priority proposals

## P0-A. Evaluation paper: protocol sensitivity and fair dLLM measurement

### Claim to test

> dLLM code scores and method rankings can move materially under seemingly
> secondary decoding choices; evaluation must freeze the sampler, separate
> reveal-order effects from token sampling, and report cross-family costs in
> comparable operational units.

### Existing evidence

- HE+ plausible spread: `26.8` points.
- MBPP+ partial local mirror: conservative plausible spread `4.5` points
  (`n=7`); this already demonstrates weaker sensitivity and poor transfer, but
  is not the final remote grid.
- HE+ order-only intervention at `T=0`: about `4.9` points.
- `top_p` optimum does not safely transfer from HE+ to MBPP+.
- NFE and success-conditioned means have already produced wrong conclusions in
  this repository.

### Minimum new experiment

1. A second dLLM family, with a **small orthogonal grid**, not another 25-cell
   Dream grid:
   - `T={0, reference}`;
   - `top_p={low, reference, 1.0}`;
   - confidence-ranked versus random reveal;
   - two deterministic/seed replicates where applicable.
2. One non-code task with a strict executable or exact-match grader.
3. Persist all raw solutions and metrics locally; do not rely on summary-only
   mirrors.
4. Add measured wall time/GPU time and a profiler-derived compute proxy.

### Kill gate

- If the second family's plausible spread is `<=3` points or not more than
  twice its seed/hardware floor, downgrade to a Dream-Coder case study.
- If order-only effects do not replicate outside HE+, remove the general
  reveal-order claim.
- Do not title the paper “sampler dominates methods” unless the comparison is
  pre-registered and reproduced across at least two models and two tasks.

### Novelty pressure as of 2026-08-08

This space is already active: CaRE audits compute/stochasticity; recent work
studies adaptive commitment, trajectory-aware gates, self-evaluation, and
experimental analysis of dLLMs. The remaining contribution must be the
**execution-graded code setting, order-only isolation, grader/cost failure
cases, and cross-benchmark non-transfer**, not a generic sampler sweep.

---

## P0-B. Method paper gate: strong-checkpoint typed local repair

This is the most promising method direction and the only one that directly
uses a distinctive repository asset.

### Core hypothesis

> A strong dLLM can repair an execution-relevant typed subtree while preserving
> the rest of a program exactly, more effectively than random remasking and
> more cheaply than full regeneration.

### Phase A: validate the repair operator with oracle localization

Use 64 held-out HumanEvalPack-Python bugs, stratified by:

- operator/value/variable/function misuse;
- missing versus excess logic;
- one-line versus multi-line gold edit;
- replacement-only versus length-changing edit.

For every item, map the gold diff to the smallest AST/typed subtree. Compare:

1. no repair;
2. matched-size random token span;
3. matched-size random AST subtree;
4. C1 oracle leaf remask where expressible;
5. **oracle typed C2 subtree collapse**;
6. flat oracle span remask with the same strong dLLM;
7. full Dream regeneration/restart;
8. Qwen native FIM or iterative AR repair.

All arms must use the same visible task information. No arm may receive gold
length unless all relevant arms do.

### Required metrics

- strict executable pass;
- paired gained / lost / unchanged;
- exact preservation outside the selected region;
- region size and expansion ratio;
- parse/nonempty/required-function rates;
- `tokens_fed`, context integral, wall time/GPU time;
- action rate and failure-inclusive costs.

### Phase A kill gate

Continue only if oracle C2:

- gains at least `5/64` tests;
- loses at most `2/64`;
- preserves all outside-region tokens exactly;
- costs at most `35%` more than the no-repair route;
- beats matched-random localization;
- and is competitive with, or offers a clear preservation/cost advantage over,
  full restart.

If oracle C2 cannot clear this gate, stop localizer training and stop the
typed-repair paper.

### Phase B: only after Phase A passes

Evaluate localizers:

- parser/static error span;
- undefined-name/type/operator rules;
- SBFL/Ochiai top-k;
- model LOO/regeneration probability;
- denoising-trajectory stability;
- execution/test-delta evidence;
- matched random.

Required gate:

- coverage `>=60%`;
- hit@3 or subtree IoU clearly above matched random;
- end-to-end gains not restricted to the oracle arm;
- learned/localized repair retains the Phase-A preservation advantage.

### Why this is still relatively open

Generic remasking and draft-refine are now crowded. Recent work includes
Targeted Remasking, Edit-Based Refinement, Multi-Block Editing, Speculative
Correction, and Detect-Remask-Repair. The defensible gap is narrower:

> **execution-grounded, typed-AST/subtree-localized, preservation-measured code
> repair against strong AR/FIM and full-regeneration controls.**

Do not present “remask then repair” alone as novel.

---

## P1-A. Full-generation architecture decomposition — one final gate only

This is not the recommended main line, but it is a clean way to decide whether
the representation itself is salvageable.

### New design

- freeze the strong Instruct backbone;
- keep the ordinary lexical LM head unchanged;
- add a separate topology head;
- add a separate length-bucket/slot head;
- eliminate lexical `[expand]/[delete]` actions;
- restore whole-text BPE or pass structure through a side channel;
- seed required function signatures and forbid an empty module root.

### 64-task decomposition

1. direct Instruct;
2. oracle topology + oracle length + current segmented BPE;
3. oracle topology + oracle length + whole-text BPE;
4. oracle topology + learned length head;
5. learned topology + learned length.

At most 256 optimizer steps for the small heads; no depth schedule, no
execution RL, and no large full-benchmark run before the gate.

### Kill gate

- If oracle topology + oracle length is below `80%` of direct-Instruct
  execution accuracy, or parse is below `95%`, terminate from-scratch Scaffold.
- If the learned length head retains less than `90%` of oracle-length execution
  or has more than `5%` generation failure, terminate dynamic length.
- Learned heads must reach parse `>=90%`, failure `<=5%`, improve at least five
  execution points over current Medium, and reduce vanilla by no more than two
  points before full evaluation.
- If a full run is still dominated by the matched-capacity AR baseline, close
  the complete-generation paper permanently.

---

## P1-B. Simultaneous non-oracle multi-site repair

The old k-span experiment did **not** test the most plausible dLLM advantage:
jointly repairing dependent sites in one canvas. DreamOn's non-oracle arm
filled holes serially.

### Proper task construction

Create fresh, AST-aligned, dependent edits such as:

- function signature plus all call sites;
- variable definition plus uses;
- paired delimiters;
- type annotation plus downstream conversions;
- duplicated condition or invariant changes.

Requirements:

- all sites are exposed simultaneously;
- `k+1` strictly retains the `k` original sites;
- no oracle length asymmetry;
- preserve unedited text exactly;
- compare native AR FIM and iterative AR repair;
- include a renamed/decontaminated version.

### Kill gate

- paired pass improvement over AR at least five points with CI above zero;
- parse no more than three points worse;
- `tokens_fed <=5x` AR and wall time `<=2x` AR;
- result survives the renamed/decontaminated set.

Otherwise terminate the multi-site line.

---

## P1-C. DreamOn length-controller falsification

> **★ 2026-08-12 STATUS: PARTLY EXECUTED on the full-program surface; VERDICT = terminate.**
> A05's K1 gate ran this diagnostic's core control (the `initial_masks` sweep + telemetry) on
> **HE+/MBPP+ full-program generation** rather than on the MultiLine infilling strata. Results:
> * The `initial_masks` sweep is decisive and **reverses the interpretation of the archived runs** —
>   MBPP+ .085 → .3545 and HE+ .122 → .2134/.2561 purely from the canvas. The pathology was budget.
> * **This section's own continue-criteria are NOT met**: median emitted/gold peaks at 0.46
>   (MBPP+ @c32) / 0.23 (HE+ @c128) versus the `0.8` bar; MBPP+ 65+ token spans stay at 0.000.
> * The "saturates near ~25 tokens" clause is **falsified as a model property** — mean emitted
>   length goes 2.35 → 12.87 → 48.53 as the canvas grows — but the ratio criterion still fails.
> * Therefore: **terminate long-span DreamOn as a capability claim**, per this section's own rule,
>   while recording that the archived collapse was a harness artifact.
> * Oracle (`per-item headroom`) arms were **NOT run** (dropped for budget; excluded from the
>   headline by the oracle invariant anyway). The **MultiLine infilling strata were NOT run** — that
>   surface belongs to B10.
> Evidence: `Mixture-of-Memory/proposal/archive/A05-structural-dllm-cost-frontier/evidence/`.

This is a small diagnostic, not a paper by itself.

Run only the long MultiLine strata with one frozen sampler implementation:

- `initial_masks={4,8,16,32}`;
- per-item headroom close to gold length plus a fixed margin;
- a common non-oracle headroom control;
- complete expand/delete and unfinished-mask telemetry.

Continue only if, on 65+ token spans:

- median emitted/gold length ratio is at least `0.8`;
- parseability is at least `90%`;
- the result is not obtained through oracle-only information.

If output still saturates near roughly 25 tokens, terminate long-span DreamOn.

~~Repository blocker: `scripts/generate_infilling.py` is missing, and the vendor~~
~~`MDMGenerator` path and model-native `diffusion_generate` path must be unified~~
~~before any new number is interpretable.~~

> **⚠️ 2026-08-12 — the "missing script" blocker was FALSE and had blocked this section since
> 2026-08-08.** `scripts/generate_infilling.py` **exists**: 392 lines, on the **zwfy6** disk, at
> `dllm_draft_104/scripts/generate_infilling.py`, already implementing all six arms
> (`dreamon_fim`, `dreamon_oracle`, `dream_fim`, `dream_prefix`, ...), `--initial_masks`, the
> oracle/non-oracle split, and first-line grading. Its companion `score_infilling.py` (197 lines)
> is there too. The original author searched **one disk**. This cluster has two genuinely different
> filesystems (wzc1 = LOCAL/.21, zwfy6 = .73/.82/.104) and `dllm_draft` exists on both with
> different contents, plus `dllm_draft_104` as a third checkout. **Never record a file as missing
> without checking both disks.** (The unification of `MDMGenerator` vs native `diffusion_generate`
> is a real, still-open item — only the "missing file" half was false.)

---

## P2. Coding-adjacent and non-coding applications

These are lower priority because recent literature already occupies much of the
surface.

### 4.1 Agent trajectory repair / text world models

Potential repository fit:

- the 100k-step, 19-environment trajectory pool is already available;
- a dLLM can condition bidirectionally on state anchors and revise an interior
  action/reasoning step;
- preservation and downstream-outcome tests are natural.

But this is **not empty territory**: recent work evaluates dLLMs in agentic
workflows and proposes masked dLLMs as text world models. A viable contribution
must be **repair of failed trajectories**, not generic dLLM agents or generic
world-model generation.

Minimum gate on at least 200 failed trajectories:

- `>=5` point outcome success gain over restart and continue-from-failure;
- no worse format/tool validity;
- changed-step count and compute are controlled;
- unchanged prefix/suffix preservation is reported.

### 4.2 Schema / JSON / XML generation

The runtime maps naturally to typed structured output, but this is crowded:
CFG-constrained diffusion decoding and multi-agent structured-data generation
already exist. Grammar-constrained AR is also a strong baseline.

Only proceed if dynamic typed generation can achieve:

- validity `>=99%`;
- semantic task quality no worse than constrained AR;
- cost no more than `1.2x` constrained AR;
- a real dynamic-structure benefit not already supplied by CFG constraints.

### 4.3 Process-based uncertainty

The repository exposes reveal order, temporal confidence, regeneration
probability, and cross-chain/hardware instability. However, temporal entropy,
sequence regeneration, trajectory-aware gating, and cross-chain localization
are now active research topics.

Only pursue a **code-specific execution-error detector**, with:

- error AUROC `>=0.75`;
- risk-coverage better than final entropy, LOO/regeneration probability, and AR
  log probability;
- explicit ablation of sampler and hardware confounds.

### 4.4 Adaptive capacity / anytime decoding

Do not revive this around the current weak Scaffold checkpoint. Recent
structured and stability-based adaptive decoding work raises the novelty bar.
Use adaptive capacity only after a task/model passes its quality gate.

Required improvement over the best fixed point:

- at most one point quality loss;
- at least `20%` token saving;
- lower failure rate.

---

## 5. Global experimental rules

Every future dLLM experiment in this repository must satisfy all of the
following.

1. **Natural baseline first.** Include a strong AR/FIM/constrained/search
   baseline before claiming a frontier.
2. **No hidden oracle advantage.** Length, hole location, test information, and
   task filtering must be symmetric.
3. **Official grading only.** For EvalPlus, Plus success is
   `base_status == pass AND plus_status == pass`.
4. **Paired accounting.** Report gained, lost, unchanged, and per-item costs.
5. **Failure-inclusive cost.** Never condition mean cost on successful
   termination.
6. **Multiple cost views.** Report NFE only as an internal descriptor; also
   report tokens, context integral, wall/GPU time, and profiler proxy where
   feasible.
7. **Preservation is a first-class metric** for editing and repair.
8. **Action-rate sanity check.** A correction policy that fires zero times has
   not been tested.
9. **Task-surface controls.** Align holes to tokens/AST where intended, freeze
   nested task sets, and measure the benchmark's own-gold ceiling.
10. **Sampler lock.** Freeze sampler configuration before comparing methods;
    report a small sensitivity envelope separately.

---

## 6. Immediate implementation order

### Week 0: repository hygiene

1. Keep all unknown/untracked pilot files untouched until they are individually
   audited.
2. Mark `*.BROKEN_VERIFIER` outputs as permanently excluded in any result
   index.
3. Do not use `TODO.md` as the scientific source of truth; several recovery
   items are stale.
4. Preserve the local MBPP sampler summary, but recover raw per-cell artifacts
   before publication.
5. Recover or rewrite the missing infilling generator before any DreamOn
   length experiment.

### Week 1: P0-B oracle repair gate

1. Build a deterministic 64-item HumanEvalPack-Python manifest.
2. Derive gold AST subtree and outside-region preservation masks.
3. Integrate strong-checkpoint flat remask and typed C2 collapse.
4. Add random-span, random-subtree, full-restart, and Qwen-FIM controls.
5. Run a 16-item CPU/model-free structural smoke.
6. Only after checking node availability, launch the 64-item neural gate.

### Week 1–2: P0-A replication

In parallel on another node:

1. select a second dLLM family;
2. run the compact orthogonal sampler grid;
3. replicate on one non-code exact/executable task;
4. package raw artifacts and a canonical analysis script.

### Decision point

- If repair Phase A passes, prioritize execution-grounded typed repair.
- If repair fails but sampler replication passes, prioritize the evaluation
  paper.
- If both fail, archive Scaffold as infrastructure/negative results and move to
  a new dLLM task rather than another code-generation retraining cycle.

---

## 7. Explicit “do not do next” list

- Do not run another depth-schedule SFT.
- Do not tune `[expand]`, `[delete]`, `[STMT]`, merge scale, or fixed hole size
  on the current shared-head architecture.
- Do not spend a node on Adaptive Scaffold before semantic quality passes.
- Do not revive k-span HumanEval with another task subset.
- Do not use confidence-only remasking as the proposed novelty.
- Do not train a fault localizer before oracle-localized repair succeeds.
- Do not report a dLLM efficiency result using NFE alone.
- Do not compare a method against only another diffusion decoder.
- Do not call the current Qwen/Dream comparison a causal AR-versus-diffusion
  proof.
- Do not infer a model property from the current DreamOn long-span harness
  without the missing length controls.

---

## 8. Key repository evidence

- `DLLM_RESULTS_20260807.md` — chronological results and retractions.
- `STAGE1_RESULTS.md` — original Scaffold/Plain/Schedule controls.
- `SEMANTIC_PRESERVATION_GATE.md` — structural-token, teacher-KL, length, and
  expansion failures.
- `CORRECTION_CALIBRATION.md` — inactive C1/C2/C3 threshold grid.
- `SAMPLER_VARIANCE_DECOMPOSITION.md` — sampler and cross-node audit.
- `SPANLEN_STRATIFIED_AUDIT.md` — task-surface and own-gold ceiling audit.
- `KSPAN_INFILLING_RESULTS.md`, `KSPAN_NONORACLE_ARM.md` — negative k-span
  record; later conclusions override earlier “survived” language.
- `scaffold_coder/decoder_runtime.py`,
  `scaffold_coder/model_sampler.py` — reusable typed revision primitives.
- `data/edit/humanevalpack_python.parquet`,
  `data/edit/canitedit.parquet` — next-gate datasets.
