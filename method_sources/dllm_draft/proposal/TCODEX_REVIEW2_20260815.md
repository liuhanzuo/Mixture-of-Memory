# dLLM Second-Opinion Review — 2026-08-15

The first review found several real seams, but its central literature claim is too strong. My bottom line is:

- **CAL has a verified ICLR-2026 DeLTa workshop venue.**
- **The universal “nobody reports NFE or wall-clock” claim is false.**
- **P1, P4, and P7 are DEAD as research proposals; P2, P3, P5, and P6 survive only after narrowing.**
- The strongest new direction is a **matched-lineage AR→diffusion suffix-utilization audit** whose first step is **0 GPU**.

## 1. Venue re-verification of the 8 unsafe rows

`NOT-FOUND-IN-OPENREVIEW` below means no native conference/workshop/journal submission note was found after exact-title, author, acronym, and arXiv-ID searches. A DBLP `CoRR` mirror is reported as a search hit but **not** treated as evidence that no accepted venue exists.

| Paper | Verdict | Venue / venueid | Exact evidence |
|---|---|---|---|
| **ρ-EOS**, 2601.22527 | **NOT-FOUND-IN-OPENREVIEW** | — | Exact-title hit `vBwjVjPF5B` is only a DBLP mirror: `CoRR 2026`, `dblp.org/journals/CORR/2026`; invitations are DBLP-only. |
| **iLLaDA**, 2606.25331 | **NOT-FOUND-IN-OPENREVIEW** | — | No exact note or DBLP mirror found under title, acronym, arXiv ID, or author searches. |
| **LLaDA-MoE**, 2509.24389 | **NOT-FOUND-IN-OPENREVIEW** | — | Exact-title hit `Q17lRHSzP1` is only a DBLP mirror: `CoRR 2025`, `dblp.org/journals/CORR/2025`. |
| **LLaDA 1.5**, 2505.19223 | **RETRIEVED: withdrawn ICLR submission** | `ICLR 2026 Conference Withdrawn Submission`; `ICLR.cc/2026/Conference/Withdrawn_Submission` | Forum `7dhe87Qjjq`, Submission 5193; `Withdrawn_Submission` invitation present; **no** `Camera_Ready_Revision`. Separate DBLP mirror `lcclq6FThF` does not override this. |
| **Dream-Coder**, 2509.01142 | **NOT-FOUND-IN-OPENREVIEW** | — | Exact-title hit `Fv8Hg10n97` is only a DBLP mirror: `CoRR 2025`, `dblp.org/journals/CORR/2025`. DreamOn is a different paper. |
| **CAL**, 2602.00476 | **RETRIEVED: ICLR 2026 DeLTa Workshop Poster** | `ICLR 2026 DeLTa Workshop Poster`; `ICLR.cc/2026/Workshop/DeLTa` | Forum `Hl8w6DC9VY`, Submission 14, under revised title *Training-Free Length Discovery for Diffusion Language Model Infilling*. Revision invitation present; no literal `Camera_Ready_Revision` string. |
| **LR-DLLM**, 2602.07546 | **NOT-FOUND-IN-OPENREVIEW** | — | No exact native note or DBLP mirror under title, acronym, arXiv ID, or author searches. |
| **ELF**, 2605.10938 | **NOT-FOUND-IN-OPENREVIEW** | — | No exact native note or DBLP mirror under title, arXiv ID, or author searches. |

**CAL identity qualification:** the OpenReview note does not carry an explicit arXiv-ID field, so linking it to 2602.00476 is formally **INFERRED**, not mechanically retrieved. The inference is nevertheless strong: identical three authors, explicit **CAL** acronym, same Oracle-Peak/Length-Bias mechanism, and the same headline improvements. The OpenReview work is therefore the workshop version of CAL, not merely an adjacent paper. ([openreview.net](https://openreview.net/attachment?id=Hl8w6DC9VY&name=pdf))

**LLaDA 1.5 is not an accepted ICLR paper.** The native OpenReview record is withdrawn, even though a separate `CoRR 2025` mirror coexists. ([openreview.net](https://openreview.net/forum?id=7dhe87Qjjq))

The arXiv identities for iLLaDA, ELF, Dream-Coder, and LLaDA-MoE themselves are verified; only their accepted-venue status remains not found. ([arxiv.org](https://arxiv.org/abs/2606.25331))

None of the eight was identified as an ACL-family publication, so the ACL-Anthology/DBLP override was not triggered.

## 2. Adversarial verdicts on P1-P7

### P1 — **DEAD as a standalone proposal**

The broad premise is no longer true: adaptive-length work now includes FLOP-aware length prediction, total adaptive-loop NFE, and wall-clock measurements. Most decisively, *VoidPadding* reports mean NFE and wall-clock for DAEDAL, ρ-EOS, and VoidExpansion; *Diffusion Language Models Are Natively Length-Aware* and *Predict-then-Diffuse* already frame adaptive length as a compute-quality problem. *VoidPadding* is within two months and is therefore **concurrent, not preemptive**, but it still falsifies the document’s present-tense “nobody reports” statement. More importantly, P1 dies on decisiveness: three already-known aggregate points, Spearman correlation with \(n=3\), and a 3× slope threshold chosen after seeing the 2.05× value cannot establish “bought rather than unlocked.” The compact `cells/*.json` files contain cell-level mean NFE, not the per-item NFE needed for the claimed paired frontier. If P1 returns exactly its predicted numbers, nothing scientific changes: larger canvases cost more and score more, which is already known. Keep it as an internal figure, not a proposal. ([arxiv.org](https://arxiv.org/abs/2606.17999))

### P2 — **NARROW**

No exact paper I found crosses DAEDAL and Fast-dLLM in the proposed factorial, so the direction is not preempted. The stated mechanism, however, is wrong: DAEDAL Stage 1 occurs **before denoising**, so parallel unmasking cannot erase the evidence used by Stage 1. Stage 2 already combines high-confidence filling with low-confidence insertion, making part of the proposed cross an ablation of DAEDAL’s own coupled sampler rather than competition between two independent methods. Moreover, adaptive length reduces sequence width while parallel decoding reduces iteration count; the gains may be multiplicative rather than substitutes. DyStruct already co-adapts length, blocks, and schedules, while recent concurrent studies indicate that cache/state refresh and prediction persistence may dominate confidence-threshold effects. Narrow P2 to a factorial over **Stage 1, Stage 2, parallel threshold, cache on/off, and refresh interval**, judged only by matched-quality NFE/FLOPs/wall frontiers. Falling expansion counts alone would not license “the methods are substitutes.” ([openreview.net](https://openreview.net/forum?id=Ic2A2gCseC))

### P3 — **NARROW**

The paper-by-paper audit remains useful, but its proposed estimand is muddled. Re-basing against the **best test-set canvas** is oracle hyperparameter selection, not a fair baseline. A validation-tuned fixed budget, a fixed deployment budget, a test-oracle budget, and an unknown-length adaptive method answer different questions and must not be collapsed into one score. LR-DLLM’s DreamOn `initial_masks=1` setting is explicitly an intentional stress test to isolate adjustment capability, not an accidentally undertuned default. The gate is also effectively pre-fired by its known LR-DLLM anchor, so it cannot falsify the proposal. Retain P3 only as a reporting-standard taxonomy that records, per paper, the selection information available to each method and the associated total compute. Do not claim that published method margins are invalid. ([arxiv.org](https://arxiv.org/abs/2602.07546))

### P4 — **DEAD**

DreamOn’s own ICLR-2026 paper already performs the central experiment. It sweeps initial mask lengths 4–64, includes `w/o Delete` and `w/o Expand` arms, shows that removing deletion sharply damages long-mask settings, and separately analyzes deletion broadcasting. Its total inference steps fall from 122.8 without broadcasting to 52.4 with broadcasting at initial length 64. Thus the proposal’s premise—“nobody has measured whether DreamOn’s delete actuator works”—is genuinely preempted by the paper under study. The proposed `oracle+128` arm is also ill-posed because DreamOn reports `L_max=128`; it would be an out-of-distribution or cap-violating stress test, not evidence for a missing practical contractor. ([openreview.net](https://openreview.net/forum?id=EQTPmqukiU))

### P5 — **NARROW**

The original “weights versus protocol” framing is already mostly dead. iLLaDA’s own Table 4 ablates confidence-based multiple-choice scoring; on ARC-Challenge the effect is only +0.6 points, far below P5’s pre-registered 3-point threshold. A residue remains because iLLaDA imports LLaDA and Dream headline numbers from their original papers instead of rerunning them under iLLaDA’s block-extension harness. But even a complete 2×2 cannot identify a causal “weight contribution”: iLLaDA simultaneously changes pretraining scale, architecture/optimization details, SFT corpus and duration, weights, and decoding. Rename it a **cross-release harness-transfer audit**, drop the already-answered ARC-C leg, and license only the conclusion that a quoted baseline was or was not protocol-dependent. ([arxiv.org](https://arxiv.org/abs/2606.25331))

### P6 — **NARROW**

ELF’s fixed slot count does not by itself prove that it inherits the same pathology as EOS-padded masked dLLMs. Its unconditional OWT examples are packed to 1024 tokens and have no per-example “correct output length”; its conditional tasks are trained with fixed target support, so sweeping slots changes both data support and the evaluator. Matching NFE is also insufficient because continuous-flow compute scales with slot count. The proposed first-step velocity norm is especially weak: velocity magnitude is entangled with diffusion time, injected noise, embedding norms, and CFG. More damagingly, *Continuous Language Diffusion as a Decoder-Interface Problem* already finds a stronger trajectory object—predicted-clean-state decoder margins—and uses it for early exit. P6 survives only as an exploratory **variable-length conditional-flow** study using a genuinely variable-target dataset, predicted-clean embeddings rather than raw velocity, and a matched slot-FLOP frontier. Correlation alone would not license a controller. ([arxiv.org](https://arxiv.org/abs/2605.10938))

### P7 — **DEAD as a causal proposal**

Reference-solution token length is not the required length of a correct program: HumanEval and MBPP permit many functionally equivalent solutions of different sizes. The proposed ratio therefore mixes truncation, reference-author style, formatting, and intrinsic difficulty. Its 159/5 HE+ strata are also too imbalanced for a credible exact test. More fundamentally, *Masked diffusion LLMs can use EoS tokens for hidden reasoning* shows—under fixed decoding-step controls and causal hidden-state interventions—that extra EOS positions can improve fully bidirectional dLLMs even when the shortest canvas already fits the answer. Spare slots can therefore be computational workspace, not merely space for longer output. P7 may remain a descriptive appendix on the 1,033-item infilling set, but it cannot support “canvas sensitivity is mainly benchmark composition.” ([arxiv.org](https://arxiv.org/html/2603.05197v2))

## 3. Is the "nobody reports NFE" seam real?

**No. The universal seam is false.**

| Paper | Absolute empirical total NFE? | Other cost accounting | Verdict |
|---|---|---|---|
| **DAEDAL** | No | Reports \(N_{\text{token}}\), \(E_{\text{token}}\), and \(E_{\text{ratio}}\); no runtime | Genuine omission |
| **ρ-EOS** | No explicit total count | Reports aggregate end-to-end evaluation runtime in seconds plus token metrics | Refutes “no wall-clock” |
| **CAL** | No | Reports 11.1–18.2 average candidate-length forwards, followed by an uncounted full decode | “Extra forwards without total” verified |
| **LR-DLLM** | Analytic, not empirical table | Explicitly derives \(F_{\text{total}}\); empirically reports `Forward Calls / Generated Tokens` | Refutes “ignores total NFE concept” |
| **DreamOn** | Yes, narrowly | Reports 122.8→52.4 total diffusion/inference steps in the deletion-broadcasting ablation | Direct counterexample |
| **VoidPadding** | Yes | Reports adaptive-method NFE and wall-clock together | Decisive counterexample |

Specific findings:

- **DAEDAL:** its algorithm has Stage-1 forwards plus Stage-2 denoising forwards, but the paper publishes only token-side efficiency. “Only effective-token ratio” is literally inaccurate because it also reports total and effective token counts; “no NFE or wall-clock” is accurate. ([openreview.net](https://openreview.net/forum?id=Ic2A2gCseC))
- **ρ-EOS:** each denoising iteration reuses one forward for unmasking and length control. Its tables report evaluation runtime in seconds for fixed-length, DAEDAL, and ρ-EOS, but not total NFE. ([arxiv.org](https://arxiv.org/html/2601.22527))
- **CAL:** the claim is verified precisely—Table 2’s `Stps.` corresponds to **11.1–18.2 additional first-step candidate forwards**, excluding formal iterative decoding. ([arxiv.org](https://arxiv.org/abs/2602.00476))
- **LR-DLLM:** Appendix F explicitly defines one forward evaluation, derives Stage-I and Stage-II calls, and writes \(F_{\text{total}}=F_I+F_{II}\). Its empirical Table 9 nevertheless reports only calls per adaptively generated token, whose denominator changes with the method. ([arxiv.org](https://arxiv.org/abs/2602.07546))
- **DreamOn:** Algorithm 2 performs one model probability evaluation per generation-loop iteration. Therefore its reported total diffusion steps are equivalent to total forwards in that ablation, although this equivalence is an inference from the algorithm and implementation rather than a column labelled “NFE.” ([openreview.net](https://openreview.net/forum?id=EQTPmqukiU))
- **VoidPadding:** under its init-64 LLaDA comparison, mean NFE is **228.82 for DAEDAL, 139.70 for ρ-EOS, and 172.10 for VoidExpansion**; VoidExpansion reports a **2.12× geometric-mean wall speedup** over DAEDAL. The paper states that NFE counts decoding forwards while wall time includes length selection and decoding. ([arxiv.org](https://arxiv.org/abs/2606.17999))

The corrected seam is:

> **Adaptive-length dLLM cost accounting is fragmented. DAEDAL reports token allocation only; CAL reports search calls but omits formal decoding; ρ-EOS reports aggregate runtime but not NFE; LR-DLLM derives total-call complexity but empirically normalizes by an endogenous generated-token count; DreamOn reports total steps only in one ablation; and VoidPadding reports aggregate total NFE plus wall-clock. None of this set provides a main-result, per-item joint distribution of absolute NFE, sequence work, synchronized wall-clock, and quality against fixed-length baselines.**

That residual is useful, but it is narrower than the document claims. Also, **NFE alone is not enough** when canvas length changes. A credible frontier should report:

\[
\text{NFE},\qquad \sum_t L_t,\qquad \sum_t L_t^2,\qquad
\text{wall-clock},\qquad \text{throughput},\qquad \text{quality}.
\]

The group’s `nfe_mean` values \(172.3\rightarrow393.7\rightarrow593.4\) remain valuable local evidence, but they no longer constitute the first disclosure of adaptive-length compute. They should support a causal or systems experiment, not P1’s three-point correlational claim.

## 4. New proposals (4-6, seven fields each)

### N1 — Matched-lineage suffix-utilization transfer under AR→diffusion conversion

1. **Title + falsifiable claim.**  
   **Claim:** converting Qwen2.5-Coder-7B into Dream-Coder changes how right context is used: under matched Base checkpoints and fixed non-oracle budgets, Dream-Coder’s FIM-minus-prefix gain is at least **5 pp larger**, while retaining at least **70%** of the items that Qwen solves only when the suffix is visible.

2. **Mechanism and named knobs.**  
   Run a paired 2×2:
   - model: `Qwen/Qwen2.5-Coder-7B` vs `Dream-org/Dream-Coder-v0-Base-7B`;
   - conditioning: native FIM vs prefix-only;
   - Dream knobs fixed at `initial_masks=32`, `max_new_tokens=64`, identical sampling temperature/top-p and post-processing;
   - benchmark: all 1,033 HumanEval-SingleLineInfilling items, grouped by parent `HumanEval/<id>`.
   
   Primary statistics are the difference-in-differences
   \[
   \Delta_{\text{surface}}
   =(D_{\text{FIM}}-D_{\text{prefix}})
   -(Q_{\text{FIM}}-Q_{\text{prefix}})
   \]
   and retention of Qwen suffix-rescue items.

3. **Why not already done; closest work and delta.**  
   Dream-Coder converts a Qwen2.5-Coder checkpoint and studies aggregate generation behavior; Repr-Align and OPDLM study knowledge preservation during AR→diffusion conversion. None reports paired item-level conservation of a conditional interface such as suffix utilization. The delta is therefore **what conditional capability transfers**, not simply whether aggregate code accuracy transfers. ([arxiv.org](https://arxiv.org/abs/2509.01142))

   Retrospective feasibility evidence—not the gate—is strong but confounded: existing maps give Qwen suffix gain \(93.51-66.02=27.49\) pp and Dream-Coder-Instruct gain \(88.00-50.24=37.75\) pp. Of 296 Qwen suffix-rescue items, 242 are also Dream suffix rescues, or **81.8%**.

4. **PRE-REGISTERED KILL GATE.**  
   **KILL if** \(\Delta_{\text{surface}}\) on HumanEval-SingleLineInfilling is **<5.0 pp**, **or** the parent-problem-grouped bootstrap 95% CI includes zero, **or** Dream retains **<70%** of Qwen’s suffix-rescue set under the fixed Base-checkpoint test.  
   **Expected outcome:** likely survives narrowly; I expect the confounded +10.26 pp pilot to shrink to roughly 5–8 pp after removing Instruct and length-handout effects.

5. **Cheapest decisive first experiment.**  
   Step 1 is **0 GPU**: finish the parent-grouped overlap analysis, freeze extraction/templates, and verify tokenizer identity. The cheapest decisive generation is two Dream-Coder-Base arms, approximately **2–4 GPU-hours on 8 cards**. The required checkpoint is already local; exact public repo: `Dream-org/Dream-Coder-v0-Base-7B`.

6. **Invalidating confounds.**  
   Instruct-versus-Base mismatch; Dream oracle or adaptive length leaking gold information; differing max output lengths; chat-template differences; tokenizer mismatch; parent-item leakage across train/test folds; and divergent code extraction or grader versions.

7. **Licenses / must not claim.**  
   A positive result licenses: “AR→diffusion conversion changed how a matched lineage exploits visible right context.” It must **not** be presented as diffusion generally outperforming AR infilling, or as proof that AR hidden representations were causally preserved.

---

### N2 — Capacity or revision budget? A per-item NFE-matched canvas intervention

1. **Title + falsifiable claim.**  
   **Claim:** at least half of DreamOn’s corrected HE+ gain from canvas 32 to 128 is caused by the accompanying increase in refinement compute rather than by extra representational space.

2. **Mechanism and named knobs.**  
   Replay canvas 128 while setting each item’s `max_nfe_i` to its observed canvas-32 NFE. Hold fixed:
   - `initial_masks=128`;
   - `expand_budget`;
   - transfer-token rule and temperature;
   - prompt/template and grader.
   
   Add a remaining-budget reveal quota so active masks are resolved before the cap. Record expansion, deletion, remasking/revision count, emitted length, parseability, NFE, \(\sum L_t\), and \(\sum L_t^2\). Include abrupt-cap and quota-finish schedule controls.

3. **Why not already done; closest work and delta.**  
   P1 is observational. Remasking audits show that gains depend strongly on settings, while *Answer First, Reason Later* crosses canvas and refinement behavior on reasoning tasks. The missing delta is a **per-item compute intervention on a trained insertion/deletion model in executable code generation**. ([arxiv.org](https://arxiv.org/abs/2606.12232))

4. **PRE-REGISTERED KILL GATE.**  
   The native corrected gap is \(48.17-25.61=22.56\) pp.  
   **KILL if** NFE-matched canvas-128 HE+ remains **>11.3 pp above** canvas 32 under the quota-finish test—i.e. if matching NFE removes less than half the native gap.  
   **Expected outcome:** I weakly expect the gate to **fire**. Canvas 128 emits much more actual code, so physical output capacity probably explains more than half the gain.

5. **Cheapest decisive first experiment.**  
   Step 1 is **0 GPU**: confirm that raw per-item c32 NFE telemetry is recoverable and unit-test NFE caps/reveal quotas. One decisive capped c128 arm is approximately **2–4 GPU-hours**; adding the schedule control makes the total **4–6 GPU-hours**. Exact model repo: `Dream-org/DreamOn-v0-7B`.

6. **Invalidating confounds.**  
   The cap changes the unmasking schedule; forced completion may degrade quality independently of compute; c32 raw per-item NFE may no longer be available in the compact evidence bundle; expansion/deletion could make the same NFE correspond to different sequence work; corrected versus as-run stitching must not be mixed.

7. **Licenses / must not claim.**  
   This can partition the DreamOn canvas effect into space and compute components. It cannot establish that DAEDAL, CAL, or ρ-EOS are inefficient, nor that the resulting capped sampler is deployable.

---

### N3 — Does “parallel generation” survive the H20→B200 hardware shift?

1. **Title + falsifiable claim.**  
   **Claim:** at a validation-selected matched-quality operating point, the Dream-Coder/Qwen latency ratio is at least **20% lower on B200 than on H20**, demonstrating that reported dLLM speed rankings are materially hardware-dependent.

2. **Mechanism and named knobs.**  
   Compare `Dream-org/Dream-Coder-v0-Instruct-7B` against `Qwen/Qwen2.5-Coder-7B` on identical code tasks with:
   - hardware: one H20 and one B200;
   - batch sizes \(\{1,8\}\);
   - output budgets \(\{64,256\}\);
   - frozen tensor-parallel degree, dtype, FlashAttention, compilation, CUDA graphs, warm-up count, and power mode;
   - validation-selected model operating points within 1 pp pass@1.
   
   Report median and p95 end-to-end latency, throughput, peak memory, kernel time, post-processing time, and pass@1.

3. **Why not already done; closest work and delta.**  
   Existing systems studies already show that open dLLMs often lose to AR models in throughput, especially at larger batches, and that context length and iterative refinement dominate. Those studies primarily profile A6000/A100/H100-class systems. The delta is a paired **sm90 H20 versus sm100 B200 crossover**, where compute-heavy dLLM passes may benefit differently from bandwidth-heavy AR decoding. ([arxiv.org](https://arxiv.org/pdf/2510.18480))

4. **PRE-REGISTERED KILL GATE.**  
   **KILL if** the geometric mean
   \[
   \frac{(\text{Dream latency}/\text{Qwen latency})_{\mathrm{B200}}}
        {(\text{Dream latency}/\text{Qwen latency})_{\mathrm{H20}}}
   \]
   is **>0.80** across the four batch/length cells, or if matched quality differs by >1 pp.  
   **Expected outcome:** genuine coin flip; I expect B200 to help the dLLM relatively more, but a 20% shift may be too aggressive and the gate may fire.

5. **Cheapest decisive first experiment.**  
   Step 1 is **0 GPU**: derive arithmetic-intensity predictions and freeze the profiling protocol. The decisive dual-hardware run costs approximately **8–12 GPU-hours total**, using one GPU of each type once nodes become free. It cannot start immediately because all 40 cards are occupied.

6. **Invalidating confounds.**  
   Different CUDA/PyTorch kernels across nodes; thermal or power caps; compilation asymmetry; tensor-parallel communication; padded batching; asynchronous CPU grading; output-length mismatch; tokenizer/post-processing overhead; or comparing a high-quality AR point against a lower-quality dLLM point.

7. **Licenses / must not claim.**  
   A positive result licenses a hardware-specific crossover statement. It must not be turned into “B200 makes dLLMs faster than AR models” unless the absolute matched-quality frontier actually crosses.

---

### N4 — Sampler transfer matrix for few-step distilled dLLMs

1. **Title + falsifiable claim.**  
   **Claim:** at least half of a few-step distilled checkpoint’s 8-step gain is sampler-specific and disappears when the checkpoint is decoded with a compatible but foreign LLaDA schedule.

2. **Mechanism and named knobs.**  
   Run a 2×2:
   - checkpoint: `GSAI-ML/LLaDA-8B-Instruct` vs `Zhouhhy/TAD-LLaDA-Insturct-Speed`;
   - sampler: vanilla LLaDA confidence schedule vs the TAD release schedule;
   - NFE: \(\{8,32,128\}\);
   - fixed canvas, prompts, tokenizer, temperature, and grader.
   
   Measure HE+/MBPP+ pass@1, TPF, NFE, wall-clock, and output agreement with the 128-step endpoint.

3. **Why not already done; closest work and delta.**  
   FS-DFM, CDLM, T3D, and TAD all optimize few-step generation, generally evaluating each trained checkpoint under its intended sampler. The missing test is a **checkpoint×sampler transfer matrix** distinguishing a model-internal gain from co-adaptation to a particular release schedule. ([openreview.net](https://openreview.net/forum?id=ue1zFeD275))

4. **PRE-REGISTERED KILL GATE.**  
   **KILL if** the TAD checkpoint retains **>50%** of its official-sampler 8-step pass@1 improvement under the vanilla sampler on both HE+ and MBPP+, with checkpoint×sampler interaction <2 pp.  
   **Expected outcome:** I slightly expect the gate to fire; much of the gain may be model-internal rather than sampler-specific.

5. **Cheapest decisive first experiment.**  
   Step 1 is **0 GPU**: diff both repositories’ timestep parameterization, scheduler, tokenizer, and custom model code, and determine whether sampler swapping is semantically valid. If valid, the full matrix is approximately **6–10 GPU-hours**.

6. **Invalidating confounds.**  
   Incompatible timestep definitions; custom output heads; different mask IDs; schedule-specific training conditioning that makes the foreign sampler nonsensical; stochastic pass@1 variance; different canvas maxima; or silently using different code-extraction routines.

7. **Licenses / must not claim.**  
   A positive interaction licenses “this checkpoint and sampler are co-adapted.” It does not imply that the method is overfit or that authors made an unfair comparison.

---

### N5 — Does preference optimization improve code by learning a different length policy?

1. **Title + falsifiable claim.**  
   **Claim:** at least half of LLaDA-1.5’s raw HumanEval improvement over LLaDA-Instruct is associated with changed termination/output-length behavior rather than a length-matched semantic improvement.

2. **Mechanism and named knobs.**  
   Evaluate:
   - `GSAI-ML/LLaDA-8B-Instruct`;
   - `GSAI-ML/LLaDA-1.5`;
   - canvases \(\{128,512\}\);
   - identical prompts, chat template, confidence sampler, steps, temperature, and EOS handling.
   
   Log emitted length, first EOS position, EOS density, NFE, \(\sum L_t\), pass@1, and model×canvas interaction. Estimate the raw paired model effect and a length/NFE-stratified effect; report the latter as descriptive adjustment, not causal mediation.

3. **Why not already done; closest work and delta.**  
   VRPO reports aggregate math, code, and alignment gains. Later post-training methods such as TraFL and RLDF study trajectory diversity or policy-loss estimation, but do not decompose gains into semantic improvement versus learned termination policy. ([openreview.net](https://openreview.net/forum?id=7dhe87Qjjq))

4. **PRE-REGISTERED KILL GATE.**  
   **KILL if** the raw LLaDA-1.5 HE+ gain is **<2 pp**, or if length/NFE matching removes **<25%** of the gain **and** the checkpoint×canvas interaction is **<3 pp**.  
   **Expected outcome:** I expect the gate to fire. VRPO’s improvements are broad and modest, so output-length policy probably explains little of them.

5. **Cheapest decisive first experiment.**  
   Step 1 is **0 GPU**: audit both configs, response-length distributions described by the training recipes, EOS IDs, and sampler compatibility. The HE+-only four-cell evaluation is approximately **6–10 GPU-hours**; add MBPP+ only if the HE+ gate survives.

6. **Invalidating confounds.**  
   LLaDA-1.5 differs from its predecessor in more than preference loss; mediator matching is not causal identification; EOS behavior may be sampler-specific; fixed canvases may interact with model calibration; Base and Instruct checkpoints must not be mixed; prompt/template differences could dominate a small raw gain.

7. **Licenses / must not claim.**  
   A positive result concerns one VRPO-trained release. It must not be generalized to “preference optimization rewards verbosity” or used to deny residual semantic improvements.

## 5. Unified ranking + the single first experiment

Dead proposals P1, P4, and P7 are excluded. Existing survivors appear only in their narrowed form.

| Rank | Proposal | First step | Decisive GPU cost | Expected information / cost |
|---:|---|---:|---:|---|
| **1** | **N1 — matched-lineage suffix-utilization transfer** | **0 GPU** | 2–4 GPU-h if screen passes | High: uses existing paired maps and asks a clean AR↔diffusion mechanism question |
| **2** | **P3-R — deployment/validation/oracle/adaptive reporting taxonomy** | **0 GPU** | 0 | Medium, but fully executable while all GPUs are busy |
| **3** | **N2 — NFE-matched canvas intervention** | **0 GPU** | 2–4 GPU-h, 4–6 with control | High causal information; cheap once a node frees |
| **4** | **P6-R — variable-length conditional ELF under matched FLOPs** | **0 GPU** | 2–4 GPU-h | High novelty, but substantial metric/support risk |
| **5** | **N4 — few-step checkpoint×sampler transfer** | **0 GPU** | 6–10 GPU-h | High information about whether distilled gains reside in weights or sampler coupling |
| **6** | **N5 — VRPO length-policy decomposition** | **0 GPU** | 6–10 GPU-h | Medium; likely negative but scientifically clean if raw gain is large enough |
| **7** | **N3 — H20/B200 wall-clock crossover** | **0 GPU** | 8–12 GPU-h | Valuable systems result, but hardware-bound and not immediately launchable |
| **8** | **P5-R — iLLaDA cross-release harness transfer** | **0 GPU** | 8–12 GPU-h | Low-medium; likely confirms only a modest protocol effect |
| **9** | **P2-R — length stages × threshold × cache/refresh factorial** | **0 GPU** | 24–36 GPU-h | Potentially high-value paper, but lowest ratio because of cost and fidelity risk |

**Run first: N1, matched-lineage suffix-utilization transfer.**

Its **first step needs no GPU**. The coarse retrospective evidence is already strong enough to justify completing a leakage-safe, parent-grouped 0-GPU analysis, while still being confounded enough that no claim should be made before the matched Base test. It is the best current use of the period in which all 40 GPUs are unavailable.

Do **not** queue N1’s GPU arms until the 0-GPU protocol has frozen:

1. the exact Qwen and Dream Base checkpoints;
2. FIM/prefix templates;
3. fixed Dream canvas and output budget;
4. parent-problem grouping;
5. code extraction and grader environment;
6. the 5-pp/70%-retention kill gate.

## 6. What I could not verify (authority + exact error)

1. **OpenReview direct note lookup remained challenge-gated.**

   Authority tried:

   ```text
   GET https://api2.openreview.net/notes?forum=Hl8w6DC9VY
   ```

   Exact result:

   ```text
   HTTP 403
   {"name":"ChallengeRequiredError",
    "message":"Challenge verification required (2026-08-15-5699687)",
    "status":403,
    ...}
   ```

   Venue verification therefore used the successful `api2 /notes/search` endpoint and inspected returned `content`, `venueid`, `domain`, and `invitations[]`.

2. **Semantic Scholar provided no evidence.**

   Authority tried:

   ```text
   GET https://api.semanticscholar.org/graph/v1/paper/ARXIV:2601.22527
       ?fields=title,venue,year,externalIds
   ```

   Exact result:

   ```text
   HTTP 429
   x-amzn-ErrorType: TooManyRequestsException

   {"message":"Too Many Requests. Please wait and try again or apply for a key
   for higher rate limits. https://www.semanticscholar.org/product/api#api-key-form",
   "code":"429"}
   ```

   Semantic Scholar silence is therefore **not** evidence of venue absence or missing prior art.

3. **CAL’s arXiv-to-OpenReview identity is not mechanically linked.** OpenReview does not expose `2602.00476` in the note. The same-work mapping is **INFERRED** from authors, acronym, mechanism, abstract, and identical headline results. Venue and venueid themselves are RETRIEVED.

4. **No native OpenReview record was found for iLLaDA, LR-DLLM, or ELF.** Searches included exact titles, acronyms, arXiv IDs, and multiple authors; all returned HTTP 200 but no exact target. This means **NOT-FOUND-IN-OPENREVIEW**, not “confirmed preprint.”

5. **ρ-EOS, LLaDA-MoE, and Dream-Coder produced only DBLP mirror records.** Those mirrors cannot exclude a lagging or retitled accepted record, as DAEDAL demonstrated. Their venue status therefore remains NOT-FOUND.

6. **Predict-then-Diffuse’s IJCNN-2026 status was not independently checked in IEEE proceedings.** Its current primary arXiv/PDF metadata states “Accepted for publication in IJCNN 2026,” which I treated only as an author-supplied acceptance statement, not an independently venue-verified record. ([arxiv.org](https://arxiv.org/abs/2605.04215))

7. **The existing compact A05 evidence does not expose per-item NFE.** `cells/*.json` contains per-item pass maps and cell-level cost aggregates. P1’s claimed paired quality-per-NFE analysis requires recovery of the original per-item metrics JSONL; I did not verify that those raw telemetry files remain reachable from the current canonical filesystem.