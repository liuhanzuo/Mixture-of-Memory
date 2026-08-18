# Scaffold-Coder TODO

The heartbeat/watchdog processes actionable items from top to bottom. An item
may move to `DONE` only after its acceptance checks pass. Failed items retain
their logs and are either retried with a documented fix or marked `BLOCKED`.

## COMPLETED CONTROL MATRIX

- [x] **BASELINE-EVAL-001 — Measure same-harness Dream-Coder/DreamOn baselines**
  - Dream-Coder 512-NFE full baseline completed:
    HE+/MBPP+ = 50.00%/65.08%.
  - Dream-Coder 64-NFE HumanEval+ = 26.22%, with 50.6% parseability.
  - DreamOn-v0 four-mask body infilling mostly deletes the body and is retained
    as an infilling control rather than a full-solution baseline.
  - Direct deltas are recorded in `STAGE1_RESULTS.md`.

- [x] **SCHEDULE-ONLY-STAGE1-001 — Run the G1 pivot**
  - Ordinary Python tokens only; no meta-token targets or deterministic
    template expansion during training.
  - Uses the same depth-banded content/structure clocks and matched data/base.
  - The five-epoch 8-GPU run completed at `global_step_4465` in 5.116 active
    hours (40.93 allocated GPU-hours), with final validation loss 0.567.
  - A resumable 16-task neural decode smoke runs before full EvalPlus and must
    load the final checkpoint with zero generation exceptions and produce at
    least one parseable nonempty program.
  - The gate completed with 0 generation failures, 16/16 nonempty, and 3/16
    parseable at 128 NFE; full 512-NFE EvalPlus is now active.
  - HumanEval/HumanEval+ at 512 NFE completed at 3.66%/3.66%, with 58.54%
    parseability and zero generation exceptions; depth-2 and depth-3+ Plus
    pass@1 are both zero.
  - MBPP/MBPP+ completed at 12.96%/11.38%, with 65.87% parseability, zero
    generation exceptions, and zero Plus passes at depth 2 or 3+.
  - HumanEval+ at 128/64 NFE is 0%/0%; parseability falls to 11.59%/3.05%.
    Dream-Coder at 128 NFE reaches 41.46% HumanEval+ and 79.88% parseability.
    The Schedule low-NFE robustness hypothesis is rejected for this run.
  - Acceptance passed operationally; the scientific result is negative.

- [x] **PLAIN-STAGE1-001 — Run the critical matched SFT control**
  - Same Dream-Coder Base, 114,363 examples, five epochs, global batch 128,
    selected micro-batch 16/GPU, and standard fixed-length Dream decoding.
  - Uniform response masking only; no depth schedule, meta-tokens, collapse, or
    template runtime.
  - A full-batch, length-bucketed 8-GPU micro-batch 16/8 sweep runs immediately
    before Stage-1. It selects the fastest non-padding-token throughput with
    at least 5 GiB/GPU headroom and median padding below 50%; Stage-1 consumes
    the resulting recommendation automatically.
  - A separate 16-task neural decode smoke gates the full Plain EvalPlus run.
  - The bucketed sweep selected micro-batch 16 (14,050 non-padding tokens/s,
    0.241% median padding, 65.04 GiB/GPU headroom); full training launched at
    19:11 +08:00 with no gradient accumulation and 1,000-step checkpoints.
  - Full Stage-1 completed at `global_step_4465` in 1.914 active hours
    (15.31 allocated GPU-hours), with 13,736 mean non-padding tokens/s,
    0.352% mean padding, and final validation loss 0.131.
  - The 128-NFE HE16 gate completed with zero generation failures, 16/16
    nonempty, and 3/16 parseable.
  - HumanEval/HumanEval+ at 512 NFE completed at 28.05%/21.95%, with 79.88%
    parseability and zero generation errors. Plain exceeds Schedule by 18.29
    points on HumanEval+ and is stronger than Scaffold at depth 2 and 3+.
  - MBPP/MBPP+ completed at 30.42%/24.34%, with 75.40% parseability and zero
    generation errors. Plain exceeds Schedule by 12.96 MBPP+ points but remains
    7.67 points below Scaffold.
  - HumanEval+ at 128/64 NFE is 4.88%/0%; parseability is 14.02%/3.05%.
  - Acceptance passed: matched Plain isolates a large negative schedule effect.

- [x] **FINAL-COMPARISON-001 — Produce the complete attribution table**
  - HumanEval: Dream 512/128/64, Scaffold, schedule-only 512/128/64,
    plain 512/128/64.
  - MBPP: Dream 512, Scaffold, schedule-only 512, plain 512.
  - Report pass@1, parseability, generation errors, NFE, wall time, nesting
    depth slices, failure taxonomy, paired bootstrap intervals, and exact
    McNemar tests over task-level pass/fail differences.
  - Attribute registered active wall time, retries, and allocated 8-GPU hours
    for all three Stage-1 training runs.
  - Summarize first/last 100-step loss windows, last-500-step trend, and final
    validation loss as diagnostic convergence evidence.
  - Emit reproducibility manifests containing dataset/config/launcher hashes
    and checkpoint inventories for all three trained checkpoints. Scaffold and
    Plain manifests also hash the exact throughput-sweep recommendation that
    selected their micro-batch configuration.
  - Recover each run's launch-time Git commit from the remote reflog and parse
    the effective resolved Hydra config from its registered log, so later code
    edits cannot be misattributed to an already-running experiment.
  - Render `ops/artifacts/FINAL_COMPARISON.md` and a deterministic
    `g1_pivot_decision.json` so the next method arm is selected even when the
    final control finishes unattended.

- [x] **C2-CALIBRATION-001 — Calibrate structural backtracking off-benchmark**
  - Deterministic 32-row educational_instruct held-out set, stratified by
    compound depth; canonical solutions pass 32/32 in the bounded executor.
  - Sweep C0 and C2 thresholds 0.05/0.10/0.20/0.30 with one backtrack maximum.
  - Select only under joint execution, parseability, NFE, and cumulative-token
    gates documented in `CORRECTION_CALIBRATION.md`.
  - All tested thresholds produced zero backtracks and identical C0 outcomes;
    C0 remained selected.

- [x] **C1C3-CALIBRATION-001 — Calibrate leaf repair and structural deferral**
  - Reuse the exact C2 held-out tasks and C0 output.
  - Sweep confidence thresholds 0.05/0.10/0.20 separately for C1 and C3.
  - Apply the same execution, parseability, NFE, and cumulative-token gates.
  - All tested thresholds produced zero C1 remasks or C3 deferrals; C0 remained
    selected. The tested grid was inactive rather than evidence of harm.

- [x] **CORRECTION-VALIDATION-001 — Validate the routed policy disjointly**
  - Build a second 64-task oracle-clean held-out set with zero calibration-task
    overlap and a separate deterministic seed.
  - Compare selected correction versus freshly generated C0 with paired
    bootstrap, exact McNemar, parse, NFE, and cumulative-token accounting.
  - Unified routing selected C0. The disjoint 64-task run therefore reproduced
    C0 exactly (31.25% pass, 84.38% parse, delta 0) and did not pass the
    correction-improvement gate.

## G1 DECISION

- [x] **G1-PIVOT-001 — Select the next method arm from matched results**
  - Current C0 Scaffold result: HE+/MBPP+ pass@1 = 18.29%/32.01%.
  - Dream-Coder matched result: HE+/MBPP+ = 50.00%/65.08%.
  - Schedule-only is rejected: HE+/MBPP+ = 3.66%/11.38% at 512 NFE.
  - Plain SFT is the primary checkpoint for ordinary quality:
    HE+/MBPP+ = 21.95%/24.34%.
  - Meta-token Scaffold remains a separately costed structural/low-NFE arm:
    HE+/MBPP+ = 18.29%/32.01% at about 59/48 mean NFE.
  - Benchmark-signature seeding reduced NFE but did not improve parseability.
  - Correction routing selected C0 because the tested C1/C2/C3 thresholds did
    not activate. Higher adaptive thresholds require new calibration evidence.
  - Do not claim functional, deep-nesting, or low-NFE superiority for the
    current Scaffold/Schedule checkpoints.

## SFT RECOVERY EXPERIMENT

- [ ] **RECOVERY-BASE-RAW-001 — Evaluate the exact Base initialization**
  - Same 512-NFE HumanEval harness; isolates the effect of any SFT.

- [ ] **RECOVERY-BASE-PLAIN1-001 — One low-LR epoch from Base**
  - One epoch, LR 2e-6, bucketed micro-batch 16; compare with raw Base and the
    completed five-epoch LR-1e-5 Plain checkpoint.

- [ ] **RECOVERY-INSTRUCT-PLAIN1-001 — Gentle SFT from Instruct**
  - One epoch and LR 2e-6; measures catastrophic forgetting of the 50% HE+
    Instruct checkpoint.

- [ ] **RECOVERY-INSTRUCT-HIGHNOISE-001 — Add prompt-only supervision**
  - One epoch from Instruct with 20% all-mask, 30% high-noise t∈[0.8,1], and
    50% uniform states.

- [ ] **RECOVERY-COMPARISON-001 — Attribute initialization and over-training**
  - Produce aggregate and paired HumanEval results for all six raw/SFT arms;
    MBPP remains held out from recipe selection.

## LATER

- [ ] Add soft gating and desynchronized clocks.
- [ ] Add clause families, docstrings, classes, `try`, `with`, and async forms.
- [ ] Add oracle-scaffold and low-NFE experiments.
- [x] Calibrate C1/C2/C3 on held-out neural decodes and evaluate each policy at
  matched NFE and cumulative model-token budgets.
- [x] Run matched baselines and ablations.

## DONE

- [x] **CORRECTION-C2-CPU-001 — Add training-consistent structural backtracking**
  - Completed model-created construct and clause subtrees retain stable anchor
    IDs, commit provenance, confidence, and per-anchor backtrack counts.
  - C2 collapses an expanded subtree to one legal line/clause mask, then lets
    the same checkpoint regenerate it without any architecture or training
    change.
  - Triggering uses bounded mean-content confidence, minimum age, global and
    per-anchor budgets; nested candidates are repaired deepest-first.
  - EvalPlus and neural-smoke CLIs expose the policy, and process metrics
    report structural backtracks separately from C1 remasks and C3 deferrals.
  - Scripted tests cover both `[FUNC]` regeneration and clause restoration;
    the integrated suite has 109 passing tests.

- [x] **EVAL-GPU-001 — Complete Stage-1 EvalPlus**
  - Full HumanEval/HumanEval+ pass@1: 19.51%/18.29%.
  - Full MBPP/MBPP+ pass@1: 38.36%/32.01%.
  - HumanEval/MBPP generation budget failures: 15/35.
  - Four-commit HumanEval smoke reduced median NFE from 62.5 to 17 but
    parseability from 14/16 to 4/16.
  - Results and interpretation are in `STAGE1_RESULTS.md`.

- [x] **SAMPLER-GPU-001 — Run neural structured decoding**
  - Stage-1 checkpoint generated `identity` in 11 NFE.
  - HumanEval exposed deterministic expand/delete cycles; added repeated-state
    edit suppression and legality-aware vocabulary support.
  - The repaired 16-task C0 smoke completed all samples with 14/16 parseable.

- [x] **TRAIN-001 — Complete strict/local-barrier Stage-1**
  - Five epochs and 4,465 optimizer steps completed on 8 H20 GPUs.
  - Checkpoint/resume, custom-code packaging, and final checkpoint all passed.
  - Selected micro-batch 8/GPU at global batch 128 from the measured sweep.

- [x] **TRAIN-PROFILE-CPU-001 — Instrument distributed training efficiency**
  - Added opt-in synchronized step-time, examples/s, non-padding tokens/s,
    supervised tokens/s, and peak allocated/reserved memory telemetry.
  - Added checkpoint-free benchmark mode and a five-candidate 8-GPU
    micro-batch sweep that survives individual OOMs.
  - Added deterministic summary/recommendation logic with a 5 GiB headroom
    constraint; stage 1 consumes the resulting artifact automatically.
  - `TRAIN-PROFILE-CPU-001` rendered the Hydra config and passed all 61 tests
    remotely; real H20 measurements remain queued before stage 1.

- [x] **CORRECTION-C1C3-CPU-001 — Add bounded inference-time correction**
  - C1 records model-call/confidence provenance on lexical cells and can re-mask
    the lowest-confidence fraction periodically or at provisional completion.
  - Rule-emitted tokens are excluded; global, per-token, age, confidence, and
    NFE bounds prevent unbounded revision.
  - C3 defers low-confidence construct/clause commits while leaving the
    position as an in-distribution mask, then force-releases after a bounded
    number of calls to avoid deadlock.
  - EvalPlus and neural-smoke CLIs expose the policies; process metrics report
    remasks, correction rounds, and structural deferrals.
  - Scripted tests show both an actual leaf correction and a `[FUNC]` proposal
    changing to `[STMT]` before expansion.
  - `CORRECTION-C1C3-CPU-001` completed on the remote server at
    2026-07-22 20:26 +08:00; all 60 tests passed.

- [x] **EVAL-HARNESS-CPU-001 — Pin and validate EvalPlus**
  - Pinned HumanEval+ v0.1.10 and MBPP+ v0.2.0 locally.
  - Oracle HumanEval/HumanEval+ pass@1 is 1.0/1.0.
  - Oracle MBPP/MBPP+ pass@1 is 1.0/0.99735; the sole plus failure is the
    dataset canonical solution for Mbpp/255.
  - Environment, hashes, schema, and artifacts are documented in
    `EVAL_HARNESS.md`.

- [x] **SAMPLER-CPU-001 — Implement mutable reverse runtime and sampler loop**
  - Added line/token holes, local barriers, templates, clauses, edits, pass
    fallback, vocabulary constraints, confidence ranking, and hard budgets.
  - A scripted provider generated valid Python through the full iterative loop.
  - Runtime details are documented in `SAMPLER_RUNTIME.md`.

- [x] **COLLATOR-003 — Add collapse hazards and desynchronization**
  - Added literal all-masked and coupled region-collapse modes.
  - Added per-top-level-subtree clock offsets with shared full-sequence context.
  - Added exact per-position `u'_l/u_l` based on node/body depth.
  - Validated sigma 0.10/0.20/0.30 and collapse rates by length/clock bin.
  - Recursive nested-sibling offsets remain an optional extension rather than a
    v0 requirement.

- [x] **BASELINE-PLAIN-001 — Implement matched plain SFT control**
  - Added uniform response masking without scaffold vocabulary or templates.
  - Reuses the same canonical data, Base checkpoint, loss normalization, and
    FSDP trainer.
  - 5,000 control states analyzed in `BASELINE_STATS.md`.
  - CPU adapter/config validation and the integrated suite pass remotely.

- [x] **BASELINE-DREAMON-CPU-001 — Audit released DreamOn collator**
  - Verified released collator on canonical data.
  - Found delete/EOS targets are about 70% of supervised targets.
  - Confirmed released padding attends to almost the full 1,024-token canvas.
  - Confirmed expand ID is a reserved model row but absent from Base tokenizer.
  - Findings and fair B3-exact/B3-dynamic requirement are documented in
    `DREAMON_BASELINE_AUDIT.md`.

- [x] **BASELINE-DREAMON-DYNAMIC-001 — Implement dynamic-padding control**
  - Preserves line-middle masking, expand targets, hidden merge positions, and
    EOS/delete supervision.
  - Adds and saves only `<|expand|>` in reserved row 151,667.
  - Reduces mean effective attention length from 1,020.3 to 167.1.
  - 5,000-state statistics are included in `DREAMON_BASELINE_AUDIT.md`.

- [x] **TRAIN-ADAPTER-001 — Integrate collator with the 8-way FSDP trainer**
  - Added stochastic parquet dataset, dynamic batch padding, explicit loss
    weights, weighted shifted CE, token-row initialization before FSDP, and
    checkpoint token manifests.
  - Added exact Base-vs-resume model path handling.
  - Remote CPU validation imported the trainer and produced an 8-example batch
    with finite weights and target IDs inside the 152,064-row vocabulary.
  - A tiny real Dream model completed a finite forward/backward/optimizer step
    with nonzero structural-output and mask-input gradients.
  - Hydra config renders successfully; the current 56-test suite passes
    remotely.

- [x] **COLLATOR-002 — Add global clocks and DreamOn edits**
  - Implemented token/line expand-delete, static/dynamic-inverse merge, partial
    line masks, depth-specific leaf clocks, and one global `t`.
  - Depth-3 content bands reproduce the design schedule exactly.
  - Added adjacent structural-band overlap and local body transition states.
  - Validated 5,000 uniform-t states; results in `GLOBAL_BAND_STATS.md`.

- [x] **COLLATOR-001 — Implement deterministic hierarchical canvases**
  - Added segmented token canvases, role IDs, node maps, target IDs, loss masks,
    eligibility, prompt composition, root planning, template skeleton, leaf
    infilling, and local body-plan states.
  - Validated 1,000 eval rows and 4,295 body states.
  - Every clean segmented canvas decodes exactly to canonical Python.
  - Canvas statistics are recorded in `CANVAS_STATS.md`.

- [x] **DATA-001 — Build and validate the normalized stage-1 dataset**
  - Pinned and copied all 118,278 educational_instruct rows.
  - V0 grammar accepts 115,369 rows (97.54%).
  - Six accepted rows exceed 1,024 prompt+code tokens.
  - Final split: 114,363 train / 1,000 eval; cached canonical code and IR JSON.
  - All 115,363 cached IRs round-trip exactly to response text and `ast.parse`;
    train/eval overlap is zero.

- [x] **RUNTIME-001 — Implement the first symbolic typed-tree runtime**
  - Added v0 IR/parser, canonical renderer, final source-role map, target-role
    constraints, local-body oracle rewrite trace, elastic line/token edits, and
    hard budgets.
  - The current integrated suite has 56 passing tests locally and remotely.
  - Tests cover canonical round trip, doc stripping/`pass`, clause order,
    rule-only-hole exclusion, line/token edits, source-map coverage, empty root,
    budget termination, unique masks, and final `ast.parse`.

- [x] **SPEC-001 — Consolidate the design into `SPEC_v0.md`**
  - The Q&A two-granularity doctrine is normative.
  - Rule-only holes and predicted line labels are disjoint.
  - Module root, target table, local-body barrier, loss caveat, tokenizer
    invariant, optional-node semantics, limits, and claim boundary are defined.

- [x] **OPS-002 — Create a durable remote workspace**
  - Remote workspace:
    `/apdcephfs_zwfy6/share_304376610/pighzliu_code/dllm_draft`
  - Installed a 30-minute heartbeat in tmux session `scaffold-heartbeat`.
  - The watchdog records GPU/disk/process/run snapshots, monitors external runs
    read-only, and only restarts or terminates explicitly registered runs.
  - CPU queue launch → success artifact → DONE → next-item progression passed
    the `HEARTBEAT-SELFTEST`.
  - Stage-1 success artifacts are accepted only when their checkpoint pointer
    resolves to a complete final-step model index, every weight shard,
    optimizer state, training state, tokenizer, and Dream custom modules.
  - READY queue items with pre-existing success artifacts are blocked rather
    than launched; `ops/audit_queue.py` also checks duplicate IDs/success paths,
    command encoding, cwd, resources, and referenced executable scripts.
  - Registered training progress records one-time 25/50/75% milestones in
    history, making midpoint and late-stage timing auditable across retries.
  - No cron/system service is exposed in the container, so tmux survives SSH
    disconnects but cannot survive replacement of the entire allocation.

- [x] **OPS-001 — Connect to the 8-GPU server and inventory it**
  - SSH as root was verified on 2026-07-22.
  - Inventory is recorded in `ops/remote_inventory.md`.
  - All eight H20 GPUs were detected.
  - A pre-existing, unrelated 8-GPU OLMo training run is healthy and currently
    occupies essentially all GPU memory; it is monitored read-only and will not
    be killed or restarted without explicit ownership/authorization.
