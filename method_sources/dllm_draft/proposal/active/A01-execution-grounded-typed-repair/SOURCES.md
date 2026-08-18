# Sources and Related-Work Audit

更新：2026-08-08。

本文件区分：

- **内部可复核证据**；
- **外部一手论文/benchmark**；
- **直接 collision**；
- **只能作为 topic signal、不能作为强证据的工作**。

---

## 1. Internal evidence and implementation

### Canonical decision records

- `../../../DLLM_SALVAGE_ROADMAP_20260808.md`
- `../../../DLLM_RESULTS_20260807.md`
- `../../../STAGE1_RESULTS.md`
- `../../../SEMANTIC_PRESERVATION_GATE.md`
- `../../../CORRECTION_CALIBRATION.md`
- `../../../SAMPLER_VARIANCE_DECOMPOSITION.md`
- `../../../SPANLEN_STRATIFIED_AUDIT.md`

### Runtime primitives

- `../../../scaffold_coder/decoder_runtime.py`
  - `remask_leaf`
  - `completed_structural_subtrees`
  - `backtrack_structural_subtree`
- `../../../scaffold_coder/model_sampler.py`
  - deepest-first C2 trigger and correction accounting
- `../../../tests/test_model_sampler.py`
  - scripted C2 collapse and final-budget tests
- `../../../scripts/generate_evalplus_scaffold.py`
  - current neural generation/cost harness

### Invalid or incomplete refinement assets

- `../../../scripts/refine_verifier_guided.py`
  - corrected verifier;
  - `restart` discards the draft;
  - `remask` is a prompt-splice proxy, not arbitrary canvas remasking.
- `../../../runs/refine_*.BROKEN_VERIFIER`
  - 16 directories; permanently excluded.
- `../../../scripts/pilot_oracle.py`
- `../../../scripts/pilot_repair.py`
- `../../../scripts/pilot_typed.py`
- `../../../scripts/pilot_mech.py`
- `../../../scripts/pilot_mut_k1.py`
  - untracked exploratory scripts;
  - stdout-only or alignment-fragile;
  - not citable evidence.

### Datasets

- `../../../data/edit/humanevalpack_python.parquet`
  - 164 Python repair pairs.
- `../../../data/edit/canitedit.parquet`
  - 105 instruction-driven edit tasks.

### Models and baseline artifacts

- `../../../models/Dream-Coder-v0-Instruct-7B`
- `../../../models/Qwen2.5-Coder-7B`
- `../../../models/Scaffold-v0-stage1-7B`
- `../../../runs/dream_coder_instruct_heplus_r2/`
- `../../../runs/dream_coder_instruct_mbppplus_r2/`
- `../../../outputs/ar_qwen25coder7b_base_252/humaneval/`

---

## 2. dLLM code models and variable-length infilling

### Dream-Coder

- **Dream-Coder 7B: An Open Diffusion Language Model for Code**
- arXiv: `2509.01142`
- Relevance:
  - released Dream-Coder checkpoints;
  - any-order code generation;
  - SFT/RL recipe.
- Boundary:
  - backbone/prior model, not local repair.

### DiffuCoder

- **DiffuCoder: Understanding and Improving Masked Diffusion Models for Code
  Generation**
- arXiv: `2506.20639`
- Relevance:
  - code-specialized dLLM;
  - generation-order analysis;
  - coupled-GRPO.
- Boundary:
  - complete generation/RL rather than typed localized repair.

### DreamOn

- **DreamOn: Diffusion Language Models For Code Infilling Beyond Fixed-size
  Canvas**
- arXiv: `2602.01326`
- Relevance:
  - variable-length code infilling;
  - expand/contract states.
- Collision:
  - length elasticity and code infilling are not novel contributions here.
- Required caveat:
  - local repository audits find strong task-surface and harness dependence;
  - do not generalize the local long-span failure without missing controls.

---

## 3. dLLM remasking, editing, and refinement

### Targeted Remasking

- **Targeted Remasking: Replacing Token Editing with Token-to-Mask Refinement
  in Discrete Diffusion Language Models**
- arXiv: `2605.26436`
- Direct collision:
  - resets suspected tokens to mask;
  - training-free detection strategies;
  - code included among evaluated domains.
- Remaining gap:
  - typed AST region;
  - execution-grounded localization;
  - outside-region program preservation.

### SCOPE + D3IM

- **Revise, Don't Freeze: Sampler-Matched Training for Self-Correcting Masked
  Diffusion Language Models**
- arXiv: `2606.01026`
- Direct collision:
  - visible-token correction;
  - preservation bias;
  - code benchmark gains.
- Remaining gap:
  - no typed program-repair unit or execution-localized patch protocol.

### Multi-Token Residual Prediction

- **Multi-Token Residual Prediction**
- arXiv: `2605.18817`
- Direct collision:
  - residual next-step prediction;
  - remasking after aggressive decoding;
  - large HumanEval gains in one regime.
- Use:
  - method-gain and compute baseline in Related Work.

### Edit-Based Refinement / ME-DLM

- **Edit-Based Refinement for Parallel Masked Diffusion Language Models**
- arXiv: `2605.09603`
- Direct collision:
  - insertion/deletion/replacement after an initial complete response;
  - HumanEval evaluation.
- Remaining gap:
  - execution-localized typed repair and exact preservation controls.

### Multi-Block Editing

- **Beyond Block Boundaries: Multi-Block Editing for Diffusion Large Language
  Models**
- arXiv: `2607.22663`
- Direct collision:
  - training-free and trained reopening of decoded spans;
  - cross-block future context.
- Remaining gap:
  - program semantics, typed regions, and fault localization.

### Speculative Correction

- **Speculative Correction: Draft-then-Refine Decoding for Diffusion Language
  Models**
- arXiv: `2608.02625`
- Direct collision:
  - full draft followed by global/local diffusion refinement;
  - MBPP result and latency-matched controls.
- Required baseline implication:
  - complete-draft refinement must be compared against typed local repair.

### Detect-Remask-Repair

- **Detect, Remask, Repair: Diffusion Editing for Faithful Summarization of
  Evolving Contexts**
- arXiv: `2606.12807`
- Direct collision:
  - localized detection/remasking/repair;
  - preservation-speed-faithfulness trade-off.
- Remaining gap:
  - execution-graded code, AST typing, regression tests.

### Confidence-remasking null result

- **Re-evaluating Confidence Remasking in Masked Diffusion Language Models**
- arXiv: `2606.12232`
- Relevance:
  - confidence-only remasking can have little benefit under standard settings;
  - motivates oracle-operator-first ordering.

---

## 4. dLLM uncertainty, search, and commit order

These works preclude a generic novelty claim around trajectory confidence.

- **Efficient Self-Evaluation for Diffusion Language Models via Sequence
  Regeneration**, arXiv `2603.02760`.
- **OSCAR: Orchestrated Self-verification and Cross-path Refinement**,
  arXiv `2604.01624`.
- **TACG: Trajectory-Aware Commit Gating for Diffusion Language Model
  Decoding**, arXiv `2607.03236`.
- **Search or Accelerate: Confidence-Switched Position Beam Search for
  Diffusion Language Models**, arXiv `2602.10953`.
- **Improving Diffusion Language Model Decoding through Joint Search in
  Generation Order and Token Space**, arXiv `2601.20339`.

Proposal boundary:

```text
trajectory/regeneration confidence is a candidate localizer feature,
not the primary methodological novelty.
```

---

## 5. Program repair and fault localization

### Graph2Diff

- **Learning to Fix Build Errors with Graph2Diff Neural Networks**
- arXiv: `1911.01205`
- Relevance:
  - graph representation;
  - AST diff prediction with code-location pointers.

### Beep

- **Beep: Fine-grained Fix Localization by Learning to Predict Buggy Code
  Elements**
- arXiv: `2111.07739`
- Direct collision:
  - AST-path token localization;
  - predicts repair actions;
  - argues fine-grained localization reduces overfitting.

### DEAR

- **DEAR: A Novel Deep Learning-based Approach for Automated Program Repair**
- arXiv: `2205.01859`
- Direct collision:
  - SBFL + data flow + tree context;
  - multi-statement and multi-hunk repair.

### AutoCodeRover

- **AutoCodeRover: Autonomous Program Improvement**
- arXiv: `2404.05427`
- Relevance:
  - AST-structured repository search;
  - optional SBFL;
  - LLM patching.

### CodePilot

- **Monte Carlo Tree Search for Execution-Guided Program Repair with Large
  Language Models**
- arXiv: `2602.00129`
- Direct collision:
  - hierarchical localization;
  - execution-guided repair search;
  - confidence-guided refinement.

### SHERLOC

- **SHERLOC: Structured Diagnostic Localization for Code Repair Agents**
- arXiv: `2606.24820`
- Relevance:
  - structured localization and diagnostics;
  - localization improves downstream repair.

### Loc2Repair

- **Loc2Repair: A Framework for Evaluating the Impact of File-Level Issue
  Localization in Repo-Level LLM Repair**
- arXiv: `2606.30963`
- Methodological collision:
  - explicitly decomposes localization and repair;
  - measures gold-localization headroom.
- Proposal implication:
  - our oracle operator gate follows the same decomposition principle at
    subtree level.

### SiblingRepair and MultiFixer

- **SiblingRepair: Sibling-Based Multi-Hunk Repair with Large Language Models**
  — arXiv `2605.06209`.
- **MultiFixer: A Coordinator-Proposer Based Multi-Agent Framework For Fixing
  Multi-Hunk Bugs** — arXiv `2607.26591`.
- Relevance:
  - simultaneous/iterative dependent multi-hunk repair;
  - mandatory future baselines for the multi-site extension.

---

## 6. Editing benchmarks and preservation

### HumanEvalPack / OctoPack

- **OctoPack: Instruction Tuning Code Large Language Models**
- arXiv: `2308.07124`
- Venue record: ICLR.
- Relevance:
  - introduces HumanEvalPack;
  - repair, explanation, and synthesis across six languages.

### CanItEdit

- **Can It Edit? Evaluating the Ability of Large Language Models to Follow Code
  Editing Instructions**
- arXiv: `2312.12450`
- Relevance:
  - instruction-driven code editing;
  - curated tests and training data.

### Editing benchmark audit

- **Edit, But Verify: An Empirical Audit of Instructed Code-Editing Benchmarks**
- arXiv: `2604.05100`
- Relevance:
  - CanItEdit/EDIT-Bench scope and test-coverage limitations;
  - prevents overclaiming deployment generality.

### Copy-as-Decode

- **Copy-as-Decode: Grammar-Constrained Parallel Prefill for LLM Editing**
- arXiv: `2604.18170`
- Direct collision:
  - preservation/copy efficiency;
  - HumanEvalPack-Fix;
  - exact resolver round trip.
- Proposal implication:
  - AR editing can exploit copying very efficiently;
  - outside-region exactness and real wall time are mandatory.

### PAIR-Bench

- **Benchmarking Code Improvement with Progressive, Adaptive, and Interactive
  Feedback**
- arXiv: `2607.01360`
- Relevance:
  - targeted repair;
  - preservation of already-correct behavior;
  - progressive feedback.

### Deletion avoidance

- **To Add Is Machine, To Delete Is Human: Measuring and Mitigating Deletion
  Avoidance in LLM Code Editing**
- arXiv: `2607.28887`
- Relevance:
  - patch intent can differ even when tests pass;
  - deletion precision/recall should be reported.

---

## 7. Evaluation and grammar constraints

### EvalPlus

- **Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of
  Large Language Models for Code Generation**
- arXiv: `2305.01210`
- Venue: NeurIPS.
- Relevance:
  - stronger hidden tests;
  - test insufficiency can change rankings.

### CFG-constrained diffusion

- **Constrained Decoding of Diffusion LLMs with Context-Free Grammars**
- arXiv: `2508.10111`.
- **EPIC: Efficient and Parallel Inference under CFG Constraints for Diffusion
  Language Models**
- arXiv: `2606.00722`.
- Boundary:
  - syntactic validity, not execution-grounded semantic repair.

---

## 8. dLLM evaluation protocol

- **CaRE Compute-aware Remasking Evaluation Protocol for Masked Diffusion
  Language Models**, arXiv `2607.24763`.
- **Diffusion Language Models: An Experimental Analysis**,
  arXiv `2606.19475`.

Proposal implication:

- sampler configuration must be frozen before method comparison;
- NFE is not sufficient for cross-family cost claims;
- report a small sensitivity envelope separately.

---

## 9. Direct topic collision with caution

- **Exploring the Power of Diffusion Large Language Models for Software
  Engineering: An Empirical Investigation**
- arXiv: `2510.04605`.
- Scope:
  - code generation, defect detection, program repair.
- Caution:
  - local audit identified weak/mismatched AR controls and citation-quality
    problems;
  - treat as a direct topic collision, not as reliable evidence that dLLMs
    dominate AR repair.

---

## 10. Claims explicitly forbidden by this audit

Do not claim:

- first dLLM program repair;
- first remasking/refinement;
- first AST/tree repair;
- first execution-guided repair;
- first preservation-aware code editing;
- first fault-localization/repair decomposition;
- general superiority over AR;
- efficiency based on NFE alone.

The proposed gap is only:

> execution-grounded, typed-subtree, mask-native dLLM repair with exact
> outside-region preservation and strong matched repair baselines.

