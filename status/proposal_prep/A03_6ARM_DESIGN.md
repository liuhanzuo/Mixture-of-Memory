# A03 6-Arm Main Study Design
# Prepared: 2026-08-09 (read-only scout; no GPU job launched, no file edited)

---

## 0. Current state summary

Gate-1 (floor certification) is **CLEARED** on four closed-book parametric-knowledge axes:
- MMLU-content (letter interface retired at 1B; content-norm only)
- PopQA EM (+ length-matched contains)
- TriviaQA EM (primary)
- NQ-open EM (gate cleared 2026-08-09, this session)

All four axes have per-example jsonl files on zwfy6 for three arms:
- `A03_1B_base` (intact 36→16L OLMo-2-0425-1B)
- `A03_1B_keep7_step200k` (pruned keep7+fresh2, 9L, fully healed @200k)
- `A03_1B_keep7_step500` (barely-healed control)

Data at:
- `/apdcephfs_zwfy6/.../olmo2_closedbook_results/A03_1B_{base,keep7_step200k,keep7_step500}/` (popqa + triviaqa)
- `/apdcephfs_zwfy6/.../olmo2_closedbook_results/A03_1B_{base,keep7_step200k,keep7_step500}_nq/` (nq_open)
- `/apdcephfs_zwfy6/.../olmo2_mmlu_content_results/A03_1B_{base,keep7_step200k,keep7_step500}/` (mmlu)

---

## 1. Six arms: concrete implementation

### Arm 1 — intact
**Definition**: OLMo-2-0425-1B base (no pruning, no additional training).
**Model**: `../models/OLMo-2-0425-1B` (5.6 GB, on BOTH disks — verified on zwfy6)
**eval_olmo2_closedbook_qa.py**: run with no `--ckpt` flag
**Status**: LAUNCH-READY. All four axes already evaluated (see §0 above).

### Arm 2 — pruned+heal
**Definition**: OLMo-2-0425-1B, front 7 layers kept + 2 fresh NTP layers, healed on
Dolmino for 200k steps.
**Checkpoint**: `/apdcephfs_zwfy6/.../outputs/olmo2_probe2_1B_keep7fresh2_16card/step200000.pt`
- Size: `12181310078` bytes (≈11.3 GB) — verified on .82 2026-08-09
- Arch: `keep_front=7, n_fresh=2 → 9 layers total`, `hidden_size=2048`, `vocab=100352`
- Training: 200k steps on Dolmino, `lr_fresh=1e-4, lr_inherited=2e-5`, `seq_len=2048`
**eval_olmo2_closedbook_qa.py**: `--ckpt outputs/olmo2_probe2_1B_keep7fresh2_16card/step200000.pt`
**Status**: LAUNCH-READY. All four axes already evaluated (see §0 above).

Note: also available as `outputs/olmo2_probe2_1B_keep7fresh2/step500.pt` (barely-healed
control; evaluated but not an arm in the 6-arm main study — it is the calibration reference
to confirm "at floor" is detectable).

### Arm 3 — pruned+heal+CPT (continued pretraining on targeted corpus)

**Design intent (A03 PROPOSAL.md)**: "pruned + CPT" — continue pretraining the
pruned+healed model to restore distributional fit and potentially parametric knowledge.

**PARTIALLY UNRESOLVED** — three sub-questions:
1. Which corpus? Per PROPOSAL.md, CPT should target recovery of "old parametric knowledge"
   (PopQA/TriviaQA/NQ-open entities). Candidates:
   - **Dolmino (already used)**: already the heal corpus; additional steps test "does
     more Dolmino help?" but may not specifically target factual entities.
   - **Wikipedia text subset**: would directly target entity knowledge but is not on disk
     and requires download (~20 GB via proxy).
   - **A "knowledge-dense" Dolmino continuation**: subsampling Dolmino segments that
     overlap with PopQA/TriviaQA entities (requires a new data pipeline script).
   **Recommended (unsettled)**: run additional 10k-50k steps on Dolmino as the cheapest
   option, then compare. Wikipedia download is feasible with proxy if Dolmino doesn't move
   the needle.

2. How many steps? The heal recipe ran 200k steps. CPT top-up is typically 5k-50k steps
   for knowledge recovery. Suggest **20k steps** as a starting point (0.5h-2h on H20),
   sweep to 50k if first checkpoint shows signal.

3. New ckpt needed (no existing Arm 3 checkpoint on either disk):
   - Script: `scripts/train_olmo2_arch_probe2.py` (same as heal, supports OLMo-2-1B via
     `--model_path ../models/OLMo-2-0425-1B`)
   - Data: `data/dolmino_now15b.npy` (on zwfy6: `/apdcephfs_zwfy6/.../data/dolmino_now15b.npy`,
     126.9 GB — verified on .82)
   - Launch: `--resume_from outputs/olmo2_probe2_1B_keep7fresh2_16card/step200000.pt
     --keep_front_layers 7 --n_fresh_layers 2 --max_steps 20000 --lr 2e-5`
   - Output dir: `outputs/olmo2_probe2_1B_keep7f2_dolmino_cpt20k/`

**Status**: NEEDS_CODE (design unresolved) + NEEDS_GPU (no checkpoint exists).
Wall for 20k steps at 7.3 win/s on 8×H20: 20000×512/(7.3×8) ≈ **34 min** (estimated;
the 7.3 win/s is from 8B distill, 1B should be faster).

### Arm 4 — pruned+heal+raw-text RAG

**Design intent**: provide retrieved Wikipedia/knowledge-base text at inference time
(open-book), using the same PopQA/TriviaQA/NQ-open questions but with retrieved gold
passages injected into the prompt.

**BLOCKED — new eval harness required**:
The existing `eval_olmo2_closedbook_qa.py` is STRICT zero-shot, no retrieval. Adding
RAG requires:
1. A retrieval index: for PopQA/TriviaQA/NQ-open, the standard oracle is to provide the
   gold Wikipedia passage (usually 100-word snippet) in the prompt. For a deployed RAG,
   a BM25 or dense retriever over a Wikipedia snapshot would be used.
2. A new eval variant: the harness must inject `"Context: {retrieved_passage}\n\nQuestion: {q}\nAnswer:"` before decoding.
3. Retrieval data: a Wikipedia passage dump is NOT on either disk. Gold passages for PopQA
   are available if the original PopQA dataset includes them (the `akariasai___pop_qa` HF
   cache on zwfy6 may include supporting passages — unverified).

No RAG eval script for OLMo-2 closed-book QA exists in the codebase. The closest is
`eval_p1_9_dense_rag.py` and `eval_p0_20_equal_latency.py`, but both are for the
QCMem/Qwen3-8B pipeline and do not apply to OLMo-2-1B directly.

**Status**: BLOCKED — needs ~1-2 days of coder work to build a retrieval-augmented
variant of `eval_olmo2_closedbook_qa.py`. The retrieval index design (BM25 vs oracle gold
passage vs dense) needs to be decided before coding.

**Minimum viable version**: Oracle RAG (use gold supporting passage from PopQA dataset if
available). This tests the best-case RAG bound without a retriever. Check the `akariasai___pop_qa`
HF cache for supporting passage fields before building a full retrieval pipeline.

### Arm 5 — pruned+heal+residual memory (CoMem-style)

**Design intent**: attach a reusable residual memory (CoMem/QCMem-style read-LoRA) to the
pruned OLMo-2-1B model, providing the retrieved context as KV-cached residual states
rather than raw text injection.

**BLOCKED — significant training required**:
The canonical CoMem Read-LoRA (`outputs/qcmem_distill_qwen_j12_r32_4k/final`) is trained
on **Qwen3-8B**, layers 12..35 of a 36-layer model. It is NOT applicable to OLMo-2-1B
(9-layer model, different architecture, different hidden_size 2048 vs Qwen3-8B 4096).

To build a CoMem arm for OLMo-2-1B:
1. Train a QCMem-style Read-LoRA on OLMo-2-1B using `scripts/train_qcmem_distill.py` or
   a 1B-adapted version. The distillation target should be the full-depth OLMo-2-1B
   teacher (intact, Arm 1).
2. Choose a resume_j for 1B: the 1B has 9 layers total; `resume_j=4` (half the layers)
   would be analogous to j=12/36 on Qwen3-8B.
3. Training cost: analogous to the Qwen3-8B distillation (4000 steps at ~7 win/s on H20,
   ~40-80 min). But this has NEVER been done for OLMo-2-1B.
4. The QCMem write/read pipeline in `src/memory/qcmem/qcmem_model.py` may need adaptation
   for the OLMo-2 Rope/architecture differences.

This arm is the most complex: it requires (a) designing the 1B-scale QCMem recipe,
(b) training the LoRA, (c) adapting the eval harness.

**Status**: BLOCKED — needs 2-4 days of research engineering. Cannot be built tonight.

**Alternative**: Use the existing `eval_qcmem_babilong.py` / `eval_qcmem_longeval.py`
harnesses with Qwen3-8B as the "residual memory" arm comparison. This would NOT be
"pruned+heal+memory" but rather "Qwen3-8B+memory vs OLMo-2-1B+RAG", which loses the
same-model comparison the PROPOSAL requires.

### Arm 6 — pruned+heal+CPT+memory (joint)

Arm 6 = Arm 3 + Arm 5. **Blocked** until both Arm 3 and Arm 5 are unblocked.

---

## 2. The four certified outcome axes

Based on `GATE_NQOPEN_VERDICT.md` and `evidence/a03_1b_floor_nulls.json`:

| Axis | Metric | n | Floor (construct-appropriate null) | Notes |
|------|--------|---|----------------------------------|-------|
| **MMLU-content** | content_norm (letter-interface RETIRED at 1B) | 14042 | longest-option split-tie = 0.2845 | Letter collapses to always-one-letter for damaged 1B arms |
| **PopQA EM** | EM (primary) + length-matched contains (secondary) | 14267 | best-constant EM ≈ 0 (rare answer match); contains = length-matched null varies by arm | EM is safer; length-matched contains null differs per arm (~6x prediction length inflation in damaged arms) |
| **TriviaQA EM** | EM | 17944 | majority-answer EM = 0.0018 (verified from evidence json) | Primary axis with highest floor-free dynamic range |
| **NQ-open EM** | EM | 3610 | majority-answer EM = 0.0053 | Verified: intact 0.1025, pruned+healed 0.0285, both above floor |

**MMLU letter interface is BANNED at 1B**: intact 1B letter 0.3807 vs content 0.3807
(nearly tied); barely-healed keep7_step500 MMLU emits always-A on 14042/14042 items.
Any paper table must use content_norm only.

**Length-matched contains null** (CRITICAL — see `analyze_1b_knowledge_floor.py` lines
482-494): The arms differ ~6x in mean prediction length (intact ~3 chars, healed ~18 chars,
barely-healed ~1 char). The `contains` metric is length-sensitive because a longer
prediction has more characters that might incidentally contain the answer. A naive best-
constant `contains` null is not arm-matched. The analyzer computes a
`lengthmatched_contains_null` per arm (finds a null string of matching average length from
the gold answer distribution). This null is MANDATORY for any arm comparison on `contains`.

---

## 3. `analyze_1b_knowledge_floor.py` patch for NQ-open

Current code (lines 427-428):
```python
    for task, expected_n, headline in (("popqa", 14267, "contains"),
                                       ("triviaqa", 17944, "em")):
```

Required patch — add `nq_open` to the same loop:
```python
    for task, expected_n, headline in (("popqa", 14267, "contains"),
                                       ("triviaqa", 17944, "em"),
                                       ("nq_open", 3610, "em")):
```

Note: the `per_example_nq_open.jsonl` files follow the same format as
`per_example_popqa.jsonl` and `per_example_triviaqa.jsonl` (confirmed from the eval
harness `eval_olmo2_closedbook_qa.py` — it writes the same dict fields for all tasks).
The `item_id` field, `gold` field, `pred` field, `em` field, and `contains` field are
all present (this is asserted by the harness before writing).

The existing `load_cb_arm`, `score_prediction`, `best_constant_qa`, and
`lengthmatched_contains_null` functions all accept any task string and expected_n — no
other code changes required beyond adding the tuple.

This is a CPU-only change (~1 line) that can be applied in a few minutes.

**Additional print statement to add** (lines 523-527 show the task loop for printing):
After the popqa/triviaqa print block, add:
```python
    if "nq_open" in diag:
        print(f"nq_open nulls: best-constant em='{diag['nq_open']['em']['best_constant']}' "
              f"{diag['nq_open']['em']['acc']:.4f} | contains="
              f"'{diag['nq_open']['contains']['best_constant']}' "
              f"{diag['nq_open']['contains']['acc']:.4f}")
```

**Where the output per_example files live** (for the analyzer's `--cb_root` arg):
The A03 1B arms have their NQ-open data at:
- `olmo2_closedbook_results/A03_1B_base_nq/per_example_nq_open.jsonl` (zwfy6, verified)
- `olmo2_closedbook_results/A03_1B_keep7_step200k_nq/per_example_nq_open.jsonl` (zwfy6, verified)
- `olmo2_closedbook_results/A03_1B_keep7_step500_nq/per_example_nq_open.jsonl` (zwfy6, verified)

BUT: the analyzer currently uses `--cb_root` pointing to a directory where arm dirs live.
The NQ-open runs used a DIFFERENT output_name pattern (`A03_1B_base_nq`, not `A03_1B_base`).
The analyzer's `arms` list must therefore be updated to point to the `_nq`-suffixed
directories for the NQ-open task, OR the NQ-open per_example files must be moved/linked
into the main arm directories.

**Simplest fix**: copy (or symlink) the NQ-open per_example jsonl files into the main
arm directories:
```bash
# Run on .82 (zwfy6):
for name in A03_1B_base A03_1B_keep7_step200k A03_1B_keep7_step500; do
  cp olmo2_closedbook_results/${name}_nq/per_example_nq_open.jsonl \
     olmo2_closedbook_results/${name}/
done
```
Then the single `--cb_root olmo2_closedbook_results` arg picks up all four tasks for
each arm.

---

## 4. Launch-ready vs blocked arms

| Arm | Status | Blocker | Tonight? |
|-----|--------|---------|---------|
| 1. intact | **LAUNCH-READY** | None (all 4 axes done) | Yes — already evaluated |
| 2. pruned+heal | **LAUNCH-READY** | None (all 4 axes done) | Yes — already evaluated |
| 3. +CPT | **NEEDS_GPU** | Design partly unresolved; no ckpt exists; ~30 min train | Next session (after design settled) |
| 4. +raw-text RAG | **NEEDS_CODE** | No RAG harness for OLMo-2 closed-book QA | 1-2 days |
| 5. +residual memory | **BLOCKED** | No OLMo-2-1B QCMem LoRA exists; needs ~1 day train + arch work | 3-4 days |
| 6. +CPT+memory | **BLOCKED** | Depends on Arm 3 + Arm 5 | After both unblocked |

**What is launch-ready tonight** (CPU-only work + data already on disk):
1. Apply the 1-line nq_open patch to `analyze_1b_knowledge_floor.py`
2. Copy per_example_nq_open.jsonl into main arm directories (symlink or cp, on .82)
3. Run `python analyze_1b_knowledge_floor.py` with all 4 axes on 3 existing arms

This produces the floor-calibrated table for the two already-healed arms on all four
axes, which is the core A03 contribution.

---

## 5. Launch plan for runnable arms (delta from existing)

The existing `_run_a03_axes_floor_82.sh` already ran NQ-open for all 3 arms.
What is NOT yet done:

**Delta 1 (CPU, ~15 min)**: Apply nq_open patch to `analyze_1b_knowledge_floor.py`
and run it to get BH-corrected floor-calibrated residuals for all 4 axes × 3 arms.

Command (on .82, after applying patch and symlinking nq files):
```bash
/opt/conda/envs/torch-base/bin/python \
  proposal/archive/A03-parametric-vs-external-memory/code/analyze_1b_knowledge_floor.py \
  --mmlu_root olmo2_mmlu_content_results \
  --cb_root olmo2_closedbook_results \
  --arms "A03_1B_base:A03_1B_base" \
         "pruned_healed_200k:A03_1B_keep7_step200k" \
         "barely_healed_step500:A03_1B_keep7_step500" \
  --out_json evidence/a03_4axis_floor_analysis.json \
  --n_boot 10000
```
(Check exact CLI of `analyze_1b_knowledge_floor.py` before running — the `--arms` format
may differ; the script uses `argparse` and its exact argument names should be verified
from the script's `main()` function.)

**Delta 2 (GPU, ~5-15 min, Arm 3 pilot)**: Launch the +CPT arm with 20k Dolmino steps.
Use `_run_olmo2_full32_dolmino_heal.sh` as a template, swapping model_path to 1B and
resuming from keep7_step200k. Wall ~30-60 min on 8×H20.

No new driver script needed for Arm 3 — use `train_olmo2_arch_probe2.py` directly
with:
```bash
python -m torch.distributed.run --standalone --nproc_per_node 8 \
  scripts/train_olmo2_arch_probe2.py \
  --model_path ../models/OLMo-2-0425-1B \
  --resume_from outputs/olmo2_probe2_1B_keep7fresh2_16card/step200000.pt \
  --keep_front_layers 7 --n_fresh_layers 2 \
  --data_path data/dolmino_now15b.npy \
  --output_dir outputs/olmo2_probe2_1B_keep7f2_dolmino_cpt20k \
  --max_steps 20000 --lr 2e-5 --lr_inherited 2e-5 --lr_fresh 2e-5 \
  --seq_len 2048 --batch_size 8 --grad_accumulation_steps 1 \
  --warmup_steps 100
```
Note: `--batch_size 8` on H20 (97.8 GB) with 1B model should be safe; verify with
`nvidia-smi` during first 5 steps.

---

## 6. Key protocol notes

**Same-evidence requirement**: All 6 arms must be evaluated on the SAME question set.
The closed-book harness uses fixed HF dataset splits (popqa default test, triviaqa
rc.nocontext validation, nq_open validation) — the item_id field ensures alignment.
The `assert [r["item_id"]...] == ref_ids_t` check in the analyzer enforces this.

**Null recalculation per arm** (MANDATORY): The PROPOSAL's "关键控制" requires that
each arm is compared against ITS OWN construct-appropriate null (best-constant or
majority-answer). The `analyze_1b_knowledge_floor.py` does this correctly — it
recomputes the null from the gold distribution, not from the intact arm's null.
This is non-trivial for contains because the length-matched null is arm-specific.

**contains length-null**: The barely-healed arm (step500) predicts near-empty text
(mean_pred_chars ≈ 1-2), while intact predicts longer answers (mean ~10-15 chars).
Do NOT compare "contains" across arms without the length-matched null — a short
prediction almost never contains the answer by chance, while a long prediction
sometimes does. The current `analyze_1b_knowledge_floor.py` handles this correctly
for popqa/triviaqa; it will also handle it for nq_open after the patch.

**MMLU-letter is BANNED at 1B**: The intact 1B letter acc 0.3807 and content_norm 0.3807
are numerically similar; the floor-certified keep7_step200k letter is below floor while
content is ABOVE floor (+3.99pp, BH-significant per STATUS.json). Any paper table must
use content_norm for MMLU comparisons.

---

## Summary

**Tonight (CPU only)**:
- Apply 1-line nq_open patch to `analyze_1b_knowledge_floor.py`
- Symlink per_example_nq_open.jsonl files into main arm directories on .82
- Run analyzer to get floor-calibrated residuals for 3 arms × 4 axes
- Result: the A03 floor-certification table is COMPLETE for Arms 1 and 2

**Next session (if Arm 3 CPT design is settled, ~30 min GPU)**:
- Launch 20k Dolmino CPT from keep7_step200k checkpoint (on .82 or .73, zwfy6)
- Eval new checkpoint on all 4 axes using existing `_run_a03_1b_floor_82.sh`-style driver

**Blocked for foreseeable future (needs coder + research time)**:
- Arm 4: RAG harness for OLMo-2-1B closed-book QA
- Arm 5: QCMem LoRA training for OLMo-2-1B
- Arm 6: Combination of 4 + 5
