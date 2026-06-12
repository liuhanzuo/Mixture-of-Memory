# legacy/

Files here are kept for historical reference but are **NOT** imported by the
active codebase. They correspond to research directions that have been
explicitly abandoned (per CLAUDE.md "已完成的工作").

**Rule**: nothing in `legacy/` should be referenced by `src/`, `scripts/`
(non-legacy), or `tests/`. If you need to revive a direction, move the files
back to their original location with `git mv`.

## Contents

### `legacy/memory/sparse_memory/` — Sparse Memory (MAG-EMA), abandoned
- EMA-based memory bank with 128/256 slots
- Attention path: `attention.py`, `memory_bank.py`, `model.py`
- Result: PPL degradation ~20%, abandoned (CLAUDE.md)
- Last commit: 2026-04-30 (snapshot before cross-attention pivot)

### `legacy/memory/sparse/` — Sparse Memory v3 (per-chunk + gated fusion), abandoned
- Same family as `sparse_memory/`, later iteration with per-chunk retrieval
- `SparseMemoryModel` exported from `__init__.py`
- Last commit: 2026-04-30 (same snapshot)

### `legacy/memory/dms/` — Dynamic Memory Sparsification, abandoned
- 8x compression via per-token decision head
- Code: `dms_attention.py`, `dms_decision_head.py`, `dms_training.py`
- Checkpoint at `outputs/dms_8x/final` is "待评估" — DO NOT remove that
- Last commit: 2026-04-23

### `legacy/memory/rmt/` — Recurrent Memory Transformer v3-v10, abandoned
- Memory tokens prepended/appended to segments
- Generation degradation (repetition pattern), abandoned (CLAUDE.md)
- `rmt_module.py` (RMTModel/RMTMemory) and `rmt_v10.py` (RMTv10 variant)
- **Note**: `src/memory/rmt_slot/` (RMT-Slot hybrid) is **active** and remains in `src/`. Do not confuse with this folder.

### `legacy/memory/selective_context.py` — Selective Context (token pruning), abandoned
- Result: PPL degradation 500-5000%, abandoned (CLAUDE.md)
- Last commit: 2026-04-22

### `legacy/memory/qfilters/` — Q-Filters (arXiv:2503.02812), abandoned 2026-05-10
- Last commit 2026-04-30. Closure documented in `ops/research_notes/20260426_s11_retraction.md`.
- 2026-05-10: user explicitly authorized abandonment ("Q-Filters checklist 不能停" rule deprecated).
- Associated scripts in `legacy/scripts/`: `eval_qfilters.py`, `eval_qfilters_streaming.py`, `_issue110_smoke_calibration.py`

### `legacy/scripts/` — training/eval/debug scripts for the abandoned directions
- `train_sparse_memory.py`, `train_gated_sparse_memory.py`, `eval_sparse_memory_ppl.py`, `eval_nih_extended_sparse.py`, `eval_phase1_gate.py`, `test_sparse_memory_import.py`, `diag_memory_write.py`, `quick_diag.py`
- `train_mag.py`, `eval_mag.py`, `eval_mac.py`, `train_mac.py`, `benchmark_mac.py`, `demo_compare.py`, `demo_single.py`
- `eval_selective_context.py`, `test_selective_context.py`, `run_sc_eval.py`
- `train_dms.py`, `eval_dms.py`
- `train_rmt*.py` (v3-v10, original, pg19), `eval_rmt*.py`, `eval_ppl_v2/v3.py`, `quick_ppl.py`, `eval_needle_haystack*.py`, `eval_nih.py`, `test_rmt_inference_debug.py`, `debug_rmt_*.py`, `debug_eval_rmt_v10.py`, `debug_eval_memory.py`, `diag_v8_mask_bug.py`
- shell wrappers: `launch_node{1,2,3}.sh`, `launch_slp.sh`, `launch_v5_bg.sh`, `launch_v8.sh`, `launch_dms_train.sh`, `run_train_{mag,mac,rmt,sparse_memory,selective_write,causal_bce,concat_fusion,multi_node,squad,recall,pg19,su,su_full,su_twostage,v3,v4*,v5*,v6*,v7,v9,v10}*.sh`, `run_rmt*.sh`, `run_debug*.sh`, `_launch_v5*.sh`, `_launch_concat_fusion_5k*.sh`, `_launch_phase1_v6.sh`, `_run_phase1_*.sh`, `_run_concat_5k.sh`, `_smoke_test.sh`

### `legacy/tests/` — unit tests for abandoned modules
- `test_arch_a_fusion.py`, `test_sparse_memory_smoke.py` (sparse_memory)
- `test_compressed_memory.py`, `test_e2e_compressed_memory.py` (MAG compressed memory)
- `test_dms_qwen3.py` (DMS)

### `legacy/launch_top/` — top-level launcher shells calling legacy scripts
- `run_eval.sh`, `run_eval_v9.sh`, `run_eval_fixed.sh` → `eval_rmt.py` / `eval_nih_extended.py`
- `run_locomo_v4*.sh` → `eval_rmt_locomo.py`
- `run_nih_smoke.sh`, `run_nih_zh.sh`, `run_v8_nih.sh`, `run_nih_extended*.sh` → eval_nih*.py / eval_rmt.py / eval_needle_haystack*.py
- `run_ppl_v4.sh`, `run_ppl_v10.sh` → quick_ppl.py / eval_rmt.py
- `run_32k_eval.sh` → eval_nih_extended_sparse.py
- `launch_2048_v4.sh` → run_train_sparse_memory.sh

## What was NOT moved (still active in `src/`)

- `src/memory/mag/` — referenced by `src/memory/scheduler.py` (lazy-import) and `src/backbone/swa_model.py`. Removing it would require refactoring the scheduler/agent stack. Marked as **MEDIUM confidence** in `CODE_CLEANUP_SUGGESTIONS.md` — see that file for follow-up.
- `src/memory/qfilters/` — ~~used by `scripts/eval_qfilters*.py` and `scripts/_issue110_smoke_calibration.py` (last touched 2026-04-30). Left in place.~~ **2026-05-10: moved to `legacy/memory/qfilters/` per user directive — Q-Filters direction officially abandoned**
- `src/memory/{l1,l2,l3,scheduler.py,state.py}` and `src/agents/`, `src/training/` — three-tier memory stack from the original MoM design. Quiet since 2026-03-22 but tested in `tests/test_l{1,2,3}.py` and `tests/test_scheduler.py`. Marked **MEDIUM confidence** in `CODE_CLEANUP_SUGGESTIONS.md`.
- `src/memory/slot/`, `src/memory/slot_memory/`, `src/memory/rmt_slot/`, `src/memory/mem_space/` — **active** (cross-attention H-series, slot infrastructure, RMT-Slot).

## Archived 2026-06-12 — repo cleanup (abandoned/superseded directions)

This batch corresponds to abandoned/superseded directions (RMT, sparse-memory,
Q-Filters, h-series dual-gate, SWA, P2/P8/P11 pre-capacity-sweep, routeA,
attention-matching). Kept for git history/reference; **NOT imported by live
code**. Active work is the `mem_space` capacity sweep (`expR1c*`) — see the
top-level `README.md`.

- `legacy/research_notes_pre0501/` — `ops/research_notes/*.md` with commit date
  ≤ 2026-04-30 (RMT/sparse/Q-Filters/mem_space-v0 era). Notes from 2026-05-11
  onward stay in `ops/research_notes/`.
- `legacy/launchers/` — dead-direction `scripts/launch_*.sh`: h-series
  (h2–h14), phase1b/phase8b, SWA, routeA, P2/P8 (pre-capacity-sweep), beacon,
  mem_scale, wbmode, hmt, rmt_slot, infini, niah_v4, v4_full_sft, exp1–exp5 L3.
- `legacy/eval_sched_dead/` — dead eval schedulers for exp2/W1/routeA/d6/diskB
  (NOT any `expR1c*` scheduler, which stay live in `scripts/`).
- `legacy/plans_superseded/` — superseded planning/design docs from `status/`
  and the repo root (attention-matching, H/V2, RMT-slot, V4 feasibility, eval
  design, locomo setup, cluster/command references, etc.).
- `legacy/versions_pre_v8/` — architecture version docs v2–v7 (v8–v20 stay in
  `versions/`).
- `legacy/root_oneoff_scripts/` — loose root-level eval/debug/test scripts
  (`run_eval_*.sh`, `run_nih_*.sh`, `eval_ruler_baseline.py`,
  `heartbeat_monitor.py`, `test_*.py`, etc.).
- `legacy/scripts_misfiled_docs/` — markdown docs that were living under
  `scripts/` (FULL_SFT_PLAN, multisegment summaries, LOCOMO/nih READMEs).

## Restoring a direction

```bash
# Example: bring sparse_memory back
git mv legacy/memory/sparse_memory src/memory/sparse_memory
git mv legacy/scripts/train_sparse_memory.py scripts/
git commit -m "revive: restore sparse_memory direction"
```
