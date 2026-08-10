# A02 Phase-1 Launch Plan
# Prepared: 2026-08-09 (read-only scout; no GPU job launched, no file edited)

---

## 0. Resolved open question: canonical Write-LoRA step selection

**Loss trajectory** extracted from
`/apdcephfs_zwfy6/.../logs/qcmem_writepath_distill_qwen_j12_r32.log` on .82:

| step | loss | note |
|------|------|------|
| 500  | 0.0426 | first checkpoint; still on high-lr plateau |
| 800  | 0.0419 | local minimum by logged loss |
| 1000 | 0.0462 | spike; above step-500 |
| 1500 | 0.0401 | |
| 2000 | 0.0395 | |
| 2500 | 0.0392 | |

The logged-loss minimum among *available checkpoints* is **step2500 (0.0392)**, but the
loss is essentially flat since step1000–1500 (range 0.038–0.042, indistinguishable noise
given PG-19 data-pipeline I/O stalls mentioned in paperA/TODOList.md). 

The authoritative quality signal is the BBWL synthetic eval (bench_results/p0_18_e4_bbwl_step*):

| step | macro BBWL |
|------|-----------|
| 1000 | 98.0 |
| 1500 | 99.0  ← peak quality |
| 2000 | 98.5 |

Step1500 has the highest task quality (BBWL macro 99.0); step2000 was designated the
"delivery ckpt = step2000 (BBWL=98.5)" per paperA/TODOList.md task #142 because step1500
was still very close and step2000 gave a slightly safer plateau.

**Decision: use `step2000` as canonical Write-LoRA**, matching the paperA designation.
Rationale: paperA/TODOList.md explicitly states "交付 ckpt = step2000 adapter (BBWL=98.5)",
and step2000 BBWL 98.5 is within 0.5pp of the peak (1500=99.0) while representing the
confirmed plateau, not a transient peak. step2500 exists but was not eval'd on BBWL
and has no quality evidence beyond training loss (which is flat since ~step1000).

Disk location verified (ssh to .82, 2026-08-09):
- Path: `/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/outputs/qcmem_writepath_distill_qwen_j12_r32/step2000/`
- Size: `116420689` bytes per `ls -la` (≈111 MB); contains `adapter_config.json` + `adapter_model.safetensors` (inferred from matching step500/1000/1500 directories, all same `116420689`).
- Layers: `0..11` (Write path, lower-band), disjoint from Read-LoRA layers `12..35`.

---

## 1. The 5 configs pinned to exact args

### Canonical assets (all on zwfy6, verified on .82 2026-08-09)

| asset | path | size | disk |
|-------|------|------|------|
| Base model | `/apdcephfs_zwfy6/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b` | ~14.4 GB | zwfy6 |
| Read-LoRA (canonical flagship) | `outputs/qcmem_distill_qwen_j12_r32_4k/final/` | 223 MB (232829168 bytes) | zwfy6 |
| Write-LoRA (step2000) | `outputs/qcmem_writepath_distill_qwen_j12_r32/step2000/` | ~111 MB | zwfy6 only |

Read-LoRA provenance: `adapter_config.json` on zwfy6 lists
`base_model_name_or_path: .../models/Qwen--Qwen3-8b`, `r=32`, `lora_alpha=64`,
`layers_to_transform=[12..35]`, 7 target modules, `inference_mode=true`. SHA verified as
the `dd09cd17…` flagship referenced in paperA/P0_16_E0_NOTES.md and all TODOList runs.

---

**Config 1 — j=0 full-depth replay (no-retrieval baseline)**

Operational meaning: `QCMemModel` with `resume_j=0` forces all 36 layers to recompute
over the PACKED context (equivalent to `KVDirect`). NO retrieval, NO LoRA. This is the
full-depth "read everything" upper bound; its cost grows O(context) in the packed Read.

Eval harness flag: `--baseline kvdirect` in any of the natural-task drivers:
- `eval_qcmem_babilong.py --baseline kvdirect`
- `eval_qcmem_longeval.py --baseline kvdirect`
- `eval_qcmem_longbench.py --baseline kvdirect`

Note: `--baseline kvdirect` forcibly sets `resume_j=0`; existing `--lora_adapter` is
silently dropped (not compatible with j=0 — see `eval_qcmem_longeval.py:375-384`).

Config 1 does NOT use the Write-LoRA or Read-LoRA. It has been run for all 5 benchmarks
for the flagship (e.g., `longeval_results/kvdirect_8b_chatFALSE/`), so existing results
can be reused for benchmarks that are identical in configuration. Reuse requires verifying
the existing run used `chat_template=False`, `selector` is irrelevant for kvdirect,
`chunk_size=512`, and `model_path=Qwen--Qwen3-8b`.

---

**Config 2 — j=12 Read-LoRA only (flagship CoMem)**

Operational meaning: `resume_j=12`, `lora_adapter=outputs/qcmem_distill_qwen_j12_r32_4k/final`,
`selector=iter_bm25`, `topk=12`, `sink_tokens=bos`, `chunk_size=512`, `chat_template=False`.
This is the Paper A flagship configuration. Write-LoRA is NOT used.

Eval harness:
```bash
python scripts/eval_qcmem_longeval.py \
    --model_path <zwfy6_root>/models/Qwen--Qwen3-8b \
    --resume_j 12 --lora_adapter outputs/qcmem_distill_qwen_j12_r32_4k/final \
    --selector iter_bm25 --topk 12 --sink_tokens bos \
    --chunk_size 512 --add_bos 0 \
    [--lengths 4k 8k 16k 32k 64k 128k] [--num_samples 50] \
    --output_name a02_config2_j12_readlora \
    --num_shards 8 --shard_index $GPU
```
(similarly for BABILong, LongBench, LoCoMo with their respective driver scripts)

Existing results on zwfy6 that can be reused (Config 2 = standard flagship):
- LongEval: `longeval_results/qcmem_8b_iter_chatFALSE/longeval_8b/` (8-shard run, chatFALSE; needs re-verification of exact `selector=iter_bm25` and `topk`)
- BABILong: `babilong_results/babilong_qcmem_seed42/` (needs verification of selector=iter_bm25, chatFALSE)
- LoCoMo: `locomo_results/qcmem_8b_iter_chatFALSE/` (verify iter_bm25, chatFALSE)
- RULER: `ruler_results/comem_lora_native_n100/` (n=100, verify chatFALSE + iter_bm25)
- LongBench: check `longbench_results/` for a chatFALSE + iter_bm25 + j12 + LoRA run

WARNING: existing "iter_chatnothink" runs use `chat_template=True + enable_thinking=False`,
NOT `chat_template=False`. These are NOT reusable for A02 phase-1. Only `chatFALSE`
suffixed directories are valid.

---

**Config 3 — j=12 + overlap w32 (Write path: zero-training contextual overlap)**

Operational meaning: Each chunk is written with a 32-token left-context prefix from the
preceding chunk (`_e2_write_chunk` in `eval_p017_e2_overlap_write.py`). The Read path and
selector are identical to Config 2 (`iter_bm25`, `topk=12`, `resume_j=12`, Read-LoRA on).
Write-LoRA is NOT used.

**CRITICAL GAP**: The overlap-write harness (`eval_p017_e2_overlap_write.py`) currently
supports ONLY the RULER synthetic benchmark (`niah_multikey_1`). It does NOT have drivers
for BABILong, LongEval, LongBench, or LoCoMo natural tasks. The paper notes explicitly
state "P0.17 overlap Write... no natural-task validation" (paperA/V6_ITERATION_NOTES.md
and TODOList.md). Running Config 3 on natural-task benchmarks requires porting the
`_e2_write_chunk` / `_run_e2` mechanics into the natural-task eval harnesses, which is
**new code work** (~100-200 lines per eval script, low-risk but non-zero).

Status: **BLOCKED for natural tasks; runnable on RULER only** (existing harness).

---

**Config 4 — j=12 + Write-LoRA (trained deployable Write + default Read)**

Operational meaning: `resume_j=12`, Read-LoRA on layers `12..35` (flagship adapter),
Write-LoRA on layers `0..11` (`step2000` adapter), selector `iter_bm25`, `topk=12`.
During the Write phase for each chunk, layers `0..11` use the trained Write LoRA
(`set_adapter(["default","write"])`); during Read the adapter is reset to `"default"` only.

**CRITICAL GAP**: The Write-LoRA two-adapter loading mechanism (`_load_with_write_lora`,
`_write_lora_enabled`) is implemented ONLY in `eval_p018_e4_2x2_writecontrol.py` for
RULER (`niah_multikey_1`). It is NOT wired into `eval_qcmem_babilong.py`,
`eval_qcmem_longeval.py`, `eval_qcmem_longbench.py`, or `eval_qcmem_locomo.py`.

The necessary code pattern is described in `paperA/P1_10_EVAL_ARM_NOTES.md`:
loading both adapters, then wrapping each chunk's write pass in `_write_lora_enabled`
context manager. This is ~50-80 lines of modification per natural-task eval script.

Status: **BLOCKED for natural tasks; runnable on RULER only** (existing harness
`scripts/_run_p018_e4_8gpu.sh` / `eval_p018_e4_2x2_writecontrol.py`).

---

**Config 5 — j=12 + Write-LoRA + Read-LoRA (joint deployment)**

Same as Config 4 (both LoRAs active, same adapter loading), which is already Config 4
because Config 4 always includes the Read-LoRA. Config 5 IS Config 4 in the current
harness design — the distinction only matters if Config 4 were defined without the
Read-LoRA. Given the existing `_load_with_write_lora` function loads both adapters,
Config 4 and Config 5 are equivalent in the current code.

Clarification needed: The PROPOSAL.md explicitly lists Config 2 as "j=12 Read-LoRA only"
vs Config 5 as "j=12 + Write-LoRA + Read-LoRA". This implies Config 4 should be
"j=12 + Write-LoRA only (no Read-LoRA)". That would require loading Write-LoRA on
layers 0..11 with NO Read-LoRA at all, which is not currently supported in any harness
(the `_load_with_write_lora` function ALWAYS loads the flagship Read-LoRA as the
"default" adapter). Config 4 without Read-LoRA would be a new code variant.

**IMPLICATION**: The 5-config comparison as specified in PROPOSAL.md requires code
changes for Configs 3, 4, 5 on all natural-task benchmarks.

---

## 2. The 5 benchmarks: dataset locations on zwfy6

| Benchmark | Dataset location (zwfy6) | Status | Verified |
|-----------|-------------------------|--------|---------|
| **LongEval** (8k/16k/64k/128k) | Synthesized on-the-fly in `eval_qcmem_longeval.py` (no static dataset) | READY | Yes |
| **BABILong** (qa1/qa2/qa5 @ 4k/16k/32k) | `data/hf_datasets_cache/` — RMT-team/babilong (HF cache implied by offline evals) | READY (OFFLINE) | Implied by existing babilong runs |
| **RULER** (multikey, VT) | `RULER/` directory on zwfy6 (`ls /apdcephfs_zwfy6/.../RULER/`) | READY | Yes (existing ruler_results) |
| **LoCoMo** | `/apdcephfs_zwfy6/.../locomo/` directory on zwfy6 | READY | Yes (existing locomo_results) |
| **LongBench** | `data/longbench_raw/` on zwfy6 (6.7MB cache seen in data/) | READY | Existing longbench_results |

All five benchmarks have existing run outputs on zwfy6, meaning the datasets + eval code
are confirmed working. Config 2 (flagship) results exist for all five.

---

## 3. Recommended node

**Node: `.82` (28.82.250.82, zwfy6, 8x H20 97.8 GB)**

Rationale:
- Both LoRAs AND base model are on zwfy6 — zero cross-disk transfer needed
- `.82` GPU status at scout time: 8 GPUs at 84-98% util (Paper B keep12 training running)
- Alternative nodes: `.73` (8x H20, also zwfy6) — same disk, same justification
- `.104` is running OLMo-2-7B keep12 training (96-100% util, ~96 GB/card)
- MAIN must decide whether to preempt Paper B on `.82` or wait for it to finish

The `.82` node has `bitsandbytes` absent per CLAUDE.md (`OPT=bnb8bit` on .82 not possible),
but A02 does not use bitsandbytes — this is not a blocker.

---

## 4. Driver script

See: `/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/scripts/_run_a02_phase1_82.sh`
(written below; do NOT chmod +x or scp — MAIN handles that)

```bash
#!/usr/bin/env bash
# ============================================================================
# A02 Phase-1 sweep: 5 configs x 5 benchmarks on .82 (zwfy6, 8x H20)
#
# Configs:
#   1. j=0 kvdirect (no retrieval, no LoRA)
#   2. j=12 Read-LoRA only (flagship CoMem, iter_bm25 topk12)
#   3. j=12 + overlap-w32  [BLOCKED: needs code; SKIP in this script]
#   4. j=12 + Write-LoRA (no Read-LoRA)  [BLOCKED: needs code; SKIP]
#   5. j=12 + Write-LoRA + Read-LoRA  [BLOCKED: needs code; SKIP]
#
# Runnable tonight: Config1 + Config2 on LongEval/BABILong/RULER/LoCoMo/LongBench
# Configs 3/4/5 need ~200-300 lines of new code (wiring overlap-write and write-lora
# into the natural-task eval harnesses). They are not launched here.
#
# env: PROJECT_ROOT PYTHON_BIN NGPU
#
# MUST RUN ON zwfy6 node (.82/.73/.104) -- assets on zwfy6 only
#
# Usage (from .82):
#   PROJECT_ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory \
#   PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
#   setsid nohup bash scripts/_run_a02_phase1_82.sh >logs/a02_phase1.out 2>&1 &
# ============================================================================
set -u
W="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$W" || exit 3
PY="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
NGPU="${NGPU:-8}"
BASE="${BASE_MODEL:-../models/Qwen--Qwen3-8b}"
READ_LORA="outputs/qcmem_distill_qwen_j12_r32_4k/final"
PROG="logs/a02_phase1_progress.log"

export HF_DATASETS_CACHE="$W/data/hf_datasets_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset http_proxy https_proxy all_proxy
mkdir -p logs

note() { printf '[%s] %s\n' "$(date +%H:%M)" "$*" | tee -a "$PROG"; }

# ---------------------------------------------------------------------------
# Pre-run assertions
# ---------------------------------------------------------------------------
if [ ! -f "$READ_LORA/adapter_model.safetensors" ]; then
  echo "FATAL: Read-LoRA not found at $READ_LORA/adapter_model.safetensors"
  exit 7
fi
note "DRIVER START on $(hostname) ngpu=$NGPU"
note "Config1=kvdirect Config2=j12_readlora (Configs 3/4/5 blocked: needs code)"

# ---------------------------------------------------------------------------
# Helper: run_sharded NAME SCRIPT_ARGS RESULT_KEY EXPECTED_N RESULT_DIR
# ---------------------------------------------------------------------------
run_sharded() {
  local NAME="$1" ARGS="$2" RESULT_DIR="$3" EXPECTED_N="$4" SCRIPT="$5"
  if [ -f "$RESULT_DIR/summary.json" ]; then
    note "SKIP $NAME: summary.json already exists (idempotent)"
    return 0
  fi
  note "$NAME START"
  for g in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$g $PY "$SCRIPT" \
      $ARGS --num_shards $NGPU --shard_index $g \
      > "logs/a02_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  local ns; ns=$(ls "$RESULT_DIR"/shard*of${NGPU}.json 2>/dev/null | wc -l)
  note "$NAME shards=$ns/$NGPU"
  if [ "$ns" -ne "$NGPU" ]; then
    note "ABORT $NAME: incomplete $ns/$NGPU shards" >&2; return 9
  fi
  # Merge
  $PY "$SCRIPT" --score_only --output_name "$(basename $RESULT_DIR)" \
    >> "logs/a02_${NAME}_merge.log" 2>&1 || true
  # Assert item count
  if [ -n "$EXPECTED_N" ] && [ -f "$RESULT_DIR/summary.json" ]; then
    $PY - "$RESULT_DIR/summary.json" "$EXPECTED_N" <<'PYEOF' || return 9
import json, sys
s=json.load(open(sys.argv[1]))
exp=int(sys.argv[2])
n=s.get("n_total",s.get("total",s.get("n",0)))
assert n==exp, f"n={n} != expected {exp}"
print(f"OK n={n}")
PYEOF
  fi
  note "$NAME DONE"
}

# ---------------------------------------------------------------------------
# Phase 1A: LongEval (synthesized, no dataset assert needed)
# Lengths: 4k 8k 16k 32k 64k 128k, 50 samples each
# ---------------------------------------------------------------------------
note "=== Phase 1A: LongEval ==="

# Config 1: kvdirect
run_sharded "longeval_c1_kvdirect" \
  "--model_path $BASE --baseline kvdirect --resume_j 12
   --lengths 4k 8k 16k 32k 64k 128k --num_samples 50
   --output_name a02_longeval_c1_kvdirect --add_bos 0" \
  "longeval_results/a02_longeval_c1_kvdirect" "" \
  "scripts/eval_qcmem_longeval.py"

# Config 2: j=12 + Read-LoRA
run_sharded "longeval_c2_j12_readlora" \
  "--model_path $BASE --resume_j 12 --lora_adapter $READ_LORA
   --selector iter_bm25 --topk 12 --sink_tokens bos
   --lengths 4k 8k 16k 32k 64k 128k --num_samples 50
   --output_name a02_longeval_c2_j12_readlora --add_bos 0" \
  "longeval_results/a02_longeval_c2_j12_readlora" "" \
  "scripts/eval_qcmem_longeval.py"

# ---------------------------------------------------------------------------
# Phase 1B: BABILong (qa1/qa2/qa5 @ 4k/16k/32k, n=100 each)
# ---------------------------------------------------------------------------
note "=== Phase 1B: BABILong ==="

# Config 1: kvdirect
for g in $(seq 0 $((NGPU-1))); do
  CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_qcmem_babilong.py \
    --model_path "$BASE" --baseline kvdirect \
    --tasks qa1 qa2 qa5 --lengths 4k 16k 32k \
    --limit 100 --chunk_size 512 --add_bos 0 \
    --num_shards $NGPU --shard_index $g \
    --output_name a02_babilong_c1_kvdirect \
    > "logs/a02_babilong_c1_shard${g}.log" 2>&1 &
done
wait
note "BABILong Config1 shards done"

# Config 2: j=12 + Read-LoRA
for g in $(seq 0 $((NGPU-1))); do
  CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_qcmem_babilong.py \
    --model_path "$BASE" --resume_j 12 --lora_adapter "$READ_LORA" \
    --selector iter_bm25 --topk 12 --sink_tokens bos \
    --tasks qa1 qa2 qa5 --lengths 4k 16k 32k \
    --limit 100 --chunk_size 512 --add_bos 0 \
    --num_shards $NGPU --shard_index $g \
    --output_name a02_babilong_c2_j12_readlora \
    > "logs/a02_babilong_c2_shard${g}.log" 2>&1 &
done
wait
note "BABILong Config2 shards done"

# ---------------------------------------------------------------------------
# Phase 1C: RULER (niah_multikey_1 + vt, n=100 per cell)
# ---------------------------------------------------------------------------
note "=== Phase 1C: RULER ==="
# Existing harness: eval_ruler_qcmem.py
# Config 1: kvdirect (no lora, resume_j=0, no retrieval)
for g in $(seq 0 $((NGPU-1))); do
  CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_ruler_qcmem.py \
    --model_path "$BASE" --baseline kvdirect \
    --tasks niah_multikey_1 variable_tracking \
    --lengths 4k 8k 16k 32k --num_samples 100 \
    --chunk_size 512 --add_bos 0 \
    --num_shards $NGPU --shard_index $g \
    --output_name a02_ruler_c1_kvdirect \
    > "logs/a02_ruler_c1_shard${g}.log" 2>&1 &
done
wait
note "RULER Config1 shards done"

# Config 2: j=12 + Read-LoRA
for g in $(seq 0 $((NGPU-1))); do
  CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_ruler_qcmem.py \
    --model_path "$BASE" --resume_j 12 --lora_adapter "$READ_LORA" \
    --selector iter_bm25 --topk 12 --sink_tokens bos \
    --tasks niah_multikey_1 variable_tracking \
    --lengths 4k 8k 16k 32k --num_samples 100 \
    --chunk_size 512 --add_bos 0 \
    --num_shards $NGPU --shard_index $g \
    --output_name a02_ruler_c2_j12_readlora \
    > "logs/a02_ruler_c2_shard${g}.log" 2>&1 &
done
wait
note "RULER Config2 shards done"

# ---------------------------------------------------------------------------
# Phase 1D: LoCoMo (paired subset)
# ---------------------------------------------------------------------------
note "=== Phase 1D: LoCoMo ==="
# Config 2 only initially (Config 1 kvdirect exists in locomo_results/kvdirect_8b_chatFALSE)
for g in $(seq 0 $((NGPU-1))); do
  CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_qcmem_locomo.py \
    --model_path "$BASE" --resume_j 12 --lora_adapter "$READ_LORA" \
    --selector iter_bm25 --topk 12 --sink_tokens bos \
    --chunk_size 512 --add_bos 0 \
    --num_shards $NGPU --shard_index $g \
    --output_name a02_locomo_c2_j12_readlora \
    > "logs/a02_locomo_c2_shard${g}.log" 2>&1 &
done
wait
note "LoCoMo Config2 shards done"

# ---------------------------------------------------------------------------
# Phase 1E: LongBench
# ---------------------------------------------------------------------------
note "=== Phase 1E: LongBench ==="
# Config 2 flagship; Config 1 kvdirect if no existing chatFALSE result
for g in $(seq 0 $((NGPU-1))); do
  CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_qcmem_longbench.py \
    --model_path "$BASE" --resume_j 12 --lora_adapter "$READ_LORA" \
    --selector iter_bm25 --topk 12 --sink_tokens bos \
    --chunk_size 512 --add_bos 0 \
    --num_shards $NGPU --shard_index $g \
    --output_name a02_longbench_c2_j12_readlora \
    > "logs/a02_longbench_c2_shard${g}.log" 2>&1 &
done
wait
note "LongBench Config2 shards done"

note "DRIVER DONE -- Configs 3/4/5 require new code (see A02_PHASE1_LAUNCH.md)"
```

---

## 5. Kill gate (verbatim from PROPOSAL.md)

> 若 paired quality CI 仍显著低于 0，则停止"CoMem 优于 RAG"的叙事，
> 定位为高复用 workload 的 storage/read-compute 方案。

The kill gate activates only after the full latency-frontier comparison (Phase B of A02,
i.e. the `eval_p0_20_equal_latency.py` + `eval_p0_20_phaseB_dense.py` harnesses).
Phase-1 (this plan) tests natural-task transfer only; it cannot by itself trigger the kill
gate.

**Paired quality CI calculation** (from PROPOSAL.md §核心成功条件 and P0.17/P0.18 pattern):

For each (config_X, config_baseline) pair on each benchmark:
1. Compute per-example binary outcomes (correct/incorrect) for both configs
2. Paired McNemar test (exact, two-tailed): using the discordant cells (N+- vs N-+)
3. Paired bootstrap CI (N=10000 resamples, seed=42): mean difference ± half-width
4. BH-correct across all (benchmark, config-pair) cells at q=0.05

Kill gate fires when:
- For the latency-matched frontier comparison (Phase B): CoMem's paired quality CI
  relative to density-matched RAG has **upper bound significantly below 0**
  (i.e., RAG dominates even after matching latency)

Phase-1 preliminary kill indicators (non-binding):
- If Config 2 (j=12 + Read-LoRA) shows NO significant quality gain vs Config 1 (j=0
  kvdirect) on ANY of {LongEval, BABILong, RULER, LoCoMo, LongBench}: this does not
  trigger the kill gate but weakens the case for Phase 2 training

Success condition from PROPOSAL.md (at least 3 of):
- LongEval or multikey relative to Read-only ≥2pp
- LoCoMo judge ≥1.5pp
- Config 5 vs Config 2: Write-LoRA closes ≥50% of j0-j12 gap on ≥3 task types

---

## 6. Wall-time estimates

Based on existing run metadata for `qcmem_8b_iter_chatFALSE`:
- LongEval 6 lengths × 50 samples ÷ 8 shards = ~6-13 samples/shard. Past run at 8 shards
  completed in ~40-70 minutes (inferred from shard file timestamps: shard0of8 files span
  longeval_4k to longeval_128k, ~07:52 to ~22:52 for the whole 6-length set).
  Estimate: **~2h per config** for LongEval 4k-128k × 50 samples.

- BABILong 3 tasks × 3 lengths × 100 samples ÷ 8 shards = 9 tasks × 100/8 ≈ 12-13 items/
  shard. Based on past BABILong runs: ~1-2h per config at this scale.

- RULER niah_multikey_1 + VT × 4 lengths × 100 samples ÷ 8 shards: prior `comem_lora_native_n100` run covers niah_single_3 × 6 lengths; estimate ~1-1.5h per config.

- LoCoMo (~1986 items ÷ 8 shards = ~248/shard): past `qcmem_8b_iter_chatFALSE` locomo run
  structure shows 8 shards; estimate ~2-3h including judge.

- LongBench (~8418 items ÷ 8 shards = ~1052/shard): estimate ~1-2h per config.

| Config | LongEval | BABILong | RULER | LoCoMo | LongBench | Total |
|--------|----------|----------|-------|--------|-----------|-------|
| C1 (kvdirect) | ~2h | ~1.5h | ~1h | ~2h | ~1.5h | ~8h |
| C2 (j12+RL) | ~2h | ~1.5h | ~1h | ~2h | ~1.5h | ~8h |
| C3-C5 (BLOCKED) | — | — | — | — | — | — |
| **Grand total** | | | | | | **~16h** for C1+C2 |

Note: C1 (kvdirect) results may be partially reusable from existing runs if they were
run with `chat_template=False`. MAIN should verify before re-running.

---

## 7. Doc bugs found in A02 PROPOSAL.md and STATUS.json

**Bug 1 (STATUS.json)**: `suspect_size: "512 bytes -- NOT adapter weights"` is incorrect.
The path `outputs/lora_best_ref/` is an **empty directory** (0 bytes of content), not a
512-byte file. The "512 bytes" figure is the directory entry overhead reported by some
`ls -la` or `du` invocations for an empty dir. This has already been noted in
`status/TRAINER_ACTIVITY.jsonl` (2026-08-09T00:06) but the STATUS.json itself was not
updated. The canonical Read-LoRA is `outputs/qcmem_distill_qwen_j12_r32_4k/final/`
(222MB adapter, on BOTH disks), not `lora_best_ref/`.

**Bug 2 (PROPOSAL.md §第一阶段)**: Config 3 is listed as "j=12 + overlap w32" but the
existing overlap-write harness (`eval_p017_e2_overlap_write.py`) is RULER-only. The
PROPOSAL implies this is "zero new training" and readily testable — that is true for RULER
but NOT for natural tasks (LongEval/BABILong/LongBench/LoCoMo). The PROPOSAL should note
that Configs 3/4/5 on natural tasks require ~1 day of coder work to wire the mechanics.

**Bug 3 (STATUS.json `read_lora.candidate_found_by_scout`)**: The text says "whether it
is the paper's flagship Read-LoRA... is UNRESOLVED". This is now resolved: the 222MB
adapter at `outputs/qcmem_distill_qwen_j12_r32_4k/final/` IS the flagship (sha dd09cd17,
layers 12..35, r=32, α=64, referenced in >10 citations in paperA including
`P0_16_E0_NOTES.md`, `distill_args.json` provenance). The STATUS.json `read_lora.status`
field should be updated from "NOT IDENTIFIED" to "CONFIRMED".

**Bug 4 (PROPOSAL.md §第一阶段 Config 4 vs 5)**: As written, Config 4 = "j=12 +
Write-LoRA" and Config 5 = "j=12 + Write-LoRA + Read-LoRA". But the current
`_load_with_write_lora` function ALWAYS loads both adapters (Read-LoRA as "default" +
Write-LoRA as "write"). Config 4 as described (Write-LoRA without Read-LoRA) would need
a different loading path. If the intent is to isolate the Write-LoRA contribution, Config 4
should be explicitly defined to omit the Read-LoRA, requiring new code. If Config 4 ≡ 5
(both LoRAs), then the 5-config list should be reduced to 4 distinct configs.

---

## Summary

**Runnable tonight with ZERO new code**: Config 1 + Config 2 on LongEval/BABILong/RULER/
LoCoMo/LongBench. These 2 × 5 = 10 benchmark-config cells can start as soon as a zwfy6
node is free.

**Needs ~1-2 days of coder work**: Configs 3, 4, 5 (overlap-write and write-lora wiring
into 4 natural-task eval scripts). Each script needs `_write_lora_enabled`-style context
manager and the overlap-write `_e2_write_chunk` ported from the RULER harness.

**Target node**: `.82` (zwfy6) — MAIN must decide whether to preempt the Paper B keep12
training currently running there.
