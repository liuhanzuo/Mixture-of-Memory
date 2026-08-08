#!/usr/bin/env bash
# ============================================================================
# A02 Phase-1 sweep: 5 configs x 5 benchmarks on .82 (zwfy6, 8x H20)
#
# IMPORTANT: Configs 3/4/5 (overlap-w32, Write-LoRA, Write+Read-LoRA) are
# BLOCKED for natural tasks. They require ~200-300 lines of new code to wire
# the overlap-write and write-lora mechanics into the natural-task eval
# harnesses. Only Configs 1 and 2 are runnable tonight.
#
# Configs (from A02 PROPOSAL.md §第一阶段):
#   1. j=0 full-depth replay (--baseline kvdirect)
#   2. j=12 + Read-LoRA only (flagship CoMem, iter_bm25 topk12)
#   3. j=12 + overlap w32      [BLOCKED: wiring not implemented in natural-task harnesses]
#   4. j=12 + Write-LoRA       [BLOCKED: same reason]
#   5. j=12 + Write-LoRA + Read-LoRA  [BLOCKED: same reason]
#
# Benchmarks:
#   A. LongEval (4k/8k/16k/32k/64k/128k, 50 samples, synthesized)
#   B. BABILong (qa1/qa2/qa5 @ 4k/16k/32k, n=100, HF cache on zwfy6)
#   C. RULER (niah_multikey_1 + variable_tracking @ 4k/8k/16k/32k, n=100)
#   D. LoCoMo (full set, ~1986 items, scored by open-weight judge)
#   E. LongBench (multi-task, ~8418 items, chat_template=False)
#
# env: PROJECT_ROOT PYTHON_BIN NGPU
#
# MUST RUN ON zwfy6 node (.82/.73/.104) -- Read-LoRA + Write-LoRA + base model
# are on zwfy6; cross-disk transfer is 12-37 MB/s (too slow for 222MB LoRA).
#
# Usage (from .82):
#   PROJECT_ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory \
#   PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
#   setsid nohup bash scripts/_run_a02_phase1_82.sh >logs/a02_phase1.out 2>&1 &
# ============================================================================
set -u
W="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$W" || { echo "FATAL: cannot cd to $W"; exit 3; }
PY="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
NGPU="${NGPU:-8}"
BASE="${BASE_MODEL:-../models/Qwen--Qwen3-8b}"
READ_LORA="${READ_LORA:-outputs/qcmem_distill_qwen_j12_r32_4k/final}"
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
  echo "  Expected: $W/$READ_LORA/adapter_model.safetensors"
  echo "  This driver must run on a zwfy6 node (.82/.73/.104) where the LoRA lives."
  exit 7
fi
if [ ! -d "$W/../models/Qwen--Qwen3-8b" ] && [ ! -d "$(dirname $BASE)/Qwen--Qwen3-8b" ]; then
  echo "WARNING: BASE_MODEL may not be at expected path: $BASE"
fi
note "DRIVER START on $(hostname) ngpu=$NGPU"
note "BASE=$BASE READ_LORA=$READ_LORA"
note "Configs 1+2 only (3/4/5 need code); Benchmarks A-E"

# ---------------------------------------------------------------------------
# Helper: run_babilong_config NAME EXTRA_ARGS
# Runs BABILong qa1/qa2/qa5 @ 4k/16k/32k for one config (8 shards)
# Idempotent: skips if per_task summary files already exist
# ---------------------------------------------------------------------------
run_babilong() {
  local NAME="$1" EXTRA="$2"
  local RD="babilong_results/${NAME}"
  # Quick idempotency check: skip if all 3 task × 3 length summaries exist
  local done=0
  for t in qa1 qa2 qa5; do
    for l in 4k 16k 32k; do
      [ -f "$RD/${t}_${l}_shard0of${NGPU}.json" ] && done=$((done+1))
    done
  done
  if [ "$done" -ge 9 ]; then
    note "SKIP babilong $NAME: shards already present"
    return 0
  fi
  note "babilong $NAME START"
  for g in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_qcmem_babilong.py \
      --model_path "$BASE" $EXTRA \
      --tasks qa1 qa2 qa5 --lengths 4k 16k 32k \
      --limit 100 --chunk_size 512 \
      --num_shards $NGPU --shard_index $g \
      --output_name "$NAME" \
      > "logs/a02_babilong_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  # Assert 8/8 shards per (task, length) -- sample check on qa1@4k
  local ns; ns=$(ls "$RD"/qa1_4k_shard*of${NGPU}.json 2>/dev/null | wc -l)
  if [ "$ns" -ne "$NGPU" ]; then
    note "ABORT babilong $NAME: qa1_4k got $ns/$NGPU shards" >&2; return 9
  fi
  note "babilong $NAME shards OK ($ns/$NGPU on qa1_4k sample check)"
  # Merge + score. NOTE: eval_qcmem_babilong.py has NO --score_only flag; the
  # canonical scorer is score_nested_babilong.py <results_dir> [--expect N].
  $PY scripts/score_nested_babilong.py "$RD" --expect -1 \
    >> "logs/a02_babilong_${NAME}_merge.log" 2>&1 || true
  note "babilong $NAME DONE"
}

# ---------------------------------------------------------------------------
# Phase 1A: LongEval
# ---------------------------------------------------------------------------
note "=== Phase 1A: LongEval ==="

# Config 1: j=0 kvdirect (no retrieval, no LoRA)
if [ ! -d "longeval_results/a02_longeval_c1_kvdirect" ]; then
  note "longeval c1_kvdirect START"
  for g in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_qcmem_longeval.py \
      --model_path "$BASE" --baseline kvdirect \
      --lengths 4k 8k 16k 32k 64k 128k --num_samples 50 \
      --output_name a02_longeval_c1_kvdirect \
      --num_shards $NGPU --shard_index $g \
      > "logs/a02_longeval_c1_shard${g}.log" 2>&1 &
  done
  wait
  $PY scripts/eval_qcmem_longeval.py --score_only \
    --output_name a02_longeval_c1_kvdirect >> "logs/a02_longeval_c1_merge.log" 2>&1 || true
  note "longeval c1_kvdirect DONE"
else
  note "SKIP longeval c1_kvdirect: dir exists"
fi

# Config 2: j=12 + Read-LoRA (flagship)
if [ ! -d "longeval_results/a02_longeval_c2_j12_readlora" ]; then
  note "longeval c2_j12_readlora START"
  for g in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_qcmem_longeval.py \
      --model_path "$BASE" --resume_j 12 --lora_adapter "$READ_LORA" \
      --selector iter_bm25 --topk 12 --sink_tokens bos \
      --lengths 4k 8k 16k 32k 64k 128k --num_samples 50 \
      --output_name a02_longeval_c2_j12_readlora \
      --num_shards $NGPU --shard_index $g \
      > "logs/a02_longeval_c2_shard${g}.log" 2>&1 &
  done
  wait
  $PY scripts/eval_qcmem_longeval.py --score_only \
    --output_name a02_longeval_c2_j12_readlora >> "logs/a02_longeval_c2_merge.log" 2>&1 || true
  note "longeval c2_j12_readlora DONE"
else
  note "SKIP longeval c2_j12_readlora: dir exists"
fi

# ---------------------------------------------------------------------------
# Phase 1B: BABILong
# ---------------------------------------------------------------------------
note "=== Phase 1B: BABILong ==="
run_babilong "a02_babilong_c1_kvdirect" "--baseline kvdirect"
run_babilong "a02_babilong_c2_j12_readlora" \
  "--resume_j 12 --lora_adapter $READ_LORA --selector iter_bm25 --topk 12 --sink_tokens bos"

# ---------------------------------------------------------------------------
# Phase 1C: RULER (niah_multikey_1 + variable_tracking)
# ---------------------------------------------------------------------------
note "=== Phase 1C: RULER ==="

# Config 1: kvdirect
if [ ! -d "ruler_results/a02_ruler_c1_kvdirect" ]; then
  note "ruler c1_kvdirect START"
  for g in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_ruler_qcmem.py \
      --model_path "$BASE" --baseline kvdirect \
      --ruler_tasks niah_multikey_1 variable_tracking \
      --lengths 4k 8k 16k 32k --limit 100 \
      --chunk_size 512 \
      --num_shards $NGPU --shard_index $g \
      --output_name a02_ruler_c1_kvdirect \
      > "logs/a02_ruler_c1_shard${g}.log" 2>&1 &
  done
  wait
  note "ruler c1_kvdirect shards done"
else
  note "SKIP ruler c1_kvdirect: dir exists"
fi

# Config 2: j=12 + Read-LoRA
if [ ! -d "ruler_results/a02_ruler_c2_j12_readlora" ]; then
  note "ruler c2_j12_readlora START"
  for g in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_ruler_qcmem.py \
      --model_path "$BASE" --resume_j 12 --lora_adapter "$READ_LORA" \
      --selector iter_bm25 --topk 12 --sink_tokens bos \
      --ruler_tasks niah_multikey_1 variable_tracking \
      --lengths 4k 8k 16k 32k --limit 100 \
      --chunk_size 512 \
      --num_shards $NGPU --shard_index $g \
      --output_name a02_ruler_c2_j12_readlora \
      > "logs/a02_ruler_c2_shard${g}.log" 2>&1 &
  done
  wait
  note "ruler c2_j12_readlora shards done"
else
  note "SKIP ruler c2_j12_readlora: dir exists"
fi

# ---------------------------------------------------------------------------
# Phase 1D: LoCoMo
# ---------------------------------------------------------------------------
note "=== Phase 1D: LoCoMo ==="

# Config 2: j=12 + Read-LoRA
# Config 1 (kvdirect) may exist as locomo_results/kvdirect_8b_chatFALSE -- MAIN should check
if [ ! -d "locomo_results/a02_locomo_c2_j12_readlora" ]; then
  note "locomo c2_j12_readlora START"
  for g in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_qcmem_locomo.py \
      --model_path "$BASE" --resume_j 12 --lora_adapter "$READ_LORA" \
      --selector iter_bm25 --topk 12 --sink_tokens bos \
      --chunk_size 512 \
      --num_shards $NGPU --shard_index $g \
      --output_dir locomo_results/a02_locomo_c2_j12_readlora \
      > "logs/a02_locomo_c2_shard${g}.log" 2>&1 &
  done
  wait
  note "locomo c2_j12_readlora shards done"
else
  note "SKIP locomo c2_j12_readlora: dir exists"
fi

# ---------------------------------------------------------------------------
# Phase 1E: LongBench
# ---------------------------------------------------------------------------
note "=== Phase 1E: LongBench ==="

# Config 2: j=12 + Read-LoRA
# Config 1 (kvdirect) may exist as longbench_results/kvdirect_8b_chatFALSE -- MAIN should check
if [ ! -d "longbench_results/a02_longbench_c2_j12_readlora" ]; then
  note "longbench c2_j12_readlora START"
  for g in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_qcmem_longbench.py \
      --model_path "$BASE" --resume_j 12 --lora_adapter "$READ_LORA" \
      --selector iter_bm25 --topk 12 --sink_tokens bos \
      --chunk_size 512 \
      --num_shards $NGPU --shard_index $g \
      --output_dir longbench_results/a02_longbench_c2_j12_readlora \
      > "logs/a02_longbench_c2_shard${g}.log" 2>&1 &
  done
  wait
  note "longbench c2_j12_readlora shards done"
else
  note "SKIP longbench c2_j12_readlora: dir exists"
fi

note "DRIVER DONE -- Configs 3/4/5 require new code; see status/proposal_prep/A02_PHASE1_LAUNCH.md"
note "Next: merge RULER + LongBench manually if needed; run locomo judge"
