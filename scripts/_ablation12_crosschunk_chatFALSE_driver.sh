#!/usr/bin/env bash
# ==========================================================================
# Phase-2 ablation #12 (tab_crosschunk) re-run at chat=False.
# Paper tab_crosschunk (draft §3.5): cross-chunk attention ablation.
#   Recompute layers[j:] with (i) FULL cross-pack attention (default) vs
#   (ii) a BLOCK-DIAGONAL mask that reuses each chunk's query-blind KV without
#   attending across chunks (--reuse_kv_blockdiag).
#   RULER  : niah_single(_2) + niah_multikey(_1), lengths 8k+16k, n=50, fixed topk=12.
#   BABILong: qa2 + qa5, lengths 8k+16k, n=100.
# Unified protocol (2026-07-17 / 2026-07-22 user directives):
#   - selector = iter_bm25 (topk 12, hop_topk 4) — this is NOT a selector-ablation
#     table, so the mandated universal selector is held fixed for both arms.
#     Selection is forward-free/lexical => identical chunks for full & blockdiag,
#     keeping the attention-recompute comparison clean.
#   - chat_template = False (NO --use_chat_template; no-think implied).
#   - flagship config: model Qwen3-8b-local, adapter j12/r32/4k, resume_j 12,
#     chunk_size 1024, sink bos (same as running tab_selector #9).
# Official scoring: RULER=string_match (eval_ruler_qcmem internal), BABILong=
#   TASK_LABELS+compare_answers (eval_qcmem_babilong internal). NO re.search.
# One cell = (arm, benchmark) evaluated over its tasks x lengths, sharded 8-ways
# across the 8 GPUs; cells run sequentially (each fully occupies the node).
# ==========================================================================
set -uo pipefail
PROJECT_ROOT="/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export http_proxy="" https_proxy="" all_proxy=""
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME="$PROJECT_ROOT/.hf_cache"
export HF_DATASETS_CACHE="$PROJECT_ROOT/.hf_cache/datasets"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
PYBIN=/opt/conda/envs/torch-base/bin/python

MODEL=models/Qwen3-8b-local
LORA=outputs/qcmem_distill_qwen_j12_r32_4k/final
RULER_RESULTS=ruler_results/ablation12_crosschunk_chatFALSE
BABI_RESULTS=babilong_results/ablation12_crosschunk_chatFALSE
LOGDIR=logs/ablation12_crosschunk
CHUNK=1024
SEL=iter_bm25
TOPK=12
HOP=4
NSHARD=8
mkdir -p "$LOGDIR" "$RULER_RESULTS" "$BABI_RESULTS"

# ---- RULER cell: one arm over niah_single+niah_multikey x {8k,16k}, n=50 ----
run_ruler () {
  local arm="$1"; local flag="$2"
  echo "[$(date +%F' '%H:%M:%S)] START RULER cell arm=$arm flag='$flag' sel=$SEL topk=$TOPK chunk=$CHUNK chat=False"
  local pids=()
  for si in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$si $PYBIN scripts/eval_ruler_qcmem.py \
      --model_path "$MODEL" --resume_j 12 --lora_adapter "$LORA" \
      --selector "$SEL" --topk "$TOPK" --iter_hop_topk "$HOP" \
      --sink_tokens bos --chunk_size "$CHUNK" $flag \
      --results_folder "$RULER_RESULTS" --output_name "$arm" \
      --ruler_tasks niah_single niah_multikey --lengths 8k 16k --limit 50 \
      --num_shards "$NSHARD" --shard_index "$si" --device cuda:0 \
      --max_new_tokens 48 \
      </dev/null >"$LOGDIR/ruler_${arm}_shard${si}.log" 2>&1 &
    pids+=($!)
  done
  local rc=0
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  echo "[$(date +%F' '%H:%M:%S)] DONE  RULER cell arm=$arm (rc=$rc)"
}

# ---- BABILong cell: one arm over qa2+qa5 x {8k,16k}, n=100 ----
run_babi () {
  local arm="$1"; local flag="$2"
  echo "[$(date +%F' '%H:%M:%S)] START BABI cell arm=$arm flag='$flag' sel=$SEL topk=$TOPK chunk=$CHUNK chat=False"
  local pids=()
  for si in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$si $PYBIN scripts/eval_qcmem_babilong.py \
      --model_path "$MODEL" --resume_j 12 --lora_adapter "$LORA" \
      --selector "$SEL" --topk "$TOPK" --iter_hop_topk "$HOP" \
      --sink_tokens bos --chunk_size "$CHUNK" $flag \
      --results_folder "$BABI_RESULTS" --output_name "$arm" \
      --tasks qa2 qa5 --lengths 8k 16k --limit 100 \
      --num_shards "$NSHARD" --shard_index "$si" --device cuda:0 \
      --max_new_tokens 20 \
      </dev/null >"$LOGDIR/babi_${arm}_shard${si}.log" 2>&1 &
    pids+=($!)
  done
  local rc=0
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  echo "[$(date +%F' '%H:%M:%S)] DONE  BABI cell arm=$arm (rc=$rc)"
}

echo "[$(date)] ablation12 crosschunk driver up (chat=False, chunk=$CHUNK, sel=$SEL, topk=$TOPK, j=12, adapter=$LORA)"
run_ruler full     ""
run_ruler blockdiag "--reuse_kv_blockdiag"
run_babi  full     ""
run_babi  blockdiag "--reuse_kv_blockdiag"
echo "[$(date)] ablation12 ALL_DONE"
touch "$LOGDIR/DONE"
