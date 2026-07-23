#!/usr/bin/env bash
# ==========================================================================
# Phase-2 ablations #10 (tab_itervt) + #11 (tab_chunk) re-run at chat=False.
# Carved onto .82 (diskB, 8xH20) after the RULER-baseline chatFALSE chain
# finished. Runs SERIALLY: tab_itervt (headline) first, then tab_chunk.
#
# tab_itervt (paper/sections/tab_itervt.tex): RULER variable_tracking, n=100.
#   Fills the two chat=False columns not covered by the other ablations:
#     (i)  iter_bm25  (multi-hop lexical BFS on the literal VAR chain; hop 4,
#          topk 16 -> ceil(16/4)=4 hop rounds), and
#     (ii) oracle-VT  (single-pass; --selector oracle degrades to recency on
#          variable_tracking by design -> reproduces the tex "Oracle" column,
#          which #9 tab_selector skipped for VT).
#   Lengths 8k/16k/32k/64k/128k. chunk_size 1024 (matches #9/#12 siblings).
#
# tab_chunk (paper/sections/tab_chunk.tex): ONLY the multikey-recall column is
#   chat-sensitive (read_len/peak/prefill/decode are pure timing -> exempt, not
#   rerun here). RULER niah_multikey, n=100, fixed k=12 (topk 12), across the
#   tex's chunk sizes {128,256,512,1024}. Lengths 8k/16k/32k/64k.
#
# Unified protocol (2026-07-17 / 2026-07-22 user directives):
#   - selector = iter_bm25 (mandated universal selector) for tab_chunk; for
#     tab_itervt the iter_bm25 vs oracle contrast IS the table, so both are run.
#   - chat_template = False (NO --use_chat_template; no-think implied).
#   - flagship: model Qwen3-8b-local, LoRA adapter j12/r32/4k, resume_j 12,
#     sink bos, 8-way GPU sharding.
# Official scoring: RULER string_match (eval_ruler_qcmem internal). NO re.search.
#   Merge 8 shards with scripts/score_ruler_taskbreadth.py (re-runs the official
#   _string_match_all_one kernel on the QUOTE_ALL CSVs + Iron-Law-2 checks).
# One cell = one (selector|chunk) config over its tasks x lengths, 8-way sharded
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
NSHARD=8

# ---------------- tab_itervt (#10): VT n=100, chat=False ------------------
IVT_RESULTS=ruler_results/ablation10_itervt_chatFALSE
IVT_LOGDIR=logs/ablation10_itervt
IVT_LENS="8k 16k 32k 64k 128k"
mkdir -p "$IVT_RESULTS" "$IVT_LOGDIR"

run_itervt () {  # $1=output_name  $2=selector  $3=topk
  local name="$1" sel="$2" tk="$3"
  echo "[$(date +%F' '%H:%M:%S)] START itervt name=$name sel=$sel topk=$tk hop=4 chunk=1024 chat=False lens=$IVT_LENS"
  local pids=()
  for si in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$si $PYBIN scripts/eval_ruler_qcmem.py \
      --model_path "$MODEL" --resume_j 12 --lora_adapter "$LORA" \
      --selector "$sel" --topk "$tk" --iter_hop_topk 4 \
      --sink_tokens bos --chunk_size 1024 \
      --results_folder "$IVT_RESULTS" --output_name "$name" \
      --ruler_tasks variable_tracking --lengths $IVT_LENS --limit 100 \
      --num_shards "$NSHARD" --shard_index "$si" --device cuda:0 \
      --max_new_tokens 48 \
      </dev/null >"$IVT_LOGDIR/${name}_shard${si}.log" 2>&1 &
    pids+=($!)
  done
  local rc=0 p
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  echo "[$(date +%F' '%H:%M:%S)] DONE  itervt name=$name (rc=$rc)"
}

# ---------------- tab_chunk (#11): niah_multikey n=100, chat=False --------
CHK_RESULTS=ruler_results/ablation11_chunk_chatFALSE
CHK_LOGDIR=logs/ablation11_chunk
CHK_LENS="8k 16k 32k 64k"
mkdir -p "$CHK_RESULTS" "$CHK_LOGDIR"

run_chunk () {  # $1=chunk_size
  local cs="$1" name="chunk${1}_multikey"
  echo "[$(date +%F' '%H:%M:%S)] START chunk chunk=$cs name=$name sel=iter_bm25 topk=12 chat=False lens=$CHK_LENS"
  local pids=()
  for si in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$si $PYBIN scripts/eval_ruler_qcmem.py \
      --model_path "$MODEL" --resume_j 12 --lora_adapter "$LORA" \
      --selector iter_bm25 --topk 12 --iter_hop_topk 4 \
      --sink_tokens bos --chunk_size "$cs" \
      --results_folder "$CHK_RESULTS" --output_name "$name" \
      --ruler_tasks niah_multikey --lengths $CHK_LENS --limit 100 \
      --num_shards "$NSHARD" --shard_index "$si" --device cuda:0 \
      --max_new_tokens 48 \
      </dev/null >"$CHK_LOGDIR/${name}_shard${si}.log" 2>&1 &
    pids+=($!)
  done
  local rc=0 p
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  echo "[$(date +%F' '%H:%M:%S)] DONE  chunk chunk=$cs (rc=$rc)"
}

echo "[$(date)] ablation10+11 driver up on .82 (chat=False, j=12, adapter=$LORA)"
# --- tab_itervt first (headline) ---
run_itervt iterbm25_vt iter_bm25 16
run_itervt oracle_vt   oracle    16
touch "$IVT_LOGDIR/DONE"
echo "[$(date)] tab_itervt (#10) DONE"
# --- tab_chunk ---
run_chunk 128
run_chunk 256
run_chunk 512
run_chunk 1024
touch "$CHK_LOGDIR/DONE"
echo "[$(date)] tab_chunk (#11) DONE"
echo "[$(date)] ablation10+11 ALL_DONE"
touch "$CHK_LOGDIR/ALL_DONE"
