#!/usr/bin/env bash
# D2a static: finish W1+W2 (14 cells) across 7 GPUs (0-5,7), GPU6 reserved for in-flight W0 32k.
# Each GPU runs a fixed 2-job sequential list (long-first), all GPUs parallel. No pool/wait-after-N.
set -uo pipefail
PROJECT_ROOT="/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
export http_proxy="http://hy-proxy.woa.com:3128"; export https_proxy="http://hy-proxy.woa.com:3128"; export all_proxy="http://hy-proxy.woa.com:3128"
export no_proxy="mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,localhost,127.0.0.1,.oa.com,.woa.com,.local"
export HF_HOME="$PROJECT_ROOT/.hf_home"
PYBIN="$PROJECT_ROOT/.venv/bin/python"
MODEL=models/Meta-Llama-3-8B
CKPT_DIR=outputs/mem_space_p11_chunk512_deltarule_normreadout
ADAPTER_CONFIG=${CKPT_DIR}/adapter_config.json
CKPT=${CKPT_DIR}/mem_space_adapter.pt
TASKS="qa1 qa2 qa5"
one() {  # W L G
  local W=$1 L=$2 G=$3
  local RES=babilong_results/p11_chunk512_final_clean_swa${W}
  local LOGD=logs/eval_d2a_clean_swa${W}; mkdir -p "$RES" "$LOGD"
  echo "[$(date)] GPU$G W=$W L=$L start"
  CUDA_VISIBLE_DEVICES=$G $PYBIN scripts/run_babilong_mem_space.py \
    --model_path $MODEL --checkpoint $CKPT --adapter_config $ADAPTER_CONFIG \
    --results_folder $RES --output_name p11_chunk512_final_clean_swa${W}_${L} \
    --tasks $TASKS --lengths $L --limit 100 --chunk_size 512 \
    --dtype bfloat16 --attn_impl sdpa --swa_eval_chunks $W \
    </dev/null >"$LOGD/${L}.log" 2>&1
}
# 14 jobs, GPU6 excluded. Long-first per GPU so the slow ones start immediately.
( one 1 32k 0; one 2 1k 0 ) &
( one 2 32k 1; one 1 1k 1 ) &
( one 1 16k 2; one 2 0k 2 ) &
( one 2 16k 3; one 1 0k 3 ) &
( one 1 8k 4; one 2 4k 4 ) &
( one 2 8k 5; one 1 4k 5 ) &
( one 1 2k 7; one 2 2k 7 ) &
wait
echo "[$(date)] D2A_STATIC_DONE"
