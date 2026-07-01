#!/usr/bin/env bash
# Eval-time SWA W={0,1,2} on P11 chunk1024 + chunk256 step500 ckpts, on .196.
# Motivation (2026-06-13 heartbeat): eval-SWA is the only winning lever (proven
# on chunk512: qa5 8k 48->67->72 monotonic). Question: does the gain hold/amplify
# across chunk sizes? chunk1024 is the strongest overall baseline. Pure offline
# eval of existing ckpts (shared diskA FS, no rsync). No training.
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
TASKS="qa1 qa2 qa5"
LENGTHS=(0k 1k 2k 4k 8k 16k 32k)
# 2 chunk-size ckpts x 3 W x 7 len = 42 jobs over 8 GPUs.
declare -A CHUNK_OF=( [c1024]=1024 [c256]=256 )
declare -A DIR_OF=( [c1024]=mem_space_p11_chunk1024_deltarule_normreadout [c256]=mem_space_p11_chunk256_deltarule_normreadout )
JOBS=()
for TAG in c1024 c256; do for W in 0 1 2; do for L in "${LENGTHS[@]}"; do JOBS+=("$TAG:$W:$L"); done; done; done
gi=0
for J in "${JOBS[@]}"; do
  TAG="${J%%:*}"; REST="${J#*:}"; W="${REST%%:*}"; L="${REST##*:}"
  CS="${CHUNK_OF[$TAG]}"; CKPT_DIR="outputs/${DIR_OF[$TAG]}"
  G=$((gi % 8)); gi=$((gi+1))
  RES=babilong_results/p11_${TAG}_step500_swa${W}
  LOGD=logs/eval_p11_${TAG}_step500_swa${W}; mkdir -p "$RES" "$LOGD"
  CUDA_VISIBLE_DEVICES=$G $PYBIN scripts/run_babilong_mem_space.py \
    --model_path $MODEL --checkpoint ${CKPT_DIR}/mem_space_adapter_step000500.pt \
    --adapter_config ${CKPT_DIR}/adapter_config.json \
    --results_folder $RES --output_name p11_${TAG}_step500_swa${W}_${L} \
    --tasks $TASKS --lengths $L --limit 100 --chunk_size $CS \
    --dtype bfloat16 --attn_impl sdpa --swa_eval_chunks $W \
    </dev/null >"$LOGD/${L}.log" 2>&1 &
  if (( gi % 8 == 0 )); then wait; fi
done
wait
echo "[$(date)] ALL_P11_CHUNKSIZE_SWA_SWEEP_DONE"
