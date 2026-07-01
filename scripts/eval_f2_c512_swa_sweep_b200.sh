#!/usr/bin/env bash
# Eval-time SWA W={0,1,2} on F2 long-doc chunk512 step500 ckpt, on B200 .188 (wzc1).
# Completes the F2 (long-doc) x chunk-size grid for the eval-SWA gain study.
# Pure offline eval. PROJECT_ROOT/PYTHON_BIN point to wzc1 .venv.
set -uo pipefail
PROJECT_ROOT="/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
export http_proxy="http://hy-proxy.woa.com:3128"; export https_proxy="http://hy-proxy.woa.com:3128"; export all_proxy="http://hy-proxy.woa.com:3128"
export no_proxy="mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,localhost,127.0.0.1,.oa.com,.woa.com,.local"
export HF_HOME="$PROJECT_ROOT/.hf_home"
PYBIN="$PROJECT_ROOT/.venv/bin/python"
MODEL=models/Meta-Llama-3-8B
CKPT_DIR=outputs/f2_longdoc_chunk512
ADAPTER_CONFIG=${CKPT_DIR}/adapter_config.json
CKPT=${CKPT_DIR}/mem_space_adapter_step000500.pt
TASKS="qa1 qa2 qa5"
LENGTHS=(0k 1k 2k 4k 8k 16k 32k)
JOBS=()
for W in 0 1 2; do for L in "${LENGTHS[@]}"; do JOBS+=("$W:$L"); done; done
gi=0
for J in "${JOBS[@]}"; do
  W="${J%%:*}"; L="${J##*:}"
  G=$((gi % 8)); gi=$((gi+1))
  RES=babilong_results/f2_c512_step500_swa${W}
  LOGD=logs/eval_f2_c512_step500_swa${W}; mkdir -p "$RES" "$LOGD"
  CUDA_VISIBLE_DEVICES=$G $PYBIN scripts/run_babilong_mem_space.py \
    --model_path $MODEL --checkpoint $CKPT --adapter_config $ADAPTER_CONFIG \
    --results_folder $RES --output_name f2_c512_step500_swa${W}_${L} \
    --tasks $TASKS --lengths $L --limit 100 --chunk_size 512 \
    --dtype bfloat16 --attn_impl sdpa --swa_eval_chunks $W \
    </dev/null >"$LOGD/${L}.log" 2>&1 &
  if (( gi % 8 == 0 )); then wait; fi
done
wait
echo "[$(date)] ALL_F2_C512_SWA_SWEEP_DONE"
