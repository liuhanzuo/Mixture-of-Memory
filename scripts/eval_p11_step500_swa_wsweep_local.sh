#!/usr/bin/env bash
# Eval-time SWA window sweep on P11 chunk512 step500 (best ckpt), LOCAL 8xH20.
# Motivation (2026-06-13 heartbeat): eval-time cross-chunk SWA is the ONLY
# non-falsified lever. W0->W1->W2 is monotonic and STILL climbing at W=2
# (qa5 8k: 48->67->72; 16k: 45->57->67). Sweep larger W to find saturation.
# Pure offline eval of existing ckpt — no training, no architecture change.
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
CKPT=${CKPT_DIR}/mem_space_adapter_step000500.pt
TASKS="qa1 qa2 qa5"
LENGTHS=(0k 1k 2k 4k 8k 16k 32k)
# Larger windows only (W0/1/2 already done). 4 W x 7 len = 28 jobs over 8 GPUs.
JOBS=()
for W in 3 4 6 8; do for L in "${LENGTHS[@]}"; do JOBS+=("$W:$L"); done; done
gi=0
for J in "${JOBS[@]}"; do
  W="${J%%:*}"; L="${J##*:}"
  G=$((gi % 8)); gi=$((gi+1))
  RES=babilong_results/p11_step500_local_swa${W}
  LOGD=logs/eval_p11_step500_swa${W}; mkdir -p "$RES" "$LOGD"
  CUDA_VISIBLE_DEVICES=$G $PYBIN scripts/run_babilong_mem_space.py \
    --model_path $MODEL --checkpoint $CKPT --adapter_config $ADAPTER_CONFIG \
    --results_folder $RES --output_name p11_step500_local_swa${W}_${L} \
    --tasks $TASKS --lengths $L --limit 100 --chunk_size 512 \
    --dtype bfloat16 --attn_impl sdpa --swa_eval_chunks $W \
    </dev/null >"$LOGD/${L}.log" 2>&1 &
  if (( gi % 8 == 0 )); then wait; fi
done
wait
echo "[$(date)] ALL_P11_SWA_WSWEEP_DONE"
