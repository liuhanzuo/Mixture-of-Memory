#!/usr/bin/env bash
# P11 chunk512 STEP500 (SOTA-peak ckpt) cross-chunk SWA W0/W1/W2 BABILong sweep.
# Mirrors scripts/eval_d2a_swa_clean_rerun.sh (FINAL ckpt) but uses the step000500
# checkpoint instead, to measure "strongest ckpt + SWA" ceiling.
#   21 jobs = W{0,1,2} x 7 lengths(0k-32k), qa1/qa2/qa5, n=100, chunk_size=512,
#   bf16 / sdpa, --swa_eval_chunks W. All run back-to-back on one node (8 GPUs) to
#   minimize cross-process bf16 nondeterminism.
#
# Node-agnostic: override PROJECT_ROOT / PYTHON_BIN for the node you land on.
#   disk-A (本机/.196):  PROJECT_ROOT=/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory
#   disk-B (.249/.76):   PROJECT_ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
#   B200 wzc1 (.188):    PROJECT_ROOT=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory
# Default PYTHON_BIN = $PROJECT_ROOT/.venv/bin/python (works on .76/.249/.188/本机).
# On .196 use PYTHON_BIN=/opt/conda/envs/torch-base/bin/python.
set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
# proxy + HF offline so rank0 babilong prefetch never blocks on net (diskB/wzc1 have no internet)
export http_proxy="${http_proxy:-http://hy-proxy.woa.com:3128}"
export https_proxy="${https_proxy:-http://hy-proxy.woa.com:3128}"
export all_proxy="${all_proxy:-http://hy-proxy.woa.com:3128}"
export no_proxy="mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,localhost,127.0.0.1,.oa.com,.woa.com,.local"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.hf_home}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
PYBIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
MODEL=models/Meta-Llama-3-8B
CKPT_DIR=outputs/mem_space_p11_chunk512_deltarule_normreadout
ADAPTER_CONFIG=${CKPT_DIR}/adapter_config.json
CKPT=${CKPT_DIR}/mem_space_adapter_step000500.pt
TASKS="qa1 qa2 qa5"
LENGTHS=(0k 1k 2k 4k 8k 16k 32k)

if [[ ! -f "$CKPT" ]]; then
  echo "[FATAL] step500 ckpt not found: $PROJECT_ROOT/$CKPT" >&2
  echo "        rsync it from disk-A before running on this node." >&2
  exit 1
fi

# 8 GPUs: assign (W,length) pairs round-robin. 3 W x 7 len = 21 jobs over 8 GPUs.
JOBS=()
for W in 0 1 2; do for L in "${LENGTHS[@]}"; do JOBS+=("$W:$L"); done; done
gi=0
for J in "${JOBS[@]}"; do
  W="${J%%:*}"; L="${J##*:}"
  G=$((gi % 8)); gi=$((gi+1))
  RES=babilong_results/p11_chunk512_step500_swaW${W}
  LOGD=logs/eval_p11_step500_swaW${W}; mkdir -p "$RES" "$LOGD"
  CUDA_VISIBLE_DEVICES=$G $PYBIN scripts/run_babilong_mem_space.py \
    --model_path $MODEL --checkpoint $CKPT --adapter_config $ADAPTER_CONFIG \
    --results_folder $RES --output_name p11_chunk512_step500_swaW${W}_${L} \
    --tasks $TASKS --lengths $L --limit 100 --chunk_size 512 \
    --dtype bfloat16 --attn_impl sdpa --swa_eval_chunks $W \
    </dev/null >"$LOGD/${L}.log" 2>&1 &
  if (( gi % 8 == 0 )); then wait; fi
done
wait
echo "[$(date)] ALL_P11_STEP500_SWA_DONE"
