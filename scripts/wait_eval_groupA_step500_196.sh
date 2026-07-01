#!/usr/bin/env bash
# Detached waiter: poll for Group-A (FIX_twostage grouped-readout) step500 ckpt,
# then launch W0 BABILong taskpool eval on .196's 8 GPUs (read-only; diskA shared
# FS so no rsync). Grouped-readout keep_all matches training (topk_chunks=0 in cfg);
# add --rawkv_grouped_readout --rawkv_subblock_size 64, NO stage1_select.
set -uo pipefail
PROJECT_ROOT="/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory"
cd "$PROJECT_ROOT"
OUTDIR="$PROJECT_ROOT/outputs/mem_space_FIX_twostage_chunk512_diskA"
CKPT="$OUTDIR/full_model_step000500.pt"
ADAPTER="$OUTDIR/adapter_config.json"
STAMP="$(date +%Y%m%d_%H%M%S)"

echo "[$(date)] waiter up; polling for $CKPT"
while [ ! -f "$CKPT" ] || [ ! -f "$ADAPTER" ]; do
  sleep 120
done
# allow torch.save to finish flushing
sleep 30
echo "[$(date)] ckpt landed -> launching W0 taskpool eval on .196 8 GPUs"

export PROJECT_ROOT
export PYTHON_BIN="/opt/conda/envs/torch-base/bin/python"
RUN_PREFIX="FIXtwostage_groupA_step500" \
CKPT_FILES="$CKPT" \
CK_NAMES="FIXtwostage_groupA_step500" \
ADAPTER_CONFIG="$ADAPTER" \
MODEL="models/Meta-Llama-3-8B" \
TASKS="qa1 qa2 qa5" \
LENGTHS="0k 1k 2k 4k 8k 16k 32k" \
CHUNK_SIZE=512 \
EXTRA_ARGS="--rawkv_grouped_readout --rawkv_subblock_size 64" \
bash scripts/_eval_taskpool_2group.sh \
  > "$PROJECT_ROOT/logs/FIXtwostage_groupA_step500_W0_sched_${STAMP}.out" 2>&1
echo "[$(date)] eval sched finished"
