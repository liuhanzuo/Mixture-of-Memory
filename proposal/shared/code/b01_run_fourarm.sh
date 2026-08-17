#!/usr/bin/env bash
# B01 four-arm gate --- the TWO ARMS THAT ARE EXECUTABLE TODAY, at full n.
#
# Arm 1 = stock + Read-LoRA        (resume_j=12, no funnel)
# Arm 2 = bottleneck only, PERSISTED (resume_j=13, --persist_bottleneck_latent)
#
# Arms 3/4 (bottleneck + Read-LoRA [+ Write-LoRA]) are NOT launched here. See
# FOURARM_VERDICT_20260817.md: the only Read-LoRA on disk was distilled against the
# STOCK upper stack, and the funnel CPT moved layers 12..35 by 1.57-8.62x MORE than
# the adapter's own correction on 48/48 matched tensors. Running them would report a
# base-mismatch as an arm effect.
#
# Runs on .25 (8x B200). Each arm = 4 GPUs x 4 strided shards, both arms concurrently.
# Full LoCoMo n=1986 per arm, identical sample set (same data file, same shard rule).
set -u

R=/apdcephfs_wzz/share_303419932/pighzliu_code/Mixture-of-Memory
PY=/opt/conda/envs/torch-base/bin/python
export PYTHONPATH="$R:/root/b01_deps"
export PROJECT_ROOT="$R"
cd "$R" || exit 9

DATA=/root/b01_deps/locomo/data/locomo10.json
BASE=/root/b01_assets/base
FUNNEL=/root/b01_assets/funnel/final.pt
READLORA=/root/b01_assets/read_lora/final
OUT=/root/b01_fourarm
NSH=4

mkdir -p "$OUT"

# fail-closed: refuse to start if anything is already on the GPUs
BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
if [ "$BUSY" -ne 0 ]; then
  echo "FATAL: $BUSY compute apps already on .25 -- refusing to launch" >&2
  exit 3
fi

launch_arm () {
  local name="$1"; shift
  local gpu0="$1"; shift
  for s in 0 1 2 3; do
    local gpu=$(( gpu0 + s ))
    CUDA_VISIBLE_DEVICES=$gpu setsid nohup $PY scripts/eval_qcmem_locomo.py \
      --model_path "$BASE" \
      --locomo_data "$DATA" \
      --selector iter_bm25 --iter_rounds 2 \
      --num_shards $NSH --shard_index $s \
      --output_dir "$OUT/$name" \
      "$@" \
      > "$OUT/${name}_shard${s}.log" 2>&1 &
    echo "  launched $name shard $s on GPU $gpu pid $!"
  done
}

echo "[b01] arm1 = stock + Read-LoRA (resume_j=12) on GPUs 0-3"
launch_arm arm1_stock_readlora 0 --resume_j 12 --lora_adapter "$READLORA"

echo "[b01] arm2 = bottleneck only, PERSISTED (resume_j=13) on GPUs 4-7"
launch_arm arm2_bottleneck_persist 4 --resume_j 13 \
  --bottleneck_ckpt "$FUNNEL" --persist_bottleneck_latent

echo "[b01] all 8 shards launched at $(date -Is)"
