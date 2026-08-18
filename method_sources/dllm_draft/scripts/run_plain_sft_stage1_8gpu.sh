#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${DREAM_ENV_DIR:-$ROOT/.venv_dream}"
PORT="${MASTER_PORT:-29642}"
OUT="${PLAIN_STAGE1_OUT:-$ROOT/outputs/plain_sft_stage1}"
METRICS="$OUT/training_metrics.jsonl"
MICRO_BATCH_FILE="$ROOT/ops/artifacts/plain_bucketed_micro_batch.txt"
MICRO_BATCH=8
if [ -s "$MICRO_BATCH_FILE" ]; then
  MICRO_BATCH="$(tr -d '[:space:]' <"$MICRO_BATCH_FILE")"
fi
if [ "$MICRO_BATCH" != 8 ] && [ "$MICRO_BATCH" != 16 ]; then
  echo "Invalid bucketed plain micro batch: $MICRO_BATCH" >&2
  exit 1
fi
if (( 128 % (8 * MICRO_BATCH) != 0 )); then
  echo "Global batch 128 is incompatible with micro batch $MICRO_BATCH" >&2
  exit 1
fi
echo "Using bucketed plain micro_batch_size_per_gpu=$MICRO_BATCH"
RESUME_ARGS=()
while IFS= read -r candidate; do
  if [ -f "$candidate/training_state.pt" ] \
    && [ -f "$candidate/optimizer_state.pt" ] \
    && [ -f "$candidate/model.safetensors.index.json" ]; then
    echo "Resuming plain Stage-1 from $candidate"
    RESUME_ARGS=(
      trainer.resume_training=true
      trainer.resume_path="$candidate"
    )
    break
  fi
done < <(
  find "$OUT" -maxdepth 1 -type d -name 'global_step_*' 2>/dev/null \
    | sort -Vr
)

export PYTHONPATH="$ROOT:$ROOT/vendor/DreamOn:$ROOT/vendor/Dream-Coder/instruct"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

"$ENV_DIR/bin/torchrun" \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  --master-port "$PORT" \
  -m scaffold_coder.training.scaffold_sft_trainer \
  scaffold.mode=plain \
  data.train_batch_size=128 \
  data.micro_batch_size_per_gpu="$MICRO_BATCH" \
  data.num_workers=4 \
  data.max_length=1024 \
  data.bucket_by_length=true \
  model.partial_pretrain="$ROOT/models/Dream-Coder-v0-Base-7B" \
  trainer.default_local_dir="$OUT" \
  trainer.total_epochs=5 \
  trainer.total_training_steps=null \
  trainer.save_checkpoint_steps=1000 \
  trainer.profile_every_steps=10 \
  trainer.metrics_jsonl="$METRICS" \
  trainer.logger="['console']" \
  "${RESUME_ARGS[@]}"

LATEST="$(find "$OUT" -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1)"
test -n "$LATEST"
printf '%s\n' "$LATEST" \
  >"$ROOT/ops/artifacts/plain_stage1_latest_checkpoint.txt"
date --iso-8601=seconds \
  >"$ROOT/ops/control/plain_stage1.done"
