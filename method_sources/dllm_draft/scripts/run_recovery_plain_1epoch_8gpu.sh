#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${DREAM_ENV_DIR:-$ROOT/.venv_dream}"
ARM="${RECOVERY_ARM:?set RECOVERY_ARM}"

case "$ARM" in
  base_plain_1ep)
    MODEL="$ROOT/models/Dream-Coder-v0-Base-7B"
    OUT="$ROOT/outputs/recovery_base_plain_1ep"
    POINTER="$ROOT/ops/artifacts/recovery_base_plain_1ep_checkpoint.txt"
    SUCCESS="$ROOT/ops/control/recovery_base_plain_1ep.done"
    PORT="${MASTER_PORT:-29701}"
    ALL_MASK=0.0
    HIGH_NOISE=0.0
    ;;
  instruct_plain_1ep)
    MODEL="$ROOT/models/Dream-Coder-v0-Instruct-7B"
    OUT="$ROOT/outputs/recovery_instruct_plain_1ep"
    POINTER="$ROOT/ops/artifacts/recovery_instruct_plain_1ep_checkpoint.txt"
    SUCCESS="$ROOT/ops/control/recovery_instruct_plain_1ep.done"
    PORT="${MASTER_PORT:-29702}"
    ALL_MASK=0.0
    HIGH_NOISE=0.0
    ;;
  instruct_highnoise_1ep)
    MODEL="$ROOT/models/Dream-Coder-v0-Instruct-7B"
    OUT="$ROOT/outputs/recovery_instruct_highnoise_1ep"
    POINTER="$ROOT/ops/artifacts/recovery_instruct_highnoise_1ep_checkpoint.txt"
    SUCCESS="$ROOT/ops/control/recovery_instruct_highnoise_1ep.done"
    PORT="${MASTER_PORT:-29703}"
    ALL_MASK=0.20
    HIGH_NOISE=0.30
    ;;
  *)
    echo "Unsupported RECOVERY_ARM=$ARM" >&2
    exit 2
    ;;
esac

METRICS="$OUT/training_metrics.jsonl"
RESUME_ARGS=()
while IFS= read -r candidate; do
  if [ -f "$candidate/training_state.pt" ] \
    && [ -f "$candidate/optimizer_state.pt" ] \
    && [ -f "$candidate/model.safetensors.index.json" ]; then
    echo "Resuming $ARM from $candidate"
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

"$ENV_DIR/bin/torchrun" \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  --master-port "$PORT" \
  -m scaffold_coder.training.scaffold_sft_trainer \
  scaffold.mode=plain \
  scaffold.plain_all_mask_probability="$ALL_MASK" \
  scaffold.plain_high_noise_probability="$HIGH_NOISE" \
  scaffold.plain_high_noise_min_t=0.80 \
  data.train_batch_size=128 \
  data.micro_batch_size_per_gpu=16 \
  data.num_workers=4 \
  data.max_length=1024 \
  data.bucket_by_length=true \
  model.partial_pretrain="$MODEL" \
  optim.lr=2.0e-6 \
  optim.warmup_steps_ratio=0.05 \
  trainer.default_local_dir="$OUT" \
  trainer.experiment_name="$ARM" \
  trainer.total_epochs=1 \
  trainer.total_training_steps=null \
  trainer.save_checkpoint_steps=1000 \
  trainer.profile_every_steps=10 \
  trainer.metrics_jsonl="$METRICS" \
  trainer.logger="['console']" \
  "${RESUME_ARGS[@]}"

LATEST="$(
  find "$OUT" -maxdepth 1 -type d -name 'global_step_*' \
    | sort -V | tail -n 1
)"
test -n "$LATEST"
printf '%s\n' "$LATEST" >"$POINTER"
date --iso-8601=seconds >"$SUCCESS"
