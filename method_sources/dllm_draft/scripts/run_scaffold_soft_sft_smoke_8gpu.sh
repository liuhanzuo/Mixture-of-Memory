#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${DREAM_ENV_DIR:-$ROOT/.venv_dream}"
PORT="${MASTER_PORT:-29621}"
OUT="${SCAFFOLD_SOFT_SMOKE_OUT:-$ROOT/outputs/scaffold_soft_sft_smoke}"
METRICS="$OUT/training_metrics.jsonl"

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
  scaffold.mode=hierarchical \
  scaffold.desync_sigma=0.20 \
  data.train_batch_size=8 \
  data.micro_batch_size_per_gpu=1 \
  data.num_workers=2 \
  data.max_length=1024 \
  model.partial_pretrain="$ROOT/models/Dream-Coder-v0-Base-7B" \
  trainer.default_local_dir="$OUT" \
  trainer.total_epochs=1 \
  trainer.total_training_steps=2 \
  trainer.save_checkpoint_steps=999999 \
  trainer.skip_checkpoint=true \
  trainer.profile_every_steps=1 \
  trainer.metrics_jsonl="$METRICS" \
  trainer.logger="['console']"

test -s "$METRICS"
date --iso-8601=seconds >"$ROOT/ops/control/scaffold_soft_sft_smoke.done"
