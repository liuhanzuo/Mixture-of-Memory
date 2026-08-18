#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${DREAM_ENV_DIR:-$ROOT/.venv_dream}"
PORT="${MASTER_PORT:-29810}"
OUT="${SEMANTIC_SCAFFOLD_SMOKE_OUT:-$ROOT/outputs/semantic_scaffold_lora_smoke}"
STEPS="${SEMANTIC_SCAFFOLD_SMOKE_STEPS:-4}"
METRICS="$OUT/training_metrics.jsonl"

rm -rf "$OUT"
mkdir -p "$OUT"

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
  scaffold.mode=rung_mixture \
  scaffold.rung.root_probability=0.10 \
  scaffold.rung.body_probability=0.20 \
  scaffold.rung.leaf_probability=0.70 \
  scaffold.rung.token_merge_base_probability=0.25 \
  scaffold.rung.line_merge_probability=0.25 \
  data.train_batch_size=128 \
  data.micro_batch_size_per_gpu=16 \
  data.num_workers=4 \
  data.max_length=1024 \
  data.bucket_by_length=true \
  model.partial_pretrain="$ROOT/models/Dream-Coder-v0-Instruct-7B" \
  model.lora_rank=16 \
  model.lora_alpha=32 \
  model.target_modules=all-linear \
  optim.lr=5.0e-5 \
  optim.warmup_steps_ratio=0.05 \
  trainer.default_local_dir="$OUT" \
  trainer.experiment_name=semantic_scaffold_lora_smoke \
  trainer.total_epochs=1 \
  trainer.total_training_steps="$STEPS" \
  trainer.save_checkpoint_steps="$STEPS" \
  trainer.profile_every_steps=1 \
  trainer.metrics_jsonl="$METRICS" \
  trainer.logger="['console']"

test -d "$OUT/global_step_$STEPS"
test -s "$METRICS"
date --iso-8601=seconds \
  >"$ROOT/ops/control/semantic_scaffold_lora_smoke.done"
