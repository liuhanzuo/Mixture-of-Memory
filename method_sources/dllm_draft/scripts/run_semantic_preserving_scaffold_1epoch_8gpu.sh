#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${DREAM_ENV_DIR:-$ROOT/.venv_dream}"
PORT="${MASTER_PORT:-29811}"
OUT="${SEMANTIC_SCAFFOLD_OUT:-$ROOT/outputs/semantic_scaffold_lora_1ep}"
MERGED_OUT="${SEMANTIC_SCAFFOLD_MERGED_OUT:-$ROOT/outputs/semantic_scaffold_1ep_merged}"
METRICS="$OUT/training_metrics.jsonl"
POINTER="$ROOT/ops/artifacts/semantic_scaffold_lora_1ep_checkpoint.txt"
MERGED_POINTER="$ROOT/ops/artifacts/semantic_scaffold_1ep_merged_checkpoint.txt"
SUCCESS="$ROOT/ops/control/semantic_scaffold_lora_1ep.done"

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
  trainer.experiment_name=semantic_scaffold_lora_1ep \
  trainer.total_epochs=1 \
  trainer.total_training_steps=null \
  trainer.save_checkpoint_steps=1000 \
  trainer.profile_every_steps=10 \
  trainer.metrics_jsonl="$METRICS" \
  trainer.logger="['console']"

LATEST="$(
  find "$OUT" -maxdepth 1 -type d -name 'global_step_*' \
    | sort -V | tail -n 1
)"
test -n "$LATEST"
printf '%s\n' "$LATEST" >"$POINTER"
"$ENV_DIR/bin/python" "$ROOT/scripts/merge_scaffold_lora.py" \
  --base "$ROOT/models/Dream-Coder-v0-Instruct-7B" \
  --adapter "$LATEST" \
  --output "$MERGED_OUT"
test -s "$MERGED_OUT/model.safetensors.index.json"
printf '%s\n' "$MERGED_OUT" >"$MERGED_POINTER"
date --iso-8601=seconds >"$SUCCESS"
