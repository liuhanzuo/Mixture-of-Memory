#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${DREAM_ENV_DIR:-$ROOT/.venv_dream}"
PORT="${MASTER_PORT:-29831}"
OUT="${SEMANTIC_TEACHER_KL_OUT:-$ROOT/outputs/semantic_teacher_kl_smoke}"
STEPS="${SEMANTIC_TEACHER_KL_STEPS:-4}"
SUCCESS="${SEMANTIC_TEACHER_KL_SUCCESS:-$ROOT/ops/control/semantic_teacher_kl_smoke.done}"
METRICS="$OUT/training_metrics.jsonl"

test ! -e "$OUT"
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
  scaffold.rung.root_probability=0.15 \
  scaffold.rung.body_probability=0.25 \
  scaffold.rung.leaf_probability=0.60 \
  scaffold.rung.token_merge_base_probability=0.25 \
  scaffold.rung.line_merge_probability=0.25 \
  data.train_batch_size=128 \
  data.micro_batch_size_per_gpu=4 \
  data.num_workers=4 \
  data.max_length=1024 \
  data.bucket_by_length=true \
  model.partial_pretrain="$ROOT/models/Dream-Coder-v0-Instruct-7B" \
  model.lora_rank=8 \
  model.lora_alpha=16 \
  model.target_modules=all-linear \
  model.teacher_kl_weight=1.0 \
  model.teacher_kl_temperature=1.0 \
  model.teacher_kl_topk=256 \
  model.teacher_sharding=replicated \
  "model.teacher_kl_roles=[TOKEN_STMT,TOKEN_HDR,TOKEN_DOC]" \
  optim.lr=2.0e-5 \
  optim.warmup_steps_ratio=0.0 \
  trainer.default_local_dir="$OUT" \
  trainer.experiment_name=semantic_teacher_kl_smoke \
  trainer.total_epochs=1 \
  trainer.total_training_steps="$STEPS" \
  trainer.save_checkpoint_steps="$STEPS" \
  trainer.profile_every_steps=1 \
  trainer.metrics_jsonl="$METRICS" \
  trainer.logger="['console']"

CHECKPOINT="$OUT/global_step_$STEPS"
test -s "$CHECKPOINT/adapter_model.safetensors"
test -s "$METRICS"
printf '%s\n' "$CHECKPOINT" \
  >"$ROOT/ops/artifacts/semantic_teacher_kl_smoke_checkpoint.txt"
date --iso-8601=seconds >"$SUCCESS"
