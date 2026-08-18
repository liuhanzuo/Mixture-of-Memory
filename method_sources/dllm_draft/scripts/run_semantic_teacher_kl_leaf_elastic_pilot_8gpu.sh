#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${DREAM_ENV_DIR:-$ROOT/.venv_dream}"
BASE="$ROOT/outputs/semantic_teacher_kl_scale_calibration/checkpoint_scale_0125"
TEACHER="$ROOT/models/Dream-Coder-v0-Instruct-7B"
OUT="$ROOT/outputs/semantic_teacher_kl_leaf_elastic_pilot"
STEPS="${SEMANTIC_TEACHER_KL_LEAF_STEPS:-64}"
SUCCESS="$ROOT/ops/control/semantic_teacher_kl_leaf_elastic_pilot.done"
METRICS="$OUT/training_metrics.jsonl"

test -s "$BASE/model.safetensors.index.json"
test -s "$TEACHER/model.safetensors.index.json"
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
  --master-port "${MASTER_PORT:-29841}" \
  -m scaffold_coder.training.scaffold_sft_trainer \
  scaffold.mode=rung_mixture \
  scaffold.rung.root_probability=0.0 \
  scaffold.rung.body_probability=0.0 \
  scaffold.rung.leaf_probability=1.0 \
  scaffold.rung.token_merge_base_probability=0.50 \
  scaffold.rung.line_merge_probability=0.0 \
  scaffold.rung.max_token_delete=1 \
  scaffold.rung.max_line_delete=0 \
  data.train_batch_size=128 \
  data.micro_batch_size_per_gpu=4 \
  data.num_workers=4 \
  data.max_length=1024 \
  data.bucket_by_length=true \
  model.partial_pretrain="$BASE" \
  model.teacher_partial_pretrain="$TEACHER" \
  model.lora_rank=8 \
  model.lora_alpha=16 \
  model.target_modules=all-linear \
  model.teacher_kl_weight=1.0 \
  model.teacher_kl_temperature=1.0 \
  model.teacher_kl_topk=256 \
  model.teacher_sharding=replicated \
  "model.teacher_kl_roles=[TOKEN_STMT,TOKEN_HDR,TOKEN_DOC]" \
  optim.lr=1.0e-5 \
  optim.warmup_steps_ratio=0.05 \
  trainer.default_local_dir="$OUT" \
  trainer.experiment_name=semantic_teacher_kl_leaf_elastic_pilot \
  trainer.total_epochs=1 \
  trainer.total_training_steps="$STEPS" \
  trainer.save_checkpoint_steps="$STEPS" \
  trainer.profile_every_steps=8 \
  trainer.metrics_jsonl="$METRICS" \
  trainer.logger="['console']"

CHECKPOINT="$OUT/global_step_$STEPS"
test -s "$CHECKPOINT/adapter_model.safetensors"
test -s "$METRICS"
printf '%s\n' "$CHECKPOINT" \
  >"$ROOT/ops/artifacts/semantic_teacher_kl_leaf_elastic_checkpoint.txt"
date --iso-8601=seconds >"$SUCCESS"
