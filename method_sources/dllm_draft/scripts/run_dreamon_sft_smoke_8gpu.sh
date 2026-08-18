#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${DREAM_ENV_DIR:-$ROOT/.venv_dream}"
PORT="${MASTER_PORT:-29620}"
OUT="${DREAMON_SFT_SMOKE_OUT:-$ROOT/outputs/dreamon_sft_smoke}"

export PYTHONPATH="$ROOT/vendor/DreamOn:$ROOT/vendor/Dream-Coder/instruct:$ROOT"
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
  -m src.trainer.fsdp_sft_expand_trainer \
  diffusion.time_reweighting=linear \
  diffusion.weight_eos=true \
  data.train_files="$ROOT/data/scaffold_edu_v0/train_data.parquet" \
  data.val_files="$ROOT/data/scaffold_edu_v0/eval_data.parquet" \
  data.train_batch_size=8 \
  data.micro_batch_size_per_gpu=1 \
  data.max_length=1024 \
  data.prompt_key=prompt \
  data.response_key=response \
  data.truncation=right \
  data.middle_line_num=null \
  data.use_uniform_merge_prob=0.5 \
  data.max_delete=64 \
  model.partial_pretrain="$ROOT/models/Dream-Coder-v0-Base-7B" \
  model.trust_remote_code=true \
  model.enable_gradient_checkpointing=true \
  trainer.default_local_dir="$OUT" \
  trainer.project_name=dreamon-matched-smoke \
  trainer.experiment_name=dreamon_matched_smoke \
  trainer.logger="['console']" \
  trainer.total_epochs=1 \
  trainer.total_training_steps=2 \
  trainer.save_checkpoint_steps=2 \
  trainer.default_hdfs_dir=null \
  ulysses_sequence_parallel_size=1 \
  use_remove_padding=false

test -d "$OUT/global_step_2"
date --iso-8601=seconds >"$ROOT/ops/control/dreamon_sft_smoke.done"

