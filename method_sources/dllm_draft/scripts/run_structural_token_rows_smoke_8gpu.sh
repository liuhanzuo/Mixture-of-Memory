#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${DREAM_ENV_DIR:-$ROOT/.venv_dream}"
BASE="$(cat "$ROOT/ops/artifacts/semantic_scaffold_selected_checkpoint.txt")"
OUT="$ROOT/outputs/structural_token_rows_native_smoke"
STEPS=4
METRICS="$OUT/training_metrics.jsonl"
VERIFY="$OUT/token_row_verification.json"

test -s "$BASE/model.safetensors.index.json"
test ! -e "$OUT"
mkdir -p "$OUT"

"$ROOT/scripts/setup_peft019_overlay.sh"
export PYTHONPATH="$ROOT/.peft019:$ROOT:$ROOT/vendor/DreamOn:$ROOT/vendor/Dream-Coder/instruct"
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
  --master-port "${MASTER_PORT:-29820}" \
  -m scaffold_coder.training.scaffold_sft_trainer \
  scaffold.mode=rung_mixture \
  scaffold.rung.root_probability=0.40 \
  scaffold.rung.body_probability=0.40 \
  scaffold.rung.leaf_probability=0.20 \
  scaffold.rung.token_merge_base_probability=0.25 \
  scaffold.rung.line_merge_probability=0.25 \
  data.train_batch_size=128 \
  data.micro_batch_size_per_gpu=16 \
  data.num_workers=4 \
  data.max_length=1024 \
  data.bucket_by_length=true \
  model.partial_pretrain="$BASE" \
  model.lora_rank=0 \
  model.modules_to_save=null \
  model.token_row_only=true \
  optim.lr=1.0e-3 \
  optim.weight_decay=0.0 \
  optim.warmup_steps_ratio=0.0 \
  trainer.default_local_dir="$OUT" \
  trainer.experiment_name=structural_token_rows_smoke \
  trainer.total_epochs=1 \
  trainer.total_training_steps="$STEPS" \
  trainer.save_checkpoint_steps="$STEPS" \
  trainer.profile_every_steps=1 \
  trainer.metrics_jsonl="$METRICS" \
  trainer.logger="['console']"

CHECKPOINT="$OUT/global_step_$STEPS"
test -s "$CHECKPOINT/adapter_model.safetensors"
TOKEN_IDS="$(
  "$ENV_DIR/bin/python" - "$CHECKPOINT/scaffold_tokens.json" <<'PY'
import json
import sys
data=json.load(open(sys.argv[1]))
print(" ".join(str(value["token_id"]) for value in data.values()))
PY
)"
"$ENV_DIR/bin/python" "$ROOT/scripts/verify_token_row_adapter.py" \
  --base "$BASE" \
  --adapter "$CHECKPOINT" \
  --output "$VERIFY" \
  --token-ids $TOKEN_IDS

date --iso-8601=seconds \
  >"$ROOT/ops/control/structural_token_rows_native_smoke.done"
