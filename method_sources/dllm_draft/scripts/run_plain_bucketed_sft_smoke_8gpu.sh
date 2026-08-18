#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${DREAM_ENV_DIR:-$ROOT/.venv_dream}"
PORT_BASE="${MASTER_PORT_BASE:-29643}"
PROBE_STEPS="${PLAIN_BUCKETED_PROBE_STEPS:-4}"
GLOBAL_BATCH_SIZE=128
MICRO_BATCH_CANDIDATES="${PLAIN_BUCKETED_MICRO_BATCH_CANDIDATES:-16 8}"
RUN_TAG="${PLAIN_BUCKETED_RUN_TAG:-$(date +%Y%m%dT%H%M%S)}"
RUN_DIR="$ROOT/outputs/plain_bucketed_sft_smoke/$RUN_TAG"
ARTIFACT="$ROOT/ops/artifacts/plain_bucketed_throughput_sweep.json"
RECOMMENDATION="$ROOT/ops/artifacts/plain_bucketed_micro_batch.txt"

mkdir -p "$RUN_DIR" "$ROOT/ops/artifacts" "$ROOT/ops/control"
STATUS="$RUN_DIR/status.tsv"
printf '%s\t%s\t%s\t%s\t%s\n' \
  micro_batch_size_per_gpu status exit_code log_file metrics_file \
  >"$STATUS"

export PYTHONPATH="$ROOT:$ROOT/vendor/DreamOn:$ROOT/vendor/Dream-Coder/instruct"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

candidate_index=0
success_count=0
for micro_batch in $MICRO_BATCH_CANDIDATES; do
  if (( GLOBAL_BATCH_SIZE % (8 * micro_batch) != 0 )); then
    echo "Skipping micro_batch=$micro_batch: global batch is not divisible"
    continue
  fi
  port=$((PORT_BASE + candidate_index))
  candidate_index=$((candidate_index + 1))
  name="micro_${micro_batch}"
  log_file="$name.log"
  metrics_file="$name.metrics.jsonl"
  out_dir="$RUN_DIR/$name.output"
  echo "=== bucketed plain probe micro_batch=$micro_batch port=$port ==="
  if "$ENV_DIR/bin/torchrun" \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=8 \
    --master-port "$port" \
    -m scaffold_coder.training.scaffold_sft_trainer \
    scaffold.mode=plain \
    data.train_batch_size="$GLOBAL_BATCH_SIZE" \
    data.micro_batch_size_per_gpu="$micro_batch" \
    data.num_workers=4 \
    data.max_length=1024 \
    data.bucket_by_length=true \
    model.partial_pretrain="$ROOT/models/Dream-Coder-v0-Base-7B" \
    trainer.default_local_dir="$out_dir" \
    trainer.total_epochs=1 \
    trainer.total_training_steps="$PROBE_STEPS" \
    trainer.save_checkpoint_steps=999999 \
    trainer.skip_checkpoint=true \
    trainer.profile_every_steps=1 \
    trainer.metrics_jsonl="$RUN_DIR/$metrics_file" \
    trainer.logger="['console']" \
    2>&1 | tee "$RUN_DIR/$log_file"
  then
    status=success
    exit_code=0
    success_count=$((success_count + 1))
  else
    exit_code=$?
    status=failed
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$micro_batch" "$status" "$exit_code" "$log_file" "$metrics_file" \
    >>"$STATUS"
done

if (( success_count == 0 )); then
  echo "No bucketed plain candidate completed successfully" >&2
  exit 1
fi

gpu_memory_mib="$(
  nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits \
    | sed -n '1p'
)"
"$ENV_DIR/bin/python" "$ROOT/scripts/summarize_throughput_sweep.py" \
  --run-dir "$RUN_DIR" \
  --output "$ARTIFACT" \
  --gpu-memory-mib "$gpu_memory_mib" \
  --minimum-headroom-gib 5.0 \
  --warmup-records 1

"$ENV_DIR/bin/python" - "$ARTIFACT" "$RECOMMENDATION" <<'PY'
import json
import sys
from pathlib import Path

artifact = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
recommended = artifact["recommended"]
micro = int(artifact["recommended_micro_batch_size_per_gpu"])
if micro not in {8, 16}:
    raise SystemExit(f"unexpected recommended micro batch: {micro}")
padding = recommended.get("median_padding_fraction")
if padding is None:
    raise SystemExit("bucketed probe did not report padding fraction")
if float(padding) >= 0.50:
    raise SystemExit(f"bucketed padding fraction is unexpectedly high: {padding}")
Path(sys.argv[2]).write_text(f"{micro}\n", encoding="utf-8")
print(
    {
        "recommended_micro_batch_size_per_gpu": micro,
        "median_padding_fraction": padding,
        "median_nonpadding_tokens_per_second": recommended[
            "median_nonpadding_tokens_per_second"
        ],
        "memory_headroom_gib": recommended["memory_headroom_gib"],
    }
)
PY

printf '%s\n' "$RUN_DIR" \
  >"$ROOT/ops/artifacts/plain_bucketed_sweep_latest_run.txt"
date --iso-8601=seconds \
  >"$ROOT/ops/control/plain_bucketed_sft_smoke.done"
