#!/usr/bin/env bash
# =============================================================================
# Cross-node reproducibility isolation for a FIXED decoding protocol.
#
# Pinned config (identical to runs/dream_coder_instruct_heplus_r2 on wzc1 and to
# runs/sampler_audit/he_ref_T0.1_p0.95_entropy_at0 on zwfy6):
#   Dream-Coder-v0-Instruct-7B, HumanEval+ n=164, chat template ON,
#   steps=512 max_new_tokens=512 temperature=0.1 top_p=0.95
#   alg=entropy alg_temp=0.0 seed=None, bf16, 8-way task sharding.
#
# Verified identical across the two disks before this ran:
#   model-0000{1..4}.safetensors md5 (all four), config/index/generation_config,
#   modeling_dream.py, generation_utils.py, tokenizer_config/vocab/merges/
#   added_tokens/special_tokens_map/tokenization_dream, and the 164 prompt
#   strings byte-for-byte. So this varies ONLY hardware + software stack.
#
# NODE var selects which leg to run; every leg writes its own OUTDIR so two
# nodes never contend for a shard path (that failure mode already bit the
# sampler audit twice; the merge coverage assertion caught it both times).
#
# ARM naming: <node>_<gpuarch>_<disk>_<env>[_dup]
# =============================================================================
set -u

ARM="${ARM:?set ARM}"
ROOT="${ROOT:?set ROOT}"
PY="${PY:?set PY}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"

CKPT="$ROOT/models/Dream-Coder-v0-Instruct-7B"
HE_DATA="${HE_DATA:-$ROOT/data/evalplus/HumanEvalPlus-v0.1.10.jsonl}"
GEN="${GEN:-$ROOT/scripts/generate_evalplus_dream_alg.py}"
SWEEP_ROOT="$ROOT/runs/xnode"
OUTDIR="$SWEEP_ROOT/$ARM"

export PYTHONPATH="$ROOT:$ROOT/vendor/Dream-Coder/instruct"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HUMANEVAL_OVERRIDE_PATH="$HE_DATA"

if [ -s "$OUTDIR/solutions.jsonl" ]; then
  echo "[$(date '+%F %T')] $ARM: solutions.jsonl exists -> SKIP generation"
else
  mkdir -p "$OUTDIR/shards"
  echo "[$(date '+%F %T')] ===== $ARM  gen start ====="
  NG=0; for g in $GPUS; do NG=$((NG+1)); done
  i=0
  for g in $GPUS; do
    # LOCAL_RANK must stay 0 when CUDA_VISIBLE_DEVICES pins one card,
    # otherwise shards 1-7 die with 'invalid device ordinal'.
    CUDA_VISIBLE_DEVICES=$g RANK=$i LOCAL_RANK=0 WORLD_SIZE=$NG \
      "$PY" -u "$GEN" \
        --checkpoint "$CKPT" \
        --dataset humaneval \
        --data-file "$HE_DATA" \
        --output-dir "$OUTDIR/shards" \
        --steps 512 --max-new-tokens 512 \
        --temperature 0.1 --top-p 0.95 \
        --alg entropy --alg-temp 0.0 \
        > "$OUTDIR/shard${i}.log" 2>&1 &
    i=$((i+1))
  done
  wait

  # Hard coverage assertion: a silently partial merge would corrupt pass@1.
  "$PY" "$ROOT/scripts/merge_evalplus_shards.py" \
    --input-dir "$OUTDIR/shards" \
    --solutions "$OUTDIR/solutions.jsonl" \
    --metrics "$OUTDIR/metrics.jsonl" \
    --expected 164 || { echo "[$ARM] MERGE FAILED"; exit 1; }
fi

# Provenance: the stack is the independent variable, so record it in-band.
"$PY" - "$OUTDIR" "$ARM" <<'PYEOF'
import json, platform, subprocess, sys
outdir, arm = sys.argv[1], sys.argv[2]
import torch, transformers
try:
    import evalplus; ev = evalplus.__version__
except Exception as exc:
    ev = f"unknown: {exc}"
try:
    gpu = torch.cuda.get_device_name(0)
    cap = ".".join(map(str, torch.cuda.get_device_capability(0)))
except Exception:
    gpu, cap = "n/a", "n/a"
meta = {
    "arm": arm,
    "host": platform.node(),
    "uname": platform.release(),
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "transformers": transformers.__version__,
    "evalplus_generation_side": ev,
    "gpu": gpu,
    "compute_capability": cap,
    "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
    "tf32_cudnn": torch.backends.cudnn.allow_tf32,
    "driver": subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip().splitlines()[:1],
}
json.dump(meta, open(f"{outdir}/stack_meta.json", "w"), indent=1)
print(json.dumps(meta))
PYEOF

echo "[$(date '+%F %T')] ===== $ARM DONE (grading is done centrally on wzc1 with ONE evalplus version) ====="
