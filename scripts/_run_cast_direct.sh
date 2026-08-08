#!/usr/bin/env bash
# CAST repro training launcher — bypasses launch_cast_llama.sh's layout assumption.
# Runs on .21 (wzc1, L20A 8x). Assumes CWD = repo root Mixture-of-Memory.
set -u

W="${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}"
CAST=$W/baselines/cast_repro
PY=${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}
TR=${TORCHRUN_BIN:-/opt/conda/envs/torch-base/bin/torchrun}
DATA=$W/data/dolmino-mix-1124-llama2
PROJ=/apdcephfs_wzc1/share_304376610/pighzliu_code   # so root/models/Llama--Llama2-7b resolves
LOG=$W/logs/cast_repro_ddp_direct_$(date +%m%d_%H%M%S).log

[ -f "$DATA/metadata.json" ] || { echo "FATAL: $DATA missing metadata.json"; exit 3; }
[ -f "$DATA/train.bin" ] || { echo "FATAL: $DATA/train.bin missing -- run tools/concat_shards_to_bin.py first"; exit 6; }
[ -d "$PROJ/models/Llama--Llama2-7b" ] || { echo "FATAL: model not at $PROJ/models/Llama--Llama2-7b"; exit 4; }

# dtype MUST come from metadata.json, not a hardcoded guess. The dolmino tokenize
# wrote uint32 (vocab is only 32000 so uint16 would have sufficed, but the writer
# chose uint32 and the file sizes prove it: shard_0000.npy is 400,000,000 bytes for
# 100,000,000 tokens = 4 bytes/token). Passing --data-dtype uint16 here would read
# each uint32 as two uint16s -- silently doubling the token stream and injecting
# zeros -- with no error. `auto` makes train_cast_llama.py read metadata.json.
DTYPE=$(python3 -c "import json;print(json.load(open('$DATA/metadata.json'))['dtype'])" 2>/dev/null || echo auto)
echo "dtype from metadata.json: $DTYPE"

cd "$CAST" || { echo "FATAL: cannot cd $CAST"; exit 5; }
echo "LOG=$LOG"
# ---------------------------------------------------------------------------
# MEMORY: this is tight and the numbers matter. Per-rank STATIC cost for
# Llama-2-7B (6.74e9 params) under plain DDP -- nothing is sharded:
#     fp32 master 25.1 + fp32 grads 25.1 + Adam m 25.1 + Adam v 25.1
#   + bf16 compute copy 12.6 + bool masks 6.3 + bf16 frozen teacher 12.6
#   = 131.8 GB   (L20A capacity 178.4 GB -> 46.6 GB left for activations)
# Measured peaks: 178.33 GB without checkpointing (OOM), 174.04 GB with
# checkpointing (still OOM by ~100 MiB). So activations+fragmentation eat the
# whole 46.6 GB headroom at seq-len 4096.
#
# expandable_segments MUST be exported here, not in the calling shell: torchrun
# spawns children and an env var set outside the `setsid nohup bash ...` wrapper
# does not reliably reach them. The 2-5 GB reclaimed from fragmentation is the
# difference between OOM and fitting.
#
# NOTE: fp32 master weights are NOT optional -- lambda=4e-7 is below bf16
# resolution, so the repo asserts require_fp32=True. Dropping to bf16 master
# would silently discard the entire selective-decay signal, which is the whole
# mechanism CAST is reproducing. Do not "fix" OOM that way.
# ---------------------------------------------------------------------------
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$TR" --nproc_per_node 8 cast/train_cast_llama.py \
  --project-root "$PROJ" \
  --data "$DATA" --data-dtype auto \
  --out outputs/cast_repro_ddp \
  --max-steps 7500 \
  --lr 2e-5 --l1-decay 4e-7 --global-batch 256 --seq-len 4096 \
  --mask-period 10 --scale-groups 2 --eta 0.3333333333333333 \
  --kl-temperature 1.0 --min-lr 2e-6 --warmup 375 \
  --micro-batch 1 \
  --gradient-checkpointing \
  --save-every 500 --diag-every 250 --log-every 10 \
  > "$LOG" 2>&1
