#!/usr/bin/env bash
# Build the n_ctx=15 PG19 teacher distill cache (group_len=(15+1)*512=8192) to
# enable the next frontier arm: longer training window (8192 vs nctx7's 4096) to
# attack the 32k long-range wall (pg19 final W0 qa5 16k=16 broke the 13 ceiling,
# but 32k=9 only tied — root cause: nctx7 effective window 4096 << 32k inference).
#
# The builder runs INDEPENDENT per-rank processes (no NCCL/init_process_group);
# it shards groups round-robin by RANK modulo WORLD_SIZE and each rank writes its
# own .npz. So we can fan out across the shared diskA FS (local + 28.59.80.196 =
# 16 GPUs) into one out_dir. rank0 writes meta.json.
#
# USAGE (per node):
#   RANK_BASE=0  WORLD_SIZE=16 bash scripts/build_distill_cache_pg19_nctx15.sh   # local: ranks 0-7
#   RANK_BASE=8  WORLD_SIZE=16 bash scripts/build_distill_cache_pg19_nctx15.sh   # .196:  ranks 8-15
# Single-node (8 GPU) fallback: RANK_BASE=0 WORLD_SIZE=8 bash ...
set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
PYBIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
RANK_BASE="${RANK_BASE:-0}"
WORLD_SIZE="${WORLD_SIZE:-8}"
OUT_DIR="${OUT_DIR:-distill_cache/pg19_512_nctx15}"
DATA="${DATA:-MemLong/data/processed/pg19_perbook_min8k/train}"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p logs
for L in 0 1 2 3 4 5 6 7; do
  R=$((RANK_BASE + L))
  RANK=$R WORLD_SIZE=$WORLD_SIZE LOCAL_RANK=$L \
    setsid nohup "$PYBIN" scripts/build_distill_cache.py \
      --dolmino_path "$DATA" \
      --model_path models/Meta-Llama-3-8B \
      --chunk_size 512 --n_ctx 15 --distill_layers 12,20,28 --topk 64 \
      --out_dir "$OUT_DIR" \
      --rank "$R" --world_size "$WORLD_SIZE" --local_rank "$L" --attn_impl sdpa \
      </dev/null >"logs/build_pg19_nctx15_rank${R}.log" 2>&1 &
done
echo "launched 8 ranks (base=$RANK_BASE world=$WORLD_SIZE) -> $OUT_DIR"
