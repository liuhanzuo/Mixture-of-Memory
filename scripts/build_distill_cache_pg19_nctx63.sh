#!/usr/bin/env bash
# Build the n_ctx=63 PG19 teacher distill cache (group_len=(63+1)*512=32768=32k) to
# push the teacher window 4x past nctx15 (8192) and attack the 32k long-range wall.
# nctx15 already broke the 16k ceiling (16k babilong qa5 W0 = 16 vs prior 13); the
# remaining gap is 32k where the effective teacher window (8192) << 32k inference.
#
# Builder runs INDEPENDENT per-rank processes (no NCCL/init_process_group); shards
# groups round-robin by RANK modulo WORLD_SIZE, each rank writes its own .npz to the
# shared diskA FS. rank0 writes meta.json. ★NO --max_groups (full build) — adding it
# causes shuffle-loader 99% cache miss + silent distill failure.
#
# USAGE:
#   Single node (8 GPU):  RANK_BASE=0 WORLD_SIZE=8 bash scripts/build_distill_cache_pg19_nctx63.sh
#   Two nodes (16 GPU, shared diskA FS):
#     local:  RANK_BASE=0  WORLD_SIZE=16 bash scripts/build_distill_cache_pg19_nctx63.sh
#     .196:   RANK_BASE=8  WORLD_SIZE=16 bash scripts/build_distill_cache_pg19_nctx63.sh
set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
PYBIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
RANK_BASE="${RANK_BASE:-0}"
WORLD_SIZE="${WORLD_SIZE:-8}"
OUT_DIR="${OUT_DIR:-distill_cache/pg19_512_nctx63}"
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
      --chunk_size 512 --n_ctx 63 --distill_layers 12,20,28 --topk 64 \
      --out_dir "$OUT_DIR" \
      --rank "$R" --world_size "$WORLD_SIZE" --local_rank "$L" --attn_impl sdpa \
      </dev/null >"logs/build_pg19_nctx63_rank${R}.log" 2>&1 &
  disown
done
echo "launched 8 ranks (base=$RANK_BASE world=$WORLD_SIZE) -> $OUT_DIR"
