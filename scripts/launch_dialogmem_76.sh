#!/usr/bin/env bash
# Dialog-memory eval launcher for node .76 (diskB, 8x H20, .venv).
# Runs mem_space (W0) on GPU0-3 and base Llama on GPU4-7, 4-way sharded each.
# Usage: BENCH=longmemeval DATA=... MAXSAMPLES=100 bash scripts/launch_dialogmem_76.sh
set -u

PROJECT_ROOT=${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}
cd "$PROJECT_ROOT"

BENCH=${BENCH:-longmemeval}
DATA=${DATA:-data/dialogmem/longmemeval/longmemeval_oracle}
MAXSAMPLES=${MAXSAMPLES:-100}
CHUNK=${CHUNK:-1024}
MODEL=${MODEL:-models/Meta-Llama-3-8B}
CKPT=${CKPT:-outputs/mem_space_p11_chunk1024_deltarule_normreadout/mem_space_adapter.pt}
ADAPTER=${ADAPTER:-outputs/mem_space_p11_chunk1024_deltarule_normreadout/adapter_config.json}
TAG=${TAG:-${BENCH}_p11}
OUTROOT=${OUTROOT:-dialogmem_results}

MEM_OUT=$OUTROOT/${TAG}_mem
BASE_OUT=$OUTROOT/${TAG}_base
mkdir -p logs "$MEM_OUT" "$BASE_OUT"

echo "[launch] bench=$BENCH data=$DATA maxsamples=$MAXSAMPLES chunk=$CHUNK"
echo "[launch] mem_out=$MEM_OUT base_out=$BASE_OUT"

# mem_space on GPU 0-3 (4 shards)
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i nohup $PYTHON_BIN scripts/eval_dialogmem_mem_space.py \
    --benchmark "$BENCH" --data "$DATA" \
    --model_path "$MODEL" --checkpoint "$CKPT" --adapter_config "$ADAPTER" \
    --output_dir "$MEM_OUT" --chunk_size "$CHUNK" --max_samples "$MAXSAMPLES" \
    --num_shards 4 --shard_index $i \
    > logs/dialogmem_${TAG}_mem_shard${i}.out 2>&1 &
done

# base Llama on GPU 4-7 (4 shards)
for i in 0 1 2 3; do
  gpu=$((i+4))
  CUDA_VISIBLE_DEVICES=$gpu nohup $PYTHON_BIN scripts/eval_dialogmem_mem_space.py \
    --base_mode --benchmark "$BENCH" --data "$DATA" \
    --model_path "$MODEL" \
    --output_dir "$BASE_OUT" --max_samples "$MAXSAMPLES" \
    --num_shards 4 --shard_index $i \
    > logs/dialogmem_${TAG}_base_shard${i}.out 2>&1 &
done

wait
echo "[launch] all shards finished, scoring..."
$PYTHON_BIN scripts/eval_dialogmem_mem_space.py --score_only --benchmark "$BENCH" --output_dir "$MEM_OUT"
$PYTHON_BIN scripts/eval_dialogmem_mem_space.py --score_only --benchmark "$BENCH" --output_dir "$BASE_OUT"
echo "[launch] DONE"
