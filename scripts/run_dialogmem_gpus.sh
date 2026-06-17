#!/usr/bin/env bash
# mem_space dialog-memory eval on a chosen set of GPUs (default 4-7).
# Env: BENCH DATA MAXSAMPLES CHUNK TAG GPUS(space sep) OUTROOT
set -u
PROJECT_ROOT=${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}
cd "$PROJECT_ROOT"
export WANDB_MODE=offline

BENCH=${BENCH:-longmemeval}
DATA=${DATA:-data/dialogmem/longmemeval/longmemeval_oracle}
MAXSAMPLES=${MAXSAMPLES:-100}
CHUNK=${CHUNK:-1024}
MODEL=${MODEL:-models/Meta-Llama-3-8B}
CKPT=${CKPT:-outputs/mem_space_p11_chunk1024_deltarule_normreadout/mem_space_adapter.pt}
ADAPTER=${ADAPTER:-outputs/mem_space_p11_chunk1024_deltarule_normreadout/adapter_config.json}
TAG=${TAG:-lme_oracle_p11}
MODE=${MODE:-mem}     # mem | base
GPUS=${GPUS:-4 5 6 7}
OUTROOT=${OUTROOT:-dialogmem_results}

gpu_arr=($GPUS)
NSH=${#gpu_arr[@]}
OUT=$OUTROOT/${TAG}_${MODE}
mkdir -p logs "$OUT"
echo "[run] $BENCH $MODE tag=$TAG gpus=$GPUS nshards=$NSH out=$OUT"

idx=0
for gpu in "${gpu_arr[@]}"; do
  if [ "$MODE" = "base" ]; then
    CUDA_VISIBLE_DEVICES=$gpu setsid nohup $PYTHON_BIN scripts/eval_dialogmem_mem_space.py \
      --base_mode --benchmark "$BENCH" --data "$DATA" --model_path "$MODEL" \
      --output_dir "$OUT" --max_samples "$MAXSAMPLES" \
      --num_shards $NSH --shard_index $idx \
      > logs/dialogmem_${TAG}_${MODE}_g${gpu}.out 2>&1 &
  else
    CUDA_VISIBLE_DEVICES=$gpu setsid nohup $PYTHON_BIN scripts/eval_dialogmem_mem_space.py \
      --benchmark "$BENCH" --data "$DATA" --model_path "$MODEL" \
      --checkpoint "$CKPT" --adapter_config "$ADAPTER" \
      --output_dir "$OUT" --chunk_size "$CHUNK" --max_samples "$MAXSAMPLES" \
      --num_shards $NSH --shard_index $idx \
      > logs/dialogmem_${TAG}_${MODE}_g${gpu}.out 2>&1 &
  fi
  idx=$((idx+1))
done
echo "[run] launched $idx shards"
