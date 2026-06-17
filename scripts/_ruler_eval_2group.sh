#!/usr/bin/env bash
# RULER full eval launcher — runs on a single 8-GPU node (.76 diskB).
# Splits work across 8 GPUs by (model, task, length) cell; each cell n=50.
# GPUs 0-3 -> mem_space SOTA ckpt ; GPUs 4-7 -> base Llama-3-8B.
# Within each 4-GPU group, cells are popped from a shared flock'd pool.
set -u

RD=${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}
PYBIN=${PYTHON_BIN:-$RD/.venv/bin/python}
cd "$RD" || exit 1
export WANDB_MODE=offline

CKPT=${CKPT:-outputs/mem_space_p11_chunk1024_deltarule_normreadout/mem_space_adapter.pt}
ACFG=outputs/mem_space_p11_chunk1024_deltarule_normreadout/adapter_config.json
MODEL=models/Meta-Llama-3-8B
TASKS=(niah_single_1 niah_single_2 niah_multikey_1 variable_tracking)
LENGTHS=(4k 8k 16k 32k)
NS=${NUM_SAMPLES:-50}

mkdir -p logs ruler_results
POOL_MEM=$(mktemp)
POOL_BASE=$(mktemp)
for t in "${TASKS[@]}"; do for L in "${LENGTHS[@]}"; do
  echo "$t $L" >> "$POOL_MEM"; echo "$t $L" >> "$POOL_BASE";
done; done
LOCK_MEM=$(mktemp); LOCK_BASE=$(mktemp)

pop() { # $1=pool $2=lock  -> prints one line and removes it, atomically
  local line
  exec 9>"$2"; flock 9
  line=$(head -n1 "$1")
  if [ -n "$line" ]; then sed -i '1d' "$1"; fi
  flock -u 9
  printf '%s' "$line"
}

worker_mem() { # $1=gpu
  local gpu=$1 cell t L
  while true; do
    cell=$(pop "$POOL_MEM" "$LOCK_MEM"); [ -z "$cell" ] && break
    t=${cell% *}; L=${cell#* }
    CUDA_VISIBLE_DEVICES=$gpu $PYBIN scripts/eval_ruler_mem_space.py --model_type mem_space \
      --model_path $MODEL --checkpoint $CKPT --adapter_config $ACFG \
      --output_name ruler_p11_c1024_final --chunk_size 1024 --swa_eval_chunks 0 \
      --tasks "$t" --lengths "$L" --num_samples $NS \
      >>logs/ruler_mem_gpu${gpu}.log 2>&1
  done
}
worker_base() { # $1=gpu
  local gpu=$1 cell t L
  while true; do
    cell=$(pop "$POOL_BASE" "$LOCK_BASE"); [ -z "$cell" ] && break
    t=${cell% *}; L=${cell#* }
    CUDA_VISIBLE_DEVICES=$gpu $PYBIN scripts/eval_ruler_mem_space.py --model_type base \
      --model_path $MODEL --base_max_window 8192 \
      --output_name ruler_base --num_samples $NS \
      --tasks "$t" --lengths "$L" \
      >>logs/ruler_base_gpu${gpu}.log 2>&1
  done
}

for g in 0 1 2 3; do worker_mem $g & done
for g in 4 5 6 7; do worker_base $g & done
wait
echo "RULER_ALL_DONE"
