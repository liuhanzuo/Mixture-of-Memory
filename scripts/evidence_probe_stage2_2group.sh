#!/usr/bin/env bash
# STAGE 2 — Slot-Routed Evidence Memory: zero-training ON-vs-OFF RULER NIAH probe.
# Same converged P11-INSTRUCT mem_space adapter loaded BOTH ways:
#   GPU 0-3  -> evidence OFF (baseline P11 readout)
#   GPU 4-7  -> evidence ON  (--use_slot_evidence, large budget)
# Tasks = single-needle exact recall (niah_single_1 noise, niah_single_2 prose).
# Within each 4-GPU group cells are popped from a shared flock'd pool (dynamic).
set -u

RD=${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}
PYBIN=${PYTHON_BIN:-$RD/.venv/bin/python}
cd "$RD" || exit 1
export WANDB_MODE=offline

CKPT=${CKPT:-outputs/mem_space_p11_chunk512_INSTRUCT/mem_space_adapter.pt}
ACFG=${ACFG:-outputs/mem_space_p11_chunk512_INSTRUCT/adapter_config.json}
MODEL=${MODEL:-models/Meta-Llama-3-8B-Instruct}
CHUNK=${CHUNK:-512}
TASKS=(${TASKS:-niah_single_1 niah_single_2})
LENGTHS=(${LENGTHS:-4k 8k})
NS=${NUM_SAMPLES:-50}
EV_BUF=${EV_BUF:-64}
EV_TOPR=${EV_TOPR:-64}
EV_LAYER=${EV_LAYER:-0}

mkdir -p logs ruler_results
POOL_OFF=$(mktemp); POOL_ON=$(mktemp)
for t in "${TASKS[@]}"; do for L in "${LENGTHS[@]}"; do
  echo "$t $L" >> "$POOL_OFF"; echo "$t $L" >> "$POOL_ON";
done; done
LOCK_OFF=$(mktemp); LOCK_ON=$(mktemp)

pop() { local line; exec 9>"$2"; flock 9
  line=$(head -n1 "$1"); [ -n "$line" ] && sed -i '1d' "$1"
  flock -u 9; printf '%s' "$line"; }

worker_off() { local gpu=$1 cell t L
  while true; do
    cell=$(pop "$POOL_OFF" "$LOCK_OFF"); [ -z "$cell" ] && break
    t=${cell% *}; L=${cell#* }
    CUDA_VISIBLE_DEVICES=$gpu $PYBIN scripts/eval_ruler_mem_space.py --model_type mem_space \
      --model_path $MODEL --checkpoint $CKPT --adapter_config $ACFG \
      --output_name ruler_evidenceOFF_p11inst --chunk_size $CHUNK --swa_eval_chunks 0 \
      --tasks "$t" --lengths "$L" --num_samples $NS \
      >>logs/stage2_evOFF_gpu${gpu}.log 2>&1
  done; }

worker_on() { local gpu=$1 cell t L
  while true; do
    cell=$(pop "$POOL_ON" "$LOCK_ON"); [ -z "$cell" ] && break
    t=${cell% *}; L=${cell#* }
    CUDA_VISIBLE_DEVICES=$gpu $PYBIN scripts/eval_ruler_mem_space.py --model_type mem_space \
      --model_path $MODEL --checkpoint $CKPT --adapter_config $ACFG \
      --output_name ruler_evidenceON_p11inst --chunk_size $CHUNK --swa_eval_chunks 0 \
      --use_slot_evidence --evidence_buffer_size $EV_BUF --evidence_topr $EV_TOPR \
      --evidence_layer $EV_LAYER \
      --tasks "$t" --lengths "$L" --num_samples $NS \
      >>logs/stage2_evON_gpu${gpu}.log 2>&1
  done; }

for g in 0 1 2 3; do worker_off $g & done
for g in 4 5 6 7; do worker_on $g & done
wait
echo "STAGE2 DONE"
