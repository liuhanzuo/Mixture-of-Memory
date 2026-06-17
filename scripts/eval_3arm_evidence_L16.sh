#!/usr/bin/env bash
# 3-arm Slot-Routed Evidence Memory probe on niah_single_1 4k (n=50), evidence
# layer 16. Runs the three decisive arms in parallel on 3 GPUs:
#   arm1 OFF                 (slot-only readout, no evidence)            -> GPU 0
#   arm2 heuristic@L16       (routed evidence, buf64 topr64, L16)        -> GPU 1
#   arm3 in-context-oracle@L16 (STEP-1 fix: gold-span hidden captured    -> GPU 2
#        from the full-context memory-ON streaming forward, sliced at the
#        needle offset — the faithful reader ceiling)
#
# Usage:
#   CKPT=outputs/<run>/mem_space_adapter.pt ACFG=outputs/<run>/adapter_config.json \
#   TAG=<short_name> bash scripts/eval_3arm_evidence_L16.sh
set -u
RD=${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}
PYBIN=${PYTHON_BIN:-$RD/.venv/bin/python}
cd "$RD" || exit 1
export WANDB_MODE=offline

CKPT=${CKPT:-outputs/mem_space_p11_chunk1024_deltarule_normreadout/mem_space_adapter.pt}
ACFG=${ACFG:-outputs/mem_space_p11_chunk1024_deltarule_normreadout/adapter_config.json}
MODEL=${MODEL:-models/Meta-Llama-3-8B}
CHUNK=${CHUNK:-1024}
TASK=${TASK:-niah_single_1}
LEN=${LEN:-4k}
NS=${NUM_SAMPLES:-50}
EV_BUF=${EV_BUF:-64}
EV_TOPR=${EV_TOPR:-64}
EV_LAYER=${EV_LAYER:-16}
TAG=${TAG:-p11frozen}
G0=${G0:-0}; G1=${G1:-1}; G2=${G2:-2}

mkdir -p logs ruler_results
echo "[3arm] ckpt=$CKPT tag=$TAG task=$TASK len=$LEN n=$NS L=$EV_LAYER chunk=$CHUNK"

# arm1 OFF
CUDA_VISIBLE_DEVICES=$G0 $PYBIN scripts/eval_ruler_mem_space.py --model_type mem_space \
  --model_path "$MODEL" --checkpoint "$CKPT" --adapter_config "$ACFG" \
  --output_name 3arm_${TAG}_OFF --chunk_size $CHUNK --swa_eval_chunks 0 \
  --tasks "$TASK" --lengths "$LEN" --num_samples $NS \
  >logs/3arm_${TAG}_OFF.out 2>&1 &
PID0=$!

# arm2 heuristic@L16
CUDA_VISIBLE_DEVICES=$G1 $PYBIN scripts/eval_ruler_mem_space.py --model_type mem_space \
  --model_path "$MODEL" --checkpoint "$CKPT" --adapter_config "$ACFG" \
  --output_name 3arm_${TAG}_heurL${EV_LAYER} --chunk_size $CHUNK --swa_eval_chunks 0 \
  --use_slot_evidence --evidence_buffer_size $EV_BUF --evidence_topr $EV_TOPR \
  --evidence_layer $EV_LAYER \
  --tasks "$TASK" --lengths "$LEN" --num_samples $NS \
  >logs/3arm_${TAG}_heurL${EV_LAYER}.out 2>&1 &
PID1=$!

# arm3 in-context-oracle@L16 (STEP-1)
CUDA_VISIBLE_DEVICES=$G2 $PYBIN scripts/eval_ruler_mem_space.py --model_type mem_space \
  --model_path "$MODEL" --checkpoint "$CKPT" --adapter_config "$ACFG" \
  --output_name 3arm_${TAG}_oracleICL${EV_LAYER} --chunk_size $CHUNK --swa_eval_chunks 0 \
  --use_slot_evidence --evidence_buffer_size $EV_BUF --evidence_topr $EV_TOPR \
  --evidence_layer $EV_LAYER \
  --oracle_evidence --oracle_incontext --oracle_layers $EV_LAYER \
  --tasks "$TASK" --lengths "$LEN" --num_samples $NS \
  >logs/3arm_${TAG}_oracleICL${EV_LAYER}.out 2>&1 &
PID2=$!

wait $PID0 $PID1 $PID2
echo "[3arm] DONE tag=$TAG"
echo "=== OFF ==="            ; grep -E "SUMMARY|niah" logs/3arm_${TAG}_OFF.out | tail -3
echo "=== heuristic@L${EV_LAYER} ===" ; grep -E "SUMMARY|niah|oracle needle" logs/3arm_${TAG}_heurL${EV_LAYER}.out | tail -3
echo "=== in-context-oracle@L${EV_LAYER} ===" ; grep -E "SUMMARY|niah|oracle needle" logs/3arm_${TAG}_oracleICL${EV_LAYER}.out | tail -4
