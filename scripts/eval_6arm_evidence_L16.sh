#!/usr/bin/env bash
# 6-arm Slot-Routed Evidence Memory probe on niah_single_1 4k (n=50), evidence
# layer 16. ONE invocation, 6 GPUs, no double-fire. Six cells =
#   {OFF, heuristic@L16, in-context-oracle@L16} x {isolate_softmax OFF, ON}.
#
# GPU map (non-overlapping):
#   iso-OFF: GPU0 OFF, GPU1 heur, GPU2 oracle
#   iso-ON : GPU3 OFF, GPU4 heur, GPU5 oracle
# (OFF arm has no evidence so isolate_softmax is a no-op there, but we run both
#  so the table is symmetric and the OFF cells confirm reproducibility.)
#
# Usage:
#   CKPT=outputs/<run>/mem_space_adapter.pt ACFG=outputs/<run>/adapter_config.json \
#   TAG=<short> PROJECT_ROOT=<root> PYTHON_BIN=<py> bash scripts/eval_6arm_evidence_L16.sh
set -u
RD=${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}
PYBIN=${PYTHON_BIN:-$RD/.venv/bin/python}
cd "$RD" || exit 1
export WANDB_MODE=offline HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

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
TAG=${TAG:-p11frz}

mkdir -p logs ruler_results
echo "[6arm] ckpt=$CKPT tag=$TAG task=$TASK len=$LEN n=$NS L=$EV_LAYER chunk=$CHUNK"

PIDS=()

run_cell () {  # $1=gpu $2=cellname $3...=extra args
  local gpu="$1"; local name="$2"; shift 2
  CUDA_VISIBLE_DEVICES="$gpu" $PYBIN scripts/eval_ruler_mem_space.py --model_type mem_space \
    --model_path "$MODEL" --checkpoint "$CKPT" --adapter_config "$ACFG" \
    --output_name 6arm_${TAG}_${name} --chunk_size $CHUNK --swa_eval_chunks 0 \
    --tasks "$TASK" --lengths "$LEN" --num_samples $NS "$@" \
    >logs/6arm_${TAG}_${name}.out 2>&1 &
  PIDS+=($!)
  echo "  launched $name on GPU$gpu pid=${PIDS[-1]}"
}

EV="--use_slot_evidence --evidence_buffer_size $EV_BUF --evidence_topr $EV_TOPR --evidence_layer $EV_LAYER"
ORACLE="--oracle_evidence --oracle_incontext --oracle_layers $EV_LAYER"
ISO="--evidence_isolate_softmax"

# iso-OFF row (GPU 0/1/2)
run_cell 0 OFF_isoOFF
run_cell 1 heur_isoOFF $EV
run_cell 2 oracle_isoOFF $EV $ORACLE
# iso-ON row (GPU 3/4/5)
run_cell 3 OFF_isoON
run_cell 4 heur_isoON $EV $ISO
run_cell 5 oracle_isoON $EV $ISO $ORACLE

wait "${PIDS[@]}"
echo "[6arm] DONE tag=$TAG"

echo "============ 6-ARM TABLE (tag=$TAG) ============"
for name in OFF_isoOFF heur_isoOFF oracle_isoOFF OFF_isoON heur_isoON oracle_isoON; do
  echo "--- $name ---"
  grep -E "SUMMARY|niah_single_1|oracle needle|inject.*pos|EVIDENCE ON|ORACLE" logs/6arm_${TAG}_${name}.out | tail -4
done
