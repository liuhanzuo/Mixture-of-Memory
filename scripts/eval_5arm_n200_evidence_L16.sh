#!/usr/bin/env bash
# Decisive n=200 5-arm evidence probe on niah_single_1 4k, frozen p11, L16.
# Resolves the heuristic +28-vs-+6 contradiction between the two n=50 runs by
# 4x-ing samples (noise ~halves to ~3pt). Evidence arms use isolate_softmax ON
# (>= OFF on oracle in the n=50 run); the only swept variable is pos0 vs realpos.
#
# 5 arms (same sample set, seed fixed):
#   OFF                       -> GPU0   slot-only readout, no evidence
#   heur@L16 pos0             -> GPU1   routed evidence, legacy pos-0 injection
#   heur@L16 realpos          -> GPU2   routed evidence, real source RoPE pos
#   oracle@L16 pos0           -> GPU3   gold-span injected at pos-0
#   oracle@L16 realpos        -> GPU4   gold-span injected at real in-chunk pos
#
# Usage:
#   CKPT=... ACFG=... MODEL=... TAG=... bash scripts/eval_5arm_n200_evidence_L16.sh
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
NS=${NUM_SAMPLES:-200}
EV_BUF=${EV_BUF:-64}
EV_TOPR=${EV_TOPR:-64}
EV_LAYER=${EV_LAYER:-16}
SEED=${SEED:-42}
TAG=${TAG:-p11frz_n200}

mkdir -p logs ruler_results
echo "[5arm-n200] ckpt=$CKPT tag=$TAG task=$TASK len=$LEN n=$NS L=$EV_LAYER chunk=$CHUNK seed=$SEED"

PIDS=()
run_cell () {  # $1=gpu $2=name $3...=extra args
  local gpu="$1"; local name="$2"; shift 2
  CUDA_VISIBLE_DEVICES="$gpu" $PYBIN scripts/eval_ruler_mem_space.py --model_type mem_space \
    --model_path "$MODEL" --checkpoint "$CKPT" --adapter_config "$ACFG" \
    --output_name 5arm_${TAG}_${name} --chunk_size $CHUNK --swa_eval_chunks 0 \
    --seed $SEED --tasks "$TASK" --lengths "$LEN" --num_samples $NS "$@" \
    >logs/5arm_${TAG}_${name}.out 2>&1 &
  PIDS+=($!)
  echo "  launched $name on GPU$gpu pid=${PIDS[-1]}"
}

EV="--use_slot_evidence --evidence_buffer_size $EV_BUF --evidence_topr $EV_TOPR --evidence_layer $EV_LAYER --evidence_isolate_softmax"
ORACLE="--oracle_evidence --oracle_incontext --oracle_layers $EV_LAYER"

run_cell 0 OFF
run_cell 1 heur_pos0   $EV --evidence_pos0
run_cell 2 heur_realpos $EV
run_cell 3 oracle_pos0   $EV $ORACLE --evidence_pos0
run_cell 4 oracle_realpos $EV $ORACLE

wait "${PIDS[@]}"
echo "[5arm-n200] DONE tag=$TAG"
echo "============ 5-ARM n=$NS TABLE (tag=$TAG, $TASK $LEN) ============"
for name in OFF heur_pos0 heur_realpos oracle_pos0 oracle_realpos; do
  printf "%-18s " "$name"
  grep -oE "score=[0-9.]+" logs/5arm_${TAG}_${name}.out 2>/dev/null | tail -1
done
