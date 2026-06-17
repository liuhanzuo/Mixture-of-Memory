#!/usr/bin/env bash
# Consolidated 6-arm Slot-Routed Evidence probe on niah_single_1 4k (n=50),
# evidence layer 16, with the Landmark position fix. Runs BOTH the non-isolated
# and isolated-EV-softmax variants on DISJOINT GPUs so nothing collides:
#   GPU 0  arm1 OFF                        (slot-only, no EV)
#   GPU 1  arm2 heuristic@L16 realpos
#   GPU 2  arm3 in-context-oracle@L16 realpos
#   GPU 3  arm4 heuristic@L16 realpos + ISOLATE softmax
#   GPU 4  arm5 in-context-oracle@L16 realpos + ISOLATE softmax
# (OFF needs no EV so there is no iso variant of it.)
#
# Usage: CKPT=... ACFG=... TAG=... bash scripts/eval_6arm_evidence_L16.sh
set -u
RD=${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}
PYBIN=${PYTHON_BIN:-$RD/.venv/bin/python}
cd "$RD" || exit 1
export WANDB_MODE=offline HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="$RD/third_party/babilong-pkg:$RD:${PYTHONPATH:-}"

CKPT=${CKPT:-outputs/mem_space_p11_chunk1024_deltarule_normreadout/mem_space_adapter.pt}
ACFG=${ACFG:-outputs/mem_space_p11_chunk1024_deltarule_normreadout/adapter_config.json}
MODEL=${MODEL:-models/Meta-Llama-3-8B}
CHUNK=${CHUNK:-1024}
TASK=${TASK:-niah_single_1}
LEN=${LEN:-4k}
NS=${NUM_SAMPLES:-50}
EV_BUF=${EV_BUF:-64}; EV_TOPR=${EV_TOPR:-64}; EV_LAYER=${EV_LAYER:-16}
TAG=${TAG:-p11frozen}

mkdir -p logs ruler_results
echo "[6arm] ckpt=$CKPT tag=$TAG task=$TASK len=$LEN n=$NS L=$EV_LAYER chunk=$CHUNK"

run_arm() {  # $1=gpu $2=outname $3...=extra args
  local gpu=$1 name=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu $PYBIN scripts/eval_ruler_mem_space.py --model_type mem_space \
    --model_path "$MODEL" --checkpoint "$CKPT" --adapter_config "$ACFG" \
    --output_name "$name" --chunk_size $CHUNK --swa_eval_chunks 0 \
    --tasks "$TASK" --lengths "$LEN" --num_samples $NS "$@" \
    >logs/${name}.out 2>&1 &
}

EVA="--use_slot_evidence --evidence_buffer_size $EV_BUF --evidence_topr $EV_TOPR --evidence_layer $EV_LAYER"
ORA="--oracle_evidence --oracle_incontext --oracle_layers $EV_LAYER"

run_arm 0 6arm_${TAG}_OFF
run_arm 1 6arm_${TAG}_heurL${EV_LAYER}            $EVA
run_arm 2 6arm_${TAG}_oracleICL${EV_LAYER}        $EVA $ORA
run_arm 3 6arm_${TAG}_heurL${EV_LAYER}_iso        $EVA --evidence_isolate_softmax
run_arm 4 6arm_${TAG}_oracleICL${EV_LAYER}_iso    $EVA $ORA --evidence_isolate_softmax
wait
echo "[6arm] DONE tag=$TAG"
for n in OFF heurL${EV_LAYER} oracleICL${EV_LAYER} heurL${EV_LAYER}_iso oracleICL${EV_LAYER}_iso; do
  echo "=== $n ==="; grep -E "score=|oracle needle" logs/6arm_${TAG}_${n}.out | tail -2
done
