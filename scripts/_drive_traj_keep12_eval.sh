#!/usr/bin/env bash
# Paper B P0.4 optional hardening: keep12 (14L shell, KF=12 NF=2) dense-trajectory
# per-item MMLU driver. Runs the KF-parameterised 8-GPU per-item harness serially
# over a list of steps, each producing olmo2_downstream_results/traj_keep12_step{S}/
# with a merged per_example_mmlu.jsonl (item_id = shard + local*num_shards aligned).
# env in: ROOT (node root), PY (python), STEPS (space-sep list of step numbers)
set -u
ROOT="${ROOT:?set ROOT}"
PY="${PY:-/opt/conda/envs/torch-base/bin/python}"
STEPS="${STEPS:?set STEPS}"
BASE="${BASE:-../models/OLMo-2-1124-7B}"
CKDIR="$ROOT/outputs/olmo2_probe2_7B_keep12fresh2"
cd "$ROOT"
for S in $STEPS; do
  CKPT="$CKDIR/step${S}.pt"
  if [ ! -f "$CKPT" ]; then
    echo "[$(date '+%F %T')] SKIP step${S}: ckpt not on disk ($CKPT)"; continue
  fi
  NAME="traj_keep12_step${S}"
  echo "[$(date '+%F %T')] === EVAL step${S} -> $NAME ==="
  ROOT="$ROOT" NAME="$NAME" BASE="$BASE" CKPT="$CKPT" KF=12 NF=2 PY="$PY" \
    bash scripts/_run_olmo2_mmlu_peritem_kf_8gpu.sh
  echo "[$(date '+%F %T')] === step${S} done ==="
done
echo "ALL_TRAJ_KEEP12_EVAL_DONE"
