#!/bin/bash
# Orchestrator (runs on .53 via setsid): wait for BOTH unfreeze-sweep training
# runs to finish, then eval every saved checkpoint — fullchain oracle qa5 16k
# (n100, 8-way sharded) + pg19 ppl guardrail. Also evals the A-model baseline
# so the harness's strict-rescore reference (~43) is reproduced on this node.
set -uo pipefail
R="/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory"
cd "$R"
LOG="logs/unfreeze_eval/orchestrator.log"
mkdir -p logs/unfreeze_eval
exec > "$LOG" 2>&1
echo "[orch] start $(date)"

# 1) Wait for both training runs to exit.
while pgrep -f "wandb_run_name mem_space_unfreeze_arm2_top16" >/dev/null 2>&1 \
   || pgrep -f "wandb_run_name mem_space_unfreeze_arm3_full" >/dev/null 2>&1; do
  sleep 120
done
echo "[orch] both training runs finished $(date)"
sleep 30  # let final ckpt writes flush

ALL_GPUS="0,1,2,3,4,5,6,7"
AMODEL_CKPT="outputs/mem_space_fifo_b25_c512_supervised_select/full_model_step002000.pt"
AMODEL_ACFG="outputs/mem_space_fifo_b25_c512_supervised_select/adapter_config.json"

# 2) A-model baseline fullchain (harness sanity: should strict-rescore ~43).
bash scripts/_eval_unfreeze_fc16k_remote53.sh Abaseline "$AMODEL_CKPT" "$AMODEL_ACFG" "$ALL_GPUS"

# 3) Each arm: step200 + final(step400) fullchain + final ppl.
for arm in arm2_top16 arm3_full; do
  OD="outputs/mem_space_unfreeze_$arm"
  ACFG="$OD/adapter_config.json"
  for step in step000200 ""; do
    if [ -z "$step" ]; then
      CKPT="$OD/full_model.pt"; TAG="${arm}_s400"
    else
      CKPT="$OD/full_model_${step}.pt"; TAG="${arm}_s200"
    fi
    if [ -f "$CKPT" ]; then
      bash scripts/_eval_unfreeze_fc16k_remote53.sh "$TAG" "$CKPT" "$ACFG" "$ALL_GPUS"
    else
      echo "[orch] MISSING ckpt $CKPT (skip $TAG)"
    fi
  done
  # ppl guardrail on the final ckpt only.
  if [ -f "$OD/full_model.pt" ]; then
    bash scripts/_eval_unfreeze_ppl_remote53.sh "${arm}_s400" "$OD/full_model.pt" "$ACFG" 0
  fi
done

echo "[orch] ALL EVALS DONE $(date)"
# 4) Strict + lenient rescore summary.
echo "======== STRICT-RESCORE SUMMARY (fullchain oracle qa5 16k) ========"
for d in babilong_results/unfreeze_*_fc16k; do
  [ -d "$d" ] || continue
  echo -n "$(basename $d): "
  .venv/bin/python scripts/strict_rescore.py "$d/qa5_16k_fullchain_oracle_n*.csv" 2>&1 | tail -1
done
echo "======== PPL GUARDRAIL ========"
for j in babilong_results/ppl_unfreeze_*.json; do
  [ -f "$j" ] || continue
  echo -n "$(basename $j): "; grep -oE '"(avg_nll|ppl)": [0-9.]+' "$j" | tr '\n' ' '; echo
done
echo "[orch] SUMMARY DONE $(date)"
