#!/bin/bash
# Wait for the HNST v2 training run to finish (process gone), then run the eval
# bundle on the LAST available checkpoint. Designed to run in the background so
# evals fire automatically when the 8-GPU training releases the cards.
set -uo pipefail
R="/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory"
cd "$R"
RUN="mem_space_hnstv2_tree_b25"
OUT="outputs/$RUN"
LOG="logs/${RUN}_autoeval.log"
echo "[autoeval] waiting for training to finish (pid pattern wandb_run_name $RUN)" | tee "$LOG"
# Wait until no training process for this run remains.
while pgrep -f "wandb_run_name $RUN" >/dev/null 2>&1; do sleep 120; done
echo "[autoeval] training process gone at $(date)" | tee -a "$LOG"
sleep 30
# Pick the highest-step checkpoint (or final full_model.pt).
CKPT=""
if [ -f "$OUT/full_model.pt" ]; then CKPT="$OUT/full_model.pt"; fi
LAST_STEP=$(ls "$OUT"/full_model_step*.pt 2>/dev/null | sort | tail -1 || true)
if [ -n "$LAST_STEP" ]; then CKPT="$LAST_STEP"; fi
if [ -z "$CKPT" ]; then echo "[autoeval] NO checkpoint found in $OUT" | tee -a "$LOG"; exit 1; fi
ACFG="$OUT/adapter_config.json"
echo "[autoeval] evaluating $CKPT" | tee -a "$LOG"
bash scripts/_eval_hnstv2_bundle.sh "$CKPT" "$ACFG" 0 "$RUN" 2>&1 | tee -a "$LOG"
echo "[autoeval] DONE $(date)" | tee -a "$LOG"
