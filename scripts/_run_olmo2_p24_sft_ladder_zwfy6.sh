#!/usr/bin/env bash
# Paper B P2.4 damage-sensitivity extension — SFT the three lower rungs
# (keep8 / keep10 / keep12) using the byte-identical recipe verified on
# .73 keep14 and shortgpt16 (both 2026-08-08 00:58 CST runs, log
# logs/p24_sft_shortgpt16.log confirmed BS=1 GA=16 eff_batch=128 max_steps=842).
#
# Called with a single ARM argument (keep8 / keep10 / keep12). Meant to be
# launched under `setsid nohup` on the zwfy6 nodes .73 / .82 / .104. Each
# invocation is one 8-GPU DDP SFT job (~40 min per arm, per PID).
#
# Recipe (identical across arms; only ckpt+output_dir vary):
#   trainer: scripts/train_olmo2_sft.py (md5 02d8b9ead6cafdf5893d6e59df6ad196)
#   data:    data/olmo2_sft/tulu3_general_clean_{input_ids,labels}.npy
#             (post-commit 0fd051a NaN fix; md5s
#              b1e6fe4e11351e208da24b03d96a762a  input_ids
#              bf7c57746f05b1ac73ccdaa07b1481b7  labels)
#   optim:   fp32 master AdamW (betas 0.9, 0.95) + bf16 autocast (NO bnb)
#   flags:   BS=1 GA=16 world=8 -> eff_batch=128, seq_len=2048, max_steps=842
#            lr=1e-5 min_lr=1e-6 warmup=100 wd=0.1 grad_clip=1.0 seed=42
#            gradient_checkpointing=1
#
# Per-arm knobs:
#   keep8   -> outputs/olmo2_probe2_7B_keep8fresh2/step121000.pt   (Table 4 headline)
#             out -> outputs/olmo2_p24_sft_keep8fresh2
#   keep10  -> outputs/olmo2_probe2_7B_keep10fresh2/step83500.pt   (Table 4 headline)
#             out -> outputs/olmo2_p24_sft_keep10fresh2
#   keep12  -> outputs/olmo2_probe2_7B_keep12fresh2/step124000.pt  (Table 4 headline)
#             out -> outputs/olmo2_p24_sft_keep12fresh2
set -euo pipefail

ARM="${1:?missing ARM (keep8 | keep10 | keep12)}"

# zwfy6 disk (H20 nodes)
ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$ROOT"

PY="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
BASE="${MODEL_PATH:-../models/OLMo-2-1124-7B}"

SFT_IDS="$ROOT/data/olmo2_sft/tulu3_general_clean_input_ids.npy"
SFT_LAB="$ROOT/data/olmo2_sft/tulu3_general_clean_labels.npy"

case "$ARM" in
  keep8)
    CKPT="$ROOT/outputs/olmo2_probe2_7B_keep8fresh2/step121000.pt"
    OUT="$ROOT/outputs/olmo2_p24_sft_keep8fresh2"
    ;;
  keep10)
    CKPT="$ROOT/outputs/olmo2_probe2_7B_keep10fresh2/step83500.pt"
    OUT="$ROOT/outputs/olmo2_p24_sft_keep10fresh2"
    ;;
  keep12)
    CKPT="$ROOT/outputs/olmo2_probe2_7B_keep12fresh2/step124000.pt"
    OUT="$ROOT/outputs/olmo2_p24_sft_keep12fresh2"
    ;;
  *)
    echo "ARM must be keep8 | keep10 | keep12, got '$ARM'" >&2
    exit 2
    ;;
esac

# preflight — fail loud if the ckpt or data is missing
[ -f "$CKPT" ]     || { echo "ERR: ckpt missing $CKPT" >&2; exit 3; }
[ -f "$SFT_IDS" ]  || { echo "ERR: sft ids missing $SFT_IDS" >&2; exit 3; }
[ -f "$SFT_LAB" ]  || { echo "ERR: sft labels missing $SFT_LAB" >&2; exit 3; }

md5_ids=$(md5sum "$SFT_IDS" | cut -d' ' -f1)
md5_lab=$(md5sum "$SFT_LAB" | cut -d' ' -f1)
[ "$md5_ids" = "b1e6fe4e11351e208da24b03d96a762a" ] || {
  echo "ERR: SFT ids md5 mismatch: got $md5_ids" >&2; exit 4; }
[ "$md5_lab" = "bf7c57746f05b1ac73ccdaa07b1481b7" ] || {
  echo "ERR: SFT labels md5 mismatch: got $md5_lab" >&2; exit 4; }

mkdir -p "$OUT" logs

export NCCL_IB_DISABLE=1
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS=1

LOG="$ROOT/logs/p24_sft_${ARM}.log"
echo "=== P2.4 SFT ladder arm=$ARM ===" | tee -a "$LOG"
echo "host=$(hostname) date=$(date -Iseconds)"     | tee -a "$LOG"
echo "ROOT=$ROOT"                                  | tee -a "$LOG"
echo "PY=$PY"                                      | tee -a "$LOG"
echo "BASE=$BASE"                                  | tee -a "$LOG"
echo "CKPT=$CKPT"                                  | tee -a "$LOG"
echo "OUT=$OUT"                                    | tee -a "$LOG"
echo "SFT_IDS=$SFT_IDS md5=$md5_ids"               | tee -a "$LOG"
echo "SFT_LAB=$SFT_LAB md5=$md5_lab"               | tee -a "$LOG"

# byte-identical to the completed keep14/shortgpt16 arms
exec "$PY" -m torch.distributed.run --standalone --nproc_per_node 8 \
  scripts/train_olmo2_sft.py \
    --base_model "$BASE" \
    --ckpt "$CKPT" \
    --arm_name "$ARM" \
    --data_mode sft --sft_ids "$SFT_IDS" --sft_labels "$SFT_LAB" \
    --output_dir "$OUT" \
    --seq_len 2048 --batch_size 1 --grad_accumulation_steps 16 \
    --max_steps 842 --lr 1e-5 --min_lr 1e-6 --warmup_steps 100 \
    --weight_decay 0.1 --grad_clip 1.0 --gradient_checkpointing 1 \
    --save_every 500 --seed 42 \
  >>"$LOG" 2>&1
