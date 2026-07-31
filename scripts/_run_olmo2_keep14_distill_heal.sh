#!/usr/bin/env bash
# keep14-distill heal: OLMo-2-7B base 32L teacher → keep14 (16L) student.
# loss = NTP + λ·KL(teacher||student) on top-k=64 logits. Same Dolmino/LR/200k as keep14-NTP.
# Online teacher forward (32L frozen, no_grad). B200 183GB fits teacher+student+fp32master.
# DRY by default; RUN=1 launches 8-GPU heal.
set -u
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/apdcephfs_wzc1/share_304376610/pighzliu_code/models/OLMo-2-1124-7B}"
TEACHER_PATH="${TEACHER_PATH:-/apdcephfs_wzc1/share_304376610/pighzliu_code/models/OLMo-2-1124-7B}"
DATA_PATH="${DATA_PATH:-/dev/shm/dolmino_now15b.npy}"
OUT_DIR="${OUT_DIR:-outputs/olmo2_probe2_7B_keep14fresh2_distill}"
LOG_FILE="${LOG_FILE:-logs/olmo2_7B_keep14_distill.log}"
NPROC="${NPROC:-8}"
BS="${BS:-16}"
GA="${GA:-1}"
RESUME_FROM="${RESUME_FROM:-}"

mkdir -p "$OUT_DIR" logs

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CMD=(
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node "$NPROC"
  scripts/train_olmo2_arch_probe2_distill.py
    --data_path "$DATA_PATH"
    --output_dir "$OUT_DIR"
    --model_path "$MODEL_PATH"
    --keep_front_layers 14
    --n_fresh_layers 2
    --distill_teacher_model "$TEACHER_PATH"
    --distill_lambda 0.6
    --teacher_topk 64
    --lr 1e-4
    --lr_inherited 2e-5
    --min_lr 1e-5
    --min_lr_inherited 2e-6
    --batch_size "$BS"
    --grad_accumulation_steps "$GA"
    --seq_len 2048
    --max_steps 200000
    --warmup_steps 150
    --weight_decay 0.1
    --grad_clip 1.0
    --save_every 5000
    --gradient_checkpointing 1
)
[ -n "$RESUME_FROM" ] && CMD+=(--resume_from "$RESUME_FROM")

echo "[_run_olmo2_keep14_distill_heal] teacher=$TEACHER_PATH lambda=0.6 topk=64 eff_bs=$((BS*GA*NPROC))"
echo "----- launch command -----"
printf '  %q' "${CMD[@]}"; echo
echo "--------------------------"

if [ "${RUN:-0}" != "1" ]; then
  echo "[_run_olmo2_keep14_distill_heal] DRY RUN (set RUN=1 to launch)."
  exit 0
fi

[ -z "$RESUME_FROM" ] && : > "$LOG_FILE"
echo "[_run_olmo2_keep14_distill_heal] LAUNCHING 8-GPU distill heal ..."
nohup "${CMD[@]}" >>"$LOG_FILE" 2>&1 &
echo "[_run_olmo2_keep14_distill_heal] launched pid=$! ; tail -f $LOG_FILE"
