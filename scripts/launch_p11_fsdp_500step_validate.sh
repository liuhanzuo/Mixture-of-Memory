#!/usr/bin/env bash
# P11 8B FSDP — 500-step validation run (FIX 2026-05-17).
#
# Identical to the original P11 8B FSDP recipe (Llama-3-8B-Instruct, qa1+qa2+qa5
# × {1k,2k,4k}, lr=2e-5, num_slots=512, top_k=64, dual_gate, L3-summary, no L2,
# pg19_mix=0.2) EXCEPT --total_steps 500 (was 5000).
#
# Purpose: validate the FSDP optimizer-param-collection fix (commit 69b396e).
# Old P11 5000-step FSDP run had 191/255 trainable handles silently frozen
# (eval mean=26.33 vs P8 DDP 500-step 59.14). With the fix, model.parameters()
# is walked post-FSDP-wrap and all ~255 handles receive grads.
#
# Output dir: outputs/babilong_sft_phase11_fsdp_500step_validate/
# Save: every 250 steps (so we get step000250.pt + step000500.pt = final).
#
# Usage:
#   bash scripts/launch_p11_fsdp_500step_validate.sh smoke   # 10-step 2-GPU smoke
#   bash scripts/launch_p11_fsdp_500step_validate.sh full    # 500-step 8-GPU full

set -euo pipefail

cd "${WORKDIR:-/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory}"

MODE="${1:-full}"
TS="$(date +%Y%m%d_%H%M)"

if [[ "$MODE" == "smoke" ]]; then
    OUTPUT_DIR="outputs/babilong_sft_phase11_fsdp_500step_validate_smoke_${TS}"
    LOG="logs/p11_fsdp_500step_validate_smoke_${TS}.log"
    NPROC=2
    TOTAL_STEPS=10
    SAVE_INTERVAL=0
    MASTER_PORT=29551
    EXTRA_ENV="CUDA_VISIBLE_DEVICES=0,1"
else
    OUTPUT_DIR="outputs/babilong_sft_phase11_fsdp_500step_validate"
    LOG="logs/p11_fsdp_500step_validate_${TS}.log"
    NPROC=8
    TOTAL_STEPS=500
    SAVE_INTERVAL=250
    MASTER_PORT=29552
    EXTRA_ENV=""
fi

mkdir -p "$OUTPUT_DIR" logs

export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1

CMD=(
    torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT"
    scripts/train_mem_space_babilong.py
    --model_path "${MODEL_PATH:-models/Meta-Llama-3-8B-Instruct}"
    --output_dir "$OUTPUT_DIR"
    # ---- BABILong task config (matches P11 fsdp_full) ----
    --babilong_tasks qa1,qa2,qa5
    --babilong_lengths 1k,2k,4k
    --use_chat_template
    --pg19_mix_fraction 0.2
    --total_steps "$TOTAL_STEPS"
    --lr 2e-5
    # ---- mem_space hyperparameters (matches P11 fsdp_full adapter_config.json) ----
    --num_slots 512
    --top_k 64
    --selector_dim 128
    --writeback_gate_max 0.3
    --writeback_warmup_steps 1000
    --load_balance_weight 0.01
    --entropy_aux_weight 0.001
    --selector_temperature 1.0
    --key_repulsion_weight 0.05
    --key_repulsion_threshold 0.3
    --peak_routing_weight 0.05
    --slot_value_norm_cap 5.0
    --slot_init random
    --slot_init_noise 0.05
    --shared_memory_bank
    --use_dual_gate
    --forget_bias_init 2.0
    --dual_gate_tanh_new
    --use_l3_summary
    --l3_n_summary 64
    --l3_n_layers 2
    --l3_n_heads 8
    # ---- L2 disabled (P11 had use_l2: false) ----
    # (no --use_l2 flag)
    # ---- FSDP + gradient checkpointing (avoids cuBLAS workspace OOM on H20) ----
    --use_fsdp
    --gradient_checkpointing
    # ---- saving ----
    --save_interval "$SAVE_INTERVAL"
)

echo "=== P11 FSDP 500-step VALIDATE launch ($MODE) ==="
echo "Output dir: $OUTPUT_DIR"
echo "Log:        $LOG"
echo "Steps:      $TOTAL_STEPS"
echo "GPUs:       $NPROC (master_port=$MASTER_PORT)"
echo "Save every: $SAVE_INTERVAL"
echo "Commit:     $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Command:"
printf '  %s \\\n' "${CMD[@]}"
echo

if [[ -n "$EXTRA_ENV" ]]; then
    nohup env $EXTRA_ENV "${CMD[@]}" > "$LOG" 2>&1 &
else
    nohup "${CMD[@]}" > "$LOG" 2>&1 &
fi

PID=$!
echo "Launched PID=$PID"
echo "Tail log: tail -f $LOG"
echo "$PID" > "${OUTPUT_DIR}/.pid"
echo "$LOG" > "${OUTPUT_DIR}/.log_path"
