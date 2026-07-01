#!/usr/bin/env bash
# P11 DDP 8B — 500-step diagnostic run (Experiment A, 2026-05-17).
#
# Hypothesis: if a vanilla DDP run with the P8 recipe scores ~59 like the
# original P8 (mean 59.14), then FSDP itself is the regression culprit
# regardless of the optimizer-param-collection bug fixed in commit 69b396e.
# If DDP also flatlines near ~33, then the recipe / data path regressed
# between P8 and the current code; needs git archaeology.
#
# Recipe = identical to scripts/launch_p11_fsdp_500step_validate.sh EXCEPT:
#   - NO --use_fsdp (so torch DDP path is used)
#   - keep --gradient_checkpointing (per user spec; matches P8 era memory-budget)
#   - export PYTORCH_CUDA_ALLOC_CONF / CUBLAS_WORKSPACE_CONFIG to avoid the
#     cuBLAS workspace OOM that originally motivated the FSDP migration
#
# Output: outputs/babilong_sft_phase11_ddp_500step_validate/
# Save:   step000250 + step000500 (final).
#
# Usage:
#   bash scripts/launch_p11_ddp_500step_validate.sh smoke   # 10-step 2-GPU smoke
#   bash scripts/launch_p11_ddp_500step_validate.sh full    # 500-step 8-GPU full

set -euo pipefail

cd "${WORKDIR:-/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory}"

MODE="${1:-full}"
TS="$(date +%Y%m%d_%H%M)"

if [[ "$MODE" == "smoke" ]]; then
    OUTPUT_DIR="outputs/babilong_sft_phase11_ddp_500step_validate_smoke_${TS}"
    LOG="logs/p11_ddp_500step_validate_smoke_${TS}.log"
    NPROC=2
    TOTAL_STEPS=10
    SAVE_INTERVAL=0
    MASTER_PORT=29553
    EXTRA_ENV="CUDA_VISIBLE_DEVICES=0,1"
    LENGTHS_ARG="1k"
else
    OUTPUT_DIR="outputs/babilong_sft_phase11_ddp_500step_validate"
    LOG="logs/p11_ddp_500step_validate_${TS}.log"
    NPROC=8
    TOTAL_STEPS=500
    SAVE_INTERVAL=250
    MASTER_PORT=29554
    EXTRA_ENV=""
    LENGTHS_ARG="1k,2k,4k"
fi

mkdir -p "$OUTPUT_DIR" logs

export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1
# Critical: avoid the cuBLAS workspace OOM that originally drove the FSDP migration.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

CMD=(
    torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT"
    scripts/train_mem_space_babilong.py
    --model_path "${MODEL_PATH:-models/Meta-Llama-3-8B-Instruct}"
    --output_dir "$OUTPUT_DIR"
    # ---- BABILong task config (matches P11 fsdp_full / P8 baseline) ----
    --babilong_tasks qa1,qa2,qa5
    --babilong_lengths "$LENGTHS_ARG"
    --use_chat_template
    --pg19_mix_fraction 0.2
    --total_steps "$TOTAL_STEPS"
    --lr 2e-5
    # ---- mem_space hyperparameters (P8 / P11 fsdp_validate adapter_config) ----
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
    # ---- DDP (no FSDP) + gradient checkpointing ----
    # NB: --use_fsdp deliberately OMITTED → vanilla torch DDP path
    --gradient_checkpointing
    # ---- saving ----
    --save_interval "$SAVE_INTERVAL"
)

echo "=== P11 DDP 500-step VALIDATE launch ($MODE) ==="
echo "Output dir: $OUTPUT_DIR"
echo "Log:        $LOG"
echo "Steps:      $TOTAL_STEPS"
echo "GPUs:       $NPROC (master_port=$MASTER_PORT)"
echo "Save every: $SAVE_INTERVAL"
echo "Commit:     $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "ALLOC:      PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "CUBLAS:     CUBLAS_WORKSPACE_CONFIG=$CUBLAS_WORKSPACE_CONFIG"
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
