#!/usr/bin/env bash
# Phase-1B v4: L1 (mem_space) + L2 (token-compressed shared KV, NSA/V4-CSA) + L3 (Q-Former pool)
# on Llama-3.2-1B-Instruct. Reproduces v2 (L1+L3) flags + adds L2.
#
# v2 baseline: outputs/babilong_sft_phase1b_v2_10k (21-cell BABILong mean=37.29)
# Diminishing returns confirmed past 5k steps on v2 → run 5000 steps here.
#
# Usage:
#   bash scripts/launch_phase1b_v4_l1l2l3.sh smoke   # 5-step single-GPU smoke
#   bash scripts/launch_phase1b_v4_l1l2l3.sh full    # 5000-step 8×GPU full run

set -euo pipefail

cd /apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory

MODE="${1:-full}"
TS="$(date +%Y%m%d_%H%M)"

if [[ "$MODE" == "smoke" ]]; then
    OUTPUT_DIR="outputs/babilong_sft_phase1b_v4_smoke"
    LOG="logs/phase1b_v4_smoke_${TS}.log"
    NPROC=1
    TOTAL_STEPS=5
    SAVE_INTERVAL=0
    MASTER_PORT=29541
    EXTRA_ENV="CUDA_VISIBLE_DEVICES=0"
else
    OUTPUT_DIR="outputs/babilong_sft_phase1b_v4_l1l2l3"
    LOG="logs/phase1b_v4_${TS}.log"
    NPROC=8
    TOTAL_STEPS=5000
    SAVE_INTERVAL=500
    MASTER_PORT=29542
    EXTRA_ENV=""
fi

mkdir -p "$OUTPUT_DIR" logs

export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1

CMD=(
    torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT"
    scripts/train_mem_space_babilong.py
    --model_path models/Llama-3.2-1B-Instruct
    --output_dir "$OUTPUT_DIR"
    # ---- v2 flags (L1 + L3 baseline) ----
    --babilong_tasks qa1,qa2,qa5
    --babilong_lengths 1k,2k,4k
    --use_chat_template
    --pg19_mix_fraction 0.2
    --total_steps "$TOTAL_STEPS"
    --lr 2e-5
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
    # ---- v4 NEW: L2 token-compressed shared KV ----
    --use_l2
    --l2_compress_ratio 16
    --l2_d_c 512
    --l2_d_h_rope 64
    --l2_init_scale 0.001
    # ---- saving ----
    --save_interval "$SAVE_INTERVAL"
)

echo "=== Phase-1B v4 launch ($MODE) ==="
echo "Output dir: $OUTPUT_DIR"
echo "Log:        $LOG"
echo "Steps:      $TOTAL_STEPS"
echo "GPUs:       $NPROC"
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
