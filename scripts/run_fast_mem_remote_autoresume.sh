#!/bin/bash
# FastMem v2 on remote H20 — auto-resume on crash
# FLA Triton kernel triggers NVLink errors every ~2h on this node.
# Strategy: save every 500 steps, auto-restart from latest checkpoint on crash.

cd /apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory
export PYTHONUNBUFFERED=1
export http_proxy="http://hy-proxy.woa.com:3128"
export https_proxy="http://hy-proxy.woa.com:3128"
export all_proxy="http://hy-proxy.woa.com:3128"
export WANDB_API_KEY="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"
export PYTHONPATH="/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory:/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory/third_party/babilong-pkg"
export TOKENIZERS_PARALLELISM=false

# NCCL settings — force all communication through TCP/bond1 (bypass NVLink entirely)
export NCCL_SOCKET_IFNAME=bond1
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=1800000
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_NET=Socket
export CUDA_P2P_DISABLE=1

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

CKPT_DIR="outputs/fast_mem_v1"
WANDB_ID_FILE="${CKPT_DIR}/.wandb_run_id"
mkdir -p "$CKPT_DIR" logs

# Auto-resume loop: find latest checkpoint and restart from it
while true; do
    # Find latest checkpoint
    LATEST_CKPT=$(ls -t ${CKPT_DIR}/mem_space_adapter_step*.pt 2>/dev/null | head -1)
    START_STEP=0

    if [ -n "$LATEST_CKPT" ]; then
        # Extract step number from filename: mem_space_adapter_step001000.pt → 1000
        START_STEP=$(echo "$LATEST_CKPT" | grep -oP 'step\K[0-9]+' | sed 's/^0*//' )
        [ -z "$START_STEP" ] && START_STEP=0
        echo "[$(date)] Resuming from checkpoint: $LATEST_CKPT (step $START_STEP)"
        INIT_CKPT_ARG="--init_checkpoint $LATEST_CKPT --start_step $START_STEP"
    else
        echo "[$(date)] No checkpoint found, starting fresh"
        INIT_CKPT_ARG=""
    fi

    # Wandb resume: reuse the same run ID across restarts
    WANDB_ARG=""
    if [ -f "$WANDB_ID_FILE" ]; then
        WANDB_RUN_ID=$(cat "$WANDB_ID_FILE")
        echo "[$(date)] Resuming wandb run: $WANDB_RUN_ID"
        WANDB_ARG="--wandb_run_id $WANDB_RUN_ID"
    fi

    # Run training
    /opt/conda/envs/torch-base/bin/python -m torch.distributed.run \
        --nproc_per_node=8 \
        --master_port=29501 \
        scripts/train_mem_space_fast_mem.py \
        --model_path models/Meta-Llama-3-8B \
        --output_dir "$CKPT_DIR" \
        --dolmino_path MemLong/data/processed/dolmino_0.5B_1024/train \
        --chunk_size 1024 \
        --curriculum "0:1,5000:2,10000:4,15000:8,25000:16" \
        --total_steps 50000 \
        --lr 5e-6 \
        --fast_mem_lr_mult 3.0 \
        --warmup_steps 1000 \
        --gradient_accumulation_steps 4 \
        --grad_clip 1.0 \
        --proj_grad_clip 0.1 \
        --batch_size 2 \
        --num_slots 512 \
        --top_k 64 \
        --selector_dim 128 \
        --selector_temperature 20.0 \
        --slot_value_norm_cap 5.0 \
        --slot_init random \
        --slot_init_noise 0.05 \
        --unfreeze_hidden_to_slot \
        --shared_memory_bank \
        --use_dual_gate \
        --forget_bias_init 2.0 \
        --input_bias_init 0.0 \
        --dual_gate_tanh_new \
        --use_l3_summary \
        --l3_n_summary 64 \
        --l3_n_layers 2 \
        --l3_n_heads 8 \
        --use_fast_mem \
        --fast_mem_heads 4 \
        --fast_mem_d_state 128 \
        --fast_mem_chunk_size 16 \
        --fast_mem_fusion_init 0.0 \
        --gradient_checkpointing \
        --babilong_mix_fraction 0.15 \
        --babilong_tasks qa1,qa2,qa5 \
        --babilong_lengths 0k,1k,2k,4k \
        --log_interval 10 \
        --save_interval 200 \
        --eval_interval 2000 \
        --eval_samples 50 \
        --wandb_project mixture-of-memory \
        --wandb_run_name fast_mem_v1 \
        --dtype bfloat16 \
        --attn_impl sdpa \
        --seed 42 \
        $INIT_CKPT_ARG $WANDB_ARG

    EXIT_CODE=$?
    echo "[$(date)] Training exited with code $EXIT_CODE at step >=$START_STEP"

    # Save wandb run ID from the first successful init (for future resumes)
    if [ ! -f "$WANDB_ID_FILE" ]; then
        # Extract run ID from wandb log in output dir
        FOUND_ID=$(grep -oP 'run-\d+_\d+-\K[a-z0-9]+' ${CKPT_DIR}/wandb/latest-run/run-*.wandb 2>/dev/null | head -1)
        if [ -z "$FOUND_ID" ]; then
            FOUND_ID=$(ls -d ${CKPT_DIR}/wandb/run-*/ 2>/dev/null | tail -1 | grep -oP 'run-\d+_\d+-\K[a-z0-9]+')
        fi
        if [ -n "$FOUND_ID" ]; then
            echo "$FOUND_ID" > "$WANDB_ID_FILE"
            echo "[$(date)] Saved wandb run ID: $FOUND_ID"
        fi
    fi

    # If exit code 0 (normal completion), stop the loop
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date)] Training completed successfully!"
        break
    fi

    # Otherwise, kill zombie processes, wait, and restart
    echo "[$(date)] Crash detected. Cleaning up zombie processes..."
    pkill -9 -f train_mem_space_fast_mem 2>/dev/null || true
    sleep 10
    # Ensure port 29501 is free
    while ss -tlnp | grep -q ":29501"; do
        echo "[$(date)] Port 29501 still in use, waiting..."
        pkill -9 -f train_mem_space_fast_mem 2>/dev/null || true
        sleep 15
    done
    # Wait 60s for GPU memory to fully release after FLA NVLink crash
    echo "[$(date)] Port free. Waiting 60s for GPU cooldown..."
    sleep 60
done
