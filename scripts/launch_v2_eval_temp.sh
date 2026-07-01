#!/bin/bash
# Launch v2 multi-ckpt eval on remote 8 GPU
# 3 ckpts × 3 tasks = 9 procs → use GPU 0-7 + reuse one (so 8 unique GPUs, qa5_final pairs with qa1_step5000)
cd /apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory
export PYTHONPATH=third_party/babilong-pkg
export HF_HUB_OFFLINE=1
TS=$(date +%Y%m%d_%H%M)

CKPT_DIR=outputs/babilong_sft_phase1b_v2_10k
CFG=$CKPT_DIR/adapter_config.json

launch() {
  local label=$1 ckpt_file=$2 task=$3 gpu=$4
  CUDA_VISIBLE_DEVICES=$gpu nohup setsid python scripts/run_babilong_mem_space.py \
    --model_path models/Llama-3.2-1B-Instruct \
    --checkpoint $ckpt_file \
    --adapter_config $CFG \
    --results_folder outputs/eval_phase1b_v2_${label} \
    --output_name p1bv2_${label}_${task} \
    --tasks $task --lengths 0k 1k 2k 4k 8k 16k 32k \
    --limit 100 --device cuda:0 --use_chat_template \
    </dev/null > logs/eval_p1bv2_${label}_${task}_${TS}.log 2>&1 &
  echo "${label} ${task} gpu=${gpu} pid=$!"
}

# Wave 1: step1000 (qa1,qa2,qa5) on GPU 0/1/2, final (qa1,qa2,qa5) on GPU 3/4/5
launch step1000 $CKPT_DIR/mem_space_adapter_step001000.pt qa1 0
launch step1000 $CKPT_DIR/mem_space_adapter_step001000.pt qa2 1
launch step1000 $CKPT_DIR/mem_space_adapter_step001000.pt qa5 2
launch final $CKPT_DIR/mem_space_adapter.pt qa1 3
launch final $CKPT_DIR/mem_space_adapter.pt qa2 4
launch final $CKPT_DIR/mem_space_adapter.pt qa5 5
launch step5000 $CKPT_DIR/mem_space_adapter_step005000.pt qa1 6
launch step5000 $CKPT_DIR/mem_space_adapter_step005000.pt qa2 7
# qa5 for step5000 waits (will launch later when GPU frees)

sleep 5
echo "=== procs ==="
ps -ef | grep run_babilong | grep -v grep | wc -l
echo "=== gpu ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | head -8
