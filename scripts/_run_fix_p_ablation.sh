#!/bin/bash
# fix_p_ablation: Fix NaN spiral caused by _sd_norms div-by-zero + temperature too sharp.
#
# Root cause of fix_o_ablation NaN spiral (step ~1178–1312 across both nodes):
#   (1) Fix P.1 (layer.py:631): _sd_norms = slot_delta.norm(...) lacks .clamp(min=1e-6).
#       When writeback gate activates post warmup (step 500+) and slot_delta≈0,
#       _sd_norms→0 → NaN injected into backward graph. Coder applied Fix P.1.
#       Fix: .clamp(min=1e-6) on _sd_norms (mirrors Fix L-1 input-side pattern).
#   (2) Fix P.2 (selector_temperature): T=1.0 still sharp; softmax gradient ∝ T.
#       At T=1.0, LM:SKRL gradient ratio ≈ 10:1 — SKRL pushes keys apart but
#       LM still dominates and can drive keys to degenerate configs (NaN under stress).
#       Fix: T=5.0 → softer routing distribution, smoother gradients.
#       Evidence: cos reached -0.012 at T=1.0 (keys spreading) but top1_sim stuck
#       at 1/N floor — routing never differentiated before NaN. T=5.0 should
#       allow gradual routing sharpening without the instability spike.
#
# Fix P.1 is already in code (coder, 2026-04-29 21:34).
# Fix P.2 is user-approved: selector_temperature = 5.0 (2026-04-30 00:22).
#
# Full fix stack (all in code, no special CLI flags needed):
#   Fix I:    hidden_to_slot in _mem_space_params() optimizer group
#   Fix J-A:  remove slots.detach() from soft-proxy einsum (layer.py:499)
#   Fix K:    strided_token slot init + _detach_banks carry-over
#   Fix L-1:  adaptive M_sel_hidden norm clip (input side)
#   Fix L-2:  per-param grad clip 0.1 for slot_to_hidden/hidden_to_slot
#   Fix L-3:  WRITEBACK_DIAG interval 200→50
#   Fix M-1:  slot_delta norm clip to bypass_h scale (output side)
#   Fix N:    SKRL re-enabled (variable weight), load_balance 10× lower
#   Fix O:    selector_temperature 10.0 → 1.0 (in code; overridden by Fix P.2 below)
#   Fix P.1:  _sd_norms .clamp(min=1e-6) at layer.py:631
#   Fix P.2:  selector_temperature → 5.0 (CLI arg, this script)  ← NEW
#
# Ablation sweep (3 nodes): T=5.0 fixed, varied SKRL weights and entropy
#   b200-1 (28.89.17.143, node0): skrl=0.10, entropy=0.001, lb=0.001
#   b200-2 (28.89.17.144, node1): skrl=0.05, entropy=0.0,   lb=0.001
#   b200-3 (28.89.17.85,  node2): skrl=0.15, entropy=0.0,   lb=0.001
#
# Success criterion (same as fix_n/fix_o):
#   step 200: mean_pairwise_cos < -0.002   (slot keys spreading)
#   step 300: top1_sim_mean > 0.003        (above 1/N = 0.001953 floor)
#   step 500: top1_sim_mean > 0.005
#   step 1000: top1_sim_mean > 0.05       → unblocks scale-up to N=1024
#
# Usage (run from project root on local machine):
#   bash scripts/_run_fix_p_ablation.sh
#
# Script launches all 3 nodes via SSH sequentially with 5s sleep between each.
# Processes run in the background (nohup); each node prints its launcher PID.

set -e
set -u

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

SSH_PASS_FILE="${PROJECT_DIR}/configs/password.txt"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=30"
SSH="sshpass -f ${SSH_PASS_FILE} ssh ${SSH_OPTS}"

# Remote workdir (cross-node shared NFS mount)
# MUST use wzc1 NFS path — /root/Mixture-of-Memory has older code that lacks
# strided_token, skrl_weight, selector_temperature, etc. (confirmed 2026-04-30 00:42)
REMOTE_DIR="/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory"

# Node IPs
IP_NODE0="28.89.17.143"   # b200-1
IP_NODE1="28.89.17.144"   # b200-2
IP_NODE2="28.89.17.85"    # b200-3

# ── Fixed training args (replicate fix_o_ablation baseline) ──────────────────
MODEL="/apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b"
DATA="${REMOTE_DIR}/data/pg19_chunks_llama3.npy"

TEMPERATURE=5.0    # Fix P.2: 1.0 → 5.0 (softer routing)
SIGMA=0.01         # slot_init_noise
LB=0.001           # load_balance_weight
WB_WARMUP=500      # writeback_warmup_steps

# Master ports (one per node; separate machines so no conflict, still unique for clarity)
PORT_NODE0=29900
PORT_NODE1=29901
PORT_NODE2=29902

echo "================================================================"
echo "fix_p_ablation SSH launcher — $(date '+%Y-%m-%d %H:%M:%S')"
echo "Fix P.1: _sd_norms .clamp(min=1e-6) (layer.py:631, already in code)"
echo "Fix P.2: selector_temperature ${TEMPERATURE} (was 1.0)"
echo ""
echo "  b200-1 (${IP_NODE0}): skrl=0.10  entropy=0.001  lb=${LB}"
echo "  b200-2 (${IP_NODE1}): skrl=0.05  entropy=0.0    lb=${LB}"
echo "  b200-3 (${IP_NODE2}): skrl=0.15  entropy=0.0    lb=${LB}"
echo "================================================================"

# ── b200-1: node0 — skrl=0.10, entropy=0.001 ─────────────────────────────────
echo ""
echo ">>> [1/3] Launching node0 on b200-1 (${IP_NODE0}) ..."
${SSH} root@${IP_NODE0} "
  cd ${REMOTE_DIR} && \
  mkdir -p logs outputs/fix_p_ablation_node0 && \
  source /opt/conda/etc/profile.d/conda.sh && conda activate torch-base && \
  export TRANSFORMERS_OFFLINE=1 && \
  export HF_DATASETS_OFFLINE=1 && \
  nohup torchrun --nproc_per_node=8 --master_port=${PORT_NODE0} \
    scripts/train_mem_space_pg19.py \
    --model ${MODEL} \
    --data ${DATA} \
    --max_chunks 200 \
    --skip_chunks 200 \
    --seq_len 4096 \
    --batch_size 1 \
    --num_slots 512 \
    --top_k 64 \
    --selector_dim 128 \
    --writeback_gate_max 0.3 \
    --writeback_warmup_steps ${WB_WARMUP} \
    --load_balance_weight ${LB} \
    --entropy_aux_weight 0.001 \
    --skrl_weight 0.10 \
    --selector_temperature ${TEMPERATURE} \
    --max_steps 10000 \
    --lr 3e-4 \
    --attn_impl sdpa \
    --dtype bfloat16 \
    --slot_init strided_token \
    --slot_init_noise ${SIGMA} \
    --niah_mix_fraction 0.10 \
    --niah_max_N 16 \
    --swa_window 512 \
    --shared_memory_bank \
    --unfreeze_hidden_to_slot \
    --output_dir ${REMOTE_DIR}/outputs/fix_p_ablation_node0 \
    > ${REMOTE_DIR}/logs/fix_p_ablation_node0_\$(date +%Y%m%d_%H%M).log 2>&1 &
  echo PID: \$!"
echo "node0 launched. Sleeping 5s ..."
sleep 5

# ── b200-2: node1 — skrl=0.05, entropy=0.0 ───────────────────────────────────
echo ""
echo ">>> [2/3] Launching node1 on b200-2 (${IP_NODE1}) ..."
${SSH} root@${IP_NODE1} "
  cd ${REMOTE_DIR} && \
  mkdir -p logs outputs/fix_p_ablation_node1 && \
  source /opt/conda/etc/profile.d/conda.sh && conda activate torch-base && \
  export TRANSFORMERS_OFFLINE=1 && \
  export HF_DATASETS_OFFLINE=1 && \
  nohup torchrun --nproc_per_node=8 --master_port=${PORT_NODE1} \
    scripts/train_mem_space_pg19.py \
    --model ${MODEL} \
    --data ${DATA} \
    --max_chunks 200 \
    --skip_chunks 200 \
    --seq_len 4096 \
    --batch_size 1 \
    --num_slots 512 \
    --top_k 64 \
    --selector_dim 128 \
    --writeback_gate_max 0.3 \
    --writeback_warmup_steps ${WB_WARMUP} \
    --load_balance_weight ${LB} \
    --entropy_aux_weight 0.0 \
    --skrl_weight 0.05 \
    --selector_temperature ${TEMPERATURE} \
    --max_steps 10000 \
    --lr 3e-4 \
    --attn_impl sdpa \
    --dtype bfloat16 \
    --slot_init strided_token \
    --slot_init_noise ${SIGMA} \
    --niah_mix_fraction 0.10 \
    --niah_max_N 16 \
    --swa_window 512 \
    --shared_memory_bank \
    --unfreeze_hidden_to_slot \
    --output_dir ${REMOTE_DIR}/outputs/fix_p_ablation_node1 \
    > ${REMOTE_DIR}/logs/fix_p_ablation_node1_\$(date +%Y%m%d_%H%M).log 2>&1 &
  echo PID: \$!"
echo "node1 launched. Sleeping 5s ..."
sleep 5

# ── b200-3: node2 — skrl=0.15, entropy=0.0 ───────────────────────────────────
echo ""
echo ">>> [3/3] Launching node2 on b200-3 (${IP_NODE2}) ..."
${SSH} root@${IP_NODE2} "
  cd ${REMOTE_DIR} && \
  mkdir -p logs outputs/fix_p_ablation_node2 && \
  source /opt/conda/etc/profile.d/conda.sh && conda activate torch-base && \
  export TRANSFORMERS_OFFLINE=1 && \
  export HF_DATASETS_OFFLINE=1 && \
  nohup torchrun --nproc_per_node=8 --master_port=${PORT_NODE2} \
    scripts/train_mem_space_pg19.py \
    --model ${MODEL} \
    --data ${DATA} \
    --max_chunks 200 \
    --skip_chunks 200 \
    --seq_len 4096 \
    --batch_size 1 \
    --num_slots 512 \
    --top_k 64 \
    --selector_dim 128 \
    --writeback_gate_max 0.3 \
    --writeback_warmup_steps ${WB_WARMUP} \
    --load_balance_weight ${LB} \
    --entropy_aux_weight 0.0 \
    --skrl_weight 0.15 \
    --selector_temperature ${TEMPERATURE} \
    --max_steps 10000 \
    --lr 3e-4 \
    --attn_impl sdpa \
    --dtype bfloat16 \
    --slot_init strided_token \
    --slot_init_noise ${SIGMA} \
    --niah_mix_fraction 0.10 \
    --niah_max_N 16 \
    --swa_window 512 \
    --shared_memory_bank \
    --unfreeze_hidden_to_slot \
    --output_dir ${REMOTE_DIR}/outputs/fix_p_ablation_node2 \
    > ${REMOTE_DIR}/logs/fix_p_ablation_node2_\$(date +%Y%m%d_%H%M).log 2>&1 &
  echo PID: \$!"
echo "node2 launched."

echo ""
echo "================================================================"
echo "All 3 nodes launched successfully."
echo ""
echo "Monitor logs (SSH into each node):"
echo "  b200-1: tail -f ${REMOTE_DIR}/logs/fix_p_ablation_node0_*.log"
echo "  b200-2: tail -f ${REMOTE_DIR}/logs/fix_p_ablation_node1_*.log"
echo "  b200-3: tail -f ${REMOTE_DIR}/logs/fix_p_ablation_node2_*.log"
echo ""
echo "Success criterion:"
echo "  step 200:  mean_pairwise_cos < -0.002   (keys diverging)"
echo "  step 500:  top1_sim_mean    > 0.005     (routing differentiating)"
echo "  step 1000: top1_sim_mean    > 0.05      (routing healthy)"
echo "================================================================"
