#!/bin/bash
# fix_t_ablation: Replace random-pair SKRL with analytical O(N·d) mean-cos formulation.
#
# Root cause of fix_q_ablation_v4 (and all prior) DEFINITIVELY FAILED:
#   slot_key_diversity_loss() sampled 512 random pairs from N*(N-1)/2 = 131,328 pairs (N=512).
#   Statistical analysis:
#     Signal magnitude:  mean_pairwise_cos target ≈ -0.005
#     Sampling std:      σ(cos_ij) ≈ 0.088 for unit vectors in d=128
#     std(mean of 512):  0.088/√512 ≈ 0.004
#     SNR ≈ 0.005/0.004 ≈ 1.25 — essentially a coin flip per step
#   Result: gradient direction was a random walk, not directed optimization.
#   Key diagnostic: oscillation amplitude ±0.015 was invariant to skrl_weight (0.05/0.10/0.15),
#   ruling out LM competition and confirming SKRL's own gradient was undirected noise.
#
# Fix T (analytical mean-cos, selector.py):
#   BEFORE: sample 512 random pairs → compute cos for each → take mean
#   AFTER:  nk = F.normalize(slot_keys, dim=-1)   # [N, d]
#           S  = nk.sum(dim=0)                     # [d]
#           mean_cos = (S.dot(S) - N) / (N*(N-1))  # scalar, exact, zero variance
#   Identity: Σᵢ≠ⱼ nkᵢ·nkⱼ = ||S||² - N  (summing outer product diagonal)
#   Cost: O(N·d) = O(512·128) = 65k ops vs O(512·128) same — but zero variance.
#   Expected: mean_pairwise_cos monotonically decreasing from step 1.
#
# Fix stack in code (all present, no special CLI flags):
#   Fix I:    hidden_to_slot in _mem_space_params() optimizer group
#   Fix J-A:  remove slots.detach() from soft-proxy einsum (layer.py:499)
#   Fix K:    strided_token slot init + _detach_banks carry-over
#   Fix L-1/2/3: adaptive norm clip + per-param grad clip + WRITEBACK_DIAG interval
#   Fix M-1:  slot_delta norm clip
#   Fix N:    SKRL re-enabled, load_balance 10x lower
#   Fix O:    selector_temperature default lowered
#   Fix P.1:  _sd_norms .clamp(min=1e-6) at layer.py:631
#   Fix P.2:  selector_temperature → 5.0 (CLI arg, this script)
#   Fix Q.2:  self.slot_keys.detach() at SOURCE in selector.py (LM grad path severed)
#   Fix S:    Forward hook registers SKRL in DDP output graph (no double-hook)
#   Fix T:    Analytical mean-cos in slot_key_diversity_loss() ← THIS RUN
#
# Ablation sweep (same hyperparameter grid as fix_q_ablation_v4):
#   b200-1 (28.89.17.143, node0): skrl=0.10, entropy=0.001, lb=0.001, T=5.0
#   b200-2 (28.89.17.144, node1): skrl=0.05, entropy=0.0,   lb=0.001, T=5.0
#   b200-3 (28.89.17.85,  node2): skrl=0.15, entropy=0.0,   lb=0.001, T=5.0
#
# Expected behavior with Fix T:
#   step 200: mean_pairwise_cos MONOTONICALLY DECREASING (zero sampling variance → smooth gradient)
#   step 300: mean_pairwise_cos < -0.005
#   step 500: top1_sim_mean > 0.005
#   step 1000: top1_sim_mean > 0.05
#
# Usage:
#   bash scripts/_run_fix_t_ablation.sh

set -e
set -u

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

SSH_PASS_FILE="${PROJECT_DIR}/configs/password.txt"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=30"
SSH="sshpass -f ${SSH_PASS_FILE} ssh ${SSH_OPTS}"

# Remote workdir (cross-node shared NFS mount)
REMOTE_DIR="/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory"

# Node IPs
IP_NODE0="28.89.17.143"   # b200-1
IP_NODE1="28.89.17.144"   # b200-2
IP_NODE2="28.89.17.85"    # b200-3

# ── Fixed training args ───────────────────────────────────────────────────────
MODEL="/apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b"
DATA="${REMOTE_DIR}/data/pg19_chunks_llama3.npy"

TEMPERATURE=5.0    # Fix P.2: softer routing
SIGMA=0.01         # slot_init_noise
LB=0.001           # load_balance_weight
WB_WARMUP=500      # writeback_warmup_steps

# Master ports (different from fix_q to avoid stale port conflicts)
PORT_NODE0=29910
PORT_NODE1=29911
PORT_NODE2=29912

echo "================================================================"
echo "fix_t_ablation SSH launcher — $(date '+%Y-%m-%d %H:%M:%S')"
echo "Fix T: analytical mean-cos in slot_key_diversity_loss() (O(N·d), zero variance)"
echo "Fix Q.2: slot_keys.detach() at source (LM grad path severed)"
echo "Fix S: forward hook (DDP double-hook resolved)"
echo ""
echo "  b200-1 (${IP_NODE0}): skrl=0.10  entropy=0.001  lb=${LB}  T=${TEMPERATURE}"
echo "  b200-2 (${IP_NODE1}): skrl=0.05  entropy=0.0    lb=${LB}  T=${TEMPERATURE}"
echo "  b200-3 (${IP_NODE2}): skrl=0.15  entropy=0.0    lb=${LB}  T=${TEMPERATURE}"
echo ""
echo "Key criterion: cos trend at step 200 MUST be MONOTONICALLY DECREASING"
echo "  (Fix T provides zero-variance gradient; oscillation would indicate new issue)"
echo "================================================================"

# ── b200-1: node0 — skrl=0.10, entropy=0.001 ─────────────────────────────────
echo ""
echo ">>> [1/3] Launching node0 on b200-1 (${IP_NODE0}) ..."
${SSH} root@${IP_NODE0} "
  cd ${REMOTE_DIR} && \
  mkdir -p logs outputs/fix_t_ablation_node0 && \
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
    --output_dir ${REMOTE_DIR}/outputs/fix_t_ablation_node0 \
    > ${REMOTE_DIR}/logs/fix_t_ablation_node0_\$(date +%Y%m%d_%H%M).log 2>&1 &
  echo PID: \$!"
echo "node0 launched. Sleeping 5s ..."
sleep 5

# ── b200-2: node1 — skrl=0.05, entropy=0.0 ───────────────────────────────────
echo ""
echo ">>> [2/3] Launching node1 on b200-2 (${IP_NODE1}) ..."
${SSH} root@${IP_NODE1} "
  cd ${REMOTE_DIR} && \
  mkdir -p logs outputs/fix_t_ablation_node1 && \
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
    --output_dir ${REMOTE_DIR}/outputs/fix_t_ablation_node1 \
    > ${REMOTE_DIR}/logs/fix_t_ablation_node1_\$(date +%Y%m%d_%H%M).log 2>&1 &
  echo PID: \$!"
echo "node1 launched. Sleeping 5s ..."
sleep 5

# ── b200-3: node2 — skrl=0.15, entropy=0.0 ───────────────────────────────────
echo ""
echo ">>> [3/3] Launching node2 on b200-3 (${IP_NODE2}) ..."
${SSH} root@${IP_NODE2} "
  cd ${REMOTE_DIR} && \
  mkdir -p logs outputs/fix_t_ablation_node2 && \
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
    --output_dir ${REMOTE_DIR}/outputs/fix_t_ablation_node2 \
    > ${REMOTE_DIR}/logs/fix_t_ablation_node2_\$(date +%Y%m%d_%H%M).log 2>&1 &
  echo PID: \$!"
echo "node2 launched."

echo ""
echo "================================================================"
echo "All 3 nodes launched."
echo ""
echo "Monitor logs (SSH into each node):"
echo "  b200-1: tail -f ${REMOTE_DIR}/logs/fix_t_ablation_node0_*.log"
echo "  b200-2: tail -f ${REMOTE_DIR}/logs/fix_t_ablation_node1_*.log"
echo "  b200-3: tail -f ${REMOTE_DIR}/logs/fix_t_ablation_node2_*.log"
echo ""
echo "Step-200 criterion: grep 'mean_pairwise_cos' <log>"
echo "  PASS: values monotonically decreasing (e.g. -0.002 → -0.005 → -0.009)"
echo "  FAIL: oscillating ± (Fix T didn't take effect)"
echo ""
echo "Success criteria:"
echo "  step 200:  mean_pairwise_cos MONOTONICALLY DECREASING"
echo "  step 300:  mean_pairwise_cos < -0.005"
echo "  step 500:  top1_sim_mean    > 0.005"
echo "  step 1000: top1_sim_mean    > 0.05"
echo "================================================================"
