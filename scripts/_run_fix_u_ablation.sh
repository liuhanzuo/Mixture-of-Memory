#!/bin/bash
# fix_u_ablation: Fix SKRL_DIAG diagnostic in layer.py to use analytical formula (Fix U).
#
# Root cause of fix_t_ablation APPARENT failure:
#   Fix T updated selector.py::slot_key_diversity_loss() to the analytical mean-cos formula,
#   but layer.py lines 679-682 (SKRL_DIAG diagnostic block) still used the old 256-pair
#   random estimator.
#   Statistical analysis of the OLD diagnostic estimator:
#     σ(cos_ij) ≈ 0.088 for unit vectors in d=128
#     std(mean of 256 pairs): 0.088/√256 ≈ 0.0055
#     ±3σ noise floor: ±0.0165
#   Result: observed ±0.015 oscillation in mean_pairwise_cos was 100% measurement noise
#   from the stale random-pair estimator — Fix T was actually working all along.
#   Confidence: HIGH (rpt_20260430_0350_fix_u_diagnosis)
#
# Fix U (analytical SKRL_DIAG, layer.py):
#   BEFORE: idx_i = torch.randint(N, (256,), ...); idx_j = ...
#           mean_pairwise_cos = (nk[idx_i] * nk[idx_j]).sum(-1).mean().item()
#   AFTER:  S_diag = nk.sum(dim=0)                    # [d]
#           mean_pairwise_cos = ((S_diag.dot(S_diag) - N) / (N*(N-1))).item()
#   Identity: Σᵢ≠ⱼ nkᵢ·nkⱼ = ||S||² - N  → exact, zero variance.
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
#   Fix T:    Analytical mean-cos in slot_key_diversity_loss() (selector.py)
#   Fix U:    Analytical SKRL_DIAG in layer.py (THIS RUN — removes ±0.0165 noise floor)
#
# Ablation sweep (same hyperparameter grid as fix_t_ablation):
#   b200-1 (28.89.17.143, node0): skrl=0.10, entropy=0.001, lb=0.001, T=5.0
#   b200-2 (28.89.17.144, node1): skrl=0.05, entropy=0.0,   lb=0.001, T=5.0
#   b200-3 (28.89.17.85,  node2): skrl=0.15, entropy=0.0,   lb=0.001, T=5.0
#
# Expected behavior with Fix T + Fix U:
#   step 200: mean_pairwise_cos MONOTONICALLY DECREASING (exact measurement, zero variance)
#   step 300: mean_pairwise_cos < -0.005
#   step 500: top1_sim_mean > 0.005
#   step 1000: top1_sim_mean > 0.05
#
# Usage:
#   bash scripts/_run_fix_u_ablation.sh

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

# Master ports (different from fix_t to avoid stale port conflicts)
PORT_NODE0=29920
PORT_NODE1=29921
PORT_NODE2=29922

echo "================================================================"
echo "fix_u_ablation SSH launcher — $(date '+%Y-%m-%d %H:%M:%S')"
echo "Fix U: analytical SKRL_DIAG in layer.py (removes ±0.0165 noise floor)"
echo "Fix T: analytical mean-cos in slot_key_diversity_loss() (zero-variance gradient)"
echo "Fix Q.2: slot_keys.detach() at source (LM grad path severed)"
echo "Fix S: forward hook (DDP double-hook resolved)"
echo ""
echo "  b200-1 (${IP_NODE0}): skrl=0.10  entropy=0.001  lb=${LB}  T=${TEMPERATURE}"
echo "  b200-2 (${IP_NODE1}): skrl=0.05  entropy=0.0    lb=${LB}  T=${TEMPERATURE}"
echo "  b200-3 (${IP_NODE2}): skrl=0.15  entropy=0.0    lb=${LB}  T=${TEMPERATURE}"
echo ""
echo "Key criterion: cos trend at step 200 MUST be MONOTONICALLY DECREASING"
echo "  (Fix T+U: zero-variance gradient AND zero-variance measurement)"
echo "================================================================"

# ── b200-1: node0 — skrl=0.10, entropy=0.001 ─────────────────────────────────
echo ""
echo ">>> [1/3] Launching node0 on b200-1 (${IP_NODE0}) ..."
${SSH} root@${IP_NODE0} "
  cd ${REMOTE_DIR} && \
  mkdir -p logs outputs/fix_u_ablation_node0 && \
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
    --output_dir ${REMOTE_DIR}/outputs/fix_u_ablation_node0 \
    > ${REMOTE_DIR}/logs/fix_u_ablation_node0_\$(date +%Y%m%d_%H%M).log 2>&1 &
  echo PID: \$!"
echo "node0 launched. Sleeping 5s ..."
sleep 5

# ── b200-2: node1 — skrl=0.05, entropy=0.0 ───────────────────────────────────
echo ""
echo ">>> [2/3] Launching node1 on b200-2 (${IP_NODE1}) ..."
${SSH} root@${IP_NODE1} "
  cd ${REMOTE_DIR} && \
  mkdir -p logs outputs/fix_u_ablation_node1 && \
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
    --output_dir ${REMOTE_DIR}/outputs/fix_u_ablation_node1 \
    > ${REMOTE_DIR}/logs/fix_u_ablation_node1_\$(date +%Y%m%d_%H%M).log 2>&1 &
  echo PID: \$!"
echo "node1 launched. Sleeping 5s ..."
sleep 5

# ── b200-3: node2 — skrl=0.15, entropy=0.0 ───────────────────────────────────
echo ""
echo ">>> [3/3] Launching node2 on b200-3 (${IP_NODE2}) ..."
${SSH} root@${IP_NODE2} "
  cd ${REMOTE_DIR} && \
  mkdir -p logs outputs/fix_u_ablation_node2 && \
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
    --output_dir ${REMOTE_DIR}/outputs/fix_u_ablation_node2 \
    > ${REMOTE_DIR}/logs/fix_u_ablation_node2_\$(date +%Y%m%d_%H%M).log 2>&1 &
  echo PID: \$!"
echo "node2 launched."

echo ""
echo "================================================================"
echo "All 3 nodes launched."
echo ""
echo "Monitor logs (SSH into each node):"
echo "  b200-1: tail -f ${REMOTE_DIR}/logs/fix_u_ablation_node0_*.log"
echo "  b200-2: tail -f ${REMOTE_DIR}/logs/fix_u_ablation_node1_*.log"
echo "  b200-3: tail -f ${REMOTE_DIR}/logs/fix_u_ablation_node2_*.log"
echo ""
echo "Step-200 criterion: grep 'mean_pairwise_cos' <log>"
echo "  PASS: values monotonically decreasing (e.g. -0.002 → -0.005 → -0.009)"
echo "  FAIL: oscillating ± (Fix U didn't take effect — check layer.py SKRL_DIAG block)"
echo ""
echo "Success criteria:"
echo "  step 200:  mean_pairwise_cos MONOTONICALLY DECREASING"
echo "  step 300:  mean_pairwise_cos < -0.005"
echo "  step 500:  top1_sim_mean    > 0.005"
echo "  step 1000: top1_sim_mean    > 0.05"
echo "================================================================"
