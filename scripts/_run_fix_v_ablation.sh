#!/bin/bash
# fix_v_ablation: Fix selector_temperature starvation (Fix V) + entropy_aux conflict (Fix W).
#
# ROOT CAUSE of fix_u_ablation STEP-500 CRITERION FAILURE:
#   Fix U confirmed: SKRL SUCCEEDED — slot_keys reached -1/(N-1) = -0.0020 (mathematical maximum
#   diversity for N=512 unit vectors). This is NOT a failure; it's SKRL completing its job.
#
#   REAL PROBLEM: top1_sim stuck at floor 0.0027 — Q_sel cannot differentiate slots.
#   ROOT CAUSE: T=5.0 starves Q_sel of gradient.
#
#   Mathematical proof (researcher rpt_20260430_0429_skrl_gradient_starvation):
#     With N=512 near-uniform slot keys, softmax Jacobian max eigenvalue ≈ 1/N = 0.002 at T=1.
#     At T=5 (current), small-key-cos-sim differences ≈ 0.002 give logit differences = T×0.002 = 0.01.
#     The routing distribution is still near-uniform → Q_sel gradient still ~1/N scale.
#     At T=10, the softmax sharpens →gradient to Q_sel is ~21× stronger per step.
#     Expected top1_sim at T=10 after 500 steps: > 0.010 (5× above floor).
#
#   Fix O rationale WAS: T=10→1 to prevent LM gradient reaching slot_keys (NaN spirals).
#   Fix O is NOW WRONG: Fix Q.2 (self.slot_keys.detach() at source) already severs LM→slot_keys.
#   With Fix Q.2 in place, T=10 is safe. Fix O overcorrected.
#
#   SECONDARY ISSUE (b200-1 LM collapse at step 280):
#     entropy_aux_weight=0.001 pushes routing toward uniform AFTER SKRL succeeds.
#     This directly opposes Q_sel differentiation via LM loss.
#     Fix W: entropy_aux_weight 0.001 → 0.0 for ALL nodes.
#
# Fix V:  selector_temperature 5.0 → 10.0 (restore gradient to Q_sel)
# Fix W:  entropy_aux_weight 0.001 → 0.0 (all nodes, stop routing interference)
#
# Full fix stack in code (no new code changes needed — hyperparameter only):
#   Fix I:    hidden_to_slot in _mem_space_params() optimizer group
#   Fix J-A:  remove slots.detach() from soft-proxy einsum (layer.py:499)
#   Fix K:    strided_token slot init + _detach_banks carry-over
#   Fix L-1/2/3: adaptive norm clip + per-param grad clip + WRITEBACK_DIAG interval
#   Fix M-1:  slot_delta norm clip
#   Fix N:    SKRL re-enabled, load_balance 10x lower
#   Fix O:    selector_temperature default (now overridden to 10.0 by Fix V)
#   Fix P.1:  _sd_norms .clamp(min=1e-6) at layer.py:631
#   Fix P.2:  selector_temperature CLI arg (now 10.0 via Fix V)
#   Fix Q.2:  self.slot_keys.detach() at SOURCE in selector.py (LM grad path severed)
#   Fix S:    Forward hook registers SKRL in DDP output graph (no double-hook)
#   Fix T:    Analytical mean-cos in slot_key_diversity_loss() (selector.py)
#   Fix U:    Analytical SKRL_DIAG in layer.py (zero-variance measurement)
#   Fix V:    selector_temperature 5.0 → 10.0 (THIS RUN — restore Q_sel gradient)
#   Fix W:    entropy_aux_weight 0.001 → 0.0 (THIS RUN — stop routing interference)
#
# Ablation sweep (same skrl_weight grid; Fix V+W applied uniformly to all nodes):
#   b200-1 (28.89.17.143, node0): skrl=0.10, entropy=0.0, lb=0.001, T=10.0
#   b200-2 (28.89.17.144, node1): skrl=0.05, entropy=0.0, lb=0.001, T=10.0
#   b200-3 (28.89.17.85,  node2): skrl=0.15, entropy=0.0, lb=0.001, T=10.0
#
# Expected behavior:
#   fwd 200:  top1_sim_mean > 0.005 (first visible routing differentiation)
#   fwd 500:  top1_sim_mean > 0.010 (5x above floor=0.00195)
#   fwd 1000: top1_sim_mean > 0.05
#
# Usage:
#   bash scripts/_run_fix_v_ablation.sh

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

TEMPERATURE=10.0   # Fix V: restore T=10.0 (Q_sel gradient restored, safe with Fix Q.2)
ENTROPY=0.0        # Fix W: remove routing interference
SIGMA=0.01         # slot_init_noise
LB=0.001           # load_balance_weight
WB_WARMUP=500      # writeback_warmup_steps

# Master ports (different from fix_u to avoid stale port conflicts)
PORT_NODE0=29930
PORT_NODE1=29931
PORT_NODE2=29932

echo "================================================================"
echo "fix_v_ablation SSH launcher — $(date '+%Y-%m-%d %H:%M:%S')"
echo "Fix V: selector_temperature 5.0 → 10.0 (restore Q_sel gradient, safe with Fix Q.2)"
echo "Fix W: entropy_aux_weight → 0.0 (all nodes, stop routing interference)"
echo "Fix Q.2: slot_keys.detach() at source (LM grad path severed — T=10 now safe)"
echo ""
echo "  b200-1 (${IP_NODE0}): skrl=0.10  entropy=${ENTROPY}  lb=${LB}  T=${TEMPERATURE}"
echo "  b200-2 (${IP_NODE1}): skrl=0.05  entropy=${ENTROPY}  lb=${LB}  T=${TEMPERATURE}"
echo "  b200-3 (${IP_NODE2}): skrl=0.15  entropy=${ENTROPY}  lb=${LB}  T=${TEMPERATURE}"
echo ""
echo "Key criterion: top1_sim_mean RISING above floor 0.00195"
echo "  fwd 200:  top1_sim_mean > 0.005"
echo "  fwd 500:  top1_sim_mean > 0.010"
echo "  fwd 1000: top1_sim_mean > 0.05"
echo "================================================================"

# ── b200-1: node0 — skrl=0.10, entropy=0.0, T=10.0 ──────────────────────────
echo ""
echo ">>> [1/3] Launching node0 on b200-1 (${IP_NODE0}) ..."
${SSH} root@${IP_NODE0} "
  cd ${REMOTE_DIR} && \
  mkdir -p logs outputs/fix_v_ablation_node0 && \
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
    --entropy_aux_weight ${ENTROPY} \
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
    --output_dir ${REMOTE_DIR}/outputs/fix_v_ablation_node0 \
    > ${REMOTE_DIR}/logs/fix_v_ablation_node0_\$(date +%Y%m%d_%H%M).log 2>&1 &
  echo PID: \$!"
echo "node0 launched. Sleeping 5s ..."
sleep 5

# ── b200-2: node1 — skrl=0.05, entropy=0.0, T=10.0 ──────────────────────────
echo ""
echo ">>> [2/3] Launching node1 on b200-2 (${IP_NODE1}) ..."
${SSH} root@${IP_NODE1} "
  cd ${REMOTE_DIR} && \
  mkdir -p logs outputs/fix_v_ablation_node1 && \
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
    --entropy_aux_weight ${ENTROPY} \
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
    --output_dir ${REMOTE_DIR}/outputs/fix_v_ablation_node1 \
    > ${REMOTE_DIR}/logs/fix_v_ablation_node1_\$(date +%Y%m%d_%H%M).log 2>&1 &
  echo PID: \$!"
echo "node1 launched. Sleeping 5s ..."
sleep 5

# ── b200-3: node2 — skrl=0.15, entropy=0.0, T=10.0 ──────────────────────────
echo ""
echo ">>> [3/3] Launching node2 on b200-3 (${IP_NODE2}) ..."
${SSH} root@${IP_NODE2} "
  cd ${REMOTE_DIR} && \
  mkdir -p logs outputs/fix_v_ablation_node2 && \
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
    --entropy_aux_weight ${ENTROPY} \
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
    --output_dir ${REMOTE_DIR}/outputs/fix_v_ablation_node2 \
    > ${REMOTE_DIR}/logs/fix_v_ablation_node2_\$(date +%Y%m%d_%H%M).log 2>&1 &
  echo PID: \$!"
echo "node2 launched."

echo ""
echo "================================================================"
echo "All 3 nodes launched."
echo ""
echo "Monitor logs (SSH into each node):"
echo "  b200-1: tail -f ${REMOTE_DIR}/logs/fix_v_ablation_node0_*.log"
echo "  b200-2: tail -f ${REMOTE_DIR}/logs/fix_v_ablation_node1_*.log"
echo "  b200-3: tail -f ${REMOTE_DIR}/logs/fix_v_ablation_node2_*.log"
echo ""
echo "Step-200 criterion: grep 'top1_sim' <log>"
echo "  PASS: top1_sim_mean > 0.005 (routing differentiation confirmed)"
echo "  FAIL: top1_sim_mean stuck at 0.002 (Q_sel still not differentiating)"
echo ""
echo "Success criteria:"
echo "  fwd 200:  top1_sim_mean > 0.005"
echo "  fwd 500:  top1_sim_mean > 0.010"
echo "  fwd 1000: top1_sim_mean > 0.05"
echo "================================================================"
