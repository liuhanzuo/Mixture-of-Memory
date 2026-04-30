#!/bin/bash
# fix_x_ablation: Fix X.1 — slot_keys.detach() removed, LM gradient restored, SKRL removed.
#
# ROOT CAUSE OF fix_v_ablation TOP1_SIM FLOOR (researcher rpt_20260430_0520_fix_x_skrl_anti_productive):
#   SKRL drove slot_keys to ETF minimum (geometrically anti-correlated with routing selectivity).
#   At the ETF minimum, slot_keys are maximally spread BUT mathematically prevent Q_sel differentiation
#   because all cosine similarities are equal at -1/(N-1) ≈ -0.002.
#
#   SKRL is anti-productive: it's accomplishing its goal (diversity) but destroying routing selectivity.
#   Fix X: remove SKRL entirely (skrl_weight=0.0) and restore LM gradient to slot_keys.
#   With LM gradient flowing, slot_keys will specialize naturally based on what content is useful.
#
# Fix X.1 code changes (coder completed 2026-04-30):
#   selector.py:~159: self.slot_keys.detach() → self.slot_keys  (LM grad flows to slot_keys)
#   train_mem_space_pg19.py: --slot_value_norm_cap CLI flag added
#   config.py, memory_bank.py, layer.py, patch.py: norm cap wired up
#
# Hypothesis: top1_sim should rise above floor within 500 fwd passes once LM gradient
# shapes slot_keys toward natural slot specialization.
#
# Ablation sweep (3 nodes):
#   b200-1 (28.89.17.143, node0): skrl=0.0, norm_cap=10.0, T=10.0 (Fix X.1 primary)
#   b200-2 (28.89.17.144, node1): skrl=0.0, norm_cap=0.0,  T=10.0 (Fix X.1 no-cap control)
#   b200-3 (28.89.17.85,  node2): skrl=0.05, norm_cap=10.0, T=10.0 (SKRL control)
#
# Expected behavior:
#   fwd 200:  top1_sim_mean > 0.005 (routing differentiation confirmed)
#   fwd 500:  top1_sim_mean > 0.010 (primary criterion)
#   fwd 1000: top1_sim_mean > 0.050 (strong selectivity)
#
# Usage:
#   bash scripts/_run_fix_x_ablation.sh node0   # launch node0 only (b200-1)
#   bash scripts/_run_fix_x_ablation.sh node1   # launch node1 only (b200-2)
#   bash scripts/_run_fix_x_ablation.sh node2   # launch node2 only (b200-3)
#   bash scripts/_run_fix_x_ablation.sh         # launch all nodes via SSH

set -e
set -u

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

SSH_PASS_FILE="${PROJECT_DIR}/configs/password.txt"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=30"
SSH_CMD="sshpass -f ${SSH_PASS_FILE} ssh ${SSH_OPTS}"

# Remote workdir (cross-node shared NFS mount)
REMOTE_DIR="/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory"

# Node IPs
IP_NODE0="28.89.17.143"   # b200-1
IP_NODE1="28.89.17.144"   # b200-2
IP_NODE2="28.89.17.85"    # b200-3

# ── Fixed training args (from fix_v_ablation baseline) ───────────────────────
MODEL="/apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b"
DATA="${REMOTE_DIR}/data/pg19_chunks_llama3.npy"

TEMPERATURE=10.0    # Same as fix_v (Q_sel gradient restored)
ENTROPY=0.0         # Same as fix_v (no routing interference)
SIGMA=0.01          # slot_init_noise
LB=0.001            # load_balance_weight
WB_WARMUP=500       # writeback_warmup_steps

# Master ports (different from fix_v to avoid stale port conflicts)
PORT_NODE0=29940
PORT_NODE1=29941
PORT_NODE2=29942

# ── If called with a node argument, run locally (used when SSH'd into node) ──
NODE_ARG="${1:-all}"

if [[ "$NODE_ARG" == "node0" ]]; then
    echo "================================================================"
    echo "fix_x_ablation node0 — $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  skrl=0.0, norm_cap=10.0, T=10.0 (Fix X.1 primary: no SKRL, with cap)"
    echo "================================================================"
    mkdir -p "${REMOTE_DIR}/logs" "${REMOTE_DIR}/outputs/fix_x_ablation_node0"
    source /opt/conda/etc/profile.d/conda.sh && conda activate torch-base
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    exec torchrun --nproc_per_node=8 --master_port=${PORT_NODE0} \
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
        --skrl_weight 0.0 \
        --slot_value_norm_cap 10.0 \
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
        --output_dir ${REMOTE_DIR}/outputs/fix_x_ablation_node0

elif [[ "$NODE_ARG" == "node1" ]]; then
    echo "================================================================"
    echo "fix_x_ablation node1 — $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  skrl=0.0, norm_cap=0.0, T=10.0 (Fix X.1 no-cap control)"
    echo "================================================================"
    mkdir -p "${REMOTE_DIR}/logs" "${REMOTE_DIR}/outputs/fix_x_ablation_node1"
    source /opt/conda/etc/profile.d/conda.sh && conda activate torch-base
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    exec torchrun --nproc_per_node=8 --master_port=${PORT_NODE1} \
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
        --skrl_weight 0.0 \
        --slot_value_norm_cap 0.0 \
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
        --output_dir ${REMOTE_DIR}/outputs/fix_x_ablation_node1

elif [[ "$NODE_ARG" == "node2" ]]; then
    echo "================================================================"
    echo "fix_x_ablation node2 — $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  skrl=0.05, norm_cap=10.0, T=10.0 (SKRL control: compare with prior runs)"
    echo "================================================================"
    mkdir -p "${REMOTE_DIR}/logs" "${REMOTE_DIR}/outputs/fix_x_ablation_node2"
    source /opt/conda/etc/profile.d/conda.sh && conda activate torch-base
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    exec torchrun --nproc_per_node=8 --master_port=${PORT_NODE2} \
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
        --slot_value_norm_cap 10.0 \
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
        --output_dir ${REMOTE_DIR}/outputs/fix_x_ablation_node2

else
    # ── SSH launcher: launch all three nodes ─────────────────────────────────
    echo "================================================================"
    echo "fix_x_ablation SSH launcher — $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Fix X.1: slot_keys.detach() removed — LM gradient now flows to slot_keys"
    echo "SKRL removed (node0/node1): natural specialization via LM gradient"
    echo "SKRL control (node2): skrl=0.05 to compare with prior runs"
    echo ""
    echo "  b200-1 (${IP_NODE0}): skrl=0.0   norm_cap=10.0  T=${TEMPERATURE} (Fix X.1 primary)"
    echo "  b200-2 (${IP_NODE1}): skrl=0.0   norm_cap=0.0   T=${TEMPERATURE} (Fix X.1 no-cap ctrl)"
    echo "  b200-3 (${IP_NODE2}): skrl=0.05  norm_cap=10.0  T=${TEMPERATURE} (SKRL control)"
    echo ""
    echo "Success criteria:"
    echo "  fwd 200:  top1_sim_mean > 0.005"
    echo "  fwd 500:  top1_sim_mean > 0.010"
    echo "  fwd 1000: top1_sim_mean > 0.050"
    echo "================================================================"

    # ── b200-1: node0 — skrl=0.0, norm_cap=10.0, T=10.0 ─────────────────────
    echo ""
    echo ">>> [1/3] Launching node0 on b200-1 (${IP_NODE0}) ..."
    ${SSH_CMD} root@${IP_NODE0} "
      cd ${REMOTE_DIR} && \
      mkdir -p logs outputs/fix_x_ablation_node0 && \
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
        --skrl_weight 0.0 \
        --slot_value_norm_cap 10.0 \
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
        --output_dir ${REMOTE_DIR}/outputs/fix_x_ablation_node0 \
        > ${REMOTE_DIR}/logs/fix_x_ablation_node0_\$(date +%Y%m%d_%H%M).log 2>&1 &
      echo PID: \$!"
    echo "node0 launched. Sleeping 5s ..."
    sleep 5

    # ── b200-2: node1 — skrl=0.0, norm_cap=0.0, T=10.0 ──────────────────────
    echo ""
    echo ">>> [2/3] Launching node1 on b200-2 (${IP_NODE1}) ..."
    ${SSH_CMD} root@${IP_NODE1} "
      cd ${REMOTE_DIR} && \
      mkdir -p logs outputs/fix_x_ablation_node1 && \
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
        --skrl_weight 0.0 \
        --slot_value_norm_cap 0.0 \
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
        --output_dir ${REMOTE_DIR}/outputs/fix_x_ablation_node1 \
        > ${REMOTE_DIR}/logs/fix_x_ablation_node1_\$(date +%Y%m%d_%H%M).log 2>&1 &
      echo PID: \$!"
    echo "node1 launched. Sleeping 5s ..."
    sleep 5

    # ── b200-3: node2 — skrl=0.05, norm_cap=10.0, T=10.0 ────────────────────
    echo ""
    echo ">>> [3/3] Launching node2 on b200-3 (${IP_NODE2}) ..."
    ${SSH_CMD} root@${IP_NODE2} "
      cd ${REMOTE_DIR} && \
      mkdir -p logs outputs/fix_x_ablation_node2 && \
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
        --skrl_weight 0.05 \
        --slot_value_norm_cap 10.0 \
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
        --output_dir ${REMOTE_DIR}/outputs/fix_x_ablation_node2 \
        > ${REMOTE_DIR}/logs/fix_x_ablation_node2_\$(date +%Y%m%d_%H%M).log 2>&1 &
      echo PID: \$!"
    echo "node2 launched."

    echo ""
    echo "================================================================"
    echo "All 3 nodes launched."
    echo ""
    echo "Monitor logs (SSH into each node):"
    echo "  b200-1: tail -f ${REMOTE_DIR}/logs/fix_x_ablation_node0_*.log"
    echo "  b200-2: tail -f ${REMOTE_DIR}/logs/fix_x_ablation_node1_*.log"
    echo "  b200-3: tail -f ${REMOTE_DIR}/logs/fix_x_ablation_node2_*.log"
    echo ""
    echo "QUERY_DIAG diagnostic lines:"
    echo "  PASS @ fwd=200: top1_sim_mean > 0.005"
    echo "  PASS @ fwd=500: top1_sim_mean > 0.010"
    echo "  PASS @ fwd=1000: top1_sim_mean > 0.050"
    echo "================================================================"
fi
