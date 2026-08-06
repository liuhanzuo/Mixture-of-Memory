#!/usr/bin/env bash
# ============================================================================
# Paper C #165 -- rtrunk keep28 (30-layer, ~6.6B) 2-node DDP launcher.
#
# WHY 2 nodes: keep28 fp32-AdamW OOMs on a single 95GiB H20 in the 1-node
# depthsweep recipe (BS=4, GA=4, eff_bs=128). Sharding params/grads/optim
# across 16 GPU (2 nodes x 8) halves per-rank state -> should fit.
#
# ---- CRITICAL: eff_bs must stay 128 (byte-identical to keep14/20/24) -------
# The whole point of the depth-sweep is a like-for-like comparison. eff_bs is
# a knob that changes gradient noise and therefore the optimisation trajectory
# -- degrading it destroys the ablation. So on 16 ranks we need BS*GA = 8:
#     eff_bs = BS * GA * WORLD_SIZE = 4 * 2 * 16 = 128  (default: BS=4, GA=2)
# If OOM at BS=4, fall back to BS=2 GA=4 or BS=1 GA=8 (keeps eff_bs=128).
# DO NOT lower EFF_BS.
#
# ---- HYPERPARAMS: byte-identical to run_paperC_depthsweep.sh rtrunk arm ----
# Recipe (from that launcher's rtrunk arm):
#   --keep_front_layers 28 --n_fresh_layers 2
#   --random_trunk --min_lr_inherited 1e-5
#   --lr 3e-4 --lr_inherited 3e-4  (uniform 3e-4, both optim groups)
#   --max_steps 1000 --warmup_steps 150 --save_every 500 --log_every 10
#   --seed 42 --gradient_checkpointing 1
#   --seq_len 2048 --optimizer adamw (fp32-master; keep28 OOM w/ 1 node)
# Data / val: P0-4 clean refusal25 split (train_refusal25_olmo2_2048.npy).
#
# ---- NCCL RECIPE (verified 2026-08-06, zwfy6 .82+.73 smoke) ----------------
#   MASTER_ADDR=<master node bond1 IP>  MASTER_PORT=29501
#   NCCL_SOCKET_IFNAME=bond1  NCCL_IB_DISABLE=1  (TCP fallback -- safe baseline)
# Bond1 is the only inter-node routable iface (bond2-9 are per-GPU private).
# IB may be faster (mlx5_0..8 up) but not yet verified on zwfy6-304; a rtrunk
# smoke should first confirm training works at all before optimising bandwidth.
#
# ---- USAGE (BOTH nodes must run this, differ only in NODE_RANK) ------------
# Assumes PROJECT_ROOT is the zwfy6 canonical workdir on each node.
#   MASTER = .82 (bond1 = 28.82.250.82, NODE_RANK=0)
#   WORKER = .73 (NODE_RANK=1)
#
# On .82:
#   NODE_RANK=0 setsid nohup bash scripts/_run_paperC_rtrunk_keep28_2node.sh \
#     >logs/paperC_rtrunk_keep28_2node0_sched.out 2>&1 </dev/null &
# On .73:
#   NODE_RANK=1 setsid nohup bash scripts/_run_paperC_rtrunk_keep28_2node.sh \
#     >logs/paperC_rtrunk_keep28_2node1_sched.out 2>&1 </dev/null &
#
# Then in ANOTHER ssh connection: tail the log, check nvidia-smi.
# ============================================================================
set -uo pipefail

: "${NODE_RANK:?must set NODE_RANK=0 (master .82) or 1 (worker .73)}"

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
PYBIN="${PYBIN:-/opt/conda/envs/torch-base/bin/python}"
BASE="${BASE:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/models/OLMo-2-1124-7B}"

# ---- data (P0-4 clean refusal25, md5-verified on both nodes) ---------------
DATA_PATH="${DATA_PATH:-$PROJECT_ROOT/data/paperC_squad_v2/train_refusal25_olmo2_2048.npy}"
VAL_PATH="${VAL_PATH:-$PROJECT_ROOT/data/paperC_squad_v2/val_refusal25.jsonl}"

# ---- NCCL: verified 2-node smoke recipe on zwfy6 .82+.73 (2026-08-06) ------
MASTER_ADDR="${MASTER_ADDR:-28.82.250.82}"   # .82 bond1
MASTER_PORT="${MASTER_PORT:-29501}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

# ---- eff_bs 128 = BS * GA * (2 nodes * 8 GPU) = 4 * 2 * 16 ------------------
BS="${BS:-4}"
GA="${GA:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NNODES="${NNODES:-2}"
NGPU_TOTAL=$(( NPROC_PER_NODE * NNODES ))
EFF_BS_TARGET=128
REAL_EFF=$(( BS * GA * NGPU_TOTAL ))

MAX_STEPS="${MAX_STEPS:-1000}"
SEQ_LEN="${SEQ_LEN:-2048}"
WARMUP="${WARMUP:-150}"
SEED="${SEED:-42}"
SAVE_EVERY="${SAVE_EVERY:-500}"
KEEP="${KEEP:-28}"
FRESH="${FRESH:-2}"
LR="${LR:-3e-4}"
LR_INH="${LR_INH:-3e-4}"

OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/outputs/paperC_depthsweep_keep28_rtrunk_refusal25}"
LOG_FILE="${LOG_FILE:-$PROJECT_ROOT/logs/paperC_rtrunk_keep28_2node${NODE_RANK}.log}"
mkdir -p "$OUT_DIR" "$(dirname "$LOG_FILE")"

export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "[keep28-2node $(date '+%F %T')] NODE_RANK=$NODE_RANK MASTER=$MASTER_ADDR:$MASTER_PORT"
echo "[keep28-2node $(date '+%F %T')] NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME NCCL_IB_DISABLE=$NCCL_IB_DISABLE"
echo "[keep28-2node $(date '+%F %T')] NNODES=$NNODES NPROC_PER_NODE=$NPROC_PER_NODE => WORLD=$NGPU_TOTAL"
echo "[keep28-2node $(date '+%F %T')] BS=$BS GA=$GA eff_bs=$REAL_EFF (target $EFF_BS_TARGET)"
if [ "$REAL_EFF" -ne "$EFF_BS_TARGET" ]; then
  echo "[keep28-2node] FATAL: eff_bs=$REAL_EFF != $EFF_BS_TARGET -- BS/GA misconfigured, would break comparability with keep14/20/24"
  exit 2
fi
echo "[keep28-2node] KEEP=$KEEP FRESH=$FRESH LR=$LR LR_INH=$LR_INH max_steps=$MAX_STEPS -> $OUT_DIR"
echo "[keep28-2node] DATA_PATH=$DATA_PATH"
echo "[keep28-2node] VAL_PATH=$VAL_PATH"

# preflight
pf=0
for f in "$DATA_PATH" "$VAL_PATH" "$BASE/config.json" \
         "$PROJECT_ROOT/scripts/train_olmo2_arch_probe2.py"; do
  [ -e "$f" ] || { echo "[keep28-2node] PREFLIGHT MISSING: $f"; pf=1; }
done
[ -x "$PYBIN" ] || { echo "[keep28-2node] PREFLIGHT: PYBIN not executable: $PYBIN"; pf=1; }
[ "$pf" = 0 ] || { echo "[keep28-2node] FATAL preflight failed"; exit 1; }

: > "$LOG_FILE"
echo "[keep28-2node] launch train_olmo2_arch_probe2.py -> tee $LOG_FILE"

exec "$PYBIN" -m torch.distributed.run \
  --nnodes "$NNODES" --node_rank "$NODE_RANK" --nproc_per_node "$NPROC_PER_NODE" \
  --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" \
  scripts/train_olmo2_arch_probe2.py \
    --data_path "$DATA_PATH" --output_dir "$OUT_DIR" --model_path "$BASE" \
    --keep_front_layers "$KEEP" --n_fresh_layers "$FRESH" \
    --batch_size "$BS" --grad_accumulation_steps "$GA" --seq_len "$SEQ_LEN" \
    --lr "$LR" --lr_inherited "$LR_INH" --max_steps "$MAX_STEPS" \
    --warmup_steps "$WARMUP" --save_every "$SAVE_EVERY" --log_every 10 --seed "$SEED" \
    --gradient_checkpointing 1 \
    --random_trunk --min_lr_inherited 1e-5 \
  2>&1 | tee -a "$LOG_FILE"
