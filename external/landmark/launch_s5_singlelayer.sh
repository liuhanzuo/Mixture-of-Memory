#!/usr/bin/env bash
# ============================================================================
# Phase 3 S5 (single-layer readout) LAUNCHER — runs FROM the isolated S5 tree.
#   external/landmark_s5_tree/llama/  (pristine anchor 99631a8 + single_layer_mem
#   patch only; physically separate from S4b's live llama-attention tree).
#
# Single axis = LM_SINGLE_LAYER=16 -> from_pretrained(single_layer_mem=16):
#   ONLY L16 runs landmark grouped-softmax; the other 31 layers run plain causal
#   softmax over the SAME retrieval+KV key set. KV/retrieval/windowing/data are
#   the faithful anchor recipe (single 512-token window, grad-ckpt ON).
#
# Faithful anchor recipe otherwise: LLaMA-1-7B base, bf16, FSDP full_shard
#   auto_wrap, per_device 2 x grad_accum 8 x 16 ranks = eff-batch 256 on 2 nodes
#   (=> set GA so per_device*GA*WORLD = 128; see GA computation below), ctx512,
#   mem_freq 63, lr 2e-5 cosine + 3% warmup, wd 0.1, RedPajama-1T-Sample (mirror).
#
# 2-NODE STATIC-RDZV IB recipe (c10d HANGS on this cluster -> MUST use static):
#   master=.76 node_rank 0, worker=.249 node_rank 1.
#   NCCL_IB_DISABLE=0 NCCL_IB_GID_INDEX=3 NCCL_P2P_DISABLE=1 (RoCE v2, 17x TCP),
#   NCCL_SOCKET_IFNAME=bond1 NCCL_DMABUF_ENABLE=0 NCCL_NET_GDR_LEVEL=0
#   GLOO_SOCKET_IFNAME=bond1.
#
# Usage (run ON EACH node; NODE_RANK differs):
#   NODE_RANK=0 MASTER_ADDR=28.49.57.76 bash launch_s5_singlelayer.sh   # on .76
#   NODE_RANK=1 MASTER_ADDR=28.49.57.76 bash launch_s5_singlelayer.sh   # on .249
# For a SINGLE-NODE 8-GPU run (if a whole group is one node), set NNODES=1.
# ============================================================================
set -euo pipefail

# ---- node/topology knobs ----
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
NNODES="${NNODES:-2}"
NODE_RANK="${NODE_RANK:?set NODE_RANK=0 on master(.76), 1 on worker(.249)}"
MASTER_ADDR="${MASTER_ADDR:-28.49.57.76}"
MASTER_PORT="${MASTER_PORT:-29533}"
NPROC="${NPROC:-8}"                          # GPUs per node

EXT="$PROJECT_ROOT/external"
S5TREE="$EXT/landmark_s5_tree/llama"         # <-- isolated S5 package dir
PY="${PY:-$EXT/landmark_venv/bin/python}"
BASE="${BASE:-$EXT/landmark_ckpts/llama1_7b_base}"
OUT="${OUT:-$EXT/landmark_ckpts/landmark_S5_L16_singlelayer}"
# Reuse the already-downloaded liang2kl RedPajama mirror cache (diskB shared FS).
CACHE="${CACHE:-$EXT/landmark-attention/llama/.hf_cache_s4}"

# ---- S5 axis + recipe knobs ----
export LM_SINGLE_LAYER="${LM_SINGLE_LAYER:-16}"     # the single readout layer
MEM_FREQ="${MEM_FREQ:-63}"
MAX_STEPS="${MAX_STEPS:-3000}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
PER_DEV_BS="${PER_DEV_BS:-2}"
# eff-batch 128 target: GA = 128 / (per_dev * world_size). world = NNODES*NPROC.
WORLD=$(( NNODES * NPROC ))
GA="${GA:-$(( 128 / (PER_DEV_BS * WORLD) ))}"
[ "$GA" -lt 1 ] && GA=1

# ---- NCCL / IB (static rdzv; c10d hangs on this cluster) ----
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=bond1
export NCCL_DMABUF_ENABLE=0
export NCCL_NET_GDR_LEVEL=0
export GLOO_SOCKET_IFNAME=bond1
export WANDB_MODE=offline
# RedPajama data files are pre-staged in CACHE, but HF `datasets` still resolves
# the dataset metadata/readme from the Hub -> needs the woa proxy (verified: with
# proxy the data is a cache hit, 930514 rows, no re-download). Model from_pretrained
# reads a LOCAL path so it is unaffected. Do NOT set HF_HUB_OFFLINE here.
export http_proxy="${http_proxy:-http://hy-proxy.woa.com:3128}"
export https_proxy="${https_proxy:-http://hy-proxy.woa.com:3128}"
export HF_HOME="${HF_HOME:-$CACHE/hf_home}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUT" "$PROJECT_ROOT/logs"
LOG="$PROJECT_ROOT/logs/landmark_S5_singlelayer_$(date +%Y%m%d_%H%M%S)_rank${NODE_RANK}.log"

cd "$S5TREE"
echo "[S5] tree=$S5TREE PY=$PY BASE=$BASE"
echo "[S5] LM_SINGLE_LAYER=$LM_SINGLE_LAYER  mem_freq=$MEM_FREQ  ctx512  grad-ckpt ON"
echo "[S5] NNODES=$NNODES NODE_RANK=$NODE_RANK MASTER=$MASTER_ADDR:$MASTER_PORT NPROC=$NPROC WORLD=$WORLD"
echo "[S5] per_dev=$PER_DEV_BS GA=$GA  => eff-batch=$(( PER_DEV_BS * GA * WORLD ))"
echo "[S5] MAX_STEPS=$MAX_STEPS SAVE_STEPS=$SAVE_STEPS  OUT=$OUT  LOG=$LOG"

setsid nohup "$PY" -m torch.distributed.run \
    --nnodes="$NNODES" --nproc_per_node="$NPROC" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    train.py \
    --model_name_or_path "$BASE" \
    --bf16 True \
    --output_dir "$OUT" \
    --cache_dir "$CACHE" \
    --model_max_length 512 \
    --mem_freq "$MEM_FREQ" \
    --num_train_epochs 1 \
    --per_device_train_batch_size "$PER_DEV_BS" \
    --per_device_eval_batch_size "$PER_DEV_BS" \
    --gradient_accumulation_steps "$GA" \
    --gradient_checkpointing True \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit 4 \
    --learning_rate 2e-5 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --max_steps "$MAX_STEPS" \
    >"$LOG" 2>&1 &
disown
echo "[S5] launched (rank $NODE_RANK), pid=$!  tail -f $LOG"
