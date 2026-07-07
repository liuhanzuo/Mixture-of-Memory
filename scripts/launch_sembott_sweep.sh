#!/usr/bin/env bash
# Launch ONE (layer j, dim) arm of the 1B semantic-bottleneck SWEEP on N GPUs.
#
# Generalises launch_semantic_bottleneck_1b.sh: OUT is derived from (j,dim) so
# multiple sweep arms never clobber the existing sembott_1b_baseline/bottleneck.
#
# Usage (eff_bs kept at 96 to match the original j6d512 run: bs*gaccum*ngpu=96):
#   local L20A (183GB, 4 GPU, bs24 gaccum1):
#     BJ=4 BD=512 GPUS=0,1,2,3 PORT=29620 bash scripts/launch_sembott_sweep.sh
#   H20 (97.8GB, 4 GPU, bs12 gaccum2 to fit VRAM, same eff_bs=96):
#     BJ=6 BD=256 GPUS=0,1,2,3 PORT=29620 BATCH_SIZE=12 GRAD_ACCUM=2 bash scripts/launch_sembott_sweep.sh
set -euo pipefail

export PATH=/opt/conda/bin:$PATH
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
cd "$PROJECT_ROOT"

export WANDB_MODE=offline
export HF_HOME="$PROJECT_ROOT/.hf_cache"
export HF_DATASETS_CACHE="$PROJECT_ROOT/.hf_cache/datasets"
export TOKENIZERS_PARALLELISM=false
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0

BJ="${BJ:-6}"                 # bottleneck layer j
BD="${BD:-512}"              # bottleneck dim
GPUS="${GPUS:-0,1,2,3}"
PORT="${PORT:-29620}"
DATA_PATH="${DATA_PATH:-data/slimpajama_chunks_4096_llama3.npy}"
SEQ_LEN="${SEQ_LEN:-2048}"
BATCH_SIZE="${BATCH_SIZE:-24}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MAX_STEPS="${MAX_STEPS:-2000}"
LR="${LR:-3e-4}"

NGPU=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
OUT="${OUT:-outputs/sembott_sweep_j${BJ}_d${BD}}"
mkdir -p logs "$OUT"

echo "[launch-sweep] j=$BJ dim=$BD GPUS=$GPUS ngpu=$NGPU bs=$BATCH_SIZE gaccum=$GRAD_ACCUM eff_bs=$((BATCH_SIZE*GRAD_ACCUM*NGPU)) out=$OUT port=$PORT"

CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON_BIN" -m torch.distributed.run \
  --nproc_per_node="$NGPU" --nnodes=1 \
  --rdzv_backend=c10d --rdzv_endpoint="127.0.0.1:$PORT" \
  scripts/train_semantic_bottleneck_1b.py \
  --data_path "$DATA_PATH" \
  --output_dir "$OUT" \
  --bottleneck_layer "$BJ" \
  --bottleneck_dim "$BD" \
  --max_steps "$MAX_STEPS" \
  --seq_len "$SEQ_LEN" \
  --batch_size "$BATCH_SIZE" \
  --grad_accumulation_steps "$GRAD_ACCUM" \
  --lr "$LR" --min_lr 3e-5 --warmup_steps 100 \
  --save_every 500 --log_every 20
