#!/usr/bin/env bash
# paperC — healed Qwen3-8B-BASE front8+fresh2 arm, to separate the
# heal-vs-no-heal confound flagged at paperC/README.md:348-350.
#
# Pre-registered in paperC/HEAL_CONFOUND_PREREGISTRATION.md (commit 9920de7,
# written BEFORE any GPU). Read that file before changing ANY value here --
# every number below is justified there and several are load-bearing:
#
#   keep_front=8   ABSOLUTE depth match to the qwen3_8b_base/k8 MMLU-Pro cell
#                  (NOT a depth-fraction match; Qwen3 has 36 layers vs OLMo-2's
#                  32, see preregistration §2). Changing this breaks the pairing.
#   n_fresh=2      matches all four OLMo-2 healed arms (keep{8,10,12,14}fresh2).
#   MAX_STEPS      200000 = the cosine HORIZON, not the stopping point. The
#                  read-out is step 121000, which is where olmo2_7b/keep8 was
#                  scored and where its LR was 8.09e-06 mid-decay. Setting
#                  max_steps=121000 would make 121000 the END of the decay = a
#                  different training state. DO NOT "tidy" this to 121000.
#   eff_bs 128     = bs2 x accum8 x 8 ranks. Matches OLMo-2 keep8 (bs16x1x8) and
#                  the historical Qwen3 armB. Preserves tokens/step = 262144.
#
# ⚠️ MODEL_PATH is Qwen3-8B-**Base**. models/Qwen--Qwen3-8b is Qwen3-8B-*Instruct*
#    (eos 151645 <|im_end|>, ctx 40960) and is INVALID for paperC, whose protocol
#    is chat_template=False. All five pre-existing qwen3_minarch_* arms used the
#    Instruct dir and are therefore unusable here. Criterion is eos_token_id +
#    ctx, NOT presence of a chat_template (both have one).
#
# ⚠️ Checkpoint rotation is MANDATORY, not optional tidiness: each ckpt is ~38 GB
#    (fp32 weights + fp32 AdamW), 242 unrotated saves would be ~9.2 TB and zwfy6
#    has ~3.4 TB free. The retention flags below bound the arm at ~420 GB.
#    keep_steps=121000 protects the read-out checkpoint from rotation forever.
#
# ⚠️ There is NO --eval_interval in this trainer (0 occurrences) -- it has no
#    inline eval at all, so the NCCL-watchdog desync hazard is structurally
#    absent. Eval is offline on checkpoints. Do not add an inline eval.
#
# Single node, 8xH20. 16-card 2-node DDP was REJECTED on measured data: 6.91
# s/step at 16 ranks vs 7.59 at 8 ranks (1.10x, not 2x) because eff_bs is held
# at 128 so accum halves to 4 and there is too little local work to hide the
# full-3.9B-fp32-gradient all-reduce over bond1 TCP; the historical 2-node run
# also died at step 20040 on a TCPStore heartbeat failure. See preregistration §10.
set -euo pipefail

PROJECT_ROOT="/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory"
cd "$PROJECT_ROOT" || exit 9

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
MODELS="/apdcephfs_zwfy6/share_304376610/pighzliu_code/models"
MODEL_PATH="${MODEL_PATH:-$MODELS/Qwen3-8B-Base}"

KEEP_FRONT="${KEEP_FRONT:-8}"
N_FRESH="${N_FRESH:-2}"
MAX_STEPS="${MAX_STEPS:-200000}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
SEQ_LEN="${SEQ_LEN:-2048}"
DATA_PATH="${DATA_PATH:-data/slimpajama_chunks_2048_qwen3.npy}"
PORT="${PORT:-29671}"
RESUME_FROM="${RESUME_FROM:-}"

OUT_DIR="${OUT_DIR:-outputs/paperC_qwen3base_heal_k${KEEP_FRONT}f${N_FRESH}}"
LOG_FILE="${LOG_FILE:-logs/paperC_qwen3base_heal_k${KEEP_FRONT}f${N_FRESH}.log}"

mkdir -p "$OUT_DIR" logs

# --- refuse to start if the Instruct model was passed by mistake --------------
EOS_ID=$("$PYTHON_BIN" - <<PY
import json
print(json.load(open("$MODEL_PATH/generation_config.json")).get("eos_token_id"))
PY
)
if [ "$EOS_ID" != "151643" ]; then
  echo "[FATAL] $MODEL_PATH has eos_token_id=$EOS_ID; expected 151643 (Qwen3-8B-Base)." >&2
  echo "        151645 / [151645, 151643] means this is Qwen3-8B-INSTRUCT, which is" >&2
  echo "        invalid for paperC (chat_template=False protocol). Refusing to launch." >&2
  exit 2
fi
echo "[launch] base model OK: eos_token_id=$EOS_ID (Qwen3-8B-Base)"

export WANDB_MODE=offline
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export HF_HOME="$PROJECT_ROOT/.hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo "[launch] keep_front=$KEEP_FRONT n_fresh=$N_FRESH -> $((KEEP_FRONT+N_FRESH)) layers"
echo "[launch] eff_bs = $BATCH_SIZE x $GRAD_ACCUM x 8 = $((BATCH_SIZE*GRAD_ACCUM*8)) (tokens/step = $((BATCH_SIZE*GRAD_ACCUM*8*SEQ_LEN)))"
echo "[launch] cosine horizon max_steps=$MAX_STEPS ; pre-registered read-out = step 121000"
echo "[launch] data=$DATA_PATH out=$OUT_DIR log=$LOG_FILE"

# fresh run truncates the log; resume APPENDS so pre-resume history survives.
[ -z "$RESUME_FROM" ] && : > "$LOG_FILE"

setsid nohup "$PYTHON_BIN" -m torch.distributed.run \
  --nproc_per_node 8 --nnodes 1 \
  --rdzv_backend c10d --rdzv_endpoint "127.0.0.1:$PORT" \
  scripts/train_qwen3_arch_probe2.py \
    --data_path "$DATA_PATH" \
    --output_dir "$OUT_DIR" \
    --model_path "$MODEL_PATH" \
    --keep_front_layers "$KEEP_FRONT" \
    --n_fresh_layers "$N_FRESH" \
    --max_steps "$MAX_STEPS" \
    --batch_size "$BATCH_SIZE" \
    --grad_accumulation_steps "$GRAD_ACCUM" \
    --seq_len "$SEQ_LEN" \
    --lr 1e-4 --min_lr 1e-5 \
    --lr_inherited 2e-5 --min_lr_inherited 2e-6 \
    --warmup_steps 150 \
    --gradient_checkpointing 1 \
    --save_every 500 \
    --milestone_every 5000 \
    --keep_last_n 3 \
    --keep_milestones 8 \
    --keep_steps 121000 \
    --log_every 20 \
    --device auto \
    ${RESUME_FROM:+--resume_from "$RESUME_FROM"} \
  >>"$LOG_FILE" 2>&1 &

echo "[launch] pid=$! ; tail -f $PROJECT_ROOT/$LOG_FILE"
