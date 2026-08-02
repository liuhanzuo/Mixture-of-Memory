#!/usr/bin/env bash
# One-click P2.4 "general SFT interface repairability" pipeline for OLMo-2
# (Paper B). DRY-by-default: prints every command and only EXECUTES when RUN=1.
# NO GPU is used unless RUN=1 -- this script is the deliverable MAIN launches on a
# FREE 8-GPU node once Paper A frees it.
#
# What it wires (identical data / token-budget / optimizer / seed across arms):
#   Stage D  prepare general instruction SFT data (once) + n-gram overlap audit
#   Stage 0  PRE-SFT eval of all arms (baseline, no SFT): PPL + both MMLU protocols
#            + core6 + PopQA/TriviaQA closed-book
#   Stage 1  SFT each arm (full32 / keep14fresh2 / shortgpt16 [+ keep14 NTP control])
#   Stage 2  POST-SFT eval of every arm's final.pt (same eval battery as Stage 0)
#
# Arms (P2.4 minimal set + compute control):
#   full32        vanilla OLMo-2-1124-7B   (no ckpt)
#   keep14fresh2  outputs/olmo2_probe2_7B_keep14fresh2/final.pt   (keep14 + fresh2)
#   shortgpt16    outputs/olmo2_probe2_7B_shortgpt16/final.pt     (keep_front16/fresh0)
#   keep14_ntp    keep14fresh2 ckpt, NTP continuation on Dolmino (equal-token compute control)
#
# Usage:
#   bash scripts/_run_olmo2_p24_sft_pipeline.sh                 # DRY: print all commands
#   RUN=1 STAGE=data  bash scripts/_run_olmo2_p24_sft_pipeline.sh   # only prep+audit
#   RUN=1 STAGE=pre   bash scripts/_run_olmo2_p24_sft_pipeline.sh   # only pre-SFT eval
#   RUN=1 STAGE=sft   ARM=keep14fresh2 bash scripts/...            # SFT one arm (8 GPU)
#   RUN=1 STAGE=post  ARM=keep14fresh2 bash scripts/...            # post-SFT eval one arm
#   (STAGE=all runs data->pre->sft(all arms serially)->post; heavy -- normally run
#    per-STAGE / per-ARM. Each 8-GPU SFT should get its own node/window.)
set -u

# --------- node-specific knobs (override per disk/node) ----------------------
ROOT="${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$ROOT"
PY="${PYTHON_BIN:-$ROOT/.venv/bin/python}"          # wzc1 B200/.venv; diskB use olmo2_venv or torch-base
BASE="${MODEL_PATH:-../models/OLMo-2-1124-7B}"
VAL="${VAL_PATH:-data/dolmino_now_val.npy}"
NTP_DATA="${NTP_DATA:-/dev/shm/dolmino_now15b.npy}" # compute-control NTP corpus
NPROC="${NPROC:-8}"
BS="${BS:-4}"; GA="${GA:-4}"                        # H20 eff128; B200 -> BS=16 GA=1
MAX_STEPS="${MAX_STEPS:-2000}"                      # SHARED token budget across arms
LR="${LR:-1e-5}"; MIN_LR="${MIN_LR:-1e-6}"; WARMUP="${WARMUP:-100}"
SEED="${SEED:-42}"
EVAL_BS="${EVAL_BS:-8}"

# --------- SFT data (shared) --------------------------------------------------
SFT_DIR="${SFT_DIR:-data/olmo2_sft}"
SFT_TAG="${SFT_TAG:-tulu3_general}"
TOKEN_BUDGET="${TOKEN_BUDGET:-250000000}"           # ~1 epoch general instructions
SFT_IDS="$SFT_DIR/${SFT_TAG}_input_ids.npy"
SFT_LAB="$SFT_DIR/${SFT_TAG}_labels.npy"
SFT_TXT="$SFT_DIR/${SFT_TAG}_text.jsonl"

STAGE="${STAGE:-all}"
ARM="${ARM:-all}"

export http_proxy="${http_proxy:-http://hy-proxy.woa.com:3128}"
export https_proxy="${https_proxy:-http://hy-proxy.woa.com:3128}"
export all_proxy="${all_proxy:-http://hy-proxy.woa.com:3128}"
export no_proxy="localhost,127.0.0.1,.oa.com,.woa.com,.local"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$ROOT/data/hf_datasets_cache}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p logs "$SFT_DIR" olmo2_ppl_results olmo2_downstream_results \
         olmo2_mmlu_content_results olmo2_closedbook_results "$HF_DATASETS_CACHE"

CORE="hellaswag,arc_challenge,arc_easy,piqa,winogrande,openbookqa"
KNOW="mmlu,lambada_openai,boolq,commonsense_qa,social_iqa"

run() { echo "+ $*"; [ "${RUN:-0}" = "1" ] && "$@"; }

# arm -> (ckpt, keep, fresh). full32 has empty ckpt (full-base mode).
arm_ckpt() { case "$1" in
  full32)        echo "" ;;
  keep14fresh2)  echo "outputs/olmo2_probe2_7B_keep14fresh2/final.pt" ;;
  shortgpt16)    echo "outputs/olmo2_probe2_7B_shortgpt16/final.pt" ;;
  keep14_ntp)    echo "outputs/olmo2_probe2_7B_keep14fresh2/final.pt" ;;
  *) echo "" ;; esac; }
arm_out()  { echo "outputs/olmo2_p24_sft_$1"; }

# ===========================================================================
# helpers: eval an arbitrary (ckpt) model. CK="" -> full-base mode.
#   eval_battery NAME_PREFIX CKPT   -- runs PPL + letter-MMLU(via downstream) +
#   content-MMLU + core6 + closedbook, 8-GPU shard+merge each.
# ===========================================================================
eval_battery() {
  local PFX=$1 CK=$2
  local CKARG=(); [ -n "$CK" ] && CKARG=(--ckpt "$CK")

  echo "### eval_battery $PFX (ckpt='${CK:-FULL-BASE}')"

  # ---- 1. held-out Dolmino PPL (8 shard + merge) ----
  for g in $(seq 0 $((NPROC-1))); do
    run env CUDA_VISIBLE_DEVICES=$g "$PY" scripts/eval_olmo2_probe2_ppl.py \
      --base_model "$BASE" "${CKARG[@]}" --val_path "$VAL" \
      --num_shards $NPROC --shard_index $g --batch_size 4 --output_name "${PFX}_ppl" \
      ">logs/p24_${PFX}_ppl_shard${g}.log" "2>&1" "&"
  done
  [ "${RUN:-0}" = "1" ] && wait
  run "$PY" scripts/eval_olmo2_probe2_ppl.py --merge --output_name "${PFX}_ppl"

  # ---- 2. letter-protocol MMLU + core6 + know5 (downstream MC harness) ----
  run "$PY" scripts/eval_olmo2_probe2_downstream.py --prepare_data --tasks "$CORE,$KNOW"
  for g in $(seq 0 $((NPROC-1))); do
    run env CUDA_VISIBLE_DEVICES=$g "$PY" scripts/eval_olmo2_probe2_downstream.py \
      --base_model "$BASE" "${CKARG[@]}" --tasks "$CORE,$KNOW" \
      --num_shards $NPROC --shard_index $g --batch_size $EVAL_BS --save_per_example \
      --output_name "${PFX}_mc" ">logs/p24_${PFX}_mc_shard${g}.log" "2>&1" "&"
  done
  [ "${RUN:-0}" = "1" ] && wait
  run "$PY" scripts/eval_olmo2_probe2_downstream.py --merge --output_name "${PFX}_mc"

  # ---- 3. content-protocol MMLU (=P0.6; call the existing harness, do NOT reimpl) ----
  run "$PY" scripts/eval_olmo2_mmlu_content.py --prepare_data
  for g in $(seq 0 $((NPROC-1))); do
    run env CUDA_VISIBLE_DEVICES=$g "$PY" scripts/eval_olmo2_mmlu_content.py \
      --base_model "$BASE" "${CKARG[@]}" \
      --num_shards $NPROC --shard_index $g --batch_size $EVAL_BS \
      --output_name "${PFX}_mmlu_content" \
      ">logs/p24_${PFX}_mmluc_shard${g}.log" "2>&1" "&"
  done
  [ "${RUN:-0}" = "1" ] && wait
  run "$PY" scripts/eval_olmo2_mmlu_content.py --merge --output_name "${PFX}_mmlu_content"

  # ---- 4. PopQA + TriviaQA closed-book (new harness; P0.3 + P2.4) ----
  run "$PY" scripts/eval_olmo2_closedbook_qa.py --prepare_data --tasks popqa,triviaqa
  for g in $(seq 0 $((NPROC-1))); do
    run env CUDA_VISIBLE_DEVICES=$g "$PY" scripts/eval_olmo2_closedbook_qa.py \
      --base_model "$BASE" "${CKARG[@]}" --tasks popqa,triviaqa \
      --num_shards $NPROC --shard_index $g --batch_size $EVAL_BS \
      --output_name "${PFX}_closedbook" \
      ">logs/p24_${PFX}_cb_shard${g}.log" "2>&1" "&"
  done
  [ "${RUN:-0}" = "1" ] && wait
  run "$PY" scripts/eval_olmo2_closedbook_qa.py --merge --output_name "${PFX}_closedbook"
  echo "### eval_battery $PFX DONE"
}

# sft one arm
sft_arm() {
  local A=$1
  local CK; CK=$(arm_ckpt "$A")
  local OUT; OUT=$(arm_out "$A")
  local CKARG=(); [ -n "$CK" ] && CKARG=(--ckpt "$CK")
  local DATAARG
  if [ "$A" = "keep14_ntp" ]; then
    DATAARG=(--data_mode ntp --data_path "$NTP_DATA")
  else
    DATAARG=(--data_mode sft --sft_ids "$SFT_IDS" --sft_labels "$SFT_LAB")
  fi
  echo "### SFT arm=$A -> $OUT  (ckpt='${CK:-FULL-BASE}')"
  run env NCCL_IB_DISABLE=1 WANDB_MODE=offline \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" -m torch.distributed.run --standalone --nproc_per_node $NPROC \
    scripts/train_olmo2_sft.py \
      --base_model "$BASE" "${CKARG[@]}" --arm_name "$A" \
      "${DATAARG[@]}" \
      --output_dir "$OUT" \
      --seq_len 2048 --batch_size $BS --grad_accumulation_steps $GA \
      --max_steps $MAX_STEPS --lr $LR --min_lr $MIN_LR --warmup_steps $WARMUP \
      --weight_decay 0.1 --grad_clip 1.0 --gradient_checkpointing 1 \
      --save_every 500 --seed $SEED \
      ">>logs/p24_sft_${A}.log" "2>&1"
}

ARMS_ALL="full32 keep14fresh2 shortgpt16 keep14_ntp"
arms_selected() { [ "$ARM" = "all" ] && echo "$ARMS_ALL" || echo "$ARM"; }

# ===========================================================================
case "$STAGE" in
  data|all)
    echo "===== STAGE data: prepare SFT data + overlap audit ====="
    run "$PY" scripts/prepare_olmo2_sft_data.py \
      --dataset allenai/tulu-3-sft-mixture --split train \
      --tokenizer_path "$BASE" --out_dir "$SFT_DIR" --tag "$SFT_TAG" \
      --seq_len 2048 --token_budget $TOKEN_BUDGET --seed $SEED
    run "$PY" scripts/audit_olmo2_sft_overlap.py \
      --sft_text "$SFT_TXT" --out "$SFT_DIR/${SFT_TAG}_overlap_audit.json" \
      --n 8 --hit_threshold 0.5
    [ "$STAGE" = "data" ] && exit 0 ;;
esac

case "$STAGE" in
  pre|all)
    echo "===== STAGE pre: PRE-SFT baseline eval of all arms ====="
    eval_battery "pre_full32"       ""
    eval_battery "pre_keep14fresh2" "$(arm_ckpt keep14fresh2)"
    eval_battery "pre_shortgpt16"   "$(arm_ckpt shortgpt16)"
    [ "$STAGE" = "pre" ] && exit 0 ;;
esac

case "$STAGE" in
  sft|all)
    echo "===== STAGE sft: fine-tune arm(s) [$(arms_selected)] ====="
    for A in $(arms_selected); do sft_arm "$A"; done
    [ "$STAGE" = "sft" ] && exit 0 ;;
esac

case "$STAGE" in
  post|all)
    echo "===== STAGE post: POST-SFT eval of arm(s) [$(arms_selected)] ====="
    for A in $(arms_selected); do
      OUT=$(arm_out "$A"); eval_battery "post_${A}" "$OUT/final.pt"
    done
    ;;
esac

echo "===== P2.4 pipeline STAGE=$STAGE DONE (RUN=${RUN:-0}) ====="
