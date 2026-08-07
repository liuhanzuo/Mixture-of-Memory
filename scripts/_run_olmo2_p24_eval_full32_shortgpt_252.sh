#!/usr/bin/env bash
# Paper B P2.4 pre/post-SFT eval battery for the wzc1-side arms full32 and
# shortgpt16 on .252 (8×L20A cc10.0, wzc1). Same-arch pairing is mandatory:
# earlier tonight (status/PAPERB_CORE6_CROSSARCH_FLOOR.md) showed L20A cc10.0
# and H20 cc9.0 give different core6 numbers for bit-identical weights (28
# items flip, +0.156 pp avg, symmetric bf16 kernel noise, signs differ per
# task). Consequence: P2.4's per-item McNemar / paired bootstrap must be
# computed within a single architecture. This driver runs BOTH pre-SFT and
# post-SFT legs for full32 and shortgpt16 on the same L20A card so the
# comparison is architecturally clean.
#
# This is the wzc1 analogue of scripts/_run_olmo2_p24_eval_keep14_73.sh
# (which runs the keep14fresh2 arm on .73, zwfy6). Do NOT run any code path
# that would overwrite that sibling's outputs.
#
# Legs (four batteries):
#   full32 pre   : vanilla base (../models/OLMo-2-1124-7B, no --ckpt)
#                  → *_wzc1 suffix; the same base all three P2.4 arms trained from.
#   full32 post  : outputs/olmo2_p24_sft_full32/final.pt         (87.6 GB, step=842)
#   shortgpt pre : outputs/olmo2_probe2_7B_shortgpt16/step200000.pt (48.7 GB, keep_front=16 n_fresh=0)
#                  → *_wzc1 suffix; existing wzc1 dir 7B_shortgpt_step200000{,_know}
#                    lacks --save_per_example → rerun.
#   shortgpt post: outputs/olmo2_p24_sft_shortgpt16/final.pt      (48.7 GB, step=842, keep_front=16 n_fresh=0)
#
# Output names (paper Table 4 anchor convention + `_wzc1` suffix on the pre-SFT
# anchors — the sibling on .73 uses no suffix, so we do NOT collide with anything
# it produced):
#
#   full32 pre   : olmo2_ppl_results/7B_full32_base_wzc1/summary.json
#                  olmo2_downstream_results/7B_full32_base_wzc1{,_know}/summary.json
#                  olmo2_mmlu_content_results/7B_full32_base_wzc1/summary.json
#                  olmo2_closedbook_results/7B_full32_base_wzc1/summary.json
#   full32 post  : 7B_p24_sft_full32_final{,_know}/summary.json  (same tree)
#   shortgpt pre : 7B_shortgpt16_step200000_wzc1{,_know}/summary.json  (same tree)
#   shortgpt post: 7B_p24_sft_shortgpt16_final{,_know}/summary.json  (same tree)
#
# Per-item predictions retained for pre/post pairing:
#   downstream MC:  per_example_<task>.jsonl (via --save_per_example)
#   MMLU content :  per_example_mmlu.jsonl   (default in harness)
#   closed-book  :  per_example_<task>.jsonl (default in harness)
#
# Shard invariant: every eval must have 8/8 shard{i}of8.json files before merge;
# a partial merge is a hard error (memory: kill-remote-gpu-job-by-pid-not-pkill).
#
# Chat template: chat_template=False everywhere; --add_bos 0 is harness default;
# OLMo-2 base is not SFT'd/RLHF'd (memory: paper-eval-chat-false-mandatory).
#
# Launch (on .252):
#   setsid nohup bash scripts/_run_olmo2_p24_eval_full32_shortgpt_252.sh \
#     > logs/p24_eval_full32_shortgpt_252.log 2>&1 &
set -u

ROOT=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory
cd "$ROOT"
PY=/opt/conda/envs/torch-base/bin/python
BASE=../models/OLMo-2-1124-7B
VAL=data/dolmino_now_val.npy

export http_proxy=http://hy-proxy.woa.com:3128
export https_proxy=http://hy-proxy.woa.com:3128
export all_proxy=http://hy-proxy.woa.com:3128
export no_proxy="localhost,127.0.0.1,.oa.com,.woa.com,.local"
export HF_DATASETS_CACHE="$ROOT/data/hf_datasets_cache"
mkdir -p logs olmo2_ppl_results olmo2_downstream_results \
         olmo2_mmlu_content_results olmo2_closedbook_results "$HF_DATASETS_CACHE"

CORE="hellaswag,arc_challenge,arc_easy,piqa,winogrande,openbookqa"
KNOW="mmlu,lambada_openai,boolq,commonsense_qa,social_iqa"

# ---- Leg definitions ----
# full32 pre-SFT: vanilla HF safetensors base — pass CKPT="" so eval scripts
# take the base-model path (no --ckpt argument).
FULL32_PRE_CKPT=""
FULL32_PRE_NAME="7B_full32_base_wzc1"

FULL32_POST_CKPT="outputs/olmo2_p24_sft_full32/final.pt"
FULL32_POST_NAME="7B_p24_sft_full32_final"

SHORTGPT_PRE_CKPT="outputs/olmo2_probe2_7B_shortgpt16/step200000.pt"
SHORTGPT_PRE_NAME="7B_shortgpt16_step200000_wzc1"

SHORTGPT_POST_CKPT="outputs/olmo2_p24_sft_shortgpt16/final.pt"
SHORTGPT_POST_NAME="7B_p24_sft_shortgpt16_final"

# ---- pre-flight ----
for CK in "$FULL32_POST_CKPT" "$SHORTGPT_PRE_CKPT" "$SHORTGPT_POST_CKPT"; do
  if [ ! -f "$CK" ]; then
    echo "[$(date '+%F %T')] FATAL: ckpt not found: $CK"; exit 2
  fi
done
if [ ! -d "$BASE" ]; then
  echo "[$(date '+%F %T')] FATAL: base model dir not found: $BASE"; exit 2
fi

# assert_8shards <results_root> <NAME> <shard_pat>
#   Confirms all 8 shard files exist; if not, ABORT the merge (partial merge =
#   silent contamination). Called before every merge step.
assert_8shards () {
  local RROOT=$1 NAME=$2 PAT=$3
  local D="$RROOT/$NAME"
  local MISS=0
  for g in 0 1 2 3 4 5 6 7; do
    if ! ls "$D"/${PAT/\{g\}/$g} >/dev/null 2>&1; then
      echo "[$(date '+%F %T')] SHARD MISSING: $D/${PAT/\{g\}/$g}"
      MISS=$((MISS+1))
    fi
  done
  if [ $MISS -gt 0 ]; then
    echo "[$(date '+%F %T')] ABORT merge for $NAME: $MISS/8 shards missing (partial merge would silently contaminate results). Rerun the failed shards."
    return 1
  fi
  return 0
}

# ---------------- eval helpers (8-GPU shard+merge; --save_per_example on MC) ----

# run_ppl NAME CKPT
run_ppl () {
  local NAME=$1 CKPT=$2
  local D="olmo2_ppl_results/$NAME"
  if [ -f "$D/summary.json" ]; then
    echo "[$(date '+%F %T')] PPL $NAME ALREADY DONE, skipping"; return 0
  fi
  echo "[$(date '+%F %T')] --- PPL $NAME (ckpt=${CKPT:-<base>}) ---"
  local CKARG=(); [ -n "$CKPT" ] && CKARG=(--ckpt "$CKPT")
  for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g LOCAL_RANK=0 RANK=$g $PY scripts/eval_olmo2_probe2_ppl.py \
      --base_model "$BASE" "${CKARG[@]}" \
      --val_path "$VAL" --num_shards 8 --shard_index $g --batch_size 4 \
      --output_name "$NAME" \
      > "logs/olmo2_ppl_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  assert_8shards "olmo2_ppl_results" "$NAME" "shard{g}of8.json" || return 1
  $PY scripts/eval_olmo2_probe2_ppl.py --merge --output_name "$NAME" 2>&1
  echo "[$(date '+%F %T')] PPL summary ($NAME):"; cat "$D/summary.json" 2>/dev/null; echo
}

# run_downstream NAME CKPT TASKS
# Uses --save_per_example so pre/post McNemar + paired bootstrap can be run.
run_downstream () {
  local NAME=$1 CKPT=$2 TASKS=$3
  local D="olmo2_downstream_results/$NAME"
  if [ -f "$D/summary.json" ]; then
    echo "[$(date '+%F %T')] downstream $NAME ALREADY DONE, skipping"; return 0
  fi
  echo "[$(date '+%F %T')] --- downstream $NAME (ckpt=${CKPT:-<base>} tasks=$TASKS) ---"
  local CKARG=(); [ -n "$CKPT" ] && CKARG=(--ckpt "$CKPT")
  $PY scripts/eval_olmo2_probe2_downstream.py --prepare_data --tasks "$TASKS" \
    > "logs/olmo2_downstream_${NAME}_prepare.log" 2>&1 || true
  for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g LOCAL_RANK=0 RANK=$g $PY scripts/eval_olmo2_probe2_downstream.py \
      --base_model "$BASE" "${CKARG[@]}" --tasks "$TASKS" \
      --num_shards 8 --shard_index $g --batch_size 8 \
      --save_per_example \
      --output_name "$NAME" \
      > "logs/olmo2_downstream_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  assert_8shards "olmo2_downstream_results" "$NAME" "shard{g}of8.json" || return 1
  $PY scripts/eval_olmo2_probe2_downstream.py --merge --output_name "$NAME" 2>&1
  echo "[$(date '+%F %T')] downstream summary ($NAME):"; cat "$D/summary.json" 2>/dev/null | head -40; echo
}

# run_mmlu_content NAME CKPT  (both letter + content protocols; per-item default)
run_mmlu_content () {
  local NAME=$1 CKPT=$2
  local D="olmo2_mmlu_content_results/$NAME"
  if [ -f "$D/summary.json" ]; then
    echo "[$(date '+%F %T')] mmlu-content $NAME ALREADY DONE, skipping"; return 0
  fi
  echo "[$(date '+%F %T')] --- mmlu-content $NAME (ckpt=${CKPT:-<base>}) ---"
  local CKARG=(); [ -n "$CKPT" ] && CKARG=(--ckpt "$CKPT")
  $PY scripts/eval_olmo2_mmlu_content.py --prepare_data \
    > "logs/olmo2_mmluc_${NAME}_prepare.log" 2>&1 || true
  for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g LOCAL_RANK=0 RANK=$g $PY scripts/eval_olmo2_mmlu_content.py \
      --base_model "$BASE" "${CKARG[@]}" \
      --num_shards 8 --shard_index $g --batch_size 8 \
      --output_name "$NAME" \
      > "logs/olmo2_mmluc_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  assert_8shards "olmo2_mmlu_content_results" "$NAME" "shard{g}of8.json" || return 1
  $PY scripts/eval_olmo2_mmlu_content.py --merge --output_name "$NAME" 2>&1
  echo "[$(date '+%F %T')] mmlu-content summary ($NAME):"; cat "$D/summary.json" 2>/dev/null | head -40; echo
}

# run_closedbook NAME CKPT
run_closedbook () {
  local NAME=$1 CKPT=$2
  local D="olmo2_closedbook_results/$NAME"
  if [ -f "$D/summary.json" ]; then
    echo "[$(date '+%F %T')] closedbook $NAME ALREADY DONE, skipping"; return 0
  fi
  echo "[$(date '+%F %T')] --- closedbook $NAME (ckpt=${CKPT:-<base>}) ---"
  local CKARG=(); [ -n "$CKPT" ] && CKARG=(--ckpt "$CKPT")
  $PY scripts/eval_olmo2_closedbook_qa.py --prepare_data --tasks popqa,triviaqa \
    > "logs/olmo2_cb_${NAME}_prepare.log" 2>&1 || true
  for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g LOCAL_RANK=0 RANK=$g $PY scripts/eval_olmo2_closedbook_qa.py \
      --base_model "$BASE" "${CKARG[@]}" --tasks popqa,triviaqa \
      --num_shards 8 --shard_index $g --batch_size 8 \
      --output_name "$NAME" \
      > "logs/olmo2_cb_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  assert_8shards "olmo2_closedbook_results" "$NAME" "shard{g}of8.json" || return 1
  $PY scripts/eval_olmo2_closedbook_qa.py --merge --output_name "$NAME" 2>&1
  echo "[$(date '+%F %T')] closedbook summary ($NAME):"; cat "$D/summary.json" 2>/dev/null | head -40; echo
}

# Runs the full 5-harness battery on one leg. NAME chosen with the paper-table
# convention plus the `_wzc1` suffix on pre-SFT anchors for provenance.
run_battery () {
  local NAME=$1 CKPT=$2
  echo "[$(date '+%F %T')] ===== battery start: $NAME (ckpt=${CKPT:-<base>}) ====="
  run_ppl          "$NAME"        "$CKPT" || echo "[$(date '+%F %T')] $NAME PPL FAILED"
  run_downstream   "$NAME"        "$CKPT" "$CORE" || echo "[$(date '+%F %T')] $NAME core6 FAILED"
  run_downstream   "${NAME}_know" "$CKPT" "$KNOW" || echo "[$(date '+%F %T')] $NAME know5 FAILED"
  run_mmlu_content "$NAME"        "$CKPT" || echo "[$(date '+%F %T')] $NAME mmlu-content FAILED"
  run_closedbook   "$NAME"        "$CKPT" || echo "[$(date '+%F %T')] $NAME closedbook FAILED"
  echo "[$(date '+%F %T')] ===== battery done: $NAME ====="
}

echo "[$(date '+%F %T')] ======== P2.4 full32+shortgpt16 EVAL BATTERY (.252, wzc1) START ========"

# ---------------------------------------------------------------------------
# LEG 1 — full32 pre-SFT (vanilla base). Expect PPL ≈ 7.398 (paper Table 4
#         full-32L base row); MMLU ≈ 0.6053. Any large deviation is loud.
# ---------------------------------------------------------------------------
run_battery "$FULL32_PRE_NAME"     "$FULL32_PRE_CKPT"

# ---------------------------------------------------------------------------
# LEG 2 — full32 post-SFT (final.pt, step 842 from vanilla base).
# ---------------------------------------------------------------------------
run_battery "$FULL32_POST_NAME"    "$FULL32_POST_CKPT"

# ---------------------------------------------------------------------------
# LEG 3 — shortgpt16 pre-SFT. Expect PPL ≈ 9.7803 (paper Table 4 ShortGPT-16
#         row); MMLU ≈ 0.4739. Ckpt meta has keep_front=16 n_fresh=0
#         (verified: arm=shortgpt, keep_layer_indices=[0..12,16,17,31],
#         num_hidden_layers=16). load_pruned_model reads meta directly, so no
#         --keep_front_layers / --n_fresh_layers overrides are required.
# ---------------------------------------------------------------------------
run_battery "$SHORTGPT_PRE_NAME"   "$SHORTGPT_PRE_CKPT"

# ---------------------------------------------------------------------------
# LEG 4 — shortgpt16 post-SFT (final.pt, step=842). Same 16L shell as leg 3.
# ---------------------------------------------------------------------------
run_battery "$SHORTGPT_POST_NAME"  "$SHORTGPT_POST_CKPT"

echo "[$(date '+%F %T')] ======== P2.4 full32+shortgpt16 EVAL BATTERY (.252, wzc1) DONE ========"
echo "[$(date '+%F %T')] Outputs:"
for N in "$FULL32_PRE_NAME" "$FULL32_POST_NAME" "$SHORTGPT_PRE_NAME" "$SHORTGPT_POST_NAME"; do
  echo "  $N:"
  echo "    PPL              olmo2_ppl_results/$N/summary.json"
  echo "    core6            olmo2_downstream_results/$N/summary.json"
  echo "    know5            olmo2_downstream_results/${N}_know/summary.json"
  echo "    MMLU dual        olmo2_mmlu_content_results/$N/summary.json"
  echo "    closedbook       olmo2_closedbook_results/$N/summary.json"
done
