#!/usr/bin/env bash
# ============================================================================
# paperC task #251 — THE POWER WALL. MMLU-Pro letter-vs-content MC eval driver.
#
# WHY THIS EXISTS
# ---------------
# #248 built MMLU's exact letter-vs-content contrast on five non-MMLU MC
# benchmarks; #250 extended it to three non-OLMo families. Both returned
# PARTIAL, and BOTH hit the SAME wall: n. MMLU has 14042 items and a CI95
# half-width of 1.15 pp, so it can resolve its own -1.389 pp headline effect.
# The five #248 benchmarks have 500-2376 items and half-widths of 1.31-6.40 pp,
# so 52 of #250's 60 damaged cells were underpowered to have detected MMLU's
# effect AT ALL. That is not a null result, it is a non-observation -- and
# critically it is NOT an effect-size problem: arc_challenge's median damaged
# effect (-3.840 pp) is LARGER than MMLU's (-3.603 pp).
#
# Adding more small benchmarks can never fix this: to get a half-width under
# 1.389 pp openbookqa would need ~10615 items and its entire test set is 500.
# The only fix is a benchmark with MMLU-scale n. MMLU-Pro has n=12032.
#
# WHY MMLU-Pro SPECIFICALLY, beyond n
# -----------------------------------
# It is 10-way (A-J), so it attacks the WEAKEST part of paperC's rhetoric.
# #248 honestly recorded that on arc/obqa/piqa/csqa the best-constant letter
# floor is only +0.43 to +2.60 pp above chance, i.e. "chance badly misstates the
# null" is WEAK there. Measured on disk before any GPU was touched, MMLU-Pro's
# floor is always-A = 0.116606 vs naive 10-way chance 0.10 = +1.661 pp
# ABSOLUTE, but 1.166x RELATIVE -- a larger relative misstatement than MMLU's
# own 0.268908/0.25 = 1.076x. It is also the hardest test of the longest-option
# content null: with ~9.5 candidates per item the tie structure (and hence the
# tokenizer dependence #250 found) is under far more pressure than at 4-way.
#
# n_opt IS NOT CONSTANT: {10:9981, 9:801, 8:320, 7:158, 6:93, 5:52, 4:606, 3:21}
# so "chance" has two defensible readings -- naive 0.10 and mean(1/n_opt) =
# 0.110877. BOTH are reported; the floor beats both.
#
# ARMS
# ----
# Exactly the six OLMo-2 arms of #248 / gate-3, so every cell here is PAIRED
# with an existing MMLU cell and an existing five-benchmark cell. Optionally the
# 15 non-OLMo cross-family arms of #250 (eval-time front-N truncation, no heal,
# no training) via MODE=crossfamily.
#
# COST NOTE
# ---------
# MMLU-Pro is ~9.5x more candidate scorings per item than a 4-way task and 5.1x
# more items than arc_challenge: 227980 candidate forward-scorings per arm vs
# ~10k for arc_challenge. Expect ~10-20x the per-arm wall time of one #248 task.
#
# env in:  ROOT   node project root (zwfy6 real path on .73/.82/.104)
#          PY     python (conda torch-base; .venv is broken on H20)
#          MODE   olmo2 (default) | crossfamily
#          BS     per-shard batch size
#          MAXLEN token cap (1536 -> 0 truncation; max letter prompt is 1226)
# ============================================================================
set -u
ROOT="${ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
PY="${PY:-/opt/conda/envs/torch-base/bin/python}"
MODELS="${MODELS:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/models}"
BASE="${BASE:-../models/OLMo-2-1124-7B}"
MODE="${MODE:-olmo2}"
BS="${BS:-48}"
NGPU="${NGPU:-8}"
N_BOOT="${N_BOOT:-10000}"
# ⚠️ MAXLEN IS A PER-TOKENIZER QUANTITY. 1536 was measured with the **OLMo-2**
# tokenizer (letter prompt mean 185.9 tok, p99 592, MAX 1226) and is correct for
# OLMo-2 and Llama-3 (max 1226) but WRONG for the other two cross-family
# tokenizers, whose 32k/151k vocabularies encode the same 12032 items longer:
#
#   family          vocab    max encoded tok @ mmlu_pro   n_trunc @1536
#   olmo2_7b       100278             1226                    0
#   llama3_8b      128256             1226                    0
#   qwen3_8b_base  151669             1660                   20  (2 items)
#   llama2_7b       32000             1678                   40  (3 items)
#
# Measured on CPU by paperC/code/mmlu_pro_trunc_audit.py, which reproduces the
# 40/20/0/0 counts of the first #251 run exactly. Only THREE distinct items
# (10500, 11603, 11790 -- all 10-option) overflow at all; n_trunc counts
# candidate encodings (10 per item per interface), which is why it is a
# multiple of 10 and constant across rungs.
#
# 2048 clears the global max (1678) with 370 tok of headroom for every family,
# so n_trunc == 0 for all four tokenizers. The scoring script now HARD-ASSERTS
# n_trunc == 0 per shard (#251 only warned, and shipped 10 truncated cells).
# Do NOT lower this: silent left-truncation of the labelled option body changes
# the letter interface itself, and because the overflow set is tokenizer-
# specific it also breaks item-matching ACROSS families.
MAXLEN="${MAXLEN:-2048}"
TASKS="${TASKS:-mmlu_pro}"
RESULTS_ROOT="${RESULTS_ROOT:-mmlu_pro_letter_content_results}"
SCRIPT=scripts/eval_olmo2_mc_letter_content.py

cd "$ROOT" || exit 1

export HF_DATASETS_CACHE="$ROOT/data/hf_datasets_cache"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
# MMLU-Pro is read straight from parquet (no HF builder), so the 8 shards cannot
# race. Point at whichever disk has it; the loader also probes both.
if [ -n "${MMLU_PRO_PARQUET:-}" ]; then export MMLU_PRO_PARQUET; fi
mkdir -p logs "$RESULTS_ROOT"

if [ "$MODE" = "olmo2" ]; then
  # arm key -> "output_name|ckpt|keep_front|n_fresh"   (ckpt "-" = full base)
  # Names MIRROR olmo2_mmlu_content_results/<name> and
  # olmo2_mc_letter_content_results/<name> so the paired comparison is a name
  # lookup, not a mapping table.
  ARMS="${ARMS:-base keep14 keep12 keep10 keep8 shortgpt16}"
  spec() {
    case "$1" in
      base)       echo "7B_base|-|-|-" ;;
      keep8)      echo "7B_keep8_step121000|outputs/olmo2_probe2_7B_keep8fresh2/step121000.pt|8|2" ;;
      # An EARLIER point on OLMo-2 keep8's own heal trajectory. Exists so the
      # "is the Qwen3 heal arm merely EARLY?" question can be answered with a
      # measurement instead of an assertion: keep8@121000 is only 1 point, and a
      # single point cannot distinguish "OLMo-2 is non-degenerate because it is
      # OLMo-2" from "OLMo-2 is non-degenerate because it healed 121k steps".
      keep8_45000) echo "7B_keep8_step45000|outputs/olmo2_probe2_7B_keep8fresh2/step45000.pt|8|2" ;;
      keep10)     echo "7B_keep10_step83500|outputs/olmo2_probe2_7B_keep10fresh2/step83500.pt|10|2" ;;
      keep12)     echo "7B_keep12_step124000|outputs/olmo2_probe2_7B_keep12fresh2/step124000.pt|12|2" ;;
      keep14)     echo "7B_keep14_step200000|outputs/olmo2_probe2_7B_keep14fresh2/step200000.pt|14|2" ;;
      shortgpt16) echo "7B_shortgpt16_step200000|outputs/olmo2_probe2_7B_shortgpt16/step200000.pt|16|0" ;;
      # paperC heal-confound arm (HEAL_CONFOUND_PREREGISTRATION.md): a HEALED
      # front-8+fresh2 **Qwen3-8B-Base** ckpt, i.e. the within-family twin of the
      # un-healed `qwen3_8b_base_k8` cell in MODE=crossfamily.
      #
      # ⚠️ It MUST go through this --ckpt path, NOT MODE=crossfamily: the latter
      # does eval-time truncation of an INTACT model via --any_family and would
      # silently score an UN-healed front-8 Qwen3, i.e. re-measure the control
      # while labelling it the treatment. `model_family` is read from the ckpt by
      # _load_pruned_dispatch, so no family flag has to be threaded here.
      #
      # `qwen3heal:<step>` is step-parameterised because the read-out is a
      # TRAJECTORY (step -> accuracy), not a single cell. Milestones are read from
      # `..._pinned/`, NOT the live `outputs/paperC_qwen3base_heal_k8f2/`, because
      # that dir is under active rotation (keep_last_n=3, milestone_every=5000,
      # keep_milestones=8): non-multiples of 5000 are deleted a few saves later,
      # so scoring straight out of the live dir races the rotator and a mid-run
      # trajectory point can vanish between enumeration and load. The pinned dir
      # holds HARDLINKS (same inodes, 0 extra bytes) and the rotator only globs
      # its own output_dir, so pinned copies survive.
      qwen3heal:*)
        _S="${1##*:}"
        case "$_S" in ''|*[!0-9]*) echo ""; return 1 ;; esac
        echo "qwen3base_heal_k8f2_step${_S}|outputs/paperC_qwen3base_heal_k8f2_pinned/step${_S}.pt|8|2" ;;
      *) echo ""; return 1 ;;
    esac
  }
else
  # #250's 15 cross-family arms. Damage is eval-time front-N truncation.
  ARMS="${ARMS:-llama2_7b:base llama3_8b:base qwen3_8b_base:base \
llama2_7b:8 llama3_8b:8 qwen3_8b_base:8 \
llama2_7b:12 llama3_8b:12 qwen3_8b_base:12 \
llama2_7b:14 llama3_8b:14 qwen3_8b_base:14 \
llama2_7b:10 llama3_8b:10 qwen3_8b_base:10}"
  # ⚠️ Qwen3-8B-**Base**, NOT models/Qwen--Qwen3-8b nor the Qwen3-8b-local
  # symlink -- those are Qwen3-8B-*Instruct* (eos 151645 = <|im_end|>, ctx
  # 40960). Under chat_template=False an Instruct model is not a valid base arm.
  # The judgement criterion is eos_token_id + ctx length, NOT the presence of a
  # chat_template (both have one).
  family_path() {
    case "$1" in
      llama2_7b)      echo "$MODELS/Llama--Llama2-7b" ;;
      llama3_8b)      echo "$MODELS/Llama--Llama3-8b" ;;
      qwen3_8b_base)  echo "$MODELS/Qwen3-8B-Base" ;;
      *) echo ""; return 1 ;;
    esac
  }
fi

echo "[$(date '+%F %T')] #251 MMLU-Pro letter/content driver on $(hostname)"
echo "[$(date '+%F %T')] root=$ROOT mode=$MODE bs=$BS maxlen=$MAXLEN tasks=$TASKS"
echo "[$(date '+%F %T')] results_root=$RESULTS_ROOT"
echo "[$(date '+%F %T')] arms: $ARMS"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

# cardinality + schema check on CPU before a single card is touched
$PY $SCRIPT --prepare_data --tasks "$TASKS" \
  > logs/mmlu_pro_lc_prepare.log 2>&1 || { echo "PREPARE FAILED"; cat logs/mmlu_pro_lc_prepare.log; exit 1; }
cat logs/mmlu_pro_lc_prepare.log

FAILED=""
for A in $ARMS; do
  if [ "$MODE" = "olmo2" ]; then
    SPEC="$(spec "$A")" || { echo "unknown arm $A"; exit 1; }
    NAME="$(echo "$SPEC" | cut -d'|' -f1)"
    CKPT="$(echo "$SPEC" | cut -d'|' -f2)"
    KF="$(echo "$SPEC" | cut -d'|' -f3)"
    NF="$(echo "$SPEC" | cut -d'|' -f4)"
    MODEL="$BASE"; MARG=""
    if [ "$CKPT" = "-" ]; then
      CKARG=""
    else
      if [ ! -f "$CKPT" ]; then
        echo "[$(date '+%F %T')] FATAL: ckpt missing on this disk: $ROOT/$CKPT"
        exit 1
      fi
      CKARG="--ckpt $CKPT --keep_front_layers $KF --n_fresh_layers $NF"
    fi
  else
    FAM="${A%%:*}"; KEEP="${A##*:}"
    MODEL="$(family_path "$FAM")" || { echo "unknown family $FAM"; exit 1; }
    [ -d "$MODEL" ] || { echo "[$(date '+%F %T')] FATAL: model dir absent: $MODEL"; exit 1; }
    MARG="--any_family"
    if [ "$KEEP" = "base" ]; then NAME="${FAM}_base"; CKARG=""
    else NAME="${FAM}_k${KEEP}"; CKARG="--keep_front_layers $KEEP"; fi
  fi

  # cheap resume: skip an arm whose per-task summaries are all present
  DONE=1
  for T in $(echo "$TASKS" | tr ',' ' '); do
    [ -f "$RESULTS_ROOT/$NAME/summary_${T}.json" ] || DONE=0
  done
  if [ "$DONE" = "1" ]; then
    echo "[$(date '+%F %T')] SKIP $NAME (all per-task summaries present)"; continue
  fi

  echo "[$(date '+%F %T')] ===== $NAME  model=$MODEL ${CKARG:-（intact）} ====="
  T0=$(date +%s)
  for g in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$g $PY $SCRIPT \
      --base_model "$MODEL" $MARG $CKARG \
      --tasks "$TASKS" --num_shards $NGPU --shard_index $g \
      --batch_size $BS --max_len $MAXLEN --add_bos 0 --desc_style none \
      --results_root "$RESULTS_ROOT" --output_name "$NAME" \
      > "logs/mmlu_pro_lc_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  T1=$(date +%s)
  echo "[$(date '+%F %T')] $NAME scoring done in $((T1-T0))s"

  # hard shard check BEFORE merge. NEVER merge a half set (memory/
  # kill-remote-gpu-job-by-pid-not-pkill: a silent 5/8 merge destroys the口径).
  NS=$(ls "$RESULTS_ROOT/$NAME"/shard*of${NGPU}.json 2>/dev/null | wc -l)
  if [ "$NS" != "$NGPU" ]; then
    echo "[$(date '+%F %T')] SHARD FAIL $NAME: ${NS}/${NGPU} shard json -- NOT merging"
    FAILED="$FAILED $NAME"; continue
  fi
  SKIP=0
  for T in $(echo "$TASKS" | tr ',' ' '); do
    NT=$(ls "$RESULTS_ROOT/$NAME"/per_example_${T}_shard*of${NGPU}.jsonl 2>/dev/null | wc -l)
    if [ "$NT" != "$NGPU" ]; then
      echo "[$(date '+%F %T')] SHARD FAIL $NAME/$T: ${NT}/${NGPU} -- NOT merging"
      FAILED="$FAILED ${NAME}/${T}"; SKIP=1
    fi
  done
  [ "$SKIP" = "1" ] && continue

  # n_trunc must be 0 at MAXLEN (global max over all four tokenizers is 1678).
  # NOTE: `bc` is NOT installed on the H20 nodes, so this uses awk -- an earlier
  # version piped to `bc` and printed a cosmetic "?" while the true sum was 0.
  # The scoring script now also hard-asserts this per shard, so a truncated arm
  # dies before writing a summary and is caught by the SHARD FAIL check above;
  # this block stays as a belt-and-braces check on the merged total.
  NTR=$(grep -ho "trunc=[0-9]*" "logs/mmlu_pro_lc_${NAME}_shard"*.log \
        | cut -d= -f2 | awk '{s+=$1} END {print s+0}')
  echo "[$(date '+%F %T')] $NAME total n_trunc across shards = $NTR (expect 0)"
  if [ "$NTR" != "0" ]; then
    echo "[$(date '+%F %T')] TRUNCATION FAILURE $NAME: n_trunc=$NTR != 0 -- the"
    echo "  labelled option body was left-truncated, which changes the letter"
    echo "  INTERFACE itself, and the overflow set is tokenizer-specific so"
    echo "  item-matching across families is broken too. Raise MAXLEN (probe with"
    echo "  paperC/code/mmlu_pro_trunc_audit.py) and re-run this arm. NOT merging."
    FAILED="$FAILED ${NAME}:trunc"
    continue
  fi

  # merge_task() asserts 8/8 shards, n_scored == EXPECTED_N, n_nan == 0, and
  # RAISES rather than merging a partial set.
  $PY $SCRIPT --merge --output_name "$NAME" --tasks "$TASKS" \
      --num_shards $NGPU --n_boot "$N_BOOT" --results_root "$RESULTS_ROOT" \
      2>&1 | tee "logs/mmlu_pro_lc_${NAME}_merge.log"
  NTASK=$(echo "$TASKS" | tr ',' ' ' | wc -w)
  NSUM=$(ls "$RESULTS_ROOT/$NAME"/summary_*.json 2>/dev/null | wc -l)
  NM=$(grep -c "\[merge\]" "logs/mmlu_pro_lc_${NAME}_merge.log" || true)
  if [ "$NM" != "$NTASK" ] || [ "$NSUM" != "$NTASK" ]; then
    echo "[$(date '+%F %T')] MERGE FAIL $NAME: ${NM}/${NTASK} merge lines, ${NSUM}/${NTASK} summaries"
    FAILED="$FAILED ${NAME}:merge"
  else
    echo "[$(date '+%F %T')] MERGE OK $NAME: ${NSUM}/${NTASK} summaries"
  fi
done

if [ -n "$FAILED" ]; then
  echo "[$(date '+%F %T')] ===== DONE WITH FAILURES:$FAILED ====="
  exit 1
fi
echo "[$(date '+%F %T')] ===== #251 MMLU-Pro letter/content ALL ARMS DONE ====="
