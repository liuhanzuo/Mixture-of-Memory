#!/usr/bin/env bash
# ============================================================================
# paperG gate-2 CROSS-FAMILY extension (task #250).
#
# WHAT THIS ADDS
# --------------
# #248 built MMLU's exact letter-vs-content contrast on five non-MMLU MC
# benchmarks (+ winogrande as a negative control) and ran it on the SIX OLMo-2-7B
# prune-then-heal arms. That closed "is the second-benchmark contrast really the
# same contrast", but it left the whole second-benchmark leg inside ONE model
# family. MMLU's headline is four-family (OLMo-2 / Llama-2-7B / Llama-3-8B /
# Qwen3-8B-Base); the second benchmark was one-family. This driver closes that
# asymmetry by running the #248 harness, unchanged, on the three NON-OLMo
# families.
#
# DAMAGE IS AN EVAL-TIME CONSTRUCTION, NOT TRAINING
# -------------------------------------------------
# Verified in eval_olmo2_probe2_ppl.py::load_truncated_any_family: it
# AutoModelForCausalLM.from_pretrained's the intact base, replaces
# `model.model.layers` with `layers[:N]`, syncs `config.num_hidden_layers` (and
# `layer_types` when present), and returns. No fresh block, no heal steps, no
# optimizer, no gradient. Identical to what gate-1's DAMAGED leg did
# (scripts/_a01_gate1_damaged_driver_21.sh), which is why the arm keys here are
# k8/k12 -- so the numbers are directly comparable with the MMLU cross-family
# table already in STATUS.json:gate1_third_model_family_DAMAGED.
#
# ARMS
# ----
# Per family: intact base, k14, k12, k10, k8 (front-N truncation, no heal).
# k8/k12 are the two rungs gate-1 DAMAGED already has on MMLU (mandatory for the
# head-to-head); k10/k14 extend the ladder so the "k14 is the last arm above its
# floor" ordering claim from #248 can be tested off OLMo too.
# Ordering is CORE FIRST (base, k8, k12 for all three families) so that a partial
# run still yields the comparable set.
#
# NOTE ON DEPTH: OLMo-2-7B / Llama-2-7B / Llama-3-8B all have 32 blocks, so kN is
# the same absolute AND relative depth. Qwen3-8B-Base has 36, so kN there is a
# slightly SMALLER fraction of the stack (k8 = 22.2% vs 25.0%). We keep absolute
# N to stay comparable with the archived MMLU cross-family numbers, and say so.
#
# env in:  ROOT   node project root (zwfy6 real path on .73/.82/.104)
#          PY     python (conda torch-base; .venv is broken on H20)
#          MODELS model root
#          BS_*   per-shard batch size (intact 8B models need the small one)
#          TASKS  comma-separated task list
#          ARMS   space-separated "<family>:<keep>" or "<family>:base"
# ============================================================================
set -u
ROOT="${ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
PY="${PY:-/opt/conda/envs/torch-base/bin/python}"
MODELS="${MODELS:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/models}"
# BS=48 is DELIBERATELY the same value #248 used on the OLMo-2 arms: bf16-autocast
# batch composition perturbs the low-order bits of the summed log-probs, so a
# different batch size is a (small) protocol difference. Measured on .73 with an
# intact Qwen3-8B-Base: peak 57-64 GiB of 97.8 (58-66%), and bs=32 vs bs=48 give
# IDENTICAL wall time on an arc_easy shard (19.8s vs 18.8s) -- the eval is
# compute-bound, not batch-bound, so raising bs further buys throughput nothing
# while adding OOM risk and a protocol delta. 8 independent single-GPU processes,
# so the per-GPU figure is the whole story.
BS_INTACT="${BS_INTACT:-48}"
BS_TRUNC="${BS_TRUNC:-48}"
NGPU="${NGPU:-8}"
N_BOOT="${N_BOOT:-10000}"
TASKS="${TASKS:-arc_challenge,arc_easy,openbookqa,commonsense_qa,piqa,winogrande}"
RESULTS_ROOT="${RESULTS_ROOT:-mc_lc_crossfamily_results}"
SCRIPT=scripts/eval_olmo2_mc_letter_content.py
# core rungs first (comparable with the archived MMLU cross-family table), then
# the ladder extension.
ARMS="${ARMS:-llama2_7b:base llama3_8b:base qwen3_8b_base:base \
llama2_7b:8 llama3_8b:8 qwen3_8b_base:8 \
llama2_7b:12 llama3_8b:12 qwen3_8b_base:12 \
llama2_7b:14 llama3_8b:14 qwen3_8b_base:14 \
llama2_7b:10 llama3_8b:10 qwen3_8b_base:10}"

cd "$ROOT" || exit 1

export HF_DATASETS_CACHE="$ROOT/data/hf_datasets_cache"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
mkdir -p logs "$RESULTS_ROOT"

family_path() {
  case "$1" in
    llama2_7b)      echo "$MODELS/Llama--Llama2-7b" ;;
    llama3_8b)      echo "$MODELS/Llama--Llama3-8b" ;;
    qwen3_8b_base)  echo "$MODELS/Qwen3-8B-Base" ;;
    *) echo ""; return 1 ;;
  esac
}

echo "[$(date '+%F %T')] crossfamily gate-2 driver on $(hostname)"
echo "[$(date '+%F %T')] root=$ROOT models=$MODELS tasks=$TASKS"
echo "[$(date '+%F %T')] arms: $ARMS"

# populate the dataset cache ONCE (CPU) so 8 shards do not race on the builder,
# and so the per-task cardinality asserts fire before any GPU is touched.
$PY $SCRIPT --prepare_data --tasks "$TASKS" \
  > logs/gate2_xf_prepare.log 2>&1 || { echo "PREPARE FAILED"; exit 1; }
cat logs/gate2_xf_prepare.log

FAILED=""
for SPEC in $ARMS; do
  FAM="${SPEC%%:*}"; KEEP="${SPEC##*:}"
  MODEL="$(family_path "$FAM")" || { echo "unknown family $FAM"; exit 1; }
  if [ ! -d "$MODEL" ]; then
    echo "[$(date '+%F %T')] FATAL: model dir absent on this disk: $MODEL"; exit 1
  fi
  if [ "$KEEP" = "base" ]; then
    NAME="${FAM}_base"; KARG=""; BS="$BS_INTACT"
  else
    NAME="${FAM}_k${KEEP}"; KARG="--keep_front_layers $KEEP"; BS="$BS_TRUNC"
  fi

  # cheap resume: skip an arm whose per-task summaries are all present
  DONE=1
  for T in $(echo "$TASKS" | tr ',' ' '); do
    [ -f "$RESULTS_ROOT/$NAME/summary_${T}.json" ] || DONE=0
  done
  if [ "$DONE" = "1" ]; then
    echo "[$(date '+%F %T')] SKIP $NAME (all per-task summaries present)"; continue
  fi

  echo "[$(date '+%F %T')] ===== $NAME  model=$MODEL keep=$KEEP bs=$BS ====="
  T0=$(date +%s)
  for g in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$g $PY $SCRIPT \
      --base_model "$MODEL" --any_family $KARG \
      --tasks "$TASKS" --num_shards $NGPU --shard_index $g \
      --batch_size $BS --add_bos 0 --desc_style none \
      --results_root "$RESULTS_ROOT" --output_name "$NAME" \
      > "logs/gate2_xf_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  T1=$(date +%s)
  echo "[$(date '+%F %T')] $NAME scoring done in $((T1-T0))s"

  # hard shard check BEFORE merge: 8/8 shard json + 8/8 per-task jsonl per task
  NS=$(ls "$RESULTS_ROOT/$NAME"/shard*of${NGPU}.json 2>/dev/null | wc -l)
  if [ "$NS" != "$NGPU" ]; then
    echo "[$(date '+%F %T')] SHARD FAIL $NAME: ${NS}/${NGPU} shard json -- NOT merging"
    FAILED="$FAILED $NAME"; continue
  fi
  for T in $(echo "$TASKS" | tr ',' ' '); do
    NT=$(ls "$RESULTS_ROOT/$NAME"/per_example_${T}_shard*of${NGPU}.jsonl 2>/dev/null | wc -l)
    if [ "$NT" != "$NGPU" ]; then
      echo "[$(date '+%F %T')] SHARD FAIL $NAME/$T: ${NT}/${NGPU} -- NOT merging"
      FAILED="$FAILED ${NAME}/${T}"
    fi
  done
  case "$FAILED" in *"$NAME"*) continue ;; esac

  # merge_task() itself asserts shard completeness, n_scored == EXPECTED_N and
  # n_nan == 0, and RAISES rather than merging a partial set.
  $PY $SCRIPT --merge --output_name "$NAME" --tasks "$TASKS" \
      --num_shards $NGPU --n_boot "$N_BOOT" --results_root "$RESULTS_ROOT" \
      2>&1 | tee "logs/gate2_xf_${NAME}_merge.log"
  # NOTE: _log() prefixes every line with a timestamp, so the marker is
  # "[merge]" ANYWHERE on the line, not at the start. Anchoring this to ^ was a
  # bug in the first version of this driver: it printed a spurious
  # "MERGE FAIL ...: 0/6 tasks merged" for arms whose six summary_<task>.json had
  # in fact all been written. The authoritative check is the summary count below.
  NM=$(grep -c "\[merge\]" "logs/gate2_xf_${NAME}_merge.log" || true)
  NTASK=$(echo "$TASKS" | tr ',' ' ' | wc -w)
  NSUM=$(ls "$RESULTS_ROOT/$NAME"/summary_*.json 2>/dev/null | wc -l)
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
echo "[$(date '+%F %T')] ===== crossfamily gate-2 ALL ARMS DONE ====="
