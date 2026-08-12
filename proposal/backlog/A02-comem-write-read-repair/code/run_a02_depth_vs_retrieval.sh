#!/usr/bin/env bash
# ============================================================================
# A02 DEPTH-vs-RETRIEVAL quality gate  (2026-08-10)
#
# THE QUESTION
# ------------
# Phase-1 compared C1 vs C2 while moving FOUR variables at once (verified from
# the on-disk eval configs, not from prose):
#   C1 = babilong_results/a02_babilong_c1_kvdirect
#        no_retrieval=True, selector=None, topk=None, resume_j=0, lora=None
#   C2 = babilong_results/a02_babilong_c2_j12_readlora
#        no_retrieval=False, selector=iter_bm25, topk=12, resume_j=12,
#        lora=outputs/qcmem_distill_qwen_j12_r32_4k/final
# So {read depth} x {LoRA} x {retrieval-vs-pack-all} x {selector} all moved.
# Nobody knows whether C2's quality losses are MID-LAYER-READ failures (A02's
# actual thesis) or merely RETRIEVAL-RECALL failures (a property of top-12
# iter_bm25 that has nothing to do with CoMem's memory).
#
# THE ARMS (each adjacent pair differs in exactly ONE thing)
# ----------------------------------------------------------
#   arm            depth  LoRA  pack              selector
#   c1_pack_all*   j=0    no    ALL chunks        (none)      <- phase-1 C1, on disk
#   j0_top12       j=0    no    top-12            iter_bm25   <- NEW
#   j12_frozen     j=12   no    top-12            iter_bm25   <- NEW
#   c2_comem*      j=12   yes   top-12            iter_bm25   <- phase-1 C2, on disk
#   (*) already on disk from phase 1; only the two middle arms are run here.
#
#   c1_pack_all -> j0_top12  : isolates RETRIEVAL (depth held at j=0, no LoRA both)
#   j0_top12    -> j12_frozen: isolates READ DEPTH (retrieval identical, no LoRA both)
#   j12_frozen  -> c2_comem  : isolates the LoRA   (depth+retrieval identical)
#
# CONFOUND NOTE (this is the improvement over the cost gate):
# The cost gate declared depth<->LoRA "irreducible" because it only had
# j0(no-LoRA) and j12(+LoRA). Adding `j12_frozen` (j=12, retrieval identical,
# LoRA DROPPED) makes the chain fully single-variable. j12_frozen is a
# functional arm -- this repo has shipped `*_j12_frozen_iterbm25_chatFALSE`
# runs before (ruler_results/qcmem_8b_zeroshot_j12_frozen_iterbm25_chatFALSE).
# We therefore run BOTH j=0 and j=12 without the LoRA and keep the LoRA as its
# own separately-attributed step, rather than leaving depth and LoRA fused.
#
# BENCHMARK CELLS (the diagnostic ones -- where phase-1's C1 wins were largest
# and protocol-clean; TIE cells are uninformative for this question):
#   BABILong qa1/qa2 x {16k,32k}  (phase-1: -35pp..-55pp, all four CIs < 0)
#   BABILong qa5     x {16k}      (phase-1: C2 WON +14pp -- sign-flip control)
#   RULER niah_multikey_1 + variable_tracking x {16k,32k} (phase-1: -5.17pp)
#
# PROTOCOL INVARIANTS
#   * chat_template=False everywhere (models have no SFT/RL).
#   * selector=iter_bm25, topk=12, iter_hop_topk=4, chunk_size=512 -- IDENTICAL
#     across every retrieving arm. That identity is the whole point.
#   * 8-shard sharding matching phase 1 exactly, so pairing is by construction:
#     BABILong CSV row r of shard s == dataset index s + r*8; RULER pairs by
#     sample_index and is additionally sha256-verified.
#   * GPU POOL BOUNDED TO 4 (GPUs 0-3). .82 hosts a live A03 eval watcher
#     (/tmp/a03_arm4_trajectory_watcher.sh) that needs all 8 GPUs for ~4 min
#     when an A03 ckpt lands. Leaving 4 GPUs free is what let the previous A02
#     gate coexist with it. DO NOT RAISE THIS.
#
# env: PROJECT_ROOT PYTHON_BIN NGPU_POOL
# Usage (on .82):
#   PROJECT_ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory \
#   PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
#   setsid nohup bash proposal/active/A02-comem-write-read-repair/code/run_a02_depth_vs_retrieval.sh \
#     >logs/a02_depth_vs_retrieval.out 2>&1 &
# ============================================================================
set -u
W="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$W" || { echo "FATAL: cannot cd to $W"; exit 3; }
PY="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
NSHARD=8                       # must match phase 1 for by-construction pairing
POOL="${NGPU_POOL:-0 1 2 3}"   # BOUNDED 4-GPU pool; leave 4 free for A03 watcher
BASE="${BASE_MODEL:-../models/Qwen--Qwen3-8b}"
READ_LORA="${READ_LORA:-outputs/qcmem_distill_qwen_j12_r32_4k/final}"
PROG=logs/a02_depth_vs_retrieval_progress.log

export HF_DATASETS_CACHE="$W/data/hf_datasets_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset http_proxy https_proxy all_proxy
mkdir -p logs

note() { printf '[%s] %s\n' "$(date '+%m-%d %H:%M:%S')" "$*" | tee -a "$PROG"; }

# --- GATE 0: Read-LoRA identity (fail-closed) -------------------------------
EXPECT_SHA=dd09cd17457c63578c0f
GOT_SHA=$(sha256sum "$READ_LORA/adapter_model.safetensors" 2>/dev/null | cut -c1-20)
if [ "$GOT_SHA" != "$EXPECT_SHA" ]; then
  echo "FATAL GATE0: Read-LoRA sha mismatch: got '$GOT_SHA' want '$EXPECT_SHA'"; exit 7
fi
note "GATE0 PASS Read-LoRA sha $GOT_SHA == flagship"
note "pool='$POOL' nshard=$NSHARD base=$BASE"

# Shared flags: IDENTICAL retrieval for every retrieving arm.
RETR="--selector iter_bm25 --topk 12 --iter_hop_topk 4 --sink_tokens bos"

# ---------------------------------------------------------------------------
# BABILong: run the 2 new arms over the diagnostic cells.
# A "unit" = (arm, task, length, shard). Units are dispatched onto the bounded
# pool; each pool slot runs one unit at a time.
# ---------------------------------------------------------------------------
run_babilong_arm() {
  local NAME="$1" EXTRA="$2"; shift 2
  local TASKS="$1" LENS="$2"
  note "babilong $NAME START tasks='$TASKS' lens='$LENS'"
  for t in $TASKS; do
    for l in $LENS; do
      # idempotency: skip cell if all NSHARD csv shards already exist
      local have; have=$(ls babilong_results/"$NAME"/${t}_${l}_*shard*of${NSHARD}.csv 2>/dev/null | wc -l)
      if [ "$have" -eq "$NSHARD" ]; then note "  SKIP $NAME $t $l (${have}/${NSHARD} shards present)"; continue; fi
      local slot=0
      for g in $POOL; do
        # each pool GPU takes the shards congruent to its slot index
        ( for s in $(seq 0 $((NSHARD-1))); do
            [ $((s % $(echo $POOL | wc -w))) -eq "$slot" ] || continue
            CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_qcmem_babilong.py \
              --model_path "$BASE" $EXTRA \
              --tasks "$t" --lengths "$l" \
              --limit 100 --chunk_size 512 \
              --num_shards $NSHARD --shard_index "$s" \
              --output_name "$NAME" \
              > "logs/a02_dvr_babilong_${NAME}_${t}_${l}_shard${s}.log" 2>&1
          done ) &
        slot=$((slot+1))
      done
      wait
      local ns; ns=$(ls babilong_results/"$NAME"/${t}_${l}_*shard*of${NSHARD}.csv 2>/dev/null | wc -l)
      if [ "$ns" -ne "$NSHARD" ]; then note "  ABORT $NAME $t $l: only $ns/$NSHARD shards" >&2; return 9; fi
      note "  OK $NAME $t $l ($ns/$NSHARD shards)"
    done
  done
  note "babilong $NAME DONE"
}

# ---------------------------------------------------------------------------
# RULER: same bounded-pool dispatch.
# ---------------------------------------------------------------------------
run_ruler_arm() {
  local NAME="$1" EXTRA="$2"; shift 2
  local TASKS="$1" LENS="$2"
  note "ruler $NAME START tasks='$TASKS' lens='$LENS'"
  local have; have=$(ls ruler_results/"$NAME"/*_shard*of${NSHARD}.records.json 2>/dev/null | wc -l)
  local want=$(( $(echo $TASKS | wc -w) * $(echo $LENS | wc -w) * NSHARD ))
  if [ "$have" -eq "$want" ]; then note "  SKIP ruler $NAME ($have/$want records present)"; return 0; fi
  local slot=0
  for g in $POOL; do
    ( for s in $(seq 0 $((NSHARD-1))); do
        [ $((s % $(echo $POOL | wc -w))) -eq "$slot" ] || continue
        CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_ruler_qcmem.py \
          --model_path "$BASE" $EXTRA \
          --ruler_tasks $TASKS --lengths $LENS \
          --limit 100 --chunk_size 512 \
          --num_shards $NSHARD --shard_index "$s" \
          --output_name "$NAME" \
          > "logs/a02_dvr_ruler_${NAME}_shard${s}.log" 2>&1
      done ) &
    slot=$((slot+1))
  done
  wait
  have=$(ls ruler_results/"$NAME"/*_shard*of${NSHARD}.records.json 2>/dev/null | wc -l)
  if [ "$have" -ne "$want" ]; then note "  ABORT ruler $NAME: only $have/$want records" >&2; return 9; fi
  note "ruler $NAME DONE ($have/$want records)"
}

BAB_TASKS="qa1 qa2 qa5"
BAB_LENS="16k 32k"
RUL_TASKS="niah_multikey_1 variable_tracking"
RUL_LENS="16k 32k"

# ARM j0_top12: j=0, NO LoRA, top-12 iter_bm25.  (retrieval isolated vs C1)
# NOTE: --baseline none keeps retrieval; resume_j=0 makes the write embeddings-only
# and the read full-depth == genuine text-RAG over the retrieved pack.
run_babilong_arm a02_dvr_babilong_j0_top12 \
  "--resume_j 0 $RETR" "$BAB_TASKS" "$BAB_LENS"
run_ruler_arm    a02_dvr_ruler_j0_top12 \
  "--resume_j 0 $RETR" "$RUL_TASKS" "$RUL_LENS"

# ARM j12_frozen: j=12, NO LoRA, SAME top-12 pack.  (depth isolated vs j0_top12)
run_babilong_arm a02_dvr_babilong_j12_frozen \
  "--resume_j 12 $RETR" "$BAB_TASKS" "$BAB_LENS"
run_ruler_arm    a02_dvr_ruler_j12_frozen \
  "--resume_j 12 $RETR" "$RUL_TASKS" "$RUL_LENS"

note "ALL ARMS DONE -- next: analyze_a02_depth_vs_retrieval.py"
