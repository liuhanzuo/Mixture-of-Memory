#!/usr/bin/env bash
# ============================================================================
# A02 Job 2.2 — DE-SATURATION cell: niah_single_3 (36-char UUID values) x {16k,32k}
#
# WHY THIS CELL. The read-tax primary cells put A0-A3 at 95-100 %, so "read tax ~ 0
# at shallow j" is partly a statement about a SATURATED benchmark (verdict caveat 2).
# A de-saturating cell must be HARDER while STAYING retrieval-closed.
#
# The CPU screen (probe_a02_desaturation_candidates.py, 0 GPU) measured recall@12
# under the identical selector:
#     niah_single_3   16k 97.5 %   32k 95.0 %   <- PASSES, and it is SINGLE-needle so
#                                                  the screen's gold locator is EXACT
#     niah_multivalue 32k 92.3 %                <- FAILS the >=95 % criterion
#     niah_multiquery 16k 100 % 32k 95.0 %      <- passes, BUT _make_niah returns only
#                                                  the FIRST of 4 queried needles as
#                                                  gold, so that recall is an UPPER
#                                                  BOUND -> not trustworthy as closed
# Hence niah_single_3 is the only candidate whose retrieval-closure is exactly screened.
# Length was deliberately NOT used as the difficulty knob: dvr measured recall
# DEGRADING with length (qa2 49.5 -> 22.9 %), so 64k/128k de-saturates by breaking
# retrieval, which is the confound the primary read-out exists to exclude.
#
# ARMS. The shallow end is what needs de-saturating, so this runs the arms that were
# saturated: A0 (no adapter), A2 (j=6), A3 (j=9), plus A4 (j=12) as the known-taxed
# reference point. A1/A6 are controls whose questions are already answered (A1 = null
# adapter, 0/400 flips; A6 = capacity, all n.s.) and A5 (j=18) is already destroyed at
# 4-42 %, so neither adds information about SHALLOW-j saturation.
#
# PROTOCOL INVARIANTS (violating any voids the result)
#   * chat_template=False -- base LM, no SFT/RL. Both eval scripts default to False;
#     nothing is passed and the analyzer asserts it (`is not False`, never `is not True`,
#     which would silently pass on a None).
#   * selector=iter_bm25 topk=12 iter_hop_topk=4 sink_tokens=bos chunk_size=512 --
#     byte-identical to the primary read-out, so the new cells are comparable.
#   * 8 shards, limit 100; shard completeness asserted before any cell is accepted.
#   * NO pooling with BABILong/LongEval, ever (standing A02 invariant).
#
# NODE. .82 ONLY. LOCAL/.21 (SparseForge #246), .73 and .104 (paperC) are never touched.
#
# Usage (on .82):
#   PROJECT_ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory \
#   PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
#   setsid nohup bash proposal/backlog/A02-comem-write-read-repair/code/run_a02_desaturation_eval.sh \
#     >logs/a02_desat_eval.out 2>&1 &
# ============================================================================
set -u
W="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$W" || { echo "FATAL: cannot cd to $W"; exit 3; }
PY="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
NSHARD=8
POOL="${NGPU_POOL:-0 1 2 3 4 5 6 7}"
NPOOL=$(echo $POOL | wc -w)
BASE="${BASE_MODEL:-../models/Qwen--Qwen3-8b}"
PROG=logs/a02_desat_eval_progress.log

export HF_DATASETS_CACHE="$W/data/hf_datasets_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$W:$W/third_party/babilong-pkg:${PYTHONPATH:-}"
unset http_proxy https_proxy all_proxy
mkdir -p logs

note() { printf '[%s] %s\n' "$(date '+%m-%d %H:%M:%S')" "$*" | tee -a "$PROG"; }

# --- GATE A: flagship Read-LoRA identity (same fail-closed check as every A02 gate)
FLAGSHIP=outputs/qcmem_distill_qwen_j12_r32_4k/final
EXPECT_SHA=dd09cd17457c63578c0f
GOT_SHA=$(sha256sum "$FLAGSHIP/adapter_model.safetensors" 2>/dev/null | cut -c1-20)
if [ "$GOT_SHA" != "$EXPECT_SHA" ]; then
  echo "FATAL GATE A: flagship sha mismatch: got '$GOT_SHA' want '$EXPECT_SHA'"; exit 7
fi
note "GATE A PASS flagship sha $GOT_SHA"

RETR="--selector iter_bm25 --topk 12 --iter_hop_topk 4 --sink_tokens bos"
RUL_TASKS="niah_single_3"
RUL_LENS="16k 32k"

run_ruler_arm() {
  local NAME="$1" EXTRA="$2"
  note "ruler $NAME START tasks='$RUL_TASKS' lens='$RUL_LENS'"
  local want=$(( $(echo $RUL_TASKS | wc -w) * $(echo $RUL_LENS | wc -w) * NSHARD ))
  local have; have=$(ls ruler_results/"$NAME"/*_shard*of${NSHARD}.records.json 2>/dev/null | wc -l)
  if [ "$have" -eq "$want" ]; then note "  SKIP $NAME ($have/$want records present)"; return 0; fi
  local slot=0
  for g in $POOL; do
    ( for s in $(seq 0 $((NSHARD-1))); do
        [ $((s % NPOOL)) -eq "$slot" ] || continue
        CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_ruler_qcmem.py \
          --model_path "$BASE" $EXTRA \
          --ruler_tasks $RUL_TASKS --lengths $RUL_LENS \
          --limit 100 --chunk_size 512 \
          --num_shards $NSHARD --shard_index "$s" \
          --output_name "$NAME" \
          > "logs/a02_desat_ruler_${NAME}_shard${s}.log" 2>&1
      done ) &
    slot=$((slot+1))
  done
  wait
  have=$(ls ruler_results/"$NAME"/*_shard*of${NSHARD}.records.json 2>/dev/null | wc -l)
  if [ "$have" -ne "$want" ]; then note "  ABORT $NAME: only $have/$want records" >&2; return 9; fi
  note "ruler $NAME DONE ($have/$want records)"
}

# A0 = no adapter (the anchor; GATE 0 established it IS the optimal j=0 adapter)
run_ruler_arm a02_desat_ruler_A0_j0    "--resume_j 0 $RETR"
run_ruler_arm a02_desat_ruler_A2_j6 \
  "--resume_j 6  --lora_adapter outputs/qcmem_distill_qwen_j6_r32_4k/final  $RETR"
run_ruler_arm a02_desat_ruler_A3_j9 \
  "--resume_j 9  --lora_adapter outputs/qcmem_distill_qwen_j9_r32_4k/final  $RETR"
run_ruler_arm a02_desat_ruler_A4_j12 \
  "--resume_j 12 --lora_adapter outputs/qcmem_distill_qwen_j12_r32_4k/final $RETR"

note "ALL DE-SATURATION ARMS DONE"
