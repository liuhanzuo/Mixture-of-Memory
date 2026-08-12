#!/usr/bin/env bash
# ============================================================================
# A02 READ-TAX eval — the step A02's own training log names as "next"
#   logs/a02_j0_depth_control_progress.log:
#     [08-12 17:38:06] ALL ARMS DONE
#     [08-12 17:38:06] next: offline eval of A0/A1/A2/A3/A4/A5/A6 on RULER
#                            (retrieval-closed) per PREREG 2.6
#
# WHAT THIS RUNS (PREREG A02_J0_DEPTH_CONTROL_PREREG.md §2.5 / §2.6)
# ------------------------------------------------------------------
#   arm  j   adapter                                   status
#   A0   0   none (= optimal j=0 adapter per GATE 0)    ON DISK, reused
#   A1   0   a02_j0control_lora_r32/final    87.29 M    NEW -> run here
#   A2   6   qcmem_distill_qwen_j6_r32_4k    72.74 M    adapter on disk, RULER never run
#   A3   9   qcmem_distill_qwen_j9_r32_4k    65.47 M    adapter on disk, RULER never run
#   A4  12   qcmem_distill_qwen_j12_r32_4k   58.20 M    ON DISK, reused (flagship)
#   A5  18   qcmem_distill_qwen_j18_r32_4k   43.65 M    adapter on disk, RULER never run
#   A6  12   a02_j12_capmatch_r40/final      72.74 M    NEW -> run here (cap-match to A2)
#
# So 5 arms are dispatched (A1 A2 A3 A5 A6); A0 and A4 are byte-on-disk from the
# dvr gate and are NOT re-run -- re-running them would change nothing but would
# risk overwriting the anchors the paired deltas are computed against.
#
# PRIMARY read-out = RULER, where dvr measured retrieval recall@12 = 99-100 %,
# i.e. retrieval is CLOSED and cannot confound the depth axis:
#     niah_multikey_1, variable_tracking x {16k, 32k}, n=100/cell, 8 shards.
# SECONDARY = BABILong qa1/qa2/qa5 x {16k,32k}, CONTRAST ONLY: dvr showed those
# cells are retrieval-dominated (recall@12 22.9-63.2 %) so they cannot support
# depth inference. They are run to show the curve is interpretable only where
# retrieval is closed. NO POOLED FIGURE is ever computed from them.
#
# PROTOCOL INVARIANTS (violating any voids the result)
#   * chat_template=False everywhere -- base LM, no SFT/RL. Default of both eval
#     scripts is store_true/False, so we pass nothing and assert it in the
#     emitted cell configs (GATE D, checked by the analyzer).
#   * selector=iter_bm25, topk=12, iter_hop_topk=4, sink_tokens=bos,
#     chunk_size=512 -- byte-identical to the A0/A4 anchors already on disk.
#     Verified against ruler_results/a02_dvr_ruler_j0_top12/*.json before use.
#   * 8 shards, limit 100 -- matches the anchors, so pairing is by construction.
#   * shard completeness is asserted before a cell is accepted (GATE C); a
#     partial cell aborts the arm rather than being merged.
#
# GPU POOL = all 8. The dvr driver bounded itself to GPUs 0-3 because a live
# A03 trajectory watcher needed the other 4. Re-checked at write time:
# `ps aux | grep -iE 'watcher|a03'` is EMPTY and all 8 cards are 0 MiB / 0 %,
# so the bound is obsolete here. Only .82 is touched.
#
# env: PROJECT_ROOT PYTHON_BIN NGPU_POOL
# Usage (on .82):
#   PROJECT_ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory \
#   PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
#   setsid nohup bash proposal/active/A02-comem-write-read-repair/code/run_a02_read_tax_eval.sh \
#     >logs/a02_read_tax_eval.out 2>&1 &
# ============================================================================
set -u
W="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$W" || { echo "FATAL: cannot cd to $W"; exit 3; }
PY="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
NSHARD=8                          # must match the A0/A4 anchors
POOL="${NGPU_POOL:-0 1 2 3 4 5 6 7}"
NPOOL=$(echo $POOL | wc -w)
BASE="${BASE_MODEL:-../models/Qwen--Qwen3-8b}"   # identical string to the anchors
PROG=logs/a02_read_tax_eval_progress.log

export HF_DATASETS_CACHE="$W/data/hf_datasets_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$W:$W/third_party/babilong-pkg:${PYTHONPATH:-}"
unset http_proxy https_proxy all_proxy
mkdir -p logs

note() { printf '[%s] %s\n' "$(date '+%m-%d %H:%M:%S')" "$*" | tee -a "$PROG"; }

# --- GATE A: flagship Read-LoRA identity (fail-closed, same as dvr/train) ----
FLAGSHIP=outputs/qcmem_distill_qwen_j12_r32_4k/final
EXPECT_SHA=dd09cd17457c63578c0f
GOT_SHA=$(sha256sum "$FLAGSHIP/adapter_model.safetensors" 2>/dev/null | cut -c1-20)
if [ "$GOT_SHA" != "$EXPECT_SHA" ]; then
  echo "FATAL GATE A: flagship sha mismatch: got '$GOT_SHA' want '$EXPECT_SHA'"; exit 7
fi
note "GATE A PASS flagship Read-LoRA sha $GOT_SHA"

# --- GATE B: every adapter we are about to evaluate has the span/rank it must -
$PY - <<'PYEOF' || { echo "FATAL GATE B"; exit 8; }
import json, sys
# arm -> (dir, resume_j, r, alpha, n_layers_expected)
want = {
 "A1": ("outputs/a02_j0control_lora_r32/final",        0, 32, 64, 36),
 "A2": ("outputs/qcmem_distill_qwen_j6_r32_4k/final",  6, 32, 64, 30),
 "A3": ("outputs/qcmem_distill_qwen_j9_r32_4k/final",  9, 32, 64, 27),
 "A5": ("outputs/qcmem_distill_qwen_j18_r32_4k/final",18, 32, 64, 18),
 "A6": ("outputs/a02_j12_capmatch_r40/final",         12, 40, 80, 24),
}
PER_R = 75776   # q8192+k5120+v5120+o8192+gate16384+up16384+down16384
bad = []
for arm,(d,j,r,a,nl) in want.items():
    c = json.load(open(f"{d}/adapter_config.json"))
    lt = sorted(c.get("layers_to_transform") or [])
    if c["r"] != r:            bad.append(f"{arm}: r={c['r']} != {r}")
    if c["lora_alpha"] != a:   bad.append(f"{arm}: alpha={c['lora_alpha']} != {a}")
    if lt != list(range(j,36)):bad.append(f"{arm}: span={lt[:2]}..{lt[-1:]} != [{j}..35]")
    if len(lt) != nl:          bad.append(f"{arm}: n_layers={len(lt)} != {nl}")
    print(f"GATE B {arm}: r={c['r']} alpha={c['lora_alpha']} span=[{lt[0]}..{lt[-1]}] "
          f"n={len(lt)} params={PER_R*c['r']*len(lt)/1e6:.2f}M")
# the capacity match the PREREG's Arm-2 argument rests on
p_a2 = PER_R*32*30; p_a6 = PER_R*40*24
print(f"GATE B capacity: A2={p_a2:,}  A6={p_a6:,}  match={p_a2==p_a6}")
if p_a2 != p_a6: bad.append("A2/A6 capacity mismatch")
if bad:
    print("GATE B FAIL:"); [print("  "+b) for b in bad]; sys.exit(1)
print("GATE B PASS")
PYEOF
note "GATE B PASS adapter spans/ranks/capacity verified"
note "pool='$POOL' ($NPOOL gpus) nshard=$NSHARD base=$BASE"

# Retrieval flags: byte-identical to the A0/A4 anchors on disk.
RETR="--selector iter_bm25 --topk 12 --iter_hop_topk 4 --sink_tokens bos"

# ---------------------------------------------------------------------------
run_babilong_arm() {
  local NAME="$1" EXTRA="$2"; shift 2
  local TASKS="$1" LENS="$2"
  note "babilong $NAME START tasks='$TASKS' lens='$LENS'"
  for t in $TASKS; do
    for l in $LENS; do
      local have; have=$(ls babilong_results/"$NAME"/${t}_${l}_*shard*of${NSHARD}.csv 2>/dev/null | wc -l)
      if [ "$have" -eq "$NSHARD" ]; then note "  SKIP $NAME $t $l (${have}/${NSHARD} shards present)"; continue; fi
      local slot=0
      for g in $POOL; do
        ( for s in $(seq 0 $((NSHARD-1))); do
            [ $((s % NPOOL)) -eq "$slot" ] || continue
            CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_qcmem_babilong.py \
              --model_path "$BASE" $EXTRA \
              --tasks "$t" --lengths "$l" \
              --limit 100 --chunk_size 512 \
              --num_shards $NSHARD --shard_index "$s" \
              --output_name "$NAME" \
              > "logs/a02_rtax_babilong_${NAME}_${t}_${l}_shard${s}.log" 2>&1
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

run_ruler_arm() {
  local NAME="$1" EXTRA="$2"; shift 2
  local TASKS="$1" LENS="$2"
  note "ruler $NAME START tasks='$TASKS' lens='$LENS'"
  local want=$(( $(echo $TASKS | wc -w) * $(echo $LENS | wc -w) * NSHARD ))
  local have; have=$(ls ruler_results/"$NAME"/*_shard*of${NSHARD}.records.json 2>/dev/null | wc -l)
  if [ "$have" -eq "$want" ]; then note "  SKIP ruler $NAME ($have/$want records present)"; return 0; fi
  local slot=0
  for g in $POOL; do
    ( for s in $(seq 0 $((NSHARD-1))); do
        [ $((s % NPOOL)) -eq "$slot" ] || continue
        CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_ruler_qcmem.py \
          --model_path "$BASE" $EXTRA \
          --ruler_tasks $TASKS --lengths $LENS \
          --limit 100 --chunk_size 512 \
          --num_shards $NSHARD --shard_index "$s" \
          --output_name "$NAME" \
          > "logs/a02_rtax_ruler_${NAME}_shard${s}.log" 2>&1
      done ) &
    slot=$((slot+1))
  done
  wait
  have=$(ls ruler_results/"$NAME"/*_shard*of${NSHARD}.records.json 2>/dev/null | wc -l)
  if [ "$have" -ne "$want" ]; then note "  ABORT ruler $NAME: only $have/$want records" >&2; return 9; fi
  note "ruler $NAME DONE ($have/$want records)"
}

RUL_TASKS="niah_multikey_1 variable_tracking"
RUL_LENS="16k 32k"
BAB_TASKS="qa1 qa2 qa5"
BAB_LENS="16k 32k"

# ---- PRIMARY: RULER for all 5 new arms, first (it is the read-out that counts)
run_ruler_arm a02_rtax_ruler_A1_j0control \
  "--resume_j 0  --lora_adapter outputs/a02_j0control_lora_r32/final       $RETR" "$RUL_TASKS" "$RUL_LENS"
run_ruler_arm a02_rtax_ruler_A2_j6 \
  "--resume_j 6  --lora_adapter outputs/qcmem_distill_qwen_j6_r32_4k/final $RETR" "$RUL_TASKS" "$RUL_LENS"
run_ruler_arm a02_rtax_ruler_A3_j9 \
  "--resume_j 9  --lora_adapter outputs/qcmem_distill_qwen_j9_r32_4k/final $RETR" "$RUL_TASKS" "$RUL_LENS"
run_ruler_arm a02_rtax_ruler_A5_j18 \
  "--resume_j 18 --lora_adapter outputs/qcmem_distill_qwen_j18_r32_4k/final $RETR" "$RUL_TASKS" "$RUL_LENS"
run_ruler_arm a02_rtax_ruler_A6_j12_r40 \
  "--resume_j 12 --lora_adapter outputs/a02_j12_capmatch_r40/final          $RETR" "$RUL_TASKS" "$RUL_LENS"
note "==== RULER (PRIMARY) COMPLETE for A1 A2 A3 A5 A6 ===="

# ---- SECONDARY: BABILong, contrast only, never pooled
run_babilong_arm a02_rtax_babilong_A1_j0control \
  "--resume_j 0  --lora_adapter outputs/a02_j0control_lora_r32/final       $RETR" "$BAB_TASKS" "$BAB_LENS"
run_babilong_arm a02_rtax_babilong_A2_j6 \
  "--resume_j 6  --lora_adapter outputs/qcmem_distill_qwen_j6_r32_4k/final $RETR" "$BAB_TASKS" "$BAB_LENS"
run_babilong_arm a02_rtax_babilong_A3_j9 \
  "--resume_j 9  --lora_adapter outputs/qcmem_distill_qwen_j9_r32_4k/final $RETR" "$BAB_TASKS" "$BAB_LENS"
run_babilong_arm a02_rtax_babilong_A5_j18 \
  "--resume_j 18 --lora_adapter outputs/qcmem_distill_qwen_j18_r32_4k/final $RETR" "$BAB_TASKS" "$BAB_LENS"
run_babilong_arm a02_rtax_babilong_A6_j12_r40 \
  "--resume_j 12 --lora_adapter outputs/a02_j12_capmatch_r40/final          $RETR" "$BAB_TASKS" "$BAB_LENS"

note "ALL ARMS DONE -- next: analyze_a02_read_tax.py"
