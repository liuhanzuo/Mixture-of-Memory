#!/usr/bin/env bash
# ============================================================================
# Round-2 baselines on 8× L20A. Fixes applied to round-1:
#   - Dream generate: +--top-p / +--no-chat flags, top_p defaults to 1.0
#   - DreamOn generate: T=0.2 top_p=0.9
#     ⚠️ CORRECTED 2026-08-12 (A05 closeout): this line used to also claim
#     "mask_expansion=True + delete_eos_token=True". THAT FIX NEVER TOOK EFFECT.
#     Both kwargs are silently swallowed by **kwargs -- verified BY EXECUTION:
#       DreamGenerationConfig().update(mask_expansion=True, delete_eos_token=True)
#       returns both keys as UNUSED, and hasattr(cfg,'mask_expansion') is False
#       before and after. (Control: update(temperature=0.2) returns {}.)
#     So r2 differs from r1 ONLY in temperature/top_p. Any statement that the r2
#     DreamOn numbers were obtained "with mask_expansion on" is VOID. The kwargs
#     have been removed from generate_evalplus_dreamon.py.
#
#   ⚠️ Two further corrections that change how these runs' outputs must be read:
#   - The per-item "nfe" these runs logged is len(output.history), NOT a
#     forward-pass count, and it is null throughout r2 (output_history=False).
#     True counted NFE at this setting: 172.3 (HE+) / 153.4 (MBPP+).
#   - HE+ used a stitch that double-indented DreamOn's already-indented body, so
#     the r2 HE+ .122 is UNDERSTATED (~+0.6 pp at this canvas; the error grows
#     sharply with canvas size). Fixed in the driver.
#
#   ⚠️ --initial-masks 8 below is the most load-bearing knob in these runs:
#   raising it to 32 alone moves MBPP+ from .085 to .3545. These runs
#   characterise "DreamOn at canvas=8", NOT DreamOn.
#
# Runs (previous runs are backed up to .r1/):
#   1. dream_coder_instruct_heplus_r2   (T=0.1 top_p=0.95, paper Instruct recipe)
#   2. dream_coder_instruct_mbppplus_r2 (T=0.1 top_p=0.95)
#   3. dream_coder_base_heplus          (T=0.2 top_p=0.9,  no-chat, paper Base recipe)
#   4. dream_coder_base_mbppplus        (T=0.1 top_p=0.9,  no-chat)
#   5. dreamon_heplus_r2                (T=0.2 top_p=0.9, canvas=8; NOT mask_expansion)
#   6. dreamon_mbppplus_r2              (same)
# ============================================================================
set -u
ROOT=/apdcephfs_wzc1/share_304376610/pighzliu_code/dllm_draft
PY=/opt/conda/envs/dllm-env/bin/python
cd "$ROOT" || exit 1

run_one () {
  local NAME="$1" SCRIPT="$2" CKPT="$3" DATASET="$4" DATAFILE="$5"; shift 5
  local OUTDIR="runs/$NAME"
  if [ -s "$OUTDIR/solutions.jsonl" ]; then
    echo "[$(date '+%F %T')] $NAME: solutions.jsonl exists -> SKIP"
    return
  fi
  mkdir -p "$OUTDIR"
  echo "[$(date '+%F %T')] ===== $NAME ====="
  local NG=8
  for g in $(seq 0 $((NG-1))); do
    CUDA_VISIBLE_DEVICES=$g RANK=$g LOCAL_RANK=0 WORLD_SIZE=$NG \
      $PY -u "$SCRIPT" \
        --checkpoint "$CKPT" \
        --dataset "$DATASET" \
        --data-file "$DATAFILE" \
        --output-dir "$OUTDIR" \
        "$@" \
        > "$OUTDIR/shard${g}.log" 2>&1 &
  done
  wait
  $PY scripts/_merge_rank_solutions.py "$OUTDIR" 2>&1 | tail -3
  # Grade with evalplus in-line
  local DSGR="humaneval"; [[ "$DATASET" == "mbpp" ]] && DSGR="mbpp"
  $PY -m evalplus.evaluate --dataset "$DSGR" \
      --samples "$OUTDIR/solutions.jsonl" > "$OUTDIR/evalplus.out" 2>&1
  echo "--- $NAME pass@1 ---"
  grep -E "pass@1|humaneval|mbpp" "$OUTDIR/evalplus.out" | grep -v "Traceback\|File \"" | tail -6
}

# 1&2 — Dream-Coder-Instruct paper recipe (top_p=0.95)
run_one dream_coder_instruct_heplus_r2 \
  scripts/generate_evalplus_dream.py \
  models/Dream-Coder-v0-Instruct-7B humaneval \
  data/evalplus/humaneval_plus.jsonl \
  --steps 512 --max-new-tokens 512 --temperature 0.1 --top-p 0.95

run_one dream_coder_instruct_mbppplus_r2 \
  scripts/generate_evalplus_dream.py \
  models/Dream-Coder-v0-Instruct-7B mbpp \
  data/evalplus/mbpp_plus.jsonl \
  --steps 512 --max-new-tokens 512 --temperature 0.1 --top-p 0.95

# 3&4 — Dream-Coder-Base paper recipe (no chat, T=0.2/0.1 top_p=0.9)
run_one dream_coder_base_heplus \
  scripts/generate_evalplus_dream.py \
  models/Dream-Coder-v0-Base-7B humaneval \
  data/evalplus/humaneval_plus.jsonl \
  --steps 512 --max-new-tokens 512 --temperature 0.2 --top-p 0.9 --no-chat

run_one dream_coder_base_mbppplus \
  scripts/generate_evalplus_dream.py \
  models/Dream-Coder-v0-Base-7B mbpp \
  data/evalplus/mbpp_plus.jsonl \
  --steps 512 --max-new-tokens 512 --temperature 0.1 --top-p 0.9 --no-chat

# 5&6 — DreamOn, T=0.2 top_p=0.9, canvas=8.
# NOT "the paper DreamOn recipe with mask_expansion": those kwargs never took
# effect (see header). canvas=8 is a choice, not a neutral default -- see A05 K1.
run_one dreamon_heplus_r2 \
  scripts/generate_evalplus_dreamon.py \
  models/DreamOn-v0-7B humaneval \
  data/evalplus/humaneval_plus.jsonl \
  --initial-masks 8 --max-new-tokens 512 --transfer-tokens 1

run_one dreamon_mbppplus_r2 \
  scripts/generate_evalplus_dreamon.py \
  models/DreamOn-v0-7B mbpp \
  data/evalplus/mbpp_plus.jsonl \
  --initial-masks 8 --max-new-tokens 512 --transfer-tokens 1

echo "[$(date '+%F %T')] ===== round-2 ALL 6 RUNS DONE ====="
