#!/usr/bin/env bash
# A05 K1 canvas sweep -- 10 cells (5 canvas settings x 2 benchmarks) on 8x H20 (.73).
#
# Cells run one at a time, 8 shards per cell (shard = index % 8, identical to the
# archived r2 protocol), with a `wait` barrier between cells so that each cell can
# be graded and cost-checked before the next one starts.
#
# Order is cheapest-first so that canvas=8 -- the reproduction check against the
# archived HE+ .122 / MBPP+ .085 -- lands within the first two minutes.
set -u

ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/dllm_draft
PY=$ROOT/.venv_dream/bin/python
OUTROOT=$ROOT/runs/a05_k1
CKPT=$ROOT/models/DreamOn-v0-7B
NG=8

cd "$ROOT" || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT/a05_k1:$ROOT"
mkdir -p "$OUTROOT"

run_cell () {
  local DS="$1" DATAFILE="$2" CANVAS="$3" NAME="$4"
  local OUTDIR="$OUTROOT/$NAME"
  if [ -f "$OUTDIR/.cell_done" ]; then
    echo "[$(date '+%F %T')] $NAME: .cell_done exists -> SKIP"
    return
  fi
  mkdir -p "$OUTDIR"
  echo "[$(date '+%F %T')] ===== CELL $NAME (dataset=$DS canvas=$CANVAS) ====="
  local T0=$SECONDS
  for g in $(seq 0 $((NG-1))); do
    CUDA_VISIBLE_DEVICES=$g RANK=$g LOCAL_RANK=0 WORLD_SIZE=$NG \
      $PY -u "$ROOT/a05_k1/a05_k1_dreamon_canvas.py" \
        --checkpoint "$CKPT" \
        --dataset "$DS" \
        --data-file "$DATAFILE" \
        --output-dir "$OUTDIR" \
        --canvas "$CANVAS" \
        --max-new-tokens 512 \
        --transfer-tokens 1 \
        --temperature 0.2 \
        --top-p 0.9 \
        --alg entropy \
        --alg-temp 0.0 \
        > "$OUTDIR/shard${g}.log" 2>&1 &
  done
  wait
  local DT=$((SECONDS-T0))
  echo "[$(date '+%F %T')] $NAME: generation done in ${DT}s wall (~$(echo "scale=2;$DT*$NG/3600"|bc) GPU-h)"
  echo "$DT" > "$OUTDIR/.wall_seconds"
  touch "$OUTDIR/.cell_done"
}

HE=$ROOT/data/evalplus/HumanEvalPlus-v0.1.10.jsonl
MB=$ROOT/data/evalplus/MbppPlus-v0.2.0.jsonl

# --- cheapest first: canvas=8 is the archive reproduction check ---
run_cell humaneval "$HE" 8      he_c8
run_cell mbpp      "$MB" 8      mbpp_c8
run_cell humaneval "$HE" 32     he_c32
run_cell mbpp      "$MB" 32     mbpp_c32
run_cell mbpp      "$MB" oracle mbpp_oracle
run_cell humaneval "$HE" oracle he_oracle
run_cell humaneval "$HE" 128    he_c128
run_cell mbpp      "$MB" 128    mbpp_c128
run_cell humaneval "$HE" 512    he_c512
run_cell mbpp      "$MB" 512    mbpp_c512

echo "[$(date '+%F %T')] ===== ALL 10 CELLS GENERATED ====="
