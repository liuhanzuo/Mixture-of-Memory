#!/usr/bin/env bash
# B10 Gate 1 -- base-axis re-score of all 6 existing arms. ZERO GPU.
#
# Pre-registered in proposal/backlog/B10-dllm-infilling-ar-dominance/PROPOSAL.md S5:
#   "Re-score all six existing arms with score_infilling.py --which base.
#    Solutions are already on disk; nothing is regenerated."
#
# Nothing here touches a GPU: CUDA_VISIBLE_DEVICES is forced empty, and the only
# work is evalplus's untrusted_check sandbox (pure-CPU subprocesses).
# Output goes to score_base.json so the pre-existing plus-axis score.json --
# the provenance for the numbers already in STATUS.json -- is NOT overwritten.
set -uo pipefail

ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/dllm_draft_104
PY=$ROOT/.venv_dream/bin/python
SPLIT_FILE=$ROOT/data/humaneval_infilling/HumanEval-SingleLineInfilling.jsonl
OUT=$ROOT/outputs/infilling_single_line

export CUDA_VISIBLE_DEVICES=""          # hard GPU lockout
export PYTHONPATH=$ROOT/scripts:$ROOT
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HUMANEVAL_OVERRIDE_PATH=$ROOT/data/evalplus/HumanEvalPlus-v0.1.10.jsonl

ARMS="dream_fim dreamon_fim dreamon_oracle dream_prefix qwen_fim qwen_prefix"
JOBS=${JOBS:-48}

for ARM in $ARMS; do
  DIR=$OUT/$ARM
  if [ -f "$DIR/score_base.json" ]; then
    echo "[$(date -Is)] SKIP $ARM (score_base.json exists)"
    continue
  fi
  echo "[$(date -Is)] BASE-SCORE $ARM"
  $PY $ROOT/scripts/score_infilling.py \
    --solutions "$DIR/solutions.jsonl" --metrics "$DIR/metrics.jsonl" \
    --data-file "$SPLIT_FILE" --output "$DIR/score_base.json" \
    --arm "$ARM" --which base --self-test-n 10 --jobs "$JOBS" \
    2>&1 | tee "$ROOT/logs/gate1_base_${ARM}_score.log"
  echo "[$(date -Is)] DONE $ARM rc=${PIPESTATUS[0]}"
done
echo "[$(date -Is)] GATE1 BASE RE-SCORE COMPLETE"
