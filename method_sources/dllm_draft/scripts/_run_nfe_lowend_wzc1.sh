#!/usr/bin/env bash
# Fill the low-NFE gap on the token-cost Pareto plot (both benchmarks).
# Currently: vanilla nfe64 = 43854/38624 tok. scaffold medium at 13774/7080 tok.
# Adds nfe16 (~10.9k/9.7k tok) and nfe32 (~21.9k/19.3k tok) so we can see
# whether vanilla can enter the low-cost segment or scaffold's exclusivity
# extends further down.
set -u
ROOT=/apdcephfs_wzc1/share_304376610/pighzliu_code/dllm_draft
PY=/opt/conda/envs/dllm-env/bin/python
cd "$ROOT" || exit 1

run_one () {
  local DS="$1" DF="$2" STEPS="$3"
  local NAME="dream_instruct_${DS/humaneval/heplus}${DS/mbpp/mbppplus}_nfe${STEPS}"
  # simpler naming
  local NAME="dream_instruct_${DS}plus_nfe${STEPS}"
  [[ "$DS" == "humaneval" ]] && NAME="dream_instruct_heplus_nfe${STEPS}"
  [[ "$DS" == "mbpp" ]] && NAME="dream_instruct_mbppplus_nfe${STEPS}"
  local OUT="runs/$NAME"
  if [ -s "$OUT/solutions.jsonl" ]; then echo "$NAME exists SKIP"; return; fi
  mkdir -p "$OUT"
  echo "[$(date '+%F %T')] ===== $NAME ====="
  for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g RANK=$g LOCAL_RANK=0 WORLD_SIZE=8 \
      $PY -u scripts/generate_evalplus_dream.py \
        --checkpoint models/Dream-Coder-v0-Instruct-7B \
        --dataset "$DS" \
        --data-file "$DF" \
        --output-dir "$OUT" \
        --steps $STEPS --max-new-tokens 512 --temperature 0.1 --top-p 0.95 \
        > "$OUT/shard${g}.log" 2>&1 &
  done
  wait
  $PY scripts/_merge_rank_solutions.py "$OUT" 2>&1 | tail -2
  $PY -m evalplus.evaluate --dataset "$DS" --samples "$OUT/solutions.jsonl" > "$OUT/evalplus.out" 2>&1
  grep -E "pass@1|humaneval|mbpp" "$OUT/evalplus.out" | grep -v "Traceback\|File \"" | tail -6
}

for STEPS in 16 32; do
  run_one humaneval data/evalplus/humaneval_plus.jsonl $STEPS
  run_one mbpp data/evalplus/mbpp_plus.jsonl $STEPS
done

echo "[$(date '+%F %T')] ===== low-NFE gap fill DONE ====="
