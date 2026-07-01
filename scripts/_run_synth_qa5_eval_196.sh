#!/bin/bash
# Synthetic-qa5 readout eval on the A-model (step2000, NO training). Calibrates
# L7 (new babilong-shaped) and L6 (current clean) synthetic difficulty against a
# REAL-babilong CONTROL, all through the IDENTICAL harness path (same generate_
# with_mem_space, same compare_answers, same readerattn setting) so the numbers
# are directly comparable. Real babilong is EVAL-ONLY here (analysis/calibration),
# never trained on — red line intact.
#
# 3 arms x per length. 8 GPUs: at each length run L7(gpu0-2), L6(gpu3-5),
# REAL(gpu6-7) — 3-shard / 3-shard / 2-shard. Short->long (4k first: decisive).
set -uo pipefail
R="/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory"
cd "$R"
export HF_HOME="$R/.hf_cache" HF_DATASETS_CACHE="$R/.hf_cache/datasets"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export PYTHONPATH="$R/third_party/babilong-pkg:$R:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="$R/.venv/bin/python"
CKPT="outputs/mem_space_fifo_b25_c512_supervised_select/full_model_step002000.pt"
ACFG="outputs/mem_space_fifo_b25_c512_supervised_select/adapter_config.json"
BG="data/pg19_chunks_llama3_noeos.npy"
# reader-attn topk: pass as $1 (0 = pure mem-chain; 2 = deployable selector like a2000_loc)
RATOPK="${1:-0}"
LIMIT=48
TAG="ra${RATOPK}"
mkdir -p logs/synth_eval babilong_results/synth_eval

# Guard: refuse to start if eval WORKERS already hold a GPU (avoid orphan
# contention that stalled the previous run at 90GB/100%). Count python workers
# via ps (excluding this shell / any pgrep self-match).
if [ "$(ps -eo args | grep -c '[p]ython scripts/eval_synth_qa5_readout.py')" -gt 0 ]; then
  echo "REFUSE: eval_synth_qa5_readout workers already running"; exit 3
fi

run_shard () {  # $1=arm(L7|L6|REAL) $2=len $3=gpu $4=shard $5=nshards
  local arm=$1 len=$2 gpu=$3 s=$4 ns=$5
  local extra=""
  case "$arm" in
    L7) extra="--level 7";;
    L6) extra="--level 6";;
    REAL) extra="--real_babilong";;
  esac
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/eval_synth_qa5_readout.py \
    --checkpoint "$CKPT" --adapter_config "$ACFG" --background "$BG" \
    $extra --target_len $len --limit $LIMIT --chunk_size 512 \
    --readerattn_topk $RATOPK --readerattn_select_layer 16 \
    --sample_timeout 300 \
    --shard_index $s --num_shards $ns \
    --out_csv babilong_results/synth_eval/${arm}_${len}_${TAG}_sh${s}.csv \
    >logs/synth_eval/${arm}_${len}_${TAG}_sh${s}.log 2>&1 &
}

for len in 4k 8k; do
  echo "[synth-eval] === $len (readerattn_topk=$RATOPK): L7 gpu0-2, L6 gpu3-5, REAL gpu6-7 ==="
  run_shard L7 $len 0 0 3; run_shard L7 $len 1 1 3; run_shard L7 $len 2 2 3
  run_shard L6 $len 3 0 3; run_shard L6 $len 4 1 3; run_shard L6 $len 5 2 3
  run_shard REAL $len 6 0 2; run_shard REAL $len 7 1 2
  wait
  echo "[synth-eval] $len done."
done
echo "[synth-eval] ALL COMPLETE (tag=$TAG)."
