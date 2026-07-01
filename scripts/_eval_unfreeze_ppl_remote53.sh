#!/bin/bash
# pg19 sliding-window PPL guardrail for one unfreeze-sweep checkpoint.
# Replicates the A-model / MVP guardrail config exactly (seq16384, window8192,
# stride4096, chunk512, pg19 noeos) so avg_nll is directly comparable to:
#   A-model step2000 = 0.4280 (ppl 1.5342)  |  5-layer MVP step800 = 0.4654 (ppl 1.5927)
# Args: $1=arm tag  $2=checkpoint .pt  $3=adapter_config.json  $4=gpu id
set -euo pipefail
R="/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory"
cd "$R"
export HF_HOME="$R/.hf_cache" HF_DATASETS_CACHE="$R/.hf_cache/datasets"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$R/third_party/babilong-pkg:$R:${PYTHONPATH:-}"
TAG="$1"; CKPT="$2"; ACFG="$3"; GPU="${4:-0}"
mkdir -p babilong_results logs/unfreeze_eval
echo "[ppl] $TAG ckpt=$CKPT gpu=$GPU"
"$R/.venv/bin/python" -u scripts/eval_sliding_ppl.py \
  --data pg19 --data_path data/pg19_chunks_llama3_noeos.npy \
  --model_path models/Meta-Llama-3-8B \
  --adapter_config "$ACFG" --checkpoint "$CKPT" \
  --seq_length 16384 --window 8192 --stride 4096 --chunk_size 512 \
  --gpu "$GPU" --dtype bfloat16 --attn_impl sdpa \
  --output_json "babilong_results/ppl_unfreeze_${TAG}.json" \
  > "logs/unfreeze_eval/ppl_${TAG}.log" 2>&1
echo "[ppl] $TAG done -> babilong_results/ppl_unfreeze_${TAG}.json"
grep -iE 'avg_nll|ppl' "logs/unfreeze_eval/ppl_${TAG}.log" | tail -3 || true
