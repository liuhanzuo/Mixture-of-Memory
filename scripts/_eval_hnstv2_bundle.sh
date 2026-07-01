#!/bin/bash
# HNST v2 eval bundle (2026-06-25): run after a checkpoint lands. Judges BOTH walls
# with official compare_answers scoring + needle-recall probe + pg19 ppl guardrail.
#   $1 = checkpoint .pt   $2 = adapter_config.json   $3 = GPU id (default 0)   $4 = tag
# Runs (each on ONE GPU; keep them serial unless you shard GPUs yourself):
#   A) needle-recall probe: v2tree vs v1max-pool vs flat vs b25 (selection wall)
#   B) standard mem-chain end-to-end (tree-token select + reforward) qa5 各档
#   C) fullchain oracle qa5 各档 (readout upper bound, excludes selector)
#   D) pg19 sliding ppl guardrail
set -euo pipefail
R="/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory"
cd "$R"
export HF_HOME="$R/.hf_cache" HF_DATASETS_CACHE="$R/.hf_cache/datasets"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$R/third_party/babilong-pkg:$R:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="$R/.venv/bin/python"
CKPT="${1:?need checkpoint}"; ACFG="${2:?need adapter_config}"; GPU="${3:-0}"; TAG="${4:-hnstv2}"
mkdir -p logs babilong_results/$TAG

echo "=== (A) needle-recall probe (selection wall) ==="
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/hnst_v2_needle_recall_probe.py \
  --adapter_config "$ACFG" --checkpoint "$CKPT" \
  --task qa5 --lengths 8k 16k 32k --limit 100 --branch 4 --beam 2 --topk 8 \
  --out logs/${TAG}_needle_recall.jsonl 2>&1 | grep -viE "warn|it/s" | tail -5

echo "=== (B) standard mem-chain end-to-end: tree-token select + reforward, qa5 ==="
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/run_babilong_mem_space.py \
  --model_path models/Meta-Llama-3-8B --adapter_config "$ACFG" --checkpoint "$CKPT" \
  --tasks qa5 --lengths 2k 4k 8k 16k --limit 100 --batch_size 1 --chunk_size 512 \
  --fifo_keep_all_buffer \
  --swa_tree_token --swa_tree_select_layer 16 --swa_tree_topk 4 --swa_tree_branch 4 --swa_tree_beam 2 \
  --results_folder babilong_results/${TAG}_memchain_tree 2>&1 | grep -viE "warn|it/s" | tail -8

echo "=== (C) fullchain oracle qa5 (readout upper bound) ==="
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/probe_fullchain_oracle_qa5.py \
  --model_path models/Meta-Llama-3-8B --checkpoint "$CKPT" --adapter_config "$ACFG" \
  --tasks qa5 --lengths 2k 4k 8k 16k --limit 100 --oracle_mode fullchain \
  --results_folder babilong_results/${TAG}_fullchain --device cuda:0 2>&1 | grep -viE "warn|it/s" | tail -8

echo "=== (D) pg19 sliding ppl guardrail (v2 vs A-model, identical config) ==="
echo "--- v2 ---"
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/eval_sliding_ppl.py --data pg19 \
  --data_path data/pg19_chunks_llama3_noeos.npy \
  --model_path models/Meta-Llama-3-8B --adapter_config "$ACFG" --checkpoint "$CKPT" \
  --seq_length 16384 --chunk_size 512 2>&1 | grep -viE "warn|it/s" | tail -6
echo "--- A-model step2000 (baseline) ---"
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/eval_sliding_ppl.py --data pg19 \
  --data_path data/pg19_chunks_llama3_noeos.npy \
  --model_path models/Meta-Llama-3-8B \
  --adapter_config outputs/mem_space_fifo_b25_c512_supervised_select/adapter_config.json \
  --checkpoint outputs/mem_space_fifo_b25_c512_supervised_select/full_model_step002000.pt \
  --seq_length 16384 --chunk_size 512 2>&1 | grep -viE "warn|it/s" | tail -6

echo "ALL EVAL DONE tag=$TAG"
