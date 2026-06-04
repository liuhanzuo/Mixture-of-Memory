#!/usr/bin/env bash
# R3-3 base LongBench anchor: plain Llama-3-8B (no adapter), middle-truncation.
# Matches R3-1 (perdoc_chunk128_local) protocol: non-instruct model, NO chat template.
# 8-GPU sharded. Run on a free disk-A node. Then scores.
set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT" CUDA_DEVICE_MAX_CONNECTIONS=1 TOKENIZERS_PARALLELISM=false
export http_proxy="http://hy-proxy.woa.com:3128" https_proxy="http://hy-proxy.woa.com:3128"
PYBIN="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
OUT=longbench_results/base_model_full_lb
DSETS="hotpotqa narrativeqa qasper multifieldqa_en 2wikimqa musique"
mkdir -p "$OUT/logs"
rm -f "$OUT"/*.jsonl
for G in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$G setsid "$PYBIN" scripts/eval_longbench_mem_space.py \
    --base_mode --no_chat_template --model_path models/Meta-Llama-3-8B \
    --checkpoint outputs/mem_space_perdoc_chunk128/mem_space_adapter.pt \
    --adapter_config outputs/mem_space_perdoc_chunk128/adapter_config.json \
    --output_dir "$OUT" --chunk_size 128 --gpu_id $G --num_gpus 8 \
    --datasets $DSETS \
    </dev/null >"$OUT/logs/gpu_${G}.log" 2>&1 &
done
echo "launched 8 base-LB shards -> $OUT"
wait
echo "all shards done; scoring"
"$PYBIN" scripts/eval_longbench_mem_space.py --score_only --output_dir "$OUT" --datasets $DSETS 2>&1 | tee "$OUT/logs/scoring.log"
