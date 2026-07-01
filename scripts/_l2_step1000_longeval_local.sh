#!/bin/bash
set -u
cd /apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory
PY=.venv/bin/python
CKPT=outputs/expL2ON_N128/mem_space_adapter.pt
LENS=(1k 2k 4k 8k 16k 32k)
for idx in "${!LENS[@]}"; do
  L=${LENS[$idx]}
  CUDA_VISIBLE_DEVICES=$idx $PY scripts/eval_longeval_mem_space.py \
    --model_path models/Meta-Llama-3-8B \
    --checkpoint $CKPT \
    --adapter_config outputs/expL2ON_N128/adapter_config.json \
    --results_folder ./longeval_results \
    --output_name L2_step1000_longeval_${L} \
    --lengths $L --num_samples 50 --chunk_size 512 \
    > logs/L2_step1000_longeval_${L}.out 2>&1 &
done
wait
echo "DONE L2 step1000 longeval all lengths"
