#!/bin/bash
set -u
cd /apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory
PY=/opt/conda/envs/torch-base/bin/python
CKPT=outputs/expL2ON_N128/mem_space_adapter_step001000.pt
for i in $(seq 1 360); do [ -f "$CKPT" ] && break; sleep 10; done
[ -f "$CKPT" ] || { echo "TIMEOUT waiting for $CKPT"; exit 1; }
sleep 30
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
