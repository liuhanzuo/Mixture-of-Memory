#!/bin/bash
# qa5 CHAT-TEMPLATE probe: {0k,8k,32k} x {20,128}, n=20, uses GPUs 0,1,2,3,6,7
cd /volume/haru/Mixture-of-Memory
export PYTHONPATH=.:third_party/babilong-pkg
export PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
PY=/volume/haru/Mixture-of-Memory/.venv_hy3/bin/python
DATA=data/babilong-1k-samples
MODEL=models/Qwen3-32B
OUT=babilong_results/qwen32_qa5_chat_probe
mkdir -p "$OUT" logs/qa5_chat_probe
gpus=(0 1 2 3 6 7)
i=0
for mnt in 20 128; do
  for len in 0k 8k 32k; do
    gpu=${gpus[$i]}
    CUDA_VISIBLE_DEVICES=$gpu $PY scripts/eval_qcmem_babilong.py \
      --model_path "$MODEL" --resume_j 16 --selector bm25 --topk 12 \
      --chunk_size 512 --sink_tokens bos --dtype bfloat16 --attn_impl sdpa --device cuda:0 \
      --tasks qa5 --lengths $len --limit 20 --max_new_tokens $mnt --use_chat_template \
      --dataset_name "$DATA" \
      --results_folder "$OUT" --output_name "chat_mnt${mnt}_${len}" \
      >logs/qa5_chat_probe/chat_mnt${mnt}_${len}.log 2>&1 &
    echo "launched gpu$gpu chat mnt=$mnt len=$len pid=$!"
    i=$((i+1))
  done
done
wait
echo "ALL_DONE"
