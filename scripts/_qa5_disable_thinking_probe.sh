#!/bin/bash
cd /volume/haru/Mixture-of-Memory
export PYTHONPATH=.:third_party/babilong-pkg
export PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
PY=/volume/haru/Mixture-of-Memory/.venv_hy3/bin/python
EVAL=scripts/_eval_qcmem_babilong_disable_thinking_tmp.py
DATA=data/babilong-1k-samples
MODEL=models/Qwen3-32B
OUT=babilong_results/qwen32_qa5_disable_thinking_probe
mkdir -p "$OUT" logs/qa5_disable_thinking_probe
gpus=(0 1 2 3 4 5)
i=0
for mnt in 20 128; do
  for len in 0k 8k 32k; do
    gpu=${gpus[$i]}
    CUDA_VISIBLE_DEVICES=$gpu $PY "$EVAL" \
      --model_path "$MODEL" --resume_j 16 --selector bm25 --topk 12 \
      --chunk_size 512 --sink_tokens bos --dtype bfloat16 --attn_impl sdpa --device cuda:0 \
      --tasks qa5 --lengths $len --limit 20 --max_new_tokens $mnt --use_chat_template \
      --dataset_name "$DATA" \
      --results_folder "$OUT" --output_name "disablethink_mnt${mnt}_${len}" \
      >logs/qa5_disable_thinking_probe/disablethink_mnt${mnt}_${len}.log 2>&1 &
    echo "launched gpu$gpu disable_thinking mnt=$mnt len=$len pid=$!"
    i=$((i+1))
  done
done
wait
echo "ALL_DONE"
