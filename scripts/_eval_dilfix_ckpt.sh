#!/usr/bin/env bash
# dilfix ckpt 待命eval: ckpt一落盘立即评 reader-attn recall@4 + token-reforward分 vs before(0.19/46)
# 用法: bash _eval_dilfix_ckpt.sh <step>  (如 300)
# 在 .7.53(训练所在, ckpt本地) 跑, 用训练腾出的卡 或 等训练间隙
set -uo pipefail
R="/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory"
cd "$R"; PY="$R/.venv/bin/python"
export PYTHONUNBUFFERED=1 PYTHONPATH="$R/third_party/babilong-pkg:$R" HF_HOME="$R/.hf_cache" HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
STEP=${1:-300}
RUN=mem_space_select_dilfix_g8k_curric24
# ckpt文件名(确认_save_adapter格式)
CKPT=$(ls outputs/$RUN/*step$(printf '%06d' $STEP)*.pt 2>/dev/null|head -1)
[ -z "$CKPT" ] && CKPT=$(ls outputs/$RUN/*step${STEP}*.pt 2>/dev/null|head -1)
[ -z "$CKPT" ] && { echo "ckpt step$STEP 未找到"; ls outputs/$RUN/*.pt 2>/dev/null; exit 1; }
ACFG="$R/outputs/$RUN/adapter_config.json"; [ -f "$ACFG" ] || ACFG="$R/outputs/mem_space_fifo_b25_c512_supervised_select/adapter_config.json"
MODEL="$R/models/Meta-Llama-3-8B"
echo "[eval dilfix ckpt=$CKPT]"
GPUS=($(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits|awk '$2<2000{print $1}'))
echo "空卡:${GPUS[*]}"
# reader-attn token-reforward 16k(主判据) topk4, 2-shard
RA="--swa_readerattn_token --swa_readerattn_topk 4 --swa_readerattn_select_layer 16"
i=0
for si in 0 1; do
  g=${GPUS[$((i%${#GPUS[@]}))]}; [ -z "$g" ]&&break
  CUDA_VISIBLE_DEVICES=$g $PY scripts/run_babilong_mem_space.py --model_path "$MODEL" --checkpoint "$CKPT" --adapter_config "$ACFG" \
    --results_folder babilong_results --output_name "DILFIX_s${STEP}/qa5_16k_readerattn" --tasks qa5 --lengths 16k --limit 50 --chunk_size 512 --batch_size 1 \
    --max_new_tokens 20 --dtype bfloat16 --attn_impl sdpa --use_instruction --use_examples --use_post_prompt \
    --num_shards 2 --shard_index $si $RA >logs/dilfix_eval_s${STEP}_ra_$si.log 2>&1 &
  i=$((i+1))
done
# P1 recall probe(reader-attn salience recall@4) 若probe在
if [ -f scripts/e2_multiscorer_probe.py ]; then
  g=${GPUS[$((i%${#GPUS[@]}))]}
  CUDA_VISIBLE_DEVICES=$g $PY scripts/e2_multiscorer_probe.py --model_path "$MODEL" --checkpoint "$CKPT" --adapter_config "$ACFG" --task qa5 --length 16k --limit 50 --output babilong_results/DILFIX_s${STEP}/p1_qa5_16k.csv >logs/dilfix_eval_s${STEP}_p1.log 2>&1 &
fi
wait
echo "DILFIX_EVAL_s${STEP}_DONE"
