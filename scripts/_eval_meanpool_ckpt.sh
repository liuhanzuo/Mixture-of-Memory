#!/usr/bin/env bash
# mean-pool ckpt eval: reader-attn recall(MEM_SALIENCE_QPOOL=mean) vs 纯推理0.37 / bm25 0.53
# 用法: bash _eval_meanpool_ckpt.sh <run_name> <step>
set -uo pipefail
R="/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory"; cd "$R"; PY="$R/.venv/bin/python"
export PYTHONUNBUFFERED=1 PYTHONPATH="$R/third_party/babilong-pkg:$R" HF_HOME="$R/.hf_cache" HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MEM_SALIENCE_QPOOL=mean   # ★关键: eval也用mean-pool探针(训练推理一致)
RUN=${1:-mem_space_select_meanpool_fmt_g8k}; STEP=${2:-300}
CKPT=$(ls outputs/$RUN/*step$(printf '%06d' $STEP)*.pt 2>/dev/null|head -1)
[ -z "$CKPT" ]&&CKPT=$(ls outputs/$RUN/*step${STEP}*.pt 2>/dev/null|head -1)
[ -z "$CKPT" ]&&{ echo "ckpt step$STEP 未找到: $(ls outputs/$RUN/*.pt 2>/dev/null)"; exit 1; }
ACFG="$R/outputs/mem_space_fifo_b25_c512_supervised_select/adapter_config.json"; MODEL="$R/models/Meta-Llama-3-8B"
sp=($(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits|awk '$2<2000{print $1}'))
echo "[eval $RUN step$STEP, QPOOL=mean, ckpt=$CKPT, 空卡${sp[*]}]"
# P1 recall probe(S1=reader-attn mean-pool, S2=bm25), qa5 16k n50
CUDA_VISIBLE_DEVICES=${sp[0]} $PY scripts/e2_meanpool_probe.py --model_path "$MODEL" --checkpoint "$CKPT" --adapter_config "$ACFG" --tasks qa5 --lengths 16k --limit 50 --scorers S1,S1mp,S2 --output_csv /tmp/eval_${RUN}_s${STEP}_qa5.csv >logs/eval_${RUN}_s${STEP}.log 2>&1 &
echo "eval started PID=$!"
