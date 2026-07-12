#!/usr/bin/env bash
# _qcmem_n100_drain.sh <partition_id> <stride> [cuda_dev]
# 本进程顺序跑所有 cell_index % stride == partition_id 且未测(无目录)的 cell.
# cuda_dev 默认=partition_id(8卡满时直接用); 手动指定物理卡时分离(如仅部分卡空).
set -u
GPU="${1:?partition_id}"; STRIDE="${2:?stride}"; CUDA_DEV="${3:-$GPU}"
ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"; cd "$ROOT"
PY="${PYTHON_BIN:-.venv/bin/python}"
M="${MODEL_PATH:?}"; CK="${LORA_CK:?}"; OUT="${OUT_DIR:?}"
SEL="${SELECTOR:-bm25}"                 # bm25(默认) 或 reader_attn 等
PFX="${NAME_PREFIX:-qcmem_n100}"        # 输出名前缀(不同selector用不同前缀避撞名)
export HF_HOME="$PWD/.hf_cache" HF_DATASETS_CACHE="$PWD/.hf_cache/datasets" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline PYTHONPATH="$PWD" PATH=/opt/conda/bin:$PATH
TASKS="niah_single niah_multikey vt"; LENS="1k 2k 4k 8k 16k 32k"; TOPKS="4 8 12 16 24"
i=0
for task in $TASKS; do for len in $LENS; do for tk in $TOPKS; do
  if [ $((i % STRIDE)) -eq "$GPU" ]; then
    o="${PFX}_${task}_tk${tk}_${len}"
    if [ ! -d "$OUT/$o" ]; then
      echo "[drain p$GPU dev$CUDA_DEV sel=$SEL] $o $(date +%H:%M:%S)"
      CUDA_VISIBLE_DEVICES="$CUDA_DEV" "$PY" scripts/eval_ruler_qcmem.py --model_path "$M" --lora_adapter "$CK" \
        --resume_j 12 --selector "$SEL" --topk "$tk" --ruler_tasks "$task" --lengths "$len" --limit 100 \
        --chunk_size 512 --device cuda:0 --output_name "$o" --results_folder "$OUT" >"logs/qcw_drain_${o}.log" 2>&1
    fi
  fi
  i=$((i+1))
done; done; done
echo "DRAIN_GPU${GPU}_DONE $(date +%H:%M:%S)"
