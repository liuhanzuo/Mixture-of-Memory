#!/usr/bin/env bash
# _qcmem_n100_drain.sh <partition_id|q> <stride> [cuda_dev]
# 两种模式:
#  - 分区模式: partition_id=数字, 跑 cell_index % stride == partition_id 的 cell (静态,partition完早会空转).
#  - ★队列模式: 第1参=q, 第3参=cuda_dev. 用 mkdir 原子锁抢任意未测 cell, 跑完继续抢下一个, 不空转.
#    多卡各起一个 q worker (共享同一 OUT), 协同排空整网格无空闲.
set -u
MODE="${1:?partition_id|q}"; STRIDE="${2:-1}"; CUDA_DEV="${3:-$MODE}"
ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"; cd "$ROOT"
PY="${PYTHON_BIN:-.venv/bin/python}"
M="${MODEL_PATH:?}"; CK="${LORA_CK:?}"; OUT="${OUT_DIR:?}"
SEL="${SELECTOR:-bm25}"                 # bm25(默认) 或 reader_attn 等
PFX="${NAME_PREFIX:-qcmem_n100}"        # 输出名前缀(不同selector用不同前缀避撞名)
CHUNK="${CHUNK_SIZE:-512}"              # QCMem chunk_size (默认512; 消融用256/1024)
export HF_HOME="$PWD/.hf_cache" HF_DATASETS_CACHE="$PWD/.hf_cache/datasets" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline PYTHONPATH="$PWD" PATH=/opt/conda/bin:$PATH
TASKS="niah_single niah_multikey vt"; LENS="1k 2k 4k 8k 16k 32k"; TOPKS="4 8 12 16 24"
mkdir -p "$OUT/.locks" 2>/dev/null
# 外层循环: 一个 worker 反复扫全网格抢未测 cell, 直到某趟一个都没抢到(全网格已测/在跑) 才退出.
# 这样 worker 不会单趟做完自己那份就退出留卡空转; 队列模式下多 worker 协同排空到 90/90 无需补卡.
while true; do
 claimed_this_pass=0
 i=0
 for task in $TASKS; do for len in $LENS; do for tk in $TOPKS; do
  run=0
  if [ "$MODE" = "q" ]; then run=1
  elif [ $((i % STRIDE)) -eq "$MODE" ]; then run=1; fi
  if [ "$run" -eq 1 ]; then
    o="${PFX}_${task}_tk${tk}_${len}"
    if [ ! -d "$OUT/$o" ] && mkdir "$OUT/.locks/$o" 2>/dev/null; then
      claimed_this_pass=$((claimed_this_pass+1))
      echo "[drain m$MODE dev$CUDA_DEV sel=$SEL] $o $(date +%H:%M:%S)"
      CUDA_VISIBLE_DEVICES="$CUDA_DEV" "$PY" scripts/eval_ruler_qcmem.py --model_path "$M" --lora_adapter "$CK" \
        --resume_j 12 --selector "$SEL" --topk "$tk" --ruler_tasks "$task" --lengths "$len" --limit 100 \
        --chunk_size "$CHUNK" --device cuda:0 --output_name "$o" --results_folder "$OUT" >"logs/qcw_drain_${o}.log" 2>&1
    fi
  fi
  i=$((i+1))
 done; done; done
 # 分区模式单趟即可(自己那份跑完); 队列模式若本趟没抢到新 cell 说明全网格已被瓜分完 → 退出
 [ "$MODE" != "q" ] && break
 [ "$claimed_this_pass" -eq 0 ] && break
done
echo "DRAIN_${MODE}_dev${CUDA_DEV}_DONE $(date +%H:%M:%S)"
