#!/bin/bash
# Qwen3-32B QCMem BABILong qa5 formal n=500, chat_template + enable_thinking=False.
# One task = (qa5, length, shard). 7 lengths x 4 shards = 28 jobs, dynamic 8-GPU pool.
set -uo pipefail
cd /volume/haru/Mixture-of-Memory
export PYTHONPATH=.:third_party/babilong-pkg
export PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
PY=/volume/haru/Mixture-of-Memory/.venv_hy3/bin/python
EVAL=scripts/_eval_qcmem_babilong_disable_thinking_tmp.py
MODEL=models/Qwen3-32B
DATA=data/babilong-1k-samples
OUT=babilong_results/qwen32_qa5_disablethinking_n500_j16_chunk512
LOG=logs/qwen32_qa5_disablethinking_n500
mkdir -p "$OUT" "$LOG"
QUEUE="$LOG/jobs.queue"
LOCK="$LOG/jobs.lock"
: > "$QUEUE"
for len in 0k 1k 2k 4k 8k 16k 32k; do
  for si in 0 1 2 3; do
    echo "$len $si" >> "$QUEUE"
  done
done
pop_job() {
  local line
  exec 9>"$LOCK"
  flock 9
  line=$(head -n 1 "$QUEUE" || true)
  if [ -n "$line" ]; then
    tail -n +2 "$QUEUE" > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"
  fi
  flock -u 9
  echo "$line"
}
worker() {
  local gpu="$1" line len si name log rc
  while true; do
    line=$(pop_job)
    [ -z "$line" ] && break
    read -r len si <<<"$line"
    name="qwen32_qa5_disablethink_${len}_shard${si}of4"
    log="$LOG/${name}.log"
    echo "[$(date -Is)] gpu${gpu} START len=${len} shard=${si}" | tee -a "$LOG/pool.log"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$EVAL" \
      --model_path "$MODEL" --resume_j 16 --selector bm25 --topk 12 \
      --chunk_size 512 --sink_tokens bos --dtype bfloat16 --attn_impl sdpa --device cuda:0 \
      --tasks qa5 --lengths "$len" --limit 500 --max_new_tokens 20 --use_chat_template \
      --dataset_name "$DATA" \
      --num_shards 4 --shard_index "$si" \
      --results_folder "$OUT" --output_name "$name" >"$log" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ]; then
      echo "[$(date -Is)] gpu${gpu} FAIL rc=${rc} len=${len} shard=${si} log=${log}" | tee -a "$LOG/pool.log"
      echo "$len $si" >> "$LOG/failed.jobs"
    else
      echo "[$(date -Is)] gpu${gpu} DONE len=${len} shard=${si}" | tee -a "$LOG/pool.log"
    fi
  done
  echo "[$(date -Is)] gpu${gpu} DRAIN" | tee -a "$LOG/pool.log"
}
for gpu in 0 1 2 3 4 5 6 7; do
  worker "$gpu" &
done
wait
"$PY" - <<"PY"
import csv, glob, json, os
from babilong.metrics import compare_answers, TASK_LABELS
base="babilong_results/qwen32_qa5_disablethinking_n500_j16_chunk512"
lengths=["0k","1k","2k","4k","8k","16k","32k"]
summary=[]
for length in lengths:
    files=sorted(glob.glob(f"{base}/qwen32_qa5_disablethink_{length}_shard*of4/*.csv"))
    correct=total=0
    for f in files:
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh):
                total += 1
                correct += bool(compare_answers(r["target"], r["output"], r["question"], TASK_LABELS["qa5"]))
    summary.append({"task":"qa5","length":length,"correct":correct,"n":total,"score":round(100*correct/total,2) if total else 0.0,"num_csv":len(files)})
os.makedirs(base, exist_ok=True)
with open(os.path.join(base,"_summary.json"),"w") as fh: json.dump(summary, fh, indent=2)
print("SUMMARY")
for r in summary:
    print(f"qa5 {r[length]:>3s}: {r[correct]}/{r[n]} = {r[score]:.2f}% ({r[num_csv]} csv)")
PY
echo "ALL_DONE $(date -Is)"
