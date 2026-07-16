#!/bin/bash
# Qwen3-32B RULER variable_tracking disable-thinking probe, n=100/cell.
set -uo pipefail
cd /volume/haru/Mixture-of-Memory
export PYTHONPATH=.:third_party/babilong-pkg
export PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
PY=/volume/haru/Mixture-of-Memory/.venv_hy3/bin/python
EVAL=scripts/_eval_ruler_qcmem_disable_thinking_tmp.py
MODEL=models/Qwen3-32B
OUT=ruler_results/qwen32_vt_disablethinking_n100_j16_chunk512
LOG=logs/qwen32_vt_disablethinking_n100
mkdir -p "$OUT" "$LOG"
QUEUE="$LOG/jobs.queue"; LOCK="$LOG/jobs.lock"; : > "$QUEUE"
for len in 8k 16k 32k; do
  for si in 0 1 2 3; do echo "$len $si" >> "$QUEUE"; done
done
pop_job(){ exec 9>"$LOCK"; flock 9; local line; line=$(head -n 1 "$QUEUE" || true); if [ -n "$line" ]; then tail -n +2 "$QUEUE" > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"; fi; flock -u 9; echo "$line"; }
worker(){ local gpu="$1" line len si name log rc; while true; do line=$(pop_job); [ -z "$line" ] && break; read -r len si <<<"$line"; name="qwen32_vt_disablethink_${len}_shard${si}of4"; log="$LOG/${name}.log"; echo "[$(date -Is)] gpu${gpu} START len=${len} shard=${si}" | tee -a "$LOG/pool.log"; CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$EVAL" --model_path "$MODEL" --resume_j 16 --selector iter_bm25 --topk 16 --iter_rounds 4 --iter_hop_topk 4 --chunk_size 512 --sink_tokens bos --dtype bfloat16 --attn_impl sdpa --device cuda:0 --ruler_tasks vt --lengths "$len" --limit 100 --max_new_tokens 60 --num_shards 4 --shard_index "$si" --results_folder "$OUT" --output_name "$name" >"$log" 2>&1; rc=$?; if [ "$rc" -ne 0 ]; then echo "[$(date -Is)] gpu${gpu} FAIL rc=${rc} len=${len} shard=${si}" | tee -a "$LOG/pool.log"; echo "$len $si" >> "$LOG/failed.jobs"; else echo "[$(date -Is)] gpu${gpu} DONE len=${len} shard=${si}" | tee -a "$LOG/pool.log"; fi; done; echo "[$(date -Is)] gpu${gpu} DRAIN" | tee -a "$LOG/pool.log"; }
for gpu in 0 1 2 3 4 5 6 7; do worker "$gpu" & done
wait
"$PY" - <<"PY"
import csv, glob, json, os
base="ruler_results/qwen32_vt_disablethinking_n100_j16_chunk512"
out=[]
for length in ["8k","16k","32k"]:
    files=sorted(glob.glob(f"{base}/qwen32_vt_disablethink_{length}_shard*of4/variable_tracking_{length}_shard*of4.csv"))
    total=0; rec=0.0
    for f in files:
        for r in csv.DictReader(open(f)):
            total += 1; rec += float(r["recall"])
    out.append({"task":"variable_tracking","length":length,"n":total,"score":round(100*rec/total,2) if total else 0.0,"num_csv":len(files)})
with open(os.path.join(base,"_summary.json"),"w") as fh: json.dump(out, fh, indent=2)
print("SUMMARY")
for r in out: print(f"VT {r[length]:>3s}: {r[score]:.2f} (n={r[n]}, csv={r[num_csv]})")
PY
echo "ALL_DONE $(date -Is)"
