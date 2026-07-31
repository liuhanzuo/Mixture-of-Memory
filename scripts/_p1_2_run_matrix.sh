#!/usr/bin/env bash
# P1.2 content-depth probe full matrix launcher (node .73, 8x H20).
# 3 models x 3 tasks = 9 jobs, one per GPU (GPU0 runs 2). All settings full.
set -u
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory || exit 1

export http_proxy="http://hy-proxy.woa.com:3128"
export https_proxy="http://hy-proxy.woa.com:3128"
export all_proxy="http://hy-proxy.woa.com:3128"
export no_proxy="mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,localhost,127.0.0.1,.oa.com,.woa.com,.local"
export PYTHONPATH=.p1_2_pylibs
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0

PY=/opt/conda/envs/torch-base/bin/python
RES=results/p1_2
mkdir -p "$RES" logs
rm -f logs/p1_2_ALLDONE

MODELS=(
  "qwen3_8b:/apdcephfs_zwfy6/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b"
  "llama3_8b:/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/models/Meta-Llama-3-8B"
  "olmo2_7b:/apdcephfs_zwfy6/share_304376610/pighzliu_code/models/OLMo-2-1124-7B"
)
TASKS=(SST2 WiC RTE)

idx=0
for mspec in "${MODELS[@]}"; do
  mname="${mspec%%:*}"; mpath="${mspec#*:}"
  for task in "${TASKS[@]}"; do
    gpu=$(( idx % 8 ))
    out="$RES/${mname}_${task}.json"
    log="logs/p1_2_${mname}_${task}.log"
    echo "[launch] idx=$idx gpu=$gpu $mname $task -> $out"
    CUDA_VISIBLE_DEVICES=$gpu setsid nohup "$PY" scripts/probe_p1_2_content_depth.py \
      --mode run --model_path "$mpath" --task "$task" \
      --device cuda:0 --dtype bf16 --max_len 128 --batch_size 32 \
      --n_pool 3000 --n_native 1000 --seeds 0,1,2,3,4 \
      --c_grid 0.1,1.0,10.0 --n_jobs 12 --results_dir "$RES" \
      --out "$out" > "$log" 2>&1 &
    idx=$(( idx + 1 ))
    sleep 3
  done
done

echo "[launched] $idx jobs; waiting..."
wait
echo "ALLDONE $(date -u +%FT%TZ)" > logs/p1_2_ALLDONE
echo "[done] all $idx jobs finished"
