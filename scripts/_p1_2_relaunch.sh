#!/usr/bin/env bash
# P1.2 RELAUNCH of the 7 incomplete (model,task) jobs with STRICT thread
# pinning (OpenBLAS was ignoring OMP/MKL -> 8 cores/worker -> oversubscription).
# The 2 finished SST2 JSONs (qwen3_8b_SST2, olmo2_7b_SST2) are kept untouched.
set -u
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory || exit 1

export http_proxy="http://hy-proxy.woa.com:3128"
export https_proxy="http://hy-proxy.woa.com:3128"
export all_proxy="http://hy-proxy.woa.com:3128"
export no_proxy="mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,localhost,127.0.0.1,.oa.com,.woa.com,.local"
export PYTHONPATH=.p1_2_pylibs
# pin EVERY BLAS backend so each loky worker uses exactly 2 threads
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export VECLIB_MAXIMUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0

PY=/opt/conda/envs/torch-base/bin/python
RES=results/p1_2
NJ=16
mkdir -p "$RES" logs
rm -f logs/p1_2_ALLDONE

QW=/apdcephfs_zwfy6/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b
LL=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/models/Meta-Llama-3-8B
OL=/apdcephfs_zwfy6/share_304376610/pighzliu_code/models/OLMo-2-1124-7B

# 7 incomplete jobs: name gpu model task
JOBS=(
  "llama3_8b:0:$LL:SST2"
  "qwen3_8b:1:$QW:WiC"
  "qwen3_8b:2:$QW:RTE"
  "llama3_8b:3:$LL:WiC"
  "llama3_8b:4:$LL:RTE"
  "olmo2_7b:5:$OL:WiC"
  "olmo2_7b:6:$OL:RTE"
)

n=0
for spec in "${JOBS[@]}"; do
  IFS=':' read -r mname gpu mpath task <<< "$spec"
  out="$RES/${mname}_${task}.json"
  log="logs/p1_2_${mname}_${task}.log"
  echo "[launch] gpu=$gpu $mname $task -> $out"
  CUDA_VISIBLE_DEVICES=$gpu setsid nohup "$PY" scripts/probe_p1_2_content_depth.py \
    --mode run --model_path "$mpath" --task "$task" \
    --device cuda:0 --dtype bf16 --max_len 128 --batch_size 32 \
    --n_pool 3000 --n_native 1000 --seeds 0,1,2,3,4 \
    --c_grid 0.1,1.0,10.0 --n_jobs $NJ --results_dir "$RES" \
    --out "$out" > "$log" 2>&1 &
  n=$(( n + 1 ))
  sleep 2
done

echo "[launched] $n jobs; waiting..."
wait
echo "ALLDONE $(date -u +%FT%TZ)" > logs/p1_2_ALLDONE
echo "[done] all $n jobs finished"
