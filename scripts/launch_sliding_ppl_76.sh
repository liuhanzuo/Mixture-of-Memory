#!/usr/bin/env bash
# Sliding-window long-context PPL matrix launcher for node .76 (diskA, 8x H20).
# 3 datasets x 2 models = 6 jobs, one GPU each:
#   base : GPU 0(pg19) 1(proofpile) 2(codeparrot)
#   mem  : GPU 4(pg19) 5(proofpile) 6(codeparrot)
# Usage: bash scripts/launch_sliding_ppl_76.sh
set -u

PROJECT_ROOT=${PROJECT_ROOT:-/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory}
PYTHON_BIN=${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}
cd "$PROJECT_ROOT"

MODEL=${MODEL:-models/Meta-Llama-3-8B}
CKPT=${CKPT:-outputs/mem_space_p11_chunk1024_deltarule_normreadout/mem_space_adapter.pt}
ADAPTER=${ADAPTER:-outputs/mem_space_p11_chunk1024_deltarule_normreadout/adapter_config.json}
SEQLEN=${SEQLEN:-32768}
WINDOW=${WINDOW:-8192}
STRIDE=${STRIDE:-4096}
CHUNK=${CHUNK:-1024}
MAXTOK=${MAXTOK:-1310720}   # 40 seqs of 32k
OUTROOT=${OUTROOT:-ppl_results}
export WANDB_MODE=offline

declare -A DPATH=(
  [pg19]=data/pg19_real_llama3_noeos.npy
  [proofpile]=data/proofpile_llama3_noeos.npy
  [codeparrot]=data/codeparrot_llama3_noeos.npy
)
mkdir -p logs/sliding_ppl "$OUTROOT"

run_base () { # $1=dataset $2=gpu
  local d=$1 g=$2
  CUDA_VISIBLE_DEVICES=$g nohup $PYTHON_BIN scripts/eval_sliding_ppl.py \
    --data "$d" --data_path "${DPATH[$d]}" --model_path "$MODEL" \
    --seq_length "$SEQLEN" --window "$WINDOW" --stride "$STRIDE" \
    --max_tokens "$MAXTOK" --gpu 0 \
    --output_json "$OUTROOT/${d}_base.json" \
    > logs/sliding_ppl/${d}_base.out 2>&1 &
}
run_mem () { # $1=dataset $2=gpu
  local d=$1 g=$2
  CUDA_VISIBLE_DEVICES=$g nohup $PYTHON_BIN scripts/eval_sliding_ppl.py \
    --data "$d" --data_path "${DPATH[$d]}" --model_path "$MODEL" \
    --adapter_config "$ADAPTER" --checkpoint "$CKPT" \
    --seq_length "$SEQLEN" --chunk_size "$CHUNK" \
    --max_tokens "$MAXTOK" --gpu 0 \
    --output_json "$OUTROOT/${d}_mem.json" \
    > logs/sliding_ppl/${d}_mem.out 2>&1 &
}

# Stagger launches: each model load transiently spikes CPU RAM (safetensors +
# 10GB mem ckpt torch.load). Loading 6 at once OOM-kills the pod cgroup, so we
# delay between launches to keep peak concurrent loads to ~1-2.
STAGGER=${STAGGER:-75}
run_base pg19 0;        sleep "$STAGGER"
run_base proofpile 1;   sleep "$STAGGER"
run_base codeparrot 2;  sleep "$STAGGER"
run_mem  pg19 4;        sleep "$STAGGER"
run_mem  proofpile 5;   sleep "$STAGGER"
run_mem  codeparrot 6

echo "[launch] 6 PPL jobs started staggered (base GPU0-2, mem GPU4-6); seqlen=$SEQLEN maxtok=$MAXTOK stagger=${STAGGER}s"
wait
echo "[launch] all PPL jobs finished"
for f in "$OUTROOT"/*.json; do echo "== $f =="; cat "$f"; echo; done
echo "[launch] DONE"
