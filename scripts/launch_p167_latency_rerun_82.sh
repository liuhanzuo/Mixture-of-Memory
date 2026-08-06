#!/usr/bin/env bash
# ============================================================================
# #167 — independent RE-RUN of the Paper A depth-replay Read-latency measurement
# that backs paperA/sections/tab_replay_latency.tex (j=0 vs j=12 Read ms).
#
# WHY THIS EXACT CONFIG (do not "fix" it to the p0_12_acceptance config):
#   The tex values 931.9 / 664.4 / 1.403x are reproduced BIT-EXACTLY by
#   bench_results/p0_13_quality_latency/latency/latency_proc{0,1,2}.json
#   (pooled median over 3 procs x 20 reads). Therefore the authoritative config
#   is the P0.13 one:  seed=42, max_new_tokens=48, iter_hop_topk=4,
#   example_index=0, task=niah_single_3, length=16k, PYTHONHASHSEED=0.
#   The p0_12_acceptance config (seed=0, n_decode=6) is a DIFFERENT experiment
#   and yields 1080.9 / 785.7 ms — it does NOT back this table.
#
# Protocol (identical to the original P0.13 latency leg):
#   3 independent processes, each on its OWN GPU, run concurrently; each does
#   3 warmups + 20 timed reads per arm on ONE fixed pack. Arms differ ONLY in
#   resume_j (0 vs 12). Pooled median/p10/p90 over the 3x20 = 60 timed reads.
#
# Node: .82 (8xH20, zwfy6). MUST be exclusive — no co-resident jobs.
# Env : /opt/conda/envs/torch-base/bin/python (torch 2.13.0 / cu13.2 / py3.14.6),
#       the same env the original measurement used.
#
# Usage on .82:
#   PROJECT_ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory \
#   setsid nohup bash scripts/launch_p167_latency_rerun_82.sh \
#       >logs/p167_latency_rerun.out 2>&1 </dev/null &
# ============================================================================
set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT" || { echo "CD_FAILED $PROJECT_ROOT"; exit 3; }

PYBIN="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0          # deterministic hash((task,length)) sample seed
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$PROJECT_ROOT/.hf_cache"
export http_proxy="" https_proxy="" all_proxy=""
export CUDA_DEVICE_MAX_CONNECTIONS=1

MODEL="${MODEL:-models/Qwen3-8b-local}"
LORA="${LORA:-outputs/qcmem_distill_qwen_j12_r32_4k/final}"
OUTDIR="${OUTDIR:-bench_results/p0_167_latency_rerun}"
DRIVER="scripts/bench_p0_13_quality_latency.py"
LOGDIR="logs/p167_latency_rerun"
mkdir -p "$LOGDIR" "$OUTDIR"

# ---- exclusivity gate: refuse to measure latency if any GPU is busy ---------
BUSY=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>200{c++}END{print c+0}')
if [ "$BUSY" -ne 0 ]; then
    echo "[p167][ABORT] $BUSY GPU(s) busy — latency measurement requires an idle node."
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
    exit 5
fi
echo "[p167] exclusivity OK: all 8 GPUs idle."

# P0.13 latency-leg config, copied field-for-field from the record that
# reproduces the tex numbers (latency_proc*.json "config").
COMMON="--model_path $MODEL --lora_adapter $LORA --resume_j_a 0 --resume_j_b 12 \
--topk 12 --iter_hop_topk 4 --chunk_size 512 --max_new_tokens 48 \
--dtype bfloat16 --attn_impl sdpa --seed 42 --output_dir $OUTDIR"

# ---- STEP 0: manifest strict-fix gate (backbone key-tensor + LoRA sha) ------
echo "[p167] STEP 0: manifest / strict-fix gate on GPU 0"
CUDA_VISIBLE_DEVICES=0 $PYBIN $DRIVER --mode manifest $COMMON \
    >"$LOGDIR/manifest.out" 2>&1
if [ $? -ne 0 ]; then
    echo "[p167] MANIFEST ABORTED — refusing to run:"; tail -20 "$LOGDIR/manifest.out"
    exit 3
fi
echo "[p167] manifest OK (LoRA sha + backbone shas match the P0.12/P0.13 record)."

# ---- STEP 1: 3 independent latency processes, one per GPU, concurrent -------
echo "[p167] STEP 1: 3 latency procs (GPU 0,1,2), warmup=3 n_repeat=20 each"
pids=()
for p in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$p $PYBIN $DRIVER --mode latency $COMMON \
        --task niah_single_3 --length 16k --proc_id "$p" \
        --example_index 0 --warmup 3 --n_repeat 20 \
        >"$LOGDIR/latency_proc$p.out" 2>&1 &
    pids+=($!)
done
rc_all=0
for i in 0 1 2; do
    wait "${pids[$i]}" || { echo "[p167] proc$i FAILED rc=$?"; rc_all=1; }
done
echo "[p167] latency procs done (rc_all=$rc_all)."
for p in 0 1 2; do echo "--- proc$p ---"; tail -3 "$LOGDIR/latency_proc$p.out"; done

# ---- STEP 2: aggregate + compare against tex / p0_13 / p0_12 ---------------
echo "[p167] STEP 2: aggregate"
$PYBIN scripts/aggregate_p167_latency.py \
    --rerun_dir "$OUTDIR/latency" \
    --orig_p0_13_dir bench_results/p0_13_quality_latency/latency \
    --p0_12_dir bench_results/p0_12_acceptance \
    --out "$OUTDIR/p167_latency_summary.json" 2>&1 | tee "$LOGDIR/aggregate.out"
echo "[p167] COMPLETE."
