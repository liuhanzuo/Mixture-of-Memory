#!/usr/bin/env bash
# One-shot orchestrator for .73 (diskB): eval keep14-distill step5000.pt on the
# base protocol (held-out PPL + core6 MC + know5 MC), THEN restart the distill
# heal from step5000.pt (BS=4 GA=4, resume). Verbatim harness of the s153500
# keep14 eval scripts (only NAME/CKPT changed) so distill@5000 is directly
# comparable to plain keep14-NTP. Runs the 3 evals sequentially on 8 GPUs, then
# hands the freed GPUs back to the distill training via the patched launcher.
#
# ⚠️ Distill must already be KILLED before launching this (GPUs must be free).
# Launch: setsid nohup bash scripts/_run_distill_step5000_eval_then_restart.sh >logs/distill_s5000_orch.log 2>&1 &
set -u
ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
cd "$ROOT"
PY=/opt/conda/envs/torch-base/bin/python
BASE="../models/OLMo-2-1124-7B"
CKPT="outputs/olmo2_probe2_7B_keep14fresh2_distill/step5000.pt"
NAME="7B_keep14distill_step5000"

export http_proxy=http://hy-proxy.woa.com:3128
export https_proxy=http://hy-proxy.woa.com:3128
export all_proxy=http://hy-proxy.woa.com:3128
export no_proxy="localhost,127.0.0.1,.oa.com,.woa.com,.local"
export HF_DATASETS_CACHE="$ROOT/data/hf_datasets_cache"
mkdir -p logs olmo2_ppl_results olmo2_downstream_results "$HF_DATASETS_CACHE"

echo "[$(date '+%F %T')] ============ ORCH START: eval $NAME (ckpt=$CKPT) then restart distill ============"

# ---------- Phase 1: held-out NTP PPL (base protocol, 2048-tok windows) ----------
VAL=data/dolmino_now_val.npy
echo "[$(date '+%F %T')] --- PHASE 1 PPL ---"
for g in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_probe2_ppl.py \
    --base_model "$BASE" --ckpt "$CKPT" \
    --keep_front_layers 14 --n_fresh_layers 2 \
    --val_path "$VAL" --num_shards 8 --shard_index $g --batch_size 4 \
    --output_name "$NAME" \
    > "logs/olmo2_ppl_${NAME}_shard${g}.log" 2>&1 &
done
wait
$PY scripts/eval_olmo2_probe2_ppl.py --merge --output_name "$NAME" 2>&1 || true
echo "[$(date '+%F %T')] PPL summary:"; cat "olmo2_ppl_results/${NAME}/summary.json" 2>/dev/null; echo

# ---------- Phase 2: core6 downstream MC ----------
CORE="hellaswag,arc_challenge,arc_easy,piqa,winogrande,openbookqa"
echo "[$(date '+%F %T')] --- PHASE 2 core6 ---"
$PY scripts/eval_olmo2_probe2_downstream.py --prepare_data --tasks "$CORE" \
  > "logs/olmo2_downstream_${NAME}_prepare.log" 2>&1 || true
for g in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_probe2_downstream.py \
    --base_model "$BASE" --ckpt "$CKPT" --tasks "$CORE" \
    --num_shards 8 --shard_index $g --batch_size 8 \
    --output_name "$NAME" \
    > "logs/olmo2_downstream_${NAME}_shard${g}.log" 2>&1 &
done
wait
$PY scripts/eval_olmo2_probe2_downstream.py --merge --output_name "$NAME" 2>&1 || true
echo "[$(date '+%F %T')] core6 summary:"; cat "olmo2_downstream_results/${NAME}/summary.json" 2>/dev/null; echo

# ---------- Phase 3: know5 downstream MC (incl. mmlu) ----------
KNOW="mmlu,lambada_openai,boolq,commonsense_qa,social_iqa"
KNAME="${NAME}_know"
echo "[$(date '+%F %T')] --- PHASE 3 know5 ---"
$PY scripts/eval_olmo2_probe2_downstream.py --prepare_data --tasks "$KNOW" \
  > "logs/olmo2_downstream_${KNAME}_prepare.log" 2>&1 || true
for g in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_probe2_downstream.py \
    --base_model "$BASE" --ckpt "$CKPT" --tasks "$KNOW" \
    --num_shards 8 --shard_index $g --batch_size 8 \
    --output_name "$KNAME" \
    > "logs/olmo2_downstream_${KNAME}_shard${g}.log" 2>&1 &
done
wait
$PY scripts/eval_olmo2_probe2_downstream.py --merge --output_name "$KNAME" 2>&1 || true
echo "[$(date '+%F %T')] know5 summary:"; cat "olmo2_downstream_results/${KNAME}/summary.json" 2>/dev/null; echo

echo "[$(date '+%F %T')] ============ EVAL DONE — restarting distill from step5000.pt (BS=4 GA=4) ============"

# ---------- Phase 4: restart distill heal, resume from step5000.pt ----------
RESUME_FROM="$ROOT/outputs/olmo2_probe2_7B_keep14fresh2_distill/step5000.pt" \
BS=4 GA=4 RUN=1 \
PROJECT_ROOT="$ROOT" \
PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
MODEL_PATH=/apdcephfs_zwfy6/share_304376610/pighzliu_code/models/OLMo-2-1124-7B \
TEACHER_PATH=/apdcephfs_zwfy6/share_304376610/pighzliu_code/models/OLMo-2-1124-7B \
DATA_PATH=/dev/shm/dolmino_now15b.npy \
  bash scripts/_run_olmo2_keep14_distill_heal.sh
echo "[$(date '+%F %T')] ============ ORCH DONE (distill relaunched) ============"
