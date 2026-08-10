#!/usr/bin/env bash
# SPEC.md S7 zero-shot 7-task suite for the finished CAST run vs dense reference.
# Tasks: hellaswag, race, piqa, winogrande, arc_easy, arc_challenge, openbookqa
# Same harness (lm-eval 0.4.8), same box, base-model settings (no chat template,
# no BOS override). Writes zeroshot_metrics.json alongside the finished
# ppl_metrics.json (which is NOT touched).
#
# The two models are run on disjoint halves of the box (GPUs 0-3 vs GPUs 4-7) in
# parallel, using `accelerate launch` for tensor parallelism within each half.
# 4 x L20A per model is more than enough for a 7B model (data-parallel with
# batch_size=auto by lm-eval).
set -u
ROOT=/apdcephfs_wzc1/share_304376610/pighzliu_code
PY=/opt/conda/envs/torch-base/bin/python
LM_EVAL=/opt/conda/envs/torch-base/bin/lm_eval
export HF_HUB_OFFLINE=0
export http_proxy=http://hy-proxy.woa.com:3128
export https_proxy=http://hy-proxy.woa.com:3128
export no_proxy='mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,localhost,127.0.0.1,.oa.com,.woa.com,.local'
TASKS=hellaswag,race,piqa,winogrande,arc_easy,arc_challenge,openbookqa

cd "$ROOT" || exit 1

run_one () {
  # name model_path gpus
  local name=$1 model=$2 gpus=$3
  local out=$ROOT/outputs/cast_eval_spec/$name
  mkdir -p "$out/lm_eval_out"
  echo "=== [$(date +%H:%M:%S)] EVAL $name on GPUs $gpus ==="
  CUDA_VISIBLE_DEVICES="$gpus" HF_ALLOW_CODE_EVAL=1 HF_DATASETS_TRUST_REMOTE_CODE=1 \
  "$LM_EVAL" \
    --model hf \
    --model_args "pretrained=$model,dtype=bfloat16,parallelize=True,trust_remote_code=True,add_bos_token=False" \
    --tasks $TASKS \
    --batch_size auto \
    --num_fewshot 0 \
    --output_path "$out/lm_eval_out" \
    --seed 0 \
    --trust_remote_code \
    --log_samples 2>&1 | tee "$out/lm_eval.log"
  echo "=== [$(date +%H:%M:%S)] DONE $name rc=${PIPESTATUS[0]} ==="
}

# CAST checkpoint uses the same tokenizer as dense (verified in cast_export_meta.json)
run_one dense_ref  "$ROOT/models/Llama--Llama2-7b"           0,1,2,3 &
run_one cast_7500  "$ROOT/outputs/cast_repro_zero2/hf_final" 4,5,6,7 &
wait
echo "=== ALL ZERO-SHOT EVALS COMPLETE $(date +%H:%M:%S) ==="
