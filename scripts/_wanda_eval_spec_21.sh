#!/usr/bin/env bash
# CAST SPEC.md §8 Wanda 2:4 baseline: prune + PPL + 7-task zeroshot on ONE harness.
# Matches the invocation pattern of _cast_eval_spec_21.sh (PPL) and
# _cast_zeroshot_spec_21.sh (zeroshot) so the three arms {dense, cast_7500, wanda}
# are directly comparable. Node: .21 (8x L20A 183 GB, wzc1 disk).
set -u
ROOT=/apdcephfs_wzc1/share_304376610/pighzliu_code
PY=/opt/conda/envs/torch-base/bin/python
LM_EVAL=/opt/conda/envs/torch-base/bin/lm_eval
HARNESS_PPL=$ROOT/baselines/eval_hf_sparse_model.py
WANDA_TOOL=$ROOT/Mixture-of-Memory/baselines/cast_repro/tools/wanda_prune.py
VERIFY_TOOL=$ROOT/Mixture-of-Memory/baselines/cast_repro/tools/verify_2of4_hf_export.py
AGG_TOOL=$ROOT/Mixture-of-Memory/baselines/cast_repro/tools/aggregate_zeroshot_metrics.py

DENSE=$ROOT/models/Llama--Llama2-7b
TRAIN_BIN=$ROOT/data/c4_llama/train.bin
WIKI=$ROOT/data/wikitext/wikitext-2-raw-v1/wiki.test.raw
OUT_MODEL=$ROOT/outputs/wanda_llama2_7b/hf_pruned
OUT_EVAL=$ROOT/outputs/cast_eval_spec/wanda
NSAMPLES=${NSAMPLES:-128}
CAL_SEQLEN=${CAL_SEQLEN:-2048}
EVAL_SEQLEN=${EVAL_SEQLEN:-4096}
SEED=${SEED:-0}
TASKS=hellaswag,race,piqa,winogrande,arc_easy,arc_challenge,openbookqa

cd "$ROOT" || exit 1
mkdir -p "$(dirname "$OUT_MODEL")" "$OUT_EVAL/lm_eval_out"

# ---- STAGE 1: prune (single GPU is enough, ~30 min for 7B x 128 samples x 2048 tokens) ----
if [ ! -f "$OUT_MODEL/model.safetensors" ] && [ ! -f "$OUT_MODEL/model.safetensors.index.json" ]; then
    echo "=== [$(date +%H:%M:%S)] STAGE 1: Wanda prune ==="
    CUDA_VISIBLE_DEVICES=0 "$PY" "$WANDA_TOOL" \
        --model "$DENSE" \
        --train_bin "$TRAIN_BIN" \
        --dtype_bin uint16 \
        --output "$OUT_MODEL" \
        --nsamples "$NSAMPLES" \
        --seqlen "$CAL_SEQLEN" \
        --seed "$SEED" \
        --storage_dtype bfloat16 \
        --device cuda:0 2>&1 | tee "$OUT_EVAL/wanda_prune.log"
    rc=${PIPESTATUS[0]}
    echo "=== [$(date +%H:%M:%S)] STAGE 1 DONE rc=$rc ==="
    if [ "$rc" -ne 0 ]; then
        echo "!!! Wanda pruning failed, aborting."
        exit "$rc"
    fi
else
    echo "=== [$(date +%H:%M:%S)] STAGE 1 SKIPPED (pruned model already exists at $OUT_MODEL) ==="
fi

# ---- STAGE 2: verify HF export is exact 2:4 (must be 224 tensors, bad_tiles=0) ----
echo "=== [$(date +%H:%M:%S)] STAGE 2: verify 2:4 on HF export ==="
CUDA_VISIBLE_DEVICES=0 "$PY" "$VERIFY_TOOL" \
    --model "$OUT_MODEL" --sample-layers 12 --seed 0 2>&1 | tee "$OUT_EVAL/verify_2of4.log"
verify_rc=${PIPESTATUS[0]}
echo "=== [$(date +%H:%M:%S)] STAGE 2 DONE rc=$verify_rc ==="
if [ "$verify_rc" -ne 0 ]; then
    echo "!!! Wanda HF export failed 2:4 verification, aborting."
    exit "$verify_rc"
fi

# ---- STAGE 3: PPL on WikiText-2 @ 4096 (same harness as dense/cast) ----
echo "=== [$(date +%H:%M:%S)] STAGE 3: WikiText-2 PPL ==="
CUDA_VISIBLE_DEVICES=0 "$PY" "$HARNESS_PPL" \
    --model "$OUT_MODEL" \
    --output_dir "$OUT_EVAL" \
    --wiki_text "$WIKI" \
    --seqlen "$EVAL_SEQLEN" \
    --wiki_tokens 100000000 \
    --device cuda:0 2>&1 | tee "$OUT_EVAL/lm_eval_ppl.log"
echo "=== [$(date +%H:%M:%S)] STAGE 3 DONE rc=${PIPESTATUS[0]} ==="

# ---- STAGE 4: zero-shot 7-task suite (lm-eval 0.4.8, parallelize=True over 8 GPUs) ----
echo "=== [$(date +%H:%M:%S)] STAGE 4: 7-task zero-shot ==="
export HF_HUB_OFFLINE=0
export http_proxy=http://hy-proxy.woa.com:3128
export https_proxy=http://hy-proxy.woa.com:3128
export no_proxy='mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,localhost,127.0.0.1,.oa.com,.woa.com,.local'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_ALLOW_CODE_EVAL=1 HF_DATASETS_TRUST_REMOTE_CODE=1 \
    "$LM_EVAL" \
        --model hf \
        --model_args "pretrained=$OUT_MODEL,dtype=bfloat16,parallelize=True,trust_remote_code=True,add_bos_token=False" \
        --tasks $TASKS \
        --batch_size auto \
        --num_fewshot 0 \
        --output_path "$OUT_EVAL/lm_eval_out" \
        --seed 0 \
        --trust_remote_code \
        --log_samples 2>&1 | tee "$OUT_EVAL/lm_eval.log"
echo "=== [$(date +%H:%M:%S)] STAGE 4 DONE rc=${PIPESTATUS[0]} ==="

# ---- STAGE 5: aggregate zeroshot_metrics.json ----
echo "=== [$(date +%H:%M:%S)] STAGE 5: aggregate zeroshot metrics ==="
"$PY" "$AGG_TOOL" \
    --lm-eval-out "$OUT_EVAL/lm_eval_out" \
    --output "$OUT_EVAL/zeroshot_metrics.json" \
    --model "wanda_llama2_7b_2of4"
echo "=== [$(date +%H:%M:%S)] STAGE 5 DONE ==="

echo "=== [$(date +%H:%M:%S)] ALL STAGES COMPLETE ==="
echo "--- PPL:"
cat "$OUT_EVAL/ppl_metrics.json" 2>/dev/null | grep -E 'wikitext2_ppl|exact_2of4_tile_ratio|linear_zero_ratio'
echo "--- Zeroshot summary:"
grep -E 'zeroshot_avg_primary|zeroshot_avg_acc' "$OUT_EVAL/zeroshot_metrics.json" 2>/dev/null
