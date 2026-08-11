#!/usr/bin/env bash
# SparseForge 5B headline checkpoint, measured on the SAME harness as the other
# three arms of the Union-9 table. Node .21 (8x L20A, wzc1).
#
# WHY: the SparseForge main table comes from
#   SparseForge_Data/tables/cast9_dense_ast_current_harness.csv   (OLD harness)
# while our CAST reproduction / dense / Wanda / AST-official arms live in
#   outputs/cast_eval_spec_union9/                                (lm-eval 0.4.8, git b86c479)
# AST-official is the only arm present in both, and its plain-acc AST-7 differs
# by -0.3460 pp between the two (57.9436 old vs 57.5976 ours), single-task max
# 1.59 pp on boolq. That offset is 0.65x SparseForge's own +0.53 pp margin, so
# NO cross-harness comparison to SparseForge is admissible. This script closes
# that by re-measuring the checkpoint here.
#
# THREE VARIANTS, because the checkpoint is not a plain 2:4 model (see
# tools/export_sparseforge_to_hf.py's module docstring for the full argument):
#   hard_drop  exact 2:4, SLoRB branch removed.  <-- the ONLY variant comparable
#              to dense / CAST-repro / AST-official / Wanda, and the only one that
#              can pass verify_2of4_hf_export.py.
#   soft_fold  continuous mask + SLoRB folded in. This is the model that produced
#              the checkpoint's own CAST-7 anchor 57.2672 (main_llama.py:2175-2215
#              rebuilds LlamaSparse from this state dict and evaluates it without
#              hardening). Included as a FAITHFULNESS CONTROL: if it lands on
#              57.27 our reading of the trainer is right and the hard_drop gap is
#              attributable, not a bug.
#   hard_fold  2:4 support + SLoRB folded in = what SparseForge would deploy.
#              NOT 2:4 (the fold writes into pruned positions), so it may never
#              be placed in a 2:4 column; reported to show the branch's value.
#
# HARNESS: byte-identical to _union9_eval_spec_21.sh / _cast_zeroshot_spec_21.sh
# except `pretrained`. lm-eval 0.4.8, --model hf, dtype=bfloat16,
# parallelize=True, trust_remote_code=True, add_bos_token=False,
# --batch_size auto, --num_fewshot 0, --seed 0, --log_samples, no chat template.
#
# PPL: seqlen 2048, because SparseForge's entire PPL column is 2048 (SPEC.md:213
# fixed this normalisation; at 4096 every arm reads ~6-7 % lower).
set -u
ROOT=/apdcephfs_wzc1/share_304376610/pighzliu_code
PY=/opt/conda/envs/torch-base/bin/python
LM_EVAL=/opt/conda/envs/torch-base/bin/lm_eval
TOOLS=$ROOT/Mixture-of-Memory/baselines/cast_repro/tools
EXPORT=$TOOLS/export_sparseforge_to_hf.py
VERIFY=$TOOLS/verify_2of4_hf_export.py
AGG9=$TOOLS/aggregate_zeroshot_union9.py
HARNESS_PPL=$ROOT/baselines/eval_hf_sparse_model.py

CKPT=$ROOT/out_llama/models_Llama--Llama2-7b_mask-unstructured_s0.5_m-hessian_obd_20260413_201320/model_best_lm_eval.pt
WIKI=$ROOT/data/wikitext/wikitext-2-raw-v1/wiki.test.raw
EVAL_SEQLEN=2048

OUT=$ROOT/outputs/cast_eval_spec/sparseforge_5b
EXPDIR=$ROOT/outputs/sparseforge_5b_hf
TASKS=boolq,rte,hellaswag,race,piqa,winogrande,arc_easy,arc_challenge,openbookqa

export HF_HUB_OFFLINE=0
export http_proxy=http://hy-proxy.woa.com:3128
export https_proxy=http://hy-proxy.woa.com:3128
export no_proxy='mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,localhost,127.0.0.1,.oa.com,.woa.com,.local'

cd "$ROOT" || exit 1
mkdir -p "$OUT" "$EXPDIR"

# ------------------------------------------------------------------ STAGE 1
# Export the three variants (CPU only).
echo "=== [$(date +%H:%M:%S)] STAGE 1: export three variants ==="
for spec in "hard drop" "soft fold" "hard fold"; do
  set -- $spec
  m=$1; s=$2
  name="${m}_${s}"
  if [ -f "$EXPDIR/$name/sparseforge_export_meta.json" ]; then
    echo "--- $name already exported, skipping"
    continue
  fi
  echo "--- [$(date +%H:%M:%S)] export $name"
  "$PY" "$EXPORT" --ckpt "$CKPT" --output "$EXPDIR/$name" \
      --mask "$m" --slorb "$s" 2>&1 | tee "$OUT/export_$name.log"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    echo "!!! export $name FAILED rc=$rc -- aborting, a wrong export is worse than no number"
    exit "$rc"
  fi
done
echo "=== [$(date +%H:%M:%S)] STAGE 1 DONE ==="

# ------------------------------------------------------------------ STAGE 2
# HARD GATE: hard_drop must be exact 2:4 before any score is computed.
echo "=== [$(date +%H:%M:%S)] STAGE 2: verify 2:4 on hard_drop (PRE-inference) ==="
CUDA_VISIBLE_DEVICES=0 "$PY" "$VERIFY" \
    --model "$EXPDIR/hard_drop" --sample-layers 12 --seed 0 2>&1 \
    | tee "$OUT/verify_2of4_hard_drop_pre.log"
pre_rc=${PIPESTATUS[0]}
echo "=== [$(date +%H:%M:%S)] STAGE 2 DONE rc=$pre_rc ==="
if [ "$pre_rc" -ne 0 ]; then
    echo "!!! hard_drop failed 2:4 verification. STOPPING per brief: do not score an unverified export."
    exit "$pre_rc"
fi

# Record what the other two variants look like sparsity-wise (expected: NOT 2:4).
for name in soft_fold hard_fold; do
  echo "=== [$(date +%H:%M:%S)] STAGE 2b: sparsity report for $name (expected NOT 2:4) ==="
  CUDA_VISIBLE_DEVICES=0 "$PY" "$VERIFY" \
      --model "$EXPDIR/$name" --sample-layers 4 --seed 0 2>&1 \
      | tee "$OUT/verify_2of4_${name}.log" || true
done

# ------------------------------------------------------------------ STAGE 3
# WikiText-2 PPL @2048 for all three variants, same harness as the other arms.
echo "=== [$(date +%H:%M:%S)] STAGE 3: WikiText-2 PPL @${EVAL_SEQLEN} ==="
for name in hard_drop soft_fold hard_fold; do
  mkdir -p "$OUT/$name"
  echo "--- [$(date +%H:%M:%S)] ppl $name"
  CUDA_VISIBLE_DEVICES=0 "$PY" "$HARNESS_PPL" \
      --model "$EXPDIR/$name" \
      --output_dir "$OUT/$name" \
      --wiki_text "$WIKI" \
      --seqlen "$EVAL_SEQLEN" \
      --wiki_tokens 100000000 \
      --device cuda:0 2>&1 | tee "$OUT/$name/ppl${EVAL_SEQLEN}.log"
done
echo "=== [$(date +%H:%M:%S)] STAGE 3 DONE ==="

# ------------------------------------------------------------------ STAGE 4
# Union-9 zero-shot. Two arms per wave on disjoint halves of the box -- the same
# topology that produced dense_ref / cast_7500 / wanda / ast_official.
run_one () {  # name model gpus
  local name=$1 model=$2 gpus=$3
  local o=$OUT/$name
  mkdir -p "$o/lm_eval_out"
  echo "=== [$(date +%H:%M:%S)] EVAL $name on GPUs $gpus ==="
  CUDA_VISIBLE_DEVICES="$gpus" HF_ALLOW_CODE_EVAL=1 HF_DATASETS_TRUST_REMOTE_CODE=1 \
  "$LM_EVAL" \
    --model hf \
    --model_args "pretrained=$model,dtype=bfloat16,parallelize=True,trust_remote_code=True,add_bos_token=False" \
    --tasks $TASKS \
    --batch_size auto \
    --num_fewshot 0 \
    --output_path "$o/lm_eval_out" \
    --seed 0 \
    --trust_remote_code \
    --log_samples 2>&1 | tee "$o/lm_eval.log"
  echo "=== [$(date +%H:%M:%S)] DONE $name rc=${PIPESTATUS[0]} ==="
}

echo "=== [$(date +%H:%M:%S)] STAGE 4 WAVE 1: hard_drop + soft_fold ==="
run_one hard_drop "$EXPDIR/hard_drop" 0,1,2,3 &
run_one soft_fold "$EXPDIR/soft_fold" 4,5,6,7 &
wait
echo "=== [$(date +%H:%M:%S)] STAGE 4 WAVE 2: hard_fold ==="
run_one hard_fold "$EXPDIR/hard_fold" 0,1,2,3
echo "=== [$(date +%H:%M:%S)] STAGE 4 COMPLETE ==="

# ------------------------------------------------------------------ STAGE 5
# Aggregate. aggregate_zeroshot_union9.py hard-fails if any of the 9 tasks is
# absent, so a partial row can never be silently averaged.
echo "=== [$(date +%H:%M:%S)] STAGE 5: aggregate (asserts all 9 tasks present) ==="
for name in hard_drop soft_fold hard_fold; do
  "$PY" "$AGG9" \
      --lm-eval-out "$OUT/$name/lm_eval_out" \
      --output "$OUT/$name/zeroshot_union9.json" \
      --model "sparseforge_5b_iter17900_$name" || exit 1
done
echo "=== [$(date +%H:%M:%S)] STAGE 5 DONE ==="

# ------------------------------------------------------------------ STAGE 6
# POST-inference 2:4 re-verify on hard_drop (same criterion as CAST/Wanda/AST).
echo "=== [$(date +%H:%M:%S)] STAGE 6: verify 2:4 on hard_drop (POST-inference) ==="
CUDA_VISIBLE_DEVICES=0 "$PY" "$VERIFY" \
    --model "$EXPDIR/hard_drop" --sample-layers 12 --seed 0 2>&1 \
    | tee "$OUT/verify_2of4_hard_drop_post.log"
post_rc=${PIPESTATUS[0]}
echo "=== [$(date +%H:%M:%S)] STAGE 6 DONE rc=$post_rc ==="

echo "=== [$(date +%H:%M:%S)] ALL STAGES COMPLETE (pre_rc=$pre_rc post_rc=$post_rc) ==="
for name in hard_drop soft_fold hard_fold; do
  echo "--- $name"
  grep -E '"wikitext2_ppl"|"exact_2of4_tile_ratio"|"linear_zero_ratio"' \
      "$OUT/$name/ppl_metrics.json" 2>/dev/null
  "$PY" -c "
import json
b=json.load(open('$OUT/$name/zeroshot_union9.json'))
for k in ('union9','cast7','ast7'):
    s=b[k]
    print(f\"    {k}: primary {s['mean_primary']*100:.4f}  plain_acc {s['mean_plain_acc']*100:.4f}\")
r=b['per_task']['rte']
print(f\"    RTE acc={r['acc']:.10f} n={r['n_samples']} k={r['acc']*r['n_samples']:.4f}\")
" 2>/dev/null
done
