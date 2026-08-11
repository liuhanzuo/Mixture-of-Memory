#!/usr/bin/env bash
# Union-9 four-arm zero-shot table + AST-arm PPL, all on ONE harness. Node .21.
#
# WHY THIS SCRIPT EXISTS
#   The published 7-task table used CAST-7 = {hellaswag, race, piqa, winogrande,
#   arc_easy, arc_challenge, openbookqa}. AST's paper reports AST-7 = {boolq, rte,
#   hellaswag, winogrande, arc_easy, arc_challenge, openbookqa}. Those two sets
#   intersect in only 5 tasks, so a CAST-7 average is NOT comparable to an AST-7
#   average. Union-9 (= 9 tasks) lets both subset means be sliced from one run.
#
# WHAT IT DOES
#   Runs all 9 tasks for all 4 arms {dense, CAST@7500, Wanda, AST-official}, even
#   though only BoolQ+RTE are strictly missing for the first three. Re-running the
#   7 known tasks is deliberate: it is a free harness-integrity check (they must
#   reproduce the on-disk numbers) and it removes any doubt that the old and new
#   halves of a row were measured under different auto-batch-size / env state.
#
# WHAT IT DOES NOT DO
#   It never writes to the three existing zeroshot_metrics.json / ppl_metrics.json
#   files -- those are already cited. New results land in a fresh tree
#   (outputs/cast_eval_spec_union9/) plus the two brief-mandated filenames:
#     outputs/cast_eval_spec/{dense_ref,cast_7500,wanda}/zeroshot_boolq_rte.json
#     outputs/cast_eval_spec/ast_official/{ppl_metrics.json,zeroshot_metrics.json}
#
# HARNESS CONFIG -- copied verbatim from _cast_zeroshot_spec_21.sh so all four arms
# are byte-identical in configuration except for `pretrained`:
#   lm-eval 0.4.8, --model hf, dtype=bfloat16, parallelize=True,
#   trust_remote_code=True, add_bos_token=False, --batch_size auto,
#   --num_fewshot 0, --seed 0, --log_samples, no chat template.
set -u
ROOT=/apdcephfs_wzc1/share_304376610/pighzliu_code
PY=/opt/conda/envs/torch-base/bin/python
LM_EVAL=/opt/conda/envs/torch-base/bin/lm_eval
HARNESS_PPL=$ROOT/baselines/eval_hf_sparse_model.py
VERIFY_TOOL=$ROOT/Mixture-of-Memory/baselines/cast_repro/tools/verify_2of4_hf_export.py
AGG9=$ROOT/Mixture-of-Memory/baselines/cast_repro/tools/aggregate_zeroshot_union9.py

WIKI=$ROOT/data/wikitext/wikitext-2-raw-v1/wiki.test.raw
EVAL_SEQLEN=${EVAL_SEQLEN:-4096}

M_DENSE=$ROOT/models/Llama--Llama2-7b
M_CAST=$ROOT/outputs/cast_repro_zero2/hf_final
M_WANDA=$ROOT/outputs/wanda_llama2_7b/hf_pruned
M_AST=$ROOT/models/AST-official-LLaMA2-7B-2of4

SPEC_DIR=$ROOT/outputs/cast_eval_spec
U9=$ROOT/outputs/cast_eval_spec_union9
AST_OUT=$SPEC_DIR/ast_official

# Union-9 = CAST-7 U AST-7.
TASKS=boolq,rte,hellaswag,race,piqa,winogrande,arc_easy,arc_challenge,openbookqa

# boolq/rte are served as parquet redirects, so the hub must be reachable.
export HF_HUB_OFFLINE=0
export http_proxy=http://hy-proxy.woa.com:3128
export https_proxy=http://hy-proxy.woa.com:3128
export no_proxy='mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,localhost,127.0.0.1,.oa.com,.woa.com,.local'

cd "$ROOT" || exit 1
mkdir -p "$AST_OUT" "$U9"/{dense_ref,cast_7500,wanda,ast_official}

# ---------------------------------------------------------------- STAGE 1
# Verify the AST checkpoint is exact 2:4 BEFORE any inference touches it.
echo "=== [$(date +%H:%M:%S)] STAGE 1: verify 2:4 on AST ckpt (PRE-inference) ==="
CUDA_VISIBLE_DEVICES=0 "$PY" "$VERIFY_TOOL" \
    --model "$M_AST" --sample-layers 12 --seed 0 2>&1 | tee "$AST_OUT/verify_2of4_pre.log"
pre_rc=${PIPESTATUS[0]}
echo "=== [$(date +%H:%M:%S)] STAGE 1 DONE rc=$pre_rc ==="
if [ "$pre_rc" -ne 0 ]; then
    echo "!!! AST ckpt failed 2:4 verification PRE-inference. Aborting: the arm would be invalid."
    exit "$pre_rc"
fi

# ---------------------------------------------------------------- STAGE 2
# WikiText-2 PPL @4096 for the AST arm, same harness/protocol as the other three.
echo "=== [$(date +%H:%M:%S)] STAGE 2: AST WikiText-2 PPL @${EVAL_SEQLEN} ==="
CUDA_VISIBLE_DEVICES=0 "$PY" "$HARNESS_PPL" \
    --model "$M_AST" \
    --output_dir "$AST_OUT" \
    --wiki_text "$WIKI" \
    --seqlen "$EVAL_SEQLEN" \
    --wiki_tokens 100000000 \
    --device cuda:0 2>&1 | tee "$AST_OUT/lm_eval_ppl.log"
echo "=== [$(date +%H:%M:%S)] STAGE 2 DONE rc=${PIPESTATUS[0]} ==="

# ---------------------------------------------------------------- STAGE 3
# Zero-shot Union-9, two arms at a time on disjoint halves of the box (4 GPUs
# each) -- the same topology that produced dense_ref/cast_7500.
run_one () {  # name model gpus
  local name=$1 model=$2 gpus=$3
  local out=$U9/$name
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

echo "=== [$(date +%H:%M:%S)] STAGE 3 WAVE 1: dense_ref + cast_7500 ==="
run_one dense_ref    "$M_DENSE" 0,1,2,3 &
run_one cast_7500    "$M_CAST"  4,5,6,7 &
wait
echo "=== [$(date +%H:%M:%S)] STAGE 3 WAVE 2: wanda + ast_official ==="
run_one wanda        "$M_WANDA" 0,1,2,3 &
run_one ast_official "$M_AST"   4,5,6,7 &
wait
echo "=== [$(date +%H:%M:%S)] STAGE 3 COMPLETE ==="

# ---------------------------------------------------------------- STAGE 4
# Aggregate. Full union-9 for every arm into the fresh tree; then the two
# brief-mandated destinations.
echo "=== [$(date +%H:%M:%S)] STAGE 4: aggregate ==="
declare -A MODEL_ID=(
  [dense_ref]="llama2_7b_dense"
  [cast_7500]="cast_repro_zero2_step7500_2of4"
  [wanda]="wanda_llama2_7b_2of4"
  [ast_official]="AST-official-LLaMA2-7B-2of4"
)
for arm in dense_ref cast_7500 wanda ast_official; do
  "$PY" "$AGG9" \
      --lm-eval-out "$U9/$arm/lm_eval_out" \
      --output "$U9/$arm/zeroshot_union9.json" \
      --model "${MODEL_ID[$arm]}" || exit 1
done

# BoolQ+RTE-only files for the three pre-existing arms (their zeroshot_metrics.json
# is left untouched).
for arm in dense_ref cast_7500 wanda; do
  "$PY" "$AGG9" \
      --lm-eval-out "$U9/$arm/lm_eval_out" \
      --output "$SPEC_DIR/$arm/zeroshot_boolq_rte.json" \
      --model "${MODEL_ID[$arm]}" \
      --tasks boolq,rte || exit 1
done

# AST arm is new, so it gets the canonical filename (all 9 tasks).
"$PY" "$AGG9" \
    --lm-eval-out "$U9/ast_official/lm_eval_out" \
    --output "$AST_OUT/zeroshot_metrics.json" \
    --model "${MODEL_ID[ast_official]}" || exit 1
echo "=== [$(date +%H:%M:%S)] STAGE 4 DONE ==="

# ---------------------------------------------------------------- STAGE 5
# Verify AST 2:4 again AFTER inference (same criterion as the CAST/Wanda arms).
echo "=== [$(date +%H:%M:%S)] STAGE 5: verify 2:4 on AST ckpt (POST-inference) ==="
CUDA_VISIBLE_DEVICES=0 "$PY" "$VERIFY_TOOL" \
    --model "$M_AST" --sample-layers 12 --seed 0 2>&1 | tee "$AST_OUT/verify_2of4_post.log"
post_rc=${PIPESTATUS[0]}
echo "=== [$(date +%H:%M:%S)] STAGE 5 DONE rc=$post_rc ==="

# ---------------------------------------------------------------- STAGE 6
# Harness-integrity check: the 7 re-run tasks must reproduce the on-disk numbers
# for the three pre-existing arms. Any drift here invalidates the union table.
echo "=== [$(date +%H:%M:%S)] STAGE 6: integrity check vs existing 7-task JSONs ==="
"$PY" - <<'PYEOF' 2>&1 | tee "$U9/integrity_check.log"
import json, os
SPEC = "/apdcephfs_wzc1/share_304376610/pighzliu_code/outputs/cast_eval_spec"
U9   = "/apdcephfs_wzc1/share_304376610/pighzliu_code/outputs/cast_eval_spec_union9"
SEVEN = ("hellaswag","race","piqa","winogrande","arc_easy","arc_challenge","openbookqa")
worst = 0.0
for arm in ("dense_ref","cast_7500","wanda"):
    old = json.load(open(f"{SPEC}/{arm}/zeroshot_metrics.json"))["per_task"]
    new = json.load(open(f"{U9}/{arm}/zeroshot_union9.json"))["per_task"]
    print(f"--- {arm}")
    for t in SEVEN:
        for m in ("acc","acc_norm"):
            o, n = old[t].get(m), new[t].get(m)
            if o is None or n is None:
                print(f"    {t:14s} {m:8s} old={o} new={n}")
                continue
            d = abs(n-o)*100
            worst = max(worst, d)
            tag = "OK " if d < 1e-9 else ("~  " if d < 0.05 else "DRIFT")
            print(f"    {t:14s} {m:8s} old={o*100:.4f} new={n*100:.4f} delta_pp={d:+.6f} {tag}")
print(f"=== worst |delta| over 3 arms x 7 tasks x 2 metrics = {worst:.6f} pp")
print("=== interpretation: 0.000000 => bit-identical re-run, harness stable, "
      "union-9 row is internally consistent. Nonzero => report the drift.")
PYEOF
echo "=== [$(date +%H:%M:%S)] STAGE 6 DONE ==="

echo "=== [$(date +%H:%M:%S)] ALL STAGES COMPLETE (pre_rc=$pre_rc post_rc=$post_rc) ==="
echo "--- AST PPL:"
grep -E 'wikitext2_ppl|wikitext2_tokens|exact_2of4_tile_ratio|linear_zero_ratio' \
    "$AST_OUT/ppl_metrics.json" 2>/dev/null
echo "--- Union-9 slices per arm:"
for arm in dense_ref cast_7500 wanda ast_official; do
  echo -n "  $arm: "
  "$PY" -c "
import json,sys
b=json.load(open('$U9/$arm/zeroshot_union9.json'))
for k in ('union9','cast7','ast7'):
    s=b[k]
    print(f\"{k}=prim {s['mean_primary']*100:.4f}/acc {s['mean_plain_acc']*100:.4f}\", end='  ')
print()
"
done
