#!/usr/bin/env bash
# SparseForge's DOLMINO-ONLY resume link (link 2 of 3), scored on the SAME harness
# as every other arm of the Union-9 table. Node .21 (8x L20A, wzc1).
#
# WHY THIS RUN EXISTS
# -------------------
# commit e44c742 established that the published SparseForge 5B arm
#   out_llama/..._20260413_201320   (link 3)
# trained on `data/qa_format_sft_llama` -- 8 multiple-choice-QA benchmark TRAIN
# splits, 129.75M tokens, repeat=3, traversed ~144.7x -- which contains
# race_middle + race_high while `race` IS a CAST-7 eval task. So the published
# SparseForge-vs-CAST/AST comparison is confounded by training data, not just by
# the SLoRB branch.
#
# But that arm is the LAST of a three-link resume chain, and the link BEFORE it
# trained only on dolmino-mix-1124-raw -- the same general-corpus family CAST
# trained on:
#
#   link 1  ..._20260331_150310   dolmino-mix-1124-raw   resume=False
#   link 2  ..._20260401_124938   dolmino-mix-1124-raw   resume_dir -> link 1
#   link 3  ..._20260413_201320   qa_format_sft_llama    resume_dir -> link 2
#
# Link 2's weights therefore NEVER saw MC-QA data. Scoring them answers the
# question "does SparseForge's algorithmic claim survive on general data?"
# WITHOUT spending ~4 GPU-days on a fresh data-matched retrain.
#
# WHICH CHECKPOINT
# ----------------
# Link 2 has TWO checkpoints and they are different models:
#   model_best_lm_eval.pt   iter_num=17600, finalization_done=True,
#                           best_lm_eval_mean=59.2153  <-- SCORED HERE
#   model.pt                iter_num=20000, carries optimizer+slorb state.
#                           This is what link 3 actually resumed from, because
#                           main_llama.py's resume_candidates tuple prefers
#                           `model.pt` over `model_best_lm_eval.pt`.
# We score model_best_lm_eval.pt because it is the one with a published-style
# anchor (best_lm_eval.json) and because it is the direct analogue of what was
# scored for link 3 (also model_best_lm_eval.pt, iter 17900).
#
# VARIANTS -- exactly the two that the link-3 row already has, so the columns
# are comparable cell-for-cell:
#   hard_fold  hard 2:4 mask + SLoRB folded into W. Numerically what SparseForge
#              DEPLOYS (per outputs/paper_v2/.../eval.log: "Set hardening_x=0 ...
#              Keeping sparse_forward mode (SLoRB enabled)"). The folded weight
#              is DENSE, so it must NEVER go in a 2:4 column.
#              <-- apples-to-apples with the existing link-3 hard_fold row.
#   hard_drop  exact 2:4, SLoRB branch AMPUTATED. Reported strictly as
#              amputation damage: the weights were TRAINED with this branch, so
#              this is NOT "SparseForge's 2:4 result".
#
# HARNESS: byte-identical to _sparseforge_same_harness_21.sh /
# _cast_zeroshot_spec_21.sh -- lm-eval 0.4.8, --model hf, dtype=bfloat16,
# parallelize=True, trust_remote_code=True, add_bos_token=False,
# --batch_size auto, --num_fewshot 0, --seed 0, --log_samples, no chat template.
# Only `pretrained` differs.
#
# PPL: measured at BOTH 2048 and 4096 and both are labelled, because commit
# 501dafb retracted a conclusion built on a PPL column that silently mixed the
# two. The headline comparison is @4096.
set -u
ROOT=/apdcephfs_wzc1/share_304376610/pighzliu_code
PY=/opt/conda/envs/torch-base/bin/python
LM_EVAL=/opt/conda/envs/torch-base/bin/lm_eval
TOOLS=$ROOT/Mixture-of-Memory/baselines/cast_repro/tools
EXPORT=$TOOLS/export_sparseforge_to_hf.py
VERIFY=$TOOLS/verify_2of4_hf_export.py
AGG9=$TOOLS/aggregate_zeroshot_union9.py
HARNESS_PPL=$ROOT/baselines/eval_hf_sparse_model.py

CKPT=$ROOT/out_llama/models_Llama--Llama2-7b_mask-unstructured_s0.5_m-hessian_obd_20260401_124938/model_best_lm_eval.pt
WIKI=$ROOT/data/wikitext/wikitext-2-raw-v1/wiki.test.raw

OUT=$ROOT/outputs/cast_eval_spec/sparseforge_dolmino_link2
EXPDIR=$ROOT/outputs/sparseforge_dolmino_link2_hf
TASKS=boolq,rte,hellaswag,race,piqa,winogrande,arc_easy,arc_challenge,openbookqa

export HF_HUB_OFFLINE=0
export http_proxy=http://hy-proxy.woa.com:3128
export https_proxy=http://hy-proxy.woa.com:3128
export no_proxy='mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,localhost,127.0.0.1,.oa.com,.woa.com,.local'

cd "$ROOT" || exit 1
mkdir -p "$OUT" "$EXPDIR"

echo "############ SparseForge dolmino-only link (link 2) same-harness eval"
echo "############ ckpt = $CKPT"
echo "############ start $(date -Is)"

# ------------------------------------------------------------------ STAGE 1
echo "=== [$(date +%H:%M:%S)] STAGE 1: export hard_fold + hard_drop ==="
for spec in "hard fold" "hard drop"; do
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
    echo "!!! export $name FAILED rc=$rc -- aborting; a wrong export is worse than no number"
    exit "$rc"
  fi
done
echo "=== [$(date +%H:%M:%S)] STAGE 1 DONE ==="

# ------------------------------------------------------------------ STAGE 2
# HARD GATE: hard_drop must pass the 2:4 deployability gate (tiles_gt2 == 0)
# before any score is computed. hard_fold is expected to FAIL by design.
echo "=== [$(date +%H:%M:%S)] STAGE 2: verify 2:4 hard_drop (PRE-inference) ==="
CUDA_VISIBLE_DEVICES=0 "$PY" "$VERIFY" \
    --model "$EXPDIR/hard_drop" --sample-layers 12 --seed 0 2>&1 \
    | tee "$OUT/verify_2of4_hard_drop_pre.log"
pre_rc=${PIPESTATUS[0]}
echo "=== [$(date +%H:%M:%S)] STAGE 2 DONE rc=$pre_rc ==="
if [ "$pre_rc" -ne 0 ]; then
    echo "!!! hard_drop failed the 2:4 gate. STOPPING -- do not score an unverified export."
    exit "$pre_rc"
fi

echo "=== [$(date +%H:%M:%S)] STAGE 2b: sparsity report for hard_fold (expected NOT 2:4) ==="
CUDA_VISIBLE_DEVICES=0 "$PY" "$VERIFY" \
    --model "$EXPDIR/hard_fold" --sample-layers 4 --seed 0 2>&1 \
    | tee "$OUT/verify_2of4_hard_fold.log" || true

# ------------------------------------------------------------------ STAGE 3
# WikiText-2 PPL at BOTH seqlens, each explicitly labelled.
for SEQ in 4096 2048; do
  echo "=== [$(date +%H:%M:%S)] STAGE 3: WikiText-2 PPL @${SEQ} ==="
  for name in hard_fold hard_drop; do
    o=$OUT/${name}/ppl${SEQ}
    mkdir -p "$o"
    echo "--- [$(date +%H:%M:%S)] ppl@${SEQ} $name"
    CUDA_VISIBLE_DEVICES=0 "$PY" "$HARNESS_PPL" \
        --model "$EXPDIR/$name" \
        --output_dir "$o" \
        --wiki_text "$WIKI" \
        --seqlen "$SEQ" \
        --wiki_tokens 100000000 \
        --device cuda:0 2>&1 | tee "$o/ppl${SEQ}.log"
  done
done
echo "=== [$(date +%H:%M:%S)] STAGE 3 DONE ==="

# ------------------------------------------------------------------ STAGE 4
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

echo "=== [$(date +%H:%M:%S)] STAGE 4: hard_fold (GPU 0-3) + hard_drop (GPU 4-7) in parallel ==="
run_one hard_fold "$EXPDIR/hard_fold" 0,1,2,3 &
run_one hard_drop "$EXPDIR/hard_drop" 4,5,6,7 &
wait
echo "=== [$(date +%H:%M:%S)] STAGE 4 COMPLETE ==="

# ------------------------------------------------------------------ STAGE 5
# aggregate_zeroshot_union9.py hard-fails if any of the 9 tasks is absent, so a
# partial row can never be silently averaged.
echo "=== [$(date +%H:%M:%S)] STAGE 5: aggregate (asserts all 9 tasks present) ==="
for name in hard_fold hard_drop; do
  "$PY" "$AGG9" \
      --lm-eval-out "$OUT/$name/lm_eval_out" \
      --output "$OUT/$name/zeroshot_union9.json" \
      --model "sparseforge_dolmino_link2_iter17600_$name" || exit 1
done
echo "=== [$(date +%H:%M:%S)] STAGE 5 DONE ==="

# ------------------------------------------------------------------ STAGE 6
echo "=== [$(date +%H:%M:%S)] STAGE 6: verify 2:4 hard_drop (POST-inference) ==="
CUDA_VISIBLE_DEVICES=0 "$PY" "$VERIFY" \
    --model "$EXPDIR/hard_drop" --sample-layers 12 --seed 0 2>&1 \
    | tee "$OUT/verify_2of4_hard_drop_post.log"
post_rc=${PIPESTATUS[0]}
echo "=== [$(date +%H:%M:%S)] STAGE 6 DONE rc=$post_rc ==="

# ------------------------------------------------------------------ SUMMARY
echo "############ SUMMARY (pre_rc=$pre_rc post_rc=$post_rc) $(date -Is)"
for name in hard_fold hard_drop; do
  echo "--- $name"
  for SEQ in 4096 2048; do
    f=$OUT/$name/ppl${SEQ}/ppl_metrics.json
    [ -f "$f" ] && "$PY" -c "
import json; d=json.load(open('$f'))
print(f\"    ppl seqlen={d['seqlen']} wikitext2_ppl={d['wikitext2_ppl']:.6f} tokens={d['wikitext2_tokens']} exact_2of4_tile_ratio={d['exact_2of4_tile_ratio']}\")"
  done
  "$PY" -c "
import json
b=json.load(open('$OUT/$name/zeroshot_union9.json'))
for k in ('ast7','cast7','union9'):
    s=b[k]; print(f\"    {k}: plain_acc {s['mean_plain_acc']*100:.4f}  (primary {s['mean_primary']*100:.4f})\")
r=b['per_task']['rte']
print(f\"    RTE acc={r['acc']:.10f} n={r['n_samples']} k={r['acc']*r['n_samples']:.4f}\")
" 2>/dev/null
done
echo "############ ALL STAGES COMPLETE $(date -Is)"
