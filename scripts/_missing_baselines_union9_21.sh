#!/usr/bin/env bash
# Re-score the SparseForge main-table baseline rows that were MISSING from the
# reproduction harness (outputs/cast_eval_spec/), on node .21, lm-eval 0.4.8.
#
# WHY THIS SCRIPT EXISTS
#   The main table (SparseForge_NIPS_2026/sections/experiments.tex) has 11 measured
#   rows. The reproduction harness only covered 4 (dense_ref / wanda / ast_official /
#   sparseforge_5b). AST-official is the one arm present in BOTH the old and the new
#   harness, and its plain-acc AST-7 mean differs by -0.3460pp (57.9436 -> 57.5976)
#   with a max single-task delta of 1.59pp (boolq). So the old CSV numbers and the
#   reproduction numbers are NOT the same measurement and cannot be pooled into one
#   column. Every row we can still reach on disk therefore has to be re-measured
#   here, under the identical invocation.
#
# INVOCATION PARITY (the whole point)
#   Copied verbatim from _union9_eval_spec_21.sh, which itself copied
#   _cast_zeroshot_spec_21.sh. Only `pretrained` varies between arms:
#     lm-eval 0.4.8, --model hf, dtype=bfloat16, parallelize=True,
#     trust_remote_code=True, add_bos_token=False, --batch_size auto,
#     --num_fewshot 0, --seed 0, --log_samples, no chat template.
#   Union-9 = CAST-7 U AST-7, so AST-7 / CAST-7 / UNION-9 means are all slices of a
#   single invocation (never stitched across runs).
#
# WIKI PPL
#   @seqlen 2048 (not 4096). The existing four arms were measured at 4096 in
#   outputs/cast_eval_spec/ and at 2048 in outputs/cast_eval_spec_ppl2048/; the
#   paper's Wiki PPL column is the 2048 convention, so 2048 is what we add here.
#
# THE 2:4 GATE, AND WHY IT IS DIRECTION-AWARE
#   Strict gate = verify_2of4_hf_export.py: requires zero_frac==0.5 AND
#   bad_tiles==0, where "bad" is any 4-wide tile whose nonzero count != 2.
#   That is too strict in ONE direction. A tile with *fewer* than 2 nonzeros is a
#   strict subset of an allowed 2:4 tile: it is sparser, has fewer live weights, and
#   runs unchanged on 2:4 sparse tensor cores. A tile with *more* than 2 nonzeros is
#   a real budget violation and means the arm is not 2:4 at all.
#   Measured (tools/_tile_dir.py -> outputs/cast_eval_spec_gate/tiledir_*.json):
#     ProxSparse-official : 68 tiles with 1 nonzero,  0 tiles with >2  -> deployable
#     SparseGPT seed0     : 462 tiles with 1 nonzero, 0 tiles with >2  -> deployable
#   Both are 100% free of budget violations, so both are scored, with the strict-gate
#   FAIL and the direction breakdown recorded per arm in gate_report.json. An arm
#   with ANY tile >2 nonzeros is refused outright.
set -u
ROOT=/apdcephfs_wzc1/share_304376610/pighzliu_code
PY=/opt/conda/envs/torch-base/bin/python
LM_EVAL=/opt/conda/envs/torch-base/bin/lm_eval
HARNESS_PPL=$ROOT/baselines/eval_hf_sparse_model.py
MOM=$ROOT/Mixture-of-Memory
VERIFY_TOOL=$MOM/baselines/cast_repro/tools/verify_2of4_hf_export.py
TILEDIR_TOOL=$MOM/baselines/cast_repro/tools/tile_direction_audit.py
AGG9=$MOM/baselines/cast_repro/tools/aggregate_zeroshot_union9.py
WIKI=$ROOT/data/wikitext/wikitext-2-raw-v1/wiki.test.raw
PPL_SEQLEN=${PPL_SEQLEN:-2048}

SPEC=$ROOT/outputs/cast_eval_spec
U9=$ROOT/outputs/cast_eval_spec_union9
GATE=$ROOT/outputs/cast_eval_spec_gate

TASKS=boolq,rte,hellaswag,race,piqa,winogrande,arc_easy,arc_challenge,openbookqa

# boolq/rte are parquet redirects -> hub must be reachable.
export HF_HUB_OFFLINE=0
export http_proxy=http://hy-proxy.woa.com:3128
export https_proxy=http://hy-proxy.woa.com:3128
export no_proxy='mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,localhost,127.0.0.1,.oa.com,.woa.com,.local'

cd "$ROOT" || exit 1
mkdir -p "$GATE"

S=$ROOT/outputs/paper_v2/staged_diskb_models/outputs/paper_v2
P=$ROOT/outputs/paper_v2/staged_diskb_models/proxsparse_models
# arm_name -> checkpoint path. All HF safetensors dirs, all already on wzc1.
declare -A MODEL=(
  [alps_seed0]="$S/alps/llama2_wandb_sf_alps_v1_alps_seed0/hf_model"
  [alps_seed1]="$S/alps/llama2_wandb_sf_alps_v1_alps_seed1/hf_model"
  [alps_seed2]="$S/alps/llama2_wandb_sf_alps_v1_alps_seed2/hf_model"
  [elsa_seed0]="$S/elsa/paper_v2_overnight_20260725_v1_elsa_full4096/Llama--Llama2-7b_pruned0.5_admm_lr0.0002_20260725_0136"
  [elsa_seed1]="$S/elsa/paper_v2_node82_elsa_3seed_20260725_full4096_seed1/Llama--Llama2-7b_pruned0.5_admm_lr0.0002_20260725_1050"
  [elsa_seed2]="$S/elsa/paper_v2_node82_elsa_3seed_20260725_full4096_seed2/Llama--Llama2-7b_pruned0.5_admm_lr0.0002_20260725_2124"
  [sparsegpt_seed0]="$ROOT/outputs/paper_v2/materialized_baselines/sparsegpt_seed0"
  [sparsegpt_seed1]="$S/trainingfree_3seed/models/sparsegpt_seed1"
  [sparsegpt_seed2]="$S/trainingfree_3seed/models/sparsegpt_seed2"
  [proxsparse_official]="$P/Llama-2-7b-hf-en_sft_final_400_len4096_batch1_lambda0.25"
)
declare -A MODEL_ID=(
  [alps_seed0]="ALPS-2of4-llama2-7b-seed0"
  [alps_seed1]="ALPS-2of4-llama2-7b-seed1"
  [alps_seed2]="ALPS-2of4-llama2-7b-seed2"
  [elsa_seed0]="ELSA-4096steps-2of4-llama2-7b-seed0"
  [elsa_seed1]="ELSA-4096steps-2of4-llama2-7b-seed1"
  [elsa_seed2]="ELSA-4096steps-2of4-llama2-7b-seed2"
  [sparsegpt_seed0]="SparseGPT-2of4-llama2-7b-seed0"
  [sparsegpt_seed1]="SparseGPT-2of4-llama2-7b-seed1"
  [sparsegpt_seed2]="SparseGPT-2of4-llama2-7b-seed2"
  [proxsparse_official]="ProxSparse-official-llama2-7b-2of4"
)
ARMS="${ARMS:-alps_seed0 alps_seed1 alps_seed2 elsa_seed0 elsa_seed1 elsa_seed2 sparsegpt_seed0 sparsegpt_seed1 sparsegpt_seed2 proxsparse_official}"

# GPU groups for the zero-shot stage. Each group runs ONE arm with parallelize=True.
# Default = the full box as two 4-GPU halves, which is the exact topology that
# produced dense_ref / cast_7500 / wanda / ast_official. Keep 4 GPUs per group when
# sharing the box with another job -- do not shrink to 1-2 GPUs per arm, because the
# auto batch-size search would then run under a different memory ceiling than the
# already-published arms.
GPU_GROUPS="${GPU_GROUPS:-0,1,2,3 4,5,6,7}"
# GPUs usable one-per-arm for the (cheap, single-device) PPL stage.
PPL_GPUS="${PPL_GPUS:-0 1 2 3 4 5 6 7}"

# ------------------------------------------------------------------ STAGE 1: gate
# Strict gate + direction audit. An arm is scorable iff it has ZERO tiles with >2
# nonzeros (i.e. no 2:4 budget violation). Tiles with <2 are allowed and recorded.
echo "=== [$(date +%H:%M:%S)] STAGE 1: 2:4 gate (strict + direction audit) ==="
declare -A OK
for arm in $ARMS; do
  out=$SPEC/$arm; mkdir -p "$out"
  if [ ! -d "${MODEL[$arm]}" ]; then
    echo "  $arm MISSING_CKPT ${MODEL[$arm]}"; OK[$arm]=0; continue
  fi
  CUDA_VISIBLE_DEVICES="" "$PY" "$VERIFY_TOOL" \
      --model "${MODEL[$arm]}" --sample-layers 12 --seed 0 \
      >"$out/verify_2of4_pre.log" 2>&1
  strict_rc=$?
  CUDA_VISIBLE_DEVICES="" "$PY" "$TILEDIR_TOOL" \
      --model "${MODEL[$arm]}" --output "$GATE/tiledir_$arm.json" \
      >"$GATE/tiledir_$arm.log" 2>&1
  dir_rc=$?
  cp -f "$GATE/tiledir_$arm.json" "$out/tile_direction_audit.json" 2>/dev/null
  if [ "$dir_rc" -eq 0 ]; then OK[$arm]=1; else OK[$arm]=0; fi
  echo "  $arm strict_rc=$strict_rc direction_rc=$dir_rc scorable=${OK[$arm]} \
$(grep -o '"tiles_gt2_TOTAL_budget_violations": [0-9]*' "$GATE/tiledir_$arm.json" 2>/dev/null) \
$(grep -o '"tiles_lt2_TOTAL_sparser_than_2of4": [0-9]*' "$GATE/tiledir_$arm.json" 2>/dev/null)"
done
echo "=== [$(date +%H:%M:%S)] STAGE 1 DONE ==="

# ------------------------------------------------------------------ STAGE 2: PPL
echo "=== [$(date +%H:%M:%S)] STAGE 2: WikiText-2 PPL @${PPL_SEQLEN} ==="
ppl_one () {  # arm gpu
  local arm=$1 gpu=$2
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$HARNESS_PPL" \
      --model "${MODEL[$arm]}" \
      --output_dir "$SPEC/$arm" \
      --wiki_text "$WIKI" \
      --seqlen "$PPL_SEQLEN" \
      --wiki_tokens 100000000 \
      --device cuda:0 >"$SPEC/$arm/lm_eval_ppl.log" 2>&1
  echo "  $arm ppl_rc=$? $(grep -o '"wikitext2_ppl": [0-9.]*' "$SPEC/$arm/ppl_metrics.json" 2>/dev/null)"
}
g=0
set -- $PPL_GPUS
n_ppl=$#
for arm in $ARMS; do
  [ "${OK[$arm]}" -ne 1 ] && { echo "  SKIP $arm (gate)"; continue; }
  eval "gpu=\${$((g+1))}"
  ppl_one "$arm" "$gpu" &
  g=$(( (g+1) % n_ppl ))
  [ "$g" -eq 0 ] && wait
done
wait
echo "=== [$(date +%H:%M:%S)] STAGE 2 DONE ==="

# ------------------------------------------------------------- STAGE 3: zero-shot
# Two arms in flight, 4 GPUs each -- same topology that produced the 4 existing arms.
run_one () {  # arm gpus
  local arm=$1 gpus=$2
  local out=$U9/$arm
  mkdir -p "$out/lm_eval_out"
  echo "=== [$(date +%H:%M:%S)] EVAL $arm on GPUs $gpus ==="
  CUDA_VISIBLE_DEVICES="$gpus" HF_ALLOW_CODE_EVAL=1 HF_DATASETS_TRUST_REMOTE_CODE=1 \
  "$LM_EVAL" \
    --model hf \
    --model_args "pretrained=${MODEL[$arm]},dtype=bfloat16,parallelize=True,trust_remote_code=True,add_bos_token=False" \
    --tasks $TASKS \
    --batch_size auto \
    --num_fewshot 0 \
    --output_path "$out/lm_eval_out" \
    --seed 0 \
    --trust_remote_code \
    --log_samples >"$out/lm_eval.log" 2>&1
  echo "=== [$(date +%H:%M:%S)] DONE $arm rc=$? ==="
}
PEND=""
for arm in $ARMS; do [ "${OK[$arm]}" -eq 1 ] && PEND="$PEND $arm"; done
# Launch one arm per GPU group, then wait for the whole wave before the next.
set -- $GPU_GROUPS
NGROUP=$#
GROUPS_ARR=("$@")
set -- $PEND
while [ "$#" -gt 0 ]; do
  wave=""
  i=0
  while [ "$i" -lt "$NGROUP" ] && [ "$#" -gt 0 ]; do
    run_one "$1" "${GROUPS_ARR[$i]}" &
    wave="$wave $1"
    shift
    i=$((i+1))
  done
  echo "=== [$(date +%H:%M:%S)] STAGE 3 WAVE:$wave ==="
  wait
done
echo "=== [$(date +%H:%M:%S)] STAGE 3 COMPLETE ==="

# ------------------------------------------------------------ STAGE 4: aggregate
echo "=== [$(date +%H:%M:%S)] STAGE 4: aggregate union-9 ==="
for arm in $ARMS; do
  [ "${OK[$arm]}" -ne 1 ] && continue
  "$PY" "$AGG9" \
      --lm-eval-out "$U9/$arm/lm_eval_out" \
      --output "$U9/$arm/zeroshot_union9.json" \
      --model "${MODEL_ID[$arm]}" || { echo "  AGG FAILED $arm"; continue; }
  cp -f "$U9/$arm/zeroshot_union9.json" "$SPEC/$arm/zeroshot_metrics.json"
done
echo "=== [$(date +%H:%M:%S)] STAGE 4 DONE ==="

# --------------------------------------------------------- STAGE 5: gate (post)
echo "=== [$(date +%H:%M:%S)] STAGE 5: strict 2:4 gate (POST-inference) ==="
for arm in $ARMS; do
  [ "${OK[$arm]}" -ne 1 ] && continue
  CUDA_VISIBLE_DEVICES="" "$PY" "$VERIFY_TOOL" \
      --model "${MODEL[$arm]}" --sample-layers 12 --seed 0 \
      >"$SPEC/$arm/verify_2of4_post.log" 2>&1
  echo "  $arm post_rc=$? $(grep -o 'VERDICT: [A-Z]*' "$SPEC/$arm/verify_2of4_post.log" | tail -1)"
done
echo "=== [$(date +%H:%M:%S)] STAGE 5 DONE ==="

# ------------------------------------------------------------- STAGE 6: summary
echo "=== [$(date +%H:%M:%S)] STAGE 6: plain-acc summary + assertions ==="
"$PY" "$MOM/baselines/cast_repro/tools/summarize_missing_baselines.py" \
    --union9-root "$U9" \
    --gate-root "$GATE" \
    --output "$U9/missing_baselines_summary.json"
echo "=== [$(date +%H:%M:%S)] ALL STAGES COMPLETE ==="
