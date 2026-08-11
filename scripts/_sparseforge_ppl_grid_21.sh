#!/usr/bin/env bash
# Complete the matched-seqlen PPL grid so no cell in the four-arm table is
# compared across context windows.
#
# WHY THIS IS NEEDED (discovered 2026-08-11 while doing task #244)
#   SparseForge's own headline PPL 6.2179 was measured at seqlen **4096**, not
#   2048. Provenance: outputs/paper_v2/ast7_eval/sparseforge_5b_table2/eval.log
#   shows `[eval_ppl]: 100%|...| 82/82`, and 82 x 4096 = 335,872 target tokens
#   (at 2048 the same corpus gives 164 sequences); the sibling
#   ast7_eval.json records "block_size": 4096.
#   But the AST row in the SAME csv (cast9_dense_ast_current_harness.csv, 6.3430)
#   comes from rebuttal_artifacts/2026-07-27/ast_official/ppl_metrics.json whose
#   own field says "seqlen": 2048.
#   => that CSV's PPL column mixes 4096 and 2048. SPEC.md:213 assumed the whole
#   column was 2048 and normalised everything to 2048 on that basis.
#
# This script fills in the two missing 2048 cells (dense, ast_official already
# exist elsewhere -> recomputed here anyway for a single consistent tree) and the
# 4096 cells for the SparseForge variants, so both rows of the final table can be
# stated at a single, explicit seqlen.
set -u
ROOT=/apdcephfs_wzc1/share_304376610/pighzliu_code
PY=/opt/conda/envs/torch-base/bin/python
HARNESS_PPL=$ROOT/baselines/eval_hf_sparse_model.py
WIKI=$ROOT/data/wikitext/wikitext-2-raw-v1/wiki.test.raw

SEQLEN=${SEQLEN:-4096}
GPU=${GPU:-1}
OUTROOT=$ROOT/outputs/cast_eval_spec_ppl${SEQLEN}_sf

declare -A MODELS=(
  [sparseforge_hard_drop]=$ROOT/outputs/sparseforge_5b_hf/hard_drop
  [sparseforge_soft_fold]=$ROOT/outputs/sparseforge_5b_hf/soft_fold
  [sparseforge_hard_fold]=$ROOT/outputs/sparseforge_5b_hf/hard_fold
  [dense_ref]=$ROOT/models/Llama--Llama2-7b
  [ast_official]=$ROOT/models/AST-official-LLaMA2-7B-2of4
  [cast_7500]=$ROOT/outputs/cast_repro_zero2/hf_final
  [wanda]=$ROOT/outputs/wanda_llama2_7b/hf_pruned
)

ARMS=${ARMS:-"sparseforge_hard_drop sparseforge_soft_fold sparseforge_hard_fold dense_ref ast_official cast_7500 wanda"}

cd "$ROOT" || exit 1
for name in $ARMS; do
  m=${MODELS[$name]:-}
  if [ -z "$m" ]; then echo "!!! unknown arm $name"; exit 1; fi
  if [ ! -d "$m" ]; then echo "--- SKIP $name (missing $m)"; continue; fi
  o=$OUTROOT/$name
  mkdir -p "$o"
  echo "=== [$(date +%H:%M:%S)] ppl@${SEQLEN} $name ==="
  CUDA_VISIBLE_DEVICES=$GPU "$PY" "$HARNESS_PPL" \
      --model "$m" \
      --output_dir "$o" \
      --wiki_text "$WIKI" \
      --seqlen "$SEQLEN" \
      --wiki_tokens 100000000 \
      --device cuda:0 2>&1 | tee "$o/ppl${SEQLEN}.log"
  echo "=== [$(date +%H:%M:%S)] DONE $name rc=${PIPESTATUS[0]} ==="
done

echo "=== SUMMARY ppl@${SEQLEN} ==="
for name in $ARMS; do
  f=$OUTROOT/$name/ppl_metrics.json
  [ -f "$f" ] || continue
  "$PY" -c "
import json
d=json.load(open('$f'))
print(f\"  {'$name':26s} seqlen={d['seqlen']} ppl={d['wikitext2_ppl']:.6f} tokens={d['wikitext2_tokens']} 2of4={d['exact_2of4_tile_ratio']}\")
"
done
