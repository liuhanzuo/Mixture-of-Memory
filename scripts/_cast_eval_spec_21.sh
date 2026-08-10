#!/usr/bin/env bash
# Score the finished CAST run against the dense reference on ONE harness.
# SPEC.md S7: WikiText-2 PPL at 4096 context. S8: judge vs the ~6.2-6.5 band
# (our harness reads ~+11% vs the paper's scale), NOT vs the paper's 5.58.
set -u
ROOT=/apdcephfs_wzc1/share_304376610/pighzliu_code
PY=/opt/conda/envs/torch-base/bin/python
HARNESS=$ROOT/baselines/eval_hf_sparse_model.py
WIKI=$ROOT/data/wikitext/wikitext-2-raw-v1/wiki.test.raw
cd "$ROOT" || exit 1

run_one () {  # name model_path gpu
  echo "=== [$(date +%H:%M:%S)] EVAL $1 on GPU $3 ==="
  "$PY" "$HARNESS" \
    --model "$2" \
    --output_dir "$ROOT/outputs/cast_eval_spec/$1" \
    --wiki_text "$WIKI" \
    --seqlen 4096 \
    --wiki_tokens 100000000 \
    --device "cuda:$3"
  echo "=== [$(date +%H:%M:%S)] DONE $1 rc=$? ==="
}

run_one cast_7500  "$ROOT/outputs/cast_repro_zero2/hf_final" 0 &
run_one dense_ref  "$ROOT/models/Llama--Llama2-7b"           1 &
wait
echo "=== ALL EVALS COMPLETE $(date +%H:%M:%S) ==="
for f in "$ROOT"/outputs/cast_eval_spec/*/*.json; do echo "--- $f"; cat "$f"; done
