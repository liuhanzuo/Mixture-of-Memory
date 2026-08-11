set -u
ROOT=/apdcephfs_wzc1/share_304376610/pighzliu_code
PY=/opt/conda/envs/torch-base/bin/python
LM_EVAL=/opt/conda/envs/torch-base/bin/lm_eval
TOOLS=$ROOT/Mixture-of-Memory/baselines/cast_repro/tools
CKPT=$ROOT/out_llama/models_Llama--Llama2-7b_mask-unstructured_s0.5_m-hessian_obd_20260401_124938/model_best_lm_eval.pt
WIKI=$ROOT/data/wikitext/wikitext-2-raw-v1/wiki.test.raw
OUT=$ROOT/outputs/cast_eval_spec/sparseforge_dolmino_link2
EXPDIR=$ROOT/outputs/sparseforge_dolmino_link2_hf
TASKS=boolq,rte,hellaswag,race,piqa,winogrande,arc_easy,arc_challenge,openbookqa
export HF_HUB_OFFLINE=0 http_proxy=http://hy-proxy.woa.com:3128 https_proxy=http://hy-proxy.woa.com:3128
export no_proxy='mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,localhost,127.0.0.1,.oa.com,.woa.com,.local'
cd "$ROOT" || exit 1
echo "=== [$(date +%H:%M:%S)] export soft_fold (faithfulness control)"
if [ ! -f "$EXPDIR/soft_fold/sparseforge_export_meta.json" ]; then
  "$PY" "$TOOLS/export_sparseforge_to_hf.py" --ckpt "$CKPT" --output "$EXPDIR/soft_fold" \
      --mask soft --slorb fold 2>&1 | tee "$OUT/export_soft_fold.log"
  [ "${PIPESTATUS[0]}" -ne 0 ] && { echo "!!! export failed"; exit 1; }
fi
for SEQ in 4096 2048; do
  o=$OUT/soft_fold/ppl${SEQ}; mkdir -p "$o"
  echo "=== [$(date +%H:%M:%S)] ppl@${SEQ} soft_fold"
  CUDA_VISIBLE_DEVICES=0 "$PY" "$ROOT/baselines/eval_hf_sparse_model.py" --model "$EXPDIR/soft_fold" \
      --output_dir "$o" --wiki_text "$WIKI" --seqlen "$SEQ" --wiki_tokens 100000000 \
      --device cuda:0 2>&1 | tee "$o/ppl${SEQ}.log"
done
o=$OUT/soft_fold; mkdir -p "$o/lm_eval_out"
echo "=== [$(date +%H:%M:%S)] EVAL soft_fold on GPUs 0-7"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_ALLOW_CODE_EVAL=1 HF_DATASETS_TRUST_REMOTE_CODE=1 \
"$LM_EVAL" --model hf \
  --model_args "pretrained=$EXPDIR/soft_fold,dtype=bfloat16,parallelize=True,trust_remote_code=True,add_bos_token=False" \
  --tasks $TASKS --batch_size auto --num_fewshot 0 --output_path "$o/lm_eval_out" \
  --seed 0 --trust_remote_code --log_samples 2>&1 | tee "$o/lm_eval.log"
"$PY" "$TOOLS/aggregate_zeroshot_union9.py" --lm-eval-out "$o/lm_eval_out" \
  --output "$o/zeroshot_union9.json" --model "sparseforge_dolmino_link2_iter17600_soft_fold" || exit 1
echo "=== [$(date +%H:%M:%S)] soft_fold COMPLETE"
