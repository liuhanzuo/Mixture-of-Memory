#!/usr/bin/env bash
# Clean re-run of the 6 OOD-pg19 PPL jobs on .252 (all GPUs free; wzc1 shared disk).
# One job per GPU, no overlap. Plus a base-wikitext cross-check on GPU6 to confirm
# transformers-version consistency vs the LOCAL runs (expect base wikitext PPL=5.9969).
set -u
cd /apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory
PY=/opt/conda/envs/torch-base/bin/python
BASE=/apdcephfs_wzc1/share_304376610/pighzliu_code/models/OLMo-2-1124-7B
PG=data/ood_ppl/pg19_test.npy
WT=data/ood_ppl/wikitext103_test.npy
mkdir -p logs ood_ppl_results

run_one(){ # gpu ckptarg out val
  local gpu="$1" ck="$2" out="$3" val="$4"
  rm -rf "ood_ppl_results/$out"
  CUDA_VISIBLE_DEVICES="$gpu" $PY scripts/eval_olmo2_probe2_ppl.py --base_model "$BASE" $ck \
    --val_path "$val" --output_name "$out" --results_root ood_ppl_results \
    --num_shards 1 --shard_index 0 --batch_size 8 > logs/ood252_${out}.log 2>&1
  CUDA_VISIBLE_DEVICES="$gpu" $PY scripts/eval_olmo2_probe2_ppl.py --merge \
    --output_name "$out" --results_root ood_ppl_results >> logs/ood252_${out}.log 2>&1
}

run_one 0 ""                                                                       base_pg19               "$PG" &
run_one 1 "--ckpt outputs/olmo2_probe2_7B_full32_dolmino/step25000.pt"             full32_step25000_pg19   "$PG" &
run_one 2 "--ckpt outputs/olmo2_probe2_7B_keep14fresh2/step200000.pt"              keep14_step200000_pg19  "$PG" &
run_one 3 "--ckpt outputs/olmo2_probe2_7B_shortgpt16/step200000.pt"                shortgpt_step200000_pg19 "$PG" &
run_one 4 "--ckpt outputs/olmo2_probe2_7B_keep14fresh2_fromscratch/step200000.pt"  random_step200000_pg19  "$PG" &
run_one 5 "--ckpt outputs/olmo2_probe2_7B_keep14fresh2_freezefront/step200000.pt"  frozen_step200000_pg19  "$PG" &
run_one 6 ""                                                                       base_wikitext103_XCHK   "$WT" &
wait
echo "PG19_252_DONE" > /tmp/pg19_252.done
{
  echo "=== .252 pg19 rerun summaries ($(date)) ==="
  for o in base_pg19 full32_step25000_pg19 keep14_step200000_pg19 shortgpt_step200000_pg19 random_step200000_pg19 frozen_step200000_pg19 base_wikitext103_XCHK; do
    s="ood_ppl_results/$o/summary.json"
    if [ -f "$s" ]; then
      echo -n "$o: "; $PY -c "import json;d=json.load(open('$s'));print('ppl',round(d['ppl'],4),'avg_nll',round(d['avg_nll'],4),'ntok',d['n_tokens'])"
    else echo "$o: MISSING"; fi
  done
} > /tmp/pg19_252_summary.txt 2>&1
cat /tmp/pg19_252_summary.txt
