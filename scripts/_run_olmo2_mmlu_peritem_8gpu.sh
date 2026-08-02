#!/usr/bin/env bash
# Paper B #103 P2.8: per-item McNemar leg. 8-GPU sharded likelihood-MC MMLU eval
# with --save_per_example, matched keep14-triad @step200000. Same harness/protocol
# as scripts/_run_olmo2_probe2_downstream_keep14_know_8gpu.sh (add_bos=0, BS=8,
# max_len=1024 default, mode=mc, 8-shard [g::8] + merge) so the aggregate MMLU acc
# reproduces the _know ledger (keep14 ~.3191 / frozen ~.2628).
#
# env in:  ROOT   node project root (wzc1 alias on .73/.104, zwfy6 on .82)
#          NAME   output_name (new dir, must NOT clobber any *_know)
#          BASE   base model path (../models/OLMo-2-1124-7B)
#          CKPT   prune-heal ckpt (.pt); keep_front/n_fresh read from ckpt meta
#          PY     python (default torch-base)
set -u
ROOT="${ROOT:?set ROOT}"
NAME="${NAME:?set NAME}"
BASE="${BASE:-../models/OLMo-2-1124-7B}"
CKPT="${CKPT:?set CKPT}"
PY="${PY:-/opt/conda/envs/torch-base/bin/python}"
cd "$ROOT"

export HF_DATASETS_CACHE="$ROOT/data/hf_datasets_cache"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
mkdir -p logs olmo2_downstream_results "$HF_DATASETS_CACHE"

TASKS="mmlu"
BS=8
DONE="logs/olmo2_mmlu_peritem_${NAME}_DONE"
rm -f "$DONE"

echo "[$(date '+%F %T')] $NAME base=$BASE ckpt=$CKPT root=$ROOT"
for g in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_probe2_downstream.py \
    --base_model "$BASE" --ckpt "$CKPT" \
    --keep_front_layers 14 --n_fresh_layers 2 \
    --tasks "$TASKS" --num_shards 8 --shard_index $g --batch_size $BS \
    --save_per_example --output_name "$NAME" \
    > "logs/olmo2_mmlu_peritem_${NAME}_shard${g}.log" 2>&1 &
done
wait
echo "[$(date '+%F %T')] $NAME shards done; merging"
$PY scripts/eval_olmo2_probe2_downstream.py --merge --output_name "$NAME" 2>&1

echo "[$(date '+%F %T')] $NAME MERGE DONE" | tee "$DONE"
$PY - <<PYEOF >> "$DONE" 2>&1
import json
d=json.load(open("olmo2_downstream_results/$NAME/summary.json"))
t=d["tasks"]["mmlu"]
print("mmlu acc=%.6f n=%d n_nan=%d n_trunc=%d" % (t["acc"], t["n"], t["n_nan"], t.get("n_trunc",0)))
PYEOF
cat "$DONE"
