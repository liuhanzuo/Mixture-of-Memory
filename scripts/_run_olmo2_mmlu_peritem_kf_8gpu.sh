#!/usr/bin/env bash
# Paper B P0.4 checkpoint-to-checkpoint paired MMLU (trajectory). 8-GPU sharded
# likelihood-MC MMLU eval with --save_per_example, generalised over keep_front /
# n_fresh so BOTH keep14 (KF=14) and keep8 (KF=8) prune-heal ckpts can be dumped
# with item_id-aligned per-example jsonl. Same harness/protocol as
# scripts/_run_olmo2_mmlu_peritem_8gpu.sh (add_bos=0, BS=8, max_len=1024, mode=mc,
# 8-shard [g::8] + merge); with KF/NF defaulting to 14/2 the keep14 behaviour is
# byte-identical. keep_front/n_fresh must AGREE with the ckpt meta (hard error
# otherwise), so KF/NF are also a guard.
#
# env in:  ROOT   node project root (wzc1 alias)
#          NAME   output_name (new dir, must NOT clobber any *_know / *_peritem)
#          BASE   base model path (../models/OLMo-2-1124-7B)
#          CKPT   prune-heal ckpt (.pt)
#          KF     keep_front_layers (default 14; set 8 for keep8)
#          NF     n_fresh_layers    (default 2)
#          PY     python (default torch-base)
set -u
ROOT="${ROOT:?set ROOT}"
NAME="${NAME:?set NAME}"
BASE="${BASE:-../models/OLMo-2-1124-7B}"
CKPT="${CKPT:?set CKPT}"
KF="${KF:-14}"
NF="${NF:-2}"
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

echo "[$(date '+%F %T')] $NAME base=$BASE ckpt=$CKPT KF=$KF NF=$NF root=$ROOT"
for g in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_probe2_downstream.py \
    --base_model "$BASE" --ckpt "$CKPT" \
    --keep_front_layers "$KF" --n_fresh_layers "$NF" \
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
