#!/usr/bin/env bash
# B-P0.7 (matched-PPL crossing test): keep14+fresh2 @ step67500, held-out PPL
# 11.5331 which is within 0.035 of Random-16L@200k (11.4983). Same base protocol,
# same harness, same BS=8, same 8-shard[g::8]+merge; only --ckpt differs from
# _run_olmo2_probe2_downstream_keep14_8gpu.sh. Runs on .252 while crossing
# monitor is between bursts (next burst ~35 min away, this eval ~15 min).
set -u
cd /apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory
PY=/opt/conda/envs/torch-base/bin/python
export http_proxy=http://hy-proxy.woa.com:3128 https_proxy=http://hy-proxy.woa.com:3128 all_proxy=http://hy-proxy.woa.com:3128
export no_proxy="localhost,127.0.0.1,.oa.com,.woa.com,.local"
export HF_DATASETS_CACHE=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/data/hf_datasets_cache
mkdir -p logs olmo2_downstream_results "$HF_DATASETS_CACHE"

CKPT=outputs/olmo2_keep14_densesave_reheal/step67500.pt
BASE=../models/OLMo-2-1124-7B
BS=8
DONE=logs/olmo2_downstream_keep14_reheal_step67500_DONE
rm -f "$DONE"
[ -f "$CKPT" ] || { echo "FATAL: ckpt missing" | tee "$DONE"; exit 1; }

LEGS=(
  "7B_keep14_reheal_step67500|hellaswag,arc_challenge,arc_easy,piqa,winogrande,openbookqa"
  "7B_keep14_reheal_step67500_know|mmlu,lambada_openai,boolq,commonsense_qa,social_iqa"
)
for leg in "${LEGS[@]}"; do
  NAME="${leg%%|*}"; TASKS="${leg#*|}"
  echo "[$(date '+%F %T')] LEG $NAME"
  $PY scripts/eval_olmo2_probe2_downstream.py --prepare_data --tasks "$TASKS" \
    > "logs/olmo2_downstream_${NAME}_prepare.log" 2>&1
  for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_probe2_downstream.py \
      --base_model "$BASE" --ckpt "$CKPT" --tasks "$TASKS" \
      --num_shards 8 --shard_index $g --batch_size $BS --output_name "$NAME" \
      > "logs/olmo2_downstream_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  NSH=$(ls olmo2_downstream_results/"$NAME"/shard*of8.json 2>/dev/null | wc -l)
  if [ "$NSH" -ne 8 ]; then echo "FATAL: only $NSH/8 shards; NOT merging." | tee -a "$DONE"; continue; fi
  echo "[$(date '+%F %T')] $NAME 8/8 shards; merging"
  $PY scripts/eval_olmo2_probe2_downstream.py --merge --output_name "$NAME" 2>&1
done

echo "[$(date '+%F %T')] KEEP14_STEP67500 DOWNSTREAM DONE" | tee "$DONE"
for leg in "${LEGS[@]}"; do
  NAME="${leg%%|*}"
  echo "--- $NAME ---" >> "$DONE"
  cat "olmo2_downstream_results/${NAME}/summary.json" >> "$DONE" 2>/dev/null
done

# ★ 决定性对照打印
$PY - >> "$DONE" 2>&1 <<'PYEOF'
import json, os
CORE6 = ["hellaswag","arc_challenge","arc_easy","piqa","winogrande","openbookqa"]
KNOW5 = ["mmlu","lambada_openai","boolq","commonsense_qa","social_iqa"]
def load(name):
    p = f"olmo2_downstream_results/{name}/summary.json"
    if not os.path.exists(p): return {}
    d = json.load(open(p)).get("tasks",{})
    return {k:v.get("acc") for k,v in d.items() if isinstance(v,dict)}
rows = {
 "keep14@200k  PPL=10.56": {**load("7B_keep14_step200000"), **load("7B_keep14_step200000_know")},
 "keep14@67500 PPL=11.53": {**load("7B_keep14_reheal_step67500"), **load("7B_keep14_reheal_step67500_know")},
 "Random16L@200k PPL=11.50": {**load("7B_scratch16L_step200000"), **load("7B_scratch16L_step200000_know")},
 "ShortGPT16@200k PPL=9.78": {**load("7B_shortgpt_step200000"), **load("7B_shortgpt_step200000_know")},
}
tasks = CORE6 + KNOW5
print(f"\n{'task':16s}" + "".join(f"{k:>26s}" for k in rows))
for t in tasks:
    line=f"{t:16s}"
    for k,r in rows.items():
        v=r.get(t)
        line+=(f"{v:26.4f}" if v is not None else f"{'--':>26s}")
    print(line)
# ★ 核心问题: 同 PPL (~11.5) 下 keep14 vs Random16L 的 MMLU 差
k14 = rows["keep14@67500 PPL=11.53"].get("mmlu")
rnd = rows["Random16L@200k PPL=11.50"].get("mmlu")
print(f"\n★ 关键: 同 PPL~11.5 下, keep14@67500 MMLU={k14} vs Random16L@200k MMLU={rnd}")
if k14 and rnd:
    print(f"  差值 = {(k14-rnd)*100:+.2f}pp")
    print(f"  keep14@200k (更好 PPL=10.56) MMLU = {rows['keep14@200k  PPL=10.56'].get('mmlu')}")
    print(f"  → 若 k14@67500 ≫ Random: 结构/init 在同 PPL 下仍带信号 (dissociation 微救)")
    print(f"  → 若 k14@67500 ≈ Random: PPL 完全预测 MMLU (dissociation 死透)")
PYEOF
echo "[$(date '+%F %T')] wrote $DONE"
