#!/bin/bash
# Lean follow-up: wait for the running A-baseline fullchain eval to finish, then
# eval arm2-final + arm3-final CONCURRENTLY (4 GPUs each) + their pg19 ppl.
# Skips step200 midpoints (final ckpt = the scientific endpoint). Runs on .53.
set -uo pipefail
R="/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory"
cd "$R"
exec > logs/unfreeze_eval/leanfollow.log 2>&1
echo "[lean] start $(date)"

# 1) Wait for the A-baseline probe wrapper to exit (all 8 shards done).
while pgrep -f "_eval_unfreeze_fc16k_remote53.sh Abaseline" >/dev/null 2>&1; do
  sleep 60
done
echo "[lean] A-baseline eval finished $(date)"
sleep 5

A2="outputs/mem_space_unfreeze_arm2_top16"
A3="outputs/mem_space_unfreeze_arm3_full"

# 2) arm2-final on GPUs 0-3, arm3-final on GPUs 4-7, in parallel.
bash scripts/_eval_unfreeze_fc16k_remote53.sh arm2_top16_s400 "$A2/full_model.pt" "$A2/adapter_config.json" 0,1,2,3 &
P2=$!
bash scripts/_eval_unfreeze_fc16k_remote53.sh arm3_full_s400 "$A3/full_model.pt" "$A3/adapter_config.json" 4,5,6,7 &
P3=$!
wait $P2 $P3
echo "[lean] both arm fullchain evals done $(date)"

# 3) pg19 ppl guardrail (fast, sequential, GPU 0 then 4).
bash scripts/_eval_unfreeze_ppl_remote53.sh arm2_top16_s400 "$A2/full_model.pt" "$A2/adapter_config.json" 0
bash scripts/_eval_unfreeze_ppl_remote53.sh arm3_full_s400 "$A3/full_model.pt" "$A3/adapter_config.json" 4
echo "[lean] ppl done $(date)"

# 4) Summary.
echo "======== STRICT-RESCORE (fullchain oracle qa5 16k n100) ========"
for d in babilong_results/unfreeze_Abaseline_fc16k babilong_results/unfreeze_arm2_top16_s400_fc16k babilong_results/unfreeze_arm3_full_s400_fc16k; do
  [ -d "$d" ] || { echo "$(basename $d): MISSING"; continue; }
  echo -n "$(basename $d): "
  .venv/bin/python scripts/strict_rescore.py "$d/qa5_16k_fullchain_oracle_n*.csv" 2>&1 | tail -1
done
echo "======== PPL GUARDRAIL ========"
for j in babilong_results/ppl_unfreeze_arm2_top16_s400.json babilong_results/ppl_unfreeze_arm3_full_s400.json; do
  [ -f "$j" ] || { echo "$(basename $j): MISSING"; continue; }
  echo -n "$(basename $j): "; grep -oE '"(avg_nll|ppl)": [0-9.]+' "$j" | tr '\n' ' '; echo
done
echo "[lean] SUMMARY DONE $(date)"
