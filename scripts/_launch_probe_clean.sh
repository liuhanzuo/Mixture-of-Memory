#!/bin/bash
# Launch sharded clean readout-wall judge: probe_fullchain_oracle_qa5 (original vs fullchain SF oracle)
# n=100 qa5 16k, both oracle modes, 8 shards on 8 local GPUs.
cd /apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory || exit 1
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
mkdir -p babilong_results/probe_clean_qa5_16k logs/probe_clean
for s in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$s setsid .venv/bin/python -u scripts/probe_fullchain_oracle_qa5.py \
    --lengths 16k --limit 100 --num_shards 8 --shard_idx "$s" --oracle_mode fullchain \
    --device cuda:0 \
    --results_folder babilong_results/probe_clean_qa5_16k \
    > "logs/probe_clean/shard${s}.log" 2>&1 &
  echo "shard$s -> GPU$s PID $!"
done
wait
