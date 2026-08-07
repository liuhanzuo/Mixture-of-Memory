#!/bin/bash
# usage: MODEL=<path> TAG=<name> bash run_embed.sh
set -u
cd /apdcephfs_wzc1/share_304376610/pighzliu_code/dllm_draft/runs/icc_track_c
PY=/apdcephfs_wzc1/share_304376610/pighzliu_code/dllm_draft/.venv_b200/bin/python
mkdir -p emb logs
for g in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$g LOCAL_RANK=0 RANK=$g \
    $PY embed_full.py "$MODEL" emb/${TAG}_shard${g}of8.npy $g 8 \
    > logs/emb_${TAG}_$g.log 2>&1 &
done
wait
echo "EMBED_DONE $TAG"
ls -la emb/${TAG}_shard*of8.npy | wc -l
