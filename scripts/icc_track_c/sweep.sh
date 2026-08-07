#!/bin/bash
set -u
cd /apdcephfs_wzc1/share_304376610/pighzliu_code/dllm_draft/runs/icc_track_c
D=/apdcephfs_wzc1/share_304376610/pighzliu_code/models
# tag:path  -- fully crossed {0.6,1.7,4,8}B x {Base,Instruct}
declare -A M=(
 [b06]=$D/Qwen3-0.6B-Base   [i06]=$D/Qwen3-0.6B-Instruct
 [b17]=$D/Qwen3-1.7B-Base   [i17]=$D/Qwen--Qwen3-1.7b
 [b40]=$D/Qwen3-4B-Base     [i40]=$D/Qwen3-4B
 [b80]=$D/Qwen3-8B-Base     [i80]=$D/Qwen--Qwen3-8b
)
for TAG in b06 i06 b17 i17 b40 i40 b80 i80; do
  if [ -f emb/${TAG}_DONE ]; then echo "skip $TAG"; continue; fi
  echo "===== $TAG ${M[$TAG]} $(date +%H:%M:%S) ====="
  MODEL=${M[$TAG]} TAG=$TAG bash run_embed.sh
  n=$(ls emb/${TAG}_shard*of8.npy 2>/dev/null | wc -l)
  if [ "$n" -eq 8 ]; then touch emb/${TAG}_DONE; echo "OK $TAG 8/8"; else echo "FAIL $TAG $n/8"; fi
done
echo SWEEP_ALLDONE
