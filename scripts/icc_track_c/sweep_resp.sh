#!/bin/bash
set -u
cd /apdcephfs_wzc1/share_304376610/pighzliu_code/dllm_draft/runs/icc_track_c
D=/apdcephfs_wzc1/share_304376610/pighzliu_code/models
PY=/apdcephfs_wzc1/share_304376610/pighzliu_code/dllm_draft/.venv_b200/bin/python
declare -A M=( [b17]=$D/Qwen3-1.7B-Base [i17]=$D/Qwen--Qwen3-1.7b [b80]=$D/Qwen3-8B-Base [i80]=$D/Qwen--Qwen3-8b )
for TAG in b17 i17 b80 i80; do
  R=r_$TAG
  if [ -f emb/${R}_DONE ]; then echo "skip $R"; continue; fi
  echo "===== $R ${M[$TAG]} $(date +%H:%M:%S) ====="
  for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g LOCAL_RANK=0 RANK=$g \
      $PY embed_resp.py "${M[$TAG]}" emb/${R}_shard${g}of8.npy $g 8 > logs/emb_${R}_$g.log 2>&1 &
  done
  wait
  n=$(ls emb/${R}_shard*of8.npy 2>/dev/null | wc -l)
  if [ "$n" -eq 8 ]; then touch emb/${R}_DONE; echo "OK $R 8/8"; else echo "FAIL $R $n/8"; fi
done
echo RESP_ALLDONE
