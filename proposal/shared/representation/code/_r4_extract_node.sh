#!/usr/bin/env bash
# Paper D / R4 driver: extract word-aligned hiddens for one model per GPU.
# Usage: MODELS="k1 k2 ... k8" bash _r4_extract_node.sh
set -u
R=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
P=/opt/conda/envs/torch-base/bin/python
S=$R/proposal/shared/representation/code/repr_alignment_multimodel.py
LOG=$R/proposal/shared/representation/logs_align
mkdir -p "$LOG"
cd "$R" || exit 1
: "${NTEXTS:=300}"
: "${MAXWORDS:=4500}"
g=0
for m in $MODELS; do
  RI=""
  case "$m" in RANDOM_*) RI="--random_init"; m="${m#RANDOM_}";; esac
  ( timeout 5400 $P "$S" --stage extract --model "$m" $RI --device "cuda:$g" \
      --n_texts "$NTEXTS" --max_words "$MAXWORDS" \
      >"$LOG/extract_${m}${RI:+_rand}.log" 2>&1
    echo "EXIT=$? model=$m${RI:+_rand}" >>"$LOG/extract_${m}${RI:+_rand}.log" ) &
  g=$((g+1))
done
wait
echo "ALL_EXTRACT_DONE $(date +%FT%T)"
