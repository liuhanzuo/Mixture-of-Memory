#!/usr/bin/env bash
# Paper D / R4 driver: one GPU per slot of pairs (slots come from _r4_pair_slots.txt).
set -u
R=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
P=/opt/conda/envs/torch-base/bin/python
S=$R/proposal/shared/representation/code/repr_alignment_multimodel.py
LOG=$R/proposal/shared/representation/logs_align
SLOTS=$R/proposal/shared/representation/code/_r4_pair_slots.txt
mkdir -p "$LOG"
cd "$R" || exit 1
: "${SLOT_LO:=0}"
: "${SLOT_HI:=7}"
g=0
for s in $(seq "$SLOT_LO" "$SLOT_HI"); do
  LINE=$(sed -n "$((s+1))p" "$SLOTS")
  [ -z "$LINE" ] && { g=$((g+1)); continue; }
  ( timeout 10800 $P "$S" --stage cka --pairs $LINE --device "cuda:$g" \
      >"$LOG/cka_slot${s}.log" 2>&1
    echo "EXIT=$? slot=$s" >>"$LOG/cka_slot${s}.log" ) &
  g=$((g+1))
done
wait
echo "ALL_CKA_DONE slots ${SLOT_LO}-${SLOT_HI} $(date +%FT%T)"
