#!/usr/bin/env bash
# Pull the keep14-distill step5000.pt (24.5 GB) from zwfy6 -> wzc1 with parallel
# byte-range `ssh dd` streams, then verify md5 against the source.
#
# WHY: .212 is a wzc1 node and the only distill checkpoint (step5000.pt, the
# frontier -- the 08-05 run reached step7780 but --save_every 5000 meant it never
# saved) lives ONLY on zwfy6. Single-stream scp measures ~17.7 MB/s (~24 min);
# 6 parallel byte-range streams measured ~92 MB/s on the keep10 corpus pull.
#
# This is READ-ONLY on the source node and touches NO GPU, so it cannot disturb
# the keep8/keep12/paperC 200k trainings running on .73/.82/.104.
#
# LOCAL runs this (it has sshpass and shares the wzc1 destination disk with .212).
set -euo pipefail

SRC_IP=${SRC_IP:-28.82.250.82}
SRC_PW=${SRC_PW:-configs/password_h20_82250.txt}
SRC=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/outputs/olmo2_probe2_7B_keep14fresh2_distill/step5000.pt
DST_DIR=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/outputs/olmo2_probe2_7B_keep14fresh2_distill
DST=$DST_DIR/step5000.pt
NSTREAM=${NSTREAM:-6}
EXPECTED_BYTES=24489312843

cd /apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory
mkdir -p "$DST_DIR"

SSH="sshpass -f $SRC_PW ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o PreferredAuthentications=password root@$SRC_IP"

# ---- assert the source really is the expected size (never trust the台账) ----
SRC_BYTES=$($SSH "stat -c %s $SRC")
if [ "$SRC_BYTES" != "$EXPECTED_BYTES" ]; then
  echo "FATAL: source bytes=$SRC_BYTES expected $EXPECTED_BYTES"; exit 1
fi
echo "source verified: $SRC_BYTES bytes on $SRC_IP"

# ---- fan out NSTREAM byte-exact ranges ----
CHUNK=$(( (EXPECTED_BYTES + NSTREAM - 1) / NSTREAM ))
PIDS=()
for i in $(seq 0 $((NSTREAM-1))); do
  OFF=$(( i * CHUNK ))
  LEN=$CHUNK
  if [ $(( OFF + LEN )) -gt $EXPECTED_BYTES ]; then LEN=$(( EXPECTED_BYTES - OFF )); fi
  echo "stream $i: offset=$OFF len=$LEN"
  $SSH "dd if=$SRC bs=4M iflag=skip_bytes,count_bytes skip=$OFF count=$LEN status=none" \
      > "$DST.part$i" &
  PIDS+=($!)
done
FAIL=0
for p in "${PIDS[@]}"; do wait "$p" || FAIL=1; done
if [ "$FAIL" != "0" ]; then echo "FATAL: a stream failed"; exit 1; fi

# ---- assert each part length, then concatenate ----
TOTAL=0
for i in $(seq 0 $((NSTREAM-1))); do
  B=$(stat -c %s "$DST.part$i"); TOTAL=$(( TOTAL + B ))
  echo "part$i = $B bytes"
done
if [ "$TOTAL" != "$EXPECTED_BYTES" ]; then
  echo "FATAL: parts total $TOTAL != $EXPECTED_BYTES"; exit 1
fi
cat $(for i in $(seq 0 $((NSTREAM-1))); do echo "$DST.part$i"; done) > "$DST"
rm -f "$DST".part*

DST_BYTES=$(stat -c %s "$DST")
if [ "$DST_BYTES" != "$EXPECTED_BYTES" ]; then
  echo "FATAL: dst bytes=$DST_BYTES"; exit 1
fi
echo "concatenated OK: $DST_BYTES bytes"

# ---- md5 both ends (this is the cross-disk identity proof) ----
echo "computing md5 on both ends (parallel) ..."
$SSH "md5sum $SRC" | awk '{print $1}' > /tmp/src_md5.txt &
SP=$!
md5sum "$DST" | awk '{print $1}' > /tmp/dst_md5.txt
wait $SP
SRC_MD5=$(cat /tmp/src_md5.txt); DST_MD5=$(cat /tmp/dst_md5.txt)
echo "src md5=$SRC_MD5"
echo "dst md5=$DST_MD5"
if [ "$SRC_MD5" != "$DST_MD5" ]; then echo "FATAL: md5 MISMATCH"; exit 1; fi
echo "TRANSFER_VERIFIED md5=$DST_MD5"
