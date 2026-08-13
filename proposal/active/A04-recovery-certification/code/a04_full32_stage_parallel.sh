#!/usr/bin/env bash
# ============================================================================
# A04 -- parallel cross-disk staging of the full32_dolmino 7B checkpoints.
#
# WHY THIS EXISTS
# ---------------
# `STATUS.json:full32_trajectory_staging_remeasured_20260813` priced the full32
# trajectory scan at 5.7 h of transfer (4 x 81.6 GiB at a MEASURED 16.3 MiB/s)
# and concluded WAIT-for-the-B200. That measurement is correct and I reproduced
# it (2 GiB in 118 s = 17.4 MiB/s), but the inference from it is wrong:
#
#     16-17 MiB/s is a PER-STREAM limit, NOT a property of the link.
#
# Measured 2026-08-13 on this dispatch, wzc1 -> zwfy6, same files:
#     1 stream                    2 GiB / 118 s =  17.4 MiB/s
#     4 streams (2x.73 + 2x.82)   4 GiB /  86 s =  47.6 MiB/s aggregate
#     8 streams (4x.73 + 4x.82)   8 GiB /  61 s = 134.3 MiB/s aggregate
#     8 streams (.73 ALONE)       6 GiB /  47 s = 130.7 MiB/s aggregate
#
# The last row is the load-bearing one: 8 concurrent streams to ONE node reach
# 130 MiB/s, i.e. 7.5x the single-stream rate, so the ceiling is per-connection
# (ssh/TCP window + per-stream cephfs read), not the interconnect. That turns
# the staging cost from 5.7 h into ~45 min for all four checkpoints and makes
# the WAIT decision obsolete. Recorded so the next agent does not re-inherit a
# single-stream number as if it were a bandwidth budget.
#
# WHY FULL FILES AND NOT JUST `model_state`
# -----------------------------------------
# `scripts/eval_olmo2_probe2_ppl.load_pruned_model` reads ONLY ck["model_state"]
# (355 tensors, 27.2 GiB fp32) plus the arch meta; the ~54 GiB of
# `optimizer_state` in each 81.57 GiB file is never touched at eval time, so a
# slimmed re-serialisation would cut the transfer 3x. It is deliberately NOT
# done: re-serialising would make the staged artefact a DIFFERENT file from the
# archived one, and the dispatch's integrity contract (full-file sha256 equal on
# both disks, zip entry count == 1435) could then no longer be stated. At
# 130 MiB/s the 3x saving is worth ~30 min and is not worth weakening the
# provenance check. Byte-for-byte copies only.
#
# METHOD
# ------
# N concurrent (dd | ssh dd) streams, each moving a disjoint byte range into a
# pre-`truncate`d destination with oflag=seek_bytes conv=notrunc, so the streams
# never interleave writes. Not `scp -O`: scp cannot express a byte range, and
# .82's sftp subsystem is broken so plain scp fails there outright.
#
# VERIFICATION (all three, in this order; any failure leaves the file in place
# for inspection and returns non-zero)
#   1. size == source size
#   2. FULL-FILE sha256 == source sha256 (computed on wzc1). NOT a head/tail
#      sample: the known failure mode on this cluster is a TRUNCATED WRITE
#      (`shortgpt16/step128000.pt` is 15.9% of its siblings on zwfy6 yet passes
#      `ls -l` inspection), and a prefix hash cannot see it.
#   3. zipfile entry count == 1435, matching the source. Catches a file that is
#      the right length and hash but whose zip central directory is unreadable.
#
# WRITES NOTHING ON wzc1. The source disk is opened read-only; every write goes
# to $DEST on the zwfy6 node. LOCAL/.21 GPUs are running SparseForge #246 and
# are not touched -- this script uses LOCAL only as a CPU/network source.
#
# usage:
#   bash a04_full32_stage_parallel.sh <node> <step> [nstream]
#     node   = 73 | 82
#     step   = 5000 | 10000 | 15000 | 20000
#     nstream= default 8
# ============================================================================
set -u
WZ=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory
ZW=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
NODE="${1:?need node: 73|82}"
STEP="${2:?need step}"
NSTREAM="${3:-8}"

case "$NODE" in
  73) IP=28.85.35.73;  PF=$WZ/configs/password_h20_853573.txt ;;
  82) IP=28.82.250.82; PF=$WZ/configs/password_h20_82250.txt ;;
  *)  echo "FATAL: node must be 73 or 82, got '$NODE'"; exit 2 ;;
esac

SRC=$WZ/outputs/olmo2_probe2_7B_full32_dolmino/step${STEP}.pt
DEST=$ZW/outputs/a04_staged/full32_step${STEP}_from_wzc1.pt
EXPECT_ENTRIES=1435

SSH() { sshpass -f "$PF" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 \
        -o PreferredAuthentications=password -o ServerAliveInterval=30 "root@$IP" "$@"; }

[ -f "$SRC" ] || { echo "FATAL: source absent: $SRC"; exit 3; }
SZ=$(stat -c %s "$SRC")
echo "[$(date '+%F %T')] STAGE step${STEP} -> .${NODE}  size=${SZ} B ($(awk -v s=$SZ 'BEGIN{printf "%.2f", s/2^30}') GiB) nstream=${NSTREAM}"

# --- already staged and intact? then do not move it again -------------------
PRE=$(SSH "stat -c %s '$DEST' 2>/dev/null || echo 0")
if [ "$PRE" = "$SZ" ]; then
  echo "[$(date '+%F %T')] destination already at full size; verifying instead of re-sending"
else
  SSH "mkdir -p '$(dirname "$DEST")' && truncate -s $SZ '$DEST'" || { echo "FATAL: cannot preallocate $DEST"; exit 4; }
  CH=$(( (SZ + NSTREAM - 1) / NSTREAM ))
  T0=$(date +%s)
  pids=()
  for i in $(seq 0 $((NSTREAM-1))); do
    OFF=$(( i * CH )); LEN=$CH
    [ $(( OFF + LEN )) -gt "$SZ" ] && LEN=$(( SZ - OFF ))
    [ "$LEN" -le 0 ] && continue
    (
      dd if="$SRC" bs=4M iflag=skip_bytes,count_bytes skip=$OFF count=$LEN 2>/dev/null \
      | SSH "dd of='$DEST' bs=4M oflag=seek_bytes conv=notrunc seek=$OFF 2>/dev/null"
    ) & pids+=($!)
  done
  fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  T1=$(date +%s); EL=$(( T1 - T0 )); [ "$EL" -eq 0 ] && EL=1
  echo "[$(date '+%F %T')] transfer done in ${EL}s = $(awk -v s=$SZ -v e=$EL 'BEGIN{printf "%.1f", s/e/2^20}') MiB/s aggregate (stream_fail=$fail)"
  [ "$fail" -eq 0 ] || { echo "FATAL: at least one stream failed"; exit 5; }
fi

# --- 1. size ----------------------------------------------------------------
GOT=$(SSH "stat -c %s '$DEST'")
if [ "$GOT" != "$SZ" ]; then echo "FATAL size mismatch: dest=$GOT src=$SZ"; exit 6; fi
echo "[$(date '+%F %T')] size OK ($GOT)"

# --- 2. FULL-FILE sha256 on both disks --------------------------------------
echo "[$(date '+%F %T')] hashing both sides (full file, ~$(awk -v s=$SZ 'BEGIN{printf "%.0f", s/2^30}') GiB each)..."
SSH "sha256sum '$DEST' | cut -d' ' -f1" > /tmp/a04_dest_${STEP}_${NODE}.sha &
DP=$!
sha256sum "$SRC" | cut -d' ' -f1 > /tmp/a04_src_${STEP}.sha
wait $DP
SH_SRC=$(cat /tmp/a04_src_${STEP}.sha); SH_DST=$(cat /tmp/a04_dest_${STEP}_${NODE}.sha)
echo "  src(wzc1)  = $SH_SRC"
echo "  dest(zwfy6)= $SH_DST"
if [ "$SH_SRC" != "$SH_DST" ]; then echo "FATAL sha256 MISMATCH -- do NOT use this file"; exit 7; fi
echo "[$(date '+%F %T')] sha256 OK (full file, identical on both disks)"

# --- 3. zip entry count -----------------------------------------------------
ENT=$(SSH "/opt/conda/envs/torch-base/bin/python -c \"import zipfile;print(len(zipfile.ZipFile('$DEST').namelist()))\"" 2>&1 | tail -1)
if [ "$ENT" != "$EXPECT_ENTRIES" ]; then echo "FATAL zip entries=$ENT != $EXPECT_ENTRIES"; exit 8; fi
echo "[$(date '+%F %T')] zip entries OK ($ENT)"
echo "[$(date '+%F %T')] STAGED_AND_VERIFIED step${STEP} on .${NODE}: $DEST"
echo "STAGE_RESULT step=${STEP} node=${NODE} size=${SZ} sha256=${SH_SRC} entries=${ENT}"
