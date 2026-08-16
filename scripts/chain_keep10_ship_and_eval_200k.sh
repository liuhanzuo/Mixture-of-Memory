#!/usr/bin/env bash
# Chain: when LOCAL's keep10 reaches step200000, SHIP the ckpt wzc1 -> zwfy6, then fire the
# Paper B ladder eval on an H20.
#
# WHY THIS IS TWO-PHASE AND THE OTHER RUNGS ARE NOT
# --------------------------------------------------
# keep8 trains on .82 and keep12 on .73 -- both zwfy6 -- so their step200000.pt lands on the
# same disk the eval must run on, and their chains are pure wait-then-eval. keep10 is the
# odd rung out: it trains on LOCAL (wzc1). Measured 2026-08-17, zwfy6's copy of
# outputs/olmo2_probe2_7B_keep10fresh2/ stops at step90000 (dated 08-12) while wzc1 is past
# 193000, so step200000.pt will be wzc1-ONLY.
#
# It cannot simply be evaluated where it lands. scripts/eval_paperb_ladder_200k.sh:85 sets
# REQUIRE_SM=9.0 and refuses any other capability, because Table 4's batteries are
# single-protocol H20 and core6 carries a measured 0.03-0.16 pp cross-arch floor on
# bit-identical weights. LOCAL is sm_100. So the ckpt has to move.
#
# THE TRANSFER IS CHEAP -- MEASURED, NOT ASSUMED
# ----------------------------------------------
# CLAUDE.md warns cross-disk copies are "12 MB/s single stream ... about 42 hours" for two
# 45.4 GiB endpoints. That note is self-inconsistent (45.4 GiB x2 at 12 MB/s is ~2.3 h, not
# 42 h) so I measured it instead of trusting either number: a 2 GiB probe, wzc1 -> .73,
# scp -O, took 112 s = 19.2 MB/s, md5 verified identical on both ends. At that rate this
# single 39.01 GB ckpt is ~34 min. That is small against the ~2 h eval it unblocks, and it
# runs while .73 is still finishing keep12, so it costs no GPU time at all.
#
# ORDERING
# --------
# Ship as soon as the ckpt is COMPLETE (size stable), which happens ~17 h before .73 frees.
# By the time keep12 finishes, the file is already in place and the eval starts immediately.
# The eval itself is only launched once .73's cards are actually free -- this script does not
# compete with keep12 and never touches it.
set -uo pipefail

WZ="${WZ:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}"
ZW="${ZW:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
ARM="${ARM:-keep10}"
EXPECT_STEP="${EXPECT_STEP:-200000}"
SRC="$WZ/outputs/olmo2_probe2_7B_${ARM}fresh2/step${EXPECT_STEP}.pt"
DST_DIR="$ZW/outputs/olmo2_probe2_7B_${ARM}fresh2"
DST="$DST_DIR/step${EXPECT_STEP}.pt"
NODE="${NODE:-28.85.35.73}"
PW="${PW:-$WZ/configs/password_h20_853573.txt}"
POLL="${POLL:-300}"
MAX_WAIT_H="${MAX_WAIT_H:-24}"
LOG="${LOG:-$WZ/logs/chain_${ARM}_ship_eval_200k.log}"

say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
ssh_n() { timeout 300 sshpass -f "$PW" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 \
            -o PreferredAuthentications=password "root@$NODE" "$@"; }

cd "$WZ" || { echo "FATAL: cannot cd $WZ"; exit 2; }
say "=== ${ARM} ship+eval chain start ==="
say "src=$SRC"
say "dst=root@${NODE}:$DST"

deadline=$(( $(date +%s) + MAX_WAIT_H * 3600 ))
prev=-1; stable=0

# ---- PHASE 1: wait for a COMPLETE local ckpt --------------------------------
while :; do
  if [ "$(date +%s)" -ge "$deadline" ]; then
    say "FATAL: ${MAX_WAIT_H}h elapsed, no complete ckpt. NOT shipping, NOT evaluating."
    exit 3
  fi
  if [ -f "$SRC" ]; then
    sz=$(stat -c %s "$SRC" 2>/dev/null || echo 0)
    if [ "$sz" -gt 0 ] && [ "$sz" -eq "$prev" ]; then
      stable=$(( stable + 1 ))
      say "ckpt present, size stable at $sz bytes ($stable consecutive polls)"
      [ "$stable" -ge 2 ] && break
    else
      stable=0
      say "ckpt present, size $sz bytes (still growing)"
    fi
    prev=$sz
  else
    st=$(tail -c 2000 logs/olmo2_7B_${ARM}fresh2_resume200k_local_0815.log 2>/dev/null \
         | tr '\r' '\n' | grep -aoE 'step [0-9]+/[0-9]+' | tail -1)
    say "ckpt absent; trainer at [${st:-unknown}]"
  fi
  sleep "$POLL"
done

# ---- PHASE 2: ship, then VERIFY BY HASH -------------------------------------
say "shipping $(stat -c %s "$SRC") bytes to $NODE (expect ~34 min at the measured 19.2 MB/s)"
ssh_n "mkdir -p '$DST_DIR'" || { say "FATAL: cannot mkdir on $NODE"; exit 4; }
t0=$(date +%s)
timeout 14400 sshpass -f "$PW" scp -O -o StrictHostKeyChecking=no -o ConnectTimeout=15 \
  -o PreferredAuthentications=password "$SRC" "root@$NODE:$DST"
rc=$?
t1=$(date +%s)
if [ "$rc" -ne 0 ]; then
  say "FATAL: scp rc=$rc after $((t1-t0))s. NOT evaluating a partial ckpt."
  exit 4
fi
say "scp done in $((t1-t0))s"

# Hash BOTH ends. A size match is not enough: a truncated-then-padded transfer or a silent
# CephFS read error both preserve size. This is the same check the 2 GiB probe passed.
say "hashing both ends (this reads 39 GB twice; a few minutes)"
h_src=$(md5sum "$SRC" | awk '{print $1}')
h_dst=$(ssh_n "md5sum '$DST' 2>/dev/null | awk '{print \$1}'" | tail -1 | tr -d '\r')
say "src md5=$h_src"
say "dst md5=$h_dst"
if [ -z "$h_dst" ] || [ "$h_src" != "$h_dst" ]; then
  say "FATAL: md5 MISMATCH (or unreadable). Removing the bad copy; NOT evaluating."
  ssh_n "rm -f '$DST'"
  exit 5
fi
say "md5 MATCH -- ckpt is intact on zwfy6"

# ---- PHASE 3: wait for .73 to free, then eval -------------------------------
say "waiting for $NODE cards to free (keep12 must finish first; this chain never kills it)"
clean=0
while :; do
  if [ "$(date +%s)" -ge "$deadline" ]; then
    say "NOTE: deadline reached while waiting for cards. The ckpt IS shipped and verified,"
    say "      so the eval can be launched by hand:"
    say "      ARM=$ARM PROJECT_ROOT=$ZW bash scripts/eval_paperb_ladder_200k.sh"
    exit 6
  fi
  n=$(ssh_n 'nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .' | tail -1 | tr -d '\r')
  case "$n" in ''|*[!0-9]*) say "could not read PID count from $NODE ('$n'); retrying"; sleep "$POLL"; continue ;; esac
  if [ "$n" -eq 0 ]; then
    clean=$(( clean + 1 ))
    say "$NODE clear: 0 compute PIDs ($clean consecutive polls)"
    [ "$clean" -ge 2 ] && break
  else
    clean=0
    say "$NODE busy: $n compute PIDs"
  fi
  sleep "$POLL"
done

say "launching the ladder eval for $ARM on $NODE"
ssh_n "cd '$ZW' && ARM=$ARM EXPECT_STEP=$EXPECT_STEP PROJECT_ROOT='$ZW' \
       setsid nohup bash scripts/eval_paperb_ladder_200k.sh \
       > logs/ladder200k_eval_${ARM}.log 2>&1 < /dev/null & sleep 5; \
       pgrep -af eval_paperb_ladder_200k | head -1"
say "=== eval dispatched. Watch $ZW/logs/ladder200k_eval_${ARM}.log on $NODE ==="
