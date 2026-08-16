#!/usr/bin/env bash
# Chain watcher: fire the B12 pilot pair (rung P, then control Dctl) the moment LOCAL's
# keep10 training finishes and frees all 8 sm_100 cards.
#
# WHY THIS EXISTS
# ---------------
# LOCAL is ~44 min from finishing olmo2_probe2_7B_keep10fresh2 at step200000 (measured
# 1.4148 s/step amortised over four consecutive 500-step ckpt intervals -> ETA 04:35).
# Nothing was waiting for that: `pgrep -af 'watch|chain|wait'` on LOCAL returns only the
# editor's file watcher. Without this, 8 B200 cards idle until a heartbeat notices.
#
# WHY THE B12 PILOT AND NOT keep10's OWN EVAL
# -------------------------------------------
# keep10's eval CANNOT run here. scripts/eval_paperb_ladder_200k.sh:85 sets REQUIRE_SM=9.0
# and dies on any other capability, because Table 4's batteries are single-protocol H20 and
# core6 has a measured 0.03-0.16 pp cross-arch floor on bit-identical weights. LOCAL is
# sm_100. So the keep10 eval belongs on an H20 (.73 frees first, ~19:57) and is chained
# separately; this node's correct successor is the one task that REQUIRES sm_100/wzc1.
#
# B12 is the only ready_gpu proposal (proposal/ready_queue.py), and its gpu_policy authorises
# exactly this: "the authorised spend is the PILOT PAIR ONLY (rung P + Dctl, 1.46 GPU-h) on
# sm_100/wzc1 (.212 or LOCAL); rungs Q/R/S are NOT authorised by this document".
# Both rungs had their pre-registered constants verified on CPU first
# (evidence/g0_leg2_rungP_selfcheck_20260817.json): P 325,844,992 / 55.0316%,
# Dctl 326,098,944 / 55.0355%, both exact.
#
# WHAT IT WAITS FOR
# -----------------
# NOT the log line, and NOT the ckpt file. It waits for the CARDS: zero compute-app PIDs on
# this node across two consecutive polls. That is the right trigger here because the driver's
# own P6 refuses to launch if ANY GPU on the node holds memory -- so triggering early would
# just make the driver die. Two consecutive clean polls also avoids the window where a
# just-exited trainer has released PIDs but not yet its memory.
#
# WHAT IT DOES NOT DO
# -------------------
# It does not kill anything and never touches the running trainer -- it only waits for the
# trainer to exit on its own. It runs rung P, and only if P succeeds does it run Dctl; a
# failed P means the pipeline is broken and running the control would waste the other 0.73
# GPU-h. It does NOT run Q/R/S under any outcome: those need the pilot verdict per
# kill_gate.clause_4_KILL_ON_PILOT.
set -uo pipefail

REPO="${REPO:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}"
POLL="${POLL:-120}"
MAX_WAIT_H="${MAX_WAIT_H:-8}"
LOG="${LOG:-$REPO/logs/chain_b12_pilot_local.log}"
# Guard: this watcher is only correct on a wzc1 sm_100 box. Checked at start, not assumed.
REQUIRE_SM="${REQUIRE_SM:-10.0}"

say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

cd "$REPO" || { echo "FATAL: cannot cd $REPO"; exit 2; }

CAPS=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sort -u | tr '\n' ',')
if [ "$CAPS" != "${REQUIRE_SM}," ]; then
  say "FATAL: this node reports compute_cap=[${CAPS%,}] but the B12 pilot requires ${REQUIRE_SM}"
  say "       (sm_100/wzc1). The 5B ckpt and both anchors are wzc1-resident and the completed"
  say "       union-9 arms were scored on sm_100; a different arch is not comparable."
  exit 2
fi
say "=== B12 pilot chain watcher start on $(hostname) (compute_cap ${CAPS%,}) ==="
say "waiting for ALL 8 cards to free; poll=${POLL}s max_wait=${MAX_WAIT_H}h"

deadline=$(( $(date +%s) + MAX_WAIT_H * 3600 ))
clean=0

while :; do
  if [ "$(date +%s)" -ge "$deadline" ]; then
    say "FATAL: ${MAX_WAIT_H}h elapsed and the node never freed. NOT launching."
    say "       Check whether keep10 stalled or a new job took the cards."
    exit 3
  fi

  npid=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . )
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
        | awk '{s+=$1} END{print s+0}')

  if [ "$npid" -eq 0 ] && [ "$mem" -lt 4096 ]; then
    clean=$(( clean + 1 ))
    say "node clear: 0 compute PIDs, ${mem} MiB held (${clean} consecutive polls)"
    [ "$clean" -ge 2 ] && break
  else
    [ "$clean" -ne 0 ] && say "not clear after all (pids=$npid mem=${mem} MiB); resetting"
    clean=0
    step=$(tail -c 2000 logs/olmo2_7B_keep10fresh2_resume200k_local_0815.log 2>/dev/null \
           | tr '\r' '\n' | grep -aoE 'step [0-9]+/[0-9]+' | tail -1)
    say "busy: ${npid} compute PIDs, ${mem} MiB held; trainer at [${step:-unknown}]"
  fi
  sleep "$POLL"
done

say "cards confirmed free. Launching the B12 PILOT PAIR (rung P, then Dctl)."
say "authorised by proposal/backlog/B12-slorb-rank-efficiency/STATUS.json .gpu_policy"

# HF proxy. MEASURED 2026-08-17, before this watcher ever fired: without it the driver's own
# P8 preflight FAILS HARD -- probe_union9_datasets.py cannot reach huggingface.co
# ("Network is unreachable", 5 retries per task) and P8 is a `die`, not a warning. That die
# happens at line ~246, i.e. AFTER P6 has confirmed the node is ours, so the pilot would have
# burned the free-node window and exited with nothing. P7's hub probe returns 000 without the
# proxy and 200 with it, but P7 only logs a WARNING, so it would not have stopped anything.
# With the proxy exported the preflight passes 9/9 tasks at exactly the expected n
# (boolq 3270, rte 277, hellaswag 10042, race 1045, piqa 1838, winogrande 1267,
#  arc_easy 2376, arc_challenge 1172, openbookqa 500) -- verified by running it here on CPU.
# Values from CLAUDE.md's proxy section.
export http_proxy="http://hy-proxy.woa.com:3128"
export https_proxy="$http_proxy"
export all_proxy="$http_proxy"
export no_proxy="mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,localhost,127.0.0.1,.oa.com,.woa.com,.local"
say "HF proxy exported ($http_proxy) -- required by P8, measured to fail without it"

for RG in P Dctl; do
  say "---- launching rung $RG (DRY_RUN=0) ----"
  RUNG="$RG" DRY_RUN=0 GPUS=0,1,2,3 GPU0=0 \
    bash scripts/launch_slorb_rank_sweep.sh >> "$REPO/logs/slorb_pilot_${RG}_launch.out" 2>&1
  rc=$?
  say "rung $RG finished rc=$rc"
  if [ "$rc" -ne 0 ]; then
    say "STOPPING: rung $RG failed (rc=$rc). Not running the remaining rung -- a broken"
    say "  pipeline would spend the other 0.73 GPU-h producing an uninterpretable cell."
    say "  See logs/slorb_rank_sweep_${RG}.log for the stage that died."
    exit 4
  fi
done

say "=== B12 PILOT PAIR COMPLETE (P + Dctl). Rungs Q/R/S remain UNAUTHORISED ==="
say "next: score both cells against kill_gate pass_bar_union9_primary=60.64 (tau=1.79 pp),"
say "      then write the verdict to the proposal's own STATUS.json before any further GPU."
