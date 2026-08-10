#!/usr/bin/env bash
# A04 Pilot One Stage B — keep12+fresh2 1B, S=3 seeds x 5000 steps.
# Per PILOT_ONE_PREREG.md (commit 2ac0b5a, PRE-DATA). Seeds pinned {101,102,103}.
#
# WHY THIS ARM: keep7 is a confirmed constant-REJECT rung (52.4 B heal tokens
# recovers only 12-39% of intact residual, NI rejects by 6-9x on every axis),
# so a rule tested only there is uninformative. keep12 = 14/16 = 87.5% depth
# is the shallowest cut plausibly close to non-inferiority, i.e. the arm where
# NI-vs-PLATEAU disagreement can *fail* -- which is what makes the test
# falsifiable. Second choice keep10 (75%) if keep12 is constant-ACCEPT.
#
# LAUNCHABILITY: no pre-existing 1B keep12 ckpt is required. train_olmo2_arch_probe2.py's
# transplant_front() prunes from --model_path (HF base) directly; --resume_from
# is optional. Verified 2026-08-10 23:40 on zwfy6.
#
# DISK: zwfy6 ONLY. dolmino_now15b.npy is 62 GB on wzc1 vs 127 GB on zwfy6 --
# same name, different file. Mixing disks silently mixes corpora (design 8).
#
# SEED PLUMBING: relies on ce5c298 (DistributedSampler seed=args.seed).
# Verified on zwfy6 md5 284b286f90b526e4e8ad93a68e2a3b16 as of 2026-08-10 16:53.
# Pre-fix runs consume byte-identical minibatch sequence, i.e. "seeds" only vary
# fresh-block init -- not true run-to-run variance.
#
# STOP CONDITION: step5000.pt intact per the v3 guard (age>=120s, size stable,
# torch.load OK). Not the racy stop-loop in _run_a03_arm4_peaklr.sh.
#
# Usage: SEED=101 bash scripts/_run_a04_stageB.sh
set -u

SEED="${SEED:?SEED must be set (integer, 101|102|103)}"
W="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$W" || { echo "FATAL: cannot cd $W"; exit 3; }
PY="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"

LOG=logs/a04_stageB_keep12_seed${SEED}.log
PROG=logs/a04_stageB_keep12_seed${SEED}_progress.log

DATA=data/dolmino_now15b.npy
BASE=/apdcephfs_zwfy6/share_304376610/pighzliu_code/models/OLMo-2-0425-1B
OUT=outputs/olmo2_probe2_1B_keep12f2_dolmino_stageB_seed${SEED}

note() { printf '[%s] %s\n' "$(date '+%m-%d %H:%M:%S')" "$*" | tee -a "$PROG"; }

# --- PREFLIGHT (fail-closed, no silent skips) --------------------------------

# 1) seed plumbing: trainer must be POST-ce5c298
FIX_LINE=$(grep -n 'DistributedSampler(ds, shuffle=True, seed=args.seed)' scripts/train_olmo2_arch_probe2.py | head -1)
[ -z "$FIX_LINE" ] && { note "FATAL trainer missing ce5c298 fix (no 'seed=args.seed' on DistributedSampler line)"; exit 4; }
note "trainer post-ce5c298 OK: $FIX_LINE"

# 2) seed must be in the pre-registered set {101,102,103}
case "$SEED" in
  101|102|103) ;;
  *) note "FATAL seed=$SEED not in pre-registered set {101,102,103}"; exit 5 ;;
esac

# 3) assets present + correct sizes
for f in "$DATA" "$BASE/config.json"; do
  [ -e "$f" ] || { note "FATAL missing asset: $f"; exit 7; }
done
DATA_SIZE=$(stat -c %s "$DATA")
if [ "$DATA_SIZE" != "126907244672" ]; then
  note "FATAL wrong dolmino corpus: $DATA is $DATA_SIZE bytes, expected 126907244672 (zwfy6 15.5M-row set)"
  note "         wzc1's dolmino_now15b.npy is 62020903040 bytes -- do NOT run Stage B on wzc1"
  exit 8
fi
note "preflight OK: SEED=$SEED dolmino=$(numfmt --to=iec $DATA_SIZE)  base=$BASE"

# 4) GPUs must be clear (< 8 GiB held elsewhere)
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}')
if [ "$used" -gt 8000 ]; then
  note "FATAL ${used}MiB GPU memory still held; not launching seed=$SEED"
  nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv | tee -a "$PROG"
  exit 9
fi
note "GPUs clear (${used}MiB held). Launching."

# 5) refuse to overwrite an existing run
if [ -d "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
  note "FATAL $OUT is non-empty. Refusing to overwrite -- move it aside if you truly want a re-run."
  exit 10
fi

export OMP_NUM_THREADS=4
export WANDB_MODE=offline

# --- LAUNCH (config per PILOT_ONE_PREREG.md §3) ------------------------------
# 5,000 steps, keep12+fresh2, lr 2e-5 -> min_lr 2e-6, warmup 150 (matches Arm 3's
# recipe for comparability), seq_len 2048, batch_size 8 (eff_bs 128 with world=8),
# save/milestone every 2500 as prereg specified (--milestone_every 2500).
note "cmd: keep12+fresh2 fresh prune from OLMo-2-0425-1B base, 5000 steps, seed=$SEED"
"$PY" -m torch.distributed.run --standalone --nproc_per_node 8 \
  scripts/train_olmo2_arch_probe2.py \
    --model_path "$BASE" \
    --keep_front_layers 12 --n_fresh_layers 2 \
    --data_path "$DATA" \
    --output_dir "$OUT" \
    --max_steps 5000 \
    --lr 2e-5 --min_lr 2e-6 \
    --lr_inherited 2e-5 --min_lr_inherited 2e-6 \
    --seq_len 2048 --batch_size 8 \
    --warmup_steps 150 \
    --save_every 2500 \
    --seed "$SEED" \
    --gradient_checkpointing 1 \
  > "$LOG" 2>&1 &
TRAIN_PID=$!
note "launched torchrun pid=$TRAIN_PID seed=$SEED"

# --- v3-guarded stop when step5000.pt lands, then verify by torch.load -------
CK=$OUT/step5000.pt
STALE_S=120
TOL=65536

ref_size() {
  local f s
  for f in $OUT/step2500.pt; do
    if [ -f "$f" ]; then
      s=$(stat -c %s "$f" 2>/dev/null) || continue
      if [ $(( $(date +%s) - $(stat -c %Y "$f") )) -ge $STALE_S ]; then
        echo "$s"; return 0
      fi
    fi
  done
  return 1
}

complete() {
  [ -f "$CK" ] || return 1
  local age s1 s2 ref diff
  age=$(( $(date +%s) - $(stat -c %Y "$CK") ))
  if [ "$age" -lt $STALE_S ]; then note "ckpt only ${age}s old; waiting"; return 1; fi
  s1=$(stat -c %s "$CK" 2>/dev/null) || return 1
  if ref=$(ref_size); then
    diff=$(( s1 > ref ? s1 - ref : ref - s1 ))
    if [ "$diff" -gt $TOL ]; then
      note "REFUSE: size $s1 vs sibling ref $ref (diff ${diff}B > 64KiB)"
      return 1
    fi
  else
    note "no settled sibling yet; relying on size stability + torch.load"
  fi
  sleep 60
  s2=$(stat -c %s "$CK" 2>/dev/null) || return 1
  if [ "$s1" != "$s2" ]; then note "still growing ($s1 -> $s2); waiting"; return 1; fi
  if ! "$PY" - "$CK" <<'PYEOF' >>"$PROG" 2>&1
import sys, torch
p = sys.argv[1]
sd = torch.load(p, map_location="cpu", weights_only=False)
n = len(sd["model_state"]) if isinstance(sd, dict) and "model_state" in sd else -1
print(f"  torch.load OK: {n} model tensors, step={sd.get('step','?')}")
PYEOF
  then
    note "REFUSE: torch.load probe FAILED"
    return 1
  fi
  note "ckpt COMPLETE (size $s1, stable, torch.load OK)"
  return 0
}

LOOPS=0
while [ $LOOPS -lt 600 ]; do
  LOOPS=$((LOOPS+1))
  if ! pgrep -f "train_olmo2_arch_probe2.py.*stageB_seed${SEED}" >/dev/null 2>&1; then
    note "no training process for seed=$SEED; exiting watcher"
    exit 0
  fi
  if complete; then
    note "stopping training; step5000.pt verified intact"
    for p in $(pgrep -f "torch.distributed.run.*stageB_seed${SEED}" 2>/dev/null); do
      note "kill -TERM torchrun $p"; kill -TERM "$p" 2>/dev/null
    done
    sleep 30
    for p in $(pgrep -f "train_olmo2_arch_probe2.py.*stageB_seed${SEED}" 2>/dev/null); do
      note "kill -9 worker $p"; kill -9 "$p" 2>/dev/null
    done
    sleep 5
    note "stopped. ckpts: $(ls $OUT/step*.pt 2>/dev/null | tr '\n' ' ')"
    note "next: 4-axis eval (triviaqa/popqa/mmlu_content/nq_open) then plug into Stage A driver at S=3"
    exit 0
  fi
  sleep 60
done
note "loop budget exhausted (600 min); NOT killing; investigate"
exit 9
