#!/usr/bin/env bash
# A04 SHALLOW RUNG LADDER — 1B keep13+fresh2 / keep14+fresh2, 5000 steps, seed 101.
#
# WHY THIS EXISTS. STATUS.json:pilot_one.pilot_two_status is BLOCKED because
# "a NEW pre-data doc [must] show a rung exists where NI can be OBSERVED TO
# ACCEPT; otherwise the gate can only ever confirm rejection". The blocker is a
# RUNG-SELECTION problem, not a variance problem:
#   * the only NI ACCEPT in A04 is full32_dolmino (ZERO structural damage);
#   * every damaged rung ever trained rejects -- 1B keep7+fresh2 (9L) and
#     keep12+fresh2 (14L, 27.0-90.4x sd_run, recovery 22.06-31.77%);
#   * keep12 was the LIGHTEST damaged 1B rung in existence (4/16 = 25% cut);
#   * shallower rungs have ZERO checkpoints on EITHER disk (verified 2026-08-13:
#     outputs/olmo2_probe2_7B_keep16fresh2/ holds only arch_meta.json on zwfy6
#     and does not exist on wzc1).
# So NI's discrimination curve has a hole between "recovery 22-31%, REJECT by
# 27-90 SE" and "zero damage, ACCEPT". This script fills it.
#
# PROTOCOL = Pilot One Stage B, VERBATIM. Every hyper-parameter below was read
# out of outputs/olmo2_probe2_1B_keep12f2_dolmino_stageB_seed101/step5000.pt's
# own `train_args` dict, not from any prose table. The ONLY two quantities that
# differ from Stage B are --keep_front_layers and (nothing else; seed is 101,
# the same seed as Stage B seed101).
#
# ARMS
#   KEEP=14 -> 14 inherited + 2 fresh = 16 layers, 2/16 = 12.5% of the base cut.
#              The LIGHTEST possible damaged rung: keep15+fresh2 would be 17
#              layers = DEEPER than the 16-layer base, at which point "damage" is
#              no longer a cut at all. NOTE 14+2 == base depth 16 -- SAME DEPTH,
#              still DAMAGED: base layers 14 and 15 are DISCARDED and replaced by
#              random-init Olmo2 layers. This is NOT a zero-damage control (that
#              would be --n_fresh_layers 0, i.e. the full32-style CPT arm).
#   KEEP=13 -> 13 inherited + 2 fresh = 15 layers, 3/16 = 18.75% cut. The
#              intermediate point, so the NI curve has FOUR rungs
#              (keep12 / keep13 / keep14 / zero-damage) instead of two.
#
# GATE0 (run 2026-08-13 18:27-18:30, 1 GPU, 20 steps, /tmp output, before any
# 8-GPU commitment): keep14 copied 157 == 3+11*14 tensors, fresh tail ids
# [14,15], max|model-base| = 0.000e+00, fresh post_attn_ln + q_norm all-ones,
# q_std 0.0200; keep13 copied 146 == 3+11*13, fresh ids [13,14], same asserts.
# Both reached step 20 and exited 0. There is NO degenerate branch at
# keep_front + n_fresh == base_layers: transplant_front() indexes the BASE
# state_dict by layer id (<keep_front) and the fresh set is
# range(keep, keep+n_fresh) computed on the NEW cfg, so 14+2 behaves exactly
# like 12+2 with two more inherited layers.
#
# DISK: zwfy6 ONLY. dolmino_now15b.npy is 126,907,244,672 B on zwfy6 vs
# 62,020,903,040 B on wzc1 -- same NAME, DIFFERENT CORPUS. Asserted below.
#
# SEED PLUMBING: requires post-ce5c298 (DistributedSampler(..., seed=args.seed)).
# Asserted POSITIVELY below and echoed into the progress log, per the Stage B
# convention. Pre-fix runs consume a byte-identical minibatch sequence, so a
# "seed" would move only the fresh-tail init.
#
# NODE BUDGET (2026-08-13 dispatch, hard):
#   ALLOWED  : .73 (28.85.35.73) and .82 (28.82.250.82) -- 8xH20, zwfy6
#   FORBIDDEN: .104 (paperC Qwen3-8B heal), .21 + LOCAL (SparseForge #246)
# The IP guard below refuses on the forbidden nodes rather than trusting the
# caller.
#
# Usage (ON the target node):  KEEP=14 bash scripts/_run_a04_shallow_ladder.sh
set -u

KEEP="${KEEP:?KEEP must be set (13 or 14)}"
SEED="${SEED:-101}"
W="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$W" || { echo "FATAL: cannot cd $W"; exit 3; }
PY="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"

LOG=logs/a04_shallow_keep${KEEP}_seed${SEED}.log
PROG=logs/a04_shallow_keep${KEEP}_seed${SEED}_progress.log

DATA=data/dolmino_now15b.npy
BASE=/apdcephfs_zwfy6/share_304376610/pighzliu_code/models/OLMo-2-0425-1B
OUT=outputs/olmo2_probe2_1B_keep${KEEP}f2_dolmino_shallow_seed${SEED}

note() { printf '[%s] %s\n' "$(date '+%m-%d %H:%M:%S')" "$*" | tee -a "$PROG"; }

# --- PREFLIGHT (fail-closed, no silent skips) --------------------------------

# 0) forbidden-node guard by IP (not by the caller's word)
HOSTIP=$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^28\.' | head -1)
note "host ip (28.x) = ${HOSTIP:-none}"
case "$HOSTIP" in
  28.83.24.104) note "FATAL .104 runs paperC Qwen3 heal; out of budget"; exit 11 ;;
  28.89.19.21)  note "FATAL .21 runs SparseForge #246; out of budget";  exit 11 ;;
esac

# 1) arm whitelist -- this script exists for exactly two rungs
case "$KEEP" in
  13|14) ;;
  *) note "FATAL KEEP=$KEEP not in the pre-registered shallow set {13,14}"; exit 5 ;;
esac

# 2) seed plumbing: trainer must be POST-ce5c298. POSITIVE assertion, echoed.
FIX_LINE=$(grep -n 'DistributedSampler(ds, shuffle=True, seed=args.seed)' scripts/train_olmo2_arch_probe2.py | head -1)
[ -z "$FIX_LINE" ] && { note "FATAL trainer missing ce5c298 fix (no 'seed=args.seed' on the DistributedSampler line)"; exit 4; }
note "PREFLIGHT-ASSERT trainer post-ce5c298: $FIX_LINE"
note "PREFLIGHT-ASSERT trainer md5: $(md5sum scripts/train_olmo2_arch_probe2.py | cut -d' ' -f1)"

# 3) assets present + correct sizes/shape
for f in "$DATA" "$BASE/config.json"; do
  [ -e "$f" ] || { note "FATAL missing asset: $f"; exit 7; }
done
DATA_SIZE=$(stat -c %s "$DATA")
if [ "$DATA_SIZE" != "126907244672" ]; then
  note "FATAL wrong dolmino corpus: $DATA is $DATA_SIZE B, expected 126907244672 (zwfy6 15.5M-row set)"
  note "         wzc1's same-named file is 62020903040 B -- do NOT run this on wzc1"
  exit 8
fi
NL=$("$PY" -c "import json;print(json.load(open('$BASE/config.json'))['num_hidden_layers'])")
if [ "$NL" != "16" ]; then
  note "FATAL base has num_hidden_layers=$NL, expected 16. The rung arithmetic (cut = (16-KEEP)/16) is void."
  exit 8
fi
note "PREFLIGHT-ASSERT base num_hidden_layers=$NL; cut = $((16-KEEP))/16; total depth = $((KEEP+2))"
note "preflight OK: KEEP=$KEEP SEED=$SEED dolmino=$(numfmt --to=iec $DATA_SIZE) base=$BASE"

# 4) GPUs must be clear (< 8 GiB held elsewhere)
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}')
if [ "$used" -gt 8000 ]; then
  note "FATAL ${used}MiB GPU memory still held; not launching KEEP=$KEEP"
  nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv | tee -a "$PROG"
  exit 9
fi
note "GPUs clear (${used}MiB held). Launching."

# 5) refuse to overwrite an existing run
if [ -d "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
  note "FATAL $OUT is non-empty. Refusing to overwrite -- move it aside for a true re-run."
  exit 10
fi

export OMP_NUM_THREADS=4
export WANDB_MODE=offline

# --- LAUNCH (flag set IDENTICAL to scripts/_run_a04_stageB.sh, KEEP swapped) --
# 5,000 steps, keep{13,14}+fresh2, lr 2e-5 -> 2e-6 on BOTH groups (Stage B is
# uniform-LR: lr == lr_inherited), warmup 150, seq_len 2048, batch_size 8,
# grad_accumulation_steps 2 (default) => eff_bs 128 at world=8, save_every 2500,
# milestone_every 5000 (default), keep_last_n 3 (default), log_every 20 (default),
# gradient_checkpointing 1, optimizer adamw (fp32), weight_decay 0.1, grad_clip 1.0.
note "cmd: keep${KEEP}+fresh2 fresh prune from OLMo-2-0425-1B base, 5000 steps, seed=$SEED"
"$PY" -m torch.distributed.run --standalone --nproc_per_node 8 \
  scripts/train_olmo2_arch_probe2.py \
    --model_path "$BASE" \
    --keep_front_layers "$KEEP" --n_fresh_layers 2 \
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
note "launched torchrun pid=$TRAIN_PID KEEP=$KEEP seed=$SEED"

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
print(f"  torch.load OK: {n} model tensors, step={sd.get('step','?')}, "
      f"keep_front={sd.get('keep_front_layers')}, n_fresh={sd.get('n_fresh_layers')}, "
      f"num_hidden_layers={sd.get('num_hidden_layers')}, seed={sd.get('seed')}")
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
  if ! pgrep -f "train_olmo2_arch_probe2.py.*keep${KEEP}f2_dolmino_shallow_seed${SEED}" >/dev/null 2>&1; then
    note "no training process for KEEP=$KEEP; exiting watcher"
    exit 0
  fi
  if complete; then
    note "stopping training; step5000.pt verified intact"
    # kill by PID from pgrep on the FULL --output_dir token (the run name appears
    # only inside --output_dir), and NEVER with a bare `pkill -f` pattern that
    # could match an eval process's --output_name.
    for p in $(pgrep -f "torch.distributed.run.*keep${KEEP}f2_dolmino_shallow_seed${SEED}" 2>/dev/null); do
      note "kill -TERM torchrun $p"; kill -TERM "$p" 2>/dev/null
    done
    sleep 30
    for p in $(pgrep -f "train_olmo2_arch_probe2.py.*keep${KEEP}f2_dolmino_shallow_seed${SEED}" 2>/dev/null); do
      note "kill -9 worker $p"; kill -9 "$p" 2>/dev/null
    done
    sleep 5
    note "stopped. ckpts: $(ls $OUT/step*.pt 2>/dev/null | tr '\n' ' ')"
    note "next: 4-axis eval (triviaqa/popqa/mmlu_content/nq_open) via code/a04_shallow_ladder_eval_driver.sh"
    exit 0
  fi
  sleep 60
done
note "loop budget exhausted (600 min); NOT killing; investigate"
exit 9
