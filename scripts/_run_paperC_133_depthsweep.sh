#!/usr/bin/env bash
# Paper C task #133 — P-C1 DEPTH SWEEP orchestrator (run ON .82, setsid nohup).
#
# Turns the single #92 keep14 point into a CURVE: freeze-graft (A4, hero) vs
# depth-matched from-scratch (A3, control) at keep in {20,24,28} with fresh=2,
# then SQuAD dev EM/F1 on each. The existing #92 keep14 pair (A4=0.2930 EM /
# A3=0.2605 EM) is the 4th depth point and is NOT retrained — it is only reused
# in the final table.
#
# Recipe is pinned to #92 for apples-to-apples comparability:
#   data=data/squad_sft_olmo2_2048_train.npy  MAX_STEPS=1000  SEQ_LEN=2048
#   eff_bs = BS*GA*nGPU == 128 for EVERY run (never relaxed -- see below)
#   A4: --freeze_front, lr=1e-4 fresh / 2e-5 inherited
#   A3: --from_scratch, lr=3e-4 both
#   eval: scripts/eval_paperC_squad_emf1.py, n=2000, chat_template=False,
#         add_bos=0, greedy, first-line completion, SQuAD-normalised EM+token-F1
#
# ⚠️ eff_bs IS NEVER REDUCED. Deeper A3 arms (all params trainable, fp32 AdamW)
# may not fit 95GiB H20. We retry with a BS/GA ladder that keeps BS*GA*nGPU==128
# exactly (4x4 -> 2x8 -> 1x16); if even BS=1 OOMs the arm is recorded
# OOM-BLOCKED and the chain continues. Silently shrinking eff_bs would break
# comparability and quietly invalidate the curve, so it is not an option.
#
# ⚠️ NODE .82 IS ON zwfy6, NOT wzc1 (CLAUDE.md's "all 5 nodes share wzc1" does
# NOT hold for .82). ROOT below is the zwfy6 checkout, which already carries the
# #92 data / base model / outputs / evidence_squad_label_prior.
#
# Usage (on .82):
#   cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
#   setsid nohup bash scripts/_run_paperC_133_depthsweep.sh \
#     > logs/paperC_133_depthsweep.log 2>&1 &
set -uo pipefail

ROOT="${ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$ROOT" || { echo "FATAL: cannot cd $ROOT"; exit 1; }
PY="${PY:-/opt/conda/envs/torch-base/bin/python}"
BASE="${BASE:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/models/OLMo-2-1124-7B}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
STEPS="${STEPS:-1000}"
KEEPS="${KEEPS:-20 24 28}"
FRESH="${FRESH_N:-2}"
SAVE_EVERY="${SAVE_EVERY:-1000}"   # only final-ish ckpts; each is 30-60GB
PORT_BASE="${PORT_BASE:-29570}"
STATUS="$ROOT/logs/paperC_133_status.tsv"

export WANDB_MODE=offline TOKENIZERS_PARALLELISM=false

log(){ echo "[orch133 $(date '+%F %T')] $*"; }
note(){ printf '%s\t%s\t%s\n' "$(date '+%F %T')" "$1" "$2" >> "$STATUS"; }

log "ROOT=$ROOT"
log "PY=$PY"
log "BASE=$BASE"
log "sweep: A4/A3 x keep{$KEEPS}+fresh$FRESH, steps=$STEPS, eff_bs pinned 128"

# ---- preflight: everything the runs depend on must exist BEFORE we start ----
fail=0
for f in "$ROOT/data/squad_sft_olmo2_2048_train.npy" \
         "$ROOT/data/squad_val.jsonl" \
         "$ROOT/scripts/run_paperC_pc1.sh" \
         "$ROOT/scripts/train_olmo2_arch_probe2.py" \
         "$ROOT/scripts/eval_paperC_squad_emf1.py" \
         "$BASE/config.json"; do
  [ -e "$f" ] || { log "PREFLIGHT MISSING: $f"; fail=1; }
done
[ "$fail" = 0 ] || { log "FATAL preflight failed"; exit 1; }
"$PY" -c 'import torch;assert torch.cuda.device_count()==8,torch.cuda.device_count()' \
  || { log "FATAL: need 8 visible GPUs"; exit 1; }
log "preflight OK"
: > "$STATUS"

# ---------------------------------------------------------------------------
# one training arm, with an eff_bs-preserving BS/GA ladder
# ---------------------------------------------------------------------------
run_arm(){            # $1=ARM(A4|A3)  $2=KEEP  $3=port
  local arm="$1" keep="$2" port="$3"
  local tag="${arm}_keep${keep}fresh${FRESH}"
  local out="$ROOT/outputs/paperC_pc1_squad_${tag}"
  local trainlog="$ROOT/logs/paperC_pc1_squad_${tag}.log"

  if [ -f "$out/final.pt" ]; then
    log "SKIP $tag (final.pt already present)"; note "$tag" "SKIP_EXISTING"; return 0
  fi

  # every pair keeps BS*GA*8 == 128
  for bsga in "4 4" "2 8" "1 16"; do
    set -- $bsga; local bs="$1" ga="$2"
    local eff=$(( bs * ga * 8 ))
    log "=== $tag : BS=$bs GA=$ga eff_bs=$eff (must be 128) ==="
    if [ "$eff" -ne 128 ]; then log "BUG: eff_bs=$eff != 128, refusing"; note "$tag" "BUG_EFFBS"; return 1; fi

    env ARM="$arm" KEEP="$keep" FRESH="$FRESH" GPUS="$GPUS" PORT="$port" \
        MAX_STEPS="$STEPS" BS="$bs" GA="$ga" SAVE_EVERY="$SAVE_EVERY" \
        PROJECT_ROOT="$ROOT" PYTHON_BIN="$PY" MODEL_PATH="$BASE" FOREGROUND=1 \
        bash scripts/run_paperC_pc1.sh

    if [ -f "$out/final.pt" ]; then
      local layers=$(( keep + FRESH ))
      log "OK $tag final.pt present (eff_bs=$eff BS=$bs GA=$ga layers=$layers)"
      note "$tag" "DONE eff_bs=$eff BS=$bs GA=$ga layers=$layers"
      grep -hE '\[freeze\]|model params|\[optim\] group|\[sanity\]' "$trainlog" | tail -8
      grep -hE '\[step ' "$trainlog" | tail -1
      return 0
    fi

    if grep -qiE 'OutOfMemoryError|CUDA out of memory' "$trainlog" 2>/dev/null; then
      log "$tag OOM at BS=$bs GA=$ga -> next rung (eff_bs stays 128)"
      note "$tag" "OOM_RETRY BS=$bs GA=$ga"
      pkill -9 -f 'train_olmo2_arch_probe2' 2>/dev/null; sleep 20
      continue
    fi
    log "$tag failed WITHOUT an OOM signature -> not a memory problem, see $trainlog"
    note "$tag" "FAIL_NON_OOM BS=$bs GA=$ga"
    grep -hiE 'Error|Traceback|assert' "$trainlog" | tail -12
    return 1
  done

  log "$tag OOM-BLOCKED: even BS=1 GA=16 (eff_bs=128) does not fit 95GiB H20."
  log "$tag NOT retried at reduced eff_bs on purpose (would break comparability)."
  note "$tag" "OOM_BLOCKED_all_rungs"
  return 1
}

# ---------------------------------------------------------------------------
# eval one finished arm (identical protocol to #92)
# ---------------------------------------------------------------------------
eval_arm(){           # $1=ARM  $2=KEEP
  local arm="$1" keep="$2"
  local tag="${arm}_keep${keep}fresh${FRESH}"
  local ck="$ROOT/outputs/paperC_pc1_squad_${tag}/final.pt"
  local name="${tag}"
  [ -f "$ck" ] || { log "eval SKIP $tag (no final.pt)"; return 0; }
  if [ -f "$ROOT/evidence_squad_label_prior/$name/summary.json" ]; then
    log "eval SKIP $tag (summary.json exists)"; return 0
  fi
  log "=== eval $tag ==="
  CUDA_VISIBLE_DEVICES=0 "$PY" scripts/eval_paperC_squad_emf1.py \
      --ckpt "$ck" --base_model "$BASE" --tokenizer "$BASE" \
      --val_path data/squad_val.jsonl --output_name "$name" --batch_size 32 \
      >> "$ROOT/logs/paperC_133_eval_${name}.log" 2>&1
  "$PY" scripts/eval_paperC_squad_emf1.py --merge --output_name "$name" \
      >> "$ROOT/logs/paperC_133_eval_${name}.log" 2>&1
  if [ -f "$ROOT/evidence_squad_label_prior/$name/summary.json" ]; then
    note "$tag" "EVAL_DONE"
    grep -hE 'pruned\]|shard 0|merge\]' "$ROOT/logs/paperC_133_eval_${name}.log" | tail -3
  else
    log "eval $tag produced no summary.json"; note "$tag" "EVAL_FAIL"
    tail -12 "$ROOT/logs/paperC_133_eval_${name}.log"
  fi
}

# ---------------------------------------------------------------------------
# 1. TRAIN — hero first at every depth (cheap + the actual claim), then control
# ---------------------------------------------------------------------------
i=0
for keep in $KEEPS; do
  i=$((i+1)); run_arm A4 "$keep" $(( PORT_BASE + i ))
done
for keep in $KEEPS; do
  i=$((i+1)); run_arm A3 "$keep" $(( PORT_BASE + i ))
done

# ---------------------------------------------------------------------------
# 2. EVAL every arm that produced a checkpoint
# ---------------------------------------------------------------------------
log "=== SQuAD dev EM/F1 evals (n=2000, base protocol) ==="
for keep in $KEEPS; do eval_arm A4 "$keep"; done
for keep in $KEEPS; do eval_arm A3 "$keep"; done

# ---------------------------------------------------------------------------
# 3. DEPTH CURVE — new points + the reused #92 keep14 point
# ---------------------------------------------------------------------------
log "=== P-C1 DEPTH CURVE (SQuAD dev EM/F1) ==="
FRESH_N="$FRESH" KEEPS="$KEEPS" "$PY" - <<'PY'
import json, os
root = os.environ.get("ROOT", ".")
fresh = int(os.environ.get("FRESH_N", "2"))
keeps = [int(k) for k in os.environ.get("KEEPS", "20 24 28").split()]
res = os.path.join(root, "evidence_squad_label_prior")

def get(name):
    p = os.path.join(res, name, "summary.json")
    if not os.path.isfile(p):
        return None
    d = json.load(open(p))
    return d.get("em"), d.get("f1"), d.get("n")

def layers_of(ck):
    p = os.path.join(root, "outputs", f"paperC_pc1_squad_{ck}", "arch_meta.json")
    if os.path.isfile(p):
        m = json.load(open(p))
        return m.get("num_hidden_layers"), m.get("n_trainable")
    return None, None

# keep14 = the #92 point, reused verbatim (NOT retrained)
rows = [(14, "A4_hero", "A3_fromscratch", "#92")]
rows += [(k, f"A4_keep{k}fresh{fresh}", f"A3_keep{k}fresh{fresh}", "#133") for k in keeps]

print(f"{'keep':>5s} {'L':>3s} {'src':>5s} "
      f"{'A4_EM':>7s} {'A4_F1':>7s} {'A3_EM':>7s} {'A3_F1':>7s} {'dEM(A4-A3)':>11s} {'dF1':>8s}")
for keep, a4, a3, src in rows:
    r4, r3 = get(a4), get(a3)
    L, _ = layers_of(a4)
    L = L if L else keep + fresh
    f = lambda v: f"{v:7.4f}" if v is not None else "     --"
    e4 = r4[0] if r4 else None; f4 = r4[1] if r4 else None
    e3 = r3[0] if r3 else None; f3 = r3[1] if r3 else None
    dem = f"{(e4-e3)*100:+10.2f}" if (e4 is not None and e3 is not None) else "        --"
    df1 = f"{(f4-f3)*100:+7.2f}" if (f4 is not None and f3 is not None) else "     --"
    print(f"{keep:5d} {L:3d} {src:>5s} {f(e4)} {f(f4)} {f(e3)} {f(f3)} {dem} {df1}")
print("\ndEM/dF1 in percentage points. '--' = not available (see status tsv: "
      "OOM_BLOCKED / FAIL / still running).")
PY

log "=== status ledger ($STATUS) ==="
cat "$STATUS" 2>/dev/null
log "=== ORCHESTRATOR #133 DONE ==="
