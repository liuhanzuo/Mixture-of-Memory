#!/usr/bin/env bash
# Paper C task #133b — A3 (from-scratch depth-matched) completion on 8×B200.
#
# WHY: task #133 on .82 (H20) finished the A4 hero side at keep∈{20,24,28}
# (EM=0.3440/0.3560/0.4190) but the A3 from-scratch CONTROL side OOM_BLOCKED on
# all three depths (ledger:
# /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/logs/paperC_133_status.tsv).
# 7B fp32 AdamW doesn't fit 95 GiB H20 even at BS=1 GA=16 (eff_bs pinned 128).
# B200 (183 GiB/card) fits it. A3 is the essential control that makes the A4
# freeze-graft claim causal (without A3 the curve is single-arm).
#
# HOW: reuses scripts/run_paperC_pc1.sh UNMODIFIED for each depth:
#   * ARM=A3 KEEP={20,24,28} FRESH=2 (override via env, exactly like #133)
#   * A3 -> --from_scratch (ignore base weights, random-init all layers, single-LR)
#   * lr = lr_inherited = 3e-4 (already hard-coded for ARM=A3)
#   * eff_bs = BS * GA * nGPU pinned to 128 (comparability with A4 #133 at each
#     depth and with #92/#134). We use BS=2 GA=8 on 8 B200s -> 128 exactly.
#   * MAX_STEPS=1000 SEQ_LEN=2048 SEED=42 -- identical to #92/#133/#134.
#   * OUTPUT_DIR = outputs/paperC_pc1_squad_A3_keep{K}fresh2 (SAME names #133
#     used on zwfy6; but those OOM'd before writing any final.pt / step*.pt, so
#     re-using is safe. Launcher REFUSES if any dir already carries a real
#     checkpoint on this disk).
#   * chain the three depths sequentially on all 8 cards, followed by eval.
#     Fault-tolerant per depth (record and continue).
#
# NOT LAUNCHED. This launcher is written for MAIN to invoke the moment .252
# (or LOCAL) frees. Guard rails:
#   * Refuses to start any depth whose OUT_DIR already carries final.pt / step*.pt.
#   * Preflights data/base_model/eval_script presence + 8 visible GPUs.
#   * Auto-tokenises the packed .npy shard if missing on wzc1.
#
# Usage:
#   setsid nohup bash scripts/_run_paperC_133b_a3b200.sh \
#     > logs/paperC_133b_a3b200.log 2>&1 &
#
# Env overrides (all optional):
#   PROJECT_ROOT   default /apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory
#   PYTHON_BIN     default /opt/conda/envs/torch-base/bin/python (fallback .venv/bin/python)
#   BASE           default $PROJECT_ROOT/../models/OLMo-2-1124-7B
#   GPUS           default 0,1,2,3,4,5,6,7
#   PORT_BASE      default 29590
#   MAX_STEPS      default 1000
#   BS/GA          default 2/8   (eff_bs=128 on 8 GPUs)
#   OPT            default adamw (bnb8bit fallback available)
#   SEQ_LEN        default 2048
#   SEED           default 42
#   KEEPS          default "20 24 28"
#   FRESH_N        default 2
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT" || { echo "FATAL: cannot cd $PROJECT_ROOT"; exit 1; }

if [ -n "${PYTHON_BIN:-}" ]; then
  :
elif [ -x /opt/conda/envs/torch-base/bin/python ]; then
  PYTHON_BIN=/opt/conda/envs/torch-base/bin/python
elif [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
  PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
else
  echo "FATAL: no conda torch-base nor .venv python found"; exit 1
fi

BASE="${BASE:-/apdcephfs_wzc1/share_304376610/pighzliu_code/models/OLMo-2-1124-7B}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
PORT_BASE="${PORT_BASE:-29590}"
MAX_STEPS="${MAX_STEPS:-1000}"
SEQ_LEN="${SEQ_LEN:-2048}"
SEED="${SEED:-42}"
BS="${BS:-2}"
GA="${GA:-8}"
OPT="${OPT:-adamw}"
KEEPS="${KEEPS:-20 24 28}"
FRESH_N="${FRESH_N:-2}"

DATA_JSONL="$PROJECT_ROOT/data/squad_train.jsonl"
DATA_NPY="$PROJECT_ROOT/data/squad_sft_olmo2_2048_train.npy"
VAL_PATH="$PROJECT_ROOT/data/squad_val.jsonl"

LOG_FILE="${LOG_FILE:-$PROJECT_ROOT/logs/paperC_133b_a3b200.log}"
STATUS="$PROJECT_ROOT/logs/paperC_133b_status.tsv"

mkdir -p "$PROJECT_ROOT/logs"
: > "$STATUS"

log(){ echo "[paperC_133b $(date '+%F %T')] $*" | tee -a "$LOG_FILE"; }
note(){ printf '%s\t%s\t%s\n' "$(date '+%F %T')" "$1" "$2" >> "$STATUS"; }

log "PROJECT_ROOT=$PROJECT_ROOT"
log "PYTHON_BIN=$PYTHON_BIN"
log "BASE=$BASE"
log "GPUS=$GPUS PORT_BASE=$PORT_BASE MAX_STEPS=$MAX_STEPS SEQ_LEN=$SEQ_LEN"
log "KEEPS=$KEEPS FRESH_N=$FRESH_N BS=$BS GA=$GA OPT=$OPT SEED=$SEED"

# ---- 0. preflight ------------------------------------------------------------
fail=0
for f in "$BASE/config.json" \
         "$PROJECT_ROOT/scripts/run_paperC_pc1.sh" \
         "$PROJECT_ROOT/scripts/train_olmo2_arch_probe2.py" \
         "$PROJECT_ROOT/scripts/eval_paperC_squad_emf1.py" \
         "$VAL_PATH"; do
  [ -e "$f" ] || { log "PREFLIGHT MISSING: $f"; fail=1; }
done
[ "$fail" = 0 ] || { log "FATAL preflight failed"; exit 1; }
CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON_BIN" -c \
  'import torch;n=torch.cuda.device_count();assert n>=8,f"need 8 GPUs, have {n}"' \
  || { log "FATAL: 8 GPUs not visible via CUDA_VISIBLE_DEVICES=$GPUS"; exit 1; }

# refuse to clobber any depth that already carries checkpoints (defence-in-depth
# against re-launching after a partial success)
for keep in $KEEPS; do
  tag="A3_keep${keep}fresh${FRESH_N}"
  od="$PROJECT_ROOT/outputs/paperC_pc1_squad_${tag}"
  if [ -f "$od/final.pt" ]; then
    log "REFUSE: $od/final.pt already exists — remove or rename before rerun."
    note "$tag" "REFUSED_EXISTING_FINAL"; exit 1
  fi
  existing_step=$(ls "$od"/step*.pt 2>/dev/null | head -1 || true)
  if [ -n "$existing_step" ]; then
    log "REFUSE: $od contains checkpoints ($existing_step) — remove/rename before rerun."
    note "$tag" "REFUSED_EXISTING_STEP"; exit 1
  fi
  # empty stubs from #133 OOM_BLOCKED runs on zwfy6 are fine (this is wzc1, they don't exist here)
done
log "preflight OK"

# eff_bs sanity
nGPU=$(awk -F, '{print NF}' <<<"$GPUS")
REAL_EFF=$(( BS * GA * nGPU ))
if [ "$REAL_EFF" -ne 128 ]; then
  log "FATAL: eff_bs=$REAL_EFF != 128 (BS=$BS GA=$GA nGPU=$nGPU). #133 explicitly bans eff_bs relaxation — comparability with A4 side breaks."
  exit 1
fi
log "eff_bs=$REAL_EFF OK (comparable to #133 A4 side and to #92/#134)"

# ---- 1. auto-tokenise packed .npy if missing on wzc1 -------------------------
if [ ! -f "$DATA_NPY" ]; then
  if [ ! -f "$DATA_JSONL" ]; then
    log "FATAL: neither $DATA_NPY nor $DATA_JSONL present; cannot tokenise."
    exit 1
  fi
  log "tokenising SQuAD SFT: $DATA_JSONL -> $DATA_NPY (seq_len=$SEQ_LEN)"
  "$PYTHON_BIN" scripts/tokenize_squad_olmo2_sft.py \
      --in_jsonl "$DATA_JSONL" \
      --out_npy  "$DATA_NPY" \
      --tokenizer "$BASE" \
      --seq_len "$SEQ_LEN" \
      >> "$LOG_FILE" 2>&1
  [ -f "$DATA_NPY" ] || { log "FATAL: tokenise failed, no $DATA_NPY"; exit 1; }
  log "tokenised OK: $(du -h "$DATA_NPY" | awk '{print $1}')"
fi

# ---- 2. TRAIN each depth via run_paperC_pc1.sh -------------------------------
run_a3(){                          # $1=KEEP $2=port
  local keep="$1" port="$2"
  local tag="A3_keep${keep}fresh${FRESH_N}"
  local od="$PROJECT_ROOT/outputs/paperC_pc1_squad_${tag}"
  local trainlog="$PROJECT_ROOT/logs/paperC_pc1_squad_${tag}.log"

  log "=== launch $tag ==="
  note "$tag" "LAUNCH BS=$BS GA=$GA OPT=$OPT"

  env ARM=A3 KEEP="$keep" FRESH="$FRESH_N" GPUS="$GPUS" PORT="$port" \
      MAX_STEPS="$MAX_STEPS" BS="$BS" GA="$GA" OPT="$OPT" \
      SEQ_LEN="$SEQ_LEN" SEED="$SEED" \
      DATA_PATH="$DATA_NPY" MODEL_PATH="$BASE" \
      PROJECT_ROOT="$PROJECT_ROOT" PYTHON_BIN="$PYTHON_BIN" \
      FOREGROUND=1 \
      bash scripts/run_paperC_pc1.sh \
      >> "$LOG_FILE" 2>&1

  if [ -f "$od/final.pt" ]; then
    local layers=$(( keep + FRESH_N ))
    log "OK $tag final.pt present (layers=$layers)"
    note "$tag" "TRAIN_DONE layers=$layers"
    return 0
  fi

  # implausible on B200, but be graceful
  if grep -qiE 'OutOfMemoryError|CUDA out of memory' "$trainlog" 2>/dev/null && [ "$OPT" != "bnb8bit" ]; then
    log "$tag fp32 OOM on B200 (very unexpected). Retrying with OPT=bnb8bit."
    note "$tag" "OOM_RETRY_BNB8BIT"
    pkill -9 -f 'train_olmo2_arch_probe2' 2>/dev/null; sleep 20
    env ARM=A3 KEEP="$keep" FRESH="$FRESH_N" GPUS="$GPUS" PORT="$port" \
        MAX_STEPS="$MAX_STEPS" BS="$BS" GA="$GA" OPT=bnb8bit \
        SEQ_LEN="$SEQ_LEN" SEED="$SEED" \
        DATA_PATH="$DATA_NPY" MODEL_PATH="$BASE" \
        PROJECT_ROOT="$PROJECT_ROOT" PYTHON_BIN="$PYTHON_BIN" \
        FOREGROUND=1 \
        bash scripts/run_paperC_pc1.sh \
        >> "$LOG_FILE" 2>&1
    if [ -f "$od/final.pt" ]; then
      log "OK $tag final.pt present (retry bnb8bit)"
      note "$tag" "TRAIN_DONE_BNB8BIT"
      return 0
    fi
    log "$tag STILL no final.pt after bnb8bit retry."
    note "$tag" "TRAIN_FAIL_EVEN_BNB8BIT"
    return 1
  fi

  log "$tag no final.pt (non-OOM failure) — see $trainlog"
  note "$tag" "TRAIN_FAIL_NON_OOM"
  return 1
}

i=0
for keep in $KEEPS; do
  i=$((i+1))
  run_a3 "$keep" $(( PORT_BASE + i )) || log "continuing after $tag failure"
done

# ---- 3. EVAL every finished depth (identical protocol to #92 / #133) ---------
eval_a3(){                         # $1=KEEP
  local keep="$1"
  local tag="A3_keep${keep}fresh${FRESH_N}"
  local ck="$PROJECT_ROOT/outputs/paperC_pc1_squad_${tag}/final.pt"
  local name="${tag}"
  [ -f "$ck" ] || { log "eval SKIP $tag (no final.pt)"; return 0; }
  if [ -f "$PROJECT_ROOT/evidence_squad_label_prior/$name/summary.json" ]; then
    log "eval SKIP $tag (summary.json exists)"; return 0
  fi
  log "=== eval $tag ==="
  CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" scripts/eval_paperC_squad_emf1.py \
      --ckpt "$ck" --base_model "$BASE" --tokenizer "$BASE" \
      --val_path "$VAL_PATH" --output_name "$name" --batch_size 32 \
      --add_bos 0 \
      >> "$PROJECT_ROOT/logs/paperC_133b_eval_${name}.log" 2>&1
  "$PYTHON_BIN" scripts/eval_paperC_squad_emf1.py --merge --output_name "$name" \
      >> "$PROJECT_ROOT/logs/paperC_133b_eval_${name}.log" 2>&1
  if [ -f "$PROJECT_ROOT/evidence_squad_label_prior/$name/summary.json" ]; then
    log "eval $tag DONE"
    note "$tag" "EVAL_DONE"
  else
    log "eval $tag FAILED — see logs/paperC_133b_eval_${name}.log"
    note "$tag" "EVAL_FAIL"
  fi
}

log "=== SQuAD dev EM/F1 evals (n=2000, add_bos=0, chat=False) ==="
for keep in $KEEPS; do eval_a3 "$keep"; done

# ---- 4. combined curve: A4 vs A3 at keep ∈ {14 (from #92), 20, 24, 28} -------
log "=== P-C1 DEPTH CURVE — A4 (hero, #133 completed) vs A3 (control, #133b B200) ==="
FRESH_N="$FRESH_N" KEEPS="$KEEPS" ROOT="$PROJECT_ROOT" "$PYTHON_BIN" - <<'PY'
import json, os
root  = os.environ["ROOT"]
fresh = int(os.environ.get("FRESH_N", "2"))
keeps = [int(k) for k in os.environ.get("KEEPS", "20 24 28").split()]
res   = os.path.join(root, "evidence_squad_label_prior")

def get(name):
    p = os.path.join(res, name, "summary.json")
    if not os.path.isfile(p): return None
    d = json.load(open(p))
    return d.get("em"), d.get("f1"), d.get("n")

# keep14 = the original #92 point (NOT retrained here, just reused for the curve)
rows = [(14, "A4_hero",              "A3_fromscratch",              "#92")]
rows += [(k, f"A4_keep{k}fresh{fresh}", f"A3_keep{k}fresh{fresh}",  "#133/#133b") for k in keeps]

print(f"{'keep':>5s} {'L':>3s} {'src':>10s} "
      f"{'A4_EM':>7s} {'A4_F1':>7s} {'A3_EM':>7s} {'A3_F1':>7s} {'dEM(A4-A3)':>11s} {'dF1':>8s}")
for keep, a4, a3, src in rows:
    r4, r3 = get(a4), get(a3)
    L = keep + fresh
    f = lambda v: f"{v:7.4f}" if v is not None else "     --"
    e4 = r4[0] if r4 else None; f4 = r4[1] if r4 else None
    e3 = r3[0] if r3 else None; f3 = r3[1] if r3 else None
    dem = f"{(e4-e3)*100:+10.2f}" if (e4 is not None and e3 is not None) else "        --"
    df1 = f"{(f4-f3)*100:+7.2f}" if (f4 is not None and f3 is not None) else "     --"
    print(f"{keep:5d} {L:3d} {src:>10s} {f(e4)} {f(f4)} {f(e3)} {f(f3)} {dem} {df1}")
print("\ndEM/dF1 in percentage points. '--' = summary.json missing (see status ledger).")
PY

log "=== status ledger ($STATUS) ==="
cat "$STATUS" 2>/dev/null | tee -a "$LOG_FILE"
log "=== #133b A3 B200 DONE ==="
