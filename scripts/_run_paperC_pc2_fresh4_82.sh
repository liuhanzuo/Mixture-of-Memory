#!/usr/bin/env bash
# Paper C P-C2 — FRESH=4 depth variants on .82 (8×H20).
#
# WHY: task #92/#133 established the FRESH=2 j-curve at keep∈{14,20,24,28}
#   (EM 0.2930/0.3440/0.3560/0.4190). P-C2 needs the (j,K) grid — i.e. the
#   FRESH=4 sister curve at keep∈{20,24,28} — to state the "optimal (j,K)"
#   claim with more than one K per depth. This launcher fills the 3 missing
#   FRESH=4 cells; keep14+fresh4 is already covered elsewhere per prompt.
#
# HOW: reuse scripts/run_paperC_pc1.sh UNMODIFIED for each depth via env:
#   ARM=A4 KEEP∈{20,24,28} FRESH=4 -> --freeze_front (HERO freeze-graft)
#   lr=1e-4 lr_inh=2e-5 (hardcoded for ARM=A4)
#   eff_bs = BS*GA*nGPU = 4*4*8 = 128 (bit-identical to #133 A4 fresh=2 side)
#   MAX_STEPS=1000 SEQ_LEN=2048 SEED=42
#   OUT_DIR = outputs/paperC_pc1_squad_A4_keep{K}fresh4 (no clash with fresh2)
# chain the three depths sequentially; fault-tolerant per depth (record and
# continue). Each depth followed by SQuAD dev EM/F1 eval @ n=2000 add_bos=0
# (same protocol as #92 / #133 A4 side, chat=False greedy).
#
# NOT LAUNCHED by this file — MAIN invokes it detached. Guard rails:
#   * Refuses to start any depth whose OUT_DIR already carries final.pt / step*.pt
#     (defence-in-depth; #133 A4 fresh2 outputs are keep{K}fresh2, no collision).
#   * Preflights base_model/data/eval_script/val presence + 8 visible GPUs.
#   * eff_bs sanity: aborts if BS*GA*nGPU != 128 (P-C2 comparability rule).
#   * fresh=4 uses same 7B fp32 AdamW as fresh=2; keep28+fresh4=32L estimated
#     ~68 GiB/97.8 GiB (headroom on H20). If OOM: retry same recipe once, then
#     report OOM_BLOCKED — DO NOT relax eff_bs.
#
# Usage (detached, on .82):
#   cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
#   setsid nohup bash scripts/_run_paperC_pc2_fresh4_82.sh \
#     > logs/paperC_pc2_fresh4_82.log 2>&1 &
#
# Env overrides (all optional):
#   PROJECT_ROOT   default /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
#   PYTHON_BIN     default /opt/conda/envs/torch-base/bin/python
#   BASE           default /apdcephfs_zwfy6/share_304376610/pighzliu_code/models/OLMo-2-1124-7B
#   GPUS           default 0,1,2,3,4,5,6,7
#   PORT_BASE      default 29600  (per-depth port = PORT_BASE + i)
#   MAX_STEPS      default 1000
#   BS/GA          default 4/4    (eff_bs=128 on 8 GPUs)
#   OPT            default adamw
#   SEQ_LEN        default 2048
#   SEED           default 42
#   KEEPS          default "20 24 28"
#   FRESH_N        default 4
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
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

BASE="${BASE:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/models/OLMo-2-1124-7B}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
PORT_BASE="${PORT_BASE:-29600}"
MAX_STEPS="${MAX_STEPS:-1000}"
SEQ_LEN="${SEQ_LEN:-2048}"
SEED="${SEED:-42}"
BS="${BS:-4}"
GA="${GA:-4}"
OPT="${OPT:-adamw}"
KEEPS="${KEEPS:-20 24 28}"
FRESH_N="${FRESH_N:-4}"
SAVE_EVERY="${SAVE_EVERY:-1000}"   # only final ckpt at step 1000 (== max_steps)

DATA_JSONL="$PROJECT_ROOT/data/squad_train.jsonl"
DATA_NPY="$PROJECT_ROOT/data/squad_sft_olmo2_2048_train.npy"
VAL_PATH="$PROJECT_ROOT/data/squad_val.jsonl"

LOG_FILE="${LOG_FILE:-$PROJECT_ROOT/logs/paperC_pc2_fresh4_82.log}"
STATUS="$PROJECT_ROOT/logs/paperC_pc2_fresh4_status.tsv"

mkdir -p "$PROJECT_ROOT/logs"
: > "$STATUS"

log(){ echo "[paperC_pc2 $(date '+%F %T')] $*" | tee -a "$LOG_FILE"; }
note(){ printf '%s\t%s\t%s\n' "$(date '+%F %T')" "$1" "$2" >> "$STATUS"; }

log "PROJECT_ROOT=$PROJECT_ROOT"
log "PYTHON_BIN=$PYTHON_BIN"
log "BASE=$BASE"
log "GPUS=$GPUS PORT_BASE=$PORT_BASE MAX_STEPS=$MAX_STEPS SEQ_LEN=$SEQ_LEN"
log "KEEPS=$KEEPS FRESH_N=$FRESH_N BS=$BS GA=$GA OPT=$OPT SEED=$SEED SAVE_EVERY=$SAVE_EVERY"

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

# refuse to clobber any depth that already carries checkpoints
for keep in $KEEPS; do
  tag="A4_keep${keep}fresh${FRESH_N}"
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
done
log "preflight OK"

# eff_bs sanity
nGPU=$(awk -F, '{print NF}' <<<"$GPUS")
REAL_EFF=$(( BS * GA * nGPU ))
if [ "$REAL_EFF" -ne 128 ]; then
  log "FATAL: eff_bs=$REAL_EFF != 128 (BS=$BS GA=$GA nGPU=$nGPU). P-C2 explicitly bans eff_bs relaxation — comparability with #133 A4 fresh=2 side breaks."
  exit 1
fi
log "eff_bs=$REAL_EFF OK (comparable to #133 A4 fresh=2 side)"

# ---- 1. auto-tokenise packed .npy if missing ---------------------------------
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
run_a4(){                          # $1=KEEP $2=port
  local keep="$1" port="$2"
  local tag="A4_keep${keep}fresh${FRESH_N}"
  local od="$PROJECT_ROOT/outputs/paperC_pc1_squad_${tag}"
  local trainlog="$PROJECT_ROOT/logs/paperC_pc1_squad_${tag}.log"

  log "=== launch $tag ==="
  note "$tag" "LAUNCH BS=$BS GA=$GA OPT=$OPT PORT=$port"

  env ARM=A4 KEEP="$keep" FRESH="$FRESH_N" GPUS="$GPUS" PORT="$port" \
      MAX_STEPS="$MAX_STEPS" BS="$BS" GA="$GA" OPT="$OPT" \
      EFF_BS=128 SAVE_EVERY="$SAVE_EVERY" \
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

  # detect OOM (do NOT relax eff_bs per prompt hard rule; retry same recipe once)
  if grep -qiE 'OutOfMemoryError|CUDA out of memory' "$trainlog" 2>/dev/null; then
    log "$tag OOM on H20 (keep=$keep total_layers=$((keep+FRESH_N))). Retrying same recipe once (no eff_bs relaxation)."
    note "$tag" "OOM_RETRY_SAME_RECIPE"
    pkill -9 -f 'train_olmo2_arch_probe2' 2>/dev/null; sleep 20
    env ARM=A4 KEEP="$keep" FRESH="$FRESH_N" GPUS="$GPUS" PORT="$port" \
        MAX_STEPS="$MAX_STEPS" BS="$BS" GA="$GA" OPT="$OPT" \
        EFF_BS=128 SAVE_EVERY="$SAVE_EVERY" \
        SEQ_LEN="$SEQ_LEN" SEED="$SEED" \
        DATA_PATH="$DATA_NPY" MODEL_PATH="$BASE" \
        PROJECT_ROOT="$PROJECT_ROOT" PYTHON_BIN="$PYTHON_BIN" \
        FOREGROUND=1 \
        bash scripts/run_paperC_pc1.sh \
        >> "$LOG_FILE" 2>&1
    if [ -f "$od/final.pt" ]; then
      log "OK $tag final.pt present (retry)"
      note "$tag" "TRAIN_DONE_RETRY"
      return 0
    fi
    log "$tag OOM_BLOCKED after retry — reporting failure (no eff_bs relaxation allowed)."
    note "$tag" "OOM_BLOCKED"
    return 1
  fi

  log "$tag no final.pt (non-OOM failure) — see $trainlog"
  note "$tag" "TRAIN_FAIL_NON_OOM"
  return 1
}

i=0
for keep in $KEEPS; do
  i=$((i+1))
  run_a4 "$keep" $(( PORT_BASE + i )) || log "continuing after keep=$keep failure"
done

# ---- 3. EVAL every finished depth (identical protocol to #92 / #133) ---------
eval_a4(){                         # $1=KEEP
  local keep="$1"
  local tag="A4_keep${keep}fresh${FRESH_N}"
  local ck="$PROJECT_ROOT/outputs/paperC_pc1_squad_${tag}/final.pt"
  local name="${tag}"
  [ -f "$ck" ] || { log "eval SKIP $tag (no final.pt)"; return 0; }
  if [ -f "$PROJECT_ROOT/evidence_squad_label_prior/$name/summary.json" ]; then
    log "eval SKIP $tag (summary.json exists)"; return 0
  fi
  log "=== eval $tag ==="
  note "$tag" "EVAL_LAUNCH"
  CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" scripts/eval_paperC_squad_emf1.py \
      --ckpt "$ck" --base_model "$BASE" --tokenizer "$BASE" \
      --val_path "$VAL_PATH" --output_name "$name" --batch_size 32 \
      --add_bos 0 \
      >> "$PROJECT_ROOT/logs/paperC_pc2_fresh4_eval_${name}.log" 2>&1
  "$PYTHON_BIN" scripts/eval_paperC_squad_emf1.py --merge --output_name "$name" \
      >> "$PROJECT_ROOT/logs/paperC_pc2_fresh4_eval_${name}.log" 2>&1
  if [ -f "$PROJECT_ROOT/evidence_squad_label_prior/$name/summary.json" ]; then
    log "eval $tag DONE"
    note "$tag" "EVAL_DONE"
  else
    log "eval $tag FAILED — see logs/paperC_pc2_fresh4_eval_${name}.log"
    note "$tag" "EVAL_FAIL"
  fi
}

log "=== SQuAD dev EM/F1 evals (n=2000, add_bos=0, chat=False, greedy) ==="
for keep in $KEEPS; do eval_a4 "$keep"; done

# ---- 4. combined table: FRESH=2 (from #133) vs FRESH=4 (this run) ------------
log "=== P-C2 (j,K) GRID — FRESH=2 (from #92/#133) vs FRESH=4 (this batch) ==="
FRESH_N="$FRESH_N" KEEPS="$KEEPS" ROOT="$PROJECT_ROOT" "$PYTHON_BIN" - <<'PY'
import json, os
root  = os.environ["ROOT"]
fresh = int(os.environ.get("FRESH_N", "4"))
keeps = [int(k) for k in os.environ.get("KEEPS", "20 24 28").split()]
res   = os.path.join(root, "evidence_squad_label_prior")

def get(name):
    p = os.path.join(res, name, "summary.json")
    if not os.path.isfile(p): return None
    d = json.load(open(p))
    return d.get("em"), d.get("f1"), d.get("n")

# keep14 from #92 (fresh=2 only)
rows = [(14, "A4_hero", None, "#92")]
rows += [(k, f"A4_keep{k}fresh2", f"A4_keep{k}fresh{fresh}", "#133/#pc2") for k in keeps]

print(f"{'keep':>5s} {'src':>10s} "
      f"{'F2_EM':>7s} {'F2_F1':>7s} {'F4_EM':>7s} {'F4_F1':>7s} {'dEM(F4-F2)':>11s} {'dF1':>8s}")
for keep, a_f2, a_f4, src in rows:
    r2 = get(a_f2) if a_f2 else None
    r4 = get(a_f4) if a_f4 else None
    f = lambda v: f"{v:7.4f}" if v is not None else "     --"
    e2 = r2[0] if r2 else None; f2 = r2[1] if r2 else None
    e4 = r4[0] if r4 else None; f4 = r4[1] if r4 else None
    dem = f"{(e4-e2)*100:+10.2f}" if (e4 is not None and e2 is not None) else "        --"
    df1 = f"{(f4-f2)*100:+7.2f}" if (f4 is not None and f2 is not None) else "     --"
    print(f"{keep:5d} {src:>10s} {f(e2)} {f(f2)} {f(e4)} {f(f4)} {dem} {df1}")
print("\ndEM/dF1 in percentage points. '--' = summary.json missing (see status ledger).")
PY

log "=== status ledger ($STATUS) ==="
cat "$STATUS" 2>/dev/null | tee -a "$LOG_FILE"
log "=== P-C2 FRESH=4 batch on .82 DONE ==="
