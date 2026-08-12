#!/usr/bin/env bash
# Paper C task #134 — A1 full-FT-32L ceiling on 8×B200 (wzc1).
#
# WHY: task #92 produced A4 (freeze-graft hero), A3 (from-scratch control) and
# A2 (LoRA r=160), but A1 (full-FT all 32 layers of OLMo-2 7B) OOM'd on H20
# because fp32 AdamW on 7B doesn't fit 95 GiB. B200 (183 GiB/card) fits it.
# A1 is the "no-graft ceiling" reference for the P-C1 depth-curve claim.
#
# HOW: reuses scripts/run_paperC_pc1.sh UNMODIFIED with ARM=A1
#   * ARM=A1 -> KEEP=32 FRESH=0 (full 32-layer transplant, no graft)
#   * single-LR: --lr==--lr_inherited==1e-5 (already handled by ARM=A1 branch)
#   * eff_bs = BS * GA * nGPU pinned to 128 for comparability with
#     #92 A2/A3/A4 and #133 A4 depth-sweep points.
#   * On B200 (183 GiB): BS=2 GA=8 nGPU=8 -> eff_bs=128 fits fp32 AdamW cleanly
#     (7B model = 14 GiB bf16 / 28 GiB fp32 params + 56 GiB fp32 Adam states +
#     activations, all comfortably under 183 GiB per card).
#   * OPT: default adamw (fp32); OPT=bnb8bit available as belt-and-braces fallback.
#   * MAX_STEPS=1000 SEQ_LEN=2048 SEED=42 -- identical to #92 A2/A3/A4 and #133.
#   * OUTPUT_DIR=outputs/paperC_pc1_squad_A1_full32ft (NEW name; #92/#133 never
#     wrote to this dir, so no risk of overwriting any prior work).
#
# NOT LAUNCHED. This launcher is written for MAIN to invoke the moment .252
# (or LOCAL) frees. Guard rails:
#   * Refuses to start if OUTPUT_DIR already contains a final.pt or step*.pt.
#   * Preflights data/base_model/eval_script presence + 8 visible GPUs.
#   * Auto-tokenises data/squad_val.jsonl into the packed .npy shard if it's
#     not on this disk (wzc1 doesn't carry the #92 tokenised .npy — that lived
#     on zwfy6 which is only mounted on .82/.104).
#
# Usage:
#   setsid nohup bash scripts/_run_paperC_134_a1fullft.sh \
#     > logs/paperC_134_a1fullft.log 2>&1 &
#
# Env overrides (all optional):
#   PROJECT_ROOT   default /apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory
#   PYTHON_BIN     default /opt/conda/envs/torch-base/bin/python (fallback .venv/bin/python)
#   BASE           default $PROJECT_ROOT/../models/OLMo-2-1124-7B
#   GPUS           default 0,1,2,3,4,5,6,7
#   PORT           default 29580
#   MAX_STEPS      default 1000
#   BS/GA          default 2/8   (eff_bs=128 on 8 GPUs)
#   OPT            default adamw ("bnb8bit" for the belt-and-braces fallback)
#   SEQ_LEN        default 2048
#   SEED           default 42
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT" || { echo "FATAL: cannot cd $PROJECT_ROOT"; exit 1; }

# Python: prefer conda (canonical); fall back to .venv only if conda is absent.
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
PORT="${PORT:-29580}"
MAX_STEPS="${MAX_STEPS:-1000}"
SEQ_LEN="${SEQ_LEN:-2048}"
SEED="${SEED:-42}"
BS="${BS:-2}"
GA="${GA:-8}"
OPT="${OPT:-adamw}"

DATA_JSONL="$PROJECT_ROOT/data/squad_train.jsonl"
DATA_NPY="$PROJECT_ROOT/data/squad_sft_olmo2_2048_train.npy"
VAL_PATH="$PROJECT_ROOT/data/squad_val.jsonl"

OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/outputs/paperC_pc1_squad_A1_full32ft}"
LOG_FILE="${LOG_FILE:-$PROJECT_ROOT/logs/paperC_134_a1fullft.log}"
STATUS="$PROJECT_ROOT/logs/paperC_134_status.tsv"

mkdir -p "$OUT_DIR" "$PROJECT_ROOT/logs"

log(){ echo "[paperC_134 $(date '+%F %T')] $*" | tee -a "$LOG_FILE"; }
note(){ printf '%s\t%s\t%s\n' "$(date '+%F %T')" "$1" "$2" >> "$STATUS"; }

log "PROJECT_ROOT=$PROJECT_ROOT"
log "PYTHON_BIN=$PYTHON_BIN"
log "BASE=$BASE"
log "GPUS=$GPUS PORT=$PORT MAX_STEPS=$MAX_STEPS SEQ_LEN=$SEQ_LEN BS=$BS GA=$GA OPT=$OPT SEED=$SEED"
log "OUT_DIR=$OUT_DIR"

# ---- 0. refuse to clobber an existing run ------------------------------------
if [ -f "$OUT_DIR/final.pt" ]; then
  log "REFUSE: $OUT_DIR/final.pt already exists — will NOT overwrite."
  note "A1_full32ft" "REFUSED_EXISTING_FINAL"
  exit 1
fi
existing_step=$(ls "$OUT_DIR"/step*.pt 2>/dev/null | head -1 || true)
if [ -n "$existing_step" ]; then
  log "REFUSE: $OUT_DIR contains checkpoints ($existing_step) — will NOT overwrite."
  note "A1_full32ft" "REFUSED_EXISTING_STEP"
  exit 1
fi

# ---- 1. preflight: paths, python, GPUs ---------------------------------------
fail=0
for f in "$BASE/config.json" \
         "$PROJECT_ROOT/scripts/run_paperC_pc1.sh" \
         "$PROJECT_ROOT/scripts/train_olmo2_arch_probe2.py" \
         "$PROJECT_ROOT/scripts/eval_paperC_squad_emf1.py" \
         "$VAL_PATH"; do
  [ -e "$f" ] || { log "PREFLIGHT MISSING: $f"; fail=1; }
done
[ "$fail" = 0 ] || { log "FATAL preflight failed"; note "A1_full32ft" "PREFLIGHT_FAIL"; exit 1; }
CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON_BIN" -c \
  'import torch;n=torch.cuda.device_count();assert n>=8,f"need 8 GPUs, have {n}"' \
  || { log "FATAL: 8 GPUs not visible via CUDA_VISIBLE_DEVICES=$GPUS"; note "A1_full32ft" "GPU_FAIL"; exit 1; }
log "preflight OK"

# ---- 2. auto-tokenise SQuAD SFT if the packed .npy is missing on wzc1 --------
# wzc1 disk does NOT carry the #92 tokenised .npy (that lived on zwfy6 which is
# mounted only on .82/.104). If we're on .252 or LOCAL (both wzc1), regenerate.
if [ ! -f "$DATA_NPY" ]; then
  if [ ! -f "$DATA_JSONL" ]; then
    log "FATAL: neither $DATA_NPY nor $DATA_JSONL present; cannot tokenise."
    note "A1_full32ft" "DATA_MISSING"
    exit 1
  fi
  log "tokenising SQuAD SFT: $DATA_JSONL -> $DATA_NPY (seq_len=$SEQ_LEN)"
  "$PYTHON_BIN" scripts/tokenize_squad_olmo2_sft.py \
      --in_jsonl "$DATA_JSONL" \
      --out_npy  "$DATA_NPY" \
      --tokenizer "$BASE" \
      --seq_len "$SEQ_LEN" \
      >> "$LOG_FILE" 2>&1
  [ -f "$DATA_NPY" ] || { log "FATAL: tokenise failed, no $DATA_NPY"; note "A1_full32ft" "TOKENISE_FAIL"; exit 1; }
  log "tokenised OK: $(du -h "$DATA_NPY" | awk '{print $1}')"
fi

# ---- 3. TRAIN: ARM=A1 via run_paperC_pc1.sh UNMODIFIED -----------------------
# eff_bs check (informational; run_paperC_pc1.sh warns if mismatch).
nGPU=$(awk -F, '{print NF}' <<<"$GPUS")
REAL_EFF=$(( BS * GA * nGPU ))
log "A1 recipe: KEEP=32 FRESH=0 (full 32L, no graft) LR=1e-5 LR_INH=1e-5 OPT=$OPT eff_bs=$REAL_EFF (target 128)"
if [ "$REAL_EFF" -ne 128 ]; then
  log "WARNING: eff_bs=$REAL_EFF != 128 — comparability with #92/#133 breaks. Aborting."
  note "A1_full32ft" "BAD_EFFBS_$REAL_EFF"
  exit 1
fi

log "=== launch A1 full-FT 32L ==="
note "A1_full32ft" "LAUNCH BS=$BS GA=$GA OPT=$OPT eff_bs=$REAL_EFF"

# Delegate to the canonical launcher. FOREGROUND=1 so we stay in this shell and
# can post-eval on the same reservation without a second nohup dance.
env ARM=A1 GPUS="$GPUS" PORT="$PORT" MAX_STEPS="$MAX_STEPS" \
    BS="$BS" GA="$GA" OPT="$OPT" SEQ_LEN="$SEQ_LEN" SEED="$SEED" \
    OUT_DIR="$OUT_DIR" \
    DATA_PATH="$DATA_NPY" MODEL_PATH="$BASE" \
    PROJECT_ROOT="$PROJECT_ROOT" PYTHON_BIN="$PYTHON_BIN" \
    FOREGROUND=1 \
    bash scripts/run_paperC_pc1.sh \
    >> "$LOG_FILE" 2>&1

if [ ! -f "$OUT_DIR/final.pt" ]; then
  # If default fp32 AdamW somehow OOMs (implausible on 183GiB), retry with bnb8bit.
  if grep -qiE 'OutOfMemoryError|CUDA out of memory' "$LOG_FILE" && [ "$OPT" != "bnb8bit" ]; then
    log "A1 fp32 OOM (unexpected on B200) — retrying with OPT=bnb8bit"
    note "A1_full32ft" "OOM_RETRY_BNB8BIT"
    pkill -9 -f 'train_olmo2_arch_probe2' 2>/dev/null; sleep 20
    env ARM=A1 GPUS="$GPUS" PORT="$PORT" MAX_STEPS="$MAX_STEPS" \
        BS="$BS" GA="$GA" OPT=bnb8bit SEQ_LEN="$SEQ_LEN" SEED="$SEED" \
        OUT_DIR="$OUT_DIR" \
        DATA_PATH="$DATA_NPY" MODEL_PATH="$BASE" \
        PROJECT_ROOT="$PROJECT_ROOT" PYTHON_BIN="$PYTHON_BIN" \
        FOREGROUND=1 \
        bash scripts/run_paperC_pc1.sh \
        >> "$LOG_FILE" 2>&1
  fi
fi

if [ -f "$OUT_DIR/final.pt" ]; then
  log "A1 training DONE, final.pt present"
  note "A1_full32ft" "TRAIN_DONE"
else
  log "A1 training FAILED (no final.pt) — see $LOG_FILE"
  note "A1_full32ft" "TRAIN_FAIL"
  exit 1
fi

# ---- 4. EVAL: SQuAD dev EM/F1 (n=2000, base protocol, chat=False, add_bos=0) --
NAME="A1_full32ft"
if [ -f "$PROJECT_ROOT/evidence_squad_label_prior/$NAME/summary.json" ]; then
  log "eval SKIP $NAME (summary.json already exists)"
else
  log "=== eval $NAME ==="
  CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" scripts/eval_paperC_squad_emf1.py \
      --ckpt "$OUT_DIR/final.pt" --base_model "$BASE" --tokenizer "$BASE" \
      --val_path "$VAL_PATH" --output_name "$NAME" --batch_size 32 \
      --add_bos 0 \
      >> "$PROJECT_ROOT/logs/paperC_134_eval_${NAME}.log" 2>&1
  "$PYTHON_BIN" scripts/eval_paperC_squad_emf1.py --merge --output_name "$NAME" \
      >> "$PROJECT_ROOT/logs/paperC_134_eval_${NAME}.log" 2>&1
  if [ -f "$PROJECT_ROOT/evidence_squad_label_prior/$NAME/summary.json" ]; then
    log "eval $NAME DONE"
    note "A1_full32ft" "EVAL_DONE"
    "$PYTHON_BIN" -c "import json;s=json.load(open('$PROJECT_ROOT/evidence_squad_label_prior/$NAME/summary.json'));print('A1 EM=%.4f F1=%.4f n=%d'%(s['em'],s['f1'],s['n']))" | tee -a "$LOG_FILE"
  else
    log "eval $NAME FAILED"
    note "A1_full32ft" "EVAL_FAIL"
  fi
fi

log "=== #134 A1 full-FT DONE ==="
