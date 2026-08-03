#!/usr/bin/env bash
# ============================================================================
# Paper C P-C1 follow-up (task #133): DEPTH-SWEEP train+eval launcher.
#
# WHY: P-C1 established freeze-graft > from-scratch at ONE depth (keep14+fresh2 =
# 16L). This launcher extends the single point into a DEPTH CURVE to test whether
# the freeze-graft advantage is robust across depth. For each keep_front in
# {20,24,28} (n_fresh=2 -> 22/26/30L), it trains TWO arms with a recipe IDENTICAL
# to the P-C1 SQuAD 4-arm:
#     graft   : freeze-graft (== P-C1 A4)  -> --freeze_front   lr_fresh=1e-4 lr_inh=2e-5
#     scratch : from-scratch (== P-C1 A3)  -> --from_scratch   single lr=3e-4 (lr==lr_inh)
# ...6 training runs total, then SQuAD dev EM/F1 (n=2000, same eval as P-C1) on
# each. Recipe is copied VERBATIM from scripts/run_paperC_pc1.sh (A4/A3 arms):
#   data=data/squad_sft_olmo2_2048_train.npy, seq_len=2048, max_steps=1000,
#   warmup_steps=150, eff_bs=128 (BS*GA*nGPU), gradient_checkpointing=1,
#   optimizer=adamw (fp32-master), seed=42.
#     A4 (graft):   KEEP=k FRESH=2 --freeze_front  LR=1e-4  LR_INH=2e-5
#     A3 (scratch): KEEP=k FRESH=2 --from_scratch  LR=3e-4  LR_INH=3e-4
#   (LRs verified against run_paperC_pc1.sh; from-scratch uses a single LR so
#    lr==lr_inherited, matching _classify_param's from_scratch='fresh' bucketing.)
#
# TARGET NODE: B200 (8xL20A 183GB, .venv torch2.10). from-scratch is a full-param
# ~deep-model train; 183GB fits fp32-AdamW, so we keep OPT=adamw (== P-C1 A3) for
# comparability. If a specific keep+scratch OOMs even on B200, set OPT=bnb8bit for
# that run (bitsandbytes 8-bit AdamW, == P-C1 A1 fallback) -- but note the optimizer
# difference in the report; the default here is adamw to stay comparable to A3.
#
# EVAL LOAD: each ckpt is a raw state_dict .pt from train_olmo2_arch_probe2 that
# STORES keep_front/n_fresh in its meta. eval_paperC_squad_emf1.py --ckpt rebuilds
# the exact (keep+fresh)-layer shell from that meta and strict-loads -> zero drift.
# We do NOT pass --keep_front_layers/--n_fresh_layers to the eval (meta drives it).
#
# FAULT-TOLERANT: each run is launched FOREGROUND and chained; a failed/OOM run is
# logged and the chain continues to the next. Eval only runs if final.pt exists.
#
# ****  DO NOT AUTO-RUN. This is only a launcher.  ****
# B200 is currently busy with P0.5; MAIN will start this by hand once B200 frees.
#
# USAGE (on B200, after it frees up):
#   cd /apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory
#   PYBIN=.venv/bin/python \
#   setsid nohup bash scripts/run_paperC_depthsweep.sh > logs/paperC_depthsweep_sched.out 2>&1 &
#   # optional overrides:
#   #   KEEPS="20 24 28"  ARMS="graft scratch"  GPUS=0,1,2,3,4,5,6,7  PORT=29560
#   #   MAX_STEPS=1000  EFF_BS=128  BS=4  OPT=adamw   EVAL_ONLY=0
# ============================================================================
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
PYBIN="${PYBIN:-${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}}"
BASE="${BASE:-/apdcephfs_wzc1/share_304376610/pighzliu_code/models/OLMo-2-1124-7B}"

DATA_PATH="${DATA_PATH:-$PROJECT_ROOT/data/squad_sft_olmo2_2048_train.npy}"
VAL_PATH="${VAL_PATH:-$PROJECT_ROOT/data/squad_val.jsonl}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
PORT_BASE="${PORT:-29560}"
MAX_STEPS="${MAX_STEPS:-1000}"
SEQ_LEN="${SEQ_LEN:-2048}"
WARMUP="${WARMUP:-150}"
SEED="${SEED:-42}"
EFF_BS="${EFF_BS:-128}"
BS="${BS:-4}"
FRESH="${FRESH:-2}"
OPT="${OPT:-adamw}"                         # adamw (== A3, default) | bnb8bit (OOM fallback)
read -r -a KEEPS <<< "${KEEPS:-20 24 28}"
read -r -a ARMS  <<< "${ARMS:-graft scratch}"
EVAL_ONLY="${EVAL_ONLY:-0}"                 # 1 -> skip training, just re-eval existing final.pt
EVAL_BATCH="${EVAL_BATCH:-32}"

export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

nGPU=$(awk -F, '{print NF}' <<< "$GPUS")
GA=$(( EFF_BS / (BS * nGPU) )); [ "$GA" -lt 1 ] && GA=1
REAL_EFF=$(( BS * GA * nGPU ))

LOGDIR="$PROJECT_ROOT/logs"; mkdir -p "$LOGDIR"
SUMMARY_LOG="$LOGDIR/paperC_depthsweep_summary.log"

log(){ echo "[depthsweep $(date '+%F %T')] $*"; }

log "PYBIN=$PYBIN nGPU=$nGPU BS=$BS GA=$GA eff_bs=$REAL_EFF (target $EFF_BS) OPT=$OPT max_steps=$MAX_STEPS"
[ "$REAL_EFF" -ne "$EFF_BS" ] && log "WARNING eff_bs=$REAL_EFF != $EFF_BS (adjust BS/EFF_BS)"
log "keeps=${KEEPS[*]} arms=${ARMS[*]} fresh=$FRESH  (DO-NOT-AUTORUN launcher; MAIN starts on free B200)"

OPT_FLAG=""
[ "$OPT" = "bnb8bit" ] && OPT_FLAG="--optimizer bnb_adamw8bit"

# per-arm recipe (verbatim from run_paperC_pc1.sh A4/A3) --------------------------
arm_extra(){ case "$1" in graft) echo "--freeze_front" ;; scratch) echo "--from_scratch" ;; esac; }
arm_lr(){    case "$1" in graft) echo "1e-4" ;;         scratch) echo "3e-4" ;; esac; }
arm_lrinh(){ case "$1" in graft) echo "2e-5" ;;         scratch) echo "3e-4" ;; esac; }

port_off=0
for K in "${KEEPS[@]}"; do
  for arm in "${ARMS[@]}"; do
    EXTRA="$(arm_extra "$arm")"; LR="$(arm_lr "$arm")"; LR_INH="$(arm_lrinh "$arm")"
    OUT_DIR="$PROJECT_ROOT/outputs/paperC_depthsweep_keep${K}_${arm}"
    LOG_FILE="$LOGDIR/paperC_depthsweep_keep${K}_${arm}.log"
    PORT=$(( PORT_BASE + port_off )); port_off=$(( port_off + 1 ))
    mkdir -p "$OUT_DIR"

    # ---- train (foreground, chained) ----
    if [ "$EVAL_ONLY" != "1" ] && [ ! -f "$OUT_DIR/final.pt" ]; then
      log "=== TRAIN keep$K/$arm KEEP=$K FRESH=$FRESH EXTRA='$EXTRA' LR=$LR LR_INH=$LR_INH PORT=$PORT -> $OUT_DIR ==="
      : > "$LOG_FILE"
      CUDA_VISIBLE_DEVICES="$GPUS" "$PYBIN" -m torch.distributed.run \
        --nnodes 1 --nproc_per_node "$nGPU" --rdzv_backend c10d --rdzv_endpoint "localhost:$PORT" \
        scripts/train_olmo2_arch_probe2.py \
          --data_path "$DATA_PATH" --output_dir "$OUT_DIR" --model_path "$BASE" \
          --keep_front_layers "$K" --n_fresh_layers "$FRESH" \
          --batch_size "$BS" --grad_accumulation_steps "$GA" --seq_len "$SEQ_LEN" \
          --lr "$LR" --lr_inherited "$LR_INH" --max_steps "$MAX_STEPS" \
          --warmup_steps "$WARMUP" --save_every 500 --log_every 10 --seed "$SEED" \
          --gradient_checkpointing 1 $EXTRA $OPT_FLAG \
        >>"$LOG_FILE" 2>&1
      rc=$?
      if [ "$rc" -ne 0 ]; then log "FAIL train keep$K/$arm (exit $rc) -> continue"; fi
    else
      log "SKIP train keep$K/$arm (EVAL_ONLY=$EVAL_ONLY, final.pt exists=$( [ -f "$OUT_DIR/final.pt" ] && echo yes || echo no ))"
    fi

    # ---- eval SQuAD dev EM/F1 (n=2000, same script/口径 as P-C1) ----
    if [ -f "$OUT_DIR/final.pt" ]; then
      ONAME="depthsweep_keep${K}_${arm}"
      log "=== EVAL keep$K/$arm SQuAD EM/F1 -> $ONAME ==="
      # meta drives keep_front/n_fresh; we pass --base_model $BASE (cfg+tokenizer).
      CUDA_VISIBLE_DEVICES="$(echo "$GPUS" | cut -d, -f1)" "$PYBIN" scripts/eval_paperC_squad_emf1.py \
        --ckpt "$OUT_DIR/final.pt" --base_model "$BASE" --tokenizer "$BASE" \
        --val_path "$VAL_PATH" --output_name "$ONAME" --batch_size "$EVAL_BATCH" \
        >>"$LOG_FILE" 2>&1 || log "FAIL eval keep$K/$arm -> continue"
      "$PYBIN" scripts/eval_paperC_squad_emf1.py --merge --output_name "$ONAME" \
        >>"$LOG_FILE" 2>&1 || log "FAIL merge keep$K/$arm -> continue"
    else
      log "no final.pt for keep$K/$arm -> skip eval"
    fi
  done
done

# ---- summary table (for MAIN to backfill) -----------------------------------
log "=== DEPTH-SWEEP SQuAD EM/F1 SUMMARY -> $SUMMARY_LOG ==="
"$PYBIN" - <<'PY' | tee "$SUMMARY_LOG"
import glob, json, os
rows = []
for sm in sorted(glob.glob("paperC_squad_results/depthsweep_*/summary.json")):
    name = os.path.basename(os.path.dirname(sm))
    try:
        s = json.load(open(sm))
    except Exception as e:
        print(f"[warn] cannot read {sm}: {e}"); continue
    meta = s.get("meta") or {}
    rows.append((name, s.get("em"), s.get("f1"), s.get("n"),
                 meta.get("keep_front_layers"), meta.get("n_fresh_layers")))
print(f"{'run':34s} {'keep':>4s} {'fresh':>5s} {'EM':>8s} {'F1':>8s} {'n':>6s}")
for name, em, f1, n, k, fr in rows:
    em = em if em is not None else float('nan')
    f1 = f1 if f1 is not None else float('nan')
    print(f"{name:34s} {str(k):>4s} {str(fr):>5s} {em:8.4f} {f1:8.4f} {(n or 0):6d}")
PY
log "=== DEPTH-SWEEP DONE ==="
