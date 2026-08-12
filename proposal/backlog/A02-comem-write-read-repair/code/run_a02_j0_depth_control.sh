#!/usr/bin/env bash
# ============================================================================
# A02 — matched-quality DEPTH CONTROL trainer driver.
#
# DERIVED FROM: scripts/_launch_p2_4_depthcurve.sh (the on-disk launcher that
# built outputs/qcmem_distill_qwen_j{6,9,18}_r32_4k). This file is that launcher
# with (a) the depth list replaced, (b) per-arm LoRA rank/alpha made settable so
# the capacity-matched arm is expressible, (c) NO rotation flags -- the zwfy6
# copy of train_qcmem_distill.py predates --keep_last_n and argparse-rejects it
# (a probe died exactly this way on 2026-08-12), and (d) a fail-closed recipe
# audit before any GPU is touched.
#
# NO NEW TRAINER. scripts/train_qcmem_distill.py is used unmodified.
#
# THE TWO ARMS (see A02_J0_DEPTH_CONTROL_PREREG.md for the full design):
#   A1  j=0  r32/a64  -> outputs/a02_j0control_lora_r32
#         The literal "LoRA distilled for j=0" that STATUS.json next_gate[3]
#         asks for. GATE 0 (a 20-step probe) established its objective is
#         ALREADY MINIMISED AT INIT (loss 0.0000 @ step1 vs flagship 0.2991):
#         teacher and student are the same path at j=0, differing only by a
#         zero-init LoRA delta. We run it to 4000 steps anyway so the vacuity
#         claim is MEASURED, and to see whether AdamW's scale-invariance turns
#         bf16 noise into a destructive random walk.
#   A6  j=12 r40/a80 -> outputs/a02_j12_capmatch_r40
#         Capacity control. 75776*40*24 == 75776*32*30 == 72,744,960 params
#         EXACTLY == the on-disk j=6 arm, so "is the j6-vs-j12 gap depth or
#         capacity?" becomes answerable.
#
# EVERYTHING ELSE IS THE FLAGSHIP RECIPE VERBATIM (asserted by GATE B below).
#
# === USAGE (on .82, zwfy6) ==================================================
#   PROJECT_ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory \
#   PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
#   setsid nohup bash proposal/active/A02-comem-write-read-repair/code/run_a02_j0_depth_control.sh \
#     >logs/a02_j0_depth_control_master.log 2>&1 &
# ============================================================================
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT" || { echo "FATAL: cannot cd $PROJECT_ROOT"; exit 3; }

export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="$PROJECT_ROOT/.hf_cache"
export HF_DATASETS_CACHE="$PROJECT_ROOT/.hf_cache/datasets"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
unset http_proxy https_proxy all_proxy

PYBIN="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
MODEL="${MODEL:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b}"
NPROC="${NPROC:-8}"
PROG=logs/a02_j0_depth_control_progress.log
mkdir -p logs
note() { printf '[%s] %s\n' "$(date '+%m-%d %H:%M:%S')" "$*" | tee -a "$PROG"; }

# ---- flagship recipe knobs (verbatim from the flagship distill_args.json) ----
TOP_PREPAY_B=0
LORA_DROPOUT=0.0
LORA_TARGETS="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
CHUNK_SIZE=512
N_CTX=3
QUERY_LOSS_TOKENS=0
TEACHER_TOPK=64
DISTILL_LAMBDA=0.6
CE_WEIGHT=0.0
TOTAL_STEPS="${TOTAL_STEPS:-4000}"
LR=8e-05
WARMUP=100
WEIGHT_DECAY=0.0
GRAD_ACCUM=1
GRAD_CLIP=1.0
SAVE_INTERVAL=500
LOG_INTERVAL=10
DTYPE=bfloat16
ATTN_IMPL=sdpa
SEED=42
# ---------------------------------------------------------------------------- #

GC_FLAG=""
if [[ "${GRAD_CKPT:-0}" == "1" ]]; then GC_FLAG="--gradient_checkpointing"; fi

FLAGSHIP=outputs/qcmem_distill_qwen_j12_r32_4k

# --- GATE A: flagship Read-LoRA identity (fail-closed) ----------------------
EXPECT_SHA=dd09cd17457c63578c0f
GOT_SHA=$(sha256sum "$FLAGSHIP/final/adapter_model.safetensors" 2>/dev/null | cut -c1-20)
if [[ "$GOT_SHA" != "$EXPECT_SHA" ]]; then
  echo "FATAL GATE A: flagship Read-LoRA sha mismatch: got '$GOT_SHA' want '$EXPECT_SHA'"
  exit 7
fi
note "GATE A PASS flagship Read-LoRA sha $GOT_SHA"

# --- GATE B: the reused ladder really is recipe-matched ----------------------
# Every depth arm we will later EVALUATE must match the flagship recipe on all
# fields except resume_j. If it does not, the "matched" claim is nominal only
# and this gate must not run. Uses the flagship's own args as ground truth --
# note the flagship distill_args.json lives on wzc1, so we compare the ladder
# arms against a hard-coded copy of it and against each other.
"$PYBIN" - <<'PYGATE' || { echo "FATAL GATE B: ladder recipe mismatch"; exit 8; }
import json, os, sys
FLAG = {  # verbatim from outputs/qcmem_distill_qwen_j12_r32_4k/distill_args.json
    "lora_rank": 32, "lora_alpha": 64, "lora_dropout": 0.0,
    "lora_targets": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    "chunk_size": 512, "n_ctx": 3, "query_loss_tokens": 0,
    "teacher_topk": 64, "distill_lambda": 0.6, "ce_weight": 0.0,
    "total_steps": 4000, "lr": 8e-05, "warmup_steps": 100, "weight_decay": 0.0,
    "grad_accum": 1, "grad_clip": 1.0, "gradient_checkpointing": False,
    "dtype": "bfloat16", "attn_impl": "sdpa", "seed": 42, "top_prepay_b": 0,
}
bad = []
for j in (6, 9, 18):
    d = f"outputs/qcmem_distill_qwen_j{j}_r32_4k"
    ap = os.path.join(d, "distill_args.json")
    if not os.path.exists(ap):
        bad.append(f"{d}: distill_args.json MISSING"); continue
    a = json.load(open(ap))
    if a.get("resume_j") != j:
        bad.append(f"{d}: resume_j={a.get('resume_j')} != {j}")
    for k, v in FLAG.items():
        if a.get(k) != v:
            bad.append(f"{d}: {k}={a.get(k)!r} != flagship {v!r}")
    cp = os.path.join(d, "final", "adapter_config.json")
    if not os.path.exists(cp):
        bad.append(f"{d}: final/adapter_config.json MISSING"); continue
    c = json.load(open(cp))
    want = list(range(j, 36))
    if c.get("r") != 32 or c.get("lora_alpha") != 64:
        bad.append(f"{d}: adapter r/alpha = {c.get('r')}/{c.get('lora_alpha')} != 32/64")
    if c.get("layers_to_transform") != want:
        bad.append(f"{d}: layers_to_transform != [{j}..35]")
if bad:
    print("GATE B FAILURES:"); [print("  -", b) for b in bad]; sys.exit(1)
print("GATE B PASS: ladder j=6,9,18 match flagship recipe on all 22 fields "
      "except resume_j; adapter spans are [j..35] r32/a64")
PYGATE
note "GATE B PASS ladder recipe-matched"

# --- GATE C: capacity arithmetic for the capacity-matched arm ----------------
"$PYBIN" - <<'PYCAP' || { echo "FATAL GATE C: capacity arithmetic wrong"; exit 8; }
PER = 75776  # sum(fan_in+fan_out) over the 7 LoRA targets, Qwen3-8B
a = PER * 40 * (36 - 12)   # j=12, r=40
b = PER * 32 * (36 - 6)    # j=6,  r=32  (the on-disk arm it must match)
assert a == b == 72_744_960, (a, b)
print(f"GATE C PASS: j12_r40 params == j6_r32 params == {a:,} (exact capacity match)")
PYCAP
note "GATE C PASS capacity arithmetic exact"

note "PROJECT_ROOT=$PROJECT_ROOT"
note "PYBIN=$PYBIN  MODEL=$MODEL  NPROC=$NPROC  TOTAL_STEPS=$TOTAL_STEPS  gc='$GC_FLAG'"
note "trainer md5 = $(md5sum scripts/train_qcmem_distill.py | cut -d' ' -f1)"

# ---------------------------------------------------------------------------
# ARMS: "run_name:resume_j:lora_rank:lora_alpha"
#   A1 = the literal j=0 control (GATE 0 predicts a vacuous objective)
#   A6 = capacity-matched j=12 (72,744,960 params == on-disk j=6)
# ---------------------------------------------------------------------------
ARMS="${ARMS:-a02_j0control_lora_r32:0:32:64 a02_j12_capmatch_r40:12:40:80}"
BASE_PORT="${BASE_PORT:-29891}"

note "START $(date -u +%FT%TZ)"
i=0
for SPEC in $ARMS; do
  IFS=':' read -r RUN J RANK ALPHA <<< "$SPEC"
  OUTPUT_DIR="outputs/$RUN"
  LOG="logs/${RUN}.log"
  PORT=$(( BASE_PORT + i ))
  mkdir -p "$OUTPUT_DIR"
  note "----------------------------------------------------------------------"
  note "=== ARM $RUN : j=$J r=$RANK alpha=$ALPHA -> $OUTPUT_DIR (port $PORT, log $LOG) ==="
  note "arm start $(date -u +%FT%TZ)"

  # NOTE: no --keep_last_n / --keep_steps. The zwfy6 trainer predates the
  # rotation args and argparse-REJECTS them (verified 2026-08-12). Adapters are
  # ~223 MB; 8 saves/arm is ~1.8 GB against 3.2 T free.
  # NOTE: there is NO --eval_interval flag in this trainer and NO inline eval in
  # its step loop (verified by grep), so the NCCL-watchdog SIGABRT failure mode
  # cannot occur. Passing --eval_interval 0 would be a bogus arg and would kill
  # the launch. All eval is offline on the saved adapters.
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
  "$PYBIN" -m torch.distributed.run --nproc_per_node="$NPROC" --master_port="$PORT" \
    scripts/train_qcmem_distill.py \
    --model_path "$MODEL" \
    --resume_j "$J" --top_prepay_b "$TOP_PREPAY_B" \
    --lora_rank "$RANK" --lora_alpha "$ALPHA" --lora_dropout "$LORA_DROPOUT" \
    --lora_targets "$LORA_TARGETS" \
    --pg19_path "$PROJECT_ROOT/data/pg19_train.jsonl" \
    --chunk_size "$CHUNK_SIZE" --n_ctx "$N_CTX" --query_loss_tokens "$QUERY_LOSS_TOKENS" \
    --teacher_topk "$TEACHER_TOPK" --distill_lambda "$DISTILL_LAMBDA" --ce_weight "$CE_WEIGHT" \
    --total_steps "$TOTAL_STEPS" --lr "$LR" --warmup_steps "$WARMUP" \
    --weight_decay "$WEIGHT_DECAY" --grad_accum "$GRAD_ACCUM" --grad_clip "$GRAD_CLIP" \
    $GC_FLAG \
    --output_dir "$OUTPUT_DIR" --save_interval "$SAVE_INTERVAL" --log_interval "$LOG_INTERVAL" \
    --dtype "$DTYPE" --attn_impl "$ATTN_IMPL" --seed "$SEED" \
    --wandb_project mixture-of-memory --wandb_run_name "" \
    >"$LOG" 2>&1
  rc=$?
  note "arm $RUN finished rc=$rc  $(date -u +%FT%TZ)"
  if [[ $rc -ne 0 ]]; then
    note "WARNING: arm $RUN exited non-zero (rc=$rc). See $LOG. Continuing."
  else
    # record the measured cadence from a REAL window (never the first 1-2 lines)
    "$PYBIN" - "$LOG" "$RUN" <<'PYRATE' | tee -a "$PROG"
import re, sys
log, run = sys.argv[1], sys.argv[2]
steps = [(int(m.group(1)), float(m.group(2))) for m in
         (re.search(r'step (\d+)/\d+ loss ([\d.]+)', l) for l in open(log, errors='ignore')) if m]
if steps:
    print(f"[{run}] loss first={steps[0][1]:.4f}@step{steps[0][0]} "
          f"mid={steps[len(steps)//2][1]:.4f}@step{steps[len(steps)//2][0]} "
          f"last={steps[-1][1]:.4f}@step{steps[-1][0]}  n_log={len(steps)}")
PYRATE
  fi
  i=$(( i + 1 ))
done

note "ALL ARMS DONE $(date -u +%FT%TZ)"
note "next: offline eval of A0/A1/A2/A3/A4/A5/A6 on RULER (retrieval-closed) per PREREG 2.6"
