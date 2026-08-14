#!/usr/bin/env bash
# ============================================================================
# Task #245 GATE0 — 20-step ALPS+SLoRB alignment probe.
#
#   bash Mixture-of-Memory/scripts/_run_alps_slorb_gate0.sh
#
# PURPOSE (and the only purpose): prove the external ALPS 2:4 mask is actually
# INSTALLED on all 224 SparseLinear modules and paired with the parameters the
# optimizer touches, before anyone spends GPU-hours on a long run.
#
# WHY A PROBE IS NECESSARY EVEN THOUGH THIS PATH LOOKS SAFE
# ---------------------------------------------------------
# status/SRSTE_SILENT_DEGRADATION_HAZARD.md: adamw.py's SR-STE branch does
# `else: mask = None` on a mask/param shape mismatch -> silently degrades to
# plain Adam, loss curve still looks fine, and CAST_REPRODUCTION_AUDIT.md 4.1
# blames exactly that pattern for a run that only revealed itself as broken via
# a final PPL of 23.45. The CAST branch was hardened to raise; the SR-STE branch
# was not. ALPS+SLoRB uses a FIXED mask and srste_decay=0.0, so it should not
# enter that branch at all -- but "should not" is a configuration property, not
# a code property, so it gets asserted:
#
#   main_llama.py `_load_external_fixed_masks` now
#     (1) prints `GATE0 mask/param alignment: aligned=N/224` and exits non-zero
#         unless N == 224 (override the expectation via
#         SPARSEFORGE_EXPECTED_SPARSELINEAR only with a reason);
#     (2) refuses to run at all if --srste_decay != 0 while a fixed external
#         mask is loaded.
#
# CONFIG PROVENANCE
# -----------------
# SLoRB knobs copied verbatim from the +SLoRB arm's own args.json
# (out_llama_tokenmatched_slorb/models_..._20260814_125037/args.json):
#   SLoRB=True  SLoRB_k=16  SLoRB_init_type=sum  trainable_projection=True
#   srste_decay=0.0  batch_size=8  global_batch_size=256  block_size=4096
#   lr=2e-5  min_lr=2e-6  warmup=375  seed=1234  dataset=dolmino-mix-1124-llama2
#
# DELIBERATE DIFFERENCES FROM THAT ARM, all forced by the fixed-mask design:
#   --change_mask False        the mask is ALPS's and must not be relearned
#   --mask_penalty_lr 0.0      no mask gradient path
#   --mask_hardening_start 0   nothing to harden; the mask is already exact 2:4
#   --freeze_non_slorb True    the reviewer-requested isolation: only
#                              SLoRB_Weight / x_proj train
#   --max_iters 20             probe only
#   --eval_interval 0          inline BABILong/lm_eval eval in a DDP loop
#                              desyncs ranks -> NCCL watchdog SIGABRT (CLAUDE.md)
# ============================================================================
set -u

ROOT="${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code}"
MOM="$ROOT/Mixture-of-Memory"
PY="${PYTHON_BIN:-$ROOT/venv_union9/bin/python}"
TR="${TORCHRUN_BIN:-/opt/conda/envs/torch-base/bin/torchrun}"

# ---------------------------------------------------------------------------
# GPU BUDGET — HARD. Watcher PID 176751 (ARM=slorb, GPUS=4,5,6,7) is live and
# refuses to score if >8000 MiB is held on its half, which would destroy the
# other half of the +/-SLoRB comparison. GPUs 0-3 only.
# ---------------------------------------------------------------------------
GPUS="${GPUS:-0,1,2,3}"
NPROC="${NPROC:-4}"
export CUDA_VISIBLE_DEVICES="$GPUS"

MASK_PATH="${MASK_PATH:-$ROOT/outputs/paper_v2/alps/llama2_wandb_sf_alps_v1_alps_seed0/mask.pt}"
MODEL="${MODEL:-models/Llama--Llama2-7b}"
DATASET="${DATASET:-dolmino-mix-1124-llama2}"
DATA_ROOT="$MOM/data"

MAX_ITERS="${MAX_ITERS:-20}"
MICRO="${MICRO:-8}"
GLOBAL_BATCH="${GLOBAL_BATCH:-256}"
BLOCK="${BLOCK:-4096}"
OUT_DIR="${OUT_DIR:-$ROOT/out_llama_alps_slorb_gate0}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG:-$MOM/logs/alps_slorb_gate0_${STAMP}.log}"

mkdir -p "$MOM/logs" "$OUT_DIR"
cd "$ROOT" || exit 1

if [ ! -f "$MASK_PATH" ]; then
  echo "FATAL: mask artifact missing: $MASK_PATH"; exit 2
fi

echo "############ ALPS+SLoRB GATE0"
echo "############ mask      = $MASK_PATH"
echo "############ gpus      = $CUDA_VISIBLE_DEVICES (nproc=$NPROC)"
echo "############ max_iters = $MAX_ITERS  micro=$MICRO  global=$GLOBAL_BATCH  block=$BLOCK"
echo "############ out_dir   = $OUT_DIR"
echo "############ log       = $LOG"
echo "############ start     = $(date -Is)"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=bond1
export GLOO_SOCKET_IFNAME=bond1
export WANDB_MODE=offline
export SPARSEFORGE_EXPECTED_SPARSELINEAR=224

"$TR" --nproc_per_node "$NPROC" --master_port "${MASTER_PORT:-29648}" main_llama.py \
  --student_model "$MODEL" --teacher_model "$MODEL" --distill_model True \
  --dataset "$DATASET" --data_root "$DATA_ROOT" --data_dtype uint32 \
  --out_dir "$OUT_DIR" \
  --block_size $BLOCK --global_batch_size $GLOBAL_BATCH --batch_size $MICRO \
  --max_iters $MAX_ITERS --final_finetune_iters 0 \
  --learning_rate 2e-5 --min_lr 2e-6 --warmup_iters 375 --lr_decay_iters 7125 \
  --weight_decay 0.1 --seed 1234 \
  \
  `# ---- THE TREATMENT: ALPS's fixed external 2:4 mask + SLoRB-only training ----` \
  --initial_mask_path "$MASK_PATH" \
  --freeze_non_slorb True \
  --SLoRB True --SLoRB_k 16 --SLoRB_init_type sum --trainable_projection True \
  \
  `# ---- mask machinery OFF: the support is ALPS's and is held fixed ----` \
  --mode sparse_forward --mask_type unstructured --hard_mask_type nm_2_4 \
  --mask_metric hessian_obd --change_mask False \
  --sparsity_ratio 0.5 --structured_n 2 --structured_m 4 --structured_exact False \
  --mask_penalty_mode nm_2_4 --mask_penalty_lr 0.0 \
  --mask_lr 0.0 --mask_update_period 1000000 \
  --mask_hardening_start 0 --mask_hardening_duration 0 \
  --hardening_period 0 --hardening_fraction 0.2 \
  --sparsity_alpha 0.0 --sparsity_warmup_steps 0 \
  --lambda_mid_max 0.0 \
  --glu_joint_mask False --weight_scaling False \
  \
  `# ---- distillation: logit KL only, as in both token-matched arms ----` \
  --hardness_task 1.0 --hardness_kldiv 1.0 --hardness_squarehead 0.0 \
  --distill_temperature 2.0 \
  \
  `# ---- srste_decay MUST be 0.0. main_llama.py now RAISES if it is not, when` \
  `#      an external fixed mask is loaded. The argparse DEFAULT is 6e-5, so` \
  `#      omitting this flag would silently arm the buggy branch. ----` \
  --srste_decay 0.0 --adaptive_l1_decay 0.0 --increase_step 4412 \
  --cast_mode False --enable_hutchinson False \
  \
  --use_fsdp True --fsdp_mode hybrid_sharded --fsdp_mixed_precision True \
  --fsdp_cpu_offload False --gradient_checkpointing True --dtype auto \
  \
  `# ---- eval_interval 0: inline eval inside DDP desyncs ranks -> NCCL abort.` \
  `#      output_flip_every 1: the tqdm postfix loss is ONLY refreshed inside` \
  `#      "if iter_num % args.output_flip_every == 0" (main_llama.py:2917), so a` \
  `#      large value freezes the displayed loss at its iter-0 value and the run` \
  `#      looks stuck when it is fine. Keep it at 1 for a 20-step probe. ----` \
  --eval_interval 0 --eval_iters 20 --log_interval 1 --output_flip_every 1 \
  --skip_wiki_ppl True --finalize_lm_eval False \
  \
  --resume False \
  --wandb_logging False \
  > "$LOG" 2>&1

rc=$?
echo "############ GATE0 finished rc=$rc $(date -Is)"
echo "############ log: $LOG"
grep -E "GATE0 mask/param alignment|Loaded [0-9]+ fixed masks|SLoRB-only isolation" "$LOG" || true
exit $rc
