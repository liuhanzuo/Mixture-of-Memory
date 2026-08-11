#!/usr/bin/env bash
# ============================================================================
# SparseForge, token-matched to the CAST reproduction. Node .21 (8x L20A, wzc1).
#
#   ARM=slorb   bash scripts/_run_sparseforge_tokenmatched.sh    # arm 2
#   ARM=noslorb bash scripts/_run_sparseforge_tokenmatched.sh    # arm 3
#
# WHY THIS RUN EXISTS
# -------------------
# The published SparseForge 5B arm cannot be compared to CAST-repro as it stands.
# Three separate confounds, each independently disqualifying:
#
#   (1) DATA. It is link 3 of a resume chain whose last leg trained on
#       `qa_format_sft_llama` -- 8 multiple-choice-QA benchmark TRAIN splits,
#       129,752,281 tokens, repeat=3, traversed ~144.7x, and containing
#       race_middle+race_high while `race` IS a CAST-7 eval task.
#       (baselines/cast_repro/AST_VS_SPARSEFORGE_DATA_CONFOUND.md)
#   (2) BUDGET. 17,900 x 256 x 4096 = 18,769,510,400 tokens nominal, on top of
#       two prior 17k+3k links. CAST-repro used 7,864,320,000. The "5B" label is
#       wrong in both directions.
#   (3) CAPACITY. SLoRB adds 848,429,056 live dense params (+13.10% over the
#       6,476,005,376 in-scope weights) and the published "2:4" checkpoint is not
#       2:4 at all -- zero_frac 0.000000000.
#       (baselines/cast_repro/SPARSEFORGE_SAME_HARNESS.md)
#
# These arms fix (1) and (2) by construction: same corpus, same tokenizer, same
# token count, same seq-len, same global batch, same harness as
# scripts/_run_cast_direct.sh -> outputs/cast_repro_zero2 (ppl@4096 6.1372,
# CAST-7 58.39 plain acc). What remains different is the mechanism, which is the
# thing we actually want to measure.
#
#   arm 2 (slorb)   vs CAST-repro  = mask machinery + SLoRB capacity
#   arm 3 (noslorb) vs CAST-repro  = mask machinery alone
#   arm 2 vs arm 3                 = SLoRB's own contribution
#
# ** Arm 3 is a FRESH TRAINING RUN with SLoRB=False, NOT a post-hoc amputation of
#    an SLoRB-trained checkpoint. ** Dropping the branch from weights that were
#    trained with it costs ~4.9 pp AST-7 and is a statement about amputation
#    damage, not about the method. The user was explicit about this. The existing
#    `hard_drop` export stays labelled as amputation damage and is NOT arm 3.
# ============================================================================
set -u

ARM="${ARM:-}"
case "$ARM" in
  slorb)   SLORB=True  ;;
  noslorb) SLORB=False ;;
  *) echo "FATAL: set ARM=slorb or ARM=noslorb"; exit 2 ;;
esac

ROOT="${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code}"
MOM="$ROOT/Mixture-of-Memory"
PY="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
TR="${TORCHRUN_BIN:-/opt/conda/envs/torch-base/bin/torchrun}"

# The corpus lives under Mixture-of-Memory/data/ but main_llama.py lives at
# $ROOT. Before the --data_root flag existed, data_dir was hardcoded
# os.path.join('data', dataset) relative to CWD, so this dir was unreachable
# without a symlink or a '../' hack. Pass it explicitly instead.
DATA_ROOT="$MOM/data"
DATASET=dolmino-mix-1124-llama2
DATA="$DATA_ROOT/$DATASET"
MODEL=models/Llama--Llama2-7b          # resolved relative to $ROOT by the trainer
OUT_DIR="out_llama_tokenmatched_$ARM"
LOG="$MOM/logs/sparseforge_tokenmatched_${ARM}_$(date +%m%d_%H%M%S).log"

# ---------------------------------------------------------------- preflight
[ -f "$DATA/metadata.json" ] || { echo "FATAL: $DATA/metadata.json missing"; exit 3; }
[ -f "$DATA/train.bin" ]     || { echo "FATAL: $DATA/train.bin missing"; exit 3; }
[ -f "$DATA/val.bin" ]       || { echo "FATAL: $DATA/val.bin missing"; exit 3; }
[ -d "$ROOT/$MODEL" ]        || { echo "FATAL: model not at $ROOT/$MODEL"; exit 4; }
[ -f "$ROOT/main_llama.py" ] || { echo "FATAL: trainer not at $ROOT/main_llama.py"; exit 4; }

# dtype is read from metadata.json by the trainer, but assert it here too so a
# silently re-tokenized corpus cannot slip past. dolmino-llama2 is uint32:
# train.bin 310,886,663,436 B / 4 = 77,721,665,859 = metadata total_tokens.
# Reading it as uint16 does NOT error -- it doubles the stream and injects a zero
# after every real token, and every value stays < 32000 so the vocab guard is
# blind to it. Measured: 50.0% zeros, 0% out-of-vocab.
DTYPE=$("$PY" -c "import json;print(json.load(open('$DATA/metadata.json'))['dtype'])")
[ "$DTYPE" = "uint32" ] || { echo "FATAL: expected uint32 corpus, metadata says '$DTYPE'"; exit 5; }
echo "preflight: corpus dtype=$DTYPE (asserted), passing --data_dtype uint32 explicitly"

# ---------------------------------------------------------------------------
# BUDGET: exactly CAST-repro's token count.
#   7500 iters x global_batch 256 x block_size 4096 = 7,864,320,000 tokens
# matching outputs/cast_repro_zero2/run_manifest.json "total_tokens": 7864320000.
#
# final_finetune_iters 0 -- deliberately. The published run's 3000-iter final
# stage is EXACTLY where the MC-QA contamination entered: link 3 switched
# `dataset` to qa_format_sft_llama for it. Running it here would either
# re-import the contamination or add 3000 uncounted iters (+3.1B tokens) that
# CAST never got. With 0, main_llama.py:3436 finalizes the mask, saves, and
# breaks cleanly (verified: the `extra > 0` branch that rebuilds the optimizer
# is simply skipped).
# ---------------------------------------------------------------------------
MAX_ITERS=7500
GLOBAL_BATCH=256
BLOCK=4096
MICRO=8                # -> grad_accum 256/8 = 32 total, 4 per rank at ws=8.
                       # Same as the published run. 256 % 8 == 0 so the
                       # divisibility assert at main_llama.py:1616 passes.
FINAL_FT=0

# ---------------------------------------------------------------------------
# ** THE SINGLE MOST DANGEROUS RESCALE IN THIS SCRIPT -- READ THIS. **
#
# The published mask-hardening schedule is calibrated to a 17000-iter horizon:
#   mask_hardening_start 12000 + mask_hardening_duration 5000 = 17000 exactly,
# i.e. hardening_x anneals 1.0 -> 0.0 and lands on 0 precisely at max_iters
# (sparse_modeling.py:1656-1672).
#
# Copying 12000/5000 onto a 7500-iter run would put the START of hardening 4500
# iters PAST THE END OF TRAINING. hardening_x would sit at 1.0 for every step,
# the mask would stay fully SOFT for the entire run, and the forward pass would
# never see a 2:4 projection (sparse_modeling.py:787 takes the
# `effective_mask = self.mask` early-out whenever hx >= 1.0-1e-6). The run would
# complete, report a healthy loss curve, save a checkpoint, and measure NOTHING
# about structured sparsity. Verified numerically:
#   hardening_x at steps 0/3750/7000/7500 with (12000,5000) = 1.0/1.0/1.0/1.0
#
# So both are rescaled by 7500/17000 = 0.441176, preserving the invariant
# "hardening completes exactly at max_iters":
#   12000 * 0.441176 = 5294      (soft-only phase = 70.6% of the run, as published)
#    5000 * 0.441176 = 2206      (anneal window   = 29.4% of the run, as published)
#   5294 + 2206 = 7500           <- lands exactly on max_iters, delta 0
# Verified: hardening_x = 1.0 at 5293, 0.5 at 6397, 0.0 at 7500.
#
# NOTE hardening_period 0 (as published) disables the OTHER, independent
# harden_fraction() path at main_llama.py:3044, so this schedule is the only
# hardening mechanism in play. Do not "helpfully" set hardening_period > 0.
# ---------------------------------------------------------------------------
HARDEN_START=5294
HARDEN_DUR=2206

# ---------------------------------------------------------------------------
# LR SCHEDULE -- a genuine judgement call, stated explicitly.
#
# Published (17000 horizon): lr 1e-4, min_lr 1e-5, warmup 2000, lr_decay 15000.
# Note lr_decay_iters 15000 < max_iters 17000, so the published run spent its
# last 2000 iters pinned at min_lr (get_lr's `it > lr_decay_iters` branch) --
# which is exactly the window where hardening was finishing. That coupling
# (mask goes hard while LR is at its floor) is plausibly load-bearing: it stops
# the pruned-away weights from being kicked back to O(lr) after binarisation.
#
# CHOICE: proportional rescale of BOTH, preserving that coupling.
#   warmup_iters   2000 * 0.441176 = 882
#   lr_decay_iters 15000 * 0.441176 = 6618      (= 88.24% of max_iters, as published)
# so the last 882 iters again sit at min_lr, and hardening again completes
# inside the min_lr tail. lr and min_lr themselves are UNCHANGED (1e-4 / 1e-5)
# because they are magnitudes, not schedule positions.
#
# THE ALTERNATIVE, and why it was rejected: keep warmup 2000 absolute. That
# would spend 26.7% of a 7500-iter run in warmup versus 11.8% in the original,
# systematically handicapping the arm we are supposedly giving a fair shot. It
# also decouples warmup from the mask-update switch at 2000.
#
# HOW TO DEFEND THIS TO A REVIEWER: the honest statement is "the schedule was
# rescaled proportionally to the compressed horizon; no LR search was run at
# 7500 iters for either arm." The comparison to CAST-repro is still not
# LR-matched in absolute terms -- CAST-repro used lr 2e-5 / min 2e-6 / warmup
# 375, i.e. a 5x smaller peak LR -- because each method keeps ITS OWN published
# optimiser hyperparameters. That is the standard choice, and it is the reason
# the primary claim must be about the mask mechanism at matched data+budget, not
# "SparseForge is better tuned". If a reviewer demands LR-matching, that is a
# third pair of runs, not a tweak to these.
# ---------------------------------------------------------------------------
LR=1e-4
MIN_LR=1e-5
WARMUP=882
LR_DECAY=6618

# Other 17000-calibrated schedule knobs, rescaled by the same 0.441176.
# Leaving these at their published absolute values would misplace them just as
# badly as the hardening schedule, only less visibly.
MASK_SWITCH=882          # 2000  -> mask_update_switch_step
SPARSITY_WARMUP=221      # 500   -> sparsity_warmup_steps
BETA_STRUCT_START=882    # 2000  -> beta_structural_start
BETA_STRUCT_END=7500     # 17000 -> beta_structural_end (= max_iters, as published)
INCREASE_STEP=4412       # 10000 -> increase_step (inert while srste_decay=0, rescaled anyway)

SEED=1234                # matches outputs/cast_repro_zero2/run_manifest.json "seed": 1234
                         # (main_llama.py default is 1337; CAST used 1234, so use 1234)

cd "$ROOT" || exit 1
mkdir -p "$MOM/logs"

echo "############ SparseForge token-matched, ARM=$ARM (SLoRB=$SLORB)"
echo "############ tokens = $MAX_ITERS x $GLOBAL_BATCH x $BLOCK = $(( MAX_ITERS * GLOBAL_BATCH * BLOCK ))"
echo "############ out_dir=$OUT_DIR  log=$LOG"
echo "############ start $(date -Is)"

# expandable_segments must be exported HERE, not in the calling shell: torchrun
# spawns children and an env var set outside the `setsid nohup bash ...` wrapper
# does not reliably reach them.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=bond1
export GLOO_SOCKET_IFNAME=bond1
export WANDB_MODE=offline          # no network dependency inside a 31 h run

"$TR" --nproc_per_node 8 --master_port 29620 main_llama.py \
  --student_model "$MODEL" --teacher_model "$MODEL" --distill_model True \
  --dataset "$DATASET" --data_root "$DATA_ROOT" --data_dtype uint32 \
  --out_dir "$OUT_DIR" \
  --block_size $BLOCK --global_batch_size $GLOBAL_BATCH --batch_size $MICRO \
  --max_iters $MAX_ITERS --final_finetune_iters $FINAL_FT \
  --learning_rate $LR --min_lr $MIN_LR --warmup_iters $WARMUP --lr_decay_iters $LR_DECAY \
  --weight_decay 0.1 \
  --seed $SEED \
  \
  `# ---- mask machinery: SparseForge's own published values (this IS the treatment) ----` \
  --mode sparse_forward --mask_type unstructured --hard_mask_type nm_2_4 \
  --mask_metric hessian_obd --change_mask True \
  --sparsity_ratio 0.5 --structured_n 2 --structured_m 4 --structured_exact False \
  --mask_penalty_mode nm_2_4 --mask_penalty_lr 0.0 \
  --mask_lr 0.05 --mask_update_period 10 \
  --mask_update_switch_step $MASK_SWITCH \
  --mask_update_period_before 10 --mask_update_period_after 10 \
  --beta 0.98 --score_ema_beta 0.99 \
  --temp_init 2.0 --temp_min 0.05 --temp_decay 0.98 \
  --sparsity_alpha 0.2 --sparsity_warmup_steps $SPARSITY_WARMUP \
  --lambda_mid_max 0.3 --tau_sample_size 262144 \
  --freeze_low 0.0 --freeze_high 1.0 \
  --beta_structural_start $BETA_STRUCT_START --beta_structural_end $BETA_STRUCT_END \
  --glu_joint_mask False --weight_scaling False \
  \
  `# ---- hardening: RESCALED to the 7500 horizon. See the block above. ----` \
  --mask_hardening_start $HARDEN_START --mask_hardening_duration $HARDEN_DUR \
  --hardening_period 0 --hardening_fraction 0.2 \
  \
  `# ---- distillation: logit KL only, as published (squarehead 0.0) ----` \
  --hardness_task 1.0 --hardness_kldiv 1.0 --hardness_squarehead 0.0 \
  \
  `# ---- SLoRB: THE ONLY DIFFERENCE BETWEEN THE TWO ARMS ----` \
  --SLoRB $SLORB --SLoRB_k 16 --SLoRB_init_type sum --trainable_projection True \
  \
  `# ---- decay paths held at published values. srste_decay 0.0 and` \
  `#      adaptive_l1_decay 0.0 mean NO optimizer path reads p.mask; see the` \
  `#      FSDP section of SPARSEFORGE_TOKENMATCHED_PREP.md -- this is why` \
  `#      SparseForge's FSDP use does not reproduce CAST's misalignment bug. ----` \
  --srste_decay 0.0 --adaptive_l1_decay 0.0 --increase_step $INCREASE_STEP \
  --cast_mode False \
  \
  `# ---- enable_hutchinson False, matching the published run. Do NOT flip this` \
  `#      on here: it is a claimed contribution that the published checkpoint` \
  `#      never used, so enabling it would confound the comparison AND silently` \
  `#      disable gradient checkpointing (main_llama.py forces it off for` \
  `#      double-backward), blowing the memory budget. ----` \
  --enable_hutchinson False \
  \
  --use_fsdp True --fsdp_mode hybrid_sharded --fsdp_mixed_precision True \
  --fsdp_cpu_offload False --gradient_checkpointing True --dtype auto \
  \
  `# ---- eval/save. eval_interval 100 as published. Inline lm_eval is left ON` \
  `#      to reproduce the published best-checkpoint selection rule; set` \
  `#      --finalize_lm_eval False if the 31 h estimate needs trimming. ----` \
  --eval_interval 100 --eval_iters 20 --log_interval 10 --output_flip_every 10 \
  --skip_wiki_ppl False --finalize_lm_eval True --lm_eval_batch_size 64 \
  --lm_eval_tasks "hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,piqa,race" \
  \
  --resume False \
  --wandb_logging False \
  > "$LOG" 2>&1

rc=$?
echo "############ ARM=$ARM finished rc=$rc $(date -Is)"
echo "############ log: $LOG"
exit $rc
