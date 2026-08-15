#!/bin/bash
# Launch Paper B keep10+fresh2 resume on LOCAL (8xB200 sm_100, wzc1 disk), 2026-08-15.
#
# WHY THIS SCRIPT EXISTS
#   keep10 was the only degraded arm with no node: .73 runs keep12, .82 runs keep8,
#   .104 runs the paperC Qwen probe. paperC/proposal were verified to have ZERO
#   pending GPU work (ready_queue.py -> 0 ready_gpu), so per CLAUDE.md the Paper B
#   resume is allowed to run.
#
# THE TWO CROSS-DISK ASSETS, BOTH RESOLVED WITH VERIFIED md5 (2026-08-15)
#   1. ckpt: the newest keep10 ckpt (step90000, 2026-08-12) lived only on zwfy6.
#      Pulled with 6 parallel `ssh dd` byte-range streams from .82 in 6.7 min
#      (~92 MB/s aggregate; a single stream measures only 17.7 MB/s).
#      md5 0112936ed6bb1e3549269bb8b6461a17 IDENTICAL on both disks.
#      (wzc1 also holds an OLDER step83500.pt -- resuming from that would silently
#      discard 6500 completed steps AND fork from the zwfy6 run of the same name.
#      The preflight below therefore DISCOVERS the newest ckpt and asserts >=90000.)
#   2. corpus: the 15,491,607-row dolmino corpus (118 GiB) also lived only on zwfy6
#      -- wzc1's data/dolmino_now15b.npy is a 7,570,911-row PARTIAL PREFIX and must
#      NOT be used. But wzc1 DOES hold all 84 source shards, so the corpus was
#      REBUILT LOCALLY instead of transferred (153 s vs ~3.2 h over the wire):
#          concat(sorted 84 shards) = 15,495,703 rows
#          [0:4096]  == data/dolmino_now_val.npy   (val split)
#          [4096:]   == the 15,491,607-row training corpus
#      Builder: scripts/build_dolmino_corpus_wzc1.py -> /dev/shm/dolmino_now15b_wzc1.npy
#      md5 7df19b217e5b0670d58bf6e01e6559d0 IDENTICAL to .82:/dev/shm/dolmino_now15b.npy.
#      The md5 assert below is what makes the arm comparable to keep8/keep12; do not
#      relax it to a size/row check.
#
# HYPERPARAMETERS ARE UNCHANGED FROM THE H20 RECIPE except batch/accum:
#   eff_batch = batch_size(16) * grad_accum(1) * world_size(8) = 128, identical to
#   the H20 arms' 4*4*8. This is mathematically EXACT, not merely "equivalent":
#   DistributedSampler yields the same per-rank index permutation, drop_last leaves
#   121028 optimizer steps/epoch in BOTH configs, and one optimizer step consumes
#   the same 16 consecutive per-rank indices either way (mean-of-4-means == mean-of-16).
#   Raising batch is what uses the B200's 178 GiB (H20 peaked at 73.5 GiB of 97.8).
#
# NOT 16 CARDS (LOCAL + .212), despite both B200 boxes being idle and same-disk:
#   this trainer is plain DDP, so every optimizer step all-reduces 13.0 GB of fp32
#   grads (26.0 GB ring traffic/rank). Measured LOCAL<->.212 TCP is 14.4 Gbps =>
#   ~14.5 s/step of pure network, ~3.6 s/step even assuming 4x multi-stream --
#   larger than the entire step. 8 cards on one node is both faster AND safer.
#
# NO bitsandbytes FLAGS: --optimizer defaults to fp32 torch AdamW. bnb is absent on
#   B200 and is only imported under `--optimizer bnb_adamw8bit`
#   (train_olmo2_arch_probe2.py:496). Do NOT confuse this trainer with
#   train_olmo2_arch_probe2_distill.py, which imports bnb at module level.
#
# There is no --eval_interval flag in this trainer (no inline BABILong eval), so the
#   known inline-eval NCCL desync hazard does not apply here.

set -e

PROJECT_ROOT=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory
PYTHON=/opt/conda/envs/torch-base/bin/python   # LOCAL .venv has no torch; conda = torch 2.13.0 / sm_100

export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN

cd $PROJECT_ROOT

LOG_FILE=$PROJECT_ROOT/logs/olmo2_7B_keep10fresh2_resume200k_local_0815.log

echo "=== keep10+fresh2 resume on LOCAL (8xB200) started at $(date) ===" | tee $LOG_FILE

# ---- data preflight: rows AND md5 (md5 is the cross-disk identity proof) ----
DATA_PATH=/dev/shm/dolmino_now15b_wzc1.npy
EXPECTED_ROWS=15491607
EXPECTED_MD5=7df19b217e5b0670d58bf6e01e6559d0

if [ ! -f "$DATA_PATH" ]; then
  echo "FATAL: $DATA_PATH not found. Rebuild it from the 84 wzc1 shards:" | tee -a $LOG_FILE
  echo "       $PYTHON scripts/build_dolmino_corpus_wzc1.py   (see header for the verified recipe)" | tee -a $LOG_FILE
  exit 1
fi
DATA_ROWS=$($PYTHON -c "import numpy as np; print(np.load('$DATA_PATH', mmap_mode='r').shape[0])")
if [ "$DATA_ROWS" != "$EXPECTED_ROWS" ]; then
  echo "FATAL: dataset rows=$DATA_ROWS expected $EXPECTED_ROWS" | tee -a $LOG_FILE; exit 1
fi
DATA_MD5=$(md5sum "$DATA_PATH" | awk '{print $1}')
if [ "$DATA_MD5" != "$EXPECTED_MD5" ]; then
  echo "FATAL: corpus md5=$DATA_MD5 expected $EXPECTED_MD5 (NOT the zwfy6 corpus keep8/keep12 train on)" | tee -a $LOG_FILE; exit 1
fi
echo "Data verified: $DATA_PATH rows=$DATA_ROWS md5=$DATA_MD5" | tee -a $LOG_FILE

# ---- ckpt preflight: DISCOVER newest, then assert (never trust a pinned filename)
OUT_DIR=$PROJECT_ROOT/outputs/olmo2_probe2_7B_keep10fresh2
NEWEST=$(ls -t $OUT_DIR/step*.pt 2>/dev/null | head -1)
if [ -z "$NEWEST" ]; then
  echo "FATAL: no step*.pt in $OUT_DIR" | tee -a $LOG_FILE; exit 1
fi
NEWSTEP=$(basename "$NEWEST" | tr -dc '0-9')
if [ "$NEWSTEP" -lt 90000 ]; then
  echo "FATAL: newest ckpt step$NEWSTEP < 90000 (the zwfy6 frontier) -- refusing to lose steps" | tee -a $LOG_FILE; exit 1
fi
if [ "$NEWSTEP" -ge 200000 ]; then
  echo "DONE: step$NEWSTEP already >= max_steps 200000; nothing to resume" | tee -a $LOG_FILE; exit 0
fi
CKPT="$NEWEST"
echo "Resuming from DISCOVERED newest ckpt: $CKPT (step $NEWSTEP)" | tee -a $LOG_FILE

torchrun \
  --standalone \
  --nproc_per_node 8 \
  scripts/train_olmo2_arch_probe2.py \
  --resume_from $CKPT \
  --keep_front_layers 10 \
  --n_fresh_layers 2 \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --seq_len 2048 \
  --lr 2e-5 \
  --min_lr 2e-6 \
  --lr_inherited 2e-5 \
  --max_steps 200000 \
  --warmup_steps 150 \
  --weight_decay 0.1 \
  --gradient_checkpointing 1 \
  --save_every 500 \
  --milestone_every 5000 \
  --keep_last_n 3 \
  --keep_milestones 8 \
  --keep_steps 83500,90000,121000,124000,150000,175000,200000 \
  --data_path $DATA_PATH \
  --output_dir $OUT_DIR \
  --model_path /apdcephfs_wzc1/share_304376610/pighzliu_code/models/OLMo-2-1124-7B \
  2>&1 | tee -a $LOG_FILE
