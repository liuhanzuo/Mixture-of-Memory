#!/bin/bash
# Paper B #99: keep14+fresh2 DISTILL heal, resumed on .212 (8xB200 sm_100, wzc1 disk).
# OLMo-2-7B base 32L teacher -> keep14 (16L) student; loss = NTP + 0.6*KL(top-k=64).
#
# ============================ WHY THIS UNBLOCKS #99 ============================
# PENDING_TASKS #99 was BLOCKED for two stated reasons. Both are now discharged:
#
# 1. "bnb locks this trainer to .73/.104" -- FALSE on B200 as of 2026-08-15.
#    train_olmo2_arch_probe2_distill.py:63 imports bitsandbytes at MODULE level and
#    hardcodes bnb.optim.AdamW8bit (line 628), and the source comment says the 8-bit
#    optimizer exists only "to fit keep14 train-all + teacher in H20 95GB".
#    bitsandbytes 0.50.1 was pip-installed on .212 and MEASURED: AdamW8bit
#    constructs AND steps on sm_100 (capability (10,0)). So we keep bnb and get a
#    FAITHFUL resume (optimizer state in the ckpt is bnb 8-bit format; switching to
#    fp32 AdamW would have forced a step-0 restart and discarded 5000 steps).
#
# 2. "--save_every 5000 vs budget: 3rd run in a row would burn the budget and save
#    NOTHING" -- this was the real blocker and it was a CADENCE problem, not a node
#    problem. The 07-31 run died at step5200 and the 08-05 run reached step7780;
#    both saved nothing because resume started exactly at 5000 and the next save was
#    at 10000. FIX: --save_every 500. On B200 that is a checkpoint every ~30-40 min.
#    (Deviation from the H20 recipe's 5000 is a CHECKPOINT CADENCE change only; it
#    does not touch the optimization path, so the arm stays comparable.)
#
# ============================== DATA (comparability) ==========================
# /dev/shm/dolmino_now15b_wzc1.npy, REBUILT on .212 from the 84 wzc1 shards in 150 s
# (scripts/build_dolmino_corpus_wzc1.py). rows=15,491,607 and
# md5 7df19b217e5b0670d58bf6e01e6559d0 -- MEASURED IDENTICAL to the zwfy6 corpus that
# keep8/keep12/keep14-NTP train on. The md5 assert below is what makes this arm
# comparable; do NOT relax it to a rows/size check.
# ⚠️ Do NOT use data/dolmino_now15b.npy -- that is a 7,570,911-row PARTIAL PREFIX.
# ⚠️ /dev/shm is tmpfs and node-local: it does NOT survive reboot and is NOT shared
#    with LOCAL even though both are wzc1. Rerun the builder after a reboot.
#
# ============================== BATCH / MEMORY ================================
# eff_batch = batch_size * grad_accum * world_size = 16 * 1 * 8 = 128, IDENTICAL to
# the H20 distill recipe (BS=16 GA=1 NPROC=8), so the optimization path is unchanged.
# Static per-rank footprint (bs-independent): student fp32 15.13 + grads fp32 15.13
# + AdamW8bit 7.56 + teacher bf16 13.59 = 51.41 GiB. H20 peaked 94.6/97.8 GiB at
# bs=16 with checkpointing. See GRADIENT_CHECKPOINTING note below.
#
# NOT 16 CARDS (LOCAL+.212): plain DDP all-reduces 15.13 GB of fp32 grads every step;
# measured LOCAL<->.212 TCP 14.4 Gbps makes cross-node exchange cost more than a whole
# step. One node is faster AND safer. (Same reasoning as launch_keep10_resume_b200_local_0815.sh.)
#
# There is NO --eval_interval flag in this trainer and NO inline eval code, so the
# known inline-BABILong NCCL-desync hazard is structurally absent here.
set -e

PROJECT_ROOT=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory
PYTHON=/opt/conda/envs/torch-base/bin/python   # .venv has no torch; conda = torch 2.13.0 / sm_100
cd $PROJECT_ROOT

export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT_DIR=$PROJECT_ROOT/outputs/olmo2_probe2_7B_keep14fresh2_distill
LOG_FILE=$PROJECT_ROOT/logs/olmo2_7B_keep14_distill_212_0815.log
DATA_PATH=/dev/shm/dolmino_now15b_wzc1.npy
EXPECTED_ROWS=15491607
EXPECTED_MD5=7df19b217e5b0670d58bf6e01e6559d0
TEACHER=/apdcephfs_wzc1/share_304376610/pighzliu_code/models/OLMo-2-1124-7B

BS=${BS:-16}
GA=${GA:-1}
GC=${GC:-1}
SAVE_EVERY=${SAVE_EVERY:-500}

mkdir -p "$OUT_DIR" logs
echo "=== keep14-distill heal resume on .212 (8xB200) started at $(date) ===" | tee -a $LOG_FILE

# ---- bnb preflight: the module-level import must succeed AND actually step ----
$PYTHON - <<'EOF' || { echo "FATAL: bnb AdamW8bit unusable"; exit 1; }
import torch, bitsandbytes as bnb
p = torch.nn.Parameter(torch.randn(256, 256, device='cuda'))
o = bnb.optim.AdamW8bit([{'params': [p], 'weight_decay': 0.1}], lr=1e-4, betas=(0.9, 0.95), eps=1e-8)
p.sum().backward(); o.step(); torch.cuda.synchronize()
print("bnb", bnb.__version__, "AdamW8bit OK on", torch.cuda.get_device_capability())
EOF

# ---- data preflight: rows AND md5 (md5 is the cross-disk comparability proof) ----
if [ ! -f "$DATA_PATH" ]; then
  echo "FATAL: $DATA_PATH missing (tmpfs is wiped by reboot)." | tee -a $LOG_FILE
  echo "       Rebuild: $PYTHON scripts/build_dolmino_corpus_wzc1.py" | tee -a $LOG_FILE
  exit 1
fi
DATA_ROWS=$($PYTHON -c "import numpy as np; print(np.load('$DATA_PATH', mmap_mode='r').shape[0])")
[ "$DATA_ROWS" = "$EXPECTED_ROWS" ] || { echo "FATAL: rows=$DATA_ROWS expected $EXPECTED_ROWS" | tee -a $LOG_FILE; exit 1; }
DATA_MD5=$(md5sum "$DATA_PATH" | awk '{print $1}')
[ "$DATA_MD5" = "$EXPECTED_MD5" ] || { echo "FATAL: corpus md5=$DATA_MD5 expected $EXPECTED_MD5" | tee -a $LOG_FILE; exit 1; }
echo "Data verified: rows=$DATA_ROWS md5=$DATA_MD5" | tee -a $LOG_FILE

# ---- teacher preflight: must be the 32L HF DIRECTORY, not a .pt ----
TLAYERS=$($PYTHON -c "import json;print(json.load(open('$TEACHER/config.json'))['num_hidden_layers'])")
[ "$TLAYERS" = "32" ] || { echo "FATAL: teacher has $TLAYERS layers, expected 32" | tee -a $LOG_FILE; exit 1; }
echo "Teacher verified: $TEACHER ($TLAYERS layers)" | tee -a $LOG_FILE

# ---- ckpt preflight: DISCOVER newest, assert >=5000, assert optimizer present ----
NEWEST=$(ls -t $OUT_DIR/step*.pt 2>/dev/null | head -1)
[ -n "$NEWEST" ] || { echo "FATAL: no step*.pt in $OUT_DIR" | tee -a $LOG_FILE; exit 1; }
NEWSTEP=$(basename "$NEWEST" | tr -dc '0-9')
[ "$NEWSTEP" -ge 5000 ] || { echo "FATAL: newest step$NEWSTEP < 5000 frontier" | tee -a $LOG_FILE; exit 1; }
if [ "$NEWSTEP" -ge 200000 ]; then echo "DONE: step$NEWSTEP >= 200000" | tee -a $LOG_FILE; exit 0; fi
echo "Resuming from DISCOVERED newest ckpt: $NEWEST (step $NEWSTEP)" | tee -a $LOG_FILE

echo "eff_batch=$((BS*GA*8)) save_every=$SAVE_EVERY gradient_checkpointing=$GC" | tee -a $LOG_FILE

exec $PYTHON -m torch.distributed.run --standalone --nproc_per_node 8 \
  scripts/train_olmo2_arch_probe2_distill.py \
  --resume_from "$NEWEST" \
  --data_path $DATA_PATH \
  --output_dir $OUT_DIR \
  --model_path $TEACHER \
  --keep_front_layers 14 \
  --n_fresh_layers 2 \
  --distill_teacher_model $TEACHER \
  --distill_lambda 0.6 \
  --teacher_topk 64 \
  --lr 1e-4 \
  --lr_inherited 2e-5 \
  --min_lr 1e-5 \
  --min_lr_inherited 2e-6 \
  --batch_size $BS \
  --grad_accumulation_steps $GA \
  --seq_len 2048 \
  --max_steps 200000 \
  --warmup_steps 150 \
  --weight_decay 0.1 \
  --grad_clip 1.0 \
  --save_every $SAVE_EVERY \
  --log_every 20 \
  --gradient_checkpointing $GC \
  >>$LOG_FILE 2>&1
