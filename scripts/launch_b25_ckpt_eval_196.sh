#!/bin/bash
# b25/c512 intermediate ckpt early-eval (overtrain-degradation test) on .196
# .196 shares 盘A FS with local host → reads ckpts at the same path post-rsync
set -e
ROOT=/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory
cd "$ROOT"
export WANDB_MODE=offline
CFG=outputs/mem_space_fifo_b25_chunk512/adapter_config.json
# eval step500 + step1000, W0 first (W6 chained after)
CK500=outputs/mem_space_fifo_b25_chunk512/full_model_step000500.pt
CK1000=outputs/mem_space_fifo_b25_chunk512/full_model_step001000.pt

# W0: both ckpts in one task-pool (2 ckpt × 3 task × 7 len = 42 tasks)
RUN_PREFIX=b25_c512_ckpt_W0 \
CKPT_FILES="$CK500 $CK1000" \
CK_NAMES="b25_c512_step500_W0 b25_c512_step1000_W0" \
ADAPTER_CONFIG="$CFG" CHUNK_SIZE=512 \
EXTRA_ARGS="--swa_eval_chunks 0" \
PROJECT_ROOT="$ROOT" PYTHON_BIN="$ROOT/.venv/bin/python" \
bash scripts/_eval_taskpool_2group.sh > logs/b25_c512_ckpt_eval_W0.out 2>&1
echo "W0_DONE_EXIT_$?" >> logs/b25_c512_ckpt_eval_W0.out
