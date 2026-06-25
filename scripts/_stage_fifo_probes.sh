#!/bin/bash
# ============================================================================
# Staged FIFO probe launcher (diskB). Heartbeat invokes with a probe name.
#   usage: bash /tmp/stage_probes.sh <P2|P3|P4|P5|NOLEAK3000_W0|NOLEAK3000_PROBES> "<GPUS>"
# GPUS = space-sep GPU ids to pin (single group). NSHARD auto = #GPUs (max 4).
# All evals: qa1/qa2/qa5 × 4k/8k/16k/32k, n=100, W0 (swa_eval_chunks 0).
# Survey verdict: P2 (reader-attn keep-set) = #1 direction (only one w/ positive evidence).
# ============================================================================
set -uo pipefail
R=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
cd "$R"; export WANDB_MODE=offline
PY=$R/.venv/bin/python
PROBE="${1:?probe name}"; GPUS="${2:?gpu list e.g. \"0 1 2 3\"}"
NG=$(echo $GPUS | wc -w); [ "$NG" -gt 4 ] && NG=4
LENS="4k 8k 16k 32k"

B25=$R/outputs/mem_space_fifo_b25_chunk512
B100=$R/outputs/mem_space_fifo_b100_chunk512
NL=$R/outputs/mem_space_fifo_b25_chunk512_noleak

case "$PROBE" in
  P2)  # reader-attn keep-set on b100 (H_DIL): keep top-25 + recency, keep-all-buffer
    CK=$B100/full_model.pt; CFG=$B100/adapter_config.json; NAME=probe_b100_P2_keepset25
    EXTRA="--swa_eval_chunks 0 --fifo_keep_set_mode flat_readerattn --fifo_keep_topk 25 --fifo_keep_recency 2 --fifo_keep_all_buffer" ;;
  P3)  # stacked: b25 packed-pos + reader-attn keep-set + keep-all (close-the-gap全胜测试)
    CK=$B25/full_model.pt; CFG=$B25/adapter_config.json; NAME=probe_b25_P3_packed_keepset
    EXTRA="--swa_eval_chunks 0 --fifo_pos_mode packed --fifo_keep_set_mode flat_readerattn --fifo_keep_topk 25 --fifo_keep_recency 2 --fifo_keep_all_buffer" ;;
  P4)  # b25 real positions (vs packed)
    CK=$B25/full_model.pt; CFG=$B25/adapter_config.json; NAME=probe_b25_P4_posReal
    EXTRA="--swa_eval_chunks 0 --fifo_pos_mode real" ;;
  P5)  # b100 keep-set top-10 (top-k sensitivity)
    CK=$B100/full_model.pt; CFG=$B100/adapter_config.json; NAME=probe_b100_P5_keepset10
    EXTRA="--swa_eval_chunks 0 --fifo_keep_set_mode flat_readerattn --fifo_keep_topk 10 --fifo_keep_recency 2 --fifo_keep_all_buffer" ;;
  NOLEAK3000_W0)  # clean b25 step3000 plain W0 — THE decisive first honest FIFO measurement
    CK=$NL/full_model.pt; CFG=$NL/adapter_config.json; NAME=noleak_b25_step3000_W0
    EXTRA="--swa_eval_chunks 0"; LENS="0k 1k 2k 4k 8k 16k 32k" ;;
  NOLEAK3000_W6)  # clean b25 step3000 W6 — clean W0/W6 gap measurement
    CK=$NL/full_model.pt; CFG=$NL/adapter_config.json; NAME=noleak_b25_step3000_W6
    EXTRA="--swa_eval_chunks 6"; LENS="4k 8k 16k 32k" ;;
  NOLEAK3000_packed)  # clean ckpt + packed pos — clean H_POS test (no train/eval mismatch)
    CK=$NL/full_model.pt; CFG=$NL/adapter_config.json; NAME=noleak_b25_step3000_packed
    EXTRA="--swa_eval_chunks 0 --fifo_pos_mode packed" ;;
  NOLEAK3000_keepset)  # clean ckpt + reader-attn keep-set (needs b>25 to matter; b25 keeps all anyway)
    CK=$NL/full_model.pt; CFG=$NL/adapter_config.json; NAME=noleak_b25_step3000_keepset
    EXTRA="--swa_eval_chunks 0 --fifo_keep_set_mode flat_readerattn --fifo_keep_topk 25 --fifo_keep_recency 2 --fifo_keep_all_buffer" ;;
  *) echo "unknown probe $PROBE"; exit 2 ;;
esac

if [ ! -f "$CK" ]; then echo "CKPT MISSING: $CK (not ready yet?)"; exit 3; fi

echo "[stage] launching $NAME on GPUs={$GPUS} NSHARD=$NG"
RUN_PREFIX=$NAME CKPT_FILES="$CK" CK_NAMES="$NAME" ADAPTER_CONFIG="$CFG" CHUNK_SIZE=512 \
  TASKS="qa1 qa2 qa5" LENGTHS="$LENS" NUM_GROUPS=1 GROUP0_GPUS="$GPUS" NSHARD=$NG \
  EXTRA_ARGS="$EXTRA" PROJECT_ROOT=$R PYTHON_BIN=$PY \
  setsid nohup bash scripts/_eval_taskpool_2group.sh > $R/logs/${NAME}.out 2>&1 &
echo "launched pid=$! log=logs/${NAME}.out"
