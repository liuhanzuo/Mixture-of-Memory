#!/usr/bin/env bash
# A04 -- driver for the CONTROL-ARMS NI pass (freezefront / fromscratch @ 200k).
#
# ZERO GPU BY DESIGN. Every input is a per-example shard set already on zwfy6,
# written 2026-08-02. Nothing is scored, no model is loaded, no CUDA context is
# created. The refuse-guard below therefore exists NOT because this job needs
# GPUs, but because the dispatch requires every driver in this tree to refuse to
# run on a node someone else is using -- so that a future copy of this file that
# DOES add a scoring step inherits the guard rather than needing it retrofitted.
#
# NODE BUDGET (2026-08-13 dispatch, hard):
#   ALLOWED : .73 (28.85.35.73), .82 (28.82.250.82)   -- 8xH20, zwfy6
#   FORBIDDEN: LOCAL and .21 (SparseForge #246), .104 (paperC Qwen3 heal)
#
# NUMPY: .73 = 2.5.1, .82 = 2.4.6, LOCAL = 2.3.5. Generator.multinomial differs
# in 19/10000 rows between 2.4.6 and 2.5.1 (max margin drift 0.005294 pp), so
# ALL cells must come from ONE node. This driver pins the expected version and
# the analysis refuses to publish from a different one.
#
# Usage (on the target node):
#   A04_NODE=.82 EXPECT_NUMPY=2.4.6 bash code/a04_control_arms_ni_driver.sh
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
A04_DIR="${A04_DIR:-$PROJECT_ROOT/proposal/active/A04-recovery-certification}"
OUT_JSON="${OUT_JSON:-$A04_DIR/evidence/a04_control_arms_ni.json}"
EXPECT_NUMPY="${EXPECT_NUMPY:-}"
MEM_LIMIT_MIB="${MEM_LIMIT_MIB:-8000}"

echo "[$(date '+%F %T')] DRIVER START a04_control_arms_ni  node=${A04_NODE:-unset}"
echo "  PROJECT_ROOT=$PROJECT_ROOT"
echo "  PYTHON_BIN=$PYTHON_BIN"
echo "  OUT_JSON=$OUT_JSON"
echo "  EXPECT_NUMPY=${EXPECT_NUMPY:-<unpinned>}"

# ---- refuse-guard: never start on a node someone else is using --------------
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[guard] nvidia-smi memory.used per GPU:"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  MAXMEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
           | sort -n | tail -1)
  if [ -z "$MAXMEM" ]; then
    echo "[guard] FATAL: could not read nvidia-smi memory.used"; exit 3
  fi
  if [ "$MAXMEM" -gt "$MEM_LIMIT_MIB" ]; then
    echo "[guard] REFUSING TO START: a GPU holds ${MAXMEM} MiB > ${MEM_LIMIT_MIB} MiB."
    echo "[guard] Another job owns this node. Not competing for it."
    exit 3
  fi
  echo "[guard] OK: max GPU memory.used=${MAXMEM} MiB <= ${MEM_LIMIT_MIB} MiB"
else
  echo "[guard] nvidia-smi absent -- CPU-only host, proceeding"
fi

# ---- refuse to run on a forbidden node --------------------------------------
HOSTIP=$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^28\.' | head -1)
echo "[guard] host ip (28.x) = ${HOSTIP:-none}"
case "$HOSTIP" in
  28.83.24.104)
    echo "[guard] REFUSING: .104 is running paperC Qwen3 heal. Out of budget."
    exit 3 ;;
  28.89.19.21)
    echo "[guard] REFUSING: .21 is running SparseForge #246. Out of budget."
    exit 3 ;;
esac

cd "$PROJECT_ROOT" || { echo "FATAL: cannot cd $PROJECT_ROOT"; exit 2; }

echo "[$(date '+%F %T')] numpy: $($PYTHON_BIN -c 'import numpy;print(numpy.__version__)')"

# ---- pre-flight: every input directory must exist --------------------------
MISSING=0
for d in \
  olmo2_mmlu_content_results/7B_base \
  olmo2_mmlu_content_results/7B_keep14_step200000 \
  olmo2_mmlu_content_results/7B_freezefront_step200000 \
  olmo2_mmlu_content_results/7B_scratch16L_step200000 \
  olmo2_closedbook_results/base_full \
  olmo2_closedbook_results/base_full_nqopen \
  olmo2_closedbook_results/keep14_step200k \
  olmo2_closedbook_results/keep14_step200k_nqopen \
  olmo2_closedbook_results/freezefront_step200k \
  olmo2_closedbook_results/freezefront_step200k_nqopen \
  olmo2_closedbook_results/fromscratch_step200k \
  olmo2_closedbook_results/fromscratch_step200k_nqopen ; do
  if [ -d "$PROJECT_ROOT/$d" ]; then
    n=$(ls "$PROJECT_ROOT/$d" | grep -c 'shard[0-7]of8.jsonl' || true)
    echo "  OK   $d  (${n} per-example shard files)"
  else
    echo "  MISS $d"; MISSING=$((MISSING+1))
  fi
done
if [ "$MISSING" -ne 0 ]; then
  echo "FATAL: $MISSING input directories missing -- refusing to publish a"
  echo "       partial comparison. (A silently merged 5-of-8 shard set has"
  echo "       corrupted results in this repo before.)"
  exit 2
fi

# ---- launcher logs the protocol asserter needs ------------------------------
for f in logs/cb_driver_104.out logs/cb_driver_73.out \
         logs/nqopen_driver_104.log logs/nqopen_scratch.log \
         scripts/p06_run_104_transferred.sh scripts/p06_run_transferred.sh \
         scripts/_run_olmo2_mmlu_content.sh \
         scripts/eval_olmo2_closedbook_qa.py \
         scripts/eval_olmo2_mmlu_content.py ; do
  [ -f "$PROJECT_ROOT/$f" ] && echo "  OK   $f" || { echo "  MISS $f"; MISSING=$((MISSING+1)); }
done
if [ "$MISSING" -ne 0 ]; then
  echo "FATAL: protocol-evidence file(s) missing. The asserter fails closed;"
  echo "       running anyway would only produce the same FATAL later."
  exit 2
fi

mkdir -p "$(dirname "$OUT_JSON")"

EXTRA=()
[ -n "$EXPECT_NUMPY" ] && EXTRA+=(--expect_numpy "$EXPECT_NUMPY")

echo "[$(date '+%F %T')] running analysis (CPU only)"
A04_NODE="${A04_NODE:-unset}" "$PYTHON_BIN" \
  "$A04_DIR/code/a04_control_arms_ni.py" \
  --raw_root "$PROJECT_ROOT" \
  --out_json "$OUT_JSON" \
  "${EXTRA[@]}"
RC=$?
echo "[$(date '+%F %T')] analysis rc=$RC"
if [ "$RC" -ne 0 ]; then
  echo "FATAL: analysis failed (rc=$RC). No output file should exist."
  ls -la "$OUT_JSON" 2>/dev/null && echo "WARNING: output file exists despite failure"
  exit "$RC"
fi

echo "[$(date '+%F %T')] sha256: $(sha256sum "$OUT_JSON")"
echo "[$(date '+%F %T')] DRIVER DONE"
