#!/usr/bin/env bash
# A04 -- sigma_run from the POST-ce5c298 keep7 data-order triplet + K2.
# CPU ONLY on .73 (zwfy6). No GPU, no model load, no torch.
# Prereg: A04_SIGMA_RUN_POSTFIX_K2_PREREG.md (commit 94839e8, PRE-DATA).
#
# The canonical a03_sigma_run_n3.json lives on WZC1 ONLY (there is no
# proposal/archive/ on zwfy6), so it is staged to /tmp by the caller and its md5
# is re-asserted inside the python.
set -euo pipefail

ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
A04=$ROOT/proposal/active/A04-recovery-certification
PY=/opt/conda/envs/torch-base/bin/python
A03_SIGMA=${A03_SIGMA:-/tmp/a04_sigma_postfix/a03_sigma_run_n3.json}
OUT=${OUT:-/tmp/a04_sigma_postfix/a04_sigma_run_postfix.json}
LOG=${LOG:-/tmp/a04_sigma_postfix/run.log}

mkdir -p "$(dirname "$OUT")"

# refuse-guard is also enforced inside the python (assert_gpu_clear), but fail
# fast here too so we never even import numpy on a busy node.
BUSY=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
        | awk '$1>8000{c++} END{print c+0}')
if [ "$BUSY" -ne 0 ]; then
  echo "REFUSE: $BUSY GPU(s) hold >8000 MiB on $(hostname). Not running." >&2
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >&2
  exit 3
fi
echo "[guard] all 8 GPUs under 8000 MiB"

cd "$ROOT"
PYTHONPATH="$A04/code:$ROOT/proposal/shared/code" \
  "$PY" "$A04/code/a04_sigma_run_postfix_k2.py" \
    --raw_root "$ROOT" \
    --a03_sigma_json "$A03_SIGMA" \
    --evidence_dir "$A04/evidence" \
    --out_json "$OUT" 2>&1 | tee "$LOG"

echo "=== done: $OUT ==="
