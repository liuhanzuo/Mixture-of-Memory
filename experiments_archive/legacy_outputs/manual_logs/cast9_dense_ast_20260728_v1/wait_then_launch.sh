#!/usr/bin/env bash
set -euo pipefail
ROOT="/apdcephfs_wzc1/share_304376610/pighzliu_code"
while pgrep -f "fixed_mask_full_healing_proxsparse_625m_20260728_v2" >/dev/null; do sleep 60; done
if [[ -f "$ROOT/outputs/paper_v2/queues/fixed_mask_full_healing_proxsparse_625m_20260728_v2/full625m.done" || ! -f "$ROOT/outputs/paper_v2/queues/fixed_mask_full_healing_proxsparse_625m_20260728_v2/queue.lock" ]]; then
  cd "$ROOT"
  CAMPAIGN_ID="cast9_dense_ast_20260728_v1" exec bash scripts/h20/run_cast9_dense_ast_queue.sh
else
  echo "upstream ProxSparse run stopped without done marker; launching CAST-9 anyway after GPU release"
  cd "$ROOT"
  CAMPAIGN_ID="cast9_dense_ast_20260728_v1" exec bash scripts/h20/run_cast9_dense_ast_queue.sh
fi
