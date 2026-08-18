#!/usr/bin/env bash
set -euo pipefail
ROOT="/apdcephfs_wzc1/share_304376610/pighzliu_code"
while [[ ! -f "/apdcephfs_wzc1/share_304376610/pighzliu_code/outputs/paper_v2/queues/cast9_dense_ast_20260728_v1/queue.done" && ! -f "/apdcephfs_wzc1/share_304376610/pighzliu_code/outputs/paper_v2/queues/cast9_dense_ast_20260728_v1/queue.failed" ]]; do sleep 30; done
if [[ -f "/apdcephfs_wzc1/share_304376610/pighzliu_code/outputs/paper_v2/queues/cast9_dense_ast_20260728_v1/queue.failed" ]]; then
  echo "CAST-9 evaluation failed; restarting ProxSparse after GPU release anyway"
fi
cd "/apdcephfs_wzc1/share_304376610/pighzliu_code"
METHOD=proxsparse INITIAL_MASK_PATH="/apdcephfs_wzc1/share_304376610/pighzliu_code/outputs/paper_v2/common_recovery_masks/proxsparse_official_top2_mask.pt" CAMPAIGN_ID="fixed_mask_full_healing_proxsparse_625m_20260728_v3" exec bash scripts/h20/run_fixed_mask_full_healing_625m.sh
