#!/usr/bin/env bash
set -euo pipefail
ROOT="/apdcephfs_wzc1/share_304376610/pighzliu_code"
while pgrep -f "fixed_mask_full_healing_proxsparse_625m_20260728_v3" >/dev/null; do sleep 60; done
cd "/apdcephfs_wzc1/share_304376610/pighzliu_code"
CAMPAIGN_ID="missing_cast9_after_proxsparse_20260728_v1" exec bash scripts/h20/run_missing_cast9_queue.sh
