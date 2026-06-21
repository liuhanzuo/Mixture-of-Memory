#!/usr/bin/env bash
# Arm B baseline already exists: offline full-context cache -> W0 pure-memory student.
# Thin wrapper for the 3-arm self-study comparison matrix.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
exec bash scripts/launch_mem_space_selfstudy_rawkv_chunk512.sh "$@"
