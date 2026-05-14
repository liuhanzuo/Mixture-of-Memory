#!/usr/bin/env bash
# Sync selected workspace/model/data paths from the main wzc1 tree to the H20 zwfy6 tree.
# Examples:
#   bash scripts/sync_workspace_to_h20.sh --repo-only
#   bash scripts/sync_workspace_to_h20.sh --models baselines/beacon-qwen-2-7b-hf models/Llama-3.2-1B-Instruct
#   bash scripts/sync_workspace_to_h20.sh --data babilong data/armt_pg19_real_tokenized_full

set -euo pipefail

SRC_ROOT="${SRC_ROOT:-/apdcephfs_wzc1/share_303098609/pighzliu_code}"
DST_ROOT="${DST_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code}"
SRC_MOM="${SRC_MOM:-${SRC_ROOT}/Mixture-of-Memory}"
DST_MOM="${DST_MOM:-${DST_ROOT}/Mixture-of-Memory}"
RSYNC_OPTS=("-av" "--delete-delay")

mkdir -p "$DST_ROOT" "$DST_MOM"

sync_path() {
  local src_rel="$1"
  local src="${SRC_ROOT}/${src_rel}"
  local dst="${DST_ROOT}/${src_rel}"
  if [[ ! -e "$src" ]]; then
    echo "Missing source: $src" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$dst")"
  rsync "${RSYNC_OPTS[@]}" "$src" "$dst"
}

sync_mom_path() {
  local src_rel="$1"
  local src="${SRC_MOM}/${src_rel}"
  local dst="${DST_MOM}/${src_rel}"
  if [[ ! -e "$src" ]]; then
    echo "Missing source: $src" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$dst")"
  rsync "${RSYNC_OPTS[@]}" "$src" "$dst"
}

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 [--repo-only] [--models <relpath> ...] [--data <relpath> ...] [--extras <relpath> ...]" >&2
  exit 1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-only)
      shift
      sync_mom_path scripts/
      sync_mom_path src/
      sync_mom_path configs/
      sync_mom_path third_party/
      sync_mom_path pyproject.toml
      sync_mom_path requirements.txt
      ;;
    --models)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        sync_path "$1"
        shift
      done
      ;;
    --data)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        sync_path "$1"
        shift
      done
      ;;
    --extras)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        sync_path "$1"
        shift
      done
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

echo "Sync complete: ${SRC_ROOT} -> ${DST_ROOT}"
