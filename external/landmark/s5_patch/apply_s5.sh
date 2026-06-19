#!/usr/bin/env bash
# Apply the S5 readout-axis gate onto a live llama/ package by patch, so it can
# be layered on top of whatever base is current (anchor OR S4b) when S5 launches.
# Usage: apply_s5.sh <path-to-llama-pkg-dir>   (e.g. .../external/landmark-attention/llama)
# Idempotent-ish: bails if single_layer_mem already present.
set -euo pipefail
PKG="${1:?usage: apply_s5.sh <llama pkg dir>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MEM="$PKG/llama_mem.py"
CFG="$PKG/llama_landmark_config.py"

if grep -q "single_layer_mem" "$MEM" && grep -q "single_layer_mem" "$CFG"; then
    echo "[apply_s5] single_layer_mem already present in $PKG — nothing to do."
    exit 0
fi

cp "$MEM" "$MEM.pre_s5.bak"
cp "$CFG" "$CFG.pre_s5.bak"

# Try a context patch first (composes with S4b). Fall back: report for manual merge.
if patch -p0 --dry-run "$MEM" < "$HERE/llama_mem.S5.diff" >/dev/null 2>&1; then
    patch -p0 "$MEM" < "$HERE/llama_mem.S5.diff"
    echo "[apply_s5] llama_mem.py patched cleanly."
else
    echo "[apply_s5] WARN: llama_mem.S5.diff did not apply cleanly to $MEM."
    echo "           Base has likely diverged (e.g. S4b). Do a manual 3-way merge:"
    echo "           the S5 hunks are (1) __init__ layer_idx threading, (2) the"
    echo "           grouped/plain gate at the normalization site. See README.md."
    exit 2
fi

if patch -p0 --dry-run "$CFG" < "$HERE/llama_landmark_config.S5.diff" >/dev/null 2>&1; then
    patch -p0 "$CFG" < "$HERE/llama_landmark_config.S5.diff"
    echo "[apply_s5] llama_landmark_config.py patched cleanly."
else
    echo "[apply_s5] WARN: config diff did not apply cleanly. Manual merge needed."
    exit 3
fi

echo "[apply_s5] done. Backups: *.pre_s5.bak"
