#!/usr/bin/env bash
# One-shot setup after `git clone`.
# Installs Python deps, third-party repos, and prints next steps for data.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "[setup] Installing Python package (editable)..."
pip install -e .

echo "[setup] Cloning third-party repositories..."
bash scripts/setup_third_party.sh

cat <<'EOF'

[setup] Done with code + third-party.

Next steps (NOT executed automatically — these need GPU / large disk):
  1. Prepare data:
       bash scripts/setup_data.sh
     (or follow data/README.md to regenerate the tokenized corpora.)

  2. Run a smoke test:
       pytest tests/ -x

  3. See README.md and INDEX.md for entry points to training / eval.
EOF
