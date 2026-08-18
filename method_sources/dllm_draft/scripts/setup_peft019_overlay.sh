#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${DREAM_ENV_DIR:-$ROOT/.venv_dream}"
OVERLAY="$ROOT/.peft019"

if PYTHONPATH="$OVERLAY" "$ENV_DIR/bin/python" - <<'PY' >/dev/null 2>&1
import peft
raise SystemExit(0 if peft.__version__ == "0.19.1" else 1)
PY
then
  exit 0
fi

mkdir -p "$OVERLAY"
WHEEL="$ROOT/wheels/peft-0.19.1-py3-none-any.whl"
if [[ -s "$WHEEL" ]]; then
  "$ENV_DIR/bin/python" -m pip install \
    --no-deps --upgrade --target "$OVERLAY" "$WHEEL"
else
  "$ENV_DIR/bin/python" -m pip install \
    --no-deps --upgrade --target "$OVERLAY" peft==0.19.1
fi
