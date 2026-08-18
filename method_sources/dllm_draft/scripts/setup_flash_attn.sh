#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${DREAM_ENV_DIR:-$ROOT/.venv_dream}"
WHEEL="$ROOT/wheels/flash_attn-2.7.3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
INDEX_URL="${PIP_INDEX_URL:-https://mirrors.cloud.tencent.com/pypi/simple}"

"$ENV_DIR/bin/python" -m pip install \
  --index-url "$INDEX_URL" \
  "einops>=0.8"
"$ENV_DIR/bin/python" -m pip install --no-deps "$WHEEL"

"$ENV_DIR/bin/python" - <<'PY'
import json
from pathlib import Path

import flash_attn
import torch
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input

report = {
    "flash_attn": flash_attn.__version__,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
    "bert_padding_imports": [
        index_first_axis.__name__,
        pad_input.__name__,
        unpad_input.__name__,
    ],
}
root = Path.cwd()
(root / "ops" / "artifacts" / "flash_attn_probe.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
PY

date --iso-8601=seconds >"$ROOT/ops/control/flash_attn_ready"

