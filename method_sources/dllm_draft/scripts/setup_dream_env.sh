#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${DREAM_ENV_DIR:-$ROOT/.venv_dream}"
CACHE_DIR="${PIP_CACHE_DIR:-$ROOT/cache/pip}"
INDEX_URL="${PIP_INDEX_URL:-https://mirrors.cloud.tencent.com/pypi/simple}"

mkdir -p "$CACHE_DIR" "$ROOT/ops/artifacts"

if [ ! -x "$ENV_DIR/bin/python" ]; then
  /usr/bin/python3.11 -m venv "$ENV_DIR"
fi

export PIP_CACHE_DIR="$CACHE_DIR"
export PIP_DEFAULT_TIMEOUT=180

"$ENV_DIR/bin/python" -m pip install \
  --index-url "$INDEX_URL" \
  --upgrade pip setuptools wheel

"$ENV_DIR/bin/python" -m pip install \
  --index-url "$INDEX_URL" \
  "torch==2.5.1" \
  "transformers==4.46.2" \
  "accelerate>=1.1,<2" \
  "safetensors>=0.4.5" \
  "omegaconf>=2.3" \
  "hydra-core>=1.3" \
  "sentencepiece>=0.2" \
  "protobuf>=4.25" \
  "psutil>=6"

"$ENV_DIR/bin/python" - <<'PY'
import json
import platform
from pathlib import Path

import torch
import transformers

root = Path.cwd()
report = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "transformers": transformers.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "devices": [],
}
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        report["devices"].append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory": properties.total_memory,
                "capability": list(torch.cuda.get_device_capability(index)),
            }
        )

path = root / "ops" / "artifacts" / "dream_env_probe.json"
path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
PY

"$ENV_DIR/bin/python" -m pip freeze \
  >"$ROOT/ops/artifacts/dream_env_freeze.txt"

date --iso-8601=seconds >"$ROOT/ops/control/dream_env_ready"

