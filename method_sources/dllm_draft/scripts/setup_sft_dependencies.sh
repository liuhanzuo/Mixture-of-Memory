#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${DREAM_ENV_DIR:-$ROOT/.venv_dream}"
INDEX_URL="${PIP_INDEX_URL:-https://mirrors.cloud.tencent.com/pypi/simple}"

"$ENV_DIR/bin/python" -m pip install \
  --index-url "$INDEX_URL" \
  "numpy==1.26.4" \
  "pandas==2.2.3" \
  "pyarrow==18.1.0" \
  "peft==0.13.2" \
  "tensordict==0.6.2" \
  "torchdata==0.10.1" \
  "datasets==3.2.0" \
  "wandb==0.18.7" \
  "ray==2.10.0" \
  "codetiming==1.4.0" \
  "dill==0.3.8" \
  "pybind11==2.13.6" \
  "pylatexenc==2.10" \
  "ninja>=1.11"

"$ENV_DIR/bin/python" -m pip install --no-deps -e "$ROOT/vendor/verl"

PYTHONPATH="$ROOT/vendor/DreamOn:$ROOT/vendor/Dream-Coder/instruct:$ROOT" \
  "$ENV_DIR/bin/python" - <<'PY'
import importlib
import json
from pathlib import Path

import datasets
import flash_attn
import numpy
import pandas
import peft
import pyarrow
import ray
import tensordict
import torch
import transformers
import verl

modules = [
    "src.trainer.sft_expand_dataset",
    "src.trainer.fsdp_sft_expand_trainer",
]
for module in modules:
    importlib.import_module(module)

report = {
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "flash_attn": flash_attn.__version__,
    "verl": getattr(verl, "__version__", "installed"),
    "numpy": numpy.__version__,
    "pandas": pandas.__version__,
    "pyarrow": pyarrow.__version__,
    "datasets": datasets.__version__,
    "peft": peft.__version__,
    "tensordict": tensordict.__version__,
    "ray": ray.__version__,
    "dreamon_imports": modules,
}
root = Path.cwd()
(root / "ops" / "artifacts" / "sft_dependency_probe.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
PY

"$ENV_DIR/bin/python" -m pip freeze \
  >"$ROOT/ops/artifacts/sft_env_freeze.txt"

date --iso-8601=seconds >"$ROOT/ops/control/sft_dependencies_ready"

