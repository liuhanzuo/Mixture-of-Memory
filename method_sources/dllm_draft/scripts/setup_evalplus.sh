#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${DREAM_ENV_DIR:-$ROOT/.venv_dream}"
INDEX_URL="${PIP_INDEX_URL:-https://mirrors.cloud.tencent.com/pypi/simple}"

"$ENV_DIR/bin/python" -m pip install \
  --index-url "$INDEX_URL" \
  wget appdirs tempdir multipledispatch termcolor fire rich \
  "tree_sitter>=0.22.0" tree-sitter-python
"$ENV_DIR/bin/python" -m pip install --no-deps -e "$ROOT/vendor/evalplus"

export HUMANEVAL_OVERRIDE_PATH="$ROOT/data/evalplus/HumanEvalPlus-v0.1.10.jsonl"
export MBPP_OVERRIDE_PATH="$ROOT/data/evalplus/MbppPlus-v0.2.0.jsonl"

"$ENV_DIR/bin/python" - <<'PY'
import json
import os
from pathlib import Path

from evalplus.data import get_human_eval_plus, get_mbpp_plus
from evalplus.data.humaneval import get_human_eval_plus_hash
from evalplus.data.mbpp import get_mbpp_plus_hash

human = get_human_eval_plus()
mbpp = get_mbpp_plus()
report = {
    "humaneval_rows": len(human),
    "mbpp_rows": len(mbpp),
    "humaneval_hash": get_human_eval_plus_hash(),
    "mbpp_hash": get_mbpp_plus_hash(),
    "humaneval_override": os.environ["HUMANEVAL_OVERRIDE_PATH"],
    "mbpp_override": os.environ["MBPP_OVERRIDE_PATH"],
}
root = Path.cwd()
(root / "ops" / "artifacts" / "evalplus_probe.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
PY

date --iso-8601=seconds >"$ROOT/ops/control/evalplus_ready"

