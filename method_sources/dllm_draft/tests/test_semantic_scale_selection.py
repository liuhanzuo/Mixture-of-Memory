from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "select_semantic_lora_scale",
    ROOT / "scripts" / "select_semantic_lora_scale.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SemanticScaleSelectionTests(unittest.TestCase):
    def test_selects_complete_checkpoint_above_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint_scale_025"
            checkpoint.mkdir()
            (checkpoint / "model.safetensors.index.json").write_text("{}")
            (root / "report.json").write_text(
                json.dumps(
                    {
                        "selected": {
                            "label": "scale_025",
                            "scale": 0.25,
                            "plus_pass1": 0.59375,
                            "parse_rate": 1.0,
                            "errors": 0,
                        }
                    }
                )
            )
            pointer = root / "pointer.txt"
            output = root / "selection.json"
            previous = sys.argv
            try:
                sys.argv = [
                    "select_semantic_lora_scale.py",
                    "--calibration-root",
                    str(root),
                    "--pointer",
                    str(pointer),
                    "--output",
                    str(output),
                ]
                MODULE.main()
            finally:
                sys.argv = previous
            self.assertEqual(pointer.read_text().strip(), str(checkpoint.resolve()))
            self.assertEqual(
                json.loads(output.read_text())["selected"]["scale"],
                0.25,
            )


if __name__ == "__main__":
    unittest.main()
