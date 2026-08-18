from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "merge_scaffold_lora",
    ROOT / "scripts" / "merge_scaffold_lora.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class _Layer:
    def __init__(self, scaling=None):
        self.scaling = scaling


class _Model:
    def __init__(self):
        self.layers = [
            _Layer({"default": 2.0}),
            _Layer({"a": 4.0, "b": 6.0}),
            _Layer(None),
        ]

    def modules(self):
        return iter(self.layers)


class MergeScaffoldLoraTests(unittest.TestCase):
    def test_scales_all_adapter_entries(self) -> None:
        model = _Model()
        count = MODULE.scale_lora_adapters(model, 0.25)
        self.assertEqual(count, 3)
        self.assertEqual(model.layers[0].scaling["default"], 0.5)
        self.assertEqual(model.layers[1].scaling["a"], 1.0)
        self.assertEqual(model.layers[1].scaling["b"], 1.5)

    def test_existing_scaffold_base_is_detectable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertFalse(MODULE.has_scaffold_token_rows(root))
            (root / "scaffold_tokens.json").write_text("{}\n")
            self.assertTrue(MODULE.has_scaffold_token_rows(root))


if __name__ == "__main__":
    unittest.main()
