from __future__ import annotations

import base64
import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_queue",
    ROOT / "ops" / "audit_queue.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class QueueAuditTests(unittest.TestCase):
    def test_detects_duplicate_id_and_stale_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = root / "scripts"
            scripts.mkdir()
            script = scripts / "run.sh"
            script.write_text("#!/bin/sh\n", encoding="utf-8")
            script.chmod(0o755)
            success = root / "done"
            success.write_text("old", encoding="utf-8")
            encoded = base64.b64encode(b"./scripts/run.sh").decode()
            rows = [
                [
                    "READY",
                    "RUN",
                    "cpu",
                    "1",
                    str(root),
                    str(success),
                    encoded,
                ],
                [
                    "READY",
                    "RUN",
                    "cpu",
                    "1",
                    str(root),
                    str(root / "other"),
                    encoded,
                ],
            ]
            report = MODULE.audit_rows(rows, root)
            self.assertFalse(report["ok"])
            text = json_text = str(report["issues"])
            self.assertIn("duplicate ID", text)
            self.assertIn("pre-existing success", json_text)


if __name__ == "__main__":
    unittest.main()
