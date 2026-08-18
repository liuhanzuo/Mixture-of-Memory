from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "write_experiment_manifest",
    ROOT / "scripts" / "write_experiment_manifest.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ExperimentManifestTests(unittest.TestCase):
    def test_file_record_has_stable_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.bin"
            path.write_bytes(b"scaffold-coder")
            record = MODULE.file_record(path)
            self.assertEqual(record["size"], len(b"scaffold-coder"))
            self.assertEqual(
                record["sha256"],
                hashlib.sha256(b"scaffold-coder").hexdigest(),
            )

    def test_extracts_resolved_config_and_last_launch(self) -> None:
        config = MODULE.extract_resolved_config(
            "noise\n"
            "{'data': {'train_batch_size': 16}, "
            "'trainer': {'save_checkpoint_steps': 2000}, "
            "'scaffold': {'mode': 'schedule_only'}}\n"
        )
        self.assertEqual(config["data"]["train_batch_size"], 16)
        with tempfile.TemporaryDirectory() as directory:
            history = Path(directory) / "history.tsv"
            history.write_text(
                "2026-01-01T00:00:00+00:00\tLAUNCHED\tRUN\tfirst\n"
                "2026-01-01T00:01:00+00:00\tRETRYING\tRUN\tretry\n"
                "2026-01-01T00:02:00+00:00\tLAUNCHED\tRUN\tsecond\n",
                encoding="utf-8",
            )
            self.assertEqual(
                MODULE.launch_timestamp(history, "RUN"),
                "2026-01-01T00:02:00+00:00",
            )

    def test_prefers_launch_config_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "run" / "global_step_10"
            checkpoint.mkdir(parents=True)
            log = root / "run.log"
            log.write_text(
                "{'data': {'train_batch_size': 1}, "
                "'trainer': {'seed': 1}}\n",
                encoding="utf-8",
            )
            artifact = checkpoint.parent / "resolved_training_config.json"
            artifact.write_text(
                '{"config":{"data":{"train_batch_size":16},'
                '"trainer":{"seed":1}},'
                '"git":{"commit":"abc"}}',
                encoding="utf-8",
            )
            config, provenance = MODULE.resolved_config_record(
                checkpoint,
                log,
                launch_commit="abc",
            )
            self.assertEqual(config["data"]["train_batch_size"], 16)
            self.assertEqual(provenance["source"], "launch_artifact")


if __name__ == "__main__":
    unittest.main()
