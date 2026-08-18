from __future__ import annotations

import os
import importlib.util
from pathlib import Path
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "heartbeat",
    ROOT / "ops" / "heartbeat.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class HeartbeatProgressTests(unittest.TestCase):
    @staticmethod
    def make_checkpoint(path: Path, step: int | None = None) -> Path:
        if step is not None:
            path = path / f"global_step_{step}"
        path.mkdir(parents=True)
        for name in MODULE.CHECKPOINT_REQUIRED_FILES:
            file = path / name
            if name == "model.safetensors.index.json":
                file.write_text(
                    '{"weight_map":{"x":"model-00001-of-00001.safetensors"}}',
                    encoding="utf-8",
                )
            else:
                file.write_text(name, encoding="utf-8")
        (path / "model-00001-of-00001.safetensors").write_text(
            "weights",
            encoding="utf-8",
        )
        return path

    def test_extracts_latest_training_progress(self) -> None:
        text = (
            "Total training steps: 4465\n"
            "Epoch 2/5:  40%| progress\n"
            "step:100 - train/loss:0.75 - train/lr(1e-3):0.01\n"
            "Epoch 2/5:  41%| progress\n"
            "step:101 - train/loss:0.625 - train/lr(1e-3):0.01\n"
        )
        progress = MODULE.extract_training_progress(text)
        self.assertEqual(progress["step"], 101)
        self.assertEqual(progress["loss"], 0.625)
        self.assertEqual(progress["total_steps"], 4465)
        self.assertAlmostEqual(progress["fraction"], 102 / 4465)
        self.assertEqual(progress["epoch"], 2)
        self.assertEqual(progress["epochs"], 5)
        self.assertEqual(progress["epoch_percent"], 41)

    def test_reuses_known_total_when_tail_omits_header(self) -> None:
        progress = MODULE.extract_training_progress(
            "step:200 - train/loss:1.25\n",
            known_total_steps=1000,
        )
        self.assertEqual(progress["total_steps"], 1000)
        self.assertAlmostEqual(progress["fraction"], 0.201)

    def test_progress_milestones_are_recorded_once(self) -> None:
        self.assertEqual(
            MODULE.pending_progress_milestones(0.53, []),
            [25, 50],
        )
        self.assertEqual(
            MODULE.pending_progress_milestones(0.80, [25, 50]),
            [75],
        )
        self.assertEqual(
            MODULE.pending_progress_milestones(0.99, [25, 50, 75]),
            [],
        )

    def test_only_stage1_training_tracks_checkpoints(self) -> None:
        self.assertTrue(
            MODULE.tracks_training_checkpoint(
                "SCHEDULE-ONLY-SFT-STAGE1-001"
            )
        )
        self.assertFalse(
            MODULE.tracks_training_checkpoint(
                "SCHEDULE-ONLY-EVALPLUS-HE16-SMOKE-001"
            )
        )

    def test_discovers_only_complete_current_or_explicit_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outputs = Path(directory)
            old = outputs / "old" / "global_step_100"
            current = outputs / "current" / "global_step_200"
            incomplete = outputs / "current" / "global_step_300"
            for path in (old, current, incomplete):
                path.mkdir(parents=True)
            now = time.time()
            for path, modified in ((old, now - 3600), (current, now)):
                for name in MODULE.CHECKPOINT_REQUIRED_FILES:
                    file = path / name
                    if name == "model.safetensors.index.json":
                        file.write_text(
                            '{"weight_map":{"x":"model-00001-of-00001.safetensors"}}',
                            encoding="utf-8",
                        )
                    else:
                        file.write_text(name, encoding="utf-8")
                    os.utime(file, (modified, modified))
                shard = path / "model-00001-of-00001.safetensors"
                shard.write_text("weights", encoding="utf-8")
                os.utime(shard, (modified, modified))
            (incomplete / "training_state.pt").write_text(
                "partial",
                encoding="utf-8",
            )
            launched = MODULE.dt.datetime.fromtimestamp(
                now - 60,
                tz=MODULE.dt.timezone.utc,
            ).isoformat()
            found = MODULE.latest_complete_checkpoint(
                outputs,
                launched_at=launched,
            )
            self.assertEqual(found["step"], 200)
            self.assertEqual(Path(found["path"]), current.resolve())
            self.assertEqual(found["integrity"], "complete")
            self.assertEqual(found["model_shards"], 1)

            resumed = MODULE.latest_complete_checkpoint(
                outputs,
                launched_at=launched,
                explicit_paths=(old,),
            )
            self.assertEqual(resumed["step"], 200)
            current.rename(outputs / "current" / "moved_step_200")
            resumed = MODULE.latest_complete_checkpoint(
                outputs,
                launched_at=launched,
                explicit_paths=(old,),
            )
            self.assertEqual(resumed["step"], 100)

    def test_rejects_checkpoint_with_missing_model_shard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "global_step_10"
            checkpoint.mkdir()
            for name in MODULE.CHECKPOINT_REQUIRED_FILES:
                path = checkpoint / name
                if name == "model.safetensors.index.json":
                    path.write_text(
                        '{"weight_map":{"x":"missing.safetensors"}}',
                        encoding="utf-8",
                    )
                else:
                    path.write_text(name, encoding="utf-8")
            self.assertIsNone(MODULE.checkpoint_integrity(checkpoint))

    def test_training_success_requires_final_complete_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = self.make_checkpoint(root / "checkpoints", step=10)
            pointer = root / "latest.txt"
            pointer.write_text(str(checkpoint), encoding="utf-8")
            valid = MODULE.validate_training_success(
                pointer,
                expected_total_steps=10,
            )
            self.assertTrue(valid["ok"])
            stale = MODULE.validate_training_success(
                pointer,
                expected_total_steps=11,
            )
            self.assertFalse(stale["ok"])


if __name__ == "__main__":
    unittest.main()
