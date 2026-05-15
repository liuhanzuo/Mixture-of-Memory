"""Smoke tests for eval_queue_watcher.py.

No GPU required — subprocess calls are mocked where needed.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from scripts.eval_queue_watcher import (
    MODE_CONFIGS,
    build_eval_command,
    build_output_name,
    discover_ckpts,
    extract_step_number,
    load_registry,
    parse_args,
    save_registry,
)


# --------------------------------------------------------------------------- #
# Test 1: Registry load/save round-trip
# --------------------------------------------------------------------------- #


class TestRegistryRoundTrip:
    def test_empty_registry(self, tmp_path):
        reg_path = tmp_path / "registry.json"
        reg = load_registry(reg_path)
        assert reg == {}

    def test_save_and_load(self, tmp_path):
        reg_path = tmp_path / "subdir" / "registry.json"
        data = {
            "/path/to/ckpt1.pt": {
                "started_at": "2026-05-15T10:00:00+00:00",
                "completed_at": "2026-05-15T10:25:00+00:00",
                "results_dir": "/results/ckpt1",
                "status": "completed",
            },
            "/path/to/ckpt2.pt": {
                "started_at": "2026-05-15T11:00:00+00:00",
                "completed_at": None,
                "results_dir": "/results/ckpt2",
                "status": "running",
            },
        }
        save_registry(data, reg_path)
        loaded = load_registry(reg_path)
        assert loaded == data

    def test_overwrite(self, tmp_path):
        reg_path = tmp_path / "registry.json"
        save_registry({"a": 1}, reg_path)
        save_registry({"b": 2}, reg_path)
        loaded = load_registry(reg_path)
        assert loaded == {"b": 2}


# --------------------------------------------------------------------------- #
# Test 2: Checkpoint discovery
# --------------------------------------------------------------------------- #


class TestCkptDiscovery:
    def test_discovers_valid_ckpt(self, tmp_path):
        """Create a fake ckpt with sibling adapter_config.json, expect discovery."""
        watch_dir = tmp_path / "outputs" / "babilong_sft_phase5b_l1_l3_v2"
        watch_dir.mkdir(parents=True)

        # Create ckpt (old enough)
        ckpt = watch_dir / "mem_space_adapter_step000100.pt"
        ckpt.write_bytes(b"fake")
        # Backdate mtime by 60 seconds
        old_time = time.time() - 60
        os.utime(ckpt, (old_time, old_time))

        # Create sibling config
        config = watch_dir / "adapter_config.json"
        config.write_text(json.dumps({"num_slots": 128}))

        results = discover_ckpts([watch_dir], registry={})
        assert len(results) == 1
        assert results[0].name == "mem_space_adapter_step000100.pt"

    def test_discovers_multiple_sorted_by_mtime(self, tmp_path):
        """Multiple ckpts sorted by mtime ascending."""
        watch_dir = tmp_path / "outputs" / "run1"
        watch_dir.mkdir(parents=True)
        (watch_dir / "adapter_config.json").write_text("{}")

        now = time.time()
        for i, step in enumerate([200, 400, 100]):
            ckpt = watch_dir / f"mem_space_adapter_step000{step:03d}.pt"
            ckpt.write_bytes(b"fake")
            os.utime(ckpt, (now - 300 + i * 100, now - 300 + i * 100))

        results = discover_ckpts([watch_dir], registry={})
        assert len(results) == 3
        # Should be sorted by mtime ascending
        assert "step000200" in results[0].name
        assert "step000400" in results[1].name
        assert "step000100" in results[2].name

    def test_skips_already_processed(self, tmp_path):
        """Ckpts already in registry are skipped."""
        watch_dir = tmp_path / "run"
        watch_dir.mkdir(parents=True)
        (watch_dir / "adapter_config.json").write_text("{}")

        ckpt = watch_dir / "mem_space_adapter_step000100.pt"
        ckpt.write_bytes(b"fake")
        old_time = time.time() - 60
        os.utime(ckpt, (old_time, old_time))

        registry = {str(ckpt.resolve()): {"status": "completed"}}
        results = discover_ckpts([watch_dir], registry=registry)
        assert len(results) == 0


# --------------------------------------------------------------------------- #
# Test 3: Skip ckpt without sibling adapter_config.json
# --------------------------------------------------------------------------- #


class TestSkipNoConfig:
    def test_skip_without_adapter_config(self, tmp_path):
        """Ckpt without adapter_config.json is skipped."""
        watch_dir = tmp_path / "run"
        watch_dir.mkdir(parents=True)

        ckpt = watch_dir / "mem_space_adapter_step000200.pt"
        ckpt.write_bytes(b"fake")
        old_time = time.time() - 60
        os.utime(ckpt, (old_time, old_time))
        # No adapter_config.json created

        results = discover_ckpts([watch_dir], registry={})
        assert len(results) == 0


# --------------------------------------------------------------------------- #
# Test 4: Skip ckpts younger than 30 sec
# --------------------------------------------------------------------------- #


class TestSkipYoungCkpts:
    def test_skip_young_ckpt(self, tmp_path):
        """Ckpts created less than 30 sec ago are skipped."""
        watch_dir = tmp_path / "run"
        watch_dir.mkdir(parents=True)
        (watch_dir / "adapter_config.json").write_text("{}")

        ckpt = watch_dir / "mem_space_adapter_step000300.pt"
        ckpt.write_bytes(b"fake")
        # Leave mtime as current (just created, < 30 sec old)

        results = discover_ckpts([watch_dir], registry={})
        assert len(results) == 0

    def test_old_ckpt_passes(self, tmp_path):
        """Ckpts older than 30 sec pass the age check."""
        watch_dir = tmp_path / "run"
        watch_dir.mkdir(parents=True)
        (watch_dir / "adapter_config.json").write_text("{}")

        ckpt = watch_dir / "mem_space_adapter_step000300.pt"
        ckpt.write_bytes(b"fake")
        old_time = time.time() - 60
        os.utime(ckpt, (old_time, old_time))

        results = discover_ckpts([watch_dir], registry={})
        assert len(results) == 1


# --------------------------------------------------------------------------- #
# Test 5: Build correct CLI args for sniff vs full mode
# --------------------------------------------------------------------------- #


class TestBuildEvalCommand:
    def test_sniff_mode_args(self, tmp_path):
        """Sniff mode should use qa1,qa2,qa5 + 8k,16k,32k + limit=20."""
        ckpt = tmp_path / "mem_space_adapter_step000400.pt"
        ckpt.write_bytes(b"fake")
        (tmp_path / "adapter_config.json").write_text("{}")

        args = parse_args([
            "--mode", "sniff",
            "--model_path", "models/Meta-Llama-3-8B-Instruct",
            "--chunk_size", "4096",
            "--max_new_tokens", "20",
            "--gpu", "7",
        ])
        results_root = tmp_path / "results"
        mode_cfg = MODE_CONFIGS["sniff"]
        cmd = build_eval_command(ckpt, "test_step000400", args, mode_cfg, results_root)

        # Verify tasks
        tasks_idx = cmd.index("--tasks")
        lengths_idx = cmd.index("--lengths")
        tasks_in_cmd = cmd[tasks_idx + 1 : lengths_idx]
        assert tasks_in_cmd == ["qa1", "qa2", "qa5"]

        # Verify lengths
        limit_idx = cmd.index("--limit")
        lengths_in_cmd = cmd[lengths_idx + 1 : limit_idx]
        assert lengths_in_cmd == ["8k", "16k", "32k"]

        # Verify limit
        assert cmd[limit_idx + 1] == "20"

        # Verify device
        assert "--device" in cmd
        device_idx = cmd.index("--device")
        assert cmd[device_idx + 1] == "cuda:0"

    def test_full_mode_args(self, tmp_path):
        """Full mode should use all 5 tasks + 7 lengths + limit=100."""
        ckpt = tmp_path / "mem_space_adapter.pt"
        ckpt.write_bytes(b"fake")
        (tmp_path / "adapter_config.json").write_text("{}")

        args = parse_args([
            "--mode", "full",
            "--model_path", "models/Meta-Llama-3-8B-Instruct",
            "--chunk_size", "4096",
            "--max_new_tokens", "20",
            "--gpu", "3",
        ])
        results_root = tmp_path / "results"
        mode_cfg = MODE_CONFIGS["full"]
        cmd = build_eval_command(ckpt, "test_final", args, mode_cfg, results_root)

        tasks_idx = cmd.index("--tasks")
        lengths_idx = cmd.index("--lengths")
        tasks_in_cmd = cmd[tasks_idx + 1 : lengths_idx]
        assert tasks_in_cmd == ["qa1", "qa2", "qa3", "qa4", "qa5"]

        limit_idx = cmd.index("--limit")
        lengths_in_cmd = cmd[lengths_idx + 1 : limit_idx]
        assert lengths_in_cmd == ["0k", "1k", "2k", "4k", "8k", "16k", "32k"]

        assert cmd[limit_idx + 1] == "100"


# --------------------------------------------------------------------------- #
# Test: output name construction
# --------------------------------------------------------------------------- #


class TestOutputName:
    def test_step_ckpt(self, tmp_path):
        d = tmp_path / "babilong_sft_phase5b_l1_l3_v2"
        d.mkdir()
        ckpt = d / "mem_space_adapter_step000400.pt"
        ckpt.write_bytes(b"x")
        assert build_output_name(ckpt) == "phase5b_l1_l3_v2_step000400"

    def test_final_ckpt(self, tmp_path):
        d = tmp_path / "babilong_sft_phase4_dual_gate"
        d.mkdir()
        ckpt = d / "mem_space_adapter.pt"
        ckpt.write_bytes(b"x")
        assert build_output_name(ckpt) == "phase4_dual_gate_final"

    def test_extract_step_number(self, tmp_path):
        d = tmp_path / "run"
        d.mkdir()
        ckpt = d / "mem_space_adapter_step000600.pt"
        ckpt.write_bytes(b"x")
        assert extract_step_number(ckpt) == 600

        final = d / "mem_space_adapter.pt"
        final.write_bytes(b"x")
        assert extract_step_number(final) is None
