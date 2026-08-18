#!/usr/bin/env python3
"""Write an auditable manifest for one trained checkpoint."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, *, include_hash: bool = True) -> dict[str, object]:
    stat = path.stat()
    record: dict[str, object] = {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if include_hash:
        record["sha256"] = sha256(path)
    return record


def command_output(command: list[str], cwd: Path) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return result.stdout.strip()


def launch_timestamp(history_path: Path, run_id: str) -> str:
    launched = []
    for line in history_path.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t", 3)
        if (
            len(fields) >= 3
            and fields[1] == "LAUNCHED"
            and fields[2] == run_id
        ):
            launched.append(fields[0])
    if not launched:
        raise ValueError(f"no LAUNCHED history entry for {run_id}")
    return launched[-1]


def commit_at_timestamp(root: Path, timestamp: str) -> str:
    target = datetime.fromisoformat(timestamp).timestamp()
    output = command_output(
        [
            "git",
            "reflog",
            "--date=unix",
            "--format=%H%x09%gd",
        ],
        root,
    )
    for line in output.splitlines():
        commit, separator, selector = line.partition("\t")
        if not separator:
            continue
        match = re.search(r"@\{(\d+)\}", selector)
        if match and int(match.group(1)) <= target:
            return commit
    raise ValueError(f"git reflog has no commit at or before {timestamp}")


def git_file_record(root: Path, path: Path, commit: str) -> dict[str, object]:
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(
            f"{relative} is unavailable at launch commit {commit}: "
            + result.stderr.decode("utf-8", errors="replace")
        )
    return {
        "path": relative,
        "commit": commit,
        "size": len(result.stdout),
        "sha256": hashlib.sha256(result.stdout).hexdigest(),
    }


def extract_resolved_config(text: str) -> dict[str, object]:
    candidates = []
    for line in text.replace("\r", "\n").splitlines():
        start = line.find("{'data':")
        if start < 0:
            continue
        try:
            value = ast.literal_eval(line[start:])
        except (SyntaxError, ValueError):
            continue
        if isinstance(value, dict) and {"data", "trainer"} <= set(value):
            candidates.append(value)
    if not candidates:
        raise ValueError("training log contains no resolved config dictionary")
    return candidates[-1]


def resolved_config_record(
    checkpoint: Path,
    run_log: Path,
    *,
    launch_commit: str,
) -> tuple[dict[str, object], dict[str, object]]:
    artifact = checkpoint.parent / "resolved_training_config.json"
    if artifact.exists():
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        config = payload.get("config")
        if not isinstance(config, dict):
            raise ValueError(f"{artifact} has no config dictionary")
        artifact_commit = payload.get("git", {}).get("commit")
        if artifact_commit and artifact_commit != launch_commit:
            raise ValueError(
                "resolved config launch commit mismatch: "
                f"{artifact_commit} != {launch_commit}"
            )
        return config, {
            "source": "launch_artifact",
            "file": file_record(artifact),
            "metadata": {
                key: value
                for key, value in payload.items()
                if key != "config"
            },
        }
    return extract_resolved_config(
        run_log.read_text(encoding="utf-8", errors="replace")
    ), {
        "source": "registered_log_recovery",
        "file": file_record(run_log),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--launcher", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-log", required=True)
    parser.add_argument("--training-metrics", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--eval-data", required=True)
    parser.add_argument(
        "--provenance-file",
        action="append",
        default=[],
        help="Additional tuning/config artifact to hash into the manifest.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    checkpoint = Path(args.checkpoint).resolve()
    launcher = Path(args.launcher).resolve()
    run_log = Path(args.run_log).resolve()
    metrics_path = Path(args.training_metrics).resolve()
    launched_at = launch_timestamp(
        root / "ops" / "history.tsv",
        args.run_id,
    )
    launch_commit = commit_at_timestamp(root, launched_at)
    resolved_config, resolved_config_provenance = resolved_config_record(
        checkpoint,
        run_log,
        launch_commit=launch_commit,
    )
    metric_rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    checkpoint_files = sorted(
        (
            {
                "name": path.name,
                "size": path.stat().st_size,
            }
            for path in checkpoint.iterdir()
            if path.is_file()
        ),
        key=lambda row: str(row["name"]),
    )
    important_checkpoint_files = {}
    for name in (
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "training_state.pt",
        "scaffold_tokens.json",
    ):
        path = checkpoint / name
        if path.exists():
            important_checkpoint_files[name] = file_record(path)

    versions: dict[str, object] = {"python": platform.python_version()}
    try:
        import torch

        versions.update(
            {
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
            }
        )
    except ImportError:
        pass
    try:
        import transformers

        versions["transformers"] = transformers.__version__
    except ImportError:
        pass

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "name": args.name,
        "mode": args.mode,
        "git": {
            "current_commit": command_output(
                ["git", "rev-parse", "HEAD"],
                root,
            ),
            "current_status": command_output(
                ["git", "status", "--short", "--branch"],
                root,
            ),
            "code_status_excluding_live_ops": command_output(
                [
                    "git",
                    "status",
                    "--short",
                    "--",
                    ".",
                    ":(exclude)ops/queue.tsv",
                    ":(exclude)ops/history.tsv",
                ],
                root,
            ),
        },
        "launch": {
            "run_id": args.run_id,
            "launched_at": launched_at,
            "git_commit": launch_commit,
            "launcher_at_launch": git_file_record(
                root,
                launcher,
                launch_commit,
            ),
            "resolved_config": resolved_config,
            "resolved_config_provenance": resolved_config_provenance,
            "run_log": file_record(run_log),
        },
        "launcher_current": file_record(launcher),
        "data": {
            "train": file_record(Path(args.train_data).resolve()),
            "eval": file_record(Path(args.eval_data).resolve()),
        },
        "checkpoint": {
            "path": str(checkpoint),
            "files": checkpoint_files,
            "important_files": important_checkpoint_files,
        },
        "training_metrics": {
            "file": file_record(metrics_path),
            "records": len(metric_rows),
            "first": metric_rows[0] if metric_rows else None,
            "last": metric_rows[-1] if metric_rows else None,
        },
        "provenance_files": [
            file_record(Path(path).resolve())
            for path in args.provenance_file
        ],
        "versions": versions,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
