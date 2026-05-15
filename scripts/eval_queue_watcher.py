#!/usr/bin/env python3
"""Eval queue watcher — auto-runs BABILong eval on new checkpoints as they appear.

Watches one or more training output directories for new mem_space adapter
checkpoints and runs BABILong evaluation on them using a dedicated GPU.

Usage:
    python scripts/eval_queue_watcher.py --gpu 7 --mode sniff
    python scripts/eval_queue_watcher.py --watch_dirs outputs/babilong_sft_phase5a_pure_l3_v2 --mode full
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BABILONG_PKG = PROJECT_ROOT / "third_party" / "babilong-pkg"

# Eval timing estimates (minutes, single GPU, n=100)
# sniff: 9 cells × n=20 ≈ 25 min
# full: 21 cells × n=100 ≈ 6.5h
ETA_MINUTES = {"sniff": 25, "full": 390}

# Mode configurations
MODE_CONFIGS = {
    "sniff": {
        "tasks": ["qa1", "qa2", "qa5"],
        "lengths": ["8k", "16k", "32k"],
        "limit": 20,
    },
    "full": {
        "tasks": ["qa1", "qa2", "qa3", "qa4", "qa5"],
        "lengths": ["0k", "1k", "2k", "4k", "8k", "16k", "32k"],
        "limit": 100,
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-eval new mem_space checkpoints on a dedicated GPU"
    )
    parser.add_argument(
        "--watch_dirs",
        type=str,
        default="outputs/babilong_sft_phase5a_pure_l3_v2,outputs/babilong_sft_phase5b_l1_l3_v2,outputs/babilong_sft_phase4_dual_gate",
        help="Comma-separated list of dirs to watch for ckpts",
    )
    parser.add_argument("--gpu", type=int, default=7, help="GPU index (CUDA_VISIBLE_DEVICES)")
    parser.add_argument(
        "--mode",
        type=str,
        default="sniff",
        choices=["sniff", "full"],
        help="sniff (n=20, 9-cell long-context subset) or full (n=100, all 21 cells)",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default=None,
        help="Where to save eval CSVs. Default: outputs/eval_queue_<mode>/",
    )
    parser.add_argument(
        "--registry_path",
        type=str,
        default="status/eval_queue_processed.json",
        help="Where to record processed ckpts",
    )
    parser.add_argument("--poll_interval", type=int, default=60, help="Seconds between polls")
    parser.add_argument(
        "--max_runs", type=int, default=0, help="Max ckpts to process before exiting (0=forever)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/Meta-Llama-3-8B-Instruct",
        help="Path to base model",
    )
    parser.add_argument("--chunk_size", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# Registry management
# --------------------------------------------------------------------------- #


def load_registry(path: Path) -> dict:
    """Load the processed-ckpt registry from disk."""
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_registry(registry: dict, path: Path) -> None:
    """Save the registry atomically (write to tmp then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(registry, f, indent=2)
    tmp.rename(path)


# --------------------------------------------------------------------------- #
# Checkpoint discovery
# --------------------------------------------------------------------------- #


def discover_ckpts(watch_dirs: list[Path], registry: dict, min_age_sec: float = 30.0) -> list[Path]:
    """Find unprocessed mem_space_adapter*.pt files sorted by mtime ascending.

    Filters:
    - Must match mem_space_adapter*.pt pattern
    - Must NOT already be in registry
    - Must have a sibling adapter_config.json
    - Must be older than min_age_sec (avoid half-written files)
    """
    candidates: list[tuple[float, Path]] = []
    now = time.time()

    for watch_dir in watch_dirs:
        if not watch_dir.is_dir():
            continue
        for pt_file in watch_dir.glob("mem_space_adapter*.pt"):
            abs_path = str(pt_file.resolve())
            if abs_path in registry:
                continue
            # Check sibling adapter_config.json
            config_file = pt_file.parent / "adapter_config.json"
            if not config_file.exists():
                continue
            # Check age
            mtime = pt_file.stat().st_mtime
            if (now - mtime) < min_age_sec:
                continue
            candidates.append((mtime, pt_file))

    # Sort by mtime ascending (process oldest first)
    candidates.sort(key=lambda x: x[0])
    return [c[1] for c in candidates]


# --------------------------------------------------------------------------- #
# Output name construction
# --------------------------------------------------------------------------- #


def build_output_name(ckpt_path: Path) -> str:
    """Build a unique output name from the checkpoint path.

    Examples:
        outputs/babilong_sft_phase5b_l1_l3_v2/mem_space_adapter_step000400.pt
        -> phase5b_l1_l3_v2_step000400

        outputs/babilong_sft_phase5b_l1_l3_v2/mem_space_adapter.pt
        -> phase5b_l1_l3_v2_final
    """
    dir_name = ckpt_path.parent.name
    # Strip common prefix "babilong_sft_"
    short_dir = dir_name
    if short_dir.startswith("babilong_sft_"):
        short_dir = short_dir[len("babilong_sft_"):]

    stem = ckpt_path.stem  # e.g. "mem_space_adapter_step000400" or "mem_space_adapter"
    if "step" in stem:
        # Extract stepNNNNNN part
        step_part = stem.split("step")[-1]  # e.g. "000400"
        return f"{short_dir}_step{step_part}"
    else:
        return f"{short_dir}_final"


def extract_step_number(ckpt_path: Path) -> int | None:
    """Extract numeric step from ckpt filename, or None for final."""
    stem = ckpt_path.stem
    if "step" in stem:
        step_str = stem.split("step")[-1]
        try:
            return int(step_str)
        except ValueError:
            return None
    return None


# --------------------------------------------------------------------------- #
# Build eval command
# --------------------------------------------------------------------------- #


def build_eval_command(
    ckpt_path: Path,
    output_name: str,
    args: argparse.Namespace,
    mode_cfg: dict,
    results_root: Path,
) -> list[str]:
    """Build the subprocess command for running BABILong eval."""
    eval_script = str(PROJECT_ROOT / "scripts" / "run_babilong_mem_space.py")
    adapter_config = str(ckpt_path.parent / "adapter_config.json")

    cmd = [
        sys.executable,
        eval_script,
        "--model_path", str(PROJECT_ROOT / args.model_path),
        "--checkpoint", str(ckpt_path),
        "--adapter_config", adapter_config,
        "--output_name", output_name,
        "--results_folder", str(results_root),
        "--tasks", *mode_cfg["tasks"],
        "--lengths", *mode_cfg["lengths"],
        "--limit", str(mode_cfg["limit"]),
        "--chunk_size", str(args.chunk_size),
        "--max_new_tokens", str(args.max_new_tokens),
        "--device", "cuda:0",
    ]
    return cmd


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def score_eval_results(results_root: Path, output_name: str, mode_cfg: dict) -> dict[str, float]:
    """Score CSVs using babilong.metrics.compare_answers.

    Returns dict like {"qa1/8k": 35.0, "qa1/16k": 32.0, ...}
    """
    # Add babilong to path for import
    babilong_path = str(BABILONG_PKG)
    if babilong_path not in sys.path:
        sys.path.insert(0, babilong_path)

    try:
        from babilong.metrics import TASK_LABELS, compare_answers
    except ImportError:
        print("[eval_queue] WARNING: cannot import babilong.metrics, skipping scoring")
        return {}

    import pandas as pd

    scores: dict[str, float] = {}
    suffix = "_instruction_yes_examples_yes_post_prompt_yes_chat_template_no_system_prompt_no.csv"
    base_dir = results_root / output_name

    for task in mode_cfg["tasks"]:
        labels = TASK_LABELS.get(task, [])
        for length in mode_cfg["lengths"]:
            csv_path = base_dir / f"{task}_{length}{suffix}"
            if not csv_path.exists():
                scores[f"{task}/{length}"] = -1.0
                continue
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                scores[f"{task}/{length}"] = -1.0
                continue
            if df.empty:
                scores[f"{task}/{length}"] = 0.0
                continue

            correct = 0
            total = 0
            for _, row in df.iterrows():
                target = row.get("target")
                output = row.get("output")
                question = row.get("question")
                if not isinstance(target, str) or not isinstance(question, str):
                    continue
                if not isinstance(output, str):
                    output = ""
                total += 1
                if compare_answers(target, output, question, labels):
                    correct += 1
            acc_pct = (correct / total * 100) if total > 0 else 0.0
            scores[f"{task}/{length}"] = round(acc_pct, 1)

    return scores


def compute_long_avg(scores: dict[str, float]) -> float:
    """Average accuracy over scored cells (excluding -1 = missing)."""
    valid = [v for v in scores.values() if v >= 0]
    return round(sum(valid) / len(valid), 1) if valid else 0.0


def load_adapter_config_summary(ckpt_path: Path) -> dict:
    """Load adapter_config.json and extract key config fields."""
    config_path = ckpt_path.parent / "adapter_config.json"
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        return {
            "use_dual_gate": cfg.get("use_dual_gate", False),
            "use_l3_summary": cfg.get("use_l3_summary", False),
            "disable_l1_inject": cfg.get("disable_l1_inject", False),
            "num_slots": cfg.get("num_slots"),
            "top_k": cfg.get("top_k"),
        }
    except Exception:
        return {}


def append_results_jsonl(
    results_jsonl_path: Path,
    ckpt_path: Path,
    step: int | None,
    mode: str,
    scores: dict[str, float],
    long_avg: float,
    config_summary: dict,
) -> None:
    """Append a summary line to the results JSONL file."""
    results_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ckpt": str(ckpt_path),
        "step": step,
        "ts_completed": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": mode,
        "scores": scores,
        "long_avg": long_avg,
        "config_summary": config_summary,
    }
    with open(results_jsonl_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Resolve paths
    os.chdir(PROJECT_ROOT)
    watch_dirs = [Path(d.strip()) for d in args.watch_dirs.split(",")]
    results_root = Path(args.results_root) if args.results_root else Path(f"outputs/eval_queue_{args.mode}")
    registry_path = Path(args.registry_path)
    results_jsonl_path = Path("status/eval_queue_results.jsonl")
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    mode_cfg = MODE_CONFIGS[args.mode]
    eta_min = ETA_MINUTES[args.mode]

    # Set CUDA_VISIBLE_DEVICES
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # Ensure babilong is importable by subprocesses
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    babilong_str = str(BABILONG_PKG)
    project_str = str(PROJECT_ROOT)
    additions = []
    if babilong_str not in pythonpath:
        additions.append(babilong_str)
    if project_str not in pythonpath:
        additions.append(project_str)
    if additions:
        env["PYTHONPATH"] = ":".join(additions + ([pythonpath] if pythonpath else []))
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Load registry
    registry = load_registry(registry_path)
    runs_completed = 0

    print(f"[{now_str()}] eval_queue_watcher started")
    print(f"  mode={args.mode}, gpu={args.gpu}, poll={args.poll_interval}s")
    print(f"  watch_dirs: {[str(d) for d in watch_dirs]}")
    print(f"  results_root: {results_root}")
    print(f"  registry: {registry_path} ({len(registry)} entries)")
    print(f"  ETA per ckpt: ~{eta_min} min")
    print()

    interrupted = False
    current_ckpt: str | None = None

    def handle_sigint(signum, frame):
        nonlocal interrupted
        interrupted = True
        print(f"\n[{now_str()}] KeyboardInterrupt received, finishing up...")

    signal.signal(signal.SIGINT, handle_sigint)

    while not interrupted:
        # Check max_runs
        if args.max_runs > 0 and runs_completed >= args.max_runs:
            print(f"[{now_str()}] max_runs={args.max_runs} reached, exiting.")
            break

        # Discover ckpts
        candidates = discover_ckpts(watch_dirs, registry)
        print(
            f"[{now_str()}] watching {len(watch_dirs)} dirs, "
            f"registry has {len(registry)} entries, scanning... "
            f"found {len(candidates)} new ckpt(s)"
            + (f": {build_output_name(candidates[0])}" if candidates else "")
        )

        if not candidates:
            time.sleep(args.poll_interval)
            continue

        # Pick first unprocessed ckpt
        ckpt_path = candidates[0].resolve()
        ckpt_key = str(ckpt_path)
        output_name = build_output_name(ckpt_path)
        step = extract_step_number(ckpt_path)
        current_ckpt = ckpt_key

        print(f"[{now_str()}] starting eval: {output_name} (ETA ~{eta_min} min)")

        # Mark started in registry
        registry[ckpt_key] = {
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "completed_at": None,
            "results_dir": str(results_root / output_name),
            "status": "running",
        }
        save_registry(registry, registry_path)

        # Build command
        cmd = build_eval_command(ckpt_path, output_name, args, mode_cfg, results_root)

        # Log file
        ts_label = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"eval_queue_{output_name}_{ts_label}.log"

        # Run eval
        try:
            with open(log_file, "w") as log_fh:
                proc = subprocess.Popen(
                    cmd,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    env=env,
                    cwd=str(PROJECT_ROOT),
                )
                retcode = proc.wait()

            if retcode == 0:
                # Success - score results
                scores = score_eval_results(results_root, output_name, mode_cfg)
                long_avg = compute_long_avg(scores)
                config_summary = load_adapter_config_summary(ckpt_path)

                registry[ckpt_key]["completed_at"] = datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                )
                registry[ckpt_key]["status"] = "completed"
                save_registry(registry, registry_path)

                append_results_jsonl(
                    results_jsonl_path, ckpt_path, step, args.mode, scores, long_avg, config_summary
                )

                runs_completed += 1
                print(f"[{now_str()}] completed: {output_name} | long_avg={long_avg}%")
                # Print per-cell scores
                for cell, acc in sorted(scores.items()):
                    if acc >= 0:
                        print(f"    {cell}: {acc}%")
            else:
                # Failed
                registry[ckpt_key]["status"] = "failed"
                registry[ckpt_key]["return_code"] = retcode
                save_registry(registry, registry_path)
                print(
                    f"[{now_str()}] FAILED: {output_name} (rc={retcode}). "
                    f"See {log_file}"
                )

        except KeyboardInterrupt:
            # Handle interrupt during eval
            registry[ckpt_key]["status"] = "interrupted"
            save_registry(registry, registry_path)
            print(f"[{now_str()}] interrupted during eval of {output_name}")
            break
        except Exception as e:
            registry[ckpt_key]["status"] = "failed"
            registry[ckpt_key]["error"] = str(e)
            save_registry(registry, registry_path)
            print(f"[{now_str()}] ERROR: {output_name}: {e}")

        current_ckpt = None

    # Final save
    if current_ckpt and current_ckpt in registry and registry[current_ckpt].get("status") == "running":
        registry[current_ckpt]["status"] = "interrupted"
        save_registry(registry, registry_path)

    print(f"[{now_str()}] eval_queue_watcher exiting. Processed {runs_completed} ckpt(s).")


if __name__ == "__main__":
    main()
