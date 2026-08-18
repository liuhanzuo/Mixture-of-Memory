#!/usr/bin/env python3
"""Reproduce paired McNemar and bootstrap analyses from released predictions."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

TASKS = ["mmlu", "lambada_openai", "boolq", "commonsense_qa", "social_iqa"]


def load_items(path: Path) -> dict[int, dict]:
    items = {}
    with path.open() as handle:
        for line in handle:
            item = json.loads(line)
            items[int(item["item_id"])] = item
    return items


def exact_mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = min(b, c)
    log_two = math.log(2.0)
    probability = sum(
        math.exp(math.lgamma(n + 1) - math.lgamma(k + 1)
                 - math.lgamma(n - k + 1) - n * log_two)
        for k in range(tail + 1)
    )
    return min(1.0, 2.0 * probability)


def paired_bootstrap(differences: np.ndarray, n_boot: int, seed: int,
                     batch_size: int = 250) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = []
    n = len(differences)
    for start in range(0, n_boot, batch_size):
        count = min(batch_size, n_boot - start)
        indices = rng.integers(0, n, size=(count, n))
        means.append(differences[indices].mean(axis=1))
    samples = np.concatenate(means)
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


def analyze(task: str, keep_dir: Path, random_dir: Path,
            n_boot: int, seed: int) -> dict:
    keep = load_items(keep_dir / f"{task}.jsonl")
    random = load_items(random_dir / f"{task}.jsonl")
    common = sorted(set(keep) & set(random))
    paired = [
        (keep[index], random[index])
        for index in common
        if not keep[index].get("nan", False)
        and not random[index].get("nan", False)
    ]
    keep_correct = np.asarray([int(a["correct"]) for a, _ in paired], dtype=np.int8)
    random_correct = np.asarray([int(b["correct"]) for _, b in paired], dtype=np.int8)
    b = int(np.sum((keep_correct == 1) & (random_correct == 0)))
    c = int(np.sum((keep_correct == 0) & (random_correct == 1)))
    differences = keep_correct - random_correct
    lo, hi = paired_bootstrap(differences, n_boot, seed)
    return {
        "task": task,
        "n_paired": len(paired),
        "keep14_acc": float(keep_correct.mean()),
        "random_acc": float(random_correct.mean()),
        "diff": float(differences.mean()),
        "mcnemar_b": b,
        "mcnemar_c": c,
        "mcnemar_p": exact_mcnemar_p(b, c),
        "bootstrap_ci95": [lo, hi],
        "bootstrap_resamples": n_boot,
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=Path("data/per_example"))
    parser.add_argument("--output", type=Path, default=Path("data/paired_analysis.json"))
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    keep_dir = args.data_root / "keep14"
    random_dir = args.data_root / "random_init"
    results = {
        task: analyze(task, keep_dir, random_dir, args.n_boot, args.seed)
        for task in TASKS
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    for task in TASKS:
        row = results[task]
        lo, hi = row["bootstrap_ci95"]
        print(
            f"{task}: n={row['n_paired']} diff={100 * row['diff']:+.2f}pp "
            f"McNemar p={row['mcnemar_p']:.3g} "
            f"CI=[{100 * lo:.2f},{100 * hi:.2f}]pp"
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
