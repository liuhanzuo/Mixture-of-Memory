#!/usr/bin/env python3
"""Dependence-aware reanalysis of the saved equal-latency paired scores.

This script performs no model inference.  It consumes the score-only exports in
``source/{bm25,bge}/paired_scores_comem12_replay10.jsonl`` and writes one
reproducible JSON summary.  All reported effects are percentage-point differences
``CoMem - replay``.

Bootstrap definitions
---------------------
pooled_iid:
    Sensitivity analysis matching the original paper: pool all 900 paired
    example differences and resample 900 pairs with replacement.
stratified_fixed_cells:
    Hold the nine observed cells fixed.  Within each cell, resample exactly that
    cell's observed n=100 paired examples; average the nine resampled cell
    means with equal weight.
hierarchical_cells_then_examples:
    For each replicate, sample nine cell labels with replacement from the nine
    observed cells.  For every selected cell occurrence, resample n=100 paired
    examples from that cell, compute its mean, and average the nine selected
    cell means.  Repeated cell labels receive independent within-cell draws.

The LoCoMo slice contains the first 100 flattened questions, all from
conversation 0.  Consequently there is only one observed conversation cluster:
conversation-cluster resampling is not identifiable and is not performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence


SCHEMA_VERSION = "paperA.equal_latency_dependence_reanalysis.v1"
SELECTORS = ("bm25", "bge")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row["selector"] != path.parent.name:
                raise ValueError(
                    f"{path}:{line_no}: selector={row['selector']!r} "
                    f"does not match directory {path.parent.name!r}"
                )
            rows.append(row)
    return rows


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated empirical percentile, NumPy-compatible."""
    if not sorted_values:
        raise ValueError("empty percentile input")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q out of range: {q}")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return float(
        sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac
    )


def summarize_draws(draws: Iterable[float], alpha: float) -> dict:
    vals = sorted(float(x) for x in draws)
    return {
        "ci_percentile_pp": [
            percentile(vals, alpha / 2.0),
            percentile(vals, 1.0 - alpha / 2.0),
        ],
        "bootstrap_mean_pp": sum(vals) / len(vals),
        "n_boot": len(vals),
        "alpha": alpha,
    }


def pooled_iid_bootstrap(
    cells: dict[str, list[float]], n_boot: int, seed: int, alpha: float
) -> dict:
    diffs = [x for cell in sorted(cells) for x in cells[cell]]
    rng = random.Random(seed)
    n = len(diffs)
    draws = [
        sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(n_boot)
    ]
    out = summarize_draws(draws, alpha)
    out.update(
        {
            "definition": (
                "Pool all 900 paired differences and resample 900 pairs IID; "
                "retained only as sensitivity to match the original analysis."
            ),
            "seed": seed,
            "n_pairs": n,
        }
    )
    return out


def stratified_fixed_cells_bootstrap(
    cells: dict[str, list[float]], n_boot: int, seed: int, alpha: float
) -> dict:
    names = sorted(cells)
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        means = []
        for name in names:
            values = cells[name]
            n = len(values)
            means.append(
                sum(values[rng.randrange(n)] for _ in range(n)) / n
            )
        draws.append(sum(means) / len(means))
    out = summarize_draws(draws, alpha)
    out.update(
        {
            "definition": (
                "Keep all nine observed cells fixed; within every cell resample "
                "exactly its observed n=100 paired examples; equal-weight the "
                "nine resampled cell means."
            ),
            "seed": seed,
            "n_cells_fixed": len(names),
            "within_cell_n": {name: len(cells[name]) for name in names},
        }
    )
    return out


def hierarchical_cells_then_examples_bootstrap(
    cells: dict[str, list[float]], n_boot: int, seed: int, alpha: float
) -> dict:
    names = sorted(cells)
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        sampled_names = [names[rng.randrange(len(names))] for _ in names]
        sampled_cell_means = []
        for name in sampled_names:
            values = cells[name]
            n = len(values)
            sampled_cell_means.append(
                sum(values[rng.randrange(n)] for _ in range(n)) / n
            )
        draws.append(sum(sampled_cell_means) / len(sampled_cell_means))
    out = summarize_draws(draws, alpha)
    out.update(
        {
            "definition": (
                "Resample nine cell labels with replacement; for each selected "
                "cell occurrence independently resample n=100 paired examples "
                "within that cell; equal-weight the nine resulting cell means."
            ),
            "seed": seed,
            "n_cell_draws_per_replicate": len(names),
            "within_cell_n": {name: len(cells[name]) for name in names},
        }
    )
    return out


def analyze_selector(
    selector: str, path: Path, n_boot: int, seed: int, alpha: float
) -> dict:
    rows = load_jsonl(path)
    if len(rows) != 900:
        raise ValueError(f"{selector}: expected 900 rows, found {len(rows)}")

    cells: dict[str, list[float]] = defaultdict(list)
    cell_scores: dict[str, list[tuple[float, float]]] = defaultdict(list)
    ids = set()
    k_comem = sorted({int(row["k_comem"]) for row in rows})
    k_replay = sorted({int(row["k_replay"]) for row in rows})
    if k_comem != [12] or k_replay != [10]:
        raise ValueError(
            f"{selector}: expected CoMem k=12 and replay k=10, got "
            f"{k_comem=} {k_replay=}"
        )
    for row in rows:
        key = (row["cell"], int(row["example_id"]))
        if key in ids:
            raise ValueError(f"{selector}: duplicate pair {key}")
        ids.add(key)
        diff = float(row["diff_comem_minus_replay_pp"])
        expected = 100.0 * (
            float(row["comem_score"]) - float(row["replay_score"])
        )
        if not math.isclose(diff, expected, abs_tol=1e-9):
            raise ValueError(
                f"{selector}: stored diff mismatch for {key}: {diff} vs {expected}"
            )
        cells[row["cell"]].append(diff)
        cell_scores[row["cell"]].append(
            (float(row["comem_score"]), float(row["replay_score"]))
        )

    if len(cells) != 9:
        raise ValueError(f"{selector}: expected 9 cells, found {len(cells)}")
    if any(len(values) != 100 for values in cells.values()):
        raise ValueError(
            f"{selector}: expected n=100 in every cell; "
            f"got { {k: len(v) for k, v in cells.items()} }"
        )

    per_cell = {}
    cell_means = {}
    for cell in sorted(cells):
        pairs = cell_scores[cell]
        comem = 100.0 * sum(x[0] for x in pairs) / len(pairs)
        replay = 100.0 * sum(x[1] for x in pairs) / len(pairs)
        diff = sum(cells[cell]) / len(cells[cell])
        cell_means[cell] = diff
        per_cell[cell] = {
            "n": len(pairs),
            "comem_pp": comem,
            "replay_pp": replay,
            "diff_comem_minus_replay_pp": diff,
        }

    macro = sum(cell_means.values()) / len(cell_means)
    pooled = sum(x for values in cells.values() for x in values) / len(rows)
    if not math.isclose(macro, pooled, abs_tol=1e-12):
        raise ValueError(
            "equal-cell macro and pooled mean should coincide because all cells "
            "have n=100"
        )

    loco_rows = [row for row in rows if row["benchmark"] == "locomo"]
    loco_clusters = sorted(
        {row.get("conversation_cluster") for row in loco_rows}
    )
    if loco_clusters != ["conv0"]:
        raise ValueError(
            f"{selector}: expected only LoCoMo conv0, got {loco_clusters}"
        )

    loco = {}
    for omitted in sorted(cells):
        retained = [value for cell, value in cell_means.items() if cell != omitted]
        loco[omitted] = sum(retained) / len(retained)
    loco_values = list(loco.values())

    selector_seed = seed + (0 if selector == "bm25" else 1_000_000)
    return {
        "source": {
            # Relative to the analysis root; avoids leaking a private
            # filesystem path and keeps the artifact relocatable.
            "path": str(path.relative_to(path.parents[2])),
            "sha256": sha256_file(path),
            "n_pairs": len(rows),
            "n_cells": len(cells),
            "k_comem": k_comem[0],
            "k_replay": k_replay[0],
            "n_per_cell": {cell: len(values) for cell, values in sorted(cells.items())},
            "contains_benchmark_text": False,
            "contains_predictions": False,
            "fields": sorted(rows[0].keys()),
        },
        "estimand": (
            "Equal-weight mean of the nine observed cell mean paired score "
            "differences, in percentage points (CoMem - replay)."
        ),
        "point_estimate_pp": macro,
        "per_cell": per_cell,
        "bootstrap": {
            "stratified_fixed_cells": stratified_fixed_cells_bootstrap(
                cells, n_boot, selector_seed + 101, alpha
            ),
            "hierarchical_cells_then_examples": (
                hierarchical_cells_then_examples_bootstrap(
                    cells, n_boot, selector_seed + 202, alpha
                )
            ),
            "pooled_iid_sensitivity": pooled_iid_bootstrap(
                cells, n_boot, selector_seed + 303, alpha
            ),
        },
        "leave_one_cell_out": {
            "definition": (
                "Drop one observed cell, then equally average the remaining "
                "eight observed cell mean paired differences."
            ),
            "estimates_pp": loco,
            "range_pp": [min(loco_values), max(loco_values)],
            "sign_counts": {
                "negative": sum(x < 0 for x in loco_values),
                "zero": sum(x == 0 for x in loco_values),
                "positive": sum(x > 0 for x in loco_values),
            },
        },
        "locomo_dependence": {
            "n_items": len(loco_rows),
            "observed_conversation_clusters": loco_clusters,
            "n_observed_conversation_clusters": len(loco_clusters),
            "conversation_cluster_bootstrap_performed": False,
            "reason": (
                "The retained first-100 LoCoMo slice is entirely conversation "
                "0; with one observed cluster, conversation-level resampling is "
                "not identifiable.  The cell is kept as one diagnostic cell "
                "rather than treated as 100 independent conversations."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing source/{bm25,bge}.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--n-boot", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    if args.n_boot <= 0:
        raise ValueError("--n-boot must be positive")
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("--alpha must lie in (0,1)")

    root = args.root.resolve()
    output = args.output or root / "equal_latency_dependence_results.json"
    results = {
        "schema_version": SCHEMA_VERSION,
        "analysis_script": {
            # Stable release-relative name rather than a host absolute path.
            "path": "reanalyze_equal_latency.py",
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "created_from_saved_scores_only": True,
        "model_evaluation_or_training_performed": False,
        "n_boot": args.n_boot,
        "base_seed": args.seed,
        "alpha": args.alpha,
        "selectors": {},
    }
    for selector in SELECTORS:
        path = (
            root
            / "source"
            / selector
            / "paired_scores_comem12_replay10.jsonl"
        )
        results["selectors"][selector] = analyze_selector(
            selector, path, args.n_boot, args.seed, args.alpha
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
        f.write("\n")

    print(json.dumps({
        "output": str(output),
        "sha256": sha256_file(output),
        "point_estimates_pp": {
            selector: results["selectors"][selector]["point_estimate_pp"]
            for selector in SELECTORS
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
