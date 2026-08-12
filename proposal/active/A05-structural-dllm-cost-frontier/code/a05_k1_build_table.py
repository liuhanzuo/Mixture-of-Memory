#!/usr/bin/env python3
"""A05 K1 -- build the sweep table and apply the pre-registered K1 decision rule.

K1 (registered in PROPOSAL.md §3 on 2026-08-12, before any A05 measurement):
    fires if DreamOn at its best NON-ORACLE canvas reaches within 5.0 pp of
    Scaffold Medium on BOTH benchmarks.

Scaffold Medium reference (NOT recomputed here -- wzc1-only checkpoint, read from
DLLM_RESULTS_20260807.md:447 / :456): HE+ .177, MBPP+ .354.

The oracle cells are tabulated but excluded from the decision, per invariant 4.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SCAFFOLD_MEDIUM = {"humaneval": 0.177, "mbpp": 0.354}
SCAFFOLD_PROVENANCE = (
    "dllm_draft/DLLM_RESULTS_20260807.md:447 (HE+ tier table) and :456 (MBPP+ tier table); "
    "checkpoint scaffold_sft_stage1 global_step_4465, wzc1-only, NOT recomputed by K1"
)
AR_CEILING = {"humaneval": 0.707, "mbpp": 0.680}
ARCHIVE_R2 = {"humaneval": 0.122, "mbpp": 0.085}
K1_THRESHOLD_PP = 5.0
# Roadmap continuation criteria for the mechanism question (PROPOSAL.md §3 K1 tail).
RATIO_CRITERION = 0.8
PARSEABILITY_CRITERION = 0.90

CANVASES = ["8", "32", "128", "512", "oracle"]
BENCHES = ["humaneval", "mbpp"]
BENCH_LABEL = {"humaneval": "HE+", "mbpp": "MBPP+"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells-dir", required=True, help="dir of per-cell *.json from the grader")
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    cells_dir = Path(args.cells_dir)
    cells = {}
    for path in sorted(cells_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        cells[payload["label"]] = payload

    table = []
    for bench in BENCHES:
        for canvas in CANVASES:
            label = f"{'he' if bench == 'humaneval' else 'mbpp'}_{'oracle' if canvas == 'oracle' else 'c' + canvas}"
            cell = cells.get(label)
            if cell is None:
                table.append({"label": label, "dataset": bench, "canvas": canvas, "status": "MISSING"})
                continue
            scores, stats = cell["scores"], cell["cost_and_behaviour"]
            table.append({
                "label": label,
                "dataset": bench,
                "benchmark": BENCH_LABEL[bench],
                "canvas": canvas,
                "is_oracle": canvas == "oracle",
                "n": scores["n"],
                "pass_at_1_base": round(scores["pass_at_1_base"], 4),
                "pass_at_1_plus": round(scores["pass_at_1_plus"], 4),
                "n_pass_plus": scores["n_pass_plus"],
                "nfe_mean": round(stats["nfe_mean"], 1),
                "nfe_median": stats["nfe_median"],
                "tokens_fed_effective_mean": round(stats["tokens_fed_effective_mean"], 1),
                "tokens_fed_padded_mean": round(stats["tokens_fed_padded_mean"], 1),
                "generated_tokens_mean": round(stats["generated_tokens_mean"], 2),
                "generated_tokens_median": stats["generated_tokens_median"],
                "emitted_gold_ratio_median": stats["emitted_gold_ratio_median"],
                "emitted_gold_ratio_mean": stats["emitted_gold_ratio_mean"],
                "parseability": round(stats["parseability"], 4),
                "empty_raw_output": stats["empty_raw_output"],
                "generation_errors": stats["generation_errors"],
                "initial_masks_observed": stats["initial_masks_observed"],
                "initial_masks_clamped_items": stats["initial_masks_clamped_items"],
                "long_span_gold_ge65": stats["long_span_gold_ge65"],
                "wall_seconds_sum": round(stats["wall_seconds_sum"], 1),
                "gpu_hours": round(stats["wall_seconds_sum"] / 3600.0, 3),
                "status": "OK",
            })

    # ---- K1 decision on non-oracle cells only ----
    decision = {}
    for bench in BENCHES:
        rows = [r for r in table if r.get("status") == "OK" and r["dataset"] == bench and not r["is_oracle"]]
        if not rows:
            decision[bench] = {"status": "NO_DATA"}
            continue
        best = max(rows, key=lambda r: r["pass_at_1_plus"])
        gap = (SCAFFOLD_MEDIUM[bench] - best["pass_at_1_plus"]) * 100.0
        decision[bench] = {
            "benchmark": BENCH_LABEL[bench],
            "scaffold_medium": SCAFFOLD_MEDIUM[bench],
            "best_nonoracle_canvas": best["canvas"],
            "best_nonoracle_pass_at_1_plus": best["pass_at_1_plus"],
            "gap_pp_scaffold_minus_dreamon": round(gap, 2),
            "within_5pp": bool(gap <= K1_THRESHOLD_PP),
            "dreamon_exceeds_scaffold": bool(gap < 0),
            "archive_r2_reference": ARCHIVE_R2[bench],
            "ar_ceiling": AR_CEILING[bench],
        }
    within = [decision[b].get("within_5pp") for b in BENCHES]
    k1_fires = all(w is True for w in within)

    # ---- mechanism question: does under-generation go away as the canvas grows? ----
    mechanism = {}
    for bench in BENCHES:
        rows = [r for r in table if r.get("status") == "OK" and r["dataset"] == bench]
        rows.sort(key=lambda r: (r["is_oracle"], 0 if r["is_oracle"] else int(r["canvas"])))
        mechanism[BENCH_LABEL[bench]] = {
            "ratio_median_by_canvas": {r["canvas"]: r["emitted_gold_ratio_median"] for r in rows},
            "parseability_by_canvas": {r["canvas"]: r["parseability"] for r in rows},
            "empty_output_by_canvas": {r["canvas"]: r["empty_raw_output"] for r in rows},
            "gen_tokens_mean_by_canvas": {r["canvas"]: r["generated_tokens_mean"] for r in rows},
            "long_span_ge65_by_canvas": {r["canvas"]: r["long_span_gold_ge65"] for r in rows},
        }
        nonoracle = [r for r in rows if not r["is_oracle"]]
        if nonoracle:
            best_ratio = max(nonoracle, key=lambda r: (r["emitted_gold_ratio_median"] or 0.0))
            ls = best_ratio["long_span_gold_ge65"]
            mechanism[BENCH_LABEL[bench]]["continuation_criteria"] = {
                "criterion_ratio_median_ge": RATIO_CRITERION,
                "criterion_parseability_ge": PARSEABILITY_CRITERION,
                "best_nonoracle_canvas_by_ratio": best_ratio["canvas"],
                "ratio_median_there": best_ratio["emitted_gold_ratio_median"],
                "ratio_criterion_met": bool((best_ratio["emitted_gold_ratio_median"] or 0) >= RATIO_CRITERION),
                "long_span_ratio_median": ls["emitted_gold_ratio_median"],
                "long_span_parseability": ls["parseability"],
                "long_span_ratio_criterion_met": bool((ls["emitted_gold_ratio_median"] or 0) >= RATIO_CRITERION),
                "long_span_parseability_criterion_met": bool((ls["parseability"] or 0) >= PARSEABILITY_CRITERION),
            }

    payload = {
        "gate": "K1",
        "proposal": "A05-structural-dllm-cost-frontier",
        "registered_rule": (
            "K1 fires iff DreamOn at its best NON-ORACLE canvas is within 5.0 pp of "
            "Scaffold Medium on BOTH benchmarks (oracle arm excluded)."
        ),
        "k1_threshold_pp": K1_THRESHOLD_PP,
        "scaffold_reference": {"values": SCAFFOLD_MEDIUM, "provenance": SCAFFOLD_PROVENANCE},
        "verdict": "K1_FIRES" if k1_fires else "K1_DOES_NOT_FIRE",
        "k1_fires": k1_fires,
        "per_benchmark_decision": decision,
        "mechanism_under_generation": mechanism,
        "table": table,
        "total_gpu_hours": round(sum(r.get("gpu_hours", 0) for r in table if r.get("status") == "OK"), 2),
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2) + "\n")

    print(f"VERDICT: {payload['verdict']}")
    for bench in BENCHES:
        d = decision[bench]
        if d.get("status") == "NO_DATA":
            print(f"  {bench}: NO DATA")
            continue
        print(f"  {d['benchmark']}: best non-oracle canvas={d['best_nonoracle_canvas']} "
              f"pass@1={d['best_nonoracle_pass_at_1_plus']:.4f} vs Scaffold {d['scaffold_medium']:.3f} "
              f"-> gap {d['gap_pp_scaffold_minus_dreamon']:+.2f} pp, within_5pp={d['within_5pp']}")
    print(f"  total GPU-h: {payload['total_gpu_hours']}")


if __name__ == "__main__":
    main()
