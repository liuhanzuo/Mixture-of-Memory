#!/usr/bin/env python3
"""Audit and regenerate Paper B appendix artifacts from merged summaries.

The script validates headline scores against `data/raw/**/summary.json`,
regenerates the detailed appendix tables, and rebuilds raw-data-driven figures.

SCOPE LIMIT (2026-08-17): this script does NOT validate the keep8/keep10/keep12
rows of Table 4 as printed. It resolves every path through `RAW = ROOT/data/raw`,
and the `close()` assertions below still target the older rungs
(7B_keep12_step111500, 7B_keep10_step10000, 7B_keep8_step44000). Commit 6d15049
re-measured those three rows from `*_v2` directories that live only on the zwfy6
disk and are not in `data/raw` at all -- and `data/raw/.../7B_keep12_step124000/`
is in fact the superseded six-of-eight-shard file. A green run of this script is
therefore NOT evidence for Table 4's three shallow rows; their provenance and
33-cell recomputation are in `paperB/TABLE4_PROVENANCE_20260817.md`.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RESULTS = ROOT.parent / "results"
PAIRED = ROOT / "data" / "paired_analysis.json"


def load(rel: str):
    return json.load(open(RAW / rel))


def close(name: str, got: float, expected: float, tol: float = 7e-4):
    if not math.isclose(got, expected, abs_tol=tol):
        raise AssertionError(f"{name}: got {got}, expected {expected}")
    print(f"OK  {name:38s} {got:.6f}")


def task(model: str, suite: str, key: str, metric: str = "acc") -> float:
    suffix = "_know" if suite == "know" else ""
    d = load(
        f"olmo2_downstream_results/{model}{suffix}/summary.json"
    )
    return d["tasks"][key][metric]


def ppl(model: str) -> float:
    return load(f"olmo2_ppl_results/{model}/summary.json")["ppl"]


def main():
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_appendix_tables.py")],
        cwd=ROOT,
        check=True,
    )

    # Full-base anchors.
    close("7B base held-out PPL", ppl("7B_base_full"), 7.398)
    close("7B base HellaSwag acc_norm",
          task("7B_base_full", "core", "hellaswag", "acc_norm"), .805)
    close("7B base MMLU",
          task("7B_base_full", "know", "mmlu"), .6053)

    # Depth frontier.
    close("keep10 PPL", ppl("7B_keep10_step10000"), 17.239)
    close("keep10 MMLU",
          task("7B_keep10_step10000", "know", "mmlu"), .2539)
    close("keep12 PPL", ppl("7B_keep12_step111500"), 11.56596)
    close("keep12 MMLU",
          task("7B_keep12_step111500", "know", "mmlu"), .2726)
    close("keep8 SocialIQA raw",
          task("7B_keep8_step44000", "know", "social_iqa"), .4002)
    close("keep12 SocialIQA raw",
          task("7B_keep12_step111500", "know", "social_iqa"), .4150)
    close("keep14 128k PPL", ppl("7B_keep14_step128000"), 10.826)
    close("keep14 128k MMLU",
          task("7B_keep14_step128000", "know", "mmlu"), .3012)
    close("keep14 153.5k PPL", ppl("7B_keep14_step153500"), 10.693)
    close("keep14 153.5k MMLU",
          task("7B_keep14_step153500", "know", "mmlu"), .3124)
    close("keep14 200k PPL", ppl("7B_keep14_step200000"), 10.5613)
    close("keep14 200k MMLU",
          task("7B_keep14_step200000", "know", "mmlu"), .3191)
    close("keep14 200k BoolQ raw",
          task("7B_keep14_step200000", "know", "boolq"), .6382)
    close("keep14 200k SIQA raw",
          task("7B_keep14_step200000", "know", "social_iqa"), .4340)
    close("fully random-init PPL", ppl("7B_scratch16L_step200000"), 11.498)
    close("fully random-init MMLU",
          task("7B_scratch16L_step200000", "know", "mmlu"), .2461)
    close("frozen-front PPL", ppl("7B_freezefront_step200000"), 12.79735)
    close("frozen-front MMLU",
          task("7B_freezefront_step200000", "know", "mmlu"), .2628)
    close("ShortGPT step0 PPL", ppl("7B_shortgpt_step0"), 401.12418)
    close("ShortGPT step0 MMLU",
          task("7B_shortgpt_step0", "know", "mmlu"), .2620)
    close("ShortGPT step0 LAMBADA",
          task("7B_shortgpt_step0", "know", "lambada_openai"), .000582)

    paired = json.loads(PAIRED.read_text())
    close("paired MMLU diff", paired["mmlu"]["diff"], .0729953)
    close("paired MMLU CI low", paired["mmlu"]["bootstrap_ci95"][0], .0634507)
    close("paired MMLU CI high", paired["mmlu"]["bootstrap_ci95"][1], .0826111)
    if not all(row["mcnemar_p"] < .05 and row["bootstrap_ci95"][0] > 0
               for row in paired.values()):
        raise AssertionError("paired significance checks failed")

    probe = json.loads((RESULTS / "probe_linguistic_olmo2_7b.json").read_text())
    division = probe["division_of_labour"]
    close("OLMo semantic sat95 depth", division["semantic_sat95_frac_depth"], .073)
    close("OLMo next-token sat95 depth", division["nexttoken_sat95_frac_depth"], 1.0)

    # Within-arm trajectories.
    for step, expected in [(10000, .2542), (25000, .2502), (44000, .2463)]:
        close(
            f"keep8 MMLU {step // 1000}k",
            task(f"7B_keep8_step{step}", "know", "mmlu"),
            expected,
        )
    for step, expected in [(50000, .2495), (100000, .2558),
                           (147000, .2529), (150000, .2480)]:
        close(
            f"1B keep7 MMLU {step / 1000:g}k",
            task(f"1B_keep7_step{step}", "know", "mmlu"),
            expected,
        )

    # Subject maps should be complete for all compared 7B knowledge runs.
    for model in [
        "7B_base_full", "7B_keep10_step10000", "7B_keep12_step111500",
        "7B_keep14_step128000", "7B_keep14_step153500",
        "7B_keep14_step200000", "7B_freezefront_step200000",
        "7B_scratch16L_step200000",
    ]:
        d = load(
            f"olmo2_downstream_results/{model}_know/summary.json"
        )
        n_subjects = len(d["tasks"]["mmlu"]["subjects"])
        if n_subjects != 57:
            raise AssertionError(f"{model}: {n_subjects} MMLU subjects")
        print(f"OK  {model:38s} 57 subjects")

    # Regenerate raw-data-driven appendix figures.
    for script in ["fig_capability_cliff.py", "fig_mmlu_subjects.py",
                   "fig_two_depths.py"]:
        subprocess.run(
            [sys.executable, str(ROOT / "figures" / script)],
            cwd=ROOT,
            check=True,
        )

    print("\nAll appendix raw-summary checks passed.")


if __name__ == "__main__":
    main()
