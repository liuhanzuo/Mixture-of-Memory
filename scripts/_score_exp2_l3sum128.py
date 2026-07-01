#!/usr/bin/env python
"""Score EXP-2 l3_n_summary=128 BABILong eval (step500 + step1000) using
third_party/babilong-pkg/babilong/metrics.py (babilong.metrics canonical).
Prints qa1/qa2/qa5 x 0k-32k grids for both ckpts + comparison vs P11 base
step500 + EXP-4 (l3_n_summary=32). Writes logs/exp2_l3sum128_scores.txt.
"""
from __future__ import annotations
import csv, sys
from pathlib import Path

ROOT = Path("/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory")
sys.path.insert(0, str(ROOT / "third_party" / "babilong-pkg"))
from babilong.metrics import TASK_LABELS, compare_answers  # type: ignore  # noqa

RESULTS = ROOT / "babilong_results" / "exp2_l3sum128"
TASKS = ["qa1", "qa2", "qa5"]
LENGTHS = ["0k", "1k", "2k", "4k", "8k", "16k", "32k"]
CKPTS = ["step500", "step1000"]
SUFFIX = "_instruction_yes_examples_yes_post_prompt_yes_chat_template_no_system_prompt_no.csv"

# Reference: P11 base step500 qa5 (chunk512 deltarule normreadout)
P11_BASE_S500_QA5 = {"0k": 74, "1k": 89, "2k": 81, "4k": 60, "8k": 48, "16k": 45, "32k": 44}
# EXP-4 (l3_n_summary=32, reduced capacity) qa5 step500 8k known-bad data point
EXP4_S500_QA5_8K = 10


def score_cell(path: Path, task: str):
    if not path.exists():
        return (-1, 0)
    labels = TASK_LABELS[task]
    correct = total = 0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            total += 1
            if compare_answers((row.get("target") or "").strip(),
                               (row.get("output") or "").strip(),
                               (row.get("question") or "").strip(), labels):
                correct += 1
    return (correct, total)


def grid_for(ck: str):
    g = {}
    for task in TASKS:
        for L in LENGTHS:
            # fine-scheduler dir (ckpt_task_length); fall back to bundled dir
            fine = RESULTS / f"exp2_l3sum128_{ck}_{task}_{L}" / f"{task}_{L}{SUFFIX}"
            bundled = RESULTS / f"exp2_l3sum128_{ck}_{L}" / f"{task}_{L}{SUFFIX}"
            path = fine if fine.exists() else bundled
            g[(task, L)] = score_cell(path, task)
    return g


def fmt_pct(c, t):
    if t == 0 or c < 0:
        return "  --"
    return f"{100.0*c/t:5.1f}"


def print_grid(ck, g, out):
    header = " task  " + "".join(f"  {l:>4}" for l in LENGTHS) + "   | mean"
    out.append(f"\n=== EXP-2 l3_n_summary=128  {ck} ===")
    out.append(header)
    out.append("-" * len(header))
    overall_s = overall_n = 0
    for task in TASKS:
        cells, ts, tn = [], 0.0, 0
        for L in LENGTHS:
            c, t = g[(task, L)]
            cells.append(fmt_pct(c, t))
            if t > 0 and c >= 0:
                pct = 100.0*c/t
                ts += pct; tn += 1; overall_s += pct; overall_n += 1
        tm = ts/tn if tn else float("nan")
        out.append(f" {task}   " + "".join(f"  {x:>4}" for x in cells) + f"   | {tm:5.2f}")
    out.append("-" * len(header))
    ov = overall_s/overall_n if overall_n else float("nan")
    out.append(f"Overall mean ({overall_n} cells): {ov:5.2f}")
    return g


def main():
    out = []
    grids = {}
    for ck in CKPTS:
        grids[ck] = grid_for(ck)
        print_grid(ck, grids[ck], out)

    # Comparison: qa5 long-range vs P11 base + EXP-4
    out.append("\n=== qa5 long-range comparison: l3_n_summary 128 vs base 64 vs EXP-4 32 ===")
    out.append(f"{'length':>6} | {'base64(P11 s500)':>16} | {'sum128 s500':>11} | {'sum128 s1000':>12}")
    out.append("-" * 56)
    for L in LENGTHS:
        c5, t5 = grids["step500"][("qa5", L)]
        c10, t10 = grids["step1000"][("qa5", L)]
        v5 = f"{100.0*c5/t5:.0f}" if t5 and c5 >= 0 else "--"
        v10 = f"{100.0*c10/t10:.0f}" if t10 and c10 >= 0 else "--"
        base = P11_BASE_S500_QA5.get(L, "--")
        out.append(f"{L:>6} | {str(base):>16} | {v5:>11} | {v10:>12}")
    out.append(f"\nEXP-4 (l3_n_summary=32) qa5 step500 8k = {EXP4_S500_QA5_8K} (reduced-capacity data point)")

    text = "\n".join(out)
    print(text)
    score_file = ROOT / "logs" / "exp2_l3sum128_scores.txt"
    score_file.parent.mkdir(parents=True, exist_ok=True)
    score_file.write_text(text + "\n")
    print(f"\nWrote {score_file}")
    (ROOT / "logs" / "eval_exp2_l3sum128" / "DONE_SCORING").write_text("done\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
