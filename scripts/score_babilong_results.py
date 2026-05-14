"""Compute BABILong accuracy from CSV files using babilong.metrics.compare_answers."""
import sys, os, json
from pathlib import Path
import pandas as pd

sys.path.insert(0, '/apdcephfs_wzc1/share_303098609/pighzliu_code/babilong')
from babilong.metrics import TASK_LABELS, compare_answers

def score_dir(results_dir, model_name, tasks, lengths):
    base = Path(results_dir) / model_name
    if not base.exists():
        print(f"NO DIR: {base}")
        return {}
    suffix = "_instruction_yes_examples_yes_post_prompt_yes_chat_template_no_system_prompt_no.csv"
    grid = {}
    for task in tasks:
        for length in lengths:
            csv = base / f"{task}_{length}{suffix}"
            if not csv.exists():
                grid[(task, length)] = None
                continue
            df = pd.read_csv(csv)
            if df.empty:
                grid[(task, length)] = None
                continue
            labels = TASK_LABELS[task]
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
            grid[(task, length)] = (correct, total, correct / total if total else 0.0)
    return grid

def print_grid(name, grid, tasks, lengths):
    print(f"\n=== {name} ===")
    print(f"task    " + "  ".join(f"{l:>5s}" for l in lengths))
    for t in tasks:
        row = []
        for l in lengths:
            v = grid.get((t, l))
            if v is None:
                row.append(f"{'-':>5s}")
            else:
                _, _, acc = v
                row.append(f"{acc*100:>4.0f}%")
        print(f"{t:6s}  " + "  ".join(f"{c:>5s}" for c in row))
    # Average per length
    avg_row = []
    for l in lengths:
        accs = [grid[(t, l)][2] for t in tasks if grid.get((t, l)) is not None]
        avg_row.append(f"{sum(accs)/len(accs)*100:>4.0f}%" if accs else f"{'-':>5s}")
    print(f"{'AVG':6s}  " + "  ".join(f"{c:>5s}" for c in avg_row))

if __name__ == "__main__":
    tasks = ["qa1", "qa2", "qa3", "qa4", "qa5"]
    lengths = ["0k", "1k", "2k", "4k", "8k", "16k", "32k"]
    base_dir = "/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/babilong_results"

    for model_name in ["Meta-Llama-3-8B", "LM2-iter12000"]:
        grid = score_dir(base_dir, model_name, tasks, lengths)
        if grid:
            print_grid(model_name, grid, tasks, lengths)
