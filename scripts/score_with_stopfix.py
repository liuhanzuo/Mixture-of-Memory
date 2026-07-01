#!/usr/bin/env python
"""Score any BABILong result CSV dir with BOTH raw-official and stop-fix judging.

Background (verified 2026-06-30): the base Llama-3-8B eval has no stop string and
greedily continues `Answer: X  Question: ... Answer: Y ...`. The official
`compare_answers` only takes the first sentence (`split('.')[0]`); its
`split('Question')` is a no-op because `preprocess_output` lowercases first. So a
correct first answer gets failed when the continuation leaks other labels.

This script reports:
  - raw-official : compare_answers on the raw output (the standard, anchor-comparable metric)
  - stop-fix     : compare_answers after truncating output to the first answer
                   (cut at first newline / 'question' (case-insensitive) / second 'answer')
The gap = format-wall (continuation pollution). The stop-fix score is NOT
anchor-comparable; it is a diagnostic of how much of the wall is format vs binding.

Usage: .venv/bin/python scripts/score_with_stopfix.py <task> <csv_glob> [<csv_glob> ...]
  e.g. .venv/bin/python scripts/score_with_stopfix.py qa5 'babilong_results/probe_clean_qa5_16k/*fullchain*.csv'
"""
import sys, glob, csv, re
sys.path.insert(0, "third_party/babilong-pkg")
from babilong.metrics import TASK_LABELS, compare_answers


def stop_fix(output: str) -> str:
    """Truncate to the first answer: cut at first newline, 'question' (any case),
    or a SECOND 'answer:' occurrence."""
    o = output
    # cut at first newline or 'question'
    o = re.split(r'(?i)\bquestion\b|\n', o)[0]
    # if there are two 'answer:' segments, keep only the first
    parts = re.split(r'(?i)answer\s*:', o)
    if len(parts) >= 3:  # text before first 'answer:', first answer, second answer...
        o = parts[0] + 'answer:' + parts[1]
    return o


def load_latest_per_shard(pat: str):
    bys = {}
    for f in sorted(glob.glob(pat)):
        sh = re.search(r'shard(\d+)', f)
        bys[sh.group(0) if sh else f] = f
    rows = []
    for f in bys.values():
        try:
            rows += list(csv.DictReader(open(f)))
        except Exception:
            pass
    return rows


def score(task: str, pat: str):
    labels = TASK_LABELS[task]
    rows_all = load_latest_per_shard(pat)
    rows = [r for r in rows_all if r.get('output') not in ('[OOM]', None, '')]
    n_oom = sum(1 for r in rows_all if r.get('output') == '[OOM]')
    if not rows:
        return None
    raw = 100 * sum(compare_answers(r['target'], r['output'], r.get('question', ''), labels) for r in rows) / len(rows)
    fix = 100 * sum(compare_answers(r['target'], stop_fix(r['output']), r.get('question', ''), labels) for r in rows) / len(rows)
    return raw, fix, len(rows), n_oom, len(glob.glob(pat))


if __name__ == "__main__":
    task = sys.argv[1]
    pats = sys.argv[2:]
    print(f"task={task} labels={TASK_LABELS[task]}")
    print(f"{'pattern':50s} {'raw-off':>8s} {'stop-fix':>9s} {'gap':>6s} {'n':>5s} {'OOM':>4s} {'files':>6s}")
    for pat in pats:
        r = score(task, pat)
        if r is None:
            print(f"{pat[:50]:50s}  (no data, {len(glob.glob(pat))} files)")
            continue
        raw, fix, n, noom, nf = r
        short = pat.split('/')[-2] if '/' in pat else pat
        print(f"{short[:50]:50s} {raw:7.0f} {fix:8.0f} {fix-raw:+5.0f} {n:5d} {noom:4d} {nf:6d}")
