#!/usr/bin/env python3
"""
enrich_per_example_normscores.py  (2026-08-08)

Post-hoc enrichment: add `norm_lens` and `norm_scores` to existing per_example_*.jsonl
files that were produced by eval_olmo2_probe2_downstream.py WITHOUT those fields.

The script:
  1. Calls load_task_examples() (same logic as the eval harness) to get norm_lens
     for each example.
  2. Reads each per_example_{task}.jsonl in the target directory.
  3. For each row, adds:
       norm_lens:   {letter: int}  -- raw candidate character count (=c[2] in cands)
       norm_scores: {letter: float}  -- option_scores[letter] / max(norm_lens[letter], 1)
  4. Writes a NEW file to the same path (atomic replace via temp file).
  5. Verifies:  for every row, the argmax of norm_scores must equal acc_norm_score
     (i.e., our computed pred_norm must match the harness's stored acc_norm_score).
  6. Verifies:  summary.json core6 numbers are unchanged (we read them before and after).

Usage:
  python enrich_per_example_normscores.py <results_dir> [<results_dir2> ...]

The script is idempotent: if `norm_lens` already exists in the first row of a file,
that file is skipped.

DOES NOT touch summary.json or shard*.json in any way.
"""

import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path

CORE6 = ["hellaswag", "arc_challenge", "arc_easy", "piqa", "winogrande", "openbookqa"]
_LETTERS = "ABCDEFGHIJKLMNOP"


def _safe_lp(x):
    return round(float(x), 6) if (x is not None and math.isfinite(float(x))) else None


# ---------------------------------------------------------------------------
# Inline copy of load_task_examples from eval_olmo2_probe2_downstream.py
# (exact same logic -- DO NOT change without also changing the harness)
# ---------------------------------------------------------------------------
def _hs_preprocess(text: str) -> str:
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def load_task_examples(task: str):
    from datasets import load_dataset

    if task == "hellaswag":
        d = load_dataset("Rowan/hellaswag", split="validation")
        out = []
        for ex in d:
            ctx = ex["ctx_a"] + " " + ex["ctx_b"].capitalize()
            query = _hs_preprocess(ex["activity_label"] + ": " + ctx)
            choices = [_hs_preprocess(e) for e in ex["endings"]]
            out.append({
                "gold": int(ex["label"]),
                "cands": [(query, " " + c, len(c)) for c in choices],
            })
        return out

    if task in ("arc_challenge", "arc_easy"):
        cfg = "ARC-Challenge" if task == "arc_challenge" else "ARC-Easy"
        d = load_dataset("allenai/ai2_arc", cfg, split="test")
        out = []
        for ex in d:
            q = "Question: " + ex["question"] + "\nAnswer:"
            texts = ex["choices"]["text"]
            labels = ex["choices"]["label"]
            ans = ex["answerKey"]
            if ans not in labels:
                continue
            out.append({
                "gold": labels.index(ans),
                "cands": [(q, " " + t, len(t)) for t in texts],
            })
        return out

    if task == "openbookqa":
        d = load_dataset("allenai/openbookqa", "main", split="test")
        out = []
        for ex in d:
            q = ex["question_stem"]
            texts = ex["choices"]["text"]
            labels = ex["choices"]["label"]
            ans = ex["answerKey"]
            if ans not in labels:
                continue
            out.append({
                "gold": labels.index(ans),
                "cands": [(q, " " + t, len(t)) for t in texts],
            })
        return out

    if task == "piqa":
        d = load_dataset("ybisk/piqa", revision="refs/convert/parquet",
                         split="validation")
        out = []
        for ex in d:
            q = "Question: " + ex["goal"] + "\nAnswer:"
            sols = [ex["sol1"], ex["sol2"]]
            out.append({
                "gold": int(ex["label"]),
                "cands": [(q, " " + s, len(s)) for s in sols],
            })
        return out

    if task == "winogrande":
        d = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
        answer_to_idx = {"1": 0, "2": 1}
        out = []
        for ex in d:
            s = ex["sentence"]
            idx = s.index("_")
            target = s[idx + 1:].strip()
            prefix = s[:idx]
            opts = [ex["option1"], ex["option2"]]
            out.append({
                "gold": answer_to_idx[ex["answer"]],
                "cands": [(prefix + o, " " + target, len(target)) for o in opts],
            })
        return out

    raise ValueError(f"unknown task: {task}")


# ---------------------------------------------------------------------------
# Build a lookup: item_id -> [norm_len_for_cand0, norm_len_for_cand1, ...]
# item_id = shard_index + ei * num_shards  (harness convention, num_shards=8)
# For a merged (non-sharded) file the item_ids run 0..N-1.
# ---------------------------------------------------------------------------
def build_norm_lens_lookup(task: str, num_shards: int = 8):
    """Returns dict: item_id -> list[int]  (one int per candidate option)."""
    examples = load_task_examples(task)
    lookup = {}
    # The harness builds item_id = shard_index + ei * num_shards
    # ei = index within the shard = position in examples_all[shard_index::num_shards]
    for shard_index in range(num_shards):
        shard = examples[shard_index::num_shards]
        for ei, ex in enumerate(shard):
            item_id = shard_index + ei * num_shards
            lookup[item_id] = [c[2] for c in ex["cands"]]
    return lookup


def enrich_file(jsonl_path: str, norm_lens_lookup: dict, task: str) -> dict:
    """
    Read the file, add norm_lens/norm_scores to every row, write back atomically.
    Returns a stats dict.
    """
    path = Path(jsonl_path)
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))

    if not rows:
        return {"skipped": True, "reason": "empty file"}

    # Idempotency check
    if "norm_lens" in rows[0]:
        return {"skipped": True, "reason": "already has norm_lens"}

    n_ok = 0
    n_nan = 0
    n_mismatch = 0

    enriched = []
    for row in rows:
        iid = row["item_id"]
        nl_list = norm_lens_lookup.get(iid)
        if nl_list is None:
            # item_id not in lookup -- fallback: derive from option_scores
            nl_list = [1] * len(row["option_scores"])
        letters = sorted(row["option_scores"].keys())
        nl_dict = {letters[k]: nl_list[k] for k in range(min(len(letters), len(nl_list)))}
        if row.get("nan"):
            ns_dict = {l: None for l in letters}
            n_nan += 1
        else:
            ns_dict = {}
            for k, l in enumerate(letters):
                raw = row["option_scores"].get(l)
                nlen = nl_list[k] if k < len(nl_list) else 1
                if raw is None:
                    ns_dict[l] = None
                else:
                    ns_dict[l] = _safe_lp(raw / max(nlen, 1))
            # Verify: pred_norm from norm_scores must match acc_norm_score
            if not row.get("nan"):
                best_ns = max(ns_dict.items(), key=lambda kv: (kv[1] if kv[1] is not None else -1e9))
                pred_norm_letter = best_ns[0]
                pred_norm_gold = (pred_norm_letter == row["gold_letter"])
                stored = row.get("acc_norm_score", 0.0)
                if abs((1.0 if pred_norm_gold else 0.0) - stored) > 0.01:
                    n_mismatch += 1
            n_ok += 1
        new_row = dict(row)
        new_row["norm_lens"] = nl_dict
        new_row["norm_scores"] = ns_dict
        enriched.append(new_row)

    # Atomic write
    tmp = path.with_suffix(".tmp_enrich")
    with open(tmp, "w") as f:
        for r in enriched:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)

    return {
        "skipped": False,
        "n_rows": len(enriched),
        "n_ok": n_ok,
        "n_nan": n_nan,
        "n_mismatch": n_mismatch,
    }


def enrich_dir(results_dir: str):
    print(f"\n=== Enriching: {results_dir} ===")
    results_dir = os.path.abspath(results_dir)

    # Read summary before
    summary_path = os.path.join(results_dir, "summary.json")
    summary_before = None
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary_before = json.load(f)

    for task in CORE6:
        fpath = os.path.join(results_dir, f"per_example_{task}.jsonl")
        if not os.path.exists(fpath):
            print(f"  {task}: MISSING per_example file -- skipping")
            continue
        print(f"  {task}: loading norm_lens lookup...", end=" ", flush=True)
        nl_lookup = build_norm_lens_lookup(task)
        print(f"OK ({len(nl_lookup)} items). Enriching...", end=" ", flush=True)
        stats = enrich_file(fpath, nl_lookup, task)
        if stats.get("skipped"):
            print(f"SKIPPED ({stats['reason']})")
        else:
            mismatch_warn = f"  !! {stats['n_mismatch']} MISMATCHES" if stats['n_mismatch'] else ""
            print(f"done: {stats['n_rows']} rows, {stats['n_nan']} nan{mismatch_warn}")

    # Verify summary unchanged
    if summary_before is not None and os.path.exists(summary_path):
        with open(summary_path) as f:
            summary_after = json.load(f)
        tasks_b = summary_before.get("tasks", {})
        tasks_a = summary_after.get("tasks", {})
        changed = []
        for t in CORE6:
            vb = tasks_b.get(t, {}).get("acc_norm")
            va = tasks_a.get(t, {}).get("acc_norm")
            if vb != va:
                changed.append(f"{t}: {vb} -> {va}")
        if changed:
            print(f"  !! SUMMARY CHANGED: {changed}")
        else:
            vals = [tasks_a.get(t, {}).get("acc_norm", 0) for t in CORE6]
            core6 = sum(v for v in vals if v) / sum(1 for v in vals if v) if any(vals) else 0
            print(f"  summary.json UNCHANGED (core6={core6:.5f})")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: enrich_per_example_normscores.py <results_dir> [...]")
        sys.exit(1)
    for d in sys.argv[1:]:
        enrich_dir(d)
    print("All done.")
