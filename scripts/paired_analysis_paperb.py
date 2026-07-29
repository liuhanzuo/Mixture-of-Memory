#!/usr/bin/env python3
"""Paired statistical analysis: keep14@200k vs fully-random-init@200k (Paper B).
Per-task McNemar test + paired bootstrap 95% CI on the accuracy difference.
Uses per-example predictions (item_id-paired) from --save_per_example eval runs.
CPU only. Output: results/paperb_paired_analysis.json + status/PAPERB_PAIRED_STATS.md
"""
import json, os
import numpy as np
from scipy.stats import binomtest

ROOT = "/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory"
KEEP_DIR = f"{ROOT}/olmo2_downstream_results/7B_keep14_step200000_perex_know"
RAND_DIR = f"{ROOT}/olmo2_downstream_results/7B_fromscratch_step200000_perex_know"
TASKS = ["mmlu", "lambada_openai", "boolq", "commonsense_qa", "social_iqa"]
N_BOOT = 10000
SEED = 0

def load_items(path):
    items = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            items[d["item_id"]] = d
    return items

def analyze_task(task):
    kp = f"{KEEP_DIR}/per_example_{task}.jsonl"
    rp = f"{RAND_DIR}/per_example_{task}.jsonl"
    if not (os.path.exists(kp) and os.path.exists(rp)):
        return None
    k = load_items(kp)
    r = load_items(rp)
    common = sorted(set(k) & set(r))
    # exclude NaN items (either side)
    paired = [(k[i], r[i]) for i in common if not k[i].get("nan", False) and not r[i].get("nan", False)]
    n = len(paired)
    kc = np.array([int(a["correct"]) for a, b in paired])
    rc = np.array([int(b["correct"]) for a, b in paired])
    # McNemar: b=keep对random错, c=keep错random对
    b = int(np.sum((kc == 1) & (rc == 0)))
    c = int(np.sum((kc == 0) & (rc == 1)))
    p_mcnemar = float(binomtest(min(b, c), b + c, 0.5).pvalue) if (b + c) > 0 else 1.0
    # paired bootstrap on per-item diff
    diffs = kc - rc
    rng = np.random.default_rng(SEED)
    if n > 0:
        idx = rng.integers(0, n, size=(N_BOOT, n))
        bs_means = diffs[idx].mean(axis=1)
        ci_lo, ci_hi = float(np.percentile(bs_means, 2.5)), float(np.percentile(bs_means, 97.5))
    else:
        ci_lo = ci_hi = 0.0
    return {
        "task": task,
        "n_paired": n,
        "keep14_acc": float(kc.mean()),
        "random_acc": float(rc.mean()),
        "diff": float(kc.mean() - rc.mean()),
        "mcnemar_b": b, "mcnemar_c": c,
        "mcnemar_p": p_mcnemar,
        "bootstrap_ci95": [ci_lo, ci_hi],
    }

results = {}
for t in TASKS:
    r = analyze_task(t)
    if r:
        results[t] = r

# write JSON
os.makedirs(f"{ROOT}/results", exist_ok=True)
with open(f"{ROOT}/results/paperb_paired_analysis.json", "w") as f:
    json.dump(results, f, indent=2)

# write markdown
lines = ["# Paper B — Paired analysis: keep14@200k vs fully-random-init@200k", "",
         "Per-task McNemar test + paired bootstrap 95% CI on accuracy difference.",
         "Item-id paired, NaN excluded. CPU only.", "",
         "| task | n | keep14 acc | random acc | diff (pp) | McNemar b/c | McNemar p | bootstrap 95% CI (pp) |",
         "|---|---:|---:|---:|---:|---|---:|---|"]
for t in TASKS:
    r = results.get(t)
    if not r:
        continue
    lines.append(f"| {t} | {r['n_paired']} | {r['keep14_acc']:.4f} | {r['random_acc']:.4f} | "
                 f"{r['diff']*100:.2f} | {r['mcnemar_b']}/{r['mcnemar_c']} | {r['mcnemar_p']:.2e} | "
                 f"[{r['bootstrap_ci95'][0]*100:.2f}, {r['bootstrap_ci95'][1]*100:.2f}] |")
lines += ["", "## Interpretation",
          "- diff > 0 means keep14 (inherited+train-all) beats fully-random-init.",
          "- McNemar p < 0.05: significant discordant pairs (one arm right where other wrong).",
          "- bootstrap CI excludes 0: significant accuracy difference.",
          "- freeze-front has no per-example predictions, so is NOT in this paired analysis."]
with open(f"{ROOT}/status/PAPERB_PAIRED_STATS.md", "w") as f:
    f.write("\n".join(lines) + "\n")

print("=== Paper B paired analysis (keep14 vs fully-random-init @200k) ===")
for t in TASKS:
    r = results.get(t)
    if r:
        print(f"{t}: n={r['n_paired']} keep14={r['keep14_acc']:.4f} random={r['random_acc']:.4f} "
              f"diff={r['diff']*100:+.2f}pp McNemar_p={r['mcnemar_p']:.2e} "
              f"CI=[{r['bootstrap_ci95'][0]*100:.2f},{r['bootstrap_ci95'][1]*100:.2f}]pp")
print("\nWrote results/paperb_paired_analysis.json + status/PAPERB_PAIRED_STATS.md")
