#!/usr/bin/env python3
"""A04 margin guard evidence: measured paired discordance rate p_disc on
MMLU-content at 1B for arms Pilot Zero did NOT score.

WHY: the guard's D6 condition asks whether Delta_x is smaller than the
achievable item-level 95% CI half-width, `hw = z2 * sqrt(p_disc/n)`. Pilot Zero
only scored well-healed arms (>= 200,000 steps), whose discordance against the
intact anchor is comparatively LOW. The gate's frozen checkpoint grid starts at
2,500 steps, where the arm is barely healed and p_disc is much HIGHER -- which
makes the CI wider and D6 MORE likely to fire. Omitting the barely-healed arm
would understate the range and produce optimistic CERTIFIABLE verdicts.

The `A03_1B_keep7_step500` arm (500 steps, the "barely healed" control in
A03's own 4-axes table) is the closest thing on disk to the gate's first
checkpoint and is used as the honest worst case.

Scorers IMPORTED from A03's canonical `analyze_1b_knowledge_floor.py`.
CPU ONLY. No GPU, no model load, no torch. Read-only on all inputs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_A03_CODE = os.path.abspath(os.path.join(
    _HERE, "..", "..", "A03-parametric-vs-external-memory", "code"))
if _A03_CODE not in sys.path:
    sys.path.insert(0, _A03_CODE)
from analyze_1b_knowledge_floor import SEED, paired_bootstrap  # noqa: E402

EXPECTED_N = 14042


def load_mmlu(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r["item_id"])
    assert len(rows) == EXPECTED_N, f"{path}: n={len(rows)} != {EXPECTED_N}"
    assert not any(r.get("nan") for r in rows), f"{path}: nan rows present"
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--intact", required=True,
                    help="A03_1B_base/per_example_mmlu.jsonl")
    ap.add_argument("--arm", action="append", default=[], metavar="LABEL=PATH",
                    help="repeatable, e.g. keep7_step500=/path/per_example_mmlu.jsonl")
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    base = load_mmlu(args.intact)
    B = np.array([1.0 if r["content_norm"]["correct"] else 0.0 for r in base])
    ids = [r["item_id"] for r in base]

    out = {
        "what": ("measured paired discordance p_disc vs the intact anchor on "
                 "MMLU-content at 1B; the input to the guard's D6 condition"),
        "date": "2026-08-10",
        "gpu_spent": 0,
        "intact_source": args.intact,
        "intact_content_norm": float(B.mean()),
        "n": len(base),
        "note": ("p_disc = fraction of items where the arm and the intact "
                 "anchor disagree. hw_95 = 1.96*sqrt(p_disc/n). A barely "
                 "healed arm has HIGHER p_disc and therefore a WIDER CI, "
                 "which is why the gate's early checkpoints are the worst "
                 "case for D6."),
        "arms": {},
    }
    for spec in args.arm:
        lab, path = spec.split("=", 1)
        rows = load_mmlu(path)
        assert [r["item_id"] for r in rows] == ids, \
            f"{lab}: item_id misaligned with the intact anchor"
        assert [r["gold_letter"] for r in rows] == \
            [r["gold_letter"] for r in base], f"{lab}: gold misaligned"
        A = np.array([1.0 if r["content_norm"]["correct"] else 0.0
                      for r in rows])
        d = A - B
        p_disc = float((d != 0).mean())
        mean, lo, hi, p = paired_bootstrap(d, seed=SEED + 300)
        out["arms"][lab] = {
            "source": path,
            "acc": float(A.mean()),
            "p_disc": p_disc,
            "diff_pp": 100 * mean,
            "ci95_pp": [100 * lo, 100 * hi],
            "halfwidth_pp": 100 * (hi - lo) / 2,
            "boot_p": p,
        }
        print(f"{lab}: acc={A.mean():.6f} p_disc={p_disc:.6f} "
              f"diff={100*mean:+.4f}pp hw={100*(hi-lo)/2:.4f}pp")

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    # the classifier consumes a flat {label: {p_disc: ...}} mapping
    json.dump(out["arms"], open(args.out_json, "w"), indent=1)
    json.dump(out, open(args.out_json.replace(".json", "_full.json"), "w"),
              indent=1)
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
