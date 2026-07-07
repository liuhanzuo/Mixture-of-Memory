#!/usr/bin/env python3
"""Sweep-aware probe driver for the semantic-bottleneck (layer j, dim) ablation.

For each (j,dim) arm checkpoint it reports, in one aggregated JSON, the three
quantities the paper §3.3 ablation needs:

  A. DIVISION OF LABOUR  (from probe_semantic_bottleneck.layerwise_curves):
       - per-hidden-state next-token top-1 acc curve (logit lens)
       - gen_acc of the first j hidden states  (should be ~0 for a clean funnel)
       - j*95 / j*99 = layer where generation reaches 95/99% of top acc
       - post-bottleneck acc jump (acc[j+2]-acc[j+1])
  B. LM COST:
       - top-layer next-token acc  (vs baseline; how much the funnel costs)
  C. QCMem-FRIENDLINESS (from probe_bottleneck_qcmem_friendly.run_arm) at the
     arm's OWN cache layer j:
       - PCA dim99 / eff_rank / int4 relerr of the cached h_j
       - readout acc DROP under PCA rank-{512,256,128,64} + int4 compression

The baseline arm (dim=0) is probed at every j that appears in the sweep so each
bottleneck arm has a same-j baseline to compare against (the funnel's effect is
local to its own layer, so a j=4 bottleneck must be compared to baseline@j=4).

Honesty: weak 1B/2000-step models -> read the RELATIVE trend across arms.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from probe_semantic_bottleneck import load_ckpt as load_ckpt_gen, layerwise_curves, jstar  # noqa: E402
from probe_bottleneck_qcmem_friendly import run_arm as qcmem_run_arm  # noqa: E402


def division_of_labour(ckpt, tokens, device, batch_size):
    model, bl, bd = load_ckpt_gen(ckpt, device)
    acc, nll = layerwise_curves(model, tokens, device, batch_size)
    j = bl
    # hidden_states: 0=embed, k+1 = output of layer k. The first j layers' outputs
    # are indices 1..j; embeddings idx 0. "gen acc of first j layers" = acc[1..j].
    pre_j_acc = [round(float(x), 4) for x in acc[1:j + 1]]
    jump = float(acc[j + 2] - acc[j + 1]) if (j + 2) < len(acc) else float("nan")
    res = {
        "bottleneck_layer": int(bl), "bottleneck_dim": int(bd),
        "gen_acc_curve": [round(float(x), 4) for x in acc],
        "nll_curve": [round(float(x), 4) for x in nll],
        "pre_j_gen_acc": pre_j_acc,
        "pre_j_gen_acc_max": round(max(pre_j_acc) if pre_j_acc else 0.0, 4),
        "jstar95": int(jstar(acc, 0.95)), "jstar99": int(jstar(acc, 0.99)),
        "top_acc": round(float(acc[-1]), 4),
        "post_bottleneck_jump": round(jump, 4),
    }
    del model
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_ckpt", default="outputs/sembott_1b_baseline/final.pt")
    # arms: "name:ckpt:j:dim" tuples
    ap.add_argument("--arms", nargs="+", required=True,
                    help="each = name:ckpt_path:j:dim")
    ap.add_argument("--val_path", default="data/slimpajama_val_4096_llama3.npy")
    ap.add_argument("--n_examples", type=int, default=96)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--ranks", type=int, nargs="+", default=[512, 256, 128, 64])
    ap.add_argument("--out_json", default="outputs/sembott_sweep_result.json")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = args.device
    arr = np.load(args.val_path, mmap_mode="r")
    tokens = torch.from_numpy(
        np.asarray(arr[: args.n_examples, : args.seq_len]).astype(np.int64))
    print(f"probe on {tokens.shape[0]} x {tokens.shape[1]} tokens")

    parsed = []
    for a in args.arms:
        # name may itself contain no colon; split from the right for j/dim/ckpt
        name, ckpt, j, dim = a.rsplit(":", 3)
        parsed.append((name, ckpt, int(j), int(dim)))

    js_needed = sorted({j for (_, _, j, _) in parsed})
    result = {"config": {"n_examples": args.n_examples, "seq_len": args.seq_len,
                         "ranks": args.ranks, "val_path": args.val_path,
                         "js_probed": js_needed},
              "arms": {}}

    # -------- baseline: division-of-labour once + QCMem-friendliness at each needed j
    print("\n########## BASELINE (dim=0) ##########")
    result["baseline"] = {
        "division_of_labour": division_of_labour(args.baseline_ckpt, tokens, device, args.batch_size),
        "qcmem_by_j": {},
    }
    for j in js_needed:
        r = qcmem_run_arm(f"baseline@j{j}", args.baseline_ckpt, tokens, j,
                          device, args.batch_size, args.ranks)
        result["baseline"]["qcmem_by_j"][str(j)] = r

    # -------- each bottleneck arm
    for (name, ckpt, j, dim) in parsed:
        print(f"\n########## ARM {name} (j={j}, dim={dim}) ##########")
        dol = division_of_labour(ckpt, tokens, device, args.batch_size)
        qc = qcmem_run_arm(name, ckpt, tokens, j, device, args.batch_size, args.ranks)
        result["arms"][name] = {"j": j, "dim": dim,
                                "division_of_labour": dol, "qcmem": qc}

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nsaved {args.out_json}")

    # ------------------------- SWEEP TABLE -------------------------
    print("\n" + "=" * 100)
    print("SWEEP TABLE  (1B / 2000-step from-scratch; read RELATIVE trend)")
    print("=" * 100)
    bdol = result["baseline"]["division_of_labour"]
    print(f"\nBASELINE(dim=0): top_acc={bdol['top_acc']}  j*95={bdol['jstar95']} j*99={bdol['jstar99']}")
    hdr = (f"{'arm':<14}{'j':>3}{'dim':>6}{'pre_j_acc_max':>14}{'jstar95':>9}"
           f"{'top_acc':>9}{'dLM_vs_base':>12}{'dim99':>7}{'base_dim99@j':>13}"
           f"{'drop@d':>9}{'base_drop@d':>13}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for name, a in result["arms"].items():
        dol = a["division_of_labour"]
        qc = a["qcmem"]
        j, dim = a["j"], a["dim"]
        base_qc = result["baseline"]["qcmem_by_j"][str(j)]
        dim99 = qc["compressibility"]["dim_99"]
        base_dim99 = base_qc["compressibility"]["dim_99"]
        # readout drop at the arm's own funnel rank (nearest available rank key <= dim)
        rk = f"pca_rank{dim}" if f"pca_rank{dim}" in qc["readout"] else None
        if rk is None:
            # pick the largest rank <= dim present
            avail = [r for r in args.ranks if r <= dim and f"pca_rank{r}" in qc["readout"]]
            rk = f"pca_rank{max(avail)}" if avail else None
        drop = qc["readout"][rk]["acc_drop"] if rk else float("nan")
        bdrop = base_qc["readout"][rk]["acc_drop"] if (rk and rk in base_qc["readout"]) else float("nan")
        dlm = round(dol["top_acc"] - bdol["top_acc"], 4)
        print(f"{name:<14}{j:>3}{dim:>6}{dol['pre_j_gen_acc_max']:>14}{dol['jstar95']:>9}"
              f"{dol['top_acc']:>9}{dlm:>12}{dim99:>7}{base_dim99:>13}"
              f"{drop:>9}{bdrop:>13}  (rank={rk})")
    print("\nlegend: pre_j_acc_max = max next-tok acc among first j hidden states "
          "(≈0 = clean funnel, back half does generation);")
    print("  dLM_vs_base = top_acc(arm) - top_acc(baseline) (funnel LM cost, <0 = worse);")
    print("  dim99 = PCA dims for 99% var of cached h_j (lower=more compressible); "
          "base_dim99@j = same-j baseline;")
    print("  drop@d = readout acc drop when h_j compressed to PCA rank≈dim; "
          "base_drop@d = baseline at same rank.")


if __name__ == "__main__":
    main()
