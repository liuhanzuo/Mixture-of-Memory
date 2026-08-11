#!/usr/bin/env python3
"""Measure how far the SparseForge checkpoint's continuous mask is from binary.

The checkpoint stores DENSE weights plus a CONTINUOUS mask, and the training
forward (``sparse_modeling.py:788``) multiplies the weight by that *soft* mask
directly whenever ``hardening_x >= 1`` or the layer is flagged finalized.  So
the model that produced the checkpoint's own ``lm_eval_results`` used
``W * mask_soft``, not ``W * hard_2of4(mask)``.

Exporting a hard 2:4 model is only a faithful representation of that model if
the soft mask is numerically binary, i.e. kept entries are ~1 and pruned
entries are ~0.  This tool quantifies exactly that, per sampled layer:

  * the mask-value distribution at the top-2-per-group ("kept") positions and at
    the bottom-2 ("pruned") positions,
  * the relative Frobenius error ``||W*soft - W*hard|| / ||W*hard||``, which is
    the actual perturbation that hardening introduces into the forward pass.

If that relative error is ~1e-7 the hard export is faithful; if it is ~1e-2 the
soft-masked model and the 2:4 model are materially different and both must be
reported separately.
"""

from __future__ import annotations

import argparse
import json

import torch

PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def nm_2_4_hard(soft: torch.Tensor) -> torch.Tensor:
    N, M = 2, 4
    out_dim, in_dim = soft.shape
    in_full = (in_dim // M) * M
    hard = torch.ones_like(soft, dtype=soft.dtype)
    if in_full == 0:
        return hard
    grouped = soft.detach().float()[:, :in_full].view(out_dim, in_full // M, M)
    topi = torch.topk(grouped, k=N, dim=-1, largest=True).indices
    gm = torch.zeros_like(grouped, dtype=soft.dtype)
    gm.scatter_(-1, topi, 1.0)
    hard[:, :in_full] = gm.view(out_dim, in_full)
    return hard


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sample", type=int, default=8)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False, mmap=True)
    sd = blob["model_state_dict"]
    scope = [k for k in sd
             if k.endswith(".weight") and any(f".{p}." in k for p in PROJECTIONS)]

    per_layer = []
    worst_rel = 0.0
    for wk in scope[: args.sample]:
        mk = wk[: -len("weight")] + "mask"
        W = sd[wk].float()
        m = sd[mk].float()
        h = nm_2_4_hard(m)
        kept = h == 1
        pruned = h == 0

        mk_vals = m[kept]
        mp_vals = m[pruned]
        Wh = W * h
        Ws = W * m
        rel = float((Ws - Wh).norm() / Wh.norm())
        worst_rel = max(worst_rel, rel)

        e = {
            "layer": wk,
            "kept_mask_min": float(mk_vals.min()),
            "kept_mask_max": float(mk_vals.max()),
            "kept_mask_mean": float(mk_vals.mean()),
            "kept_mask_frac_below_0.99": float((mk_vals < 0.99).float().mean()),
            "kept_mask_frac_below_0.5": float((mk_vals < 0.5).float().mean()),
            "pruned_mask_min": float(mp_vals.min()),
            "pruned_mask_max": float(mp_vals.max()),
            "pruned_mask_mean": float(mp_vals.mean()),
            "pruned_mask_frac_above_0.01": float((mp_vals > 0.01).float().mean()),
            "pruned_mask_frac_above_0.5": float((mp_vals > 0.5).float().mean()),
            "rel_fro_soft_vs_hard": rel,
        }
        per_layer.append(e)
        print(f"[maskdist] {wk}")
        print(f"[maskdist]   kept  : min={e['kept_mask_min']:.6f} max={e['kept_mask_max']:.6f} "
              f"mean={e['kept_mask_mean']:.6f} frac<0.99={e['kept_mask_frac_below_0.99']:.3e} "
              f"frac<0.5={e['kept_mask_frac_below_0.5']:.3e}")
        print(f"[maskdist]   pruned: min={e['pruned_mask_min']:.3e} max={e['pruned_mask_max']:.6f} "
              f"mean={e['pruned_mask_mean']:.3e} frac>0.01={e['pruned_mask_frac_above_0.01']:.3e} "
              f"frac>0.5={e['pruned_mask_frac_above_0.5']:.3e}")
        print(f"[maskdist]   ||W*soft - W*hard|| / ||W*hard|| = {rel:.6e}")

    summary = {
        "ckpt": args.ckpt,
        "iter_num": blob.get("iter_num"),
        "sampled": len(per_layer),
        "worst_rel_fro_soft_vs_hard": worst_rel,
        "per_layer": per_layer,
        "interpretation": (
            "worst_rel_fro < 1e-4 => hardening the mask is numerically a no-op, so a hard "
            "2:4 export faithfully represents the soft-masked model that was evaluated during "
            "training. Larger values mean the two are different models."
        ),
    }
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[maskdist] wrote {args.out_json}")
    print(f"[maskdist] worst relative Frobenius error over {len(per_layer)} layers = {worst_rel:.6e}")
    print("[maskdist] VERDICT: hardening is "
          + ("NUMERICALLY NEUTRAL (<1e-4)" if worst_rel < 1e-4
             else f"MATERIAL ({worst_rel:.3e}) -- soft and hard are different models"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
