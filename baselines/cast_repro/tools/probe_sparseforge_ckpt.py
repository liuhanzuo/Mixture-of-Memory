#!/usr/bin/env python3
"""Probe the SparseForge headline checkpoint's structure before exporting it.

The SparseForge trainer (``sparse_modeling.py``) stores a *dense* weight plus a
*continuous* soft mask per in-scope projection, so a naive
``threshold(mask, 0.5)`` hardening does NOT reproduce the model that was
evaluated during training. The correct projection is the ``nm_2_4`` branch of
``sparse_modeling.py`` (exact per-group top-2 via ``topk(2)+scatter_``).

This tool prints the state-dict layout and, for a sample of in-scope
projections, compares the two hardenings so the discrepancy is on the record.
"""

from __future__ import annotations

import argparse
import collections
import json
import re

import torch

PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def nm_2_4_hard(soft: torch.Tensor) -> torch.Tensor:
    """Exact port of the ``nm_2_4`` branch in sparse_modeling.py:594-621."""
    N, M = 2, 4
    out_dim, in_dim = soft.shape
    in_full = (in_dim // M) * M
    hard = torch.ones_like(soft, dtype=soft.dtype)
    if in_full == 0:
        return hard
    core = soft.detach().float()[:, :in_full]
    groups = in_full // M
    grouped = core.view(out_dim, groups, M)
    topi = torch.topk(grouped, k=N, dim=-1, largest=True).indices
    group_mask = torch.zeros_like(grouped, dtype=soft.dtype)
    group_mask.scatter_(-1, topi, 1.0)
    hard[:, :in_full] = group_mask.view(out_dim, in_full)
    return hard


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sample", type=int, default=6)
    ap.add_argument("--full-scan", action="store_true",
                    help="scan every in-scope tensor (slow, ~41GB read) and report global tile stats")
    args = ap.parse_args()

    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False, mmap=True)
    print(f"[probe] top-level keys: {list(blob.keys())}")
    for k, v in blob.items():
        if k == "model_state_dict":
            continue
        print(f"[probe]   {k} = {repr(v)[:300]}")

    sd = blob["model_state_dict"]
    print(f"[probe] model_state_dict: {len(sd)} tensors")

    pat = collections.Counter()
    for k in sd:
        pat[re.sub(r"\.\d+\.", ".N.", k)] += 1
    print("[probe] key patterns:")
    for k, c in sorted(pat.items()):
        print(f"[probe]   {c:5d}  {k}")

    # In-scope weights and their masks.
    scope = [k for k in sd
             if k.endswith(".weight") and any(f".{p}." in k for p in PROJECTIONS)]
    masks = [k for k in sd if k.endswith(".mask")]
    scales = [k for k in sd if k.endswith(".cast_scale") or k.endswith("_scale")]
    print(f"[probe] in-scope weights: {len(scope)}  masks: {len(masks)}  scale-like: {len(scales)}")

    targets = scope if args.full_scan else scope[: args.sample]
    tot_elems = tot_tiles = 0
    thr_bad = nm_bad = 0
    nm_kept = thr_kept = 0
    w_zero = 0
    mask_min, mask_max, mask_sum, mask_n = 1e30, -1e30, 0.0, 0
    for wk in targets:
        mk = wk[: -len("weight")] + "mask"
        w = sd[wk]
        if mk not in sd:
            print(f"[probe]   {wk}: NO MASK SIBLING")
            continue
        m = sd[mk].float()
        wf = w.float()
        r, c = wf.shape
        tot_elems += wf.numel()
        w_zero += int((wf == 0).sum())
        mask_min = min(mask_min, float(m.min()))
        mask_max = max(mask_max, float(m.max()))
        mask_sum += float(m.sum())
        mask_n += m.numel()

        thr = (m > 0.5).float()
        nm = nm_2_4_hard(m)
        tiles = (r * c) // 4
        tot_tiles += tiles
        thr_nz = thr.reshape(r, c // 4, 4).sum(-1)
        nm_nz = nm.reshape(r, c // 4, 4).sum(-1)
        thr_bad += int((thr_nz != 2).sum())
        nm_bad += int((nm_nz != 2).sum())
        thr_kept += int(thr.sum())
        nm_kept += int(nm.sum())
        if not args.full_scan:
            print(f"[probe]   {wk} shape={r}x{c} w_zero_frac={(wf==0).float().mean():.9f} "
                  f"mask[min={m.min():.3e},max={m.max():.3e},mean={m.mean():.6f}] "
                  f"thr0.5_bad_tiles={int((thr_nz!=2).sum())} nm24_bad_tiles={int((nm_nz!=2).sum())}")

    summary = {
        "ckpt": args.ckpt,
        "iter_num": blob.get("iter_num"),
        "finalization_done": blob.get("finalization_done"),
        "n_state_tensors": len(sd),
        "n_scope_weights": len(scope),
        "n_masks": len(masks),
        "scanned_tensors": len(targets),
        "scanned_elems": tot_elems,
        "weight_zero_frac": w_zero / tot_elems if tot_elems else None,
        "mask_min": mask_min if mask_n else None,
        "mask_max": mask_max if mask_n else None,
        "mask_mean": mask_sum / mask_n if mask_n else None,
        "tiles": tot_tiles,
        "threshold05_bad_tiles": thr_bad,
        "threshold05_kept_frac": thr_kept / tot_elems if tot_elems else None,
        "nm24_bad_tiles": nm_bad,
        "nm24_kept_frac": nm_kept / tot_elems if tot_elems else None,
    }
    print("[probe] SUMMARY " + json.dumps(summary, indent=2))
    verdict = (nm_bad == 0 and abs(summary["nm24_kept_frac"] - 0.5) < 1e-9)
    print(f"[probe] nm_2_4 projection is EXACT 2:4: {'PASS' if verdict else 'FAIL'}")
    return 0 if verdict else 2


if __name__ == "__main__":
    raise SystemExit(main())
