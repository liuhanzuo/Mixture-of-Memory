#!/usr/bin/env python3
"""Read the fp32 norm of the gist scorer projections from saved ckpt(s).

The in-training [gist_traj] probe reads bf16 params whose norm quantizes near
14.5 (can't resolve a ~0.02-0.03 drift). Saved checkpoints are fp32 (FSDP
gathers + upcasts), so the gist proj norm trajectory is read accurately HERE.

Usage:
  python scripts/gist_norm_from_ckpt.py ckptA.pt ckptB.pt ...
Prints, per ckpt, the fp32 norm of gist query_proj / key_proj + drift from the
fresh-init reference (std=0.02, [128,4096] -> ~14.4702).
"""
from __future__ import annotations

import sys
import torch
import torch.nn as nn

INIT_REF = None  # computed below


def _init_ref(gist_dim: int, d_model: int) -> float:
    torch.manual_seed(0)
    w = nn.Linear(d_model, gist_dim, bias=False).weight
    nn.init.normal_(w, std=0.02)
    return float(w.detach().norm())


def main():
    ckpts = sys.argv[1:]
    if not ckpts:
        print("usage: gist_norm_from_ckpt.py <ckpt.pt> [...]")
        sys.exit(1)
    for path in ckpts:
        try:
            sd = torch.load(path, map_location="cpu")
        except Exception as e:  # noqa: BLE001
            print(f"{path}: LOAD ERROR {e}")
            continue
        if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        rows = {}
        for k, v in sd.items():
            if "gist_readout" in k and "proj" in k and "weight" in k:
                # strip any wrapper prefixes
                nk = k
                for m in ("_fsdp_wrapped_module.", "module.", "model.", "_gist_readout.", "gist_readout."):
                    nk = nk.replace(m, "")
                if not isinstance(v, torch.Tensor):
                    continue
                w = v.float()
                if nk not in rows:  # dedup (model.* and _gist_readout.* alias)
                    rows[nk] = (float(w.norm()), tuple(w.shape))
        if not rows:
            print(f"{path}: no gist proj keys found")
            continue
        for nk, (norm, shape) in rows.items():
            gist_dim, d_model = shape
            ref = _init_ref(gist_dim, d_model)
            print(f"{path} | {nk} fp32_norm={norm:.6f} init_ref={ref:.6f} "
                  f"drift={norm - ref:+.6f} shape={shape}")


if __name__ == "__main__":
    main()
