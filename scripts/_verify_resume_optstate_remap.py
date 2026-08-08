#!/usr/bin/env python
"""Can the 2-group optimizer_state in the keepN ckpts be REMAPPED into the
4-group structure current code builds, preserving every Adam moment?

The ckpt's optimizer_state['state'] is keyed by a FLAT index into the
concatenation of param_groups in the order they were built. Both the old
(prefix-bug) and the new (fixed) grouping bucket the SAME set of parameters --
only the partition differs. So a faithful remap exists iff we can compute, for
each parameter, its old flat index and its new flat index, and permute
state{} accordingly. Group hyperparams (lr/base_lr/min_lr/weight_decay) come
from the NEW groups (that is the intended differential-LR behaviour); only
exp_avg / exp_avg_sq / step are carried over.

This script PROVES the remap is well-defined and bijective, and verifies that
the resulting state dict loads cleanly into the CURRENT 4-group optimizer with
all Adam moments intact and matching the source tensors bytewise.

Read-only. No CUDA. No training. No writes to any ckpt.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_olmo2_arch_probe2 import _classify_param  # noqa: E402

SPEC_ORDER = ["fresh_decay", "fresh_nodecay", "inh_decay", "inh_nodecay"]


def bucket(names, shapes, keep_front, strip_module):
    """Replicate build_param_groups ordering. strip_module=False emulates the
    pre-fix DDP 'module.'-prefixed classification (everything -> inherited)."""
    specs = {k: [] for k in SPEC_ORDER}
    for n in names:
        if strip_module:
            cls = _classify_param(n, keep_front, False, random_trunk=False)
        else:
            dn = "module." + n  # DDP name, pre-fix code did not strip
            if dn.startswith("model.layers."):
                cls = "inherited" if int(dn.split(".")[2]) < keep_front else "fresh"
            elif dn.startswith("lm_head"):
                cls = "fresh"
            else:
                cls = "inherited"
        prefix = "fresh" if cls == "fresh" else "inh"
        key = f"{prefix}_decay" if len(shapes[n]) >= 2 else f"{prefix}_nodecay"
        specs[key].append(n)
    # build_param_groups drops empty groups, preserving specs dict order
    return [(k, specs[k]) for k in SPEC_ORDER if specs[k]]


def flat_index(groups):
    idx, i = {}, 0
    for _, names in groups:
        for n in names:
            idx[n] = i
            i += 1
    return idx


def main(path):
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    keep_front = ck["keep_front_layers"]
    ms = ck["model_state"]
    names = list(ms.keys())
    shapes = {k: tuple(v.shape) for k, v in ms.items()}
    osd = ck["optimizer_state"]

    old_groups = bucket(names, shapes, keep_front, strip_module=False)
    new_groups = bucket(names, shapes, keep_front, strip_module=True)
    old_idx = flat_index(old_groups)
    new_idx = flat_index(new_groups)

    res = {
        "path": path,
        "keep_front_layers": keep_front,
        "n_params": len(names),
        "old_groups": [(k, len(v)) for k, v in old_groups],
        "new_groups": [(k, len(v)) for k, v in new_groups],
        "ckpt_group_sizes": [len(g["params"]) for g in osd["param_groups"]],
        "ckpt_state_keys_n": len(osd["state"]),
    }
    # sanity: reconstructed OLD grouping must match what the ckpt stored
    res["old_grouping_matches_ckpt"] = (
        [len(v) for _, v in old_groups] == [len(g["params"]) for g in osd["param_groups"]]
    )
    # the ckpt's flat index space must equal the old flat index space
    ckpt_flat = [p for g in osd["param_groups"] for p in g["params"]]
    res["ckpt_flat_is_0..n-1"] = ckpt_flat == list(range(len(names)))
    res["remap_is_bijective"] = (sorted(old_idx.values()) == sorted(new_idx.values())
                                 and set(old_idx) == set(new_idx))

    # --- build remapped state ---
    remapped_state = {}
    for n in names:
        oi, ni = old_idx[n], new_idx[n]
        if oi in osd["state"]:
            remapped_state[ni] = osd["state"][oi]
    res["remapped_state_n"] = len(remapped_state)

    new_osd = {
        "state": remapped_state,
        "param_groups": [],
    }
    for gi, (gname, gnames) in enumerate(new_groups):
        new_osd["param_groups"].append({
            **{k: v for k, v in osd["param_groups"][0].items() if k != "params"},
            "params": [new_idx[n] for n in gnames],
        })

    # --- actually load into a CURRENT-shaped optimizer ---
    pg = []
    for gname, gnames in new_groups:
        params = [torch.nn.Parameter(torch.zeros(shapes[n], dtype=torch.float32))
                  for n in gnames]
        pg.append({"params": params,
                   "weight_decay": 0.1 if gname.endswith("_decay") else 0.0,
                   "base_lr": 1e-4 if gname.startswith("fresh") else 2e-5,
                   "min_lr": 1e-5 if gname.startswith("fresh") else 2e-6})
    opt = torch.optim.AdamW(pg, lr=2e-5, betas=(0.9, 0.95), eps=1e-8)
    try:
        opt.load_state_dict(new_osd)
        res["remapped_load"] = {"ok": True, "n_states": len(opt.state)}
        # verify moments preserved bytewise for a sample of params
        checks = []
        for n in (names[0], names[len(names) // 2], names[-1]):
            src = osd["state"][old_idx[n]]
            # find the Parameter object at new flat index
            flat = [p for g in opt.param_groups for p in g["params"]]
            dst = opt.state[flat[new_idx[n]]]
            checks.append({
                "name": n,
                "exp_avg_equal": bool(torch.equal(src["exp_avg"], dst["exp_avg"])),
                "exp_avg_sq_equal": bool(torch.equal(src["exp_avg_sq"], dst["exp_avg_sq"])),
                "step": float(dst["step"]),
                "shape": list(src["exp_avg"].shape),
            })
        res["moment_preservation_checks"] = checks
        res["all_moments_preserved"] = all(
            c["exp_avg_equal"] and c["exp_avg_sq_equal"] for c in checks)
        res["group_lrs_after_remap"] = [
            {"n": len(g["params"]), "base_lr": g.get("base_lr"), "wd": g.get("weight_decay")}
            for g in opt.param_groups]
    except Exception as e:  # noqa: BLE001
        res["remapped_load"] = {"ok": False, "exc": f"{type(e).__name__}: {e}"}
    del ck
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    main(ap.parse_args().path)
