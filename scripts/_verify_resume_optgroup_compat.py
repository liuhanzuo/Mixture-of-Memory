#!/usr/bin/env python
"""Resume-fidelity test: would CURRENT train_olmo2_arch_probe2.py be able to
restore the optimizer state stored in these keepN ckpts?

The ckpts were written by a PRE-`module.`-strip build_param_groups (2 groups:
inh_decay / inh_nodecay). Current _classify_param strips 'module.' so it
classifies the fresh tail correctly -> 4 groups. torch's
Optimizer.load_state_dict raises ValueError on a group-count / group-size
mismatch, which train_olmo2_arch_probe2.py catches -> silent WARM RESTART.

This script does NOT build the 7B model. It reconstructs the exact parameter
name list from the ckpt's own model_state keys, runs the REAL _classify_param
from the trainer, buckets by (fresh|inh) x (decay|nodecay) using the real
ndim rule, and then runs a real torch AdamW.load_state_dict against the
ckpt's saved optimizer_state using dummy params of the right shapes.
Read-only w.r.t. the ckpts. No CUDA. No training.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from train_olmo2_arch_probe2 import _classify_param  # noqa: E402


def analyze(path):
    out = {"path": path}
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    keep_front = ck["keep_front_layers"]
    n_fresh = ck["n_fresh_layers"]
    ms = ck["model_state"]
    names = list(ms.keys())
    shapes = {k: tuple(v.shape) for k, v in ms.items()}
    out["keep_front_layers"] = keep_front
    out["n_fresh_layers"] = n_fresh
    out["n_tensors"] = len(names)

    # ---- what the CKPT actually stored ----
    osd = ck.get("optimizer_state")
    if osd is None:
        out["ckpt_has_optimizer_state"] = False
        del ck
        return out
    out["ckpt_has_optimizer_state"] = True
    out["ckpt_n_groups"] = len(osd["param_groups"])
    out["ckpt_group_sizes"] = [len(g["params"]) for g in osd["param_groups"]]
    out["ckpt_group_base_lr"] = [g.get("base_lr") for g in osd["param_groups"]]

    # ---- what CURRENT code would build ----
    specs = {"fresh_decay": [], "fresh_nodecay": [], "inh_decay": [], "inh_nodecay": []}
    for name in names:
        cls = _classify_param(name, keep_front, False, random_trunk=False)
        prefix = "fresh" if cls == "fresh" else "inh"
        ndim = len(shapes[name])
        key = f"{prefix}_decay" if ndim >= 2 else f"{prefix}_nodecay"
        specs[key].append(name)
    cur_groups = {k: v for k, v in specs.items() if v}
    out["current_code_n_groups"] = len(cur_groups)
    out["current_code_group_sizes"] = {k: len(v) for k, v in cur_groups.items()}
    out["current_code_fresh_names_sample"] = (
        cur_groups.get("fresh_decay", [])[:3] + cur_groups.get("fresh_nodecay", [])[:2]
    )

    # ---- what the PRE-FIX (module.-prefixed) code would have built ----
    specs_old = {"fresh_decay": [], "fresh_nodecay": [], "inh_decay": [], "inh_nodecay": []}
    for name in names:
        # DDP-wrapped named_parameters() -> 'module.' + name; pre-fix code did NOT strip
        dname = "module." + name
        # emulate pre-fix _classify_param: no strip
        if dname.startswith("model.layers."):
            lid = int(dname.split(".")[2])
            cls = "inherited" if lid < keep_front else "fresh"
        elif dname.startswith("lm_head"):
            cls = "fresh"
        else:
            cls = "inherited"
        prefix = "fresh" if cls == "fresh" else "inh"
        ndim = len(shapes[name])
        key = f"{prefix}_decay" if ndim >= 2 else f"{prefix}_nodecay"
        specs_old[key].append(name)
    old_groups = {k: v for k, v in specs_old.items() if v}
    out["prefix_bug_code_n_groups"] = len(old_groups)
    out["prefix_bug_code_group_sizes"] = {k: len(v) for k, v in old_groups.items()}

    # ---- REAL load_state_dict test with CURRENT grouping ----
    def try_load(grouping, label):
        pg = []
        for gname, gnames in grouping.items():
            params = [torch.nn.Parameter(torch.zeros(shapes[n], dtype=torch.float32))
                      for n in gnames]
            pg.append({"params": params, "weight_decay": 0.1 if "decay" in gname
                       and "nodecay" not in gname else 0.0,
                       "base_lr": 2e-5, "min_lr": 2e-6})
        opt = torch.optim.AdamW(pg, lr=2e-5, betas=(0.9, 0.95), eps=1e-8)
        try:
            opt.load_state_dict(osd)
            return {"ok": True, "n_states": len(opt.state)}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "exc": f"{type(e).__name__}: {e}"}

    out["load_with_CURRENT_grouping"] = try_load(cur_groups, "current")
    out["load_with_PREFIXBUG_grouping"] = try_load(old_groups, "prefix_bug")
    del ck
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    a = ap.parse_args()
    print(json.dumps([analyze(p) for p in a.paths], indent=2, default=str))
