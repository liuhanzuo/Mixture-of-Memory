#!/usr/bin/env python
"""Read-only ckpt metadata probe for resume-feasibility audit.

Prints, per ckpt path: presence + size of optimizer_state / rng_state /
train_args / step / epoch / max_steps / warmup_steps, plus the key train_args
that govern LR schedule + data path. Uses mmap so we never materialize the
weights. NO training, NO writes.
"""
import argparse
import json
import os
import sys

import torch


def probe(path):
    out = {"path": path}
    try:
        st = os.stat(path)
        out["bytes"] = st.st_size
    except OSError as e:
        out["error"] = f"stat failed: {e}"
        return out
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except Exception as e:  # noqa: BLE001
        out["error"] = f"torch.load failed: {type(e).__name__}: {e}"
        return out
    out["top_keys"] = sorted(ck.keys())
    out["step"] = ck.get("step")
    out["epoch"] = ck.get("epoch")
    out["max_steps_field"] = ck.get("max_steps")
    out["warmup_steps_field"] = ck.get("warmup_steps")
    out["has_optimizer_state"] = "optimizer_state" in ck
    out["has_rng_state"] = "rng_state" in ck
    out["has_train_args"] = "train_args" in ck
    out["n_model_tensors"] = len(ck.get("model_state", {}))
    if out["has_optimizer_state"]:
        osd = ck["optimizer_state"]
        try:
            out["opt_n_param_states"] = len(osd.get("state", {}))
            out["opt_n_param_groups"] = len(osd.get("param_groups", []))
            # sample one state entry to prove real moments present
            k0 = next(iter(osd["state"]))
            ent = osd["state"][k0]
            out["opt_state_entry_keys"] = sorted(
                (kk, tuple(vv.shape) if torch.is_tensor(vv) else str(vv))
                for kk, vv in ent.items()
            )
            out["opt_group_lrs"] = [
                {kk: g.get(kk) for kk in ("lr", "base_lr", "min_lr", "weight_decay")}
                for g in osd.get("param_groups", [])
            ]
        except Exception as e:  # noqa: BLE001
            out["opt_probe_error"] = f"{type(e).__name__}: {e}"
    if out["has_rng_state"]:
        rng = ck["rng_state"]
        out["rng_keys"] = sorted(rng.keys()) if isinstance(rng, dict) else str(type(rng))
        if isinstance(rng, dict) and "cuda" in rng:
            out["rng_n_cuda"] = len(rng["cuda"])
    if out["has_train_args"]:
        ta = ck["train_args"]
        keep = ("lr", "min_lr", "lr_inherited", "min_lr_inherited", "warmup_steps",
                "max_steps", "batch_size", "grad_accumulation_steps", "seq_len",
                "weight_decay", "grad_clip", "data_path", "output_dir",
                "keep_front_layers", "n_fresh_layers", "seed", "optimizer",
                "gradient_checkpointing", "save_every", "milestone_every",
                "model_path", "freeze_front", "from_scratch")
        out["train_args_subset"] = {k: ta.get(k) for k in keep if k in ta}
    out["arch"] = {k: ck.get(k) for k in ("keep_front_layers", "n_fresh_layers",
                                          "num_hidden_layers", "seq_len", "seed")}
    del ck
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    a = ap.parse_args()
    res = [probe(p) for p in a.paths]
    print(json.dumps(res, indent=2, default=str))
    sys.stdout.flush()
