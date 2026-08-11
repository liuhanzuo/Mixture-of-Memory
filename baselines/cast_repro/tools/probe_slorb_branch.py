#!/usr/bin/env python3
"""Quantify the SLoRB branch in a SparseForge checkpoint.

WHY THIS MATTERS FOR THE FOUR-ARM TABLE
---------------------------------------
``sparse_modeling.py:819-822`` adds, on every in-scope projection::

    out = masked_linear(x, W, mask) + (x @ x_proj.T) @ SLoRB_Weight.T

i.e. an *extra, fully dense, rank-(in/k)* term whose effective weight is
``SLoRB_Weight @ x_proj`` (shape out x in).  Consequences:

1. The model that produced the training-time ``lm_eval_results`` (57.27) is
   ``2:4 sparse W`` **plus** this dense low-rank branch -- it is not a pure 2:4
   model, and it has more live parameters than the AST / CAST / Wanda arms.
2. Folding the branch into ``W`` destroys 2:4 (it writes into the pruned
   positions), so the "exact 2:4" gate cannot pass on a folded export.
3. ``deploy_sparse_24/convert.py:189`` drops ``.SLoRB_Weight`` / ``.x_proj``
   as "training auxiliaries" -- that yields a genuinely 2:4 model, but a
   *different* one from the checkpoint's own reported accuracy.

So the export decision is not cosmetic and must be made on measured numbers.
This tool reports, per sampled layer and in aggregate:
  * whether SLoRB_Weight is nonzero at all (if it is all-zero the branch is a
    no-op and the whole question is moot),
  * the extra live parameter count the branch represents,
  * whether x_proj is still the fixed block-sum 0/1 pattern or has been trained
    away from it (``trainable_projection: true`` in this run's args),
  * ||SLoRB_eff|| / ||W_masked|| in Frobenius norm, i.e. how much signal the
    branch carries relative to the sparse weight it supplements.
"""

from __future__ import annotations

import argparse
import json

import torch

PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def nm_2_4_hard(soft: torch.Tensor) -> torch.Tensor:
    """Exact port of sparse_modeling.py's ``nm_2_4`` hard-projection branch."""
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
    ck_args = blob.get("args", {}) or {}
    print(f"[slorb] SLoRB={ck_args.get('SLoRB')} k={ck_args.get('SLoRB_k')} "
          f"init={ck_args.get('SLoRB_init_type')} "
          f"trainable_projection={ck_args.get('trainable_projection')}")

    scope = [k for k in sd
             if k.endswith(".weight") and any(f".{p}." in k for p in PROJECTIONS)]

    # ---- aggregate: how many extra live params does the branch add? --------
    n_slorb_w = sum(sd[k].numel() for k in sd if k.endswith(".SLoRB_Weight"))
    n_xproj = sum(sd[k].numel() for k in sd if k.endswith(".x_proj"))
    n_scope = sum(sd[k].numel() for k in scope)
    print(f"[slorb] in-scope W elements      : {n_scope:,}")
    print(f"[slorb] SLoRB_Weight elements    : {n_slorb_w:,}")
    print(f"[slorb] x_proj elements          : {n_xproj:,}")

    per_layer = []
    any_nonzero = False
    xproj_is_blocksum_all = True
    for wk in scope[: args.sample]:
        base = wk[: -len("weight")]
        sk, xk, mk = base + "SLoRB_Weight", base + "x_proj", base + "mask"
        if sk not in sd or xk not in sd:
            print(f"[slorb] {wk}: no SLoRB tensors")
            continue
        S = sd[sk].float()
        X = sd[xk].float()
        W = sd[wk].float()
        m = nm_2_4_hard(sd[mk].float())
        Wm = W * m

        s_nz = int((S != 0).sum())
        any_nonzero |= s_nz > 0

        # effective dense weight contributed by the branch
        eff = S @ X                                   # (out, in)
        fro_eff = float(eff.norm())
        fro_wm = float(Wm.norm())

        # how much of the branch lands on positions the 2:4 mask pruned away
        pruned_pos = (m == 0)
        fro_eff_pruned = float(eff[pruned_pos].norm())

        # is x_proj still the fixed block-sum 0/1 pattern?
        x_is_binary = bool(torch.all((X == 0) | (X == 1)))
        k = int(ck_args.get("SLoRB_k", 16))
        rows, cols = X.shape
        ref = torch.zeros_like(X)
        idx = torch.arange(rows) * k
        ref[torch.arange(rows)[:, None], idx[:, None] + torch.arange(k)] = 1
        x_is_blocksum = bool(torch.equal(X, ref))
        xproj_is_blocksum_all &= x_is_blocksum

        e = {
            "layer": wk,
            "W_shape": list(W.shape),
            "SLoRB_Weight_shape": list(S.shape),
            "x_proj_shape": list(X.shape),
            "SLoRB_Weight_nonzero": s_nz,
            "SLoRB_Weight_absmax": float(S.abs().max()),
            "x_proj_is_binary01": x_is_binary,
            "x_proj_equals_fixed_blocksum": x_is_blocksum,
            "fro_W_masked": fro_wm,
            "fro_SLoRB_effective": fro_eff,
            "ratio_SLoRB_over_Wmasked": fro_eff / fro_wm if fro_wm else None,
            "fro_SLoRB_on_pruned_positions": fro_eff_pruned,
            "frac_SLoRB_energy_on_pruned_positions": (fro_eff_pruned ** 2) / (fro_eff ** 2)
                if fro_eff else None,
        }
        per_layer.append(e)
        print(f"[slorb] {wk}")
        print(f"[slorb]     S{list(S.shape)} nonzero={s_nz:,} absmax={S.abs().max():.4e}  "
              f"x_proj binary01={x_is_binary} ==fixed_blocksum={x_is_blocksum}")
        print(f"[slorb]     ||SLoRB_eff||/||W*m|| = {e['ratio_SLoRB_over_Wmasked']:.6f}   "
              f"energy on pruned positions = "
              f"{e['frac_SLoRB_energy_on_pruned_positions']*100:.2f}%")

    summary = {
        "ckpt": args.ckpt,
        "iter_num": blob.get("iter_num"),
        "SLoRB_enabled_in_args": ck_args.get("SLoRB"),
        "SLoRB_k": ck_args.get("SLoRB_k"),
        "SLoRB_init_type": ck_args.get("SLoRB_init_type"),
        "trainable_projection": ck_args.get("trainable_projection"),
        "in_scope_W_elements": n_scope,
        "SLoRB_Weight_elements": n_slorb_w,
        "x_proj_elements": n_xproj,
        "SLoRB_Weight_any_nonzero_in_sample": any_nonzero,
        "x_proj_all_fixed_blocksum_in_sample": xproj_is_blocksum_all,
        "per_layer": per_layer,
    }
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[slorb] wrote {args.out_json}")

    print("[slorb] VERDICT: SLoRB branch is "
          + ("ACTIVE (nonzero) -> dropping it changes the model; keeping it means the "
             "arm is NOT a pure 2:4 model" if any_nonzero
             else "ALL-ZERO in the sample -> a no-op, safe to drop"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
