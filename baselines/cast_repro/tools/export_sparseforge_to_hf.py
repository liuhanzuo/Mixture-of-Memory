#!/usr/bin/env python3
"""Export the SparseForge headline checkpoint to plain HF LlamaForCausalLM models.

WHY THIS IS NOT A ONE-LINE ADAPTATION OF ``export_final_to_hf.py``
------------------------------------------------------------------
``export_final_to_hf.py`` assumes a CAST checkpoint that has already been
through ``finalize_all()``: masked entries are *already* exact zeros in the
saved weight, so exporting is just "drop the ``.mask`` buffers". The SparseForge
checkpoint
``out_llama/models_Llama--Llama2-7b_mask-unstructured_s0.5_m-hessian_obd_20260413_201320/model_best_lm_eval.pt``
is a different animal in three ways, each of which changes what "the model"
even means:

1. **The saved weights are DENSE** (zero fraction 0.000000000) and the ``.mask``
   buffers are **continuous** bf16 in ``[4e-11, 1.0]``, not binary. The trainer's
   forward (``sparse_modeling.py:786-789``) uses ``effective_mask = self.mask``
   verbatim whenever ``hardening_x >= 1``, and a freshly constructed
   ``SparseLinear`` has ``hardening_x = 1.0`` (``sparse_modeling.py:389``). The
   checkpoint's own ``lm_eval_results`` (CAST-7 mean 57.2672) was produced by
   ``main_llama.py:2175-2215``, which rebuilds a ``LlamaSparse`` from this state
   dict and calls ``run_lm_eval_benchmarks`` **without** going through
   ``eval_wiki_ppl.optimize_sparse_model_for_inference``. So the anchored 57.27
   is a **soft-masked** measurement, not a 2:4 measurement.

   Hardening is not numerically free: ``||W*soft - W*hard|| / ||W*hard||`` is
   1e-3..8.5e-3 per layer (see ``probe_mask_binariness.py``).

2. **There is an active dense low-rank branch (SLoRB).** ``sparse_modeling.py:819-822``
   adds ``(x @ x_proj.T) @ SLoRB_Weight.T`` to every in-scope projection. In this
   checkpoint ``SLoRB_Weight`` is fully nonzero, and ``x_proj`` has been *trained*
   away from its fixed block-sum init (``trainable_projection: true``), so it is
   not a reconstructible constant. The branch carries a Frobenius norm of
   0.20-0.47 x the masked weight, and ~50 % of its energy lands on positions the
   2:4 mask prunes. It adds **848.4 M** live parameters (``SLoRB_Weight``
   404,750,336 + ``x_proj`` 443,678,720) on top of the 3.238 G surviving weights
   — i.e. **+26 % live in-scope parameters** versus a pure 2:4 model.

   ``deploy_sparse_24/convert.py:189`` discards these two tensors as "training
   auxiliaries". That silently produces a *different, weaker* model than the one
   the checkpoint reports accuracy for.

3. The state dict is nested one extra ``model.`` deep (``model.model.layers.*``,
   ``model.lm_head.weight``) and carries five more per-layer training buffers
   (``hessian_diag``, ``frozen_mask_flags``, ``grad_ema``, ``scaler_row``).

Because (1) and (2) are *model-defining*, this tool refuses to pick one for you.
You choose the variant, and the meta file records it:

  --mask hard --slorb drop   exact 2:4, no extra params.  The ONLY variant that
                             is apples-to-apples with the dense / CAST-repro /
                             AST-official / Wanda arms, and the only one that can
                             pass verify_2of4_hf_export.py.
  --mask hard --slorb fold   2:4 support + SLoRB folded into W. Numerically
                             equal to what SparseForge deploys, but the folded
                             weight is DENSE, so it is not a 2:4 model and must
                             never be put in a 2:4 column.
  --mask soft --slorb fold   the exact model that produced the checkpoint's own
                             57.2672 CAST-7 anchor. Faithfulness reference.
  --mask soft --slorb drop   ablation: soft mask, no branch.

Folding is exact algebra, not an approximation: the branch is linear, so
``W_eff = W*mask + SLoRB_Weight @ x_proj`` reproduces the two-matmul forward
exactly up to the final bf16 rounding (accumulation is done in fp32 here).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
AUX_SUFFIXES = (".mask", ".hessian_diag", ".frozen_mask_flags", ".grad_ema",
                ".scaler_row", ".SLoRB_Weight", ".x_proj", ".cast_scale")
EXPECTED_SCOPE_TENSORS = 224
EXPECTED_SCOPE_ELEMENTS = 6_476_005_376


def nm_2_4_hard(soft: torch.Tensor) -> torch.Tensor:
    """Exact port of the ``nm_2_4`` branch of
    ``sparse_modeling.SparseLinear._hard_mask_from_soft`` (lines 594-621):
    per group of 4 consecutive input columns, keep the top-2 by mask value via
    ``topk(2)`` + ``scatter_``.

    This is NOT the same as ``soft > 0.5``: on this checkpoint thresholding
    leaves 26,726 groups off-pattern, because the continuous mask does not have
    exactly two entries above 0.5 in every group.
    """
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


def strip_prefix(key: str) -> str:
    """``model.model.layers.0.…`` -> ``model.layers.0.…``; ``model.lm_head.weight``
    -> ``lm_head.weight``. The trainer wraps LlamaForCausalLM in a ``LlamaSparse``
    whose submodule is itself named ``model``."""
    return key[len("model."):] if key.startswith("model.") else key


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--mask", choices=["hard", "soft"], required=True,
                    help="hard = exact 2:4 via nm_2_4 top-2-per-4; soft = the "
                         "continuous mask the trainer's forward actually used")
    ap.add_argument("--slorb", choices=["drop", "fold"], required=True,
                    help="drop = discard the low-rank branch (yields a pure 2:4 "
                         "model); fold = add SLoRB_Weight@x_proj into W (exact, "
                         "but destroys 2:4)")
    ap.add_argument("--model", default="models/Llama--Llama2-7b")
    ap.add_argument("--project-root", default="/apdcephfs_wzc1/share_304376610/pighzliu_code")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    root = Path(args.project_root)
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = root / model_path
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = root / ckpt_path
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = root / out_path

    print(f"[export] ckpt   = {ckpt_path}")
    print(f"[export] variant= mask={args.mask} slorb={args.slorb}")
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    sd = blob["model_state_dict"]
    print(f"[export] iter_num={blob.get('iter_num')} "
          f"finalization_done={blob.get('finalization_done')} "
          f"best_lm_eval_mean={blob.get('best_lm_eval_mean')}")
    print(f"[export] {len(sd)} raw tensors")

    clean: dict[str, torch.Tensor] = {}
    n_scope = 0
    n_elements = 0
    zeros = 0
    violations = 0
    slorb_folded = 0
    stats = {"max_abs_fold_delta": 0.0}

    for k in sorted(sd):
        if any(k.endswith(s) for s in AUX_SUFFIXES):
            continue
        nk = strip_prefix(k)
        v = sd[k]
        is_scope = k.endswith(".weight") and any(f".{p}." in k for p in PROJECTIONS)
        if not is_scope:
            clean[nk] = v.clone()
            continue

        base = k[: -len("weight")]
        mk = base + "mask"
        if mk not in sd:
            raise SystemExit(f"{k} is in scope but has no sibling mask -- refusing to guess")
        soft = sd[mk].float()
        w = v.float()

        m = nm_2_4_hard(soft) if args.mask == "hard" else soft
        eff = w * m

        if args.slorb == "fold":
            sk, xk = base + "SLoRB_Weight", base + "x_proj"
            if sk not in sd or xk not in sd:
                raise SystemExit(f"--slorb fold requested but {sk}/{xk} missing")
            add = sd[sk].float() @ sd[xk].float()
            if add.shape != eff.shape:
                raise SystemExit(
                    f"SLoRB effective weight shape {tuple(add.shape)} != W {tuple(eff.shape)}")
            stats["max_abs_fold_delta"] = max(stats["max_abs_fold_delta"], float(add.abs().max()))
            eff = eff + add
            slorb_folded += 1

        out = eff.to(getattr(torch, args.dtype))
        clean[nk] = out

        n_scope += 1
        n_elements += out.numel()
        ow = out.float()
        r, c = ow.shape
        zeros += int((ow == 0).sum())
        nz = (ow != 0).reshape(r, c // 4, 4).sum(-1)
        violations += int((nz != 2).sum())

    print(f"[export] kept {len(clean)} tensors; in-scope processed {n_scope}")
    if n_scope != EXPECTED_SCOPE_TENSORS or n_elements != EXPECTED_SCOPE_ELEMENTS:
        raise SystemExit(f"scope mismatch: {n_scope}/{n_elements} != "
                         f"{EXPECTED_SCOPE_TENSORS}/{EXPECTED_SCOPE_ELEMENTS}")
    zero_frac = zeros / n_elements
    print(f"[export] in-scope zero_fraction={zero_frac:.9f} exact-2:4 violations={violations}")
    if args.slorb == "fold":
        print(f"[export] folded SLoRB into {slorb_folded} tensors; "
              f"max |SLoRB_eff| = {stats['max_abs_fold_delta']:.6e}")

    exact_2of4 = (violations == 0 and abs(zero_frac - 0.5) < 1e-6)
    if args.mask == "hard" and args.slorb == "drop" and not exact_2of4:
        raise SystemExit("mask=hard slorb=drop must yield exact 2:4 -- it did not; refusing to write")
    if args.slorb == "fold" and exact_2of4:
        raise SystemExit("folding SLoRB left the weight exactly 2:4, which is impossible "
                         "unless the branch is a no-op -- investigate before trusting this export")

    # ---- key-set invariant vs the dense reference -------------------------
    index = json.loads((model_path / "model.safetensors.index.json").read_text())["weight_map"]
    ref = set(index)
    missing = {k for k in ref - set(clean) if not k.endswith("rotary_emb.inv_freq")}
    extra = set(clean) - ref
    if missing or extra:
        raise SystemExit(f"key-set mismatch: missing={sorted(missing)[:8]} extra={sorted(extra)[:8]}")
    print("[export] key set matches dense reference (modulo rotary inv_freq)")

    dtype = getattr(torch, args.dtype)
    print(f"[export] materializing reference skeleton ({args.dtype})", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, low_cpu_mem_usage=True, local_files_only=True,
    )
    incompat = model.load_state_dict({k: v.to(dtype) for k, v in clean.items()}, strict=False)
    still_missing = [k for k in incompat.missing_keys if not k.endswith("rotary_emb.inv_freq")]
    if incompat.unexpected_keys or still_missing:
        raise SystemExit(f"load_state_dict mismatch: unexpected={list(incompat.unexpected_keys)[:8]} "
                         f"missing={still_missing[:8]}")

    # Re-verify AFTER the cast: bf16 rounding must not move a zero.
    post_viol = 0
    post_zeros = 0
    post_elems = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear) or mod.weight.ndim != 2:
            continue
        if not any(name.endswith(p) for p in PROJECTIONS):
            continue
        w = mod.weight.detach().float()
        r, c = w.shape
        post_elems += w.numel()
        post_zeros += int((w == 0).sum())
        nz = (w != 0).reshape(r, c // 4, 4).sum(-1)
        post_viol += int((nz != 2).sum())
    print(f"[export] post-cast: zero_frac={post_zeros/post_elems:.9f} violations={post_viol}")
    if args.mask == "hard" and args.slorb == "drop" and post_viol:
        raise SystemExit(f"{post_viol} groups broke 2:4 after the {args.dtype} cast")

    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_path, safe_serialization=True)

    copied = []
    for fname in ("tokenizer.model", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json"):
        src = model_path / fname
        if src.exists():
            shutil.copyfile(src, out_path / fname)
            copied.append(fname)
    if "tokenizer.model" not in copied:
        raise SystemExit(f"{model_path} has no tokenizer.model; refusing to export a broken tokenizer")
    print(f"[export] copied tokenizer files verbatim: {copied}")

    (out_path / "sparseforge_export_meta.json").write_text(json.dumps({
        "source_ckpt": str(ckpt_path),
        "source_iter_num": blob.get("iter_num"),
        "source_best_lm_eval_mean": blob.get("best_lm_eval_mean"),
        "source_lm_eval_results": blob.get("lm_eval_results"),
        "dense_reference": str(model_path),
        "dtype": args.dtype,
        "mask_variant": args.mask,
        "mask_hardening": ("nm_2_4 exact top-2-per-4 (sparse_modeling.py:594-621)"
                           if args.mask == "hard"
                           else "none -- continuous mask used verbatim, as the trainer forward does"),
        "slorb_variant": args.slorb,
        "slorb_note": ("SLoRB_Weight @ x_proj folded into W (exact linear algebra; "
                       "makes the weight dense)" if args.slorb == "fold"
                       else "SLoRB_Weight and x_proj discarded -- this REMOVES an active "
                            "forward-pass branch that was present when the checkpoint's own "
                            "lm_eval_results were measured"),
        "scope_tensors": n_scope,
        "scope_elements": n_elements,
        "scope_zero_fraction": zero_frac,
        "exact_2of4_violations": violations,
        "exact_2of4_violations_post_cast": post_viol,
        "is_exact_2of4": bool(exact_2of4 and post_viol == 0),
        "max_abs_slorb_effective": stats["max_abs_fold_delta"] if args.slorb == "fold" else None,
    }, indent=2))
    print(f"[export] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
