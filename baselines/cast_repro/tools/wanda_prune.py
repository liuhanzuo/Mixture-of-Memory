#!/usr/bin/env python3
"""Training-free Wanda pruning under exact 2:4 sparsity, packaged for CAST SPEC.md §8.

Reference: Sun et al., "A Simple and Effective Pruning Approach for Large Language
Models", arXiv:2306.11695 (Wanda). Under 2:4 the algorithm is:

    For each in-scope Linear W in {q,k,v,o,gate,up,down} × 32 layers:
      1. Capture per-input-channel activation L2 norm ‖X_j‖_2 over K
         calibration sequences (layer-sequential; each layer sees the OUTPUT
         of the previously pruned layer, which is Wanda's original recipe).
      2. Score  S_ij = |W_ij| · ‖X_j‖_2
      3. Reshape rows into contiguous 4-column groups; keep top-2 by S,
         zero the other two. No gradient updates.

Scope Θ is the same 224 in-block projections that CAST puts under mask
(embed_tokens and lm_head stay dense). Save layout matches
``tools/export_final_to_hf.py``'s ``hf_final`` — safetensors from the
pruned model, tokenizer files copied verbatim from the dense reference,
plus a ``wanda_export_meta.json`` breadcrumb.

Refuses to save unless the exact-2:4 property holds over all 224 tensors
(0 bad tiles). This is the same criterion ``verify_2of4_hf_export.py``
enforces, applied here at export time so we cannot ship a broken artifact.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

#: The 7 in-block projections that CAST/Wanda both put in scope (see SPEC.md §5(e)).
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
EXPECTED_CAST_TENSORS = 224
EXPECTED_CAST_ELEMENTS = 6_476_005_376


def find_layers(module: nn.Module, layers=(nn.Linear,), name: str = "") -> dict[str, nn.Linear]:
    """Return {name: linear} for every ``nn.Linear`` in a decoder layer subtree."""
    if isinstance(module, tuple(layers)):
        return {name: module}
    res: dict[str, nn.Linear] = {}
    for child_name, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=f"{name}.{child_name}" if name else child_name))
    return res


def sample_memmap_sequences(
    path: Path,
    *,
    nsamples: int,
    seqlen: int,
    seed: int,
    dtype: str = "uint16",
) -> list[torch.Tensor]:
    data = np.memmap(path, dtype=np.dtype(dtype), mode="r")
    rng = random.Random(seed)
    samples: list[torch.Tensor] = []
    for _ in range(nsamples):
        start = rng.randint(0, len(data) - seqlen - 2)
        ids = torch.from_numpy(np.asarray(data[start : start + seqlen], dtype=np.int64)).unsqueeze(0)
        samples.append(ids)
    return samples


@torch.no_grad()
def wanda_prune_2of4(
    model: nn.Module,
    calibration: list[torch.Tensor],
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Apply Wanda 2:4 pruning in-place on every ``nn.Linear`` under a decoder layer.

    Follows the layer-sequential recipe from Sun et al.: capture activations at
    each layer with a forward hook, compute score = |W| * ||X||_2 per column,
    apply exact 2:4 with topk(2)+scatter over row-major 4-groups.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    nsamples = len(calibration)
    seqlen = int(calibration[0].shape[1])
    hidden = int(model.config.hidden_size)

    model.model.embed_tokens = model.model.embed_tokens.to(device)
    model.model.norm = model.model.norm.to(device)
    if getattr(model.model, "rotary_emb", None) is not None:
        model.model.rotary_emb = model.model.rotary_emb.to(device)
    layers[0] = layers[0].to(device)
    dtype = next(model.parameters()).dtype

    # Buffer for per-sample layer-0 inputs. We reuse this pair of buffers per
    # (input, output) to avoid allocating 2 GB * n_layers of intermediates.
    inps = torch.zeros((nsamples, seqlen, hidden), dtype=dtype, device=device)
    cache: dict[str, Any] = {"i": 0, "kwargs": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            idx = cache["i"]
            inps[idx].copy_(inp[0] if inp.dim() == 3 else inp)
            cache["i"] += 1
            if cache["kwargs"] is None:
                cache["kwargs"] = kwargs
            raise ValueError("capture-complete")

    layers[0] = Catcher(layers[0])
    for batch in calibration:
        try:
            model(batch.to(device))
        except ValueError as exc:
            if str(exc) != "capture-complete":
                raise
    layers[0] = layers[0].module.cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    layer_kwargs = cache["kwargs"] or {}
    total_tiles = 0
    total_bad = 0
    total_masked_elements = 0
    total_elements = 0
    started = time.time()

    for layer_idx, layer_cpu in enumerate(layers):
        layer_start = time.time()
        layer = layer_cpu.to(device)
        subset = find_layers(layer)
        # Per-column activation L2^2 accumulator, kept in fp64 for stability.
        scalers = {
            name: torch.zeros(module.in_features, device=device, dtype=torch.float64)
            for name, module in subset.items()
        }
        counts = {name: 0 for name in subset}

        def make_hook(target_name: str):
            def hook(_, inp, __):
                x = inp[0].detach()
                x = x.reshape(-1, x.shape[-1]).double()
                scalers[target_name].add_(x.pow(2).sum(dim=0))
                counts[target_name] += x.shape[0]
            return hook

        handles = [module.register_forward_hook(make_hook(name)) for name, module in subset.items()]
        for sample_idx in range(nsamples):
            outs[sample_idx] = layer(inps[sample_idx].unsqueeze(0), **layer_kwargs)[0]
        for handle in handles:
            handle.remove()

        # Compute + apply the 2:4 Wanda mask per in-scope linear in this layer.
        for name, module in subset.items():
            weight = module.weight.data
            out_dim, in_dim = weight.shape
            full = (in_dim // 4) * 4
            if full <= 0:
                continue
            # ||X||_2 per column, sqrt(mean-square). counts is total (batch*seq).
            activation_scale = torch.sqrt(
                (scalers[name][:full] / max(counts[name], 1)).float() + 1e-8
            )
            score = weight[:, :full].abs().float() * activation_scale.unsqueeze(0)
            grouped = score.view(out_dim, -1, 4)
            keep = torch.topk(grouped, k=2, dim=-1, largest=True).indices
            mask = torch.zeros_like(grouped, dtype=torch.bool)
            mask.scatter_(-1, keep, True)
            flat_mask = mask.view(out_dim, full)
            # Multiply the weight in place. Zeros stay exact zeros regardless
            # of dtype (0 * x = 0 for bf16/fp16 too).
            weight[:, :full].mul_(flat_mask.to(weight.dtype))

            # Local accounting.
            n_tiles = flat_mask.numel() // 4
            n_bad = int(((flat_mask.sum(dim=-1).view(-1, 1)) != 2).sum().item())  # will be 0 by construction
            # More precise: per-4-group sum
            per_group_sum = flat_mask.view(out_dim, -1, 4).sum(dim=-1)
            n_bad_precise = int((per_group_sum != 2).sum().item())
            total_tiles += per_group_sum.numel()
            total_bad += n_bad_precise
            total_masked_elements += int((weight[:, :full] == 0).sum().item())
            total_elements += weight[:, :full].numel()

        # Re-run the (now-pruned) layer to get the outputs consumed by the
        # next layer's activation stats -- this is what "sequential Wanda" means.
        for sample_idx in range(nsamples):
            outs[sample_idx] = layer(inps[sample_idx].unsqueeze(0), **layer_kwargs)[0]
        layers[layer_idx] = layer.cpu()
        del layer, scalers, counts
        torch.cuda.empty_cache()
        inps, outs = outs, inps

        cumulative_zero_frac = total_masked_elements / max(total_elements, 1)
        print(
            f"[Wanda] layer={layer_idx + 1}/{len(layers)} "
            f"elapsed={time.time() - layer_start:.1f}s "
            f"cumulative_zero_frac={cumulative_zero_frac:.6f} "
            f"bad_tiles_so_far={total_bad}",
            flush=True,
        )

    model.config.use_cache = use_cache
    return {
        "total_tiles": total_tiles,
        "bad_tiles": total_bad,
        "exact_2of4_frac": 1.0 - total_bad / max(total_tiles, 1),
        "linear_zero_frac_in_scope": total_masked_elements / max(total_elements, 1),
        "prune_seconds": time.time() - started,
    }


def verify_hf_scope_2of4(model: nn.Module) -> dict[str, Any]:
    """Recompute the 2:4 stats on the pruned in-memory model, in-scope only."""
    n_tensors = 0
    n_elements = 0
    zeros = 0
    total_tiles = 0
    bad = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or mod.weight.ndim != 2:
            continue
        if not any(name.endswith(p) for p in PROJECTIONS):
            continue
        w = mod.weight.detach().float()
        r, c = w.shape
        n_tensors += 1
        n_elements += w.numel()
        zeros += int((w == 0).sum().item())
        nz = (w != 0).reshape(r, c // 4, 4).sum(-1)
        total_tiles += nz.numel()
        bad += int((nz != 2).sum().item())
    return {
        "n_tensors": n_tensors,
        "n_elements": n_elements,
        "zeros": zeros,
        "zero_frac": zeros / max(n_elements, 1),
        "total_tiles": total_tiles,
        "bad_tiles": bad,
        "exact_2of4_frac": 1.0 - bad / max(total_tiles, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="dense LLaMA-2-7B HF path")
    ap.add_argument("--train_bin", required=True,
                    help="calibration token memmap (LLaMA-tokenized, uint16)")
    ap.add_argument("--dtype_bin", default="uint16")
    ap.add_argument("--output", required=True, help="HF export directory to write")
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--storage_dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    model_path = Path(args.model)
    out_path = Path(args.output)

    print(f"[wanda] loading dense model: {model_path}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=getattr(torch, args.storage_dtype),
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval()
    model.seqlen = args.seqlen

    print(f"[wanda] sampling {args.nsamples} sequences of {args.seqlen} tokens "
          f"from {args.train_bin} (seed={args.seed})", flush=True)
    calibration = sample_memmap_sequences(
        Path(args.train_bin),
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        seed=args.seed,
        dtype=args.dtype_bin,
    )

    prune_stats = wanda_prune_2of4(model, calibration, device=device)
    print(f"[wanda] prune stats: {json.dumps(prune_stats, indent=2)}", flush=True)

    # Full in-memory in-scope verify -- must be exact.
    verify = verify_hf_scope_2of4(model)
    print(f"[wanda] in-memory verify (in-scope only): {json.dumps(verify, indent=2)}", flush=True)

    if verify["n_tensors"] != EXPECTED_CAST_TENSORS or verify["n_elements"] != EXPECTED_CAST_ELEMENTS:
        raise SystemExit(
            f"scope mismatch: got {verify['n_tensors']} tensors / {verify['n_elements']} elements; "
            f"expected {EXPECTED_CAST_TENSORS} / {EXPECTED_CAST_ELEMENTS}"
        )
    if verify["bad_tiles"] != 0:
        raise SystemExit(
            f"exact-2:4 violated: {verify['bad_tiles']} of {verify['total_tiles']} "
            "tiles are not 2:4 -- refusing to save"
        )
    if abs(verify["zero_frac"] - 0.5) > 1e-4:
        raise SystemExit(
            f"in-scope zero_frac={verify['zero_frac']:.9f} not within 1e-4 of 0.5 -- refusing"
        )

    # Save.
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"[wanda] saving pruned HF model to {out_path}", flush=True)
    model.save_pretrained(out_path, safe_serialization=True)

    # Copy tokenizer files verbatim -- same reason as export_final_to_hf.py:
    # transformers >= 5.13 silently drops tokenizer.model on round-trip, which
    # breaks use_fast=False loads in some code paths.
    copied = []
    for fname in ("tokenizer.model", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json"):
        src = model_path / fname
        if src.exists():
            shutil.copyfile(src, out_path / fname)
            copied.append(fname)
    if "tokenizer.model" not in copied:
        raise SystemExit(
            f"{model_path} has no tokenizer.model; refusing to write a broken tokenizer"
        )
    print(f"[wanda] copied tokenizer files: {copied}", flush=True)

    (out_path / "wanda_export_meta.json").write_text(json.dumps({
        "algorithm": "Wanda 2:4 (Sun et al., arXiv:2306.11695)",
        "dense_reference": str(model_path),
        "calibration": {
            "source": str(args.train_bin),
            "dtype": args.dtype_bin,
            "nsamples": args.nsamples,
            "seqlen": args.seqlen,
            "seed": args.seed,
        },
        "storage_dtype": args.storage_dtype,
        "in_scope_tensors": verify["n_tensors"],
        "in_scope_elements": verify["n_elements"],
        "in_scope_zero_frac": round(verify["zero_frac"], 9),
        "in_scope_bad_tiles": verify["bad_tiles"],
        "in_scope_exact_2of4_frac": round(verify["exact_2of4_frac"], 9),
        "prune_seconds": round(prune_stats["prune_seconds"], 2),
    }, indent=2))
    print(f"[wanda] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
