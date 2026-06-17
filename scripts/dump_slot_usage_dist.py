"""Dump + plot the 128-slot cumulative usage distribution of a mem_space ckpt.

DIAGNOSTIC-ONLY. This script does NOT change any eval/train numeric path: it
reuses ``run_babilong_mem_space``'s model loader + bsz=1 streaming generation
verbatim, and only READS the layer-0 ``MemorySpaceLayer._cum_usage`` tensor
([B, num_slots]) that the forward pass already maintains as no_grad telemetry.

For each BABILong sample we stream the haystack through the bank (exactly the
real eval ingestion path), then read ``_cum_usage[0]`` — the per-slot top-k
selection count accumulated over EVERY chunk of that ONE sample. ``_cum_usage``
is sample-scoped and layer-0-only (the bank is shared across all 32 layers, so
layer-0 drives a single un-double-counted accumulator). We snapshot it right
after ingestion (before generation, which freezes the bank, so it does not add
selections) and then ``_reset_banks`` for the next sample re-zeros it.

We average the per-sample [128] histograms over all evaluated samples to get a
mean "how often is each slot selected per sample" profile, AND keep the summed
total. Output:
  * report/figs/slot_usage_dist.png / .pdf  — sorted bar chart of mean per-slot
    selection count, dead-slot count + top-16 concentration annotated.
  * report/figs/slot_usage_dist.npy         — raw [n_samples, 128] matrix.
  * stdout text stats.

Usage:
    python scripts/dump_slot_usage_dist.py \
        --model_path models/Meta-Llama-3-8B \
        --checkpoint outputs/T2_recall_chunk512_MASS_N128/mem_space_adapter.pt \
        --adapter_config outputs/T2_recall_chunk512_MASS_N128/adapter_config.json \
        --task qa5 --length 32k --limit 40 --chunk_size 512
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
_BABILONG_PKG = os.path.join(PROJECT_ROOT, "third_party", "babilong-pkg")
if os.path.isdir(_BABILONG_PKG) and _BABILONG_PKG not in sys.path:
    sys.path.insert(0, _BABILONG_PKG)

import datasets  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from babilong.prompts import (  # noqa: E402
    DEFAULT_PROMPTS,
    DEFAULT_TEMPLATE,
    get_formatted_input,
)

from scripts.run_babilong_mem_space import (  # noqa: E402
    build_mem_space_config,
    load_mem_space_model,
    _reset_banks,
    _reset_l2,
)


def _get_layer0_cum_usage(model) -> torch.Tensor | None:
    """Return the layer-0 MemorySpaceLayer's _cum_usage [B, N] (or None).

    _cum_usage is maintained ONLY on the layer whose _layer_idx == 0 (the bank
    is shared, so a single layer drives the accumulator). We scan _mem_space_layers
    and return the first one carrying a non-None _cum_usage.
    """
    root = getattr(model, "module", model)
    for w in getattr(root, "_mem_space_layers", []) or []:
        cu = getattr(w, "_cum_usage", None)
        if cu is not None and getattr(w, "_layer_idx", None) == 0:
            return cu
    # fallback: any layer with a non-None accumulator
    for w in getattr(root, "_mem_space_layers", []) or []:
        cu = getattr(w, "_cum_usage", None)
        if cu is not None:
            return cu
    return None


@torch.no_grad()
def stream_and_read_usage(model, input_ids, chunk_size, device):
    """Stream ALL chunks of a sample through the bank, then read _cum_usage[0].

    Mirrors generate_with_mem_space's ingestion exactly (reset -> stream
    chunks), but we stream ALL chunks (including the last) because we only want
    the usage histogram, not generation. Returns a [num_slots] numpy array, or
    None if no accumulator was found.
    """
    _reset_banks(model)
    _reset_l2(model)
    tokens = input_ids[0]
    chunks = list(tokens.split(chunk_size))
    for chunk in chunks:
        ct = chunk.unsqueeze(0).to(device)
        _ = model(input_ids=ct, use_cache=False)
    cu = _get_layer0_cum_usage(model)
    if cu is None:
        return None
    return cu[0].detach().float().cpu().numpy().copy()  # [num_slots]


def make_plot(usage_mat: np.ndarray, top_k: int, num_slots: int,
              task: str, length: str, n_samples: int, out_base: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm

    for fp in ["/usr/share/fonts/google-noto-cjk/NotoSansCJKsc-Regular.otf",
               "/usr/share/fonts/google-noto-cjk/NotoSansCJKsc-Light.otf"]:
        try:
            fm.fontManager.addfont(fp)
        except Exception:
            pass
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
    plt.rcParams["axes.unicode_minus"] = False

    mean_usage = usage_mat.mean(axis=0)          # [num_slots] mean per-sample
    order = np.argsort(mean_usage)[::-1]          # descending
    sorted_usage = mean_usage[order]
    dead = int((mean_usage == 0).sum())
    total = mean_usage.sum()
    top16_frac = sorted_usage[:top_k].sum() / total if total > 0 else 0.0

    fig, ax = plt.subplots(figsize=(9, 4.2))
    colors = ["#c0392b" if i < top_k else "#2980b9"
              for i in range(num_slots)]
    ax.bar(range(num_slots), sorted_usage, color=colors, width=0.9)
    ax.set_xlabel("slot 排名 (按平均被选中次数降序; 0=最热)")
    ax.set_ylabel("平均每样本被 top-k 选中次数")
    ax.set_title(
        f"128-slot 路由调用分布  |  {task} {length}, n={n_samples}, "
        f"top_k={top_k}/{num_slots}"
    )
    txt = (
        f"dead slot (=0): {dead}/{num_slots}\n"
        f"前 {top_k} 热点占比: {top16_frac*100:.1f}%\n"
        f"max={sorted_usage[0]:.1f}  median={np.median(mean_usage):.2f}"
    )
    ax.text(0.97, 0.95, txt, transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    ax.axvline(top_k - 0.5, color="gray", ls="--", lw=1, alpha=0.7)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_base + ".png", dpi=140)
    fig.savefig(out_base + ".pdf")
    print(f"[dump_slot_usage] saved {out_base}.png / .pdf")
    return mean_usage, dead, top16_frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--adapter_config", required=True)
    ap.add_argument("--dataset_name", default="RMT-team/babilong")
    ap.add_argument("--task", default="qa5")
    ap.add_argument("--length", default="32k")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--chunk_size", type=int, default=512)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--out_base", default="report/figs/slot_usage_dist")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    with open(args.adapter_config) as f:
        adapter_cfg = json.load(f)
    mem_config = build_mem_space_config(adapter_cfg)
    mem_config.l3_recon_max_positions = args.chunk_size
    num_slots = mem_config.num_slots
    top_k = mem_config.top_k
    print(f"[dump_slot_usage] num_slots={num_slots} top_k={top_k} "
          f"chunk_size={args.chunk_size}")

    model = load_mem_space_model(
        model_path=args.model_path,
        checkpoint_path=args.checkpoint,
        mem_config=mem_config,
        device=device,
        dtype=dtype,
    )

    prompt_cfg = {
        "instruction": DEFAULT_PROMPTS[args.task]["instruction"],
        "examples": DEFAULT_PROMPTS[args.task]["examples"],
        "post_prompt": DEFAULT_PROMPTS[args.task]["post_prompt"],
        "template": DEFAULT_TEMPLATE,
    }
    data = datasets.load_dataset(args.dataset_name, args.length)
    task_data = data[args.task]
    n = len(task_data) if args.limit <= 0 else min(len(task_data), args.limit)
    print(f"[dump_slot_usage] streaming {n} samples of {args.task}/{args.length}")

    rows = []
    for i in range(n):
        s = task_data[i]
        text = get_formatted_input(
            s["input"], s["question"], prompt_cfg["examples"],
            prompt_cfg["instruction"], prompt_cfg["post_prompt"],
            template=prompt_cfg["template"],
        )
        ids = tok.encode(text, add_special_tokens=True, return_tensors="pt")
        if isinstance(ids, list):
            ids = torch.tensor([ids], dtype=torch.long)
        ids = ids.to(device)
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            u = stream_and_read_usage(model, ids, args.chunk_size, device)
        if u is None:
            print("[dump_slot_usage] ERROR: no _cum_usage accumulator found "
                  "(disable_l1_inject? wrong config?). Aborting.")
            sys.exit(2)
        rows.append(u)
        if (i + 1) % 5 == 0 or i == n - 1:
            print(f"  [{i+1}/{n}] tokens={ids.shape[1]} "
                  f"dead={(u==0).sum()} max={u.max():.0f}")

    usage_mat = np.stack(rows, axis=0)  # [n, num_slots]
    os.makedirs(os.path.dirname(args.out_base), exist_ok=True)
    np.save(args.out_base + ".npy", usage_mat)
    print(f"[dump_slot_usage] saved raw matrix {args.out_base}.npy "
          f"shape={usage_mat.shape}")

    mean_usage, dead, top16_frac = make_plot(
        usage_mat, top_k, num_slots, args.task, args.length, n, args.out_base
    )

    # ---- text statistics ----
    sorted_mean = np.sort(mean_usage)[::-1]
    total = mean_usage.sum()
    # per-sample dead count (avg of how many slots stay 0 within a single sample)
    per_sample_dead = (usage_mat == 0).sum(axis=1)
    print("\n" + "=" * 64)
    print(f"SLOT USAGE DISTRIBUTION  ({args.task} {args.length}, n={n})")
    print("=" * 64)
    print(f"num_slots={num_slots}  top_k={top_k}  chunk_size={args.chunk_size}")
    print(f"-- aggregated over all {n} samples (mean per-slot selections):")
    print(f"  dead slots (mean usage == 0)        : {dead}/{num_slots}")
    print(f"  max per-slot selections (mean)      : {sorted_mean[0]:.2f}")
    print(f"  median                              : {np.median(mean_usage):.2f}")
    for q in (10, 25, 50, 75, 90):
        print(f"  p{q:<2d}                                : "
              f"{np.percentile(mean_usage, q):.2f}")
    print(f"  top-{top_k} hotspot share of all selections : "
          f"{top16_frac*100:.1f}%")
    print(f"  (uniform baseline = {top_k}/{num_slots} = "
          f"{top_k/num_slots*100:.1f}%)")
    print(f"-- per-sample dead-slot count (within one 32k sample):")
    print(f"  mean={per_sample_dead.mean():.1f}  min={per_sample_dead.min()} "
          f"max={per_sample_dead.max()}  (of {num_slots})")
    print("=" * 64)


if __name__ == "__main__":
    main()
