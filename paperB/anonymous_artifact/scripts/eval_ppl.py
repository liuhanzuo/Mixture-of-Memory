#!/usr/bin/env python3
"""Held-out PPL eval driver for the OLMo-2 prune-then-heal probe (Paper B).

Self-contained (does NOT import scripts/train_olmo2_arch_probe2.py, to avoid
version drift): the pruned-arch construction logic below is copied verbatim from
that trainer's build_olmo2_minimal / _save contract (verified 2026-07-19).

Two modes
---------
* Full-model base (no --ckpt, no --keep_front_layers): load the pretrained
  OLMo-2 base with Olmo2ForCausalLM.from_pretrained and score held-out val PPL.
  This is the "full-depth" denominator for the prune-then-heal comparison.
* Pruned prune-then-heal ckpt (--ckpt path): rebuild the (keep_front+n_fresh)-layer
  OLMo-2 shell exactly as the trainer does (Olmo2Config.from_pretrained(base) ->
  cfg.num_hidden_layers = keep+fresh -> Olmo2ForCausalLM(cfg)) then strict-load the
  trained state_dict from the .pt. keep_front/n_fresh are read from the ckpt meta
  when present (falls back to CLI args); a mismatch vs CLI args is a hard error.

Scoring
-------
Teacher-forced next-token CE over each 2048-token window: forward under bf16
autocast (fp32 weights, matching the trainer's fp32-master-weight + bf16-autocast
setup), shift logits[:, :-1] against input_ids[:, 1:], accumulate a token-level
sum of NLL (reduction='sum', in fp32) plus token count. 8-GPU sharding is
process-per-GPU: window set is strided ``windows[shard_index::num_shards]``; each
shard writes its own json. --merge then token-weight-combines the shards
(ppl = exp(sum_nll / sum_tokens) -- NEVER a plain average of per-shard ppl).

OLMo-2 layout (POST-norm, NO input_layernorm; QK-norm; untied embeddings) is
irrelevant to eval because we load the full trained state_dict, but the config
shrink must match the trainer so the shell has the identical layer count.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Olmo2Config, Olmo2ForCausalLM


def _log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# pruned-arch construction (copied from train_olmo2_arch_probe2.build_olmo2_minimal)
# ---------------------------------------------------------------------------
def build_pruned_shell(base_path, keep_front_layers, n_fresh_layers, dtype):
    """Rebuild the (keep_front + n_fresh)-layer OLMo-2 shell exactly like the
    trainer: shrink the pretrained config's num_hidden_layers, then instantiate
    Olmo2ForCausalLM(cfg). No transplant here -- the caller strict-loads the full
    trained state_dict, which already contains every (front + fresh) layer."""
    cfg = Olmo2Config.from_pretrained(base_path, local_files_only=True)
    total_layers = keep_front_layers + n_fresh_layers
    cfg.num_hidden_layers = total_layers
    if getattr(cfg, "layer_types", None) is not None:
        cfg.layer_types = ["full_attention"] * total_layers
        assert len(cfg.layer_types) == total_layers
    model = Olmo2ForCausalLM(cfg).to(dtype)
    return model, cfg


def load_pruned_model(ckpt_path, base_path, keep_front_layers, n_fresh_layers, device):
    """Load a prune-then-heal ckpt. Reads keep/fresh from the ckpt meta when
    present (CLI args, if given, must match). fp32 master weights (matches the
    trainer); forward runs under bf16 autocast at eval time."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or "model_state" not in ck:
        raise ValueError(
            f"ckpt {ckpt_path} not in expected {{'model_state': ...}} format; "
            f"got type={type(ck)} keys={list(ck.keys())[:6] if isinstance(ck, dict) else '-'}"
        )
    sd = ck["model_state"]

    # prefer ckpt-recorded arch meta; CLI args (if provided) must agree.
    ck_keep = ck.get("keep_front_layers")
    ck_fresh = ck.get("n_fresh_layers")
    if ck_keep is not None:
        if keep_front_layers is not None and int(keep_front_layers) != int(ck_keep):
            raise ValueError(
                f"--keep_front_layers={keep_front_layers} != ckpt meta {ck_keep}"
            )
        keep_front_layers = int(ck_keep)
    if ck_fresh is not None:
        if n_fresh_layers is not None and int(n_fresh_layers) != int(ck_fresh):
            raise ValueError(
                f"--n_fresh_layers={n_fresh_layers} != ckpt meta {ck_fresh}"
            )
        n_fresh_layers = int(ck_fresh)
    if keep_front_layers is None:
        raise ValueError("keep_front_layers unknown (not in ckpt meta, none passed)")
    if n_fresh_layers is None:
        n_fresh_layers = 2

    model, cfg = build_pruned_shell(base_path, keep_front_layers, n_fresh_layers,
                                    torch.float32)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, (
        f"state_dict mismatch: missing={missing[:6]} unexpected={unexpected[:6]}"
    )
    step = ck.get("step")
    _log(f"[pruned] loaded ckpt step={step} keep_front={keep_front_layers} "
         f"n_fresh={n_fresh_layers} num_hidden_layers={cfg.num_hidden_layers} "
         f"({len(sd)} tensors, strict) from {ckpt_path}")
    model = model.to(device)
    model.eval()
    return model, {
        "mode": "pruned",
        "keep_front_layers": keep_front_layers,
        "n_fresh_layers": n_fresh_layers,
        "num_hidden_layers": cfg.num_hidden_layers,
        "ckpt_step": step,
        "ckpt": ckpt_path,
    }


def load_base_model(base_path, device):
    """Full-depth pretrained OLMo-2 (the denominator). fp32 weights + bf16
    autocast forward, matching the pruned-model path for a fair comparison."""
    model = Olmo2ForCausalLM.from_pretrained(
        base_path, torch_dtype=torch.float32, local_files_only=True
    )
    n_layers = model.config.num_hidden_layers
    _log(f"[base] loaded full-depth base {base_path} "
         f"num_hidden_layers={n_layers} vocab={model.config.vocab_size}")
    model = model.to(device)
    model.eval()
    return model, {
        "mode": "base",
        "num_hidden_layers": n_layers,
        "base_model": base_path,
    }


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
@torch.no_grad()
def score_windows(model, windows, device, batch_size):
    """Teacher-forced NTP CE over [n, seq_len] int token windows. Returns
    (sum_nll, n_tokens, n_windows). bf16 autocast forward; fp32 reduction='sum'
    CE over shifted (predict pos 1..T-1) targets -- no padding, all tokens valid."""
    sum_nll = 0.0
    n_tokens = 0
    n_windows = 0
    n = windows.shape[0]
    for i in range(0, n, batch_size):
        chunk = windows[i:i + batch_size]
        input_ids = torch.from_numpy(chunk.astype(np.int64)).to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=input_ids)
        logits = out.logits[:, :-1, :].float()          # [B, T-1, V]
        targets = input_ids[:, 1:].contiguous()         # [B, T-1]
        V = logits.shape[-1]
        loss = F.cross_entropy(
            logits.reshape(-1, V), targets.reshape(-1), reduction="sum"
        )
        sum_nll += float(loss.item())
        n_tokens += int(targets.numel())
        n_windows += int(chunk.shape[0])
    return sum_nll, n_tokens, n_windows


def merge_shards(results_dir):
    """Token-weighted merge of shard{i}of{N}.json -> summary.json. ppl is
    exp(sum_nll / sum_tokens), NOT a mean of per-shard ppl."""
    shard_files = sorted(glob.glob(os.path.join(results_dir, "shard*of*.json")))
    if not shard_files:
        raise FileNotFoundError(f"no shard*of*.json in {results_dir}")
    parsed = []
    for sf in shard_files:
        match = re.fullmatch(r"shard(\d+)of(\d+)\.json", os.path.basename(sf))
        if match:
            parsed.append((int(match.group(1)), int(match.group(2)), sf))
    if len(parsed) != len(shard_files):
        raise ValueError(f"unexpected shard filename(s) in {results_dir}")
    totals = {total for _, total, _ in parsed}
    if len(totals) != 1:
        raise ValueError(f"inconsistent num_shards in {results_dir}: {sorted(totals)}")
    expected_total = totals.pop()
    indices = sorted(index for index, _, _ in parsed)
    if indices != list(range(expected_total)):
        raise ValueError(
            f"incomplete shards in {results_dir}: got {indices}, expected 0..{expected_total - 1}"
        )

    tot_nll = 0.0
    tot_tok = 0
    tot_win = 0
    meta = None
    for _, _, sf in parsed:
        with open(sf) as f:
            d = json.load(f)
        tot_nll += float(d["sum_nll"])
        tot_tok += int(d["n_tokens"])
        tot_win += int(d["n_windows"])
        if meta is None:
            meta = d.get("meta")
        elif d.get("meta") != meta:
            raise ValueError(f"metadata mismatch across shards in {results_dir}")
    if tot_tok <= 0:
        raise ValueError(f"merged n_tokens={tot_tok} <= 0 (bad shards in {results_dir})")
    ppl = math.exp(tot_nll / tot_tok)
    summary = {
        "output_name": os.path.basename(results_dir.rstrip("/")),
        "n_shards": expected_total,
        "sum_nll": tot_nll,
        "n_tokens": tot_tok,
        "n_windows": tot_win,
        "ppl": ppl,
        "avg_nll": tot_nll / tot_tok,
        "meta": meta,
        "shard_files": [os.path.basename(sf) for _, _, sf in parsed],
    }
    out = os.path.join(results_dir, "summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    _log(f"[merge] {expected_total} shards | n_windows={tot_win} "
         f"n_tokens={tot_tok} | PPL={ppl:.4f} avg_nll={tot_nll/tot_tok:.4f} -> {out}")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", type=str, required=False,
                   help="pretrained OLMo-2 path (cfg source for pruned mode; "
                        "the full model itself in base mode)")
    p.add_argument("--ckpt", type=str, default="",
                   help="prune-then-heal .pt (omit -> full-model base mode)")
    p.add_argument("--keep_front_layers", type=int, default=None,
                   help="pruned mode; default read from ckpt meta")
    p.add_argument("--n_fresh_layers", type=int, default=None,
                   help="pruned mode; default read from ckpt meta (else 2)")
    p.add_argument("--val_path", type=str, default="data/dolmino_now_val.npy")
    p.add_argument("--limit", type=int, default=0,
                   help=">0 caps windows scored by THIS shard (post-striding); sanity only")
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--output_name", type=str, required=False,
                   help="results dir = olmo2_ppl_results/<output_name>/")
    p.add_argument("--results_root", type=str, default="olmo2_ppl_results")
    p.add_argument("--merge", action="store_true",
                   help="merge shard jsons in olmo2_ppl_results/<output_name>/ and exit")
    args = p.parse_args()

    if args.merge:
        if not args.output_name:
            raise ValueError("--merge requires --output_name")
        merge_shards(os.path.join(args.results_root, args.output_name))
        return

    if not args.output_name:
        raise ValueError("--output_name required")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError(
            f"invalid shard_index={args.shard_index} for num_shards={args.num_shards}"
        )
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")

    # ---- load full val windows into RAM (small; avoid ceph mmap random reads) ----
    arr = np.load(args.val_path, mmap_mode="r")
    windows_all = np.array(arr)  # materialise fully into memory
    assert windows_all.ndim == 2, windows_all.shape
    n_total = windows_all.shape[0]
    idx = np.arange(args.shard_index, n_total, args.num_shards)
    shard_windows = windows_all[idx]
    if args.limit and args.limit > 0:
        shard_windows = shard_windows[: args.limit]
    _log(f"val={args.val_path} shape={windows_all.shape} | shard "
         f"{args.shard_index}/{args.num_shards} -> {shard_windows.shape[0]} windows "
         f"(seq_len={windows_all.shape[1]}) batch_size={args.batch_size}")

    # ---- build / load model ----
    if args.ckpt:
        model, meta = load_pruned_model(
            args.ckpt, args.base_model, args.keep_front_layers,
            args.n_fresh_layers, device)
    else:
        if not args.base_model:
            raise ValueError("base mode requires --base_model")
        model, meta = load_base_model(args.base_model, device)
    meta["base_model"] = args.base_model
    meta["val_path"] = args.val_path

    # ---- score ----
    t0 = time.time()
    sum_nll, n_tokens, n_windows = score_windows(
        model, shard_windows, device, args.batch_size)
    dt = time.time() - t0
    assert n_tokens > 0, f"n_tokens={n_tokens} (empty shard?)"
    ppl_shard = math.exp(sum_nll / n_tokens)
    assert math.isfinite(ppl_shard), (
        f"ppl_shard={ppl_shard} non-finite (sum_nll={sum_nll} n_tokens={n_tokens}) "
        f"-> check dtype / ckpt key mismatch"
    )

    results_dir = os.path.join(args.results_root, args.output_name)
    os.makedirs(results_dir, exist_ok=True)
    out = os.path.join(results_dir, f"shard{args.shard_index}of{args.num_shards}.json")
    payload = {
        "sum_nll": sum_nll,
        "n_tokens": n_tokens,
        "n_windows": n_windows,
        "ppl_shard": ppl_shard,
        "avg_nll": sum_nll / n_tokens,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "seconds": dt,
        "meta": meta,
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    _log(f"[shard {args.shard_index}/{args.num_shards}] n_windows={n_windows} "
         f"n_tokens={n_tokens} ppl_shard={ppl_shard:.4f} ({dt:.1f}s) -> {out}")


if __name__ == "__main__":
    main()
