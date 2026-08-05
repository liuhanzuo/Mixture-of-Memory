#!/usr/bin/env python
"""Paper C P-C2 adaptation-onset probe.

Forward-only, no training. Given a base OLMo-2 and a fine-tuned checkpoint
(pruned/keep-front+fresh trainer produced by scripts/train_olmo2_arch_probe2.py,
or a full 32L HF folder for A2 = LoRA-merged), compute per-layer linear CKA
between the residual-stream hidden states of the two models on a held-out slice
of text.

Layer alignment:
  * Full 32L base is always the "reference".
  * Pruned models with num_hidden_layers = keep + fresh:
      - transformer output_hidden_states has length num_hidden_layers + 1
        (index 0 = embedding output; index k = residual stream AFTER layer k-1).
      - For k in [0 .. keep], ft.hidden_states[k] corresponds directly to
        base.hidden_states[k] (front `keep` layers were transplanted from base
        indices 0..keep-1). We compute CKA for these indices.
      - For k in [keep+1 .. keep+fresh] the ft residual passes through fresh
        randomly-initialised (or SFT-updated) layers with no direct base
        counterpart; we skip these.
      - A "post_norm" point (final RMSNorm output, right before lm_head) is
        also compared to base's post-final-norm point (=hidden_states[-1] on
        both sides but taken through their own final norms).
  * Full 32L ft (A2 merged, A1 full-FT): all 33 hidden-state indices (0..32)
    are compared to base.

CKA is linear centered kernel alignment (Kornblith et al. 2019), computed with
double precision on all-gathered flattened vectors. Cosine similarity mean and
||delta_H|| / ||H|| are computed as side channels.

Data slice: `data/squad_val.jsonl` -> concatenate each row's `memory_texts`
(joined by "\n\n"), tokenize with OLMo-2 tokenizer, shuffle rows with seed=42
(reproducible), pack into `--n_windows` windows of `--seq_len` tokens each.

Usage (DDP, one rank per GPU):
  torchrun --nproc_per_node=8 --master_port=29509 \
      scripts/probe_paperC_adaptation_onset.py \
      --base_path /path/to/OLMo-2-1124-7B \
      --ft_path outputs/paperC_pc1_squad_A4/final.pt \
      --ft_mode pruned  \
      --data_path data/squad_val.jsonl \
      --n_windows 512 --seq_len 2048 \
      --out_dir paperC_probe_results/onset_A4 \
      --tag A4_hero

Or ft_mode=hf_dir for A2 merged, ft_mode=base for self-check (base vs base).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoTokenizer, Olmo2Config, Olmo2ForCausalLM


def _log(rank: int, msg: str) -> None:
    if rank == 0:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# model loading (copied structurally from scripts/eval_olmo2_probe2_ppl.py)
# ---------------------------------------------------------------------------
def build_pruned_shell(base_path, keep_front_layers, n_fresh_layers, dtype):
    cfg = Olmo2Config.from_pretrained(base_path, local_files_only=True)
    total_layers = keep_front_layers + n_fresh_layers
    cfg.num_hidden_layers = total_layers
    if getattr(cfg, "layer_types", None) is not None:
        cfg.layer_types = ["full_attention"] * total_layers
        assert len(cfg.layer_types) == total_layers
    model = Olmo2ForCausalLM(cfg).to(dtype)
    return model, cfg


def load_ft_pruned(ckpt_path, base_path, dtype, rank):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or "model_state" not in ck:
        raise ValueError(
            f"ckpt {ckpt_path} not in expected format; keys={list(ck.keys())[:6]}"
        )
    sd = ck["model_state"]
    ck_keep = int(ck.get("keep_front_layers"))
    ck_fresh = int(ck.get("n_fresh_layers"))
    model, cfg = build_pruned_shell(base_path, ck_keep, ck_fresh, dtype)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, (
        f"state_dict mismatch: missing={missing[:6]} unexpected={unexpected[:6]}"
    )
    step = ck.get("step")
    _log(rank, f"[ft pruned] loaded ckpt step={step} keep={ck_keep} fresh={ck_fresh} "
               f"({len(sd)} tensors, strict) from {ckpt_path}")
    return model, {"mode": "pruned", "keep_front_layers": ck_keep,
                   "n_fresh_layers": ck_fresh,
                   "num_hidden_layers": cfg.num_hidden_layers,
                   "ckpt_step": step, "ckpt": ckpt_path}


def load_hf_dir(hf_dir, dtype, rank):
    model = Olmo2ForCausalLM.from_pretrained(
        hf_dir, torch_dtype=dtype, local_files_only=True
    )
    _log(rank, f"[ft hf_dir] loaded {hf_dir} "
               f"num_hidden_layers={model.config.num_hidden_layers}")
    return model, {"mode": "hf_dir", "num_hidden_layers": model.config.num_hidden_layers,
                   "ckpt": hf_dir}


def load_base(base_path, dtype, rank):
    model = Olmo2ForCausalLM.from_pretrained(
        base_path, torch_dtype=dtype, local_files_only=True
    )
    _log(rank, f"[base] loaded {base_path} num_hidden_layers={model.config.num_hidden_layers}")
    return model, {"mode": "base", "num_hidden_layers": model.config.num_hidden_layers,
                   "base_model": base_path}


# ---------------------------------------------------------------------------
# data: squad_val.jsonl -> [n_windows, seq_len] token windows
# ---------------------------------------------------------------------------
def prepare_windows(data_path, tok_path, n_windows, seq_len, seed, rank):
    tok = AutoTokenizer.from_pretrained(tok_path, local_files_only=True)
    rows = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    rng = random.Random(seed)
    rng.shuffle(rows)  # deterministic given seed

    windows: list[list[int]] = []
    buf: list[int] = []
    for r in rows:
        # concat all memory_texts to get raw passage text (context, not the Q)
        texts = r.get("memory_texts") or []
        if not texts:
            continue
        passage = "\n\n".join(str(t) for t in texts)
        ids = tok.encode(passage, add_special_tokens=False)
        buf.extend(ids)
        while len(buf) >= seq_len:
            windows.append(buf[:seq_len])
            buf = buf[seq_len:]
            if len(windows) >= n_windows:
                break
        if len(windows) >= n_windows:
            break
    if len(windows) < n_windows:
        raise RuntimeError(
            f"only produced {len(windows)} windows from {data_path}; need {n_windows}. "
            f"Try lowering --n_windows or --seq_len."
        )
    arr = np.asarray(windows[:n_windows], dtype=np.int64)
    _log(rank, f"[data] {arr.shape[0]} windows x {arr.shape[1]} tokens "
               f"(tokenizer={tok_path}, seed={seed})")
    return arr


# ---------------------------------------------------------------------------
# CKA: linear centered kernel alignment
# ---------------------------------------------------------------------------
def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """X: [N, D1] fp64, Y: [N, D2] fp64. Both already row-centered? We center
    here for safety. Returns HSIC(X,Y) / sqrt(HSIC(X,X) * HSIC(Y,Y)) which is
    equivalent to ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F) with centered X,Y."""
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    # linear CKA closed form (Kornblith 2019 eq. before eq. 6)
    #   CKA = ||Y^T X||_F^2 / ( ||X^T X||_F * ||Y^T Y||_F )
    cross = (Y.T @ X)          # [D2, D1]
    num = (cross ** 2).sum()
    xx = (X.T @ X)             # [D1, D1]
    yy = (Y.T @ Y)             # [D2, D2]
    denom = torch.sqrt((xx ** 2).sum() * (yy ** 2).sum())
    if denom.item() == 0.0:
        return float("nan")
    return float((num / denom).item())


def cos_mean(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Mean cosine over rows. X,Y in fp64 already."""
    xn = F.normalize(X, dim=1)
    yn = F.normalize(Y, dim=1)
    return float((xn * yn).sum(dim=1).mean().item())


def delta_norm_ratio(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Mean over rows of ||y-x||/||x||."""
    dn = (Y - X).norm(dim=1)
    xn = X.norm(dim=1).clamp_min(1e-9)
    return float((dn / xn).mean().item())


# ---------------------------------------------------------------------------
# forward + collect hidden state pool
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_hiddens(model, windows_local, device, batch_size, per_batch_subsample,
                    n_layers_expected):
    """Run model.forward(output_hidden_states=True) on windows on THIS rank and
    return a list of length (n_layers_expected+1) of torch.float64 tensors on
    CPU, each [N_local, D]. N_local = num_batches * per_batch_subsample. We
    subsample positions per batch to bound memory (2048 tokens * 4096 dim *
    batch=1 * 33 layers * fp32 = 1.1 GB per batch is fine; but pool grows so we
    keep only `per_batch_subsample` positions per batch)."""
    model.eval()
    n = windows_local.shape[0]
    pool = None  # list of accumulating tensors, filled on first batch

    gen = torch.Generator(device="cpu").manual_seed(20260805)  # deterministic subsample

    for i in range(0, n, batch_size):
        chunk = windows_local[i:i + batch_size]
        input_ids = torch.from_numpy(chunk).to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=input_ids, output_hidden_states=True,
                         use_cache=False)
        hs = out.hidden_states  # tuple length = num_hidden_layers+1
        assert len(hs) == n_layers_expected + 1, (
            f"expected {n_layers_expected+1} hidden states, got {len(hs)}"
        )
        B, T = chunk.shape
        # pick fixed positions (same across ranks & runs for this batch index)
        idx = torch.randperm(B * T, generator=gen)[:per_batch_subsample]
        if pool is None:
            pool = [[] for _ in range(len(hs))]
        for li, h in enumerate(hs):
            hf = h.float().reshape(B * T, -1).index_select(0, idx.to(h.device))
            pool[li].append(hf.to(torch.float64).cpu())
        del out, hs
    # concat per layer
    per_layer = [torch.cat(pl, dim=0) for pl in pool]
    return per_layer  # list of [N_local, D_layer] fp64 CPU


def gather_and_concat(t_cpu: torch.Tensor, world_size, device):
    """All-gather a [N_local, D] fp64 tensor across ranks, return [N_total, D]
    concatenated fp64 tensor on CPU (rank 0 gets the real result, others get
    same-shape tensor for simplicity)."""
    t = t_cpu.to(device)
    # ensure all ranks have same D
    D = torch.tensor([t.shape[1]], device=device)
    Ds = [torch.zeros_like(D) for _ in range(world_size)]
    dist.all_gather(Ds, D)
    assert all(int(x.item()) == int(D.item()) for x in Ds), f"D mismatch: {[int(x.item()) for x in Ds]}"
    N = torch.tensor([t.shape[0]], device=device)
    Ns = [torch.zeros_like(N) for _ in range(world_size)]
    dist.all_gather(Ns, N)
    max_n = int(max(int(x.item()) for x in Ns))
    pad = torch.zeros(max_n - int(N.item()), int(D.item()), dtype=t.dtype, device=device)
    padded = torch.cat([t, pad], dim=0)
    out_bufs = [torch.zeros_like(padded) for _ in range(world_size)]
    dist.all_gather(out_bufs, padded)
    trimmed = [out_bufs[r][:int(Ns[r].item())] for r in range(world_size)]
    result = torch.cat(trimmed, dim=0).cpu()
    return result


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_path", required=True, help="OLMo-2 base model dir")
    p.add_argument("--ft_path", required=True, help="ckpt .pt file OR HF dir")
    p.add_argument("--ft_mode", choices=["pruned", "hf_dir", "base"], required=True,
                   help="pruned=paperC A4/A3 keep+fresh .pt ; hf_dir=A2 LoRA-merged folder ; base=self (sanity)")
    p.add_argument("--data_path", required=True, help="squad_val.jsonl")
    p.add_argument("--tok_path", default=None,
                   help="tokenizer dir; default=base_path")
    p.add_argument("--n_windows", type=int, default=512)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--per_batch_subsample", type=int, default=16,
                   help="positions kept per batch (per rank)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--tag", required=True, help="label for this ft, e.g. A4_hero")
    args = p.parse_args()

    tok_path = args.tok_path or args.base_path

    # DDP init
    if dist.is_available() and "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    _log(rank, f"DDP world={world_size} rank={rank} local_rank={local_rank}")

    out_dir = Path(args.out_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    # ---- data ----
    all_windows = prepare_windows(args.data_path, tok_path, args.n_windows,
                                   args.seq_len, args.seed, rank)
    # shard along the sample dim
    shard = all_windows[rank::world_size]
    _log(rank, f"[data] rank {rank} owns {shard.shape[0]} windows")

    # ---- load base ----
    base_model, base_meta = load_base(args.base_path, torch.float32, rank)
    base_model.to(device)
    n_base_layers = base_meta["num_hidden_layers"]

    # ---- forward base, collect hidden pool ----
    _log(rank, "[fwd] base ...")
    t0 = time.time()
    base_pool = collect_hiddens(base_model, shard, device, args.batch_size,
                                 args.per_batch_subsample, n_base_layers)
    _log(rank, f"[fwd] base done in {time.time()-t0:.1f}s "
               f"(pool: {len(base_pool)} layers x [{base_pool[0].shape}])")
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    # ---- load ft ----
    if args.ft_mode == "pruned":
        ft_model, ft_meta = load_ft_pruned(args.ft_path, args.base_path,
                                             torch.float32, rank)
    elif args.ft_mode == "hf_dir":
        ft_model, ft_meta = load_hf_dir(args.ft_path, torch.float32, rank)
    elif args.ft_mode == "base":
        ft_model, ft_meta = load_base(args.base_path, torch.float32, rank)
    ft_model.to(device)
    n_ft_layers = ft_meta["num_hidden_layers"]

    _log(rank, "[fwd] ft ...")
    t0 = time.time()
    ft_pool = collect_hiddens(ft_model, shard, device, args.batch_size,
                              args.per_batch_subsample, n_ft_layers)
    _log(rank, f"[fwd] ft done in {time.time()-t0:.1f}s")
    del ft_model
    gc.collect()
    torch.cuda.empty_cache()

    # ---- all-gather each layer's pool ----
    _log(rank, "[gather] all-gather layer pools ...")
    base_full = [gather_and_concat(t, world_size, device) for t in base_pool]
    ft_full = [gather_and_concat(t, world_size, device) for t in ft_pool]

    if rank != 0:
        _log(rank, "[done] non-master exits")
        dist.destroy_process_group()
        return

    # ---- alignment map: which base index each ft hidden index corresponds to ----
    # hidden_states[k] for k=0..N is the residual AFTER passing through the
    # first k layers (k=0 = embedding output).
    # For pruned ft with keep=K, fresh=F: ft layers 0..K-1 == base layers 0..K-1.
    #   -> ft.hidden_states[k] and base.hidden_states[k] are directly comparable
    #      for k in [0..K]. k=K is "after K transplanted layers" which corresponds
    #      to base.hidden_states[K].
    #   -> k in [K+1 .. K+F] passes through fresh layers -> NOT compared to base.
    # For full 32L ft (A2/A1): compare all k in [0..32].
    pairs: list[tuple[str, int, int]] = []
    if args.ft_mode == "pruned":
        K = ft_meta["keep_front_layers"]
        F_ = ft_meta["n_fresh_layers"]
        for k in range(0, K + 1):  # 0..K inclusive
            pairs.append((f"L{k:02d}", k, k))
        for k in range(K + 1, K + F_ + 1):
            pairs.append((f"L{k:02d}_fresh", k, -1))  # -1 = no base match
    else:
        n = n_ft_layers
        assert n == n_base_layers, (
            f"non-pruned ft has {n} layers != base {n_base_layers}"
        )
        for k in range(0, n + 1):
            pairs.append((f"L{k:02d}", k, k))

    # ---- compute per-layer metrics ----
    per_layer_results = []
    _log(rank, "[cka] computing per-layer CKA / cos / dnorm ...")
    for name, ft_idx, base_idx in pairs:
        X_ft = ft_full[ft_idx]
        if base_idx < 0:
            cka = None
            cos = None
            dn = None
            n_used = int(X_ft.shape[0])
            dim_ft = int(X_ft.shape[1])
            dim_base = None
        else:
            X_base = base_full[base_idx]
            assert X_base.shape[0] == X_ft.shape[0], (
                f"row mismatch at {name}: base={X_base.shape} ft={X_ft.shape}"
            )
            cka = linear_cka(X_base, X_ft)
            if X_base.shape[1] == X_ft.shape[1]:
                cos = cos_mean(X_base, X_ft)
                dn = delta_norm_ratio(X_base, X_ft)
            else:
                cos = None
                dn = None
            n_used = int(X_base.shape[0])
            dim_ft = int(X_ft.shape[1])
            dim_base = int(X_base.shape[1])
        per_layer_results.append({
            "name": name,
            "ft_hidden_index": ft_idx,
            "base_hidden_index": base_idx,
            "n_vectors": n_used,
            "dim_ft": dim_ft,
            "dim_base": dim_base,
            "linear_cka": cka,
            "mean_cosine": cos,
            "mean_dnorm_ratio": dn,
        })
        _log(0, f"  {name} ft[{ft_idx}] vs base[{base_idx}] "
                f"CKA={cka if cka is None else f'{cka:.5f}'} "
                f"cos={cos if cos is None else f'{cos:.5f}'} "
                f"dnorm={dn if dn is None else f'{dn:.5f}'}")

    # ---- sanity flags ----
    sanity = {}
    if args.ft_mode == "base":
        # base-vs-base: every layer should have CKA >= 0.999
        min_cka = min(r["linear_cka"] for r in per_layer_results
                      if r["linear_cka"] is not None)
        sanity["base_vs_base_min_cka"] = min_cka
        sanity["base_vs_base_pass"] = bool(min_cka >= 0.999)
    if args.ft_mode == "pruned":
        K = ft_meta["keep_front_layers"]
        # transplanted layers 0..K-1 pre-FT are IDENTICAL to base; if front is
        # frozen the transplanted layers stay identical post-FT. But even for
        # freeze_front=True the hidden STATES at k=1..K depend on the same
        # weights (front frozen) AND the same embedding (also inherited & not
        # updated for keep32/A1... embedding IS `inherited`, at low LR, so it
        # can drift). For A4 (freeze_front=True), lr_inherited applies only to
        # trainable inherited params; front-block params are frozen. embed is
        # inherited and trainable -> can drift slightly. So we assert front
        # transplanted CKA is very high (>= 0.99) but not necessarily 1.0.
        front_ckas = [r["linear_cka"] for r in per_layer_results
                       if r["name"].startswith("L")
                       and r["base_hidden_index"] >= 0
                       and r["base_hidden_index"] <= K
                       and r["linear_cka"] is not None]
        min_front = min(front_ckas) if front_ckas else float("nan")
        sanity["pruned_front_min_cka"] = min_front
        sanity["pruned_front_threshold"] = 0.999  # user-requested strong check
        sanity["pruned_front_pass_0999"] = bool(min_front >= 0.999)
        sanity["pruned_front_pass_099"] = bool(min_front >= 0.99)

    result = {
        "tag": args.tag,
        "ft_mode": args.ft_mode,
        "base_meta": base_meta,
        "ft_meta": ft_meta,
        "config": {
            "n_windows": args.n_windows,
            "seq_len": args.seq_len,
            "seed": args.seed,
            "per_batch_subsample": args.per_batch_subsample,
            "world_size": world_size,
            "tok_path": tok_path,
            "data_path": args.data_path,
            "total_vectors_per_layer": int(base_full[0].shape[0]),
        },
        "per_layer": per_layer_results,
        "sanity": sanity,
    }

    with open(out_dir / "cka_per_layer.json", "w") as f:
        json.dump(result, f, indent=2)
    _log(0, f"[write] {out_dir/'cka_per_layer.json'}")

    # sanity-flag summary
    if args.ft_mode == "base":
        _log(0, f"[sanity] base-vs-base min CKA = {sanity['base_vs_base_min_cka']:.6f} "
                f"(pass>=0.999: {sanity['base_vs_base_pass']})")
    if args.ft_mode == "pruned":
        _log(0, f"[sanity] pruned front (indices 0..{ft_meta['keep_front_layers']}) "
                f"min CKA = {sanity['pruned_front_min_cka']:.6f} "
                f"(pass>=0.999: {sanity['pruned_front_pass_0999']}, "
                f"pass>=0.99: {sanity['pruned_front_pass_099']})")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
