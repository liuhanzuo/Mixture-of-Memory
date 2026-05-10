#!/usr/bin/env python3
"""Q-Filters **streaming** PPL eval at long sequence lengths (≥ 32k).

Companion to `scripts/eval_qfilters.py` (which is a *chunked* eval — each 4096-
token chunk is scored as an independent document with its own fresh cache).
This driver instead processes a single long document as a continuous stream:

    ┌───────────────── document (L tokens, e.g. 32768) ─────────────────┐
    │ sub-window 0 │ sub-window 1 │ ... │ sub-window K-1 │
       cache ═════ carries over ═════════════════════════════════════════╗
                       (compression hook fires after each sub-window)   ║

Key behavior vs. the chunked driver:
    * **Never resets cache** between sub-windows of the same document. The
      `QFiltersCache` post-forward hook fires after every sub-window; when
      `len(cache) > kv_budget` it compresses down to `kv_budget` tokens. This
      is how the cache stays bounded even as L grows to 32k / 65k / 128k.
    * **Per-position PPL curve.** We compute per-token losses manually
      (instead of relying on HF's averaged `loss`) so we can bucket by
      absolute position in the document. The main output is PPL vs. position,
      which is the right thing to look at for a streaming long-context eval —
      it tells you whether compression degrades LM quality as the prefix
      grows past the training window.
    * **Document source:** we do NOT require a new .npy. Instead we derive
      long streams by concatenating contiguous 4096-token chunks from the
      existing `data/pg19_chunks_llama3_noeos.npy` (default). For `L=32768`
      we concat 8 consecutive chunks; for `L=65536`, 16 chunks. See
      `--stream_length` and `--num_streams`.

Distribution (torchrun 8 GPU):
    Documents are sharded across ranks (rank k takes every `world_size`-th
    stream). Each rank independently processes its streams, accumulates
    per-bucket loss/token counters on GPU, and we all-reduce at the end.

RoPE caveat (read before trusting the numbers):
    In `qfilters` and `sliding_window` modes, Patch A re-rotates preserved
    keys to new positions [0, kv_budget), so the cache physical length is
    always ≤ `kv_budget + sub_window_len` regardless of document length — RoPE
    positions stay well inside the model's trained range. **No naïve
    extrapolation in these two modes.**
    In `dense` mode (no compression), positions grow with the document and
    *will* exceed the training window (Llama-3-8B trained at 8192) once
    `stream_length > 8192`. We still support the mode for completeness (short
    streams / sanity checks) but it naively extrapolates RoPE. Do NOT cite a
    dense-mode number past the training window as a fair baseline.

Smoke contract (1 GPU, 1 stream × 32k, ~2 min on H20):
    python scripts/eval_qfilters_streaming.py \\
        --model models/Llama--Llama3-8b \\
        --data data/pg19_chunks_llama3_noeos.npy \\
        --filters_cache outputs/rank1_kv_ext_llama3/qf_r1_b1024_rw64/filters.pt \\
        --stream_length 32768 --num_streams 1 \\
        --kv_budget 512 --filter_rank 1 --recent_window 64 \\
        --calibration_chunks 64 --sub_window_len 1024 \\
        --output_dir outputs/streaming_smoke --single_gpu --bf16

Full eval (8 GPU, 16 streams × 32k, ~10 min est on 8× L20A):
    torchrun --nproc_per_node=8 scripts/eval_qfilters_streaming.py \\
        --model /…/Llama--Llama3-8b \\
        --data data/pg19_chunks_llama3_noeos.npy \\
        --filters_cache outputs/rank1_kv_ext_llama3/qf_r1_b1024_rw64/filters.pt \\
        --stream_length 32768 --num_streams 16 \\
        --kv_budget 512 --filter_rank 1 --recent_window 64 \\
        --calibration_chunks 64 --sub_window_len 1024 \\
        --bucket_tokens 2048 --warmup_tokens 4096 \\
        --mode qfilters --output_dir outputs/streaming_llama3_r1_b512_32k --bf16

Output (`eval_results.json`):
    {
      "ppl":              <aggregate PPL after dropping warmup>,
      "ppl_raw":          <aggregate PPL over all scored positions>,
      "ppl_by_bucket":    [{"start": 0, "end": 2048, "ppl": …, "tokens": …}, …],
      "num_tokens":       <tokens scored across ranks>,
      "num_streams":      <requested number of streams>,
      "stream_length":    …,
      "bucket_tokens":    …,
      "warmup_tokens":    …,
      "mode":             "qfilters" | "sliding_window" | "dense",
      "kv_budget":        …,
      "filter_rank":      …,
      "recent_window":    …,
      "sub_window_len":   …,
      "wall_time_sec":    …,
      "tokens_per_sec":   …,
      ...
    }
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from typing import Iterable, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.memory.qfilters import (
    QFiltersConfig,
    compute_filters,
    patch_model,
)
from src.memory.qfilters.layer import make_qfilters_cache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data — build long streams by concatenating contiguous chunks
# --------------------------------------------------------------------------- #


def build_streams(
    npy_path: str,
    stream_length: int,
    num_streams: int,
    skip_chunks: int,
) -> np.ndarray:
    """Return a [num_streams, stream_length] int64 array.

    Source: `npy_path` is the existing 4096-token chunked corpus
    (e.g. pg19_chunks_llama3_noeos.npy, shape [N, 4096]).
    Stream k concatenates `(stream_length // 4096)` consecutive chunks starting
    at `skip_chunks + k * chunks_per_stream`. This keeps documents disjoint
    across streams and disjoint from the head-of-file calibration shard.

    Chunks on disk are uint32 (Llama-3 128k-vocab) or uint16 (Llama-2 32k-vocab);
    we upcast to int64 for torch indexing.
    """
    data = np.load(npy_path, mmap_mode="r")
    n_chunks_total, chunk_len = data.shape
    if stream_length % chunk_len != 0:
        raise ValueError(
            f"stream_length={stream_length} must be a multiple of "
            f"on-disk chunk_len={chunk_len}"
        )
    chunks_per_stream = stream_length // chunk_len
    need = skip_chunks + num_streams * chunks_per_stream
    if need > n_chunks_total:
        raise ValueError(
            f"Not enough chunks: need {need} (skip={skip_chunks} + "
            f"{num_streams} streams × {chunks_per_stream} chunks/stream), "
            f"have {n_chunks_total} in {npy_path}"
        )
    streams = np.empty((num_streams, stream_length), dtype=np.int64)
    for k in range(num_streams):
        base = skip_chunks + k * chunks_per_stream
        flat = data[base : base + chunks_per_stream].reshape(-1).astype(np.int64)
        streams[k] = flat
    logger.info(
        "Built %d streams of %d tokens each from %s (chunks_per_stream=%d, "
        "skip_chunks=%d, n_chunks_total=%d)",
        num_streams, stream_length, npy_path, chunks_per_stream,
        skip_chunks, n_chunks_total,
    )
    return streams


class CalibIterable:
    """Head-of-file calibration loader. Mirror of eval_qfilters.py's version."""

    def __init__(self, npy_path: str, seq_length: int, n_chunks: int) -> None:
        data = np.load(npy_path, mmap_mode="r")
        take = data[:n_chunks].astype(np.int32)
        self.items = [
            torch.tensor(take[i], dtype=torch.long)[:seq_length]
            for i in range(n_chunks)
        ]
        logger.info(
            "Calibration loader: %d chunks of up to %d tokens from head of %s",
            n_chunks, seq_length, npy_path,
        )

    def __iter__(self) -> Iterable:
        for t in self.items:
            yield {"input_ids": t.unsqueeze(0)}

    def __len__(self) -> int:
        return len(self.items)


# --------------------------------------------------------------------------- #
# Distributed helpers (copy of eval_qfilters.py's)
# --------------------------------------------------------------------------- #


def init_distributed(single_gpu: bool) -> Tuple[int, int, int]:
    """Return (rank, world_size, local_rank). world_size==1 if not distributed."""
    if single_gpu or "RANK" not in os.environ:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def is_main(rank: int) -> bool:
    return rank == 0


# --------------------------------------------------------------------------- #
# Streaming core
# --------------------------------------------------------------------------- #


@torch.no_grad()
def stream_one_document(
    model,
    doc_ids: torch.Tensor,           # [1, L] long
    device: torch.device,
    sub_window_len: int,
    bucket_tokens: int,
    num_buckets: int,
    mode: str,                        # "qfilters" | "sliding_window" | "dense"
    bucket_loss_sum: torch.Tensor,    # [num_buckets] float64 on device (accum)
    bucket_tokens_sum: torch.Tensor,  # [num_buckets] float64 on device (accum)
) -> None:
    """Process ONE document as a single stream, accumulating per-bucket stats.

    Streaming mechanic:
        * One `QFiltersCache` lives for the full document (all sub-windows
          share it). The cache's post-forward hook compresses it to
          `kv_budget` entries after each layer's forward, so sub-window k+1
          reads the compressed state left by sub-window k.
        * For `mode == "dense"` we allocate a plain `DynamicCache` instead
          and never compress. Expect PPL blowup past the model's training
          window; see module docstring.

    Per-token loss:
        We compute raw logits (labels=None) and then, inside each sub-window,
            loss[t] = CE(logits[:, t], sw_input[:, t+1])   for t ∈ [0, Ts-2]
        which predicts absolute position `start + t + 1` in the stream. loss
        is bucketed by absolute predicted position. We drop the very last
        logit of each sub-window (which would predict the first token of the
        *next* sub-window) because its label isn't known locally — this
        drops positions {sw_len, 2·sw_len, …, (K-1)·sw_len} plus position 0,
        about K/L ≈ 0.1% of tokens at sw_len=1024 / L=32k. Matches the
        chunked driver's known sub-window boundary accounting; negligible
        for PPL.

    Notes on label alignment:
        HF's `LlamaForCausalLM.forward(labels=...)` internally shifts
        (logits[:-1] / labels[1:]). We deliberately pass `labels=None` and
        do the shift ourselves so we can produce un-reduced per-token losses
        without double-shifting (double-shift was the 2026-04-25 PPL blowup
        bug on Llama-3). See eval_qfilters.py §dataset docstring.
    """
    if mode == "dense":
        from transformers.cache_utils import DynamicCache
        cache = DynamicCache()
    else:
        cache = make_qfilters_cache(model)

    L = doc_ids.size(1)
    sw_len = sub_window_len if sub_window_len > 0 else L

    for start in range(0, L, sw_len):
        end = min(start + sw_len, L)
        sw_input = doc_ids[:, start:end].to(device, non_blocking=True)

        outputs = model(
            input_ids=sw_input,
            past_key_values=cache,
            use_cache=True,
        )
        logits = outputs.logits  # [1, Ts, V]
        Ts = logits.size(1)
        if Ts < 2:
            continue

        # Per-token CE: predict position `start + t + 1` from logits[t].
        # t ∈ [0, Ts-2] → absolute position predicted = start + t + 1.
        shift_logits = logits[:, :-1, :].contiguous().float()  # [1, Ts-1, V]
        shift_labels = sw_input[:, 1:].contiguous()             # [1, Ts-1]
        loss_per_tok = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        )  # [Ts-1]

        # Guard against non-finite losses (shouldn't happen in bf16 sdpa with
        # fp32 CE, but we saw sporadic inf in earlier harness iterations).
        finite = torch.isfinite(loss_per_tok)
        if not finite.all():
            n_bad = int((~finite).sum().item())
            logger.warning(
                "Non-finite per-token loss count=%d at stream start=%d (keeping finite only)",
                n_bad, start,
            )
            loss_per_tok = torch.where(
                finite, loss_per_tok, loss_per_tok.new_zeros(()),
            )

        # Bucket by absolute predicted position.
        # position[t] = start + t + 1
        positions = torch.arange(
            start + 1, start + Ts, device=device, dtype=torch.long
        )
        bucket_idx = (positions // bucket_tokens).clamp(max=num_buckets - 1)

        # Scatter-add loss and count into buckets.
        contribution = loss_per_tok.double()
        if not finite.all():
            # Mask out non-finite positions from the token count too.
            token_mask = finite.double()
        else:
            token_mask = torch.ones_like(contribution)

        bucket_loss_sum.scatter_add_(0, bucket_idx, contribution * token_mask)
        bucket_tokens_sum.scatter_add_(0, bucket_idx, token_mask)


# --------------------------------------------------------------------------- #
# Args
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Q-Filters streaming PPL eval at seq_length ≥ 32k."
    )
    # Model / data
    p.add_argument("--model", type=str, required=True,
                   help="HF model dir (absolute or project-relative).")
    p.add_argument("--data", type=str, required=True,
                   help="Pre-chunked .npy (e.g. data/pg19_chunks_llama3_noeos.npy, [N, 4096]).")
    p.add_argument("--stream_length", type=int, default=32768,
                   help="Total tokens per stream. Must be a multiple of on-disk chunk length.")
    p.add_argument("--num_streams", type=int, default=16,
                   help="How many independent long documents to score.")
    p.add_argument("--skip_chunks", type=int, default=200,
                   help="Skip this many head-of-file chunks (avoids calibration overlap).")
    # Compression config
    p.add_argument("--kv_budget", type=int, default=512)
    p.add_argument("--filter_rank", type=int, default=1,
                   help="Default 1 since §11.4.3 shows rank=1 dominant on Llama-3.")
    p.add_argument("--recent_window", type=int, default=64)
    p.add_argument("--calibration_chunks", type=int, default=64)
    p.add_argument("--sub_window_len", type=int, default=1024,
                   help="Length of each streaming sub-window. Cache carries over "
                        "across sub-windows within a document.")
    # Bucketing
    p.add_argument("--bucket_tokens", type=int, default=2048,
                   help="Size of each position bucket for the PPL-vs-position curve.")
    p.add_argument("--warmup_tokens", type=int, default=4096,
                   help="Drop positions < this from the summary 'ppl' (kept in "
                        "'ppl_raw' and 'ppl_by_bucket').")
    # Mode / IO
    p.add_argument("--mode", type=str, default="qfilters",
                   choices=["qfilters", "sliding_window", "dense"])
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--filters_cache", type=str, default=None,
                   help="Optional path to load/save calibration filters.pt. "
                        "Rank-0 writes; other ranks read.")
    p.add_argument("--single_gpu", action="store_true")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--attn_impl", type=str, default="sdpa",
                   choices=["sdpa", "eager", "flash_attention_2"])
    # Calibration uses this seq_length (independent of stream_length)
    p.add_argument("--calib_seq_length", type=int, default=4096,
                   help="Seq length for head-of-file calibration chunks.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = init_distributed(args.single_gpu)
    is_dist = world_size > 1

    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    if is_main(rank):
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(
            "Q-Filters STREAMING eval | model=%s | mode=%s | "
            "stream_length=%d num_streams=%d sub_window_len=%d | "
            "kv_budget=%d filter_rank=%d recent_window=%d | world_size=%d",
            args.model, args.mode, args.stream_length, args.num_streams,
            args.sub_window_len, args.kv_budget, args.filter_rank,
            args.recent_window, world_size,
        )

    # ---- tokenizer ---- #
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- model ---- #
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation=args.attn_impl,
        device_map={"": device} if device.type == "cuda" else None,
    )
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)

    # ---- calibration / filters ---- #
    qcfg = QFiltersConfig(
        kv_budget=args.kv_budget,
        filter_rank=args.filter_rank,
        recent_window=args.recent_window,
        calibration_chunks=args.calibration_chunks,
    )

    if args.mode in ("sliding_window", "dense"):
        if is_main(rank):
            logger.info(
                "Mode=%s: skipping calibration and filter I/O.", args.mode,
            )
        filters: dict = {}
    else:
        filters = None
        cache_path = args.filters_cache or os.path.join(args.output_dir, "filters.pt")

        if is_main(rank):
            if os.path.exists(cache_path):
                logger.info("Loading cached filters from %s", cache_path)
                filters = torch.load(cache_path, map_location="cpu")
            else:
                logger.info(
                    "Running calibration (chunks=%d, rank=%d)...",
                    qcfg.calibration_chunks, qcfg.filter_rank,
                )
                calib = CalibIterable(
                    args.data, args.calib_seq_length, qcfg.calibration_chunks,
                )
                num_kv_heads = getattr(
                    model.config, "num_key_value_heads",
                    model.config.num_attention_heads,
                )
                filters = compute_filters(
                    model=model,
                    calib_loader=calib,
                    rank=qcfg.filter_rank,
                    num_kv_heads=num_kv_heads,
                    device=device,
                )
                torch.save(filters, cache_path)
                logger.info("Saved filters to %s (%d layers)", cache_path, len(filters))

        if is_dist:
            dist.barrier()
            if not is_main(rank):
                filters = torch.load(cache_path, map_location="cpu")

    # ---- patch (only needed for qfilters / sliding_window) ---- #
    if args.mode != "dense":
        patch_model(model, filters, qcfg)

    # ---- build streams ---- #
    # Main builds the shape once so we can log; every rank actually reloads
    # (cheap — it's mmap'd). We slice by rank below.
    streams_np = build_streams(
        args.data,
        stream_length=args.stream_length,
        num_streams=args.num_streams,
        skip_chunks=args.skip_chunks,
    )
    # Shard streams across ranks.
    my_stream_ids = list(range(rank, args.num_streams, world_size))
    if is_main(rank):
        logger.info(
            "Sharding %d streams across %d ranks; rank 0 takes %d streams",
            args.num_streams, world_size, len(my_stream_ids),
        )

    # ---- per-bucket accumulators ---- #
    num_buckets = math.ceil(args.stream_length / args.bucket_tokens)
    bucket_loss = torch.zeros(num_buckets, dtype=torch.float64, device=device)
    bucket_tok = torch.zeros(num_buckets, dtype=torch.float64, device=device)

    t0 = time.time()
    for idx, sid in enumerate(my_stream_ids):
        doc = torch.from_numpy(streams_np[sid]).long().unsqueeze(0)  # [1, L]
        stream_one_document(
            model=model,
            doc_ids=doc,
            device=device,
            sub_window_len=args.sub_window_len,
            bucket_tokens=args.bucket_tokens,
            num_buckets=num_buckets,
            mode=args.mode,
            bucket_loss_sum=bucket_loss,
            bucket_tokens_sum=bucket_tok,
        )
        if is_main(rank):
            logger.info(
                "  [rank 0] stream %d/%d (sid=%d) done in %.1fs",
                idx + 1, len(my_stream_ids), sid, time.time() - t0,
            )
    wall = time.time() - t0

    # ---- all-reduce ---- #
    if is_dist:
        dist.all_reduce(bucket_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(bucket_tok, op=dist.ReduceOp.SUM)

    # ---- summary ---- #
    if is_main(rank):
        # Per-bucket PPL.
        ppl_by_bucket: List[dict] = []
        warm = max(0, args.warmup_tokens // args.bucket_tokens)
        total_tokens = float(bucket_tok.sum().item())
        # All-positions PPL (raw)
        safe_tok = bucket_tok.clamp_min(1)
        avg_loss_raw = (bucket_loss.sum() / bucket_tok.sum().clamp_min(1)).item()
        ppl_raw = math.exp(avg_loss_raw) if total_tokens > 0 else float("nan")

        # Past-warmup PPL.
        if warm < num_buckets:
            post_loss = bucket_loss[warm:].sum()
            post_tok = bucket_tok[warm:].sum()
        else:
            post_loss = torch.zeros((), dtype=torch.float64, device=device)
            post_tok = torch.zeros((), dtype=torch.float64, device=device)
        post_tok_f = float(post_tok.item())
        avg_loss_post = (
            (post_loss / post_tok.clamp_min(1)).item() if post_tok_f > 0 else float("nan")
        )
        ppl = math.exp(avg_loss_post) if post_tok_f > 0 else float("nan")

        for b in range(num_buckets):
            n = float(bucket_tok[b].item())
            if n <= 0:
                ppl_b = float("nan")
                avg_b = float("nan")
            else:
                avg_b = float((bucket_loss[b] / safe_tok[b]).item())
                ppl_b = math.exp(avg_b)
            ppl_by_bucket.append({
                "start": b * args.bucket_tokens,
                "end": min((b + 1) * args.bucket_tokens, args.stream_length),
                "avg_loss": avg_b,
                "ppl": ppl_b,
                "tokens": int(n),
            })

        tokens_per_sec = (total_tokens / wall) if wall > 0 else float("nan")
        result = {
            "ppl": ppl,
            "ppl_raw": ppl_raw,
            "avg_loss": avg_loss_post,
            "avg_loss_raw": avg_loss_raw,
            "num_tokens": int(total_tokens),
            "num_streams": args.num_streams,
            "stream_length": args.stream_length,
            "bucket_tokens": args.bucket_tokens,
            "warmup_tokens": args.warmup_tokens,
            "ppl_by_bucket": ppl_by_bucket,
            "mode": args.mode,
            "kv_budget": args.kv_budget,
            "filter_rank": args.filter_rank,
            "recent_window": args.recent_window,
            "calibration_chunks": args.calibration_chunks,
            "sub_window_len": args.sub_window_len,
            "model": args.model,
            "attn_impl": args.attn_impl,
            "bf16": bool(args.bf16),
            "wall_time_sec": wall,
            "tokens_per_sec": tokens_per_sec,
            "world_size": world_size,
            "skip_chunks": args.skip_chunks,
        }

        out_path = os.path.join(args.output_dir, "eval_results.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        # Human-readable curve.
        logger.info(
            "RESULT: PPL=%.4f (past warmup=%d tokens) | PPL_raw=%.4f | "
            "tokens=%d | streams=%d × %d | mode=%s kv=%d rank=%d recent=%d | "
            "wall=%.1fs (%.1f tok/s) | -> %s",
            ppl, args.warmup_tokens, ppl_raw,
            int(total_tokens), args.num_streams, args.stream_length,
            args.mode, args.kv_budget, args.filter_rank, args.recent_window,
            wall, tokens_per_sec, out_path,
        )
        logger.info("PPL curve by bucket:")
        for row in ppl_by_bucket:
            logger.info(
                "  [%6d, %6d) tokens=%5d  ppl=%.4f  avg_loss=%.4f",
                row["start"], row["end"], row["tokens"],
                row["ppl"], row["avg_loss"],
            )

        # PPL red-line self-check (CLAUDE.md §PPL 级别洞察).
        if not math.isnan(ppl) and ppl > 100:
            logger.warning(
                "PPL > 100 may indicate model contamination (attention/RoPE/"
                "cache bug) — see CLAUDE.md §PPL 级别洞察. Before retuning "
                "hyperparameters, dispatch /researcher to investigate root "
                "cause. In 'dense' mode at stream_length > training_window, "
                "naive RoPE extrapolation can also cause this — check the "
                "mode flag first."
            )

        print(
            f"\nQ-Filters STREAMING PPL: {ppl:.4f} "
            f"(model={os.path.basename(args.model.rstrip('/'))}, "
            f"mode={args.mode}, kv_budget={args.kv_budget}, "
            f"rank={args.filter_rank}, stream_length={args.stream_length}, "
            f"num_streams={args.num_streams})"
        )

    if is_dist:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
