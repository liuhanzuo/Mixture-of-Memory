#!/usr/bin/env python3
"""Q-Filters PPL eval on pg19 chunks (mirror of scripts/eval_baseline_ppl.py).

Flow:
    1. Load Llama-2-7B (bf16) on the requested device(s).
    2. Run offline calibration on `--calibration_chunks` chunks from the same
       pg19 shard and compute per-layer Q-filters via SVD.
    3. Patch the model with `QFiltersCache` and run PPL on `--max_chunks` chunks
       of `--seq_length` tokens.
    4. Write `<output_dir>/eval_results.json`.

Smoke contract (brief §6):
    python scripts/eval_qfilters.py \
        --model models/Llama--Llama2-7b \
        --data data/pg19_chunks.npy \
        --max_chunks 10 \
        --seq_length 4096 \
        --kv_budget 512 \
        --filter_rank 2 \
        --recent_window 64 \
        --calibration_chunks 8 \
        --output_dir outputs/qfilters_smoke \
        --single_gpu

Full eval (brief §7):
    torchrun --nproc_per_node=8 scripts/eval_qfilters.py \
        --model models/Llama--Llama2-7b \
        --data data/pg19_chunks.npy \
        --max_chunks 200 \
        --seq_length 4096 \
        --kv_budget 512 \
        --filter_rank 2 \
        --recent_window 64 \
        --calibration_chunks 64 \
        --output_dir outputs/qfilters_baseline \
        --bf16
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from typing import Iterable

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

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
# Data
# --------------------------------------------------------------------------- #


class PreTokenizedEvalDataset(Dataset):
    """Same layout as scripts/eval_baseline_ppl.py's dataset."""

    def __init__(
        self,
        npy_path: str,
        seq_length: int,
        skip_chunks: int,
        max_chunks: int,
    ) -> None:
        data = np.load(npy_path, mmap_mode="r")
        self.data = data[skip_chunks : skip_chunks + max_chunks].astype(np.int32)
        self.seq_length = seq_length
        logger.info(
            "Loaded %d chunks of %d tokens from %s",
            len(self.data), self.seq_length, npy_path,
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        # IMPORTANT: do NOT pre-shift. `LlamaForCausalLM.forward(labels=...)`
        # applies its own `shift_logits=logits[..., :-1, :]` /
        # `shift_labels=labels[..., 1:]`. Pre-shifting here is a DOUBLE SHIFT:
        # the model would be trained/scored to predict 2 tokens ahead, which
        # on Llama-3 (128K vocab, rope_theta=500000) explodes PPL to 4e7+ and
        # on Llama-2 inflates it from ~60 to ~3600. See bug investigation
        # 2026-04-25: bare-forward PPL 1.14 on noeos chunks vs 5e7 under the
        # old pre-shift alignment.
        tokens = torch.tensor(self.data[idx], dtype=torch.long)
        return {"input_ids": tokens, "labels": tokens.clone()}


def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


class CalibIterable:
    """Tiny iterable yielding {'input_ids': [1,T]} for the calibration loop.

    Separate from the eval dataset: we slice from the start of the chunks file
    so calibration never overlaps with the eval shard (eval uses
    skip_chunks=40000 by default, matching the baseline script).
    """

    def __init__(self, npy_path: str, seq_length: int, n_chunks: int) -> None:
        data = np.load(npy_path, mmap_mode="r")
        take = data[:n_chunks].astype(np.int32)
        self.items = [torch.tensor(take[i], dtype=torch.long)[:seq_length] for i in range(n_chunks)]
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
# Distributed helpers
# --------------------------------------------------------------------------- #


def init_distributed(single_gpu: bool) -> tuple[int, int, int]:
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
# Core
# --------------------------------------------------------------------------- #


@torch.no_grad()
def evaluate_ppl(
    model: LlamaForCausalLM,
    loader: DataLoader,
    device: torch.device,
    pad_token_id: int,
    world_size: int,
    sub_window_len: int,
) -> tuple[float, float, int]:
    """Return (ppl, avg_loss, total_tokens) summed across ranks if distributed.

    Sub-window carryover (fix for issue_20260425_qfilters_harness_noop):
        `QFiltersAttention.make_forward` installs compression as a POST-forward
        callback on `QFiltersCache`. If we call the model exactly once per
        chunk, compression fires AFTER loss is computed on a cache that is then
        discarded — the eval collapses to dense attention.

        We split each chunk into contiguous sub-windows of `sub_window_len`
        tokens, sharing the SAME cache across sub-windows. After sub-window k
        returns, the post-forward hook compresses `cache` down to `kv_budget`
        entries, so sub-window k+1 reads the compressed KV. This is what
        actually exercises Q-Filters scoring / sliding-window truncation.

        Note on label alignment: HF's LlamaForCausalLM does `shift_logits =
        logits[..., :-1, :]` / `shift_labels = labels[..., 1:]` internally. We
        already pre-shifted in the dataset, so we lose the last token of each
        sub-window (predicting across the boundary is skipped). At K=4
        sub-windows per 4095-token chunk that is a ~0.07% token drop vs the
        single-forward path — accepted. See issue_20260425 option (a).
    """
    from src.memory.qfilters.layer import make_qfilters_cache

    model.eval()
    total_loss = torch.zeros((), device=device, dtype=torch.float64)
    total_tokens = torch.zeros((), device=device, dtype=torch.float64)

    for i, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)  # [1, T]
        labels = batch["labels"].to(device, non_blocking=True)         # [1, T]
        # Fresh cache per chunk: we are scoring independent documents, and we
        # also want Q-Filters compression to act within each document only.
        cache = make_qfilters_cache(model)

        T = input_ids.size(1)
        sw_len = sub_window_len if sub_window_len > 0 else T  # 0 disables split (legacy)
        chunk_loss_report = None
        for start in range(0, T, sw_len):
            end = min(start + sw_len, T)
            sw_input = input_ids[:, start:end]
            sw_labels = labels[:, start:end]
            outputs = model(
                input_ids=sw_input,
                labels=sw_labels,
                past_key_values=cache,
                use_cache=True,
            )
            loss = outputs.loss.detach()
            if not torch.isfinite(loss):
                logger.warning(
                    "Non-finite loss at chunk %d sub-window %d (value=%s); skipping",
                    i, start // sw_len, loss.item(),
                )
                continue
            n_tokens = (sw_labels != pad_token_id).sum()
            total_loss += loss.double() * n_tokens.double()
            total_tokens += n_tokens.double()
            chunk_loss_report = loss.item()
        # No need to manually reset cache — next chunk allocates a fresh one above.

        if (i + 1) % 50 == 0 and chunk_loss_report is not None:
            cur_ppl = math.exp((total_loss / total_tokens.clamp_min(1)).item())
            logger.info(
                "  chunk %d: last_sw_loss=%.4f cumul_ppl=%.4f",
                i + 1, chunk_loss_report, cur_ppl,
            )

    if world_size > 1:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)

    tot_tok = int(total_tokens.item())
    if tot_tok == 0:
        raise RuntimeError("evaluate_ppl: 0 tokens scored (all chunks dropped?)")
    avg_loss = (total_loss / total_tokens).item()
    ppl = math.exp(avg_loss)
    return ppl, avg_loss, tot_tok


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Q-Filters PPL eval (Llama-2-7B / Llama-3.0-8B / pg19).")
    p.add_argument("--model", type=str, required=True,
                   help="HF model dir; accepts absolute paths (e.g. /apdcephfs_wzc1/.../Llama--Llama3-8b) "
                        "or project-relative (models/Llama--Llama2-7b).")
    p.add_argument("--data", type=str, required=True, help="pg19 chunks .npy (uint16)")
    p.add_argument("--max_chunks", type=int, default=200)
    p.add_argument("--seq_length", type=int, default=4096)
    p.add_argument("--skip_chunks", type=int, default=40000,
                   help="match eval_baseline_ppl.py offset (keeps eval shard disjoint "
                        "from calibration head-of-file)")
    p.add_argument("--kv_budget", type=int, default=512)
    p.add_argument("--filter_rank", type=int, default=2)
    p.add_argument("--recent_window", type=int, default=64)
    p.add_argument("--calibration_chunks", type=int, default=8)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--single_gpu", action="store_true",
                   help="force single-GPU mode regardless of torchrun env")
    p.add_argument("--bf16", action="store_true", default=True,
                   help="load model in bf16 (default True; kept for flag parity)")
    p.add_argument("--attn_impl", type=str, default="sdpa",
                   choices=["sdpa", "eager", "flash_attention_2"],
                   help="HF attn_implementation; sdpa is the safest default")
    p.add_argument("--filters_cache", type=str, default=None,
                   help="optional path to load/save the calibration filters.pt "
                        "(rank-0 writes, other ranks read)")
    p.add_argument("--mode", type=str, default="qfilters",
                   choices=["qfilters", "sliding_window"],
                   help="'qfilters' = full Q-Filters (default); "
                        "'sliding_window' = no filter scoring, keep last kv_budget tokens.")
    p.add_argument("--sub_window_len", type=int, default=1024,
                   help="Split each chunk into sub-windows of this length. "
                        "Cache carries over across sub-windows within a chunk, "
                        "so post-forward compression gates the NEXT sub-window's attention. "
                        "Set 0 to disable split (reverts to the pre-2026-04-25-15:31 broken "
                        "single-forward-per-chunk behavior — for diagnostic parity only).")
    p.add_argument("--seed", type=int, default=None,
                   help="Optional global seed. When provided, torch / cuda / numpy / random "
                        "RNGs are all seeded at process start (before model load or "
                        "calibration). Used to characterise residual stochasticity in the "
                        "Q-Filters pipeline (Issue #110 follow-up, 2026-04-26). When None "
                        "the legacy behaviour is preserved — no explicit seeding is done.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = init_distributed(args.single_gpu)
    is_dist = world_size > 1

    # Issue #110 follow-up (2026-04-26): opt-in explicit global seeding so that
    # multi-seed sweeps can probe residual stochasticity in the Q-Filters
    # pipeline. When --seed is unset we preserve the original (unseeded)
    # behaviour so this flag is a pure extension.
    if args.seed is not None:
        import random
        seed = int(args.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if is_main(rank):
            logger.info("Seeded RNGs (python/numpy/torch/cuda) with seed=%d", seed)

    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    if is_main(rank):
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info("Q-Filters eval | model=%s | mode=%s | kv_budget=%d filter_rank=%d "
                    "recent_window=%d calibration_chunks=%d | world_size=%d",
                    args.model, args.mode, args.kv_budget, args.filter_rank,
                    args.recent_window, args.calibration_chunks, world_size)

    # ---- tokenizer (needed for pad_token_id) ---- #
    # Use AutoTokenizer so Llama-2 (sentencepiece) and Llama-3 (tiktoken-BPE)
    # both resolve to the correct class. LlamaTokenizer is sentencepiece-only
    # and would fail on Llama-3.
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    # ---- model ---- #
    # AutoModelForCausalLM resolves to LlamaForCausalLM for both Llama-2 and
    # Llama-3.0 checkpoints; idempotent for the Llama-2 path.
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation=args.attn_impl,
        device_map={"": device} if device.type == "cuda" else None,
    )
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)

    # ---- calibration ---- #
    qcfg = QFiltersConfig(
        kv_budget=args.kv_budget,
        filter_rank=args.filter_rank,
        recent_window=args.recent_window,
        calibration_chunks=args.calibration_chunks,
    )

    if args.mode == "sliding_window":
        # No calibration, no filter I/O: every compress_layer call will hit the
        # `filters is None` fallback in QFiltersCache and keep the last
        # kv_budget tokens — i.e. plain sliding-window attention. All ranks
        # build the empty dict locally; nothing to broadcast.
        if is_main(rank):
            logger.info("Mode=sliding_window: skipping calibration and filter I/O; "
                        "all layers will fall back to last-%d keys.", args.kv_budget)
        filters: dict = {}
    else:
        filters = None
        cache_path = args.filters_cache or os.path.join(args.output_dir, "filters.pt")

        if is_main(rank):
            if os.path.exists(cache_path):
                logger.info("Loading cached filters from %s", cache_path)
                filters = torch.load(cache_path, map_location="cpu")
            else:
                logger.info("Running calibration (chunks=%d, rank=%d)...",
                            qcfg.calibration_chunks, qcfg.filter_rank)
                calib = CalibIterable(args.data, args.seq_length, qcfg.calibration_chunks)
                num_kv_heads = getattr(model.config, "num_key_value_heads",
                                       model.config.num_attention_heads)
                filters = compute_filters(
                    model=model,
                    calib_loader=calib,
                    rank=qcfg.filter_rank,
                    num_kv_heads=num_kv_heads,
                    device=device,
                )
                torch.save(filters, cache_path)
                logger.info("Saved filters to %s (%d layers)", cache_path, len(filters))

        # Broadcast filters path-based load across ranks.
        if is_dist:
            dist.barrier()
            if not is_main(rank):
                filters = torch.load(cache_path, map_location="cpu")

    # ---- patch ---- #
    patch_model(model, filters, qcfg)

    # ---- eval data ---- #
    dataset = PreTokenizedEvalDataset(
        args.data,
        seq_length=args.seq_length,
        skip_chunks=args.skip_chunks,
        max_chunks=args.max_chunks,
    )
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False) if is_dist else None
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=0,
    )

    ppl, avg_loss, n_tokens = evaluate_ppl(
        model=model,
        loader=loader,
        device=device,
        pad_token_id=pad_id,
        world_size=world_size,
        sub_window_len=args.sub_window_len,
    )

    if is_main(rank):
        result = {
            "ppl": ppl,
            "avg_loss": avg_loss,
            "num_tokens": n_tokens,
            "num_chunks": len(dataset),
            "mode": args.mode,
            "kv_budget": args.kv_budget,
            "filter_rank": args.filter_rank,
            "recent_window": args.recent_window,
            "calibration_chunks": args.calibration_chunks,
            "model": args.model,
            "seq_length": args.seq_length,
            "attn_impl": args.attn_impl,
            "bf16": bool(args.bf16),
            "sub_window_len": args.sub_window_len,
            "seed": args.seed,
        }
        out_path = os.path.join(args.output_dir, "eval_results.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("RESULT: PPL=%.4f | tokens=%d | chunks=%d | mode=%s | -> %s",
                    ppl, n_tokens, len(dataset), args.mode, out_path)
        print(f"\nQ-Filters PPL: {ppl:.4f} "
              f"(model={os.path.basename(args.model.rstrip('/'))}, "
              f"mode={args.mode}, kv_budget={args.kv_budget}, rank={args.filter_rank})")

    if is_dist:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
