#!/usr/bin/env python
"""Offline teacher cache builder for self-study distillation (v21).

Runs a PURE frozen Llama-3-8B (NOT patched with memory) over the dolmino
per-doc corpus and dumps, for the answer segment (= the whole target chunk of
every (n_ctx+1)-chunk group), the teacher's top-64 next-token logits + selected
decoder-layer hidden states. The training script later reads these .npz files
and distills the closed-book memory readout (student) toward the open-book
full-context teacher.

Design doc: versions/v21_selfstudy_distillation.md  (§3 §4 §5 §6).

Key correctness points
-----------------------
* teacher forward = ONE flat sequence concat(ctx_chunks, target_chunk), original
  causal attention, output_hidden_states=True, use_cache=False. answer segment =
  the LAST chunk_size positions (= target chunk). This is OPEN-BOOK on purpose.
* hidden_states off-by-one: HF returns a tuple of length n_layers+1 where index 0
  is the embedding output, so decoder layer L's output is hidden_states[L+1].
  --distill_layers indices are decoder-layer indices (e.g. 12,20,28).
* sample_id stability: keyed by (doc_idx, group_pos) which is INVARIANT to the
  per-epoch shuffle / DDP sharding the training loader applies. doc_idx = the
  Arrow row's original index (0..num_samples-1); group_pos = the index of the
  (n_ctx+1)*chunk_size window WITHIN that document (0,1,2,...). We walk every doc
  in original order, NO shuffle, so the cache key matches the training loader's
  per-sample (doc_idx, group_pos) regardless of how it shuffles/shards at runtime.

Output: <out_dir>/<doc_idx>_<group_pos>.npz with
    logit_idx  int32 [chunk_size, topk]
    logit_val  bf16  [chunk_size, topk]   (raw logits; T/softmax applied at train time)
    hidden     bf16  [chunk_size, n_layers_sel, 4096]
    answer_mask bool [chunk_size]          (all True for dolmino; kept for parity)
plus group-level metadata (doc_idx, group_pos, n_ctx, chunk_size, layers).

Multi-rank: shard groups round-robin by a running global group counter modulo
world_size; each rank writes its own .npz, no inter-rank communication.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List

import numpy as np
import torch

# Make repo root importable (mirror train script's sys.path handling).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from transformers import AutoTokenizer, LlamaForCausalLM  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Build teacher distill cache (v21).")
    p.add_argument("--dolmino_path", type=str, required=True,
                   help="HF Arrow dataset dir (per-doc), e.g. "
                        "MemLong/data/processed/dolmino_per_doc/train")
    p.add_argument("--model_path", type=str, default="models/Meta-Llama-3-8B",
                   help="Pure frozen teacher backbone (NOT patched with memory).")
    p.add_argument("--chunk_size", type=int, default=512)
    p.add_argument("--n_ctx", type=int, default=3,
                   help="Number of context chunks per group (must match the "
                        "n_ctx the training loader uses for these samples, so "
                        "group_pos windows line up). For curriculum runs, build "
                        "one cache per n_ctx the training will actually use.")
    p.add_argument("--distill_layers", type=str, default="12,20,28",
                   help="Comma-separated DECODER-layer indices (0-indexed). "
                        "hidden_states[L+1] is taken (index 0 is embeddings).")
    p.add_argument("--topk", type=int, default=64)
    p.add_argument("--out_dir", type=str, required=True,
                   help="Output dir, e.g. distill_cache/512/")
    p.add_argument("--max_docs", type=int, default=-1,
                   help="Limit number of documents (for smoke tests). -1 = all.")
    p.add_argument("--max_groups", type=int, default=-1,
                   help="Limit total groups written (smoke). -1 = all.")
    p.add_argument("--attn_impl", type=str, default="sdpa",
                   choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--rank", type=int, default=int(os.environ.get("RANK", 0)),
                   help="Data-parallel shard index (groups modulo world_size).")
    p.add_argument("--world_size", type=int,
                   default=int(os.environ.get("WORLD_SIZE", 1)),
                   help="Number of parallel cache-builder processes.")
    p.add_argument("--local_rank", type=int,
                   default=int(os.environ.get("LOCAL_RANK", 0)),
                   help="CUDA device index for this process.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-dump even if the .npz already exists.")
    return p.parse_args()


def main():
    args = parse_args()
    layers = [int(x) for x in args.distill_layers.split(",") if x.strip() != ""]
    os.makedirs(args.out_dir, exist_ok=True)

    # (meta.json is written AFTER the dataset is loaded below, so it can record
    # the dataset _fingerprint — sample_id=(doc_idx,group_pos) is a POSITIONAL
    # index into this exact Arrow, so the cache is only valid for a dataset with
    # the identical fingerprint/row-order.)

    device = torch.device(
        f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"[rank {args.rank}/{args.world_size}] loading teacher {args.model_path} "
          f"on {device} ({dtype}) ...", flush=True)
    model = LlamaForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_impl,
        local_files_only=True,
    ).to(device)
    model.eval()
    for pp in model.parameters():
        pp.requires_grad_(False)

    n_decoder_layers = model.config.num_hidden_layers
    for L in layers:
        assert 0 <= L < n_decoder_layers, (
            f"--distill_layers index {L} out of range [0,{n_decoder_layers})")

    # Tokenizer not strictly needed (data is pre-tokenised) but loaded for parity
    # / vocab-size sanity. Non-fatal if absent.
    try:
        _ = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    except Exception as e:  # noqa: BLE001
        print(f"[rank {args.rank}] tokenizer load skipped ({e}).", flush=True)

    import datasets
    ds = datasets.load_from_disk(args.dolmino_path)
    num_docs = len(ds)
    if args.max_docs > 0:
        num_docs = min(num_docs, args.max_docs)
    print(f"[rank {args.rank}] dataset has {len(ds)} docs; scanning {num_docs}.",
          flush=True)

    # Drop meta.json (rank 0 only) describing exactly how this cache was sliced,
    # so the training script can assert n_ctx / chunk_size / distill_layers AND
    # the dataset identity match. sample_id=(doc_idx,group_pos) is a POSITIONAL
    # index into THIS Arrow; if training uses a dataset with a different
    # _fingerprint (different row order, e.g. an independently reprocessed copy)
    # every key points at a different document -> 100% silent cache miss. Record
    # the fingerprint so training fails fast instead of training garbage.
    if args.rank == 0:
        import json
        ds_fingerprint = getattr(ds, "_fingerprint", None)
        meta_path = os.path.join(args.out_dir, "meta.json")
        meta = {
            "n_ctx": int(args.n_ctx),
            "chunk_size": int(args.chunk_size),
            "distill_layers": layers,
            "model_path": args.model_path,
            "topk": int(args.topk),
            "group_len": (int(args.n_ctx) + 1) * int(args.chunk_size),
            "dataset_fingerprint": ds_fingerprint,
            "dataset_num_docs": int(len(ds)),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[rank 0] wrote cache meta {meta_path}: {meta}", flush=True)

    chunk_size = args.chunk_size
    group_size = args.n_ctx + 1
    group_len = group_size * chunk_size

    global_group_counter = -1  # increments for EVERY group across the whole corpus
    n_written = 0
    n_total_groups = 0

    for doc_idx in range(num_docs):
        tokens = ds[int(doc_idx)]["input_ids"]
        doc_len = len(tokens)

        pos = 0
        group_pos = 0
        while pos + group_len <= doc_len:
            global_group_counter += 1
            this_group_pos = group_pos
            this_pos = pos
            pos += group_len
            group_pos += 1
            n_total_groups += 1

            # Round-robin shard across parallel cache builders.
            if (global_group_counter % args.world_size) != args.rank:
                continue

            out_path = os.path.join(
                args.out_dir, f"{doc_idx}_{this_group_pos}.npz")
            if (not args.overwrite) and os.path.exists(out_path):
                continue

            flat_tokens = tokens[this_pos: this_pos + group_len]
            flat = torch.tensor(flat_tokens, dtype=torch.long,
                                device=device).unsqueeze(0)  # [1, group_len]

            with torch.no_grad():
                out = model(input_ids=flat, output_hidden_states=True,
                            use_cache=False)

            # answer segment = LAST chunk_size positions (= target chunk).
            logits = out.logits[0, -chunk_size:, :]  # [chunk_size, V]
            topv, topi = torch.topk(logits, args.topk, dim=-1)  # both [cs, topk]

            # hidden_states is a tuple len n_decoder_layers+1; index 0 = embed,
            # decoder layer L output = hidden_states[L+1]. Off-by-one固化:
            hs = out.hidden_states
            assert len(hs) == n_decoder_layers + 1, (
                f"expected {n_decoder_layers+1} hidden_states, got {len(hs)}")
            hid_layers = []
            for L in layers:
                h = hs[L + 1][0, -chunk_size:, :]  # [chunk_size, 4096]
                hid_layers.append(h)
            hidden = torch.stack(hid_layers, dim=1)  # [chunk_size, n_sel, 4096]

            answer_mask = np.ones((chunk_size,), dtype=bool)  # dolmino: all target

            np.savez(
                out_path,
                logit_idx=topi.to(torch.int32).cpu().numpy(),
                logit_val=topv.to(torch.bfloat16).view(torch.int16).cpu().numpy(),
                hidden=hidden.to(torch.bfloat16).view(torch.int16).cpu().numpy(),
                answer_mask=answer_mask,
                meta_doc_idx=np.int64(doc_idx),
                meta_group_pos=np.int64(this_group_pos),
                meta_n_ctx=np.int64(args.n_ctx),
                meta_chunk_size=np.int64(chunk_size),
                meta_layers=np.array(layers, dtype=np.int32),
            )
            n_written += 1
            if n_written % 50 == 0:
                print(f"[rank {args.rank}] wrote {n_written} groups "
                      f"(scanned {n_total_groups} total) ...", flush=True)

            if args.max_groups > 0 and n_written >= args.max_groups:
                print(f"[rank {args.rank}] hit --max_groups={args.max_groups}, stop.",
                      flush=True)
                print(f"[rank {args.rank}] DONE. wrote {n_written} npz.", flush=True)
                return

    print(f"[rank {args.rank}] DONE. wrote {n_written} npz "
          f"(scanned {n_total_groups} total groups across {num_docs} docs).",
          flush=True)


if __name__ == "__main__":
    main()
