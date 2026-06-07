#!/usr/bin/env python
"""Build a LONG-DOCUMENT subset of the per-document Dolmino dataset (F2 prep).

Background
----------
``MemLong/data/processed/dolmino_per_doc`` is a HF ``DatasetDict`` (train +
validation) produced by ``scripts/reprocess_dolmino_per_doc.py``. Each row has a
single column ``input_ids`` = one COMPLETE document (variable length, trailing
EOS=128001 included), kept only if length >= 512 tokens.

The mem_space training loop (``--per_doc_data`` in
``scripts/train_mem_space_dolmino_cpt.py`` -> ``DolminoCurriculumDataset`` in
``src/memory/mem_space/dolmino_dataset.py::_iter_per_doc``) slices each document
into consecutive non-overlapping windows of ``(n_ctx+1)*chunk_size`` tokens
(n_ctx context chunks + 1 target). So the number of *chunks* a document of
``L`` tokens contributes is ``L // chunk_size`` (any leftover < a full group is
dropped). To stress-test "write many chunks -> hold -> read back across chunks"
we want documents that produce DOZENS of chunks per sample, i.e. large ``L``:

    chunks_produced(L) = L // chunk_size

    chunk_size=512  : >=40 chunks  needs L >= 40*512  = 20480 tok
    chunk_size=512  : >=80 chunks  needs L >= 80*512  = 40960 tok
    chunk_size=1024 : >=40 chunks  needs L >= 40*1024 = 40960 tok

CAVEAT (measured 2026-06-08): the existing per-doc dataset is HARD-TRUNCATED at
4096 tokens (the original SlimPajama/Dolmino tokeniser ran with max_length=4096;
~6.8% of train docs sit exactly at 4096). NO document exceeds 4096 tokens, so the
most chunks any single sample can produce is 16 (chunk256), 8 (chunk512) or 4
(chunk1024). Reaching the F2 goal of "dozens-hundreds of chunks per sample"
CANNOT be done by filtering this dataset — it requires RE-TOKENISING raw Dolmino
with a much larger max_length. This script still builds the longest-tail subset
(useful for chunk256 stress tests), but read the report before relying on it.

This script
-----------
1. Loads the per-doc DatasetDict with ``datasets.load_from_disk``.
2. For each split (train/validation) computes per-document token length
   (``len(input_ids)``) and prints the length distribution
   (min/median/p90/p99/max + token-count + chunk-count at common chunk sizes).
3. Filters: keep documents with ``len >= --min_tokens``; optionally keep only
   the longest ``--top_n`` of those (per split).
4. Writes a new ``DatasetDict`` (same single-column ``input_ids`` schema, so the
   training script's ``--per_doc_data`` can read it directly) via
   ``save_to_disk`` to ``--output_dir``.

``--dry_run`` scans + prints the distribution ONLY (no write), so you can pick a
threshold before materialising the (potentially large) subset.

Run with the project venv on a CPU node:
    .venv/bin/python scripts/build_dolmino_longdoc_subset.py --dry_run
"""
from __future__ import annotations

import argparse
import os
from typing import List, Optional

import numpy as np

CHUNK_SIZES = (256, 512, 1024)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--input_dir",
        type=str,
        default="MemLong/data/processed/dolmino_per_doc",
        help="Path to the per-document Dolmino DatasetDict (load_from_disk).",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="MemLong/data/processed/dolmino_longdoc",
        help="Output directory for the long-doc subset DatasetDict (save_to_disk).",
    )
    p.add_argument(
        "--min_tokens",
        type=int,
        default=3072,
        # NOTE: the upstream per-doc dataset is hard-truncated at 4096 tokens
        # (tokenizer max_length during the original SlimPajama/Dolmino tokenise
        # step), so NO document exceeds 4096 tokens. A default of 3072 keeps the
        # longest ~tail (docs near the 4096 ceiling) and yields a non-empty
        # subset; anything >= ~4097 returns zero docs. See the report / F2 note:
        # reaching "dozens-hundreds of chunks per sample" requires RE-TOKENISING
        # raw Dolmino with a larger max_length, not just filtering this dataset.
        help="Keep only documents with at least this many tokens. Default 3072 "
             "(the per-doc data is capped at 4096 tok; see header docstring). "
             "At chunk256 a 4096-tok doc yields 16 chunks; chunk512 -> 8.",
    )
    p.add_argument(
        "--top_n",
        type=int,
        default=5000,
        help="After the --min_tokens filter, keep only the LONGEST top_n "
             "documents PER SPLIT. Use 0 or a negative value (or --no_top_n) to "
             "keep ALL documents that pass --min_tokens.",
    )
    p.add_argument(
        "--no_top_n",
        action="store_true",
        default=False,
        help="Disable the top_n cap; keep every document passing --min_tokens.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="Only scan + print length distributions; do NOT write any output.",
    )
    return p.parse_args()


def _print_dist(tag: str, lengths: np.ndarray) -> None:
    """Print the token-length distribution + chunk-count implications."""
    n = lengths.shape[0]
    if n == 0:
        print(f"[longdoc] {tag}: 0 documents", flush=True)
        return
    print(
        f"[longdoc] {tag}: n={n}  tokens "
        f"min/median/mean/p90/p99/max = "
        f"{int(lengths.min())}/{int(np.median(lengths))}/{lengths.mean():.0f}/"
        f"{int(np.percentile(lengths, 90))}/{int(np.percentile(lengths, 99))}/"
        f"{int(lengths.max())}",
        flush=True,
    )
    # How many docs reach >= X chunks at each chunk_size.
    for cs in CHUNK_SIZES:
        chunks = lengths // cs
        for thr in (10, 20, 40, 80):
            frac = float((chunks >= thr).mean())
            cnt = int((chunks >= thr).sum())
            print(
                f"[longdoc]   chunk{cs}: docs producing >={thr:>3} chunks "
                f"(>= {thr * cs:>6} tok): {cnt} ({frac * 100:.1f}%)",
                flush=True,
            )


def _lengths_for_split(ds) -> np.ndarray:
    """Compute per-document token lengths for an HF Dataset (memory-bounded)."""
    lengths: List[int] = []
    # Iterate in batches to avoid materialising all rows at once.
    for batch in ds.iter(batch_size=2000):
        for row in batch["input_ids"]:
            lengths.append(len(row))
    return np.asarray(lengths, dtype=np.int64)


def main() -> None:
    args = parse_args()

    import datasets

    ddict = datasets.load_from_disk(args.input_dir)
    if isinstance(ddict, datasets.Dataset):
        # Single split on disk -> wrap so the rest of the logic is uniform.
        ddict = datasets.DatasetDict({"train": ddict})

    split_names = list(ddict.keys())
    print(f"[longdoc] loaded DatasetDict from {args.input_dir}: splits={split_names}",
          flush=True)

    use_top_n = (not args.no_top_n) and (args.top_n is not None) and (args.top_n > 0)

    out_splits = {}
    for name in split_names:
        ds = ddict[name]
        lengths = _lengths_for_split(ds)
        print(f"[longdoc] ===== split '{name}' (raw) =====", flush=True)
        _print_dist(f"{name} raw", lengths)

        # Filter by min_tokens.
        keep_mask = lengths >= args.min_tokens
        keep_idx = np.flatnonzero(keep_mask)
        n_pass = keep_idx.shape[0]
        print(
            f"[longdoc] split '{name}': {n_pass}/{lengths.shape[0]} docs pass "
            f"--min_tokens={args.min_tokens}",
            flush=True,
        )

        if n_pass == 0:
            print(f"[longdoc] WARNING split '{name}': 0 docs pass filter", flush=True)
            out_splits[name] = ([], lengths[keep_idx])
            continue

        # Optionally keep only the longest top_n among the passing docs.
        if use_top_n and n_pass > args.top_n:
            pass_lengths = lengths[keep_idx]
            # indices (into keep_idx) of the top_n longest passing docs.
            order = np.argsort(pass_lengths)[::-1][: args.top_n]
            sel_idx = keep_idx[order]
            print(
                f"[longdoc] split '{name}': capping to longest --top_n="
                f"{args.top_n} of the {n_pass} passing docs",
                flush=True,
            )
        else:
            sel_idx = keep_idx

        sel_idx_sorted = np.sort(sel_idx)  # keep original row order
        sel_lengths = lengths[sel_idx_sorted]
        print(f"[longdoc] ----- split '{name}' (kept subset) -----", flush=True)
        _print_dist(f"{name} kept", sel_lengths)

        out_splits[name] = (sel_idx_sorted.tolist(), sel_lengths)

    if args.dry_run:
        print("[longdoc] --dry_run set: scanned + printed distributions only, "
              "no output written.", flush=True)
        return

    # Materialise the selected documents into a new DatasetDict.
    new_dd = {}
    for name in split_names:
        sel_idx, _ = out_splits[name]
        new_dd[name] = ddict[name].select(sel_idx)
    out_ddict = datasets.DatasetDict(new_dd)

    os.makedirs(os.path.dirname(args.output_dir) or ".", exist_ok=True)
    out_ddict.save_to_disk(args.output_dir)
    sizes = {k: len(v) for k, v in out_ddict.items()}
    print(f"[longdoc] wrote DatasetDict to {args.output_dir}: {sizes}", flush=True)


if __name__ == "__main__":
    main()
