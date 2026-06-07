#!/usr/bin/env python
"""Re-tokenise RAW Dolmino text (NO truncation) into a long-document subset (F2).

Background
----------
The F2 step of the plan needs single training samples that produce DOZENS to
HUNDREDS of chunks, so we can stress-test the memory's "write many chunks ->
hold -> read back across chunks" capability. The number of chunks a document of
``L`` tokens contributes at a given ``chunk_size`` is ``L // chunk_size``:

    chunk_size=512  : >=40 chunks  needs L >= 40*512  = 20480 tok
    chunk_size=512  : >=80 chunks  needs L >= 80*512  = 40960 tok
    chunk_size=1024 : >=40 chunks  needs L >= 40*1024 = 40960 tok

The 4096-truncation problem (measured 2026-06-08)
-------------------------------------------------
``MemLong/data/processed/dolmino_per_doc`` (produced by
``scripts/reprocess_dolmino_per_doc.py`` from the PACKED Arrow dataset
``dolmino_0.5B_1024``) is HARD-TRUNCATED at 4096 tokens: the original packing /
tokenise step ran with ``max_length=4096`` so the recovered per-document rows all
sit at p99 = max = 4096. NO document exceeds 4096 tokens, so the most chunks any
single sample can ever produce is 16 (chunk256), 8 (chunk512) or 4 (chunk1024).
``scripts/build_dolmino_longdoc_subset.py`` (a pure FILTER over that dataset)
therefore cannot reach the F2 goal — filtering a 4096-capped dataset can never
yield a >=20k-token document.

The fix: re-tokenise the RAW text directly, with NO max_length / truncation. The
raw long-document sources are already on local disk:

    MemLong/data/raw/dolmino_pes2o_wiki/raw/data/pes2o/pes2o-00XX.json.gz  # academic papers, naturally long
    MemLong/data/raw/dolmino_pes2o_wiki/raw/data/wiki/wiki-000X.json.gz

Each ``.json.gz`` is a gzip-compressed JSONL (one JSON object per line); the text
field is ``text`` (verified 2026-06-08 on pes2o-0000).

This script
-----------
1. Streams each ``--raw_glob`` file as gzip JSONL (line by line, never loading a
   whole file into memory), reads ``--text_field`` (default ``text``); probes the
   first record and errors out listing the available fields if it is missing.
2. Tokenises each text with ``tokenizer(text, add_special_tokens=False)`` and
   manually prepends BOS=128000 + appends EOS=128001 (matching the
   ``reprocess_dolmino_per_doc.py`` schema: each per-doc row is one complete
   document with a leading BOS and a trailing EOS). NO max_length / truncation.
3. Keeps documents with token length >= ``--min_tokens`` (default 8192; F2 wants
   long docs). Optional ``--max_docs`` cap and ``--max_files`` scan cap.
4. Prints the token-length distribution (min/median/mean/p90/p99/max) and, for
   chunk_size in {256, 512, 1024}, the fraction of kept docs producing
   >=10/>=20/>=40/>=80 chunks (chunks = L // chunk_size). Style follows
   ``build_dolmino_longdoc_subset.py``.
5. Writes a HF ``DatasetDict`` (train + validation, document-level
   ``--val_frac`` split) with a single column ``input_ids`` (list[int]) via
   ``save_to_disk(--out_path)``. The schema is byte-for-byte compatible with
   ``dolmino_per_doc`` so the training script's ``--per_doc_data`` reads it
   directly (``src/memory/mem_space/dolmino_dataset.py::_iter_per_doc``).

``--dry_run`` scans + tokenises + prints the distribution ONLY (no write), so you
can confirm pes2o really has >=20k-token docs and pick ``--min_tokens`` before
materialising the (potentially large) dataset.

Run with the project venv on a CPU node:
    .venv/bin/python scripts/build_dolmino_longdoc_raw_retokenize.py --dry_run \
        --max_files 2 \
        --raw_glob 'MemLong/data/raw/dolmino_pes2o_wiki/raw/data/pes2o/*.json.gz'
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
from typing import Iterator, List, Optional

import numpy as np

EOS_TOKEN_ID = 128001
BOS_TOKEN_ID = 128000

CHUNK_SIZES = (256, 512, 1024)

DEFAULT_TOKENIZER_CANDIDATES = (
    "models/Meta-Llama-3-8B-Instruct",
    "models/Meta-Llama-3-8B",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--raw_glob",
        type=str,
        nargs="+",
        default=["MemLong/data/raw/dolmino_pes2o_wiki/raw/data/pes2o/*.json.gz"],
        help="One or more globs (space-separated; each may also be a "
             "comma-separated list) for the raw .json.gz JSONL files. "
             "Default: all pes2o shards.",
    )
    p.add_argument(
        "--text_field",
        type=str,
        default="text",
        help="JSON field holding the document text (default 'text'). The first "
             "record is probed; if the field is missing the script errors out "
             "and lists the available fields.",
    )
    p.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Local Llama-3 tokenizer directory. Default: auto-detect the first "
             "existing of " + ", ".join(DEFAULT_TOKENIZER_CANDIDATES) + ". "
             "Loaded with local_files_only=True.",
    )
    p.add_argument(
        "--min_tokens",
        type=int,
        default=8192,
        help="Keep only documents with at least this many tokens (incl. BOS/EOS). "
             "Default 8192 (F2 wants long docs).",
    )
    p.add_argument(
        "--max_docs",
        type=int,
        default=None,
        help="Optional cap on the number of KEPT documents (defence against an "
             "over-large dataset). Default None = no cap.",
    )
    p.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Optional cap on the number of raw files scanned (handy for "
             "--dry_run, e.g. --max_files 2). Default None = all matched files.",
    )
    p.add_argument(
        "--out_path",
        type=str,
        default="MemLong/data/processed/dolmino_longdoc",
        help="Output directory for the long-doc DatasetDict (save_to_disk).",
    )
    p.add_argument(
        "--val_frac",
        type=float,
        default=0.05,
        help="Fraction of documents held out for validation (document-level "
             "split). Default 0.05.",
    )
    p.add_argument(
        "--num_proc",
        type=int,
        default=8,
        help="Tokenisation parallelism (this is a CPU node). Default 8.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="Only scan + tokenise + print the length distribution; do NOT write "
             "any output.",
    )
    p.add_argument("--seed", type=int, default=42, help="Seed for the split.")
    return p.parse_args()


def resolve_tokenizer_path(arg: Optional[str]) -> str:
    if arg:
        if not os.path.isdir(arg):
            raise FileNotFoundError(f"--tokenizer path does not exist: {arg}")
        return arg
    for cand in DEFAULT_TOKENIZER_CANDIDATES:
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError(
        "No tokenizer found. Tried: " + ", ".join(DEFAULT_TOKENIZER_CANDIDATES)
        + ". Pass --tokenizer explicitly."
    )


def expand_globs(raw_glob: List[str], max_files: Optional[int]) -> List[str]:
    files: List[str] = []
    for entry in raw_glob:
        for part in entry.split(","):
            part = part.strip()
            if not part:
                continue
            matched = sorted(glob.glob(part))
            files.extend(matched)
    # De-duplicate while preserving order.
    seen = set()
    uniq: List[str] = []
    for f in files:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    if max_files is not None:
        uniq = uniq[:max_files]
    return uniq


def iter_texts(
    files: List[str], text_field: str
) -> Iterator[str]:
    """Stream raw text strings line-by-line from gzip JSONL files.

    The first record (across all files) is probed for ``text_field``; if it is
    missing a RuntimeError listing the available keys is raised.
    """
    probed = False
    for path in files:
        print(f"[longdoc-raw] reading {path}", flush=True)
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not probed:
                    if text_field not in obj:
                        raise RuntimeError(
                            f"--text_field '{text_field}' not found in first "
                            f"record. Available fields: {sorted(obj.keys())}"
                        )
                    print(
                        f"[longdoc-raw] field probe OK: using text_field="
                        f"'{text_field}' (available: {sorted(obj.keys())})",
                        flush=True,
                    )
                    probed = True
                text = obj.get(text_field)
                if isinstance(text, str) and text:
                    yield text


def _print_dist(tag: str, lengths: np.ndarray) -> None:
    """Print token-length distribution + chunk-count implications."""
    n = lengths.shape[0]
    if n == 0:
        print(f"[longdoc-raw] {tag}: 0 documents", flush=True)
        return
    print(
        f"[longdoc-raw] {tag}: n={n}  tokens "
        f"min/median/mean/p90/p99/max = "
        f"{int(lengths.min())}/{int(np.median(lengths))}/{lengths.mean():.0f}/"
        f"{int(np.percentile(lengths, 90))}/{int(np.percentile(lengths, 99))}/"
        f"{int(lengths.max())}",
        flush=True,
    )
    for cs in CHUNK_SIZES:
        chunks = lengths // cs
        for thr in (10, 20, 40, 80):
            frac = float((chunks >= thr).mean())
            cnt = int((chunks >= thr).sum())
            print(
                f"[longdoc-raw]   chunk{cs}: docs producing >={thr:>3} chunks "
                f"(>= {thr * cs:>6} tok): {cnt} ({frac * 100:.1f}%)",
                flush=True,
            )


def main() -> None:
    args = parse_args()

    from transformers import AutoTokenizer

    tok_path = resolve_tokenizer_path(args.tokenizer)
    print(f"[longdoc-raw] loading tokenizer from {tok_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(tok_path, local_files_only=True)

    files = expand_globs(args.raw_glob, args.max_files)
    if not files:
        raise RuntimeError(
            f"No files matched --raw_glob={args.raw_glob} (max_files={args.max_files})"
        )
    print(
        f"[longdoc-raw] {len(files)} file(s) to scan; min_tokens={args.min_tokens}, "
        f"max_docs={args.max_docs}, num_proc={args.num_proc}",
        flush=True,
    )

    # Stream + tokenise. We accumulate KEPT docs into an in-memory list. For
    # --dry_run with --max_files this is bounded; for a full materialise the
    # --max_docs cap guards against an over-large dataset.
    kept_docs: List[List[int]] = []
    kept_lengths: List[int] = []
    n_seen = 0
    n_kept = 0

    for text in iter_texts(files, args.text_field):
        n_seen += 1
        body = tokenizer(text, add_special_tokens=False)["input_ids"]
        ids = [BOS_TOKEN_ID] + body + [EOS_TOKEN_ID]
        L = len(ids)
        if L >= args.min_tokens:
            kept_docs.append(ids)
            kept_lengths.append(L)
            n_kept += 1
            if n_kept % 5000 == 0:
                print(f"[longdoc-raw] kept {n_kept} docs so far "
                      f"(scanned {n_seen})", flush=True)
            if args.max_docs is not None and n_kept >= args.max_docs:
                print(f"[longdoc-raw] reached --max_docs={args.max_docs}, "
                      f"stopping scan", flush=True)
                break
        if n_seen % 50000 == 0:
            print(f"[longdoc-raw] scanned {n_seen} docs, kept {n_kept}", flush=True)

    print("[longdoc-raw] ============ statistics ============", flush=True)
    print(f"[longdoc-raw] scanned {n_seen} docs; kept {n_kept} "
          f"(>= {args.min_tokens} tok)", flush=True)
    lengths_np = np.asarray(kept_lengths, dtype=np.int64)
    _print_dist("kept", lengths_np)

    if args.dry_run:
        print("[longdoc-raw] --dry_run set: scanned + tokenised + printed "
              "distribution only, no output written.", flush=True)
        return

    if n_kept == 0:
        raise RuntimeError(
            "No documents survived the --min_tokens filter — lower the threshold "
            "or check the data."
        )

    import datasets

    full = datasets.Dataset.from_dict({"input_ids": kept_docs})
    # Document-level split. shuffle only selects WHICH docs go to validation;
    # token order within each document is untouched.
    split = full.train_test_split(
        test_size=args.val_frac, shuffle=True, seed=args.seed
    )
    ddict = datasets.DatasetDict(
        {"train": split["train"], "validation": split["test"]}
    )

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    ddict.save_to_disk(args.out_path)
    print(
        f"[longdoc-raw] wrote DatasetDict to {args.out_path}: "
        f"train={len(ddict['train'])}, validation={len(ddict['validation'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
