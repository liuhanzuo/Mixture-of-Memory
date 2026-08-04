#!/usr/bin/env python3
"""Pre-tokenize OOD corpora into [N, seq_len] uint32 windows that match the
in-domain Dolmino held-out packing EXACTLY, so OOD PPL is directly comparable to
the in-domain held-out PPL (scripts/eval_olmo2_probe2_ppl.py, val=data/dolmino_now_val.npy).

Packing convention (verbatim from scripts/tokenize_dolmino_olmo2.py, verified 2026-08-04):
  * tokenize each document's text with add_special_tokens=False (OLMo-2 has NO BOS;
    bos_token_id=None in config -> matches the in-domain build, which also never
    prepends BOS),
  * append EOS (100257) after each document,
  * pack the contiguous token stream into seq_len windows, drop the <seq_len tail,
  * dtype uint32.

Corpora (each yields an iterable of documents = strings):
  * wikitext : Salesforce/wikitext, wikitext-103-raw-v1, split=test. General /
    encyclopedic text. Documents = articles (split on the " = Title = " header
    lines that wikitext uses); this mirrors dolmino's per-doc+EOS boundary.
  * pg19     : local data/pg19_train.jsonl (one book per line, field "text").
    Long-form narrative (public-domain books) -> OOD narrative relative to the
    Dolmino/DCLM web-text continuation distribution. Streamed; stops at budget.
  * c4       : allenai/c4, en, split=validation (optional general alternative).

--max_windows / --max_docs cap the token budget (PPL is token-weighted so the
value is corpus-size-independent; the cap only bounds noise/runtime).
"""
from __future__ import annotations

import argparse
import array
import json
import os
import re
import sys
import time

import numpy as np

os.environ.setdefault("RAYON_NUM_THREADS", "8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

EOS_ID = 100257  # OLMo-2 <|endoftext|>


def _log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------------------- #
# document iterators
# --------------------------------------------------------------------------- #
def iter_wikitext():
    """wikitext-103-raw-v1 test -> documents split on article header lines.

    wikitext raw uses ' = Title = \n' (level-1) headers between articles. We
    accumulate lines into an article buffer, flushing on each new level-1 header.
    """
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    hdr = re.compile(r"^ = [^=].* = $")  # level-1 header (single ' = ' pair)
    buf = []
    for row in ds:
        line = row["text"]
        if hdr.match(line.rstrip("\n")):
            if buf:
                txt = "".join(buf).strip()
                if txt:
                    yield txt
                buf = []
        buf.append(line)
    if buf:
        txt = "".join(buf).strip()
        if txt:
            yield txt


def iter_pg19(path, max_docs, byte_budget=0, block_chars=400_000, num_seeks=80):
    """data/pg19_train.jsonl is (despite the name) a raw line-wrapped text dump of
    concatenated public-domain books with NO per-document boundary and NO metadata
    (verified 2026-08-04: first bytes are the KJV Bible; json.loads fails).

    We sample ~block_chars-sized blocks from num_seeks byte offsets spread evenly
    across the WHOLE file (not just the head -- the head is the Bible, whose PPL is
    degenerate/memorized and unrepresentative). At each offset we discard the
    partial first line, then read a block and yield it as a pseudo-document (EOS is
    appended between blocks; boundary count is negligible and identical across all
    scored models, so it cannot bias the model-to-model comparison). byte_budget>0
    stops early once enough raw text has been read to fill the requested windows."""
    size = os.path.getsize(path)
    # spread offsets; skip the first 1% (Bible) and the last block_chars.
    lo = int(size * 0.01)
    hi = max(lo + 1, size - block_chars)
    offsets = [lo + int((hi - lo) * i / max(num_seeks - 1, 1)) for i in range(num_seeks)]
    n = 0
    read = 0
    with open(path, "rb") as fb:
        for off in offsets:
            fb.seek(off)
            fb.readline()  # discard partial line owned by previous region
            raw = fb.read(block_chars)
            tail = fb.readline()
            if tail:
                raw += tail
            block = raw.decode("utf-8", "replace").strip()
            if block:
                yield block
                n += 1
                read += len(raw)
            if max_docs and n >= max_docs:
                return
            if byte_budget and read >= byte_budget:
                return


def iter_c4(max_docs):
    from datasets import load_dataset
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    n = 0
    for row in ds:
        t = row.get("text", "")
        if t and t.strip():
            yield t.strip()
            n += 1
            if max_docs and n >= max_docs:
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=["wikitext", "pg19", "c4"])
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--max_windows", type=int, default=0, help=">0 cap output windows")
    ap.add_argument("--max_docs", type=int, default=0, help=">0 cap documents read")
    ap.add_argument("--pg19_path", default="data/pg19_train.jsonl")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    _log(f"tokenizer {args.tokenizer} bos={tok.bos_token_id} eos={tok.eos_token_id} "
         f"(we append EOS_ID={EOS_ID}, add_special_tokens=False)")

    if args.corpus == "wikitext":
        docs = iter_wikitext()
    elif args.corpus == "pg19":
        # ~4-5 chars/token; read ~2x the char budget for the requested windows.
        bb = (args.max_windows * args.seq_len * 10) if args.max_windows else 0
        docs = iter_pg19(args.pg19_path, args.max_docs, byte_budget=bb)
    else:
        docs = iter_c4(args.max_docs)

    seq_len = args.seq_len
    cap_tokens = args.max_windows * seq_len if args.max_windows else 0
    buf = array.array("I")
    n_docs = 0
    t0 = time.time()
    batch = []
    B = 64

    def flush_batch():
        enc = tok(batch, add_special_tokens=False)["input_ids"]
        for ids in enc:
            buf.extend(ids)
            buf.append(EOS_ID)

    for d in docs:
        batch.append(d)
        n_docs += 1
        if len(batch) >= B:
            flush_batch()
            batch = []
            if cap_tokens and len(buf) >= cap_tokens + seq_len:
                break
    if batch and (not cap_tokens or len(buf) < cap_tokens + seq_len):
        flush_batch()

    n_tokens = len(buf)
    n_chunks = n_tokens // seq_len
    if args.max_windows and n_chunks > args.max_windows:
        n_chunks = args.max_windows
    if n_chunks == 0:
        raise SystemExit(f"no full windows produced (n_tokens={n_tokens} < seq_len={seq_len})")
    arr = np.frombuffer(buf, dtype=np.uint32)[: n_chunks * seq_len].reshape(n_chunks, seq_len)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, arr)
    _log(f"[{args.corpus}] docs={n_docs} tokens={n_tokens:,} -> windows={n_chunks} "
         f"seq_len={seq_len} shape={arr.shape} ({time.time()-t0:.1f}s) -> {args.out}")
    # tiny provenance sidecar
    with open(args.out + ".meta.json", "w") as f:
        json.dump({"corpus": args.corpus, "tokenizer": args.tokenizer,
                   "seq_len": seq_len, "n_docs": n_docs, "n_tokens": n_tokens,
                   "n_windows": int(n_chunks), "eos_id": EOS_ID,
                   "add_special_tokens": False,
                   "pg19_path": args.pg19_path if args.corpus == "pg19" else None},
                  f, indent=2)


if __name__ == "__main__":
    main()
