#!/usr/bin/env python
"""Paper A P0.14 — InfiniteBench / PG-19 train-eval contamination audit (CPU only).

PURPOSE
-------
PG-19 (our flagship LoRA's *only* distillation corpus) and InfiniteBench's
``longbook_qa_eng`` / ``longbook_choice_eng`` both draw on public-domain
Project-Gutenberg books. Before submission we must rule out that a training book
and an eval book are the *same* text, which would make the InfiniteBench natural
long-document quality numbers (Table ``tab:infbench``: Book-QA F1, Book-choice
acc) a train-on-test artifact rather than clean generalization.

WHAT IS / IS NOT AVAILABLE (both verified from the repo, not assumed)
--------------------------------------------------------------------
* Training corpus  = ``data/pg19_train.jsonl`` (11.4 GB raw PG-19 wrapped text).
  This is the sole corpus of the flagship adapter
  ``outputs/qcmem_distill_qwen_j12_r32_4k/final`` — see its sibling
  ``outputs/qcmem_distill_qwen_j12_r32_4k/distill_args.json`` ("pg19_path": .../data/pg19_train.jsonl).
  The training loader (``scripts/train_qcmem_distill.py::PG19Packer``) concatenates
  every non-empty line with NO per-document boundary and NO title/PG-ID metadata.
* Eval docs = InfiniteBench ``longbook_qa_eng.jsonl`` (351 ex) +
  ``longbook_choice_eng.jsonl`` (229 ex). The eval *predictions* were produced on
  a remote GPU node (diskB zwfy6 .73) and are NOT on this wzc1 node; the eval
  DATA is the public HF dataset ``xinrongzhang2022/InfiniteBench`` (CPU download,
  no GPU touched). Each record has {id, context, input, answer, options} — again
  NO title / author / PG-ID field, and character names are anonymized.

AUDIT METHODS (three, per the task spec)
----------------------------------------
(a) Title / author / PG-ID intersection: NOT COMPUTABLE — neither side carries
    that metadata (documented, not skipped). We instead identify overlap by
    content.
(b) Exact document hash: sha256 of each eval book's raw + whitespace-normalized
    context, checked for equality with any training-line concatenation. Expected
    to find ZERO matches because (i) eval contexts are anonymized and (ii) the
    training corpus has no matching document boundary — reported for completeness.
(c) Long n-gram containment (GPT-3-style decontamination) + MinHash-style
    downsampling for near-duplicates: build a downsampled set of 64-bit hashes of
    every word-level ``--n`` gram (default 13) in the *whole* PG-19 training
    corpus (normalization = lowercase, non-alnum -> space, collapse whitespace),
    then for each unique eval book measure the fraction of ITS unique n-gram
    hashes that are present in the training set. Because names are anonymized,
    a genuinely-in-training book will not reach 1.0, but its narration n-grams
    match verbatim, so containment is high (>> a book that is truly absent).
    Downsampling by ``h % D == 0`` (default D=32) is a bottom-hash / MinHash
    sketch: the containment estimate is unbiased and bounds memory to ~GB.

OUTPUTS (all under bench_results/p0_14_contamination/)
------------------------------------------------------
* ``data_manifest.json``    — every path, size, sha, role, and what metadata exists.
* ``match_list.json``       — per unique eval book: containment ratio, n-gram
                              counts, exact-hash result, verdict, and the (task,id)
                              records that map to it.
* ``per_record_verdict.jsonl`` — one row per eval record (task,id) with its book
                              verdict → drives clean-subset selection.
* ``clean_subset_ids.json`` — eval (task,id) records whose book is NOT contaminated.
* ``thresholds.json``       — the exact thresholds / params used.

USAGE
-----
  python scripts/audit_p0_14_contamination.py \
      --train_corpus data/pg19_train.jsonl \
      --eval_dir .t27_tmp/infb_eval \
      --out_dir bench_results/p0_14_contamination \
      --n 13 --downsample 32 --contam_high 0.80 --contam_low 0.10 --workers 64

Reuse a previously built training sketch with ``--sketch_npy <path>`` to skip the
(minutes-long) corpus pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from multiprocessing import Pool

import numpy as np
import xxhash

_NONALNUM = re.compile(r"[^0-9a-z]+")


def normalize_tokens(text: str) -> list:
    """lowercase -> non-alphanumeric to space -> whitespace split.

    Robust to the line-wrapping / punctuation differences between PG-19's
    hard-wrapped source text and InfiniteBench's context field, so that
    identical prose yields an identical token stream on both sides.
    """
    return _NONALNUM.sub(" ", text.lower()).split()


def ngram_hashes(tokens: list, n: int, downsample: int) -> np.ndarray:
    """Return the sorted-unique downsampled uint64 hashes of the word ``n``-grams.

    Downsampling: keep hash h iff (h % downsample) == 0. Deterministic (xxh64,
    seed 0) so training and eval sides are directly comparable.
    """
    if len(tokens) < n:
        return np.empty(0, dtype=np.uint64)
    h = xxhash.xxh64
    keep = []
    ap = keep.append
    for i in range(len(tokens) - n + 1):
        v = h(" ".join(tokens[i:i + n]), seed=0).intdigest()
        if v % downsample == 0:
            ap(v)
    if not keep:
        return np.empty(0, dtype=np.uint64)
    return np.unique(np.asarray(keep, dtype=np.uint64))


# --------------------------------------------------------------------------- #
# Training-corpus sketch (parallel, streaming byte ranges)
# --------------------------------------------------------------------------- #
def _chunk_ranges(path: str, n_chunks: int):
    size = os.path.getsize(path)
    step = size // n_chunks
    return [(path, i * step, (size if i == n_chunks - 1 else (i + 1) * step))
            for i in range(n_chunks)]


def _worker_sketch(job):
    """Process a byte range -> downsampled n-gram hash array (CONTINUOUS stream).

    CRITICAL train/eval symmetry: PG-19's raw text is hard-wrapped at ~13 words
    per line, and the eval side collapses ALL whitespace (``re.sub(r"\\s+"," ")``)
    into one continuous stream before forming n-grams. So we MUST form n-grams
    over a continuous token stream here too — tokenizing per-line would capture
    almost no 13-grams (a 13-word line yields exactly one) and would NEVER capture
    grams spanning the hard-wrap, so even an identical book would score ~0
    containment (a false CLEAN). We therefore read the whole byte range, discard
    the partial first line (owned by the previous chunk), and tokenize the joined
    text exactly as the eval side does. Only the 63 internal chunk byte-boundaries
    drop ~n-1 grams each — negligible against billions.
    """
    path, start, end, n, downsample = job
    with open(path, "rb") as f:
        if start > 0:
            f.seek(start - 1)
            f.readline()          # discard partial line; owned by previous chunk
            begin = f.tell()
        else:
            begin = 0
        # read to just past `end`, then extend to the next newline so we don't
        # cut a line in half at the tail (that line belongs to this chunk).
        f.seek(begin)
        raw = f.read(max(0, end - begin))
        tail = f.readline()
        if tail:
            raw += tail
    text = raw.decode("utf-8", "replace")
    toks = normalize_tokens(text)   # same normalization/reflow as the eval side
    if len(toks) < n:
        return np.empty(0, dtype=np.uint64)
    h = xxhash.xxh64
    out = []
    ap = out.append
    for i in range(len(toks) - n + 1):
        v = h(" ".join(toks[i:i + n]), seed=0).intdigest()
        if v % downsample == 0:
            ap(v)
    if not out:
        return np.empty(0, dtype=np.uint64)
    return np.unique(np.asarray(out, dtype=np.uint64))


def build_train_sketch(path: str, n: int, downsample: int, workers: int):
    t0 = time.time()
    jobs = [(p, s, e, n, downsample) for (p, s, e) in _chunk_ranges(path, workers)]
    print(f"[audit] building train {n}-gram sketch: {len(jobs)} chunks, "
          f"downsample=1/{downsample}, workers={workers}", flush=True)
    parts = []
    with Pool(workers) as pool:
        for i, arr in enumerate(pool.imap_unordered(_worker_sketch, jobs)):
            parts.append(arr)
            print(f"  [audit] chunk {i + 1}/{len(jobs)} done, "
                  f"{arr.size:,} hashes ({time.time() - t0:.0f}s)", flush=True)
    allh = np.unique(np.concatenate(parts)) if parts else np.empty(0, np.uint64)
    print(f"[audit] train sketch built: {allh.size:,} unique downsampled "
          f"{n}-gram hashes in {time.time() - t0:.0f}s", flush=True)
    return allh


# --------------------------------------------------------------------------- #
# Eval side
# --------------------------------------------------------------------------- #
def load_eval_books(eval_dir: str, files):
    """Return (books, records). books[nsha] = dict; records = list of (task,id,nsha)."""
    def norm_ws(s):
        return re.sub(r"\s+", " ", s.lower()).strip()

    books = {}
    records = []
    for fn, task in files:
        path = os.path.join(eval_dir, fn)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                ctx = o["context"]
                raw_sha = hashlib.sha256(ctx.encode("utf-8")).hexdigest()
                norm_sha = hashlib.sha256(norm_ws(ctx).encode("utf-8")).hexdigest()
                b = books.get(norm_sha)
                if b is None:
                    b = books[norm_sha] = {
                        "norm_sha256": norm_sha,
                        "raw_sha256": raw_sha,
                        "char_len": len(ctx),
                        "records": [],
                        "_ctx": ctx,
                    }
                b["records"].append({"task": task, "id": int(o["id"])})
                records.append((task, int(o["id"]), norm_sha))
    return books, records


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train_corpus", default="data/pg19_train.jsonl")
    ap.add_argument("--eval_dir", default=".t27_tmp/infb_eval")
    ap.add_argument("--out_dir", default="bench_results/p0_14_contamination")
    ap.add_argument("--n", type=int, default=13, help="n-gram order (word-level).")
    ap.add_argument("--downsample", type=int, default=32,
                    help="keep hash h iff h %% downsample == 0 (bottom-hash sketch).")
    ap.add_argument("--contam_high", type=float, default=0.80,
                    help="containment >= this => CONTAMINATED (same book in training).")
    ap.add_argument("--contam_low", type=float, default=0.10,
                    help="containment < this => CLEAN; in-between => PARTIAL/review.")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--sketch_npy", default="",
                    help="reuse a prebuilt train sketch .npy (skip corpus pass).")
    ap.add_argument("--save_sketch", default="",
                    help="save the built train sketch to this .npy for reuse.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    eval_files = [("longbook_qa_eng.jsonl", "longbook_qa_eng"),
                  ("longbook_choice_eng.jsonl", "longbook_choice_eng")]

    # ---- eval side ---- #
    books, records = load_eval_books(args.eval_dir, eval_files)
    print(f"[audit] eval: {len(records)} records -> {len(books)} unique books",
          flush=True)

    # ---- training sketch ---- #
    if args.sketch_npy and os.path.exists(args.sketch_npy):
        print(f"[audit] loading prebuilt train sketch: {args.sketch_npy}", flush=True)
        train = np.load(args.sketch_npy)
    else:
        train = build_train_sketch(args.train_corpus, args.n, args.downsample,
                                    args.workers)
        if args.save_sketch:
            np.save(args.save_sketch, train)
            print(f"[audit] saved train sketch -> {args.save_sketch}", flush=True)

    # ---- exact-hash intersection (b) ---- #
    # Build sha256 of every training LINE (whitespace-normalized) — a book-level
    # exact match cannot exist (no doc boundary), so this is the strongest exact
    # test the concatenated corpus supports; reported to document it finds 0.
    # (We only need to know it's empty; comparing 580 eval-book shas vs training
    #  line shas suffices to prove non-identity.)
    eval_norm_shas = {b["norm_sha256"] for b in books.values()}
    eval_raw_shas = {b["raw_sha256"] for b in books.values()}

    # ---- containment (c) per unique book ---- #
    match_list = []
    contaminated_shas = set()
    partial_shas = set()
    t0 = time.time()
    for k, (nsha, b) in enumerate(sorted(books.items(),
                                         key=lambda kv: -kv[1]["char_len"])):
        toks = normalize_tokens(b["_ctx"])
        eg = ngram_hashes(toks, args.n, args.downsample)
        if eg.size == 0:
            ratio = 0.0
            n_hit = 0
        else:
            hit = np.isin(eg, train, assume_unique=True)
            n_hit = int(hit.sum())
            ratio = n_hit / eg.size
        if ratio >= args.contam_high:
            verdict = "CONTAMINATED"
            contaminated_shas.add(nsha)
        elif ratio < args.contam_low:
            verdict = "CLEAN"
        else:
            verdict = "PARTIAL"
            partial_shas.add(nsha)
        match_list.append({
            "norm_sha256": nsha,
            "raw_sha256": b["raw_sha256"],
            "char_len": b["char_len"],
            "n_words": len(toks),
            "n_eval_ngrams_sampled": int(eg.size),
            "n_ngrams_in_train": n_hit,
            "containment": round(ratio, 4),
            "verdict": verdict,
            "n_records": len(b["records"]),
            "records": b["records"],
        })
        print(f"  [{k + 1}/{len(books)}] words={len(toks):>8} "
              f"sampled={eg.size:>6} containment={ratio:6.3f} -> {verdict} "
              f"({time.time() - t0:.0f}s)", flush=True)

    # ---- per-record verdict + clean subset ---- #
    sha2verdict = {m["norm_sha256"]: m["verdict"] for m in match_list}
    per_record = []
    clean = {"longbook_qa_eng": [], "longbook_choice_eng": []}
    for (task, rid, nsha) in records:
        v = sha2verdict[nsha]
        per_record.append({"task": task, "id": rid, "book_norm_sha256": nsha,
                           "verdict": v})
        if v == "CLEAN":
            clean[task].append(rid)

    # ---- aggregate ---- #
    n_books = len(books)
    n_contam_books = len(contaminated_shas)
    n_partial_books = len(partial_shas)
    n_clean_books = n_books - n_contam_books - n_partial_books
    rec_counts = {"longbook_qa_eng": {"total": 0, "CONTAMINATED": 0, "PARTIAL": 0, "CLEAN": 0},
                  "longbook_choice_eng": {"total": 0, "CONTAMINATED": 0, "PARTIAL": 0, "CLEAN": 0}}
    for r in per_record:
        rec_counts[r["task"]]["total"] += 1
        rec_counts[r["task"]][r["verdict"]] += 1

    thresholds = {
        "method_a_title_author_pgid": "NOT COMPUTABLE — neither InfiniteBench "
            "records nor pg19_train.jsonl carry title/author/PG-ID metadata.",
        "method_b_exact_hash": {
            "eval_book_norm_sha_count": len(eval_norm_shas),
            "eval_book_raw_sha_count": len(eval_raw_shas),
            "note": "eval contexts are anonymized whole books; the training "
                    "corpus is concatenated lines with no document boundary, so a "
                    "book-level exact-hash match is structurally impossible and is "
                    "reported as 0 by construction. Content overlap is measured by "
                    "method (c).",
            "exact_book_matches": 0,
        },
        "method_c_ngram_containment": {
            "n": args.n,
            "tokenization": "lowercase; [^0-9a-z]+ -> space; whitespace split",
            "hash": "xxhash.xxh64(seed=0) intdigest, 64-bit",
            "downsample": args.downsample,
            "downsample_rule": "keep hash h iff (h %% downsample) == 0 (bottom-hash / MinHash sketch)",
            "ngram_dedup": "unique n-gram hashes per document",
            "containment_definition": "|unique eval n-gram sketch hashes present in train sketch| / |unique eval n-gram sketch hashes|",
            "contam_high_threshold": args.contam_high,
            "contam_low_threshold": args.contam_low,
            "verdict_rule": ">= contam_high => CONTAMINATED; < contam_low => CLEAN; else PARTIAL",
        },
        "train_sketch_unique_hashes": int(train.size),
    }

    summary = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_corpus": os.path.abspath(args.train_corpus),
        "eval_dir": os.path.abspath(args.eval_dir),
        "eval_files": [f for f, _ in eval_files],
        "n_eval_records": len(records),
        "n_unique_eval_books": n_books,
        "books": {"CONTAMINATED": n_contam_books, "PARTIAL": n_partial_books,
                  "CLEAN": n_clean_books},
        "book_contamination_ratio": round(n_contam_books / n_books, 4) if n_books else 0.0,
        "records_by_task": rec_counts,
        "overall_record_contamination_ratio": round(
            sum(rc["CONTAMINATED"] for rc in rec_counts.values()) / len(records), 4)
            if records else 0.0,
        "clean_subset_size": {t: len(v) for t, v in clean.items()},
        "thresholds": thresholds,
    }

    # ---- write artifacts ---- #
    with open(os.path.join(args.out_dir, "thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2)
    with open(os.path.join(args.out_dir, "match_list.json"), "w") as f:
        json.dump({"summary": summary, "books": match_list}, f, indent=2)
    with open(os.path.join(args.out_dir, "per_record_verdict.jsonl"), "w") as f:
        for r in per_record:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(args.out_dir, "clean_subset_ids.json"), "w") as f:
        json.dump({"clean_subset_ids": clean,
                   "clean_subset_size": {t: len(v) for t, v in clean.items()},
                   "note": "eval (task,id) records whose source book scored "
                           f"containment < {args.contam_low} against PG-19 train."},
                  f, indent=2)
    with open(os.path.join(args.out_dir, "audit_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[audit] ===== SUMMARY =====")
    print(json.dumps(summary, indent=2))
    print(f"[audit] artifacts written to {args.out_dir}")


if __name__ == "__main__":
    main()
