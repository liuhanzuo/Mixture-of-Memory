#!/usr/bin/env python3
"""OLMo-2 prune-then-heal: train-eval CONTAMINATION / overlap audit (CPU only).

Training corpus  = the Dolmino (DCLM subset) continuation stream the probe arms
were healed on, in its EXACT tokenized form: data/dolmino_now15b.npy
(shape [7570911, 2048] uint32, 15.5B OLMo-2 tokens; packed contiguous with an EOS
between docs -- verified from scripts/tokenize_dolmino_olmo2.py). Using the
tokenized corpus (not decoded text) means the overlap test is over the exact token
stream the model saw, with the SAME tokenizer on both sides (no tokenizer-mismatch
false negatives).

Eval benchmarks  = MMLU (cais/mmlu all/test), PopQA (akariasai/PopQA/test),
TriviaQA (mandarjoshi/trivia_qa rc.nocontext/validation), NQ-open
(google-research-datasets/nq_open/validation). Overlap is measured on the QUESTION
text (GPT-3-style question-overlap decontamination); answers are facts that occur
ubiquitously and are not themselves a contamination signal.

METHOD (long token-n-gram containment, exact hashing, INVERTED single streaming pass)
------------------------------------------------------------------------------------
1. Tokenize every eval question with the OLMo-2 tokenizer (add_special_tokens=False,
   matching the training tokenization). Form its word... token n-grams for each n in
   --ns (default 13,8). Questions shorter than n contribute their whole token seq as
   one gram. Hash each n-gram with a 64-bit polynomial (Horner) hash.
2. Pool ALL eval n-gram hashes (across benchmarks) into one sorted uint64 array per n
   (the "eval sketch"). Because we do NOT downsample, short questions keep full
   resolution -- unlike a MinHash bottom-hash sketch, which is only valid for long
   documents.
3. Stream the ENTIRE training token corpus once (multiprocess over row-blocks): for
   each block compute all token n-gram hashes (vectorized numpy Horner), and mark
   which EVAL n-grams are present (searchsorted membership). OR-accumulate across
   blocks. 64-bit hash collisions between the ~1e6 eval grams and ~1e10 train grams
   are ~1e-4 expected total -> negligible false positives.
4. Per question: containment = (# of its n-grams present in training) / (# n-grams).
   Verdict: >= --contam_high => CONTAMINATED; < --contam_low => CLEAN; else PARTIAL.
5. Emit per-benchmark contamination rates, clean-subset id lists, thresholds, and the
   highest-containment example questions for manual inspection.

Outputs under --out_dir:
  contamination_summary.json, per_benchmark_<bench>_n<n>.json,
  clean_subset_ids.json, thresholds.json, examples.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

P64 = np.uint64(1099511628211)   # FNV-ish odd 64-bit multiplier
MASK = np.uint64(0xFFFFFFFFFFFFFFFF)

# globals set per worker
_NPY = None
_NS = None
_UNIQ = None  # dict n -> sorted uint64 array


def _log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def ngram_hashes_1d(x_u64: np.ndarray, n: int) -> np.ndarray:
    """Vectorized 64-bit polynomial (Horner) hash of every length-n token n-gram in
    the 1-D uint64 token array x. Returns hashes for positions 0..len-n (uint64,
    wraps mod 2^64). Identical recurrence to _ngram_hashes_seq (eval side)."""
    L = x_u64.shape[0]
    if L < n:
        return np.empty(0, dtype=np.uint64)
    acc = x_u64[0:L - n + 1].copy()
    for k in range(1, n):
        acc = acc * P64 + x_u64[k:L - n + 1 + k]
    return acc


def _ngram_hashes_seq(tokens, n):
    """Same Horner hash for a single python token list -> list of uint64.
    Returns [] when len<n (a length-L<n gram could never match a fixed-length-n
    train gram, so such questions are 'undecidable' at order n and handled by the
    caller as SHORT rather than silently CLEAN)."""
    L = len(tokens)
    if L < n:
        return []
    out = []
    for i in range(L - n + 1):
        h = np.uint64(0)
        for k in range(n):
            h = h * P64 + np.uint64(int(tokens[i + k]))
        out.append(int(h & MASK))
    return out


def _init_worker(npy_path, ns, uniq_map):
    global _NPY, _NS, _UNIQ
    _NPY = np.load(npy_path, mmap_mode="r")
    _NS = ns
    _UNIQ = uniq_map


def _process_block(rng):
    start, end = rng
    block = np.asarray(_NPY[start:end]).astype(np.uint64).reshape(-1)
    hits = {}
    for n in _NS:
        uniq = _UNIQ[n]
        h = ngram_hashes_1d(block, n)
        if h.size == 0:
            hits[n] = np.empty(0, dtype=np.int64)
            continue
        idx = np.searchsorted(uniq, h)
        idx = np.clip(idx, 0, len(uniq) - 1)
        valid = uniq[idx] == h
        hit_idx = np.unique(idx[valid])
        hits[n] = hit_idx.astype(np.int64)
    return hits


def load_benchmarks():
    from datasets import load_dataset
    B = {}
    d = load_dataset("cais/mmlu", "all", split="test")
    B["mmlu"] = [(i, ex["question"].strip()) for i, ex in enumerate(d)]
    _log(f"mmlu: {len(B['mmlu'])} questions (item_id = dataset index)")
    d = load_dataset("akariasai/PopQA", split="test")
    B["popqa"] = [(int(ex["id"]), ex["question"].strip()) for ex in d]
    _log(f"popqa: {len(B['popqa'])}")
    d = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    B["triviaqa"] = [(i, ex["question"].strip()) for i, ex in enumerate(d)]
    _log(f"triviaqa: {len(B['triviaqa'])}")
    d = load_dataset("google-research-datasets/nq_open", split="validation")
    B["nq_open"] = [(i, ex["question"].strip()) for i, ex in enumerate(d)]
    _log(f"nq_open: {len(B['nq_open'])}")
    return B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_npy", default="/dev/shm/dolmino_now15b.npy")
    ap.add_argument("--tokenizer",
                    default="/apdcephfs_wzc1/share_304376610/pighzliu_code/models/OLMo-2-1124-7B")
    ap.add_argument("--ns", default="13,8", help="comma list of token n-gram orders")
    ap.add_argument("--contam_high", type=float, default=0.80)
    ap.add_argument("--contam_low", type=float, default=0.10)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--block_rows", type=int, default=8000)
    ap.add_argument("--out_dir", default="bench_results/olmo2_dolmino_contamination")
    ap.add_argument("--limit_train_rows", type=int, default=0, help=">0 = sanity cap")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    ns = [int(x) for x in args.ns.split(",")]

    # ---- eval side ----
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    benches = load_benchmarks()

    # per (bench): list of dicts {qid, n_tokens, grams:{n:[hashes]}}
    all_hashes = {n: [] for n in ns}
    qrecs = {}
    for bench, items in benches.items():
        recs = []
        for (qid, q) in items:
            toks = tok(q, add_special_tokens=False)["input_ids"]
            grams = {}
            for n in ns:
                hs = _ngram_hashes_seq(toks, n)
                grams[n] = hs
                all_hashes[n].extend(hs)
            recs.append({"qid": qid, "q": q, "n_tokens": len(toks), "grams": grams})
        qrecs[bench] = recs
    uniq_map = {}
    for n in ns:
        arr = np.array(all_hashes[n], dtype=np.uint64) if all_hashes[n] else np.empty(0, np.uint64)
        uniq_map[n] = np.unique(arr)
        _log(f"eval sketch n={n}: {uniq_map[n].size:,} unique n-gram hashes "
             f"(from {len(all_hashes[n]):,} total)")
    del all_hashes

    # ---- train streaming pass ----
    arr = np.load(args.train_npy, mmap_mode="r")
    nrows = arr.shape[0] if not args.limit_train_rows else min(args.limit_train_rows, arr.shape[0])
    seq_len = arr.shape[1]
    _log(f"train {args.train_npy} shape={arr.shape} scanning {nrows} rows "
         f"({nrows*seq_len/1e9:.2f}B tokens) | ns={ns} workers={args.workers} "
         f"block_rows={args.block_rows}")
    blocks = [(s, min(s + args.block_rows, nrows)) for s in range(0, nrows, args.block_rows)]
    present = {n: np.zeros(uniq_map[n].size, dtype=bool) for n in ns}
    t0 = time.time()
    done = 0
    with Pool(args.workers, initializer=_init_worker,
              initargs=(args.train_npy, ns, uniq_map)) as pool:
        for hits in pool.imap_unordered(_process_block, blocks, chunksize=1):
            for n in ns:
                hi = hits[n]
                if hi.size:
                    present[n][hi] = True
            done += 1
            if done % 20 == 0 or done == len(blocks):
                el = time.time() - t0
                rate = done * args.block_rows * seq_len / max(el, 1e-9)
                _log(f"  block {done}/{len(blocks)} | {rate/1e6:.1f}M tok/s | "
                     f"present n{ns[0]}={int(present[ns[0]].sum()):,}/{uniq_map[ns[0]].size:,} "
                     f"({el:.0f}s)")

    # ---- per-question containment + verdicts ----
    summary = {"date": time.strftime("%Y-%m-%d %H:%M:%S"),
               "train_npy": os.path.abspath(args.train_npy),
               "train_rows_scanned": int(nrows), "seq_len": int(seq_len),
               "train_tokens_scanned": int(nrows * seq_len),
               "ns": ns, "contam_high": args.contam_high,
               "contam_low": args.contam_low, "benchmarks": {}}
    clean_ids = {}
    examples = {}
    for bench, recs in qrecs.items():
        summary["benchmarks"][bench] = {}
        clean_ids[bench] = {}
        examples[bench] = {}
        for n in ns:
            uniq = uniq_map[n]
            pres = present[n]
            n_c = n_p = n_cl = n_short = 0
            conts = []
            clean_list = []          # strictly CLEAN (< low)
            contaminated_list = []   # >= high
            per = []
            for r in recs:
                hs = r["grams"][n]
                if not hs:
                    per.append((-1.0, r["qid"], r["q"], "SHORT"))
                    n_short += 1
                    continue
                idx = np.searchsorted(uniq, np.array(hs, dtype=np.uint64))
                idx = np.clip(idx, 0, len(uniq) - 1)
                ok = pres[idx]
                cont = float(ok.mean())
                conts.append(cont)
                if cont >= args.contam_high:
                    n_c += 1; verdict = "CONTAMINATED"; contaminated_list.append(r["qid"])
                elif cont < args.contam_low:
                    n_cl += 1; verdict = "CLEAN"; clean_list.append(r["qid"])
                else:
                    n_p += 1; verdict = "PARTIAL"
                per.append((cont, r["qid"], r["q"], verdict))
            conts = np.array(conts) if conts else np.array([0.0])
            keep = [r["qid"] for r in recs if r["qid"] not in set(contaminated_list)]
            summary["benchmarks"][bench][f"n{n}"] = {
                "n_questions": len(recs),
                "n_decidable": len(recs) - n_short,
                "CONTAMINATED": n_c, "PARTIAL": n_p, "CLEAN": n_cl, "SHORT": n_short,
                "contam_rate": round(n_c / max(len(recs), 1), 5),
                "partial_rate": round(n_p / max(len(recs), 1), 5),
                "clean_rate": round(n_cl / max(len(recs), 1), 5),
                "mean_containment": round(float(conts.mean()), 5),
                "p95_containment": round(float(np.percentile(conts, 95)), 5),
                "max_containment": round(float(conts.max()), 5),
                "keep_for_recompute_size": len(keep),
            }
            clean_ids[bench][f"n{n}"] = {
                "contaminated_ids": sorted(contaminated_list),
                "clean_strict_ids": sorted(clean_list),
                "keep_for_recompute_ids": sorted(keep),
                "note": "keep_for_recompute = all qids EXCEPT CONTAMINATED "
                        "(the standard decontaminated subset for gap recompute)",
            }
            top = sorted(per, key=lambda z: -z[0])[:10]
            examples[bench][f"n{n}"] = [
                {"containment": round(c, 4), "qid": q, "verdict": v, "question": qq[:200]}
                for (c, q, qq, v) in top]
            # write per-benchmark per-record verdict jsonl (primary n only, to bound size)
        # per-record verdict for the primary n
        pn = ns[0]
        uniq = uniq_map[pn]; pres = present[pn]
        with open(os.path.join(args.out_dir, f"per_record_{bench}_n{pn}.jsonl"), "w") as f:
            for r in recs:
                hs = r["grams"][pn]
                if hs:
                    idx = np.clip(np.searchsorted(uniq, np.array(hs, np.uint64)), 0, len(uniq) - 1)
                    cont = float(pres[idx].mean())
                else:
                    cont = 0.0
                v = ("CONTAMINATED" if cont >= args.contam_high
                     else "CLEAN" if cont < args.contam_low else "PARTIAL")
                f.write(json.dumps({"qid": r["qid"], "containment": round(cont, 4),
                                    "verdict": v, "n_tokens": r["n_tokens"]}) + "\n")

    thresholds = {
        "method": "long token-n-gram containment vs the exact tokenized Dolmino "
                  "continuation corpus (inverted single streaming pass, no downsampling)",
        "tokenizer": args.tokenizer,
        "ns": ns,
        "hash": "64-bit polynomial (Horner), P=1099511628211, mod 2^64; "
                "collision FP ~1e-4 total (negligible)",
        "unit": "question text (add_special_tokens=False)",
        "containment_def": "|question n-grams present in training| / |question n-grams|; "
                           "questions with <n tokens use their whole token sequence as one gram",
        "verdict_rule": ">= contam_high CONTAMINATED; < contam_low CLEAN; else PARTIAL",
        "contam_high": args.contam_high, "contam_low": args.contam_low,
        "eval_sketch_unique_hashes": {str(n): int(uniq_map[n].size) for n in ns},
        "train_tokens_scanned": int(nrows * seq_len),
    }
    with open(os.path.join(args.out_dir, "thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2)
    with open(os.path.join(args.out_dir, "contamination_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out_dir, "clean_subset_ids.json"), "w") as f:
        json.dump(clean_ids, f, indent=2)
    with open(os.path.join(args.out_dir, "examples.json"), "w") as f:
        json.dump(examples, f, indent=2)
    _log("===== CONTAMINATION SUMMARY =====")
    print(json.dumps(summary["benchmarks"], indent=2))
    _log(f"artifacts -> {args.out_dir}")


if __name__ == "__main__":
    main()
