#!/usr/bin/env python3
"""N-gram overlap + dedup audit of the OLMo-2 SFT data vs the eval benchmarks
(MMLU / PopQA / TriviaQA). No GPU. Satisfies the P2.4 data-contamination check.

Given the SFT text dump (<tag>_text.jsonl from prepare_olmo2_sft_data.py) and the
three eval sets, this quantifies how much SFT content overlaps each eval:

  * n-gram containment: fraction of EVAL questions whose (default 8-gram) set has
    >= --hit_threshold Jaccard-free containment in the pooled SFT n-gram vocabulary
    (i.e. share of the eval question's n-grams that also appear anywhere in SFT).
    An eval item is a "hit" if its max per-item containment >= threshold.
  * exact-question dedup: fraction of eval questions whose normalised text appears
    verbatim as a substring of some SFT record (prompt or response).
  * deduped SFT scale: number of SFT records that would remain if we dropped every
    record sharing >= --hit_threshold n-gram containment with ANY eval question
    (the "cleaned" training scale, reported for transparency).

Report -> --out (json): per-eval hit rate, exact-match rate, deduped SFT counts,
n-gram config, plus a few example hits for manual inspection. This is descriptive
(the SFT sources are already filtered to exclude subject-MC / closed-book-QA by
prepare_olmo2_sft_data.py); the audit is the empirical backstop.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import string
import time

_PUNCT = str.maketrans("", "", string.punctuation)


def _log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def norm_tokens(text: str) -> list[str]:
    return text.lower().translate(_PUNCT).split()


def ngrams(tokens, n):
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def load_sft_text(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_eval_questions(proxy_env):
    """Return {eval_name: [question_str, ...]} for mmlu/popqa/triviaqa via HF."""
    from datasets import load_dataset
    out = {}
    try:
        d = load_dataset("cais/mmlu", "all", split="test")
        out["mmlu"] = [ex["question"].strip() for ex in d]
        _log(f"mmlu: {len(out['mmlu'])} questions")
    except Exception as e:
        _log(f"mmlu load FAILED: {e}")
    try:
        d = load_dataset("akariasai/PopQA", split="test")
        out["popqa"] = [ex["question"].strip() for ex in d]
        _log(f"popqa: {len(out['popqa'])} questions")
    except Exception as e:
        _log(f"popqa load FAILED: {e}")
    try:
        d = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
        out["triviaqa"] = [ex["question"].strip() for ex in d]
        _log(f"triviaqa: {len(out['triviaqa'])} questions")
    except Exception as e:
        _log(f"triviaqa load FAILED: {e}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sft_text", type=str, required=True,
                   help="<tag>_text.jsonl produced by prepare_olmo2_sft_data.py")
    p.add_argument("--out", type=str, default="data/olmo2_sft/overlap_audit.json")
    p.add_argument("--n", type=int, default=8, help="n-gram order")
    p.add_argument("--hit_threshold", type=float, default=0.5,
                   help="an eval item is a 'hit' if this fraction of its n-grams "
                        "appears in the pooled SFT n-gram vocabulary")
    p.add_argument("--max_examples", type=int, default=8,
                   help="how many example hits to record per eval for inspection")
    args = p.parse_args()

    sft_rows = load_sft_text(args.sft_text)
    _log(f"loaded {len(sft_rows)} SFT records from {args.sft_text}")

    # pooled SFT n-gram vocabulary + per-record n-gram sets (for dedup).
    sft_ngram_vocab = set()
    sft_record_ngrams = []
    sft_norm_texts = []
    for r in sft_rows:
        combined = (r.get("prompt", "") + " " + r.get("response", ""))
        toks = norm_tokens(combined)
        g = ngrams(toks, args.n)
        sft_record_ngrams.append(g)
        sft_ngram_vocab |= g
        sft_norm_texts.append(" ".join(toks))
    _log(f"SFT pooled {args.n}-gram vocab size = {len(sft_ngram_vocab):,}")

    evals = load_eval_questions(None)

    report = {
        "sft_text": args.sft_text,
        "n_sft_records": len(sft_rows),
        "ngram_n": args.n,
        "hit_threshold": args.hit_threshold,
        "sft_pooled_ngram_vocab": len(sft_ngram_vocab),
        "evals": {},
    }
    # records to drop for the "cleaned scale": any SFT record sharing >= threshold
    # of an eval question's n-grams (union across all evals).
    dropped_records = set()

    for ename, questions in evals.items():
        n_q = len(questions)
        n_hit = 0
        n_exact = 0
        containments = []
        examples = []
        for q in questions:
            qtoks = norm_tokens(q)
            qgrams = ngrams(qtoks, args.n)
            if not qgrams:
                containments.append(0.0)
                continue
            inter = len(qgrams & sft_ngram_vocab)
            cont = inter / len(qgrams)
            containments.append(cont)
            qnorm = " ".join(qtoks)
            exact = False
            if cont >= args.hit_threshold:
                n_hit += 1
                # find which SFT records overlap -> dedup + exact check
                for ri, rg in enumerate(sft_record_ngrams):
                    if not rg:
                        continue
                    if len(qgrams & rg) / len(qgrams) >= args.hit_threshold:
                        dropped_records.add(ri)
                        if qnorm and qnorm in sft_norm_texts[ri]:
                            exact = True
                if len(examples) < args.max_examples:
                    examples.append({"question": q, "containment": round(cont, 3)})
            if not exact:
                # cheap exact substring check even for sub-threshold items
                if qnorm and any(qnorm in t for t in sft_norm_texts):
                    exact = True
            if exact:
                n_exact += 1
        mean_cont = sum(containments) / max(len(containments), 1)
        report["evals"][ename] = {
            "n_questions": n_q,
            "n_hit": n_hit,
            "hit_rate": n_hit / max(n_q, 1),
            "n_exact_substring": n_exact,
            "exact_rate": n_exact / max(n_q, 1),
            "mean_containment": round(mean_cont, 4),
            "example_hits": examples,
        }
        _log(f"{ename}: hit_rate={n_hit/max(n_q,1):.4f} ({n_hit}/{n_q}) "
             f"exact={n_exact} mean_containment={mean_cont:.4f}")

    report["dedup"] = {
        "n_sft_records": len(sft_rows),
        "n_records_dropped_overlap": len(dropped_records),
        "n_records_clean": len(sft_rows) - len(dropped_records),
        "note": "records dropped if they share >= hit_threshold of ANY eval "
                "question's n-grams (union across mmlu/popqa/triviaqa)",
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    _log(f"WROTE audit -> {args.out}")
    _log(f"dedup: {len(dropped_records)} / {len(sft_rows)} SFT records overlap "
         f"an eval question -> clean scale {len(sft_rows)-len(dropped_records)}")


if __name__ == "__main__":
    main()
