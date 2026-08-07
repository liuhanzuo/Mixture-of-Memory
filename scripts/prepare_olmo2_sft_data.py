#!/usr/bin/env python3
"""Prepare GENERAL instruction SFT data for the OLMo-2 P2.4 diagnostic (no GPU).

Streams a general instruction-tuning mixture (default: allenai/tulu-3-sft-mixture),
FILTERS OUT forbidden sources (subject multiple-choice / closed-book factual-QA /
academic-exam corpora that would contaminate the MMLU / PopQA / TriviaQA probes),
formats each conversation in a fixed plain-text role template, tokenises with the
OLMo-2 tokenizer, and PACKS response-only-masked sequences into fixed [N, seq_len]
arrays until a configurable TOKEN BUDGET is reached.

Why streaming + budget: the full Tulu-3 mix is ~939k conversations (multi-GB). We
only need ~1 epoch of a few hundred M tokens shared identically across the three
P2.4 arms, so we stream + shuffle-buffer + cap. Deterministic given --seed.

Outputs (under --out_dir, prefix from --tag):
  <tag>_input_ids.npy   [N, seq_len] uint32   packed token ids
  <tag>_labels.npy      [N, seq_len] int32    response-only labels (-100 elsewhere)
  <tag>_text.jsonl      one {source, prompt, response} row per USED conversation
                        (drives the n-gram overlap audit; NOT read at train time)
  <tag>_manifest.json   config + n_sequences / n_tokens / n_supervised_tokens +
                        per-source histogram + tokenizer/eos ids + git provenance

Response-only masking (standard open-instruct/Tulu convention): only assistant
turn tokens (and the trailing EOS after each assistant turn) carry a label; the
role tags, user turns, and pad carry -100. The eval AFTER SFT still uses the Paper
B base protocol (chat_template=False / no BOS / LL-based MC) unchanged -- the
format here is purely a training-time interface.

FORBIDDEN (never used as SFT data; --deny_sources default): any source whose name
matches a subject-MC / exam / closed-book-QA pattern (flan_v2, sciriff, mmlu, arc,
triviaqa, popqa, natural_questions, sciq, hendrycks, openbookqa, hellaswag,
commonsense, squad, coqa, drop, exam). The n-gram audit (audit_olmo2_sft_overlap.py)
is the empirical backstop that quantifies residual overlap with MMLU/PopQA/TriviaQA.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time

import numpy as np


def _log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# substring patterns -> a source is FORBIDDEN if its name contains any of these.
DEFAULT_DENY = [
    "flan", "sciriff", "mmlu", "arc", "triviaqa", "popqa",
    "natural_questions", "naturalquestions", "sciq", "hendrycks",
    "openbookqa", "hellaswag", "commonsense", "squad", "coqa", "drop_",
    "exam", "_qa_", "qa_converted",
]


def is_denied(source: str, deny_patterns) -> bool:
    s = (source or "").lower()
    return any(pat in s for pat in deny_patterns)


# fixed plain-text role template (base model has no chat template). Supervised
# span = the assistant content + trailing EOS after each assistant turn.
USER_PREFIX = "<|user|>\n"
ASSIST_PREFIX = "<|assistant|>\n"


def format_and_tokenize(messages, tok, eos_id):
    """Return (ids: list[int], label_mask: list[bool]) for one conversation.
    label_mask[i]=True iff token i is a supervised (assistant/eos) token."""
    ids: list[int] = []
    mask: list[bool] = []

    def _emit(text, supervised):
        toks = tok.encode(text, add_special_tokens=False)
        ids.extend(toks)
        mask.extend([supervised] * len(toks))

    for turn in messages:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            _emit(USER_PREFIX + content + "\n", supervised=False)
        elif role == "assistant":
            _emit(ASSIST_PREFIX, supervised=False)
            _emit(content, supervised=True)
            ids.append(eos_id)
            mask.append(True)  # supervise the EOS so the model learns to stop
        elif role == "system":
            _emit(content + "\n", supervised=False)
        # ignore any other roles
    return ids, mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="allenai/tulu-3-sft-mixture")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--tokenizer_path", type=str, default="../models/OLMo-2-1124-7B")
    p.add_argument("--out_dir", type=str, default="data/olmo2_sft")
    p.add_argument("--tag", type=str, default="tulu3_general")
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--token_budget", type=int, default=250_000_000,
                   help="stop after packing >= this many total tokens (~1 epoch "
                        "worth; identical budget must be reused across the 3 arms)")
    p.add_argument("--max_conversations", type=int, default=0,
                   help=">0 also caps the number of conversations consumed")
    p.add_argument("--shuffle_buffer", type=int, default=50000,
                   help="streaming shuffle buffer for source diversity (0=off)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deny_sources", type=str, default=",".join(DEFAULT_DENY),
                   help="comma-separated substring patterns; a source containing "
                        "any is EXCLUDED (subject-MC / closed-book-QA guard)")
    p.add_argument("--max_response_chars", type=int, default=0,
                   help=">0 drops conversations whose assistant text is longer "
                        "(sanity guard against pathological rows); 0=off")
    p.add_argument("--dry_run", action="store_true",
                   help="consume only 200 conversations + budget<=1M to smoke the "
                        "pipeline WITHOUT downloading/tokenising the whole set")
    args = p.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    deny = [x.strip().lower() for x in args.deny_sources.split(",") if x.strip()]
    os.makedirs(args.out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    eos_id = tok.eos_token_id
    assert eos_id is not None, "OLMo-2 tokenizer must have an eos id"

    budget = 1_000_000 if args.dry_run else args.token_budget
    max_conv = 200 if args.dry_run else args.max_conversations

    _log(f"streaming {args.dataset}:{args.split} | budget={budget:,} tok "
         f"seq_len={args.seq_len} deny={deny}")
    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    if args.shuffle_buffer and args.shuffle_buffer > 0:
        ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    L = args.seq_len
    packed_ids: list[np.ndarray] = []   # each [L] uint32
    packed_lab: list[np.ndarray] = []   # each [L] int32
    buf_ids: list[int] = []
    buf_lab: list[int] = []             # token id where supervised else -100

    n_rows_dropped_zero_sup = 0

    def _flush_full():
        # ★ A packed row MUST contain >=1 supervised token. We pack by
        # CONCATENATION across conversation boundaries, and the any(mask) check
        # upstream is PER CONVERSATION -- so an [L]-token row can still land
        # entirely inside an unsupervised span (e.g. a long user turn), yielding
        # an all -100 label row. F.cross_entropy(..., ignore_index=-100) on an
        # all-ignored target returns NaN, and under DDP that NaN all-reduces into
        # every rank's gradients and poisons ALL weights (observed 2026-08-07:
        # 14330/122070 = 11.74% such rows -> loss=nan from step 1).
        nonlocal n_rows_dropped_zero_sup
        while len(buf_ids) >= L:
            chunk_ids = np.asarray(buf_ids[:L], dtype=np.uint32)
            chunk_lab = np.asarray(buf_lab[:L], dtype=np.int32)
            if bool((chunk_lab != -100).any()):
                packed_ids.append(chunk_ids)
                packed_lab.append(chunk_lab)
            else:
                n_rows_dropped_zero_sup += 1
            del buf_ids[:L]
            del buf_lab[:L]

    n_seen = n_used = n_denied = n_empty = n_toolong = 0
    n_total_tok = n_sup_tok = 0
    src_hist: dict[str, int] = {}
    text_path = os.path.join(args.out_dir, f"{args.tag}_text.jsonl")
    tf = open(text_path, "w")

    for ex in ds:
        n_seen += 1
        if max_conv and n_used >= max_conv:
            break
        if n_total_tok >= budget:
            break
        source = ex.get("source", "?")
        if is_denied(source, deny):
            n_denied += 1
            continue
        messages = ex.get("messages")
        if not messages:
            n_empty += 1
            continue
        if args.max_response_chars > 0:
            resp_chars = sum(len(m.get("content") or "") for m in messages
                             if m.get("role") == "assistant")
            if resp_chars > args.max_response_chars:
                n_toolong += 1
                continue
        ids, mask = format_and_tokenize(messages, tok, eos_id)
        if not ids or not any(mask):
            n_empty += 1
            continue
        labels = [ids[i] if mask[i] else -100 for i in range(len(ids))]
        buf_ids.extend(ids)
        buf_lab.extend(labels)
        _flush_full()
        n_used += 1
        n_total_tok += len(ids)
        n_sup_tok += sum(mask)
        src_hist[source] = src_hist.get(source, 0) + 1
        # compact text record for the overlap audit (prompt=user text, response=asst)
        prompt = " ".join((m.get("content") or "") for m in messages
                          if m.get("role") in ("user", "system"))
        resp = " ".join((m.get("content") or "") for m in messages
                        if m.get("role") == "assistant")
        tf.write(json.dumps({"source": source, "prompt": prompt[:4000],
                             "response": resp[:4000]}) + "\n")
        if n_used % 5000 == 0:
            _log(f"  used={n_used:,} seq={len(packed_ids):,} "
                 f"tok={n_total_tok:,}/{budget:,} denied={n_denied:,}")
    tf.close()

    if not packed_ids:
        raise RuntimeError("no packed sequences produced (budget too small / all "
                           "sources denied?)")
    ids_arr = np.stack(packed_ids)   # [N, L] uint32
    lab_arr = np.stack(packed_lab)   # [N, L] int32
    ids_path = os.path.join(args.out_dir, f"{args.tag}_input_ids.npy")
    lab_path = os.path.join(args.out_dir, f"{args.tag}_labels.npy")
    np.save(ids_path, ids_arr)
    np.save(lab_path, lab_arr)

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        commit = "unknown"

    n_packed_sup = int((lab_arr != -100).sum())
    manifest = {
        "dataset": args.dataset,
        "split": args.split,
        "tag": args.tag,
        "seq_len": L,
        "token_budget": budget,
        "seed": args.seed,
        "shuffle_buffer": args.shuffle_buffer,
        "deny_sources": deny,
        "tokenizer_path": args.tokenizer_path,
        "eos_id": eos_id,
        "n_sequences": int(ids_arr.shape[0]),
        "n_rows_dropped_zero_supervised": n_rows_dropped_zero_sup,
        "n_conversations_used": n_used,
        "n_conversations_seen": n_seen,
        "n_conversations_denied": n_denied,
        "n_conversations_empty": n_empty,
        "n_conversations_toolong": n_toolong,
        "n_tokens_formatted": n_total_tok,
        "n_supervised_tokens_formatted": n_sup_tok,
        "n_tokens_packed": int(ids_arr.size),
        "n_supervised_tokens_packed": n_packed_sup,
        "source_histogram": dict(sorted(src_hist.items(),
                                        key=lambda kv: -kv[1])),
        "input_ids_path": ids_path,
        "labels_path": lab_path,
        "text_path": text_path,
        "commit": commit,
        "dry_run": args.dry_run,
    }
    man_path = os.path.join(args.out_dir, f"{args.tag}_manifest.json")
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)
    _log(f"WROTE {ids_arr.shape} ids -> {ids_path}")
    _log(f"WROTE {lab_arr.shape} labels ({n_packed_sup:,} supervised) -> {lab_path}")
    _log(f"WROTE manifest -> {man_path}")
    _log(f"used={n_used:,} conversations from {len(src_hist)} sources | "
         f"denied={n_denied:,} | packed {ids_arr.shape[0]:,} seqs "
         f"({ids_arr.size:,} tok, {n_packed_sup:,} supervised)")


if __name__ == "__main__":
    main()
