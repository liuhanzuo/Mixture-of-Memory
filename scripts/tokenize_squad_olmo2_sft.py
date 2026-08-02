#!/usr/bin/env python
"""Tokenise the on-disk SQuAD jsonl into packed [N, seq_len] uint32 chunks that
match NpyChunkDataset (scripts/train_semantic_bottleneck_1b.py), for Paper C
P-C1 supervised finetuning.

Why packed full-LM format: the tested trainer scripts/train_olmo2_arch_probe2.py
(A1/A3/A4 arms) consumes NpyChunkDataset, which does input_ids=labels (full-LM
loss over the packed chunk). To reuse that trainer VERBATIM (zero risk to the
concurrent Paper B runs), we produce the identical packed uint32 shard format the
dolmino tokeniser produces -- just sourced from SQuAD SFT strings instead of DCLM.

SFT string per example (English; the on-disk jsonl carries a Chinese task prefix
"根据以下对话记录，回答问题：<English question>" that we strip on the fullwidth
colon so the SFT text is clean English):

    Context: {context}\n\nQuestion: {question}\n\nAnswer: {answer}

where context = " ".join(memory_texts) (the SQuAD passage sentences),
question = the English question, answer = target_text. An OLMo-2 EOS (100257) is
appended after each example, then the token stream is packed into contiguous
seq_len chunks (the <seq_len tail is dropped, matching the dolmino packer).

Loss note: this is FULL-LM loss over the packed chunk (prompt + answer), the
minimal-viable choice the task permits; all four P-C1 arms train on the identical
packed shard + identical loss, so the freeze-graft/LoRA/full-FT/from-scratch
comparison is clean. (Answer-only masking would need a labels array + a trainer
fork; deferred to keep the tested trainer untouched.)
"""
import argparse
import array
import json
import os

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

EOS_ID = 100257  # OLMo-2 <|endoftext|>
CN_PREFIX_SEP = "："  # fullwidth colon separating the CN task prefix from the EN question


def _clean_question(input_text: str) -> str:
    """Strip the Chinese task prefix ('...：') if present, return the English question."""
    if CN_PREFIX_SEP in input_text:
        return input_text.split(CN_PREFIX_SEP, 1)[1].strip()
    return input_text.strip()


def _build_sft_text(rec: dict) -> str:
    q = _clean_question(rec.get("input_text", ""))
    ans = (rec.get("target_text") or "").strip()
    mem = rec.get("memory_texts") or []
    context = " ".join(m.strip() for m in mem if isinstance(m, str) and m.strip())
    if context:
        return f"Context: {context}\n\nQuestion: {q}\n\nAnswer: {ans}"
    return f"Question: {q}\n\nAnswer: {ans}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_npy", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=1000)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)

    texts = []
    with open(args.in_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            texts.append(_build_sft_text(rec))
    print(f"[tok] {len(texts)} SFT examples from {args.in_jsonl}", flush=True)

    buf = array.array("I")  # uint32
    n_ans_tok = 0
    for i in range(0, len(texts), args.batch):
        enc = tok(texts[i:i + args.batch], add_special_tokens=False)["input_ids"]
        for ids in enc:
            buf.extend(ids)
            buf.append(EOS_ID)
    n_tokens = len(buf)
    n_chunks = n_tokens // args.seq_len
    if n_chunks == 0:
        raise SystemExit(f"[tok] only {n_tokens} tokens < seq_len {args.seq_len}")
    arr = np.frombuffer(buf, dtype=np.uint32)[: n_chunks * args.seq_len]
    arr = arr.reshape(n_chunks, args.seq_len)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_npy)), exist_ok=True)
    tmp = args.out_npy + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, args.out_npy)
    approx_epochs_2000 = 2000 * 128 / n_chunks
    print(f"[tok] wrote {args.out_npy} shape={arr.shape} dtype={arr.dtype} "
          f"| {n_tokens/1e6:.2f}M tokens -> {n_chunks} chunks of {args.seq_len} "
          f"| eff_bs128 x 2000 steps ~= {approx_epochs_2000:.0f} epochs over these chunks",
          flush=True)


if __name__ == "__main__":
    main()
