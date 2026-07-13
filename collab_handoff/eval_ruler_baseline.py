#!/usr/bin/env python
"""Self-contained long-context BASELINE eval on RULER-style synthetic tasks.

For an external collaborator (A100 40GB). NO dependency on the QCMem repo —
only `transformers` + `torch`. Generates RULER-style NIAH / VT samples on the
fly, runs an HF causal LM under one of several long-context BASELINE methods,
and scores with the OFFICIAL RULER `string_match` recall (substring match of
gold answers in the model output). Writes one CSV per (method,task,length).

Methods
-------
  full        : full context, no compression (upper-bound reference; may OOM at
                long lengths on 40GB — cap length or use --dtype bfloat16 + sdpa).
  streaming   : StreamingLLM — model sees only [first --sink_tokens] ++
                [last --window_tokens] of the context (attention-sink + sliding
                window; same KV budget). Faithful for retrieval tasks: a needle
                outside the window is unrecoverable, which is exactly the point.

Usage (one cell)
----------------
  python eval_ruler_baseline.py \
      --model_path Qwen/Qwen3-8B --method full \
      --task niah_single --length 16k --num_samples 100 \
      --out_dir results/

Scoring
-------
RULER string_match recall = mean over samples of
  (# gold answer strings appearing as substrings in output) / (# gold answers).
This matches the official RULER `string_match_all` metric. DO NOT change it to
a regex / fuzzy match — comparability with our numbers depends on this exact
metric.
"""
from __future__ import annotations
import argparse, csv, json, os, random, string, time
import torch

# ----------------------------------------------------------------------------- #
# RULER-style synthetic sample generation (self-contained, deterministic per seed)
# ----------------------------------------------------------------------------- #
_HAYSTACK_SENT = (
    "The grass is green. The sky is blue. The sun is yellow. Here we go. "
    "There and back again. The quick brown fox jumps over the lazy dog. "
)

def _approx_len_tokens(s):  # rough char->token
    return len(s) // 4

def _pad_haystack(target_tokens):
    reps = max(1, (target_tokens * 4) // len(_HAYSTACK_SENT) + 1)
    return (_HAYSTACK_SENT * reps)

def make_niah_single(rng, ctx_tokens):
    """One needle: 'The special magic {word} number is {NUM}.' Question asks NUM."""
    key = "".join(rng.choice(string.ascii_lowercase) for _ in range(8))
    val = str(rng.randint(1000000, 9999999))
    needle = f" The special magic {key} number is {val}. "
    hay = _pad_haystack(ctx_tokens)
    pos = rng.randint(len(hay) // 5, 4 * len(hay) // 5)
    ctx = hay[:pos] + needle + hay[pos:]
    q = (f"\n\nWhat is the special magic {key} number mentioned in the text above? "
         f"The special magic {key} number is")
    return ctx + q, [val]

def make_niah_multikey(rng, ctx_tokens, n_keys=4):
    keys = ["".join(rng.choice(string.ascii_lowercase) for _ in range(8)) for _ in range(n_keys)]
    vals = [str(rng.randint(1000000, 9999999)) for _ in range(n_keys)]
    hay = _pad_haystack(ctx_tokens)
    parts, cur = [], 0
    step = max(1, len(hay) // (n_keys + 1))
    for i, (k, v) in enumerate(zip(keys, vals)):
        parts.append(hay[cur:cur + step]); cur += step
        parts.append(f" The special magic {k} number is {v}. ")
    parts.append(hay[cur:])
    ctx = "".join(parts)
    tgt = rng.randint(0, n_keys - 1)
    q = (f"\n\nWhat is the special magic {keys[tgt]} number mentioned in the text above? "
         f"The special magic {keys[tgt]} number is")
    return ctx + q, [vals[tgt]]

def make_vt(rng, ctx_tokens, chain_len=4):
    """Variable-tracking chain: VAR_0=NUM; VAR_1=VAR_0; ... ask final var's value."""
    val = str(rng.randint(10000, 99999))
    varnames = ["VAR_" + "".join(rng.choice(string.ascii_uppercase) for _ in range(4))
                for _ in range(chain_len)]
    stmts = [f" {varnames[0]} = {val}. "]
    for i in range(1, chain_len):
        stmts.append(f" {varnames[i]} = {varnames[i-1]}. ")
    hay = _pad_haystack(ctx_tokens)
    step = max(1, len(hay) // (chain_len + 1)); parts, cur = [], 0
    order = list(range(chain_len)); rng.shuffle(order)  # scatter chain out of order
    for oi in order:
        parts.append(hay[cur:cur + step]); cur += step; parts.append(stmts[oi])
    parts.append(hay[cur:]); ctx = "".join(parts)
    q = (f"\n\nAll variables above are assigned. What is the value of {varnames[-1]}? "
         f"The value of {varnames[-1]} is")
    return ctx + q, [val]

_GEN = {"niah_single": make_niah_single, "niah_multikey": make_niah_multikey, "vt": make_vt}
_LEN = {"1k":1000,"2k":2000,"4k":4000,"8k":8000,"16k":16000,"32k":32000,"64k":64000,"128k":128000}

# ----------------------------------------------------------------------------- #
def string_match_recall(output, golds):
    o = output.lower()
    return sum(1 for g in golds if str(g).lower() in o) / max(1, len(golds))

def build_streaming_ids(ids, sink, window):
    """Keep first `sink` + last `window` tokens (StreamingLLM sink+window)."""
    if ids.shape[1] <= sink + window:
        return ids
    return torch.cat([ids[:, :sink], ids[:, -window:]], dim=1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--method", choices=["full", "streaming"], default="full")
    ap.add_argument("--task", choices=list(_GEN), required=True)
    ap.add_argument("--length", choices=list(_LEN), required=True)
    ap.add_argument("--num_samples", type=int, default=100)
    ap.add_argument("--sink_tokens", type=int, default=4)
    ap.add_argument("--window_tokens", type=int, default=4096,
                    help="streaming window (KV budget). Match QCMem read budget for fair compare.")
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn", default="sdpa", help="sdpa or flash_attention_2 (flash needs pip install)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    dt = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dt, device_map="cuda",
        attn_implementation=args.attn, trust_remote_code=True).eval()
    dev = next(model.parameters()).device
    ctx_tok = _LEN[args.length]
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{args.method}_{args.task}_{args.length}"
    rows, recall_sum = [], 0.0
    t0 = time.time()
    for i in range(args.num_samples):
        rng = random.Random(args.seed * 100003 + i)
        text, golds = _GEN[args.task](rng, ctx_tok)
        ids = tok(text, return_tensors="pt", add_special_tokens=True).input_ids.to(dev)
        if args.method == "streaming":
            ids = build_streaming_ids(ids, args.sink_tokens, args.window_tokens)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        r = string_match_recall(gen, golds)
        recall_sum += r
        rows.append({"idx": i, "recall": r, "gold": "|".join(golds),
                     "n_ctx_tok": int(ids.shape[1]), "output": gen[:200].replace("\n", " ")})
    recall = 100.0 * recall_sum / args.num_samples
    with open(os.path.join(args.out_dir, tag + ".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    summary = {"model": args.model_path, "method": args.method, "task": args.task,
               "length": args.length, "num_samples": args.num_samples,
               "window_tokens": args.window_tokens if args.method == "streaming" else None,
               "recall": round(recall, 2), "secs": round(time.time() - t0, 1)}
    with open(os.path.join(args.out_dir, tag + ".json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"RESULT {tag}: recall={recall:.2f} ({args.num_samples} samples) -> {args.out_dir}/{tag}.csv")

if __name__ == "__main__":
    main()
