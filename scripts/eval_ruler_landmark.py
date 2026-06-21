"""RULER NIAH/variable_tracking evaluation for faithful Landmark Attention.

Run with external/landmark_venv/bin/python on an H20 node. This script is
intentionally separate from eval_ruler_mem_space.py so the Landmark torch2.1 /
transformers4.28 environment does not import the mem_space Llama-3 path.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANDMARK_LLAMA_DIR = os.path.join(PROJECT_ROOT, "external", "landmark-attention", "llama")
for _p in (PROJECT_ROOT, LANDMARK_LLAMA_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.run_babilong_landmark import (  # noqa: E402
    generate_landmark,
    insert_landmark_tokens,
    load_landmark_pipeline,
)

NIAH_TEMPLATE = (
    "Some special magic {type_needle_v} are hidden within the following text. "
    "Make sure to memorize it. I will quiz you about the {type_needle_v} "
    "afterwards.\n{context}\nWhat are all the special magic {type_needle_v} for "
    "{query} mentioned in the provided text?"
)
NIAH_ANSWER_PREFIX = (
    " The special magic {type_needle_v} for {query} mentioned in the provided "
    "text are"
)
NIAH_NEEDLE = "One of the special magic {type_needle_v} for {key} is: {value}."
VT_TEMPLATE = (
    "Memorize and track the chain(s) of variable assignment hidden in the "
    "following text.\n\n{context}\nQuestion: Find all variables that are "
    "assigned the value {query} in the text above."
)
VT_ANSWER_PREFIX = (
    " Answer: According to the chain(s) of variable assignment in the text "
    "above, {num_v} variables are assigned the value {query}, they are: "
)
NOISE_HAYSTACK = (
    "The grass is green. The sky is blue. The sun is yellow. Here we go. "
    "There and back again."
)
_LENGTH_TOKENS = {"1k": 1024, "2k": 2048, "4k": 4096, "8k": 8192, "16k": 16384, "32k": 32768}
DEPTHS = [int(round(x)) for x in __import__("numpy").linspace(0, 100, num=40)]
_ESSAY_WORDS_CACHE: list[str] | None = None
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def _stable_hash(text: str) -> int:
    return sum((i + 1) * ord(c) for i, c in enumerate(text)) % 100000


def _load_essay_words() -> list[str]:
    global _ESSAY_WORDS_CACHE
    if _ESSAY_WORDS_CACHE is not None:
        return _ESSAY_WORDS_CACHE
    path = os.path.join(PROJECT_ROOT, "data", "pg19_train.jsonl")
    with open(path, "r", errors="ignore") as f:
        text = f.read(8_000_000)
    text = re.sub(r"\s+", " ", text).strip()
    _ESSAY_WORDS_CACHE = text.split(" ")
    return _ESSAY_WORDS_CACHE


def _sent_tokenize(text: str) -> list[str]:
    return [s for s in _SENT_RE.split(text.strip()) if s]


def _rand_word(rng: random.Random) -> str:
    def w() -> str:
        return "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(4, 8)))
    return f"{w()}-{w()}"


def _rand_number(rng: random.Random, num_digits: int = 7) -> str:
    return str(rng.randint(10 ** (num_digits - 1), 10 ** num_digits - 1))


def _make_niah(num_haystack: int, type_haystack: str, num_needle_k: int, rng: random.Random):
    keys, values, needles = [], [], []
    for _ in range(num_needle_k):
        k = _rand_word(rng)
        v = _rand_number(rng)
        keys.append(k)
        values.append([v])
        needles.append(NIAH_NEEDLE.format(type_needle_v="numbers", key=k, value=v))
    random.Random(rng.randint(0, 10 ** 9)).shuffle(needles)

    if type_haystack == "noise":
        sentences = [NOISE_HAYSTACK] * num_haystack
        idxs = sorted(rng.sample(range(num_haystack), len(needles)), reverse=True)
        for index, element in zip(idxs, needles):
            sentences.insert(index, element)
        context = "\n".join(sentences)
    else:
        words = _load_essay_words()
        text = " ".join((words * ((num_haystack + len(words) - 1) // len(words)))[:num_haystack])
        sents = _sent_tokenize(text) or [text]
        chosen = rng.sample(DEPTHS, min(len(needles), len(DEPTHS)))
        ins = [0] + sorted(int(len(sents) * (d / 100)) for d in chosen) + [len(sents)]
        parts = []
        for i in range(1, len(ins)):
            parts.append(" ".join(sents[ins[i - 1]:ins[i]]))
            if i - 1 < len(needles):
                parts.append(needles[i - 1])
        context = " ".join(parts)
    return context, keys[0], values[0]


def _render_niah(context: str, query: str) -> str:
    full = NIAH_TEMPLATE + NIAH_ANSWER_PREFIX
    full = full.replace("Some", "A").replace("are all", "is").replace("are", "is")
    full = full.replace("answers", "answer")
    return full.format(type_needle_v="number", context=context, query=query)


def _gen_chain(num_hops: int, rng: random.Random, icl: bool = False):
    k = 3 if icl else 5
    nvars = num_hops + 1
    vars_all: list[str] = []
    while len(set(vars_all)) < nvars:
        vars_all.append("".join(rng.choices(string.ascii_uppercase, k=k)))
    vars_all = list(dict.fromkeys(vars_all))[:nvars]
    first_val = "12345" if icl else str(rng.randint(10000, 99999))
    chain = [f"VAR {vars_all[0]} = {first_val}"]
    for j in range(num_hops):
        chain.append(f"VAR {vars_all[j + 1]} = VAR {vars_all[j]} ")
    return vars_all, chain, first_val


def _make_vt(num_noises: int, num_hops: int, rng: random.Random):
    vars_all, chain, value = _gen_chain(num_hops, rng)
    sentences = [NOISE_HAYSTACK] * num_noises
    positions = sorted(rng.sample(range(len(sentences)), len(chain)))
    for offset, (pos, c) in enumerate(zip(positions, chain)):
        sentences.insert(pos + offset, c)
    return "\n".join(sentences).replace(". \n", ".\n"), value, vars_all, num_hops + 1


def _make_vt_icl(rng: random.Random, num_hops: int) -> str:
    nh = min(num_hops, 10)
    vars_all, chain, value = _gen_chain(nh, rng, icl=True)
    sentences = [NOISE_HAYSTACK] * 5
    positions = sorted(rng.sample(range(len(sentences)), len(chain)))
    for offset, (pos, c) in enumerate(zip(positions, chain)):
        sentences.insert(pos + offset, c)
    ctx = "\n".join(sentences).replace(". \n", ".\n")
    body = VT_TEMPLATE.format(context=ctx, query=value)
    prefix = VT_ANSWER_PREFIX.format(num_v=nh + 1, query=value)
    return body + prefix + " ".join(vars_all) + "\n"


def _render_vt(context: str, query: str, num_v: int, icl_block: str) -> str:
    body = VT_TEMPLATE.format(context=context, query=query)
    prefix = VT_ANSWER_PREFIX.format(num_v=num_v, query=query)
    return (icl_block + body + prefix) if icl_block else (body + prefix)


def _build_sample(task: str, target_tokens: int, tokenizer, rng: random.Random, vt_icl: str | None):
    if task == "variable_tracking":
        num_hops = 4
        incremental = 5
        def render(n):
            ctx, val, vars_all, num_v = _make_vt(n, num_hops, rng)
            return _render_vt(ctx, val, num_v, vt_icl or ""), vars_all
    else:
        if task == "niah_single_1":
            type_haystack, num_k = "noise", 1
        elif task == "niah_single_2":
            type_haystack, num_k = "essay", 1
        elif task == "niah_multikey_1":
            type_haystack, num_k = "essay", 4
        else:
            raise ValueError(f"unknown task {task}")
        incremental = 25 if type_haystack == "noise" else 500
        def render(n):
            ctx, query, answers = _make_niah(n, type_haystack, num_k, rng)
            return _render_niah(ctx, query), answers

    n = incremental
    last_ok = None
    while True:
        text, answers = render(n)
        ntok = len(tokenizer.encode(text, add_special_tokens=True))
        if ntok >= target_tokens:
            break
        last_ok = (text, answers)
        n += incremental if n < incremental * 8 else incremental * 4
        if n > 400_000:
            break
    lo, hi = max(incremental, n // 2), n
    best = last_ok or (text, answers)
    while lo <= hi:
        mid = (lo + hi) // 2
        text, answers = render(mid)
        ntok = len(tokenizer.encode(text, add_special_tokens=True))
        if ntok <= target_tokens:
            best = (text, answers)
            lo = mid + incremental
        else:
            hi = mid - incremental
    return best


def _string_match_all_one(pred: str, refs: list[str]) -> float:
    pl = pred.lower()
    return sum(1.0 for r in refs if r.lower() in pl) / len(refs)


def main():
    p = argparse.ArgumentParser(description="RULER eval for Landmark Attention")
    p.add_argument("--ckpt_path", type=str, default=os.path.join(PROJECT_ROOT, "external", "landmark_ckpts", "landmark_tuned"))
    p.add_argument("--results_folder", type=str, default="./ruler_results")
    p.add_argument("--output_name", type=str, required=True)
    p.add_argument("--tasks", nargs="+", default=["niah_single_1", "variable_tracking"])
    p.add_argument("--lengths", nargs="+", default=["4k", "8k", "16k", "32k"])
    p.add_argument("--num_samples", type=int, default=100)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--max_new_tokens", type=int, default=48)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--insert_landmarks", action="store_true", default=True)
    p.add_argument("--no_insert_landmarks", dest="insert_landmarks", action="store_false")
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    args = p.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    print(f"[ruler-landmark] ckpt={args.ckpt_path} tasks={args.tasks} lengths={args.lengths} n={args.num_samples} top_k={args.top_k}")
    pipe, tokenizer = load_landmark_pipeline(args.ckpt_path, args.top_k, args.device, dtype=dtype)
    mem_freq = pipe.model.config.mem_freq

    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        raise ValueError("require num_shards>=1 and 0<=shard_index<num_shards")
    shard_tag = f"_shard{args.shard_index}of{args.num_shards}" if args.num_shards > 1 else ""

    outdir = Path(args.results_folder) / args.output_name
    outdir.mkdir(parents=True, exist_ok=True)
    summary: dict = {}
    for task in args.tasks:
        summary[task] = {}
        for length in args.lengths:
            if length not in _LENGTH_TOKENS:
                print(f"[WARN] unknown length {length}, skipping")
                continue
            base_seed = args.seed + _stable_hash(f"{task}/{length}")
            vt_icl = _make_vt_icl(random.Random(base_seed + 777), 4) if task == "variable_tracking" else None
            target_tokens = _LENGTH_TOKENS[length]
            records = []
            recall_sum = 0.0
            n_tok_seen = 0
            sample_indices = set(range(args.num_samples)[args.shard_index::args.num_shards])
            for i in tqdm(sorted(sample_indices), desc=f"{task}/{length}{shard_tag}", leave=False):
                rng = random.Random(base_seed * 1000 + i)
                prompt, answers = _build_sample(task, target_tokens, tokenizer, rng, vt_icl)
                n_tok_seen = len(tokenizer.encode(prompt, add_special_tokens=True))
                if args.insert_landmarks:
                    prompt = insert_landmark_tokens(prompt, tokenizer, mem_freq)
                mnt = args.max_new_tokens if task != "variable_tracking" else max(args.max_new_tokens, 60)
                try:
                    out = generate_landmark(pipe, prompt, mnt)
                except Exception as exc:
                    print(f"[ERROR] {task}/{length} sample={i} generation failed: {exc}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    out = ""
                rec = _string_match_all_one(out, answers)
                recall_sum += rec
                records.append({"sample_index": i, "answers": answers, "output": out, "recall": rec, "n_tokens": int(n_tok_seen)})
            score = recall_sum / len(records) * 100.0 if records else 0.0
            summary[task][length] = {"score": round(score, 2), "n": len(records), "approx_tokens": n_tok_seen}
            outfile = outdir / f"{task}_{length}{shard_tag}.json"
            with open(outfile, "w") as f:
                json.dump({"task": task, "length": length, "summary": summary[task][length], "records": records}, f, indent=2)
            print(f"[ruler-landmark] {task}/{length}: score={score:.2f} ({len(records)} samples, ~{n_tok_seen} tok) -> {outfile}")
    print("\n[ruler-landmark] SUMMARY")
    for task, vals in summary.items():
        print(f"  {task:>20}: " + "  ".join(f"{L}={vals[L]['score']:.1f}" for L in vals))
    with open(outdir / f"_summary{shard_tag}.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
