"""RULER (NVIDIA long-context benchmark) evaluation for mem_space + base Llama-3.

Self-contained, faithful re-implementation of the two RULER task families that
most precisely localise our 32k long-range failure mode:

  * NIAH (needle-in-a-haystack, pure retrieval) — 3 variants:
        - ``niah_single_1``  : noise haystack, single word-key / numeric-value
        - ``niah_single_2``  : real-prose (PG19) haystack, single key/value
        - ``niah_multikey_1``: real-prose haystack, 4 distractor keys, retrieve 1
    Directly comparable to our LongEval single-hop retrieval conclusion.

  * ``variable_tracking`` (multi-hop) — 1 chain, 4 hops (5 variables), noise
    haystack + 1 in-context example. Comparable to BABILong qa2/qa3 multi-hop.

Why a local re-implementation (vs ``git clone NVIDIA/RULER``)?
  The RULER repo's task templates, needle format and scoring (``string_match_all``
  recall) are short and fully reproduced below verbatim from the upstream source
  (``data/synthetic/{niah,variable_tracking,constants}.py`` and
  ``eval/synthetic/constants.py``). The only upstream asset that is *not* in the
  git tree is ``json/PaulGrahamEssays.json`` (download-gated, returns 404 from
  raw.githubusercontent). We substitute PG19 natural prose (``data/pg19_train``),
  which is the same spirit of essay-style distractor text and keeps the pipeline
  dependency-free. The ``noise`` haystack used by single_1 + variable_tracking is
  RULER's exact literal string, so those tasks are bit-faithful.

Scoring == RULER ``string_match_all``: for each sample, recall = fraction of the
reference answer strings that appear (case-insensitive substring) in the model
output; the cell score is the mean recall over samples, ×100.

Two inference paths, selected by ``--model_type``:
  * ``mem_space`` : reuse the EXACT BABILong W0 closed-book memory-readout path
    (``run_babilong_mem_space.generate_with_mem_space``): chunk the prompt into
    ``chunk_size`` segments, stream chunks[:-1] into the memory bank, then
    generate from the last chunk only (``swa_eval_chunks=0`` => earlier context
    reaches the final forward solely through the 128-slot memory bank). This is
    identical in口径 to our BABILong qa5 32k / LongEval numbers.
  * ``base`` : plain HF ``LlamaForCausalLM`` full-attention greedy generation.
    The context is left-truncated to the model window (8192) when longer — the
    honest "full-attention upper bound" (and it too loses needles past 8k, which
    is exactly the comparison we want at 16k/32k).

Usage (mem_space SOTA ckpt):
    python scripts/eval_ruler_mem_space.py --model_type mem_space \
        --model_path models/Meta-Llama-3-8B \
        --checkpoint outputs/mem_space_p11_chunk1024_deltarule_normreadout/mem_space_adapter.pt \
        --adapter_config outputs/mem_space_p11_chunk1024_deltarule_normreadout/adapter_config.json \
        --output_name ruler_p11_c1024 --chunk_size 1024 --swa_eval_chunks 0 \
        --tasks niah_single_1 niah_single_2 niah_multikey_1 variable_tracking \
        --lengths 4k 8k 16k 32k --num_samples 50

Usage (base full-attention upper bound):
    python scripts/eval_ruler_mem_space.py --model_type base \
        --model_path models/Meta-Llama-3-8B \
        --output_name ruler_base --base_max_window 8192 \
        --tasks niah_single_1 niah_single_2 niah_multikey_1 variable_tracking \
        --lengths 4k 8k 16k 32k --num_samples 50
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
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoTokenizer, LlamaForCausalLM  # noqa: E402

# Reuse the EXACT mem_space W0 model-loading + streaming-generation path.
from scripts.run_babilong_mem_space import (  # noqa: E402
    build_mem_space_config,
    load_mem_space_model,
    generate_with_mem_space,
)


# --------------------------------------------------------------------------- #
# RULER constants (verbatim from NVIDIA/RULER data/synthetic/constants.py)
# --------------------------------------------------------------------------- #

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

# Length bucket -> approximate max sequence length (tokens), BABILong-aligned.
_LENGTH_TOKENS = {
    "1k": 1024, "2k": 2048, "4k": 4096,
    "8k": 8192, "16k": 16384, "32k": 32768,
}

DEPTHS = [int(round(x)) for x in __import__("numpy").linspace(0, 100, num=40)]


# --------------------------------------------------------------------------- #
# Essay (PG19 prose) haystack — substitute for RULER's gated PaulGrahamEssays.
# --------------------------------------------------------------------------- #

_ESSAY_WORDS_CACHE: list[str] | None = None


def _load_essay_words() -> list[str]:
    """Load a chunk of PG19 natural prose, return it as a whitespace-split word
    list (mirrors RULER's ``re.sub(r'\\s+', ' ', essay).split(' ')``)."""
    global _ESSAY_WORDS_CACHE
    if _ESSAY_WORDS_CACHE is not None:
        return _ESSAY_WORDS_CACHE
    path = os.path.join(PROJECT_ROOT, "data", "pg19_train.jsonl")
    with open(path, "r", errors="ignore") as f:
        text = f.read(8_000_000)  # ~8 MB of prose is plenty for 32k contexts
    text = re.sub(r"\s+", " ", text).strip()
    _ESSAY_WORDS_CACHE = text.split(" ")
    return _ESSAY_WORDS_CACHE


_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def _sent_tokenize(text: str) -> list[str]:
    """Lightweight sentence splitter (avoids nltk/punkt dependency). Recall
    scoring is invariant to exact sentence boundaries."""
    return [s for s in _SENT_RE.split(text.strip()) if s]


# --------------------------------------------------------------------------- #
# Random value/key generators (RULER-faithful: word keys, numeric values)
# --------------------------------------------------------------------------- #


def _rand_word(rng: random.Random) -> str:
    """adj-noun style identifier (two random lowercase words, hyphen-joined)."""
    def w() -> str:
        return "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(4, 8)))
    return f"{w()}-{w()}"


def _rand_number(rng: random.Random, num_digits: int = 7) -> str:
    return str(rng.randint(10 ** (num_digits - 1), 10 ** num_digits - 1))


# --------------------------------------------------------------------------- #
# NIAH sample generation
# --------------------------------------------------------------------------- #


def _make_niah(num_haystack: int, type_haystack: str, num_needle_k: int,
               rng: random.Random):
    """Build one NIAH (context, query, answers) with ``num_needle_k`` keys
    (1 queried, rest distractors); single value per key, numeric.

    Faithful to RULER niah.generate_input_output for
    num_needle_v=num_needle_q=1.
    """
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
    else:  # essay (PG19 prose)
        words = _load_essay_words()
        if num_haystack > len(words):
            reps = (num_haystack + len(words) - 1) // len(words)
            text = " ".join((words * reps)[:num_haystack])
        else:
            text = " ".join(words[:num_haystack])
        sents = _sent_tokenize(text)
        if not sents:
            sents = [text]
        chosen = rng.sample(DEPTHS, min(len(needles), len(DEPTHS)))
        ins = [0] + sorted(int(len(sents) * (d / 100)) for d in chosen) + [len(sents)]
        parts = []
        for i in range(1, len(ins)):
            parts.append(" ".join(sents[ins[i - 1]:ins[i]]))
            if i - 1 < len(needles):
                parts.append(needles[i - 1])
        context = " ".join(parts)

    # Query the first key; answer = its value list.
    query = keys[0]
    answers = values[0]
    return context, query, answers


def _render_niah(context: str, query: str) -> str:
    """template + answer_prefix with the single-needle text fixups RULER does."""
    full = NIAH_TEMPLATE + NIAH_ANSWER_PREFIX
    # Single-needle replacements (num_needle_q*num_needle_v == 1):
    full = full.replace("Some", "A").replace("are all", "is").replace("are", "is")
    full = full.replace("answers", "answer")
    full = full.format(type_needle_v="number", context=context, query=query)
    return full


# --------------------------------------------------------------------------- #
# variable_tracking sample generation (1 chain, num_hops hops)
# --------------------------------------------------------------------------- #


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
    context = "\n".join(sentences).replace(". \n", ".\n")
    return context, value, vars_all, num_hops + 1


def _make_vt_icl(rng: random.Random, num_hops: int) -> str:
    """Tiny in-context worked example (RULER prepends one to fix output format):
    a full mini VT prompt INCLUDING its answer (the chain's variable names)."""
    nh = min(num_hops, 10)
    vars_all, chain, value = _gen_chain(nh, rng, icl=True)
    sentences = [NOISE_HAYSTACK] * 5
    positions = sorted(rng.sample(range(len(sentences)), len(chain)))
    for offset, (pos, c) in enumerate(zip(positions, chain)):
        sentences.insert(pos + offset, c)
    ctx = "\n".join(sentences).replace(". \n", ".\n")
    body = VT_TEMPLATE.format(context=ctx, query=value)
    prefix = VT_ANSWER_PREFIX.format(num_v=nh + 1, query=value)
    answer = " ".join(vars_all)
    return body + prefix + answer + "\n"


def _render_vt(context: str, query: str, num_v: int, icl_block: str) -> str:
    body = VT_TEMPLATE.format(context=context, query=query)
    prefix = VT_ANSWER_PREFIX.format(num_v=num_v, query=query)
    return (icl_block + body + prefix) if icl_block else (body + prefix)


# --------------------------------------------------------------------------- #
# Sizing: binary-search haystack size so total tokens ~ target (RULER style)
# --------------------------------------------------------------------------- #


def _build_sample(task: str, target_tokens: int, tokenizer, rng: random.Random,
                  vt_icl: str | None):
    """Return (prompt_text, answers:list[str]) sized to ~target_tokens."""
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

    # Grow geometrically until we exceed the target, then back off.
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
    # Pick the largest size whose token count does not blow far past target:
    # binary-refine between (n - step) and n.
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


# --------------------------------------------------------------------------- #
# Scoring: RULER string_match_all (recall of reference substrings)
# --------------------------------------------------------------------------- #


def _string_match_all_one(pred: str, refs: list[str]) -> float:
    pl = pred.lower()
    return sum(1.0 for r in refs if r.lower() in pl) / len(refs)


# --------------------------------------------------------------------------- #
# Base full-attention generation
# --------------------------------------------------------------------------- #


@torch.no_grad()
def _generate_base(model, input_ids, tokenizer, max_new_tokens, device,
                   max_window: int):
    ids = input_ids
    if ids.shape[1] > max_window:
        ids = ids[:, -max_window:]  # left-truncate, keep question + answer prefix
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen = out[0, ids.shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main():
    p = argparse.ArgumentParser(description="RULER eval for mem_space + base Llama-3")
    p.add_argument("--model_type", choices=["mem_space", "base"], required=True)
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--adapter_config", type=str, default=None)
    p.add_argument("--results_folder", type=str, default="./ruler_results")
    p.add_argument("--output_name", type=str, required=True)
    p.add_argument("--tasks", type=str, nargs="+",
                   default=["niah_single_1", "niah_single_2",
                            "niah_multikey_1", "variable_tracking"])
    p.add_argument("--lengths", type=str, nargs="+",
                   default=["4k", "8k", "16k", "32k"])
    p.add_argument("--num_samples", type=int, default=50)
    p.add_argument("--chunk_size", type=int, default=1024)
    p.add_argument("--swa_eval_chunks", type=int, default=0)
    p.add_argument("--max_new_tokens", type=int, default=48)
    p.add_argument("--base_max_window", type=int, default=8192)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn_impl", type=str, default="sdpa",
                   choices=["sdpa", "eager", "flash_attention_2"])
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    # Slot-Routed Evidence Memory eval-time override (2026-06-17). adapter_config
    # carries no evidence fields, so they default OFF; these flags turn the
    # evidence path ON at eval to probe whether routed raw evidence recovers the
    # exact needle even with a frozen (evidence-naive) checkpoint.
    p.add_argument("--use_slot_evidence", action="store_true", default=False)
    p.add_argument("--evidence_buffer_size", type=int, default=8)
    p.add_argument("--evidence_topr", type=int, default=0)
    p.add_argument("--evidence_layer", type=int, default=0)
    args = p.parse_args()

    print(f"[ruler] model_type={args.model_type} tasks={args.tasks} "
          f"lengths={args.lengths} n={args.num_samples} "
          f"chunk={args.chunk_size} swa={args.swa_eval_chunks}")

    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.model_type == "mem_space":
        assert args.checkpoint and args.adapter_config, \
            "mem_space requires --checkpoint and --adapter_config"
        with open(args.adapter_config) as f:
            adapter_cfg = json.load(f)
        mem_config = build_mem_space_config(adapter_cfg)
        mem_config.l3_recon_max_positions = args.chunk_size
        # Eval-time evidence override (see CLI flags above).
        if args.use_slot_evidence:
            mem_config.use_slot_evidence = True
            mem_config.evidence_buffer_size = args.evidence_buffer_size
            mem_config.evidence_topr = args.evidence_topr
            mem_config.evidence_layer = args.evidence_layer
            print(f"[ruler] EVIDENCE ON: buffer_size={args.evidence_buffer_size} "
                  f"topr={args.evidence_topr} layer={args.evidence_layer}")
        model = load_mem_space_model(
            model_path=args.model_path, checkpoint_path=args.checkpoint,
            mem_config=mem_config, device=device, dtype=dtype,
            attn_impl=args.attn_impl,
        )
    else:
        print(f"[ruler] loading base Llama-3 (full attention, window={args.base_max_window})")
        model = LlamaForCausalLM.from_pretrained(
            args.model_path, torch_dtype=dtype, attn_implementation=args.attn_impl,
        ).to(device)
        model.eval()

    outdir = Path(args.results_folder) / args.output_name
    outdir.mkdir(parents=True, exist_ok=True)

    sharded = args.num_shards > 1
    shard_tag = f"_shard{args.shard_index}of{args.num_shards}" if sharded else ""

    summary: dict = {}
    for task in args.tasks:
        summary[task] = {}
        for length in args.lengths:
            if length not in _LENGTH_TOKENS:
                print(f"[WARN] unknown length {length}, skipping")
                continue
            target_tokens = _LENGTH_TOKENS[length]
            # Deterministic per-(task,length) RNG so shards share the sample set.
            base_seed = args.seed + (hash((task, length)) % 100000)

            # Pre-build a fixed in-context example for VT (shared across samples).
            vt_icl = None
            if task == "variable_tracking":
                vt_icl = _make_vt_icl(random.Random(base_seed + 777), 4)

            sample_indices = list(range(args.num_samples))[args.shard_index::args.num_shards]
            records = []
            recall_sum = 0.0
            total = 0
            n_tok_seen = 0
            for i in tqdm(range(args.num_samples), desc=f"{task}/{length}", leave=False):
                rng = random.Random(base_seed * 1000 + i)
                prompt, answers = _build_sample(task, target_tokens, tokenizer, rng, vt_icl)
                if i not in sample_indices:
                    continue
                ids = tokenizer.encode(prompt, add_special_tokens=True,
                                       return_tensors="pt").to(device)
                n_tok_seen = ids.shape[1]
                mnt = args.max_new_tokens if task != "variable_tracking" else max(args.max_new_tokens, 60)
                with torch.amp.autocast(device_type="cuda", dtype=dtype):
                    if args.model_type == "mem_space":
                        out = generate_with_mem_space(
                            model=model, input_ids=ids, tokenizer=tokenizer,
                            chunk_size=args.chunk_size, max_new_tokens=mnt,
                            device=device, swa_eval_chunks=args.swa_eval_chunks,
                        )
                    else:
                        out = _generate_base(
                            model, ids, tokenizer, mnt, device, args.base_max_window,
                        )
                rec = _string_match_all_one(out, answers)
                recall_sum += rec
                total += 1
                records.append({
                    "answers": answers, "output": out, "recall": rec,
                    "n_tokens": int(ids.shape[1]),
                })

            score = (recall_sum / total * 100.0) if total else 0.0
            summary[task][length] = {
                "score": round(score, 2), "n": total, "approx_tokens": n_tok_seen,
            }
            outfile = outdir / f"{task}_{length}{shard_tag}.json"
            with open(outfile, "w") as f:
                json.dump({"task": task, "length": length,
                           "summary": summary[task][length],
                           "records": records}, f, indent=2)
            print(f"[ruler] {task}/{length}: score={score:.2f} "
                  f"({total} samples, ~{n_tok_seen} tok) -> {outfile}")

    print("\n[ruler] SUMMARY")
    for task in summary:
        row = "  ".join(f"{L}={summary[task][L]['score']:.1f}" for L in summary[task])
        print(f"  {task:>20}: {row}")
    with open(outdir / f"_summary{shard_tag}.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
