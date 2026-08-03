"""LongEval (LongChat) lines-retrieval evaluation for the mem_space architecture.

Probes the *cleanest possible* long-context retrieval ability: in a record of
N lines of the form

    line <random-label>: REGISTER_CONTENT is <6-digit number>

the model is asked, after the whole record, to return the REGISTER_CONTENT of
one specific line. Pure single-hop exact retrieval — no multi-hop, no semantic
mixing — so accuracy at 4k/8k/16k/32k directly quantifies "can the fixed-size
memory bank retrieve one exact fact under length L".

Task format follows DachengLi1/LongChat's ``longeval`` lines-retrieval task. We
generate the prompts locally (the task is trivial to synthesize and this avoids
clone/data-format friction) and size each length bucket with the Llama-3
tokenizer so the buckets align with our BABILong 4k/8k/16k/32k cells.

Eval口径 matches BABILong W0 (pure memory readout, closed-book):
    chunk_size=512, swa_eval_chunks=0 — the generation window is only the last
    chunk and everything earlier reaches the final forward solely through the
    memory bank. This makes LongEval accuracy directly comparable to our
    BABILong qa5 32k=9 conclusion.

We reuse the model-loading + streaming-generation helpers verbatim from
``run_babilong_mem_space.py`` so the inference path is identical to BABILong W0.

Usage:
    python scripts/eval_longeval_mem_space.py \
        --model_path models/Meta-Llama-3-8B \
        --checkpoint outputs/mem_space_p11_chunk512_deltarule_normreadout/mem_space_adapter_step000500.pt \
        --adapter_config outputs/mem_space_p11_chunk512_deltarule_normreadout/adapter_config.json \
        --output_name longeval_p11_step500 \
        --lengths 4k 8k 16k 32k --num_samples 50 --chunk_size 512
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
import sys
import zlib
from pathlib import Path

import torch
from tqdm.auto import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoTokenizer  # noqa: E402

# Reuse the exact W0 model-loading + streaming-generation path from BABILong.
from scripts.run_babilong_mem_space import (  # noqa: E402
    build_mem_space_config,
    load_mem_space_model,
    generate_with_mem_space,
)


# --------------------------------------------------------------------------- #
# LongEval lines-retrieval prompt generation
# --------------------------------------------------------------------------- #

# Approximate token budget per BABILong-aligned length bucket.
_LENGTH_TOKENS = {
    "1k": 1024,
    "2k": 2048,
    "4k": 4096,
    "8k": 8192,
    "16k": 16384,
    "32k": 32768,
}

_PROMPT_HEADER = (
    "Below is a record of lines I want you to remember. Each line begins with "
    "'line <line index>' and contains a '<REGISTER_CONTENT>' at the end of the "
    "line as a numerical value. For each line index, memorize its corresponding "
    "<REGISTER_CONTENT>. At the end of the record, I will ask you to retrieve the "
    "corresponding <REGISTER_CONTENT> of a certain line index. Now the record "
    "start:\n\n"
)


def _random_label(rng: random.Random) -> str:
    """Two random lowercase words joined by a hyphen, LongChat-style."""
    def word() -> str:
        return "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(5, 9)))
    return f"{word()}-{word()}"


def build_lines_prompt(target_tokens: int, tokenizer, rng: random.Random):
    """Build one lines-retrieval sample sized to ~target_tokens.

    Returns (prompt_text, expected_value:str, target_label:str, n_lines:int).

    Adds lines until the tokenized header+lines+query reaches target_tokens,
    then picks a random target line (uniform over depth) and appends the query.
    """
    labels: list[str] = []
    lines: list[str] = []
    values: list[str] = []

    def render(query_label: str) -> str:
        query = (
            f"\nNow the record is over. Tell me what is the <REGISTER_CONTENT> in "
            f"line {query_label}? I need the number.\nThe <REGISTER_CONTENT> in line "
            f"{query_label} is"
        )
        return _PROMPT_HEADER + "".join(lines) + query

    # Cheap token estimate per line to avoid re-tokenizing every iteration.
    # Grow in blocks, re-measure, until we slightly exceed the target.
    while True:
        # add a block of lines
        for _ in range(64):
            label = _random_label(rng)
            value = str(rng.randint(100000, 999999))
            labels.append(label)
            values.append(value)
            lines.append(f"line {label}: REGISTER_CONTENT is <{value}>\n")
        # measure with a placeholder query (length-stable: any label same size-ish)
        approx = render(labels[len(labels) // 2])
        n_tok = len(tokenizer.encode(approx, add_special_tokens=True))
        if n_tok >= target_tokens:
            break

    # Pick a random target line (uniform depth) and finalize.
    ti = rng.randrange(len(labels))
    target_label = labels[ti]
    expected = values[ti]
    prompt = render(target_label)
    return prompt, expected, target_label, len(labels)


_NUM_RE = re.compile(r"\d{4,}")


def extract_prediction(output: str) -> str:
    """Pull the first >=4-digit run from the model output (the 6-digit answer)."""
    m = _NUM_RE.search(output)
    return m.group(0) if m else ""


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main():
    p = argparse.ArgumentParser(description="LongEval lines-retrieval eval for mem_space")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--adapter_config", type=str, required=True)
    p.add_argument("--results_folder", type=str, default="./longeval_results")
    p.add_argument("--output_name", type=str, required=True)
    p.add_argument("--lengths", type=str, nargs="+",
                   default=["4k", "8k", "16k", "32k"])
    p.add_argument("--num_samples", type=int, default=50)
    p.add_argument("--chunk_size", type=int, default=512)
    p.add_argument("--max_new_tokens", type=int, default=16)
    p.add_argument("--swa_eval_chunks", type=int, default=0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn_impl", type=str, default="sdpa",
                   choices=["sdpa", "eager", "flash_attention_2"])
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    args = p.parse_args()

    print("[longeval-mem_space] Configuration:")
    print(f"  model={args.model_path}")
    print(f"  ckpt={args.checkpoint}")
    print(f"  lengths={args.lengths} num_samples={args.num_samples}")
    print(f"  chunk_size={args.chunk_size} swa={args.swa_eval_chunks}")

    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16,
             "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(args.adapter_config) as f:
        adapter_cfg = json.load(f)
    mem_config = build_mem_space_config(adapter_cfg)
    mem_config.l3_recon_max_positions = args.chunk_size

    model = load_mem_space_model(
        model_path=args.model_path,
        checkpoint_path=args.checkpoint,
        mem_config=mem_config,
        device=device,
        dtype=dtype,
        attn_impl=args.attn_impl,
    )

    outdir = Path(args.results_folder) / args.output_name
    outdir.mkdir(parents=True, exist_ok=True)

    summary: dict = {}
    for length in args.lengths:
        if length not in _LENGTH_TOKENS:
            print(f"[WARN] unknown length {length}, skipping")
            continue
        target_tokens = _LENGTH_TOKENS[length]
        # Deterministic per-(length) RNG so shards generate the SAME sample set
        # and each shard evaluates a stride slice [shard::num_shards].
        # Use zlib.crc32 (PYTHONHASHSEED-independent) not built-in hash():
        # hash(str) is per-process salted unless PYTHONHASHSEED is pinned, so
        # separate shard processes would draw different seeds and generate
        # misaligned sample sets. crc32 is stable across processes.
        rng = random.Random(args.seed + zlib.crc32(length.encode()) % 100000)

        sample_indices = list(range(args.num_samples))[args.shard_index::args.num_shards]
        sharded = args.num_shards > 1
        shard_tag = f"_shard{args.shard_index}of{args.num_shards}" if sharded else ""

        records = []
        correct = 0
        total = 0
        n_lines_seen = 0
        n_tok_seen = 0
        for i in tqdm(range(args.num_samples), desc=f"{length}", leave=False):
            prompt, expected, label, n_lines = build_lines_prompt(
                target_tokens, tokenizer, rng
            )
            if i not in sample_indices:
                continue  # generated (to keep RNG aligned) but not this shard's
            ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt").to(device)
            n_lines_seen = n_lines
            n_tok_seen = ids.shape[1]
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                out = generate_with_mem_space(
                    model=model,
                    input_ids=ids,
                    tokenizer=tokenizer,
                    chunk_size=args.chunk_size,
                    max_new_tokens=args.max_new_tokens,
                    device=device,
                    swa_eval_chunks=args.swa_eval_chunks,
                )
            pred = extract_prediction(out)
            ok = pred == expected
            correct += int(ok)
            total += 1
            records.append({
                "label": label, "expected": expected,
                "output": out, "pred": pred, "correct": ok,
                "n_lines": n_lines, "n_tokens": int(ids.shape[1]),
            })

        acc = correct / total if total else 0.0
        summary[length] = {
            "accuracy": acc, "correct": correct, "total": total,
            "approx_lines": n_lines_seen, "approx_tokens": n_tok_seen,
        }
        outfile = outdir / f"longeval_{length}{shard_tag}.json"
        with open(outfile, "w") as f:
            json.dump({"length": length, "summary": summary[length],
                       "records": records}, f, indent=2)
        print(f"[longeval-mem_space] {length}: acc={acc:.3f} "
              f"({correct}/{total}) ~{n_lines_seen} lines / {n_tok_seen} tok "
              f"-> {outfile}")

    print("\n[longeval-mem_space] SUMMARY")
    for length, s in summary.items():
        print(f"  {length:>4}: acc={s['accuracy']:.3f}  "
              f"({s['correct']}/{s['total']})  ~{s['approx_tokens']} tok")
    with open(outdir / f"_summary{('_shard%dof%d' % (args.shard_index, args.num_shards)) if args.num_shards>1 else ''}.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
