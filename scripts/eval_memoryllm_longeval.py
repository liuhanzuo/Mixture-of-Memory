#!/usr/bin/env python
"""MemoryLLM — LongEval (LongChat lines-retrieval) long-context eval driver.

The MemoryLLM companion to ``scripts/eval_qcmem_longeval.py``: it runs the SAME
LongEval lines-retrieval task (a record of N lines

    line <random-label>: REGISTER_CONTENT is <6-digit number>

after which the model must return the REGISTER_CONTENT of one queried line) under
the SAME prompt synthesis + digit-extraction judging, so MemoryLLM lands as a
*same-class* (compress-long-context-into-a-fixed-memory) baseline next to QCMem's
retrieval memory on the cleanest possible single-hop exact-retrieval benchmark.

Design — a thin composition of two existing, unmodified pieces:

  LongEval task framework (imported from ``scripts/eval_longeval_mem_space.py``):
    * ``build_lines_prompt`` — synthesize one lines-retrieval sample sized to
                               ~target tokens -> (prompt, expected_value,
                               target_label, n_lines).
    * ``extract_prediction`` — pull the first >=4-digit run from the output.
    * ``_LENGTH_TOKENS``     — length-bucket -> target token budget.
    * ``_PROMPT_HEADER``     — the fixed record-preamble; used here only to
                               recover the (deterministic) record boundary so we
                               can peel the record off the prompt.
  A sample is correct iff ``extract_prediction(output) == expected_value``
  (identical to eval_longeval_mem_space / eval_qcmem_longeval).

  MemoryLLM forward path (imported from ``scripts/eval_memoryllm_common.py``):
    * ``load_memoryllm`` / ``reset_memory`` / ``inject_context`` / ``generate_answer``
      — the same ported-MemoryLLM (transformers-5.5.4 port on this L20A/B200
      .venv) load + per-sample clean-pool reset + chunked memory injection +
      greedy decode used by the RULER MemoryLLM driver.

MemoryLLM usage is faithful to ``scripts/run_babilong_memoryllm.py``: the long
RECORD of lines is injected into the fixed memory pool (FIFO drop on overflow —
the fixed-capacity behaviour we contrast against QCMem retrieval), and the SHORT
preamble + query is what we generate from. We recover the record span by
splitting at the fixed template boundaries (the constant ``_PROMPT_HEADER`` and
the fixed ``"\\nNow the record is over."`` query marker), so nothing about the
LongEval sample is re-implemented.

Sharding matches ``eval_qcmem_longeval.py``: STABLE per-length ``zlib.crc32``
seed, shard ``s`` evaluates ``range(num_samples)[s::num_shards]``, and
``--score_only`` merges the disjoint per-shard JSON records.

Usage (single GPU, small validation):
    python scripts/eval_memoryllm_longeval.py \
        --lengths 8k 16k 32k --num_samples 50 \
        --output_name longeval_memoryllm --device cuda:0

Full 8-GPU sharded launch (one process per GPU):
    for s in 0 1 2 3 4 5 6 7; do
      CUDA_VISIBLE_DEVICES=$s python scripts/eval_memoryllm_longeval.py \
        --lengths 8k 16k 32k --num_samples 50 \
        --num_shards 8 --shard_index $s \
        --output_name longeval_memoryllm --device cuda:0 &
    done; wait
    python scripts/eval_memoryllm_longeval.py --score_only \
        --lengths 8k 16k 32k --output_name longeval_memoryllm
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import zlib
from pathlib import Path

import torch
from tqdm.auto import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJECT_ROOT,
           os.path.join(PROJECT_ROOT, "scripts"),
           os.path.join(PROJECT_ROOT, "third_party", "babilong-pkg")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# LongEval task framework (prompt synthesis + judging) — reused, unmodified.
import scripts.eval_longeval_mem_space as le  # noqa: E402
# MemoryLLM forward path — reused verbatim, unmodified.
import scripts.eval_memoryllm_common as mem  # noqa: E402

build_lines_prompt = le.build_lines_prompt
extract_prediction = le.extract_prediction
_LENGTH_TOKENS = dict(le._LENGTH_TOKENS)
_LENGTH_TOKENS.setdefault("64k", 64 * 1024)
_LENGTH_TOKENS.setdefault("128k", 128 * 1024)

# Fixed query marker emitted by build_lines_prompt.render() right after the
# record (the record itself is only "line ...\n" lines, so this is unambiguous).
_QUERY_MARKER = "\nNow the record is over."


def _split_context(prompt: str):
    """Return (record_text, gen_prompt) for a LongEval prompt.

    ``record_text`` = the N lines (streamed into the memory pool); ``gen_prompt``
    = the fixed preamble + the query (the short prompt we generate from). Falls
    back to injecting the whole prompt if the deterministic layout ever changes.
    """
    header = le._PROMPT_HEADER
    if not prompt.startswith(header):
        return prompt, prompt[:0]
    q_start = prompt.index(_QUERY_MARKER, len(header))
    record_text = prompt[len(header):q_start]
    gen_prompt = header + prompt[q_start:]
    return record_text, gen_prompt


# --------------------------------------------------------------------------- #
# shard-merge scorer (used by --score_only and single-shard auto-score)
# --------------------------------------------------------------------------- #
def score_longeval(outdir: Path, lengths):
    """Merge every ``longeval_<length>*.json`` shard and recompute per-length
    accuracy over the concatenated (disjoint) records."""
    summary: dict = {}
    for length in lengths:
        shard_files = sorted(outdir.glob(f"longeval_{length}_shard*.json"))
        single = outdir / f"longeval_{length}.json"
        if single.exists():
            shard_files = [single] + list(shard_files)
        if not shard_files:
            print(f"[score] {length}: no shard files found in {outdir}")
            continue
        records, seen = [], set()
        for sf in shard_files:
            try:
                with open(sf) as f:
                    payload = json.load(f)
            except Exception as e:
                print(f"[score][WARN] failed to read {sf}: {e}")
                continue
            for r in payload.get("records", []):
                key = r.get("sample_index")
                if key is not None:
                    if key in seen:
                        continue
                    seen.add(key)
                records.append(r)
        correct = sum(int(r.get("correct", False)) for r in records)
        total = len(records)
        acc = correct / total if total else 0.0
        summary[length] = {"accuracy": round(acc, 4), "correct": correct,
                           "total": total, "n_shards": len(shard_files)}
    print("\n[MemoryLLM-LongEval] SUMMARY (merged)")
    for length, s in summary.items():
        print(f"  {length:>4}: acc={s['accuracy']:.3f}  "
              f"({s['correct']}/{s['total']})  ({s['n_shards']} shards)")
    with open(outdir / "_summary_merged.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[MemoryLLM-LongEval] merged summary -> {outdir / '_summary_merged.json'}")
    return summary


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="MemoryLLM — LongEval lines-retrieval eval driver "
                    "(QCMem same-class baseline)")
    parser.add_argument("--model_path", type=str, default=mem.DEFAULT_MEMORYLLM_PATH,
                        help="On-disk MemoryLLM-8b-chat snapshot (config + tokenizer "
                             "+ safetensors).")
    parser.add_argument("--lengths", type=str, nargs="+",
                        default=["8k", "16k", "32k"],
                        help=f"Length buckets (subset of {sorted(_LENGTH_TOKENS)}).")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Samples per length bucket.")
    parser.add_argument("--chunk_size", type=int, default=1024,
                        help="Token chunk size for MemoryLLM context injection.")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--no_chat_template", action="store_true", default=False,
                        help="Generate from the raw prompt instead of the Llama-3 "
                             "chat template (default: chat template).")
    parser.add_argument("--results_folder", type=str,
                        default=str(Path(PROJECT_ROOT) / "longeval_results"))
    parser.add_argument("--output_name", type=str, default="longeval_memoryllm")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn_impl", type=str, default="sdpa")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Base RNG seed (per-sample seed derived stably with "
                             "zlib.crc32 so shards align across processes).")
    parser.add_argument("--verify_memory_reset", action="store_true", default=False)
    parser.add_argument("--verify_resets", type=int, default=3)
    parser.add_argument("--print_examples", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--score_only", action="store_true", default=False,
                        help="Only merge existing per-shard JSON + recompute acc.")
    args = parser.parse_args()

    outdir = Path(args.results_folder) / args.output_name

    if args.score_only:
        outdir.mkdir(parents=True, exist_ok=True)
        score_longeval(outdir, args.lengths)
        return

    if args.num_shards < 1:
        parser.error("--num_shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        parser.error(f"--shard_index must be in [0, {args.num_shards})")

    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    print(f"[MemoryLLM-LongEval] model_path={args.model_path}")
    print(f"[MemoryLLM-LongEval] lengths={args.lengths} "
          f"num_samples={args.num_samples} chunk_size={args.chunk_size} "
          f"shard={args.shard_index}/{args.num_shards} dtype={dtype}")

    model, tokenizer, initial_state = mem.load_memoryllm(
        args.model_path, device, dtype, attn_impl=args.attn_impl)

    outdir.mkdir(parents=True, exist_ok=True)
    sharded = args.num_shards > 1
    shard_tag = f"_shard{args.shard_index}of{args.num_shards}" if sharded else ""
    use_chat_template = not args.no_chat_template

    with open(outdir / f"eval_config{shard_tag}.json", "w") as f:
        cfg = dict(vars(args))
        cfg.update({"num_layers": int(model.config.num_hidden_layers),
                    "use_chat_template": use_chat_template})
        json.dump(cfg, f, indent=2)

    reset_checks_done = 0
    summary: dict = {}
    for length in args.lengths:
        if length not in _LENGTH_TOKENS:
            print(f"[WARN] unknown length {length}, skipping")
            continue
        target_tokens = _LENGTH_TOKENS[length]
        # STABLE per-length seed (NOT process-randomised hash()) so every shard
        # process derives the same per-sample seeds -> identical sample set.
        length_seed = args.seed + (zlib.crc32(length.encode()) % 100000)

        outfile = outdir / f"longeval_{length}{shard_tag}.json"
        if outfile.exists() and not args.overwrite:
            print(f"[MemoryLLM-LongEval] Skip existing {outfile}")
            continue

        sample_indices = list(range(args.num_samples))[args.shard_index::args.num_shards]
        if sharded:
            print(f"[MemoryLLM-LongEval] {length} shard {args.shard_index}/"
                  f"{args.num_shards}: {len(sample_indices)} of {args.num_samples} samples")

        records = []
        correct = 0
        total = 0
        n_tok_seen = 0
        t0 = time.time()
        for pos, i in enumerate(tqdm(sample_indices, desc=f"{length}", leave=False)):
            # Per-sample RNG (stable across processes) — build ONLY this shard's
            # samples (no wasteful build-then-skip of the full set).
            rng = random.Random(length_seed * 1000 + i)
            prompt, expected, target_label, n_lines = build_lines_prompt(
                target_tokens, tokenizer, rng)
            n_tok_seen = len(tokenizer.encode(prompt, add_special_tokens=True))

            record_text, gen_prompt = _split_context(prompt)

            verify = args.verify_memory_reset and reset_checks_done < args.verify_resets
            mem.reset_memory(model, initial_state, verify=verify)
            if verify:
                reset_checks_done += 1
                print(f"[MemoryLLM-LongEval] verified clean memory reset (i={i})")

            try:
                mem.inject_context(model, tokenizer, record_text, device,
                                   chunk_size=args.chunk_size)
                output = mem.generate_answer(
                    model, tokenizer, gen_prompt, device,
                    max_new_tokens=args.max_new_tokens,
                    use_chat_template=use_chat_template)
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                output = "[OOM]"
                print(f"[OOM] i={i} length={length} n_tok={n_tok_seen}: {e}", flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            pred = extract_prediction(output)
            ok = (pred == expected)
            correct += int(ok)
            total += 1
            records.append({
                "sample_index": i, "label": target_label, "expected": expected,
                "output": output, "pred": pred, "correct": ok,
                "n_lines": n_lines, "n_tokens": n_tok_seen,
            })
            if total <= args.print_examples:
                print(f"[MemoryLLM-LongEval][example] i={i} label={target_label} "
                      f"expected={expected} pred={pred} correct={ok} output={output!r}")
            if (pos + 1) % 10 == 0 or pos == len(sample_indices) - 1:
                acc_cur = correct / total if total else 0.0
                with open(outfile, "w") as f:
                    json.dump({"length": length,
                               "summary": {"accuracy": round(acc_cur, 4),
                                           "correct": correct, "total": total},
                               "records": records}, f, indent=2)

        acc = correct / total if total else 0.0
        summary[length] = {"accuracy": round(acc, 4), "correct": correct,
                           "total": total, "approx_tokens": n_tok_seen}
        with open(outfile, "w") as f:
            json.dump({"length": length, "summary": summary[length],
                       "records": records,
                       "config": {"model_path": args.model_path,
                                  "chunk_size": args.chunk_size,
                                  "max_new_tokens": args.max_new_tokens,
                                  "use_chat_template": use_chat_template,
                                  "num_shards": args.num_shards,
                                  "shard_index": args.shard_index,
                                  "seed": args.seed}}, f, indent=2)
        print(f"[MemoryLLM-LongEval] {length}: acc={acc:.3f} ({correct}/{total}) "
              f"~{n_tok_seen} tok ({time.time()-t0:.1f}s) -> {outfile}")

    print("\n[MemoryLLM-LongEval] SUMMARY (this shard)")
    for length, s in summary.items():
        print(f"  {length:>4}: acc={s['accuracy']:.3f}  "
              f"({s['correct']}/{s['total']})  ~{s['approx_tokens']} tok")
    with open(outdir / f"_summary{shard_tag}.json", "w") as f:
        json.dump(summary, f, indent=2)

    if args.num_shards == 1:
        print("\n[MemoryLLM-LongEval] merged scoring (single-shard mode)...")
        score_longeval(outdir, args.lengths)
    print("\n[MemoryLLM-LongEval] Evaluation complete!")


if __name__ == "__main__":
    main()
