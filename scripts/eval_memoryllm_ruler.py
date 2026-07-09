#!/usr/bin/env python
"""MemoryLLM — RULER (NIAH / variable_tracking) long-context eval driver.

The MemoryLLM companion to ``scripts/eval_ruler_qcmem.py``: it runs the SAME
RULER task family (NIAH needle-in-a-haystack + variable_tracking) under the SAME
generator + token-sizing + ``string_match_all`` scoring, so MemoryLLM lands as a
*same-class* (compress-long-context-into-a-fixed-memory) baseline next to QCMem's
retrieval memory on RULER — a genuine cross-benchmark memory-vs-memory
comparison, not just a KV-cache baseline.

Design — a thin composition of two existing, unmodified pieces:

  RULER task framework (imported from ``scripts/eval_ruler_mem_space.py``):
    * ``_build_sample``          — RULER-faithful (prompt, answers, gold_needle)
                                   sized to ~target tokens (NIAH single/multikey
                                   on noise|PG19-prose haystack, variable_tracking
                                   on a noise haystack with a fixed in-context
                                   worked example).
    * ``_make_vt_icl``           — the fixed VT in-context example.
    * ``_render_niah`` / ``_render_vt`` — render skeletons, used here only to
                                   recover the (deterministic) context boundary
                                   so we can peel the haystack off the prompt.
    * ``_string_match_all_one``  — RULER ``string_match_all`` recall scoring.
    * ``_LENGTH_TOKENS``         — length-bucket -> target token budget.

  MemoryLLM forward path (imported from ``scripts/eval_memoryllm_common.py``):
    * ``load_memoryllm``   — load the PORTED MemoryLLM (transformers-5.5.4 port,
                             runs on this L20A/B200 .venv) + tokenizer, snapshot
                             the clean memory pool.
    * ``reset_memory``     — restore the clean pool before every sample (no state
                             leak sample-to-sample).
    * ``inject_context``   — stream the haystack chunk-by-chunk into the memory
                             pool (FIFO drop on overflow — the fixed-capacity
                             behaviour we contrast against QCMem retrieval).
    * ``generate_answer``  — greedy-decode the answer from the (short) question
                             prompt with the memory pool injected.

MemoryLLM usage is faithful to ``scripts/run_babilong_memoryllm.py``: the long
HAYSTACK is injected into the memory pool, and the SHORT instruction + question
(+ VT in-context example) is what we generate from. We recover the haystack span
by splitting the rendered RULER prompt at its deterministic template boundaries
(the instruction head before ``{context}`` and the fixed question marker after
it), so the injected text is exactly the haystack and the generation prompt is
exactly the task's instruction+question — nothing about the RULER sample is
re-implemented.

Sharding: samples for each (task,length) are generated with a STABLE per-cell
seed (``zlib.crc32``-derived, not Python's process-randomised ``hash()``), so
every shard process derives the same sample set and shard ``s`` evaluates only
``range(limit)[s::num_shards]``. Per-cell JSON output mirrors
``eval_ruler_mem_space`` (``{task}_{length}{shard_tag}.json``); merge shards with
``--score_only``.

Usage (single GPU, small validation):
    python scripts/eval_memoryllm_ruler.py \
        --ruler_tasks niah_single niah_multikey --lengths 8k 16k 32k \
        --limit 50 --output_name ruler_memoryllm --device cuda:0

Full 8-GPU sharded launch (one process per GPU, shard_index 0..7):
    for s in 0 1 2 3 4 5 6 7; do
      CUDA_VISIBLE_DEVICES=$s python scripts/eval_memoryllm_ruler.py \
        --ruler_tasks niah_single niah_multikey --lengths 8k 16k 32k \
        --limit 50 --num_shards 8 --shard_index $s \
        --output_name ruler_memoryllm --device cuda:0 &
    done; wait
    # then merge:
    python scripts/eval_memoryllm_ruler.py --score_only \
        --ruler_tasks niah_single niah_multikey --lengths 8k 16k 32k \
        --output_name ruler_memoryllm
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
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

# RULER task framework (generation + scoring) — reused verbatim, unmodified.
import scripts.eval_ruler_mem_space as ruler  # noqa: E402
# MemoryLLM forward path — reused verbatim, unmodified.
import scripts.eval_memoryllm_common as mem  # noqa: E402


# --------------------------------------------------------------------------- #
# RULER task-name aliases -> canonical eval_ruler_mem_space task ids
# (mirrors scripts/eval_ruler_qcmem.py so the CLI is identical).
# --------------------------------------------------------------------------- #
_TASK_ALIAS = {
    "niah_single": "niah_single_2",        # default to the realistic prose haystack
    "niah_single_noise": "niah_single_1",
    "niah_single_essay": "niah_single_2",
    "niah_multi": "niah_multikey_1",
    "niah_multikey": "niah_multikey_1",
    "vt": "variable_tracking",
}
_CANONICAL_TASKS = {
    "niah_single_1", "niah_single_2", "niah_multikey_1", "variable_tracking",
}


def _resolve_task(name: str) -> str:
    if name in _CANONICAL_TASKS:
        return name
    if name in _TASK_ALIAS:
        return _TASK_ALIAS[name]
    raise ValueError(
        f"unknown ruler task {name!r}; expected one of {sorted(_CANONICAL_TASKS)} "
        f"or aliases {sorted(_TASK_ALIAS)}")


# --------------------------------------------------------------------------- #
# Split a rendered RULER prompt into (haystack -> memory, instruction+question).
# --------------------------------------------------------------------------- #
_CTX_SENTINEL = "\x00__RULER_CTX__\x00"
_QRY_SENTINEL = "\x00__RULER_QRY__\x00"


def _split_context(task: str, prompt: str, vt_icl: str | None):
    """Return (memory_text, gen_prompt) for a rendered RULER prompt.

    RULER templates have a deterministic layout
        <instruction-head> {context} <question-tail>
    where the instruction head and the marker immediately preceding the question
    are context/query-INDEPENDENT. We render the skeleton with sentinel context /
    query to recover those two fixed strings, then peel the haystack off the real
    prompt: ``memory_text`` is exactly the haystack (streamed into the memory
    pool) and ``gen_prompt`` is the instruction + question (+ VT in-context
    example) with the haystack removed — the short prompt we generate from.
    """
    if task == "variable_tracking":
        # num_v value is irrelevant to the head / pre-query marker (it appears in
        # the answer prefix, AFTER the query); 5 = num_hops(4)+1 for concreteness.
        skel = ruler._render_vt(_CTX_SENTINEL, _QRY_SENTINEL, 5, vt_icl or "")
    else:
        skel = ruler._render_niah(_CTX_SENTINEL, _QRY_SENTINEL)
    head, _, after = skel.partition(_CTX_SENTINEL)
    q_head = after.split(_QRY_SENTINEL)[0]  # fixed text between context and query

    if not prompt.startswith(head):
        # Deterministic templates should always match; fall back to injecting the
        # whole prompt as context if the layout ever changes upstream.
        return prompt, prompt[:0]
    ctx_start = len(head)
    ctx_end = prompt.index(q_head, ctx_start)
    memory_text = prompt[ctx_start:ctx_end]
    gen_prompt = head + prompt[ctx_end:]
    return memory_text, gen_prompt


# --------------------------------------------------------------------------- #
# shard-merge scorer (used by --score_only)
# --------------------------------------------------------------------------- #
def score_ruler(outdir: Path, tasks, lengths):
    """Merge every ``{task}_{length}*.json`` shard in ``outdir`` and recompute the
    per-cell ``string_match_all`` recall over the concatenated records."""
    summary: dict = {}
    for task in tasks:
        summary[task] = {}
        for length in lengths:
            shard_files = sorted(outdir.glob(f"{task}_{length}_shard*.json"))
            single = outdir / f"{task}_{length}.json"
            if single.exists():
                shard_files = [single] + list(shard_files)
            if not shard_files:
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
            total = len(records)
            recall_sum = sum(float(r.get("recall", 0.0)) for r in records)
            score = (recall_sum / total * 100.0) if total else 0.0
            summary[task][length] = {"score": round(score, 2), "n": total,
                                     "n_shards": len(shard_files)}
    print("\n[MemoryLLM-RULER] SUMMARY (merged)")
    for task in summary:
        row = "  ".join(f"{ln}={summary[task][ln]['score']:.1f}"
                        for ln in summary[task])
        print(f"  {task:>18}: {row}")
    with open(outdir / "_summary_merged.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[MemoryLLM-RULER] merged summary -> {outdir / '_summary_merged.json'}")
    return summary


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="MemoryLLM — RULER (NIAH/VT) eval driver (QCMem same-class baseline)")
    parser.add_argument("--model_path", type=str, default=mem.DEFAULT_MEMORYLLM_PATH,
                        help="On-disk MemoryLLM-8b-chat snapshot (config + tokenizer "
                             "+ safetensors).")
    parser.add_argument("--ruler_tasks", type=str, nargs="+",
                        default=["niah_single", "niah_multikey"],
                        help="RULER tasks / aliases: niah_single(_1/_2), "
                             "niah_multi(key_1), vt(variable_tracking).")
    parser.add_argument("--lengths", type=str, nargs="+",
                        default=["8k", "16k", "32k"])
    parser.add_argument("--limit", type=int, default=50,
                        help="Samples per (task,length) cell (RULER's num_samples).")
    parser.add_argument("--chunk_size", type=int, default=1024,
                        help="Token chunk size for MemoryLLM context injection "
                             "(matches run_babilong_memoryllm default).")
    parser.add_argument("--max_new_tokens", type=int, default=48,
                        help="Greedy decode budget (RULER uses 48; VT is bumped to "
                             ">=60 to fit the variable list, as in eval_ruler_mem_space).")
    parser.add_argument("--no_chat_template", action="store_true", default=False,
                        help="Generate from the raw prompt instead of wrapping the "
                             "question in the Llama-3 chat template (default: chat "
                             "template, matching MemoryLLM-8b-chat on BABILong).")
    parser.add_argument("--results_folder", type=str,
                        default=str(Path(PROJECT_ROOT) / "ruler_results"))
    parser.add_argument("--output_name", type=str, default="ruler_memoryllm")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn_impl", type=str, default="sdpa")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base RNG seed (per-cell seed is derived stably with "
                             "zlib.crc32 so shards align across processes).")
    parser.add_argument("--verify_memory_reset", action="store_true", default=False,
                        help="Exact-check the first few clean memory resets.")
    parser.add_argument("--verify_resets", type=int, default=3)
    parser.add_argument("--print_examples", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--score_only", action="store_true", default=False,
                        help="Only merge existing per-shard JSON + recompute recall.")
    args = parser.parse_args()

    tasks = [_resolve_task(t) for t in args.ruler_tasks]
    outdir = Path(args.results_folder) / args.output_name

    if args.score_only:
        outdir.mkdir(parents=True, exist_ok=True)
        score_ruler(outdir, tasks, args.lengths)
        return

    if args.num_shards < 1:
        parser.error("--num_shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        parser.error(f"--shard_index must be in [0, {args.num_shards})")

    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    print(f"[MemoryLLM-RULER] model_path={args.model_path}")
    print(f"[MemoryLLM-RULER] tasks={tasks} lengths={args.lengths} "
          f"limit={args.limit} chunk_size={args.chunk_size} "
          f"shard={args.shard_index}/{args.num_shards} dtype={dtype}")

    model, tokenizer, initial_state = mem.load_memoryllm(
        args.model_path, device, dtype, attn_impl=args.attn_impl)

    outdir.mkdir(parents=True, exist_ok=True)
    sharded = args.num_shards > 1
    shard_tag = f"_shard{args.shard_index}of{args.num_shards}" if sharded else ""
    use_chat_template = not args.no_chat_template

    reset_checks_done = 0
    summary: dict = {}
    for task in tqdm(tasks, desc="tasks"):
        summary[task] = {}
        for length in tqdm(args.lengths, desc="lengths", leave=False):
            if length not in ruler._LENGTH_TOKENS:
                print(f"[WARN] unknown length {length}, skipping")
                continue
            target_tokens = ruler._LENGTH_TOKENS[length]
            # STABLE per-(task,length) seed (NOT process-randomised hash()) so
            # every shard process derives the SAME sample set.
            base_seed = args.seed + (
                zlib.crc32(f"{task}\x00{length}".encode()) % 100000)

            vt_icl = None
            if task == "variable_tracking":
                vt_icl = ruler._make_vt_icl(random.Random(base_seed + 777), 4)

            outfile = outdir / f"{task}_{length}{shard_tag}.json"
            if outfile.exists() and not args.overwrite:
                print(f"[MemoryLLM-RULER] Skip existing {outfile}")
                continue

            sample_indices = set(
                list(range(args.limit))[args.shard_index::args.num_shards])
            if sharded:
                print(f"[MemoryLLM-RULER] {task}/{length} shard "
                      f"{args.shard_index}/{args.num_shards}: "
                      f"{len(sample_indices)} of {args.limit} samples")

            mnt = (max(args.max_new_tokens, 60)
                   if task == "variable_tracking" else args.max_new_tokens)

            records = []
            recall_sum = 0.0
            total = 0
            n_tok_seen = 0
            for i in tqdm(range(args.limit), desc=f"{task}/{length}", leave=False):
                # Build EVERY sample (fixed per-i seed) so shard sample sets align,
                # then process only this shard's indices.
                rng = random.Random(base_seed * 1000 + i)
                prompt, answers, gold_needle = ruler._build_sample(
                    task, target_tokens, tokenizer, rng, vt_icl)
                if i not in sample_indices:
                    continue

                memory_text, gen_prompt = _split_context(task, prompt, vt_icl)
                n_tok_seen = len(tokenizer.encode(prompt, add_special_tokens=True))

                verify = args.verify_memory_reset and reset_checks_done < args.verify_resets
                mem.reset_memory(model, initial_state, verify=verify)
                if verify:
                    reset_checks_done += 1
                    print(f"[MemoryLLM-RULER] verified clean memory reset (i={i})")

                try:
                    mem.inject_context(model, tokenizer, memory_text, device,
                                       chunk_size=args.chunk_size)
                    output = mem.generate_answer(
                        model, tokenizer, gen_prompt, device,
                        max_new_tokens=mnt, use_chat_template=use_chat_template)
                except RuntimeError as e:
                    if "out of memory" not in str(e).lower():
                        raise
                    output = "[OOM]"
                    print(f"[OOM] i={i} task={task} length={length}: {e}", flush=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                rec = ruler._string_match_all_one(output, answers)
                recall_sum += rec
                total += 1
                records.append({
                    "sample_index": i, "answers": answers, "output": output,
                    "recall": rec, "n_tokens": n_tok_seen,
                })
                if total <= args.print_examples:
                    print(f"[MemoryLLM-RULER][example] i={i} answers={answers} "
                          f"recall={rec:.2f} output={output!r}")
                if total % 10 == 0:
                    with open(outfile, "w") as f:
                        json.dump({"task": task, "length": length,
                                   "summary": {"score": round(recall_sum / total * 100.0, 2),
                                               "n": total},
                                   "records": records}, f, indent=2)

            score = (recall_sum / total * 100.0) if total else 0.0
            summary[task][length] = {"score": round(score, 2), "n": total,
                                     "approx_tokens": n_tok_seen}
            with open(outfile, "w") as f:
                json.dump({"task": task, "length": length,
                           "summary": summary[task][length],
                           "records": records,
                           "config": {"model_path": args.model_path,
                                      "chunk_size": args.chunk_size,
                                      "max_new_tokens": mnt,
                                      "use_chat_template": use_chat_template,
                                      "num_shards": args.num_shards,
                                      "shard_index": args.shard_index,
                                      "seed": args.seed}}, f, indent=2)
            print(f"[MemoryLLM-RULER] {task}/{length}: recall={score:.2f} "
                  f"({total} samples, ~{n_tok_seen} tok) -> {outfile}")

    print("\n[MemoryLLM-RULER] SUMMARY (this shard)")
    for task in summary:
        row = "  ".join(f"{ln}={summary[task][ln]['score']:.1f}"
                        for ln in summary[task])
        print(f"  {task:>18}: {row}")
    with open(outdir / f"_summary{shard_tag}.json", "w") as f:
        json.dump(summary, f, indent=2)

    if args.num_shards == 1:
        print("\n[MemoryLLM-RULER] merged scoring (single-shard mode)...")
        score_ruler(outdir, tasks, args.lengths)
    print("\n[MemoryLLM-RULER] Evaluation complete!")


if __name__ == "__main__":
    main()
