#!/usr/bin/env python
"""QCMem mid-depth resume — LongEval (LongChat lines-retrieval) eval driver.

The pure-single-hop-exact-retrieval companion to
``scripts/eval_qcmem_babilong.py`` (synthetic BABILong recall),
``scripts/eval_ruler_qcmem.py`` (synthetic RULER NIAH/VT) and
``scripts/eval_qcmem_longbench.py`` (real long-document QA): this runs the SAME
QCMem write/read primitive (``src/memory/qcmem/qcmem_model.py``) on the LongEval
lines-retrieval task — a record of N lines

    line <random-label>: REGISTER_CONTENT is <6-digit number>

after which the model must return the REGISTER_CONTENT of one queried line.
There is no multi-hop, no semantic mixing, no distractor answer — just "can the
fixed read retrieve one EXACT fact under length L". This is the cleanest possible
retrieval benchmark and the natural home turf of a retrieval memory: bm25 has a
rare, discriminative needle key (the queried line label) to lock onto, while the
no-retrieval baselines must pack the whole record.

Like the RULER / LongBench QCMem drivers, this is a thin composition of two
existing, unmodified drivers — nothing about the QCMem forward path or the
LongEval prompt synthesis / judging口径 is re-implemented:

  QCMem forward path (imported from ``scripts/eval_qcmem_babilong.py``):
    * ``qcmem_generate``  — chunk the prompt -> write_chunk each chunk to depth j
                            -> selector picks topk context chunks (bm25/recency/
                            oracle/reader_attn) -> read (pack [sink; selected h_j;
                            query h_j], resume layers[j:]) -> greedy decode. Its
                            ``no_retrieval`` arm packs EVERY context chunk (the
                            KV-Direct / HCache baselines).
    * ``run_self_test``   — j=0 correctness gate (QCMem read == full forward,
                            fp32 max|logit diff| < 1e-4).
    * ``QCMemModel``      — the write/read orchestrator (read-only backbone).
    * ``harness``         — ``_locate_needle_chunks`` (oracle needle locator).

  LongEval prompt synthesis + judging (imported from
  ``scripts/eval_longeval_mem_space.py``):
    * ``build_lines_prompt`` — synthesize one lines-retrieval sample sized to
                               ~target tokens, returning (prompt, expected_value,
                               target_label, n_lines).
    * ``extract_prediction`` — pull the first >=4-digit run from the output.
    * ``_LENGTH_TOKENS``     — length-bucket -> target token budget.
  A sample is correct iff ``extract_prediction(output) == expected_value``
  (identical to eval_longeval_mem_space's判分).

Baselines (``--baseline``, mirrors eval_ruler_qcmem.py / eval_qcmem_longbench.py):
  * ``none``     — normal QCMem (retrieval topk + resume_j + optional LoRA).
  * ``kvdirect`` (2603.19664) — full-depth recompute (forces resume_j=0) + NO
                                retrieval (packs every chunk) + no LoRA
                                (training-free). Read grows O(context).
  * ``hcache``   (2410.05004) — mid-layer recompute (keeps --resume_j) + NO
                                retrieval (packs every chunk) + no LoRA (post-hoc).
                                Read grows O(context).

Model arms (mutually exclusive, mirrors the other QCMem drivers):
  * plain ``--model_path`` backbone (zero-training QCMem),
  * ``--lora_adapter``    — a trained QCMem-distill LoRA (Direction A),
  * ``--bottleneck_ckpt`` — a continued-pretrain funnel-Qwen checkpoint
                            (RECOMMENDED --resume_j == bottleneck_layer+1).

Sharding: samples for each length are generated with a STABLE per-sample RNG
(``zlib.crc32``-derived, not Python's process-randomized ``hash()``), so every
shard sees an identical sample set and shard ``s`` evaluates only the strided
slice ``range(num_samples)[s::num_shards]``. Merge with ``--score_only`` (globs
every ``longeval_<length>*.json`` shard, concatenates the disjoint records and
recomputes per-length accuracy).

Usage (QCMem-distill LoRA on Qwen3-8B, three-way head-to-head):
    # QCMem (retrieval, fixed read)
    python scripts/eval_qcmem_longeval.py \
        --model_path /apdcephfs_wzc1/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b \
        --resume_j 12 --lora_adapter outputs/qcmem_distill_qwen_j12_r32_4k/final \
        --selector bm25 --topk 12 --sink_tokens bos \
        --lengths 4k 8k 16k 32k --num_samples 50 \
        --output_name longeval_qcmem_j12
    # HCache baseline (no retrieval, mid-layer recompute)
    python scripts/eval_qcmem_longeval.py --baseline hcache \
        --model_path .../Qwen--Qwen3-8b --resume_j 12 \
        --lengths 4k 8k 16k 32k --num_samples 50 \
        --output_name longeval_hcache_j12
    # Score only (merge all shards):
    python scripts/eval_qcmem_longeval.py --score_only \
        --lengths 4k 8k 16k 32k --results_folder ./longeval_results \
        --output_name longeval_qcmem_j12
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

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

# QCMem forward path — reused verbatim, unmodified (same import the RULER driver
# uses; loads its explicit-file-path babilong harness on import).
import scripts.eval_qcmem_babilong as qcb  # noqa: E402

# LongEval prompt synthesis + judging — reused verbatim, unmodified. Importing
# this module pulls in the mem_space model helpers but does NOT patch any model
# (QCMem never calls apply_mem_space_to_model / load_mem_space_model).
import scripts.eval_longeval_mem_space as le  # noqa: E402

QCMemModel = qcb.QCMemModel
qcmem_generate = qcb.qcmem_generate
run_self_test = qcb.run_self_test

build_lines_prompt = le.build_lines_prompt
extract_prediction = le.extract_prediction
_LENGTH_TOKENS = le._LENGTH_TOKENS


# --------------------------------------------------------------------------- #
# retrieval query + oracle needle
# --------------------------------------------------------------------------- #
def _bm25_query(target_label: str) -> str:
    """The bm25 lexical query for a LongEval sample == the queried line label.

    The needle chunk is exactly ``line {target_label}: REGISTER_CONTENT is
    <value>`` and the trailing question repeats ``line {target_label}``, so the
    (rare, hyphenated) label subwords are the maximally-discriminative retrieval
    key — bm25 over token ids will lock onto the one chunk that contains them."""
    return f"line {target_label}"


def _oracle_needle_chunks(input_ids, expected_value, target_label,
                          tokenizer, chunk_size):
    """Document-absolute chunk index set containing the queried line, for the
    NIAH-style oracle selector. LongEval has a single, unambiguous gold needle
    (the target line), so oracle IS well-defined here (unlike LongBench). Try the
    exact 6-digit REGISTER_CONTENT value first (most distinctive), then the
    bracketed ``<value>`` form, then the line label. Returns None if nothing can
    be located (caller's selector then degrades to recency)."""
    probes = []
    if expected_value:
        probes.extend([expected_value, f"<{expected_value}>"])
    if target_label:
        probes.append(f"line {target_label}")
    for probe in probes:
        if not probe:
            continue
        chunks = qcb.harness._locate_needle_chunks(
            input_ids, probe, tokenizer, chunk_size)
        if chunks:
            return chunks
    return None


# --------------------------------------------------------------------------- #
# shard-merge scorer (used by --score_only and single-shard auto-score)
# --------------------------------------------------------------------------- #
def score_longeval(outdir: Path, lengths):
    """Merge every ``longeval_<length>*.json`` shard in ``outdir`` and recompute
    per-length accuracy over the concatenated (disjoint) records.

    Shards are generated over disjoint strided sample slices of an identical
    sample set, so no dedup is needed — the union of records IS the full set.
    Prints a SUMMARY table and writes ``_summary_merged.json``."""
    summary: dict = {}
    for length in lengths:
        shard_files = sorted(outdir.glob(f"longeval_{length}_shard*.json"))
        single = outdir / f"longeval_{length}.json"
        if single.exists():
            shard_files = [single] + [f for f in shard_files]
        if not shard_files:
            print(f"[score] {length}: no shard files found in {outdir}")
            continue
        records = []
        seen_keys = set()
        for sf in shard_files:
            try:
                with open(sf) as f:
                    payload = json.load(f)
            except Exception as e:
                print(f"[score][WARN] failed to read {sf}: {e}")
                continue
            for r in payload.get("records", []):
                # dedup guard: (sample_index) if present else the record identity
                key = r.get("sample_index")
                if key is not None:
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                records.append(r)
        correct = sum(int(r.get("correct", False)) for r in records)
        total = len(records)
        acc = correct / total if total else 0.0
        read_lens = [r["read_len"] for r in records if r.get("read_len") is not None]
        summary[length] = {
            "accuracy": round(acc, 4), "correct": correct, "total": total,
            "avg_read_len": round(sum(read_lens) / len(read_lens), 1) if read_lens else None,
            "n_shards": len(shard_files),
        }
    print("\n[QCMem-LongEval] SUMMARY (merged)")
    for length, s in summary.items():
        print(f"  {length:>4}: acc={s['accuracy']:.3f}  "
              f"({s['correct']}/{s['total']})  "
              f"read_len~{s['avg_read_len']}  ({s['n_shards']} shards)")
    with open(outdir / "_summary_merged.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[QCMem-LongEval] merged summary -> {outdir / '_summary_merged.json'}")
    return summary


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="QCMem mid-depth resume — LongEval lines-retrieval eval driver"
    )
    # --- model arm (aligned with the other QCMem drivers) ---
    parser.add_argument("--model_path", type=str, default="",
                        help="Path to plain backbone weights (Qwen3-8B / "
                             "Llama-3-8B). Required unless --score_only.")
    parser.add_argument("--resume_j", type=int, default=12,
                        help="Layer split index j (0=RAG upper bound, L=closed-book).")
    parser.add_argument("--top_prepay_b", type=int, default=0,
                        help="Direction-B top-prepay: run the top b layers "
                             "query-local at read (0=exact connective resume).")
    parser.add_argument("--reuse_kv_blockdiag", action="store_true", default=False,
                        help="QCMem ablation (ii): block-diagonal read attention "
                             "(sink global; each chunk within-block only; query "
                             "reads sink+all chunks+itself). Only valid with "
                             "--top_prepay_b 0 and --baseline none.")
    parser.add_argument("--lora_adapter", type=str, default="",
                        help="Optional path to a trained QCMem-distill LoRA "
                             "adapter dir (Direction A). Mutually exclusive with "
                             "--bottleneck_ckpt.")
    parser.add_argument("--bottleneck_ckpt", type=str, default="",
                        help="Optional path to a continued-pretrain funnel-Qwen "
                             "checkpoint (*.pt with 'model_state' + arch_meta.json "
                             "next to it, from scripts/train_qwen_bottleneck_"
                             "continued.py). Rebuilds 'stock backbone + mid-layer "
                             "BottleneckLayer funnel' then load_state_dict. "
                             "RECOMMENDED --resume_j == bottleneck_layer+1. "
                             "Mutually exclusive with --lora_adapter.")
    parser.add_argument("--baseline", type=str, default="none",
                        choices=["none", "kvdirect", "hcache"],
                        help="Mechanism-level head-to-head baseline (mirrors "
                             "eval_ruler_qcmem.py / eval_qcmem_babilong.py). "
                             "'none' = normal QCMem (retrieval topk + resume_j + "
                             "optional LoRA). 'kvdirect' (2603.19664) = full-depth "
                             "recompute (forces resume_j=0) + NO retrieval (packs "
                             "every chunk) + no LoRA (training-free) — read grows "
                             "O(context). 'hcache' (2410.05004) = mid-layer "
                             "recompute (keeps --resume_j) + NO retrieval (packs "
                             "every chunk) + no LoRA (post-hoc) — read grows "
                             "O(context).")
    parser.add_argument("--selector", type=str, default="bm25",
                        choices=["bm25", "recency", "oracle", "reader_attn"],
                        help="Chunk selector for the read pack. LongEval has a "
                             "single gold needle (the queried line), so oracle IS "
                             "supported (locates the line by its exact "
                             "REGISTER_CONTENT value / label). bm25 is the "
                             "deployable default and the natural fit here.")
    parser.add_argument("--topk", type=int, default=12,
                        help="Number of context chunks to pack into the read.")
    parser.add_argument("--sink_tokens", type=str, default="bos",
                        choices=["bos", "none"],
                        help="Attention-sink anchor at packed position 0.")
    parser.add_argument("--chunk_size", type=int, default=512,
                        help="QCMem chunk size (prompt split into chunk_size "
                             "segments; matches the other QCMem drivers).")
    # --- LongEval task framework ---
    parser.add_argument("--lengths", type=str, nargs="+",
                        default=["4k", "8k", "16k", "32k"],
                        help=f"Length buckets to eval (subset of {sorted(_LENGTH_TOKENS)}).")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Samples per length bucket.")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234,
                        help="Base RNG seed (per-sample seed is derived stably "
                             "with zlib.crc32 so shards align across processes).")
    parser.add_argument("--results_folder", type=str, default="./longeval_results")
    parser.add_argument("--output_name", type=str, required=True)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn_impl", type=str, default="sdpa")
    parser.add_argument("--score_only", action="store_true",
                        help="Only merge existing per-shard JSON + recompute acc.")
    parser.add_argument("--self_test", action="store_true", default=False,
                        help="Run the shared QCMem j=0 correctness gate and exit.")
    args = parser.parse_args()

    outdir = Path(args.results_folder) / args.output_name

    # --- score-only: merge shards + recompute per-length accuracy, then exit ---
    if args.score_only:
        score_longeval(outdir, args.lengths)
        return

    if args.num_shards < 1:
        parser.error("--num_shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        parser.error(f"--shard_index must be in [0, {args.num_shards})")
    if not args.model_path:
        parser.error("--model_path is required unless --score_only")

    # --- head-to-head baseline resolution (identical to the other QCMem drivers) --
    no_retrieval = (args.baseline != "none")
    if args.bottleneck_ckpt and args.lora_adapter:
        parser.error("--bottleneck_ckpt (funnel-Qwen arm) and --lora_adapter "
                     "(stock-Qwen LoRA arm) are mutually exclusive; pick one.")
    if args.baseline == "kvdirect":
        if args.resume_j != 0:
            print(f"[QCMem-LongEval] baseline=kvdirect -> forcing resume_j "
                  f"{args.resume_j} -> 0 (full-depth K/V recompute).")
        args.resume_j = 0
        if args.lora_adapter:
            print("[QCMem-LongEval] baseline=kvdirect is training-free -> "
                  f"ignoring --lora_adapter {args.lora_adapter!r}.")
            args.lora_adapter = ""
    elif args.baseline == "hcache":
        if args.lora_adapter:
            print("[QCMem-LongEval] baseline=hcache is post-hoc (no training) -> "
                  f"ignoring --lora_adapter {args.lora_adapter!r}.")
            args.lora_adapter = ""
    if no_retrieval and args.reuse_kv_blockdiag:
        parser.error("--reuse_kv_blockdiag is a QCMem ablation and is incompatible "
                     "with --baseline (kvdirect/hcache pack all chunks with the "
                     "standard causal read).")

    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16,
             "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    if args.self_test:
        dtype = torch.float32  # tight <1e-4 gate needs fp32

    model_path = args.model_path
    if not os.path.isabs(model_path):
        model_path = os.path.join(PROJECT_ROOT, model_path)

    print(f"[QCMem-LongEval] model_path={model_path}")
    print(f"[QCMem-LongEval] baseline={args.baseline} "
          f"(no_retrieval={no_retrieval}) resume_j={args.resume_j} "
          f"top_prepay_b={args.top_prepay_b} reuse_kv_blockdiag={args.reuse_kv_blockdiag} "
          f"selector={args.selector} topk={args.topk} sink={args.sink_tokens} "
          f"chunk_size={args.chunk_size} dtype={dtype} attn_impl={args.attn_impl}")
    print(f"[QCMem-LongEval] lengths={args.lengths} num_samples={args.num_samples} "
          f"shard={args.shard_index}/{args.num_shards}")

    # local_files_only=True: offline nodes otherwise treat a local dir path as an
    # HF repo_id and error ("Repo id must be in the form ...").
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_impl,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device).eval()

    L = int(model.config.num_hidden_layers)
    if not (0 <= args.resume_j <= L):
        parser.error(f"--resume_j must be in [0, {L}] for this model; got {args.resume_j}")
    if not (0 <= args.top_prepay_b <= L - args.resume_j):
        parser.error(f"--top_prepay_b must be in [0, {L - args.resume_j}]; got {args.top_prepay_b}")
    if args.reuse_kv_blockdiag and args.top_prepay_b != 0:
        parser.error("--reuse_kv_blockdiag requires --top_prepay_b 0")

    # Direction A: load a trained QCMem-distill LoRA adapter onto the backbone.
    if args.lora_adapter:
        from peft import PeftModel
        print(f"[QCMem-LongEval] loading LoRA adapter: {args.lora_adapter}")
        peft_model = PeftModel.from_pretrained(model, args.lora_adapter).eval()
        model = peft_model.base_model.model

    # Funnel-Qwen arm: rebuild "stock backbone + mid-layer BottleneckLayer funnel"
    # exactly as continued-pretrain saved it, then load the full state_dict.
    # Identical to the eval_ruler_qcmem.py / eval_qcmem_longbench.py funnel loader.
    if args.bottleneck_ckpt:
        from scripts.train_qwen_bottleneck_continued import inject_bottleneck
        meta_path = os.path.join(
            os.path.dirname(os.path.abspath(args.bottleneck_ckpt)), "arch_meta.json")
        if not os.path.exists(meta_path):
            parser.error(f"--bottleneck_ckpt given but arch_meta.json not found "
                         f"next to it at {meta_path}")
        with open(meta_path) as f:
            meta = json.load(f)
        b_layer = int(meta["bottleneck_layer"])
        b_dim = int(meta["bottleneck_dim"])
        print(f"[QCMem-LongEval] funnel-Qwen: arch_meta {meta_path} -> "
              f"bottleneck_layer={b_layer} bottleneck_dim={b_dim} "
              f"num_hidden_layers={meta.get('num_hidden_layers')}")
        inject_bottleneck(model, b_layer, b_dim, dtype)
        ck = torch.load(args.bottleneck_ckpt, map_location="cpu")
        state = ck.get("model_state", ck)
        missing, unexpected = model.load_state_dict(state, strict=False)
        bad_missing = [k for k in missing if "inv_freq" not in k]
        if bad_missing or unexpected:
            print(f"[QCMem-LongEval][WARN] load_state_dict missing={bad_missing[:8]}"
                  f"{'...' if len(bad_missing) > 8 else ''} "
                  f"unexpected={unexpected[:8]}"
                  f"{'...' if len(unexpected) > 8 else ''}")
        model = model.to(device).eval()
        step = ck.get("step")
        print(f"[QCMem-LongEval] funnel-Qwen loaded from {args.bottleneck_ckpt} "
              f"(step={step}). RECOMMENDED --resume_j == bottleneck_layer+1 "
              f"(={b_layer + 1}); you passed --resume_j {args.resume_j}.")

    if args.self_test:
        ok = run_self_test(model, tokenizer, device, args.chunk_size)
        sys.exit(0 if ok else 1)

    qc = QCMemModel(model, resume_j=args.resume_j, top_prepay_b=args.top_prepay_b,
                    block_diagonal=args.reuse_kv_blockdiag)

    outdir.mkdir(parents=True, exist_ok=True)
    sharded = args.num_shards > 1
    shard_tag = f"_shard{args.shard_index}of{args.num_shards}" if sharded else ""

    # Record the eval config next to the predictions.
    with open(outdir / f"eval_config{shard_tag}.json", "w") as f:
        cfg = dict(vars(args))
        cfg.update({"no_retrieval": bool(no_retrieval), "num_layers": L,
                    "resolved_model_path": model_path})
        json.dump(cfg, f, indent=2)

    summary: dict = {}
    for length in args.lengths:
        if length not in _LENGTH_TOKENS:
            print(f"[WARN] unknown length {length}, skipping")
            continue
        target_tokens = _LENGTH_TOKENS[length]
        # STABLE per-length seed (NOT Python's process-randomized hash()) so every
        # shard process derives the SAME per-sample seeds -> identical sample set.
        length_seed = args.seed + (zlib.crc32(length.encode()) % 100000)

        sample_indices = list(range(args.num_samples))[args.shard_index::args.num_shards]
        if sharded:
            print(f"[QCMem-LongEval] {length} shard {args.shard_index}/"
                  f"{args.num_shards}: {len(sample_indices)} of {args.num_samples} samples")

        records = []
        correct = 0
        total = 0
        n_tok_seen = 0
        read_len_sum = 0
        read_len_last = 0
        t0 = time.time()
        for pos, i in enumerate(tqdm(sample_indices, desc=f"{length}", leave=False)):
            # Per-sample RNG (stable across processes) — build ONLY this shard's
            # samples (no wasteful build-then-skip of the full set).
            rng = random.Random(length_seed * 1000 + i)
            prompt, expected, target_label, n_lines = build_lines_prompt(
                target_tokens, tokenizer, rng)

            ids = tokenizer.encode(prompt, add_special_tokens=True,
                                   return_tensors="pt")
            if isinstance(ids, list):
                ids = torch.tensor([ids], dtype=torch.long)
            input_ids = ids.to(device)
            n_tok_seen = int(input_ids.shape[1])

            bare_q_ids = tokenizer.encode(
                _bm25_query(target_label), add_special_tokens=False)

            # oracle needle chunks (LongEval has a single gold line).
            needle_set = None
            if args.selector == "oracle":
                needle_set = _oracle_needle_chunks(
                    input_ids, expected, target_label, tokenizer, args.chunk_size)

            gen_stats: dict = {}
            try:
                output = qcmem_generate(
                    qc=qc, tokenizer=tokenizer, input_ids=input_ids,
                    chunk_size=args.chunk_size, max_new_tokens=args.max_new_tokens,
                    selector=args.selector, topk=args.topk,
                    sink_tokens=args.sink_tokens,
                    needle_chunk_set=needle_set, bare_question_ids=bare_q_ids,
                    no_retrieval=no_retrieval, stats=gen_stats,
                )
                if "read_len" in gen_stats:
                    read_len_last = int(gen_stats["read_len"])
                    read_len_sum += read_len_last
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                output = "[OOM]"
                print(f"[OOM] i={i} length={length} n_tok={n_tok_seen}: {e}",
                      flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            pred = extract_prediction(output)
            ok = (pred == expected)
            correct += int(ok)
            total += 1
            records.append({
                "sample_index": i,
                "label": target_label, "expected": expected,
                "output": output, "pred": pred, "correct": ok,
                "n_lines": n_lines, "n_tokens": n_tok_seen,
                "read_len": gen_stats.get("read_len"),
                "n_selected_chunks": gen_stats.get("n_selected_chunks"),
                "n_context_chunks": gen_stats.get("n_context_chunks"),
            })

            if (pos + 1) % 10 == 0 or pos == len(sample_indices) - 1:
                acc_cur = correct / total if total else 0.0
                with open(outdir / f"longeval_{length}{shard_tag}.json", "w") as f:
                    json.dump({"length": length,
                               "summary": {"accuracy": round(acc_cur, 4),
                                           "correct": correct, "total": total},
                               "records": records}, f, indent=2)

        acc = correct / total if total else 0.0
        avg_read_len = round(read_len_sum / total, 1) if total else 0
        summary[length] = {
            "accuracy": round(acc, 4), "correct": correct, "total": total,
            "approx_tokens": n_tok_seen, "avg_read_len": avg_read_len,
            "last_read_len": read_len_last,
        }
        with open(outdir / f"longeval_{length}{shard_tag}.json", "w") as f:
            json.dump({"length": length, "summary": summary[length],
                       "records": records}, f, indent=2)
        print(f"[QCMem-LongEval] {length}: acc={acc:.3f} ({correct}/{total}) "
              f"~{n_tok_seen} tok  read_len~{avg_read_len}  "
              f"({time.time()-t0:.1f}s) -> longeval_{length}{shard_tag}.json")

    print("\n[QCMem-LongEval] SUMMARY (this shard)")
    for length, s in summary.items():
        print(f"  {length:>4}: acc={s['accuracy']:.3f}  "
              f"({s['correct']}/{s['total']})  ~{s['approx_tokens']} tok  "
              f"read_len~{s['avg_read_len']}")
    with open(outdir / f"_summary{shard_tag}.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Single-shard: auto-merge (multi-shard: run --score_only after all finish).
    if args.num_shards == 1:
        print("\n[QCMem-LongEval] Running merged scoring (single-shard mode)...")
        score_longeval(outdir, args.lengths)

    print("\n[QCMem-LongEval] Evaluation complete!")


if __name__ == "__main__":
    main()
