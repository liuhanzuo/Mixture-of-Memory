#!/usr/bin/env python
"""QCMem mid-depth resume — LongBench (real long-document) eval driver.

The real-long-document companion to ``scripts/eval_qcmem_babilong.py`` (synthetic
BABILong recall) and ``scripts/eval_ruler_qcmem.py`` (synthetic RULER NIAH/VT):
this runs the SAME QCMem write/read primitive (``src/memory/qcmem/qcmem_model.py``)
on LongBench's genuine long-context QA tasks (narrativeqa / qasper / hotpotqa /
2wikimqa / musique / multifieldqa_en), which reviewers expect over the synthetic
benchmarks.

Like the RULER driver, this is a thin composition of two existing, unmodified
drivers — nothing about the QCMem forward path or the LongBench scoring口径 is
re-implemented:

  QCMem forward path (imported from ``scripts/eval_qcmem_babilong.py``):
    * ``qcmem_generate``  — chunk the prompt -> write_chunk each chunk to depth j
                            -> selector picks topk context chunks (bm25/recency/
                            reader_attn) -> read (pack [sink; selected h_j; query
                            h_j], resume layers[j:]) -> greedy decode. Its
                            ``no_retrieval`` arm packs EVERY context chunk (the
                            KV-Direct / HCache baselines).
    * ``run_self_test``   — j=0 correctness gate (QCMem read == full forward,
                            fp32 max|logit diff| < 1e-4).
    * ``QCMemModel``      — the write/read orchestrator (read-only backbone).

  LongBench task framework (imported from ``scripts/eval_longbench_mem_space.py``):
    * ``load_longbench_dataset`` — offline JSONL loading from data/longbench_raw.
    * ``DATASET2PROMPT`` / ``DATASET2MAXGEN`` / ``format_prompt`` — the MemoryLLM
                                   prompt templates + per-dataset gen budgets.
    * ``compute_f1_multi`` / ``run_scoring`` — SQuAD-style token-F1 + EM (the
                                   official LongBench ``qa_f1_score`` metric for
                                   every QA subtask here; the driver writes the
                                   same ``{index, pred, answers, dataset}`` JSONL
                                   layout so ``run_scoring`` merges shards as-is).

Baselines (``--baseline``, mirrors eval_ruler_qcmem.py exactly):
  * ``none``     — normal QCMem (retrieval topk + resume_j + optional LoRA).
  * ``kvdirect`` (2603.19664) — full-depth recompute (forces resume_j=0) + NO
                                retrieval (packs every chunk) + no LoRA
                                (training-free). Read grows O(context).
  * ``hcache``   (2410.05004) — mid-layer recompute (keeps --resume_j) + NO
                                retrieval (packs every chunk) + no LoRA (post-hoc).
                                Read grows O(context).

Model arms (mutually exclusive, mirrors eval_ruler_qcmem.py):
  * plain ``--model_path`` backbone (zero-training QCMem),
  * ``--lora_adapter``    — a trained QCMem-distill LoRA (Direction A),
  * ``--bottleneck_ckpt`` — a continued-pretrain funnel-Qwen checkpoint
                            (RECOMMENDED --resume_j == bottleneck_layer+1).

Usage (QCMem-distill LoRA on Qwen3-8B, head-to-head three-way):
    # QCMem (retrieval, fixed read)
    python scripts/eval_qcmem_longbench.py \
        --model_path /apdcephfs_wzc1/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b \
        --resume_j 12 --lora_adapter outputs/qcmem_distill_qwen_j12_r32_4k/final \
        --selector bm25 --topk 12 --sink_tokens bos \
        --tasks narrativeqa qasper hotpotqa 2wikimqa \
        --output_dir longbench_results/qcmem_j12 --num_shards 1 --shard_index 0
    # HCache baseline (no retrieval, mid-layer recompute)
    python scripts/eval_qcmem_longbench.py --baseline hcache \
        --model_path .../Qwen--Qwen3-8b --resume_j 12 \
        --tasks narrativeqa qasper hotpotqa 2wikimqa \
        --output_dir longbench_results/hcache_j12
    # Score only (merge all shards):
    python scripts/eval_qcmem_longbench.py --score_only \
        --tasks narrativeqa qasper hotpotqa 2wikimqa \
        --output_dir longbench_results/qcmem_j12
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
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

# LongBench task framework (data loading + prompt templates + F1/EM scoring) —
# reused verbatim, unmodified. Importing it defines mem_space classes but does
# NOT patch any model (QCMem never calls apply_mem_space_to_model).
import scripts.eval_longbench_mem_space as lb  # noqa: E402

QCMemModel = qcb.QCMemModel
qcmem_generate = qcb.qcmem_generate
run_self_test = qcb.run_self_test


def _bare_question(sample: dict) -> str:
    """The bm25 lexical query for a LongBench sample == the question ('input').

    LongBench templates place the (short) question in the ``{input}`` slot and
    the (long) document in ``{context}``; the question terms are exactly what
    bm25 should retrieve context chunks on."""
    return (sample.get("input") or "").strip()


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="QCMem mid-depth resume — LongBench eval driver"
    )
    # --- model arm (aligned with scripts/eval_ruler_qcmem.py) ---
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
                        choices=["bm25", "recency", "reader_attn"],
                        help="Chunk selector for the read pack (oracle is NOT "
                             "supported on LongBench — no single gold-answer span "
                             "to locate). bm25 is the deployable default.")
    parser.add_argument("--topk", type=int, default=12,
                        help="Number of context chunks to pack into the read.")
    parser.add_argument("--sink_tokens", type=str, default="bos",
                        choices=["bos", "none"],
                        help="Attention-sink anchor at packed position 0.")
    parser.add_argument("--chunk_size", type=int, default=512,
                        help="QCMem chunk size (prompt split into chunk_size "
                             "segments; matches eval_qcmem_babilong/ruler).")
    # --- LongBench task framework ---
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        help="LongBench subtasks to eval (default: the 6 QA tasks "
                             f"{lb.DEFAULT_DATASETS}). All defaults use the official "
                             "qa_f1_score metric reproduced by compute_f1_multi.")
    parser.add_argument("--output_dir", type=str,
                        default="longbench_results/qcmem",
                        help="Directory for per-shard prediction JSONL + scores.")
    parser.add_argument("--hf_dataset", type=str, default="THUDM/LongBench")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Max samples per task (-1 = all).")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--use_chat_template", action="store_true", default=False,
                        help="Wrap each prompt in the tokenizer chat template. "
                             "Default OFF, matching QCMem's BABILong/RULER drivers "
                             "(raw-completion prompts). Turn ON when evaluating an "
                             "instruct backbone that expects the chat wrapper.")
    parser.add_argument("--enable_thinking", action="store_true", default=False,
                        help="When --use_chat_template is set, keep the backbone's "
                             "thinking/reasoning mode ON (Qwen3 enable_thinking=True). "
                             "Default OFF: pass enable_thinking=False to apply_chat_template "
                             "so Qwen3 does not emit <think>...</think> that pollutes "
                             "scoring and wastes the generation budget. "
                             "Silently ignored for tokenizers that do not support the kwarg.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn_impl", type=str, default="sdpa")
    parser.add_argument("--score_only", action="store_true",
                        help="Only merge existing per-shard JSONL + compute F1/EM.")
    parser.add_argument("--self_test", action="store_true", default=False,
                        help="Run the shared QCMem j=0 correctness gate and exit.")
    args = parser.parse_args()

    datasets_list = args.tasks if args.tasks else lb.DEFAULT_DATASETS

    # --- score-only: reuse the LongBench shard-merge + F1/EM scorer verbatim ---
    if args.score_only:
        lb.run_scoring(args.output_dir, datasets_list)
        return

    if args.num_shards < 1:
        parser.error("--num_shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        parser.error(f"--shard_index must be in [0, {args.num_shards})")
    if not args.model_path:
        parser.error("--model_path is required unless --score_only")

    # --- head-to-head baseline resolution (identical to eval_ruler_qcmem.py) ---
    no_retrieval = (args.baseline != "none")
    if args.bottleneck_ckpt and args.lora_adapter:
        parser.error("--bottleneck_ckpt (funnel-Qwen arm) and --lora_adapter "
                     "(stock-Qwen LoRA arm) are mutually exclusive; pick one.")
    if args.baseline == "kvdirect":
        if args.resume_j != 0:
            print(f"[QCMem-LongBench] baseline=kvdirect -> forcing resume_j "
                  f"{args.resume_j} -> 0 (full-depth K/V recompute).")
        args.resume_j = 0
        if args.lora_adapter:
            print("[QCMem-LongBench] baseline=kvdirect is training-free -> "
                  f"ignoring --lora_adapter {args.lora_adapter!r}.")
            args.lora_adapter = ""
    elif args.baseline == "hcache":
        if args.lora_adapter:
            print("[QCMem-LongBench] baseline=hcache is post-hoc (no training) -> "
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

    print(f"[QCMem-LongBench] model_path={model_path}")
    print(f"[QCMem-LongBench] baseline={args.baseline} "
          f"(no_retrieval={no_retrieval}) resume_j={args.resume_j} "
          f"top_prepay_b={args.top_prepay_b} reuse_kv_blockdiag={args.reuse_kv_blockdiag} "
          f"selector={args.selector} topk={args.topk} sink={args.sink_tokens} "
          f"chunk_size={args.chunk_size} chat_template={args.use_chat_template} "
          f"dtype={dtype} attn_impl={args.attn_impl}")
    print(f"[QCMem-LongBench] tasks={datasets_list} max_samples={args.max_samples} "
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
        print(f"[QCMem-LongBench] loading LoRA adapter: {args.lora_adapter}")
        peft_model = PeftModel.from_pretrained(model, args.lora_adapter).eval()
        model = peft_model.base_model.model

    # Funnel-Qwen arm: rebuild "stock backbone + mid-layer BottleneckLayer funnel"
    # exactly as continued-pretrain saved it, then load the full state_dict. Reuses
    # ``inject_bottleneck`` from the train script (structure must match the ckpt
    # keys). Identical to the eval_ruler_qcmem.py funnel loader.
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
        print(f"[QCMem-LongBench] funnel-Qwen: arch_meta {meta_path} -> "
              f"bottleneck_layer={b_layer} bottleneck_dim={b_dim} "
              f"num_hidden_layers={meta.get('num_hidden_layers')}")
        inject_bottleneck(model, b_layer, b_dim, dtype)
        ck = torch.load(args.bottleneck_ckpt, map_location="cpu")
        state = ck.get("model_state", ck)
        missing, unexpected = model.load_state_dict(state, strict=False)
        bad_missing = [k for k in missing if "inv_freq" not in k]
        if bad_missing or unexpected:
            print(f"[QCMem-LongBench][WARN] load_state_dict missing={bad_missing[:8]}"
                  f"{'...' if len(bad_missing) > 8 else ''} "
                  f"unexpected={unexpected[:8]}"
                  f"{'...' if len(unexpected) > 8 else ''}")
        model = model.to(device).eval()
        step = ck.get("step")
        print(f"[QCMem-LongBench] funnel-Qwen loaded from {args.bottleneck_ckpt} "
              f"(step={step}). RECOMMENDED --resume_j == bottleneck_layer+1 "
              f"(={b_layer + 1}); you passed --resume_j {args.resume_j}.")

    if args.self_test:
        ok = run_self_test(model, tokenizer, device, args.chunk_size)
        sys.exit(0 if ok else 1)

    qc = QCMemModel(model, resume_j=args.resume_j, top_prepay_b=args.top_prepay_b,
                    block_diagonal=args.reuse_kv_blockdiag)

    # Load LongBench data (offline JSONL) via the shared loader.  When
    # --hf_dataset names a local directory, use it as data_dir explicitly;
    # otherwise retain the historical HF dataset-id fallback.
    local_data_dir = args.hf_dataset if os.path.isdir(args.hf_dataset) else None
    all_data = lb.load_longbench_dataset(
        args.hf_dataset, datasets_list, data_dir=local_data_dir)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    sharded = args.num_shards > 1
    # Per-shard file: run_scoring globs "{ds}_*.jsonl" and dedups by "index".
    shard_tag = f"shard{args.shard_index}of{args.num_shards}" if sharded else "0"

    # Record the eval config next to the predictions.  Keep the literal CLI
    # values for provenance, while also exposing normalized protocol fields
    # that downstream validators can check without interpreting empty strings.
    task_tag = datasets_list[0] if len(datasets_list) == 1 else "multi"
    with open(output_path / f"eval_config_{task_tag}_{shard_tag}.json", "w") as f:
        raw_args = dict(vars(args))
        zero_training_no_adapter = (
            args.baseline == "none"
            and not args.lora_adapter
            and not args.bottleneck_ckpt
        )
        cfg = dict(raw_args)
        cfg.update({"no_retrieval": bool(no_retrieval), "num_layers": L,
                    "resolved_model_path": model_path,
                    "raw_args": raw_args,
                    "lora_adapter": args.lora_adapter or None,
                    "bottleneck_ckpt": args.bottleneck_ckpt or None,
                    "resume_j": int(args.resume_j),
                    "chunk_size": int(args.chunk_size),
                    "selector": args.selector,
                    "topk": int(args.topk),
                    "chat_template": bool(args.use_chat_template),
                    "zero_training_no_adapter": zero_training_no_adapter})
        json.dump(cfg, f, indent=2)

    for ds_name in datasets_list:
        samples = all_data.get(ds_name, [])
        if not samples:
            print(f"[QCMem-LongBench] Skipping {ds_name} (no data)")
            continue
        if args.max_samples > 0:
            samples = samples[:args.max_samples]

        # Strided shard (matches the babilong/ruler [shard_index::num_shards]
        # convention; global index recorded so run_scoring dedups correctly).
        sample_indices = list(range(len(samples)))[args.shard_index::args.num_shards]
        if sharded:
            print(f"[QCMem-LongBench] {ds_name} shard {args.shard_index}/"
                  f"{args.num_shards}: {len(sample_indices)} of {len(samples)} samples")

        max_gen = lb.DATASET2MAXGEN.get(ds_name, 64)
        outfile = output_path / f"{ds_name}_{shard_tag}.jsonl"
        results_buffer = []
        t0 = time.time()

        for pos, idx in enumerate(tqdm(sample_indices, desc=f"{ds_name}", leave=True)):
            sample = samples[idx]
            prompt = lb.format_prompt(sample, ds_name, tokenizer,
                                      use_chat_template=args.use_chat_template,
                                      enable_thinking=args.enable_thinking)
            ids = tokenizer.encode(prompt, add_special_tokens=True,
                                   return_tensors="pt")
            if isinstance(ids, list):
                ids = torch.tensor([ids], dtype=torch.long)
            input_ids = ids.to(device)
            n_tokens = int(input_ids.shape[1])
            n_chunks = (n_tokens + args.chunk_size - 1) // args.chunk_size

            bare_q = _bare_question(sample)
            bare_q_ids = tokenizer.encode(bare_q, add_special_tokens=False)

            gen_stats: dict = {}
            try:
                pred = qcmem_generate(
                    qc=qc, tokenizer=tokenizer, input_ids=input_ids,
                    chunk_size=args.chunk_size, max_new_tokens=max_gen,
                    selector=args.selector, topk=args.topk,
                    sink_tokens=args.sink_tokens,
                    needle_chunk_set=None, bare_question_ids=bare_q_ids,
                    no_retrieval=no_retrieval, stats=gen_stats,
                )
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                pred = "[OOM]"
                print(f"[OOM] idx={idx} task={ds_name} n_tok={n_tokens}: {e}",
                      flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            results_buffer.append({
                "index": idx,
                "pred": pred,
                "answers": sample["answers"],
                "dataset": ds_name,
                "n_tokens": n_tokens,
                "n_chunks": n_chunks,
                "read_len": gen_stats.get("read_len"),
                "n_selected_chunks": gen_stats.get("n_selected_chunks"),
                "n_context_chunks": gen_stats.get("n_context_chunks"),
            })

            if (pos + 1) % 10 == 0 or pos == len(sample_indices) - 1:
                with open(outfile, "w") as f:
                    for r in results_buffer:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if (pos + 1) % 20 == 0:
                cur_f1s = [lb.compute_f1_multi(r["pred"], r["answers"])
                           for r in results_buffer]
                avg_f1 = sum(cur_f1s) / len(cur_f1s) * 100
                speed = (pos + 1) / (time.time() - t0)
                print(f"  [{ds_name}] {pos+1}/{len(sample_indices)} | "
                      f"{speed:.2f} samples/s | running F1={avg_f1:.1f}% | "
                      f"read_len~{gen_stats.get('read_len')} | "
                      f"last_pred='{pred[:60]}'")

        with open(outfile, "w") as f:
            for r in results_buffer:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f1s = [lb.compute_f1_multi(r["pred"], r["answers"]) for r in results_buffer]
        avg_f1 = (sum(f1s) / len(f1s) * 100) if f1s else 0.0
        elapsed_seconds = time.time() - t0
        task_metrics = {
            "dataset": ds_name,
            "shard_index": int(args.shard_index),
            "num_shards": int(args.num_shards),
            "num_samples": len(results_buffer),
            "f1": avg_f1,
            "elapsed_seconds": elapsed_seconds,
            "output_file": str(outfile),
            "oom_count": sum(r["pred"] == "[OOM]" for r in results_buffer),
            "empty_prediction_count": sum(not r["pred"].strip()
                                          for r in results_buffer),
            "resume_j": int(args.resume_j),
            "chunk_size": int(args.chunk_size),
            "selector": args.selector,
            "topk": int(args.topk),
            "chat_template": bool(args.use_chat_template),
            "lora_adapter": args.lora_adapter or None,
            "bottleneck_ckpt": args.bottleneck_ckpt or None,
            "zero_training_no_adapter": zero_training_no_adapter,
        }
        metrics_file = output_path / f"{ds_name}_{shard_tag}_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(task_metrics, f, indent=2)
        print(f"[QCMem-LongBench] {ds_name}: F1={avg_f1:.2f}% "
              f"({len(results_buffer)} samples, {elapsed_seconds:.1f}s) -> {outfile}")

    print(f"\n[QCMem-LongBench] Shard {args.shard_index}/{args.num_shards} complete!")
    # Single-shard: auto-score (multi-shard: run --score_only after all finish).
    if args.num_shards == 1:
        print("\n[QCMem-LongBench] Running scoring (single-shard mode)...")
        lb.run_scoring(args.output_dir, datasets_list)


if __name__ == "__main__":
    main()
