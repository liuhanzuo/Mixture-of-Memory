#!/usr/bin/env python
"""Paper A — #144 CoMem dense-selector swap (single-variable baseline).

NEW FILE (2026-08-04). A single-variable re-parameterisation of the FLAGSHIP
CoMem reader whose ONLY changed variable is the SELECTOR:

    flagship CoMem : selector = iter_bm25 (lexical, hop=4) -> fills h12
    #144 (this)    : selector = dense_bge (BGE-large-en-v1.5, CLS+L2+cosine) -> fills h12

Everything else is held byte-identical to flagship CoMem:
  * reader   : models/Qwen3-8b-local + flagship LoRA
               (outputs/qcmem_distill_qwen_j12_r32_4k/final), resume_j=12,
               adapter ENABLED, sink=bos, chunk_size=512, topk=12, bf16 + sdpa,
               seed 42, chat_template OFF (BASE LM — no SFT/RL).
  * examples : each (family,task,length) sample is built by P1.9's UNMODIFIED
               family iterators (== the flagship drivers' sample builders), same
               seed/shard convention -> example i here == example i in the
               iter_bm25 CoMem run (1:1 pairing).
  * reader interface : the dense-selected top-k DOCUMENT-ABSOLUTE chunk indices
               are fed as ``needle_chunk_set`` into the UNMODIFIED
               ``eval_qcmem_babilong.qcmem_generate`` with ``selector="oracle"``.
               The oracle branch packs EXACTLY the supplied indices and
               ``qc.write_chunk/write_chunks`` writes them to depth-12 -> the
               dense-selected chunks become CoMem's ``h12``.

Why a NEW file and not a P1.9 flag: ``eval_p1_9_dense_rag.run_cell`` HARD-GUARDS
``resume_j==0`` and empty LoRA (two ``raise SystemExit`` guards) precisely to keep
P1.9 the pure training-free RAG reference. #144 needs the OPPOSITE reader
(resume_j=12 + flagship LoRA), so it reuses P1.9's parts (DenseRetriever, family
iterators + scorers, recall locator, aggregate) UNCHANGED and only re-writes the
~40-line reader construction to the flagship CoMem reader.

This module IMPORTS the shared eval modules; it MODIFIES NONE of them.

Reconciliation (NOT a duplicate of #140/#141):
  * #140 P1.9 = dense -> resume_j=0, LoRA-OFF full-recompute RAG reader (dense
                never touches CoMem's h12).
  * #141 P0.20 PhaseB = swaps only the TEXT-RAG arm's selector to dense; the CoMem
                arm stays iter_bm25 (dense never touches h12).
  * #144 (this) = dense selection feeds the resume_j=12 + LoRA CoMem reader, i.e.
                dense chooses which chunks land in h12. It answers: does CoMem's
                own quality change when its slots are filled by dense retrieval
                instead of BM25?

--------------------------------------------------------------------------------
Usage (one shard = one (family,task,length,shard) cell):
    python scripts/eval_comem_dense_selector.py --mode run \
        --family babilong --task qa5 --length 8k \
        --model_path models/Qwen3-8b-local \
        --retriever_path models/bge-large-en-v1.5 \
        --lora_adapter outputs/qcmem_distill_qwen_j12_r32_4k/final \
        --resume_j 12 --topk 12 --chunk_size 512 --limit 100 \
        --num_shards 4 --shard_index 0 \
        --output_dir bench_results/dense_selector \
        --index_dir retrieval_results/dense_selector

Aggregate (CPU, after all shards of the requested cells finish) — reuses P1.9's
aggregate verbatim (recall/e2e/CI decomposition is selector-agnostic):
    python scripts/eval_comem_dense_selector.py --mode aggregate \
        --output_dir bench_results/dense_selector \
        --require_family babilong:qa5 ruler:niah_single_2,niah_multikey_1,variable_tracking locomo:

The 8-GPU task-pool launcher is scripts/_run_comem_dense_selector_8gpu.sh (DRY by
default; RUN=1 to execute on a free diskB H20 node).
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJECT_ROOT,
           os.path.join(PROJECT_ROOT, "scripts"),
           os.path.join(PROJECT_ROOT, "third_party", "babilong-pkg")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---- reuse P1.9's dense-RAG infrastructure UNCHANGED (imported, never edited) --
import scripts.eval_p1_9_dense_rag as p19            # noqa: E402

DenseRetriever = p19.DenseRetriever
_FAMILY_ITER = p19._FAMILY_ITER
_FAMILY_SCORE = p19._FAMILY_SCORE
_gold_chunk_set = p19._gold_chunk_set
_sha256_str = p19._sha256_str
aggregate = p19.aggregate                            # selector-agnostic; reused verbatim
qcmem_generate = p19.qcmem_generate
QCMemModel = p19.QCMemModel
qcb = p19.qcb
EXPECTED_BGE_SHA256 = p19.EXPECTED_BGE_SHA256
EXPECTED_BGE_REVISION = p19.EXPECTED_BGE_REVISION
BGE_QUERY_INSTRUCTION = p19.BGE_QUERY_INSTRUCTION

# ---- reuse the flagship CoMem LoRA loader + sha gate UNCHANGED -----------------
# _load_with_peft mirrors p013._load's ordering (base -> PeftModel -> inner) so the
# reader is numerically identical to the flagship CoMem arm; EXPECTED_LORA_SHA is
# the flagship adapter's fail-closed sha (p013).
import scripts.eval_p0_20_equal_latency as p020      # noqa: E402

_load_with_peft = p020._load_with_peft
EXPECTED_LORA_SHA = p020.EXPECTED_LORA_SHA

FLAGSHIP_LORA = "outputs/qcmem_distill_qwen_j12_r32_4k/final"


# --------------------------------------------------------------------------- #
# run mode — one (family,task,length,shard) cell of the CoMem dense-selector arm.
# --------------------------------------------------------------------------- #
def run_cell(args):
    import torch

    # ---- fail-closed the OPPOSITE way vs P1.9: this IS the CoMem reader -------
    if args.resume_j != 12:
        raise SystemExit(
            f"[#144][ABORT] #144 reader is the flagship CoMem mid-depth resume "
            f"reader; --resume_j must be 12 (got {args.resume_j}).")
    if not args.lora_adapter:
        raise SystemExit(
            f"[#144][ABORT] #144 reader is the flagship CoMem reader; "
            f"--lora_adapter must be the flagship LoRA (got empty).")

    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    model_path = args.model_path
    if not os.path.isabs(model_path):
        model_path = os.path.join(PROJECT_ROOT, model_path)
    retriever_path = args.retriever_path
    if not os.path.isabs(retriever_path):
        retriever_path = os.path.join(PROJECT_ROOT, retriever_path)
    lora_adapter = args.lora_adapter
    if not os.path.isabs(lora_adapter):
        lora_adapter = os.path.join(PROJECT_ROOT, lora_adapter)

    use_chat = (args.reader_prompt == "native")

    print(f"[#144] family={args.family} task={args.task} length={args.length} "
          f"shard={args.shard_index}/{args.num_shards} reader_prompt={args.reader_prompt}")
    print(f"[#144] reader={model_path} + LoRA (resume_j=12, adapter ENABLED, "
          f"sink={args.sink_tokens}, chunk={args.chunk_size}, topk={args.topk}, "
          f"{args.dtype}/{args.attn_impl})")

    # ---- reader: flagship CoMem = LoRA-applied Qwen3-8B, resume_j=12 ----------
    # _load_with_peft returns the LoRA-APPLIED inner model with the adapter
    # ENABLED by default (PeftModel enables adapters on load) — exactly the
    # flagship CoMem arm of eval_p0_20_equal_latency (:325-332).
    tokenizer, model, peft_model, lora_sha256, lora_layers = _load_with_peft(
        model_path, dtype, args.attn_impl, device, lora_adapter)
    if lora_sha256 != EXPECTED_LORA_SHA:
        raise SystemExit(
            f"[#144][ABORT] LoRA sha {lora_sha256} != expected {EXPECTED_LORA_SHA}")
    qc = QCMemModel(model, resume_j=12)   # CoMem reader, LoRA enabled
    print(f"[#144] LoRA sha OK ({lora_sha256[:12]}…) layers_to_transform={lora_layers}")

    gen_boundary_ids = None
    enable_thinking = args.enable_thinking
    if use_chat:
        gen_boundary_ids = qcb._chat_generation_boundary_ids(
            tokenizer, enable_thinking)

    # ---- frozen dense retriever (identical to P1.9) --------------------------
    retriever = DenseRetriever(
        retriever_path, device, dtype,
        allow_sha_mismatch=args.allow_retriever_sha_mismatch)
    print(f"[#144] retriever=BGE-large-en-v1.5 sha_ok={retriever.sha_ok} "
          f"sha256={retriever.weight_sha256} pooling={retriever.pooling} "
          f"hidden={retriever.hidden} max_len={retriever.max_len}")

    provenance = {
        "retriever_model": "BAAI/bge-large-en-v1.5",
        "retriever_path": retriever_path,
        "retriever_revision": EXPECTED_BGE_REVISION,
        "retriever_weight_sha256": retriever.weight_sha256,
        "retriever_weight_sha256_expected": EXPECTED_BGE_SHA256,
        "retriever_sha_ok": retriever.sha_ok,
        "pooling": "cls",
        "normalization": "l2",
        "distance_metric": "cosine (dot of L2-normalized CLS)",
        "query_instruction": BGE_QUERY_INSTRUCTION,
        "passage_instruction": "",
        "retriever_max_tokens": retriever.max_len,
        "retriever_dtype": args.dtype,
        "hidden_dim": retriever.hidden,
        "index_type": "flat brute-force cosine (exact, per-query rebuild)",
        "hardware": {
            "node": socket.gethostname(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device": args.device,
            "gpu_name": (torch.cuda.get_device_name(0)
                         if torch.cuda.is_available() else None),
        },
    }
    reader_cfg = {
        "reader_model_path": model_path,
        "reader_lora_adapter": lora_adapter,
        "lora_sha256": lora_sha256,
        "expected_lora_sha256": EXPECTED_LORA_SHA,
        "lora_sha_match": lora_sha256 == EXPECTED_LORA_SHA,
        "lora_layers_to_transform": lora_layers,
        "resume_j": 12,
        "adapter_enabled": True,
        "sink_tokens": args.sink_tokens,
        "chunk_size": args.chunk_size,
        "topk": args.topk,
        "selector": "dense_bge (via oracle needle_chunk_set injection into CoMem h12)",
        "reader_prompt": args.reader_prompt,
        "use_chat_template": use_chat,
        "enable_thinking": enable_thinking,
        "add_special_tokens": True,
        "max_new_tokens": args.max_new_tokens,
        "dtype": args.dtype,
        "attn_impl": args.attn_impl,
        "seed": args.seed,
        "num_layers": int(model.config.num_hidden_layers),
        "single_variable_vs_flagship_comem": "selector iter_bm25 -> dense_bge",
    }

    outdir = Path(args.output_dir) / f"{args.family}"
    outdir.mkdir(parents=True, exist_ok=True)
    sharded = args.num_shards > 1
    shard_tag = f"_shard{args.shard_index}of{args.num_shards}" if sharded else ""
    cell = f"{args.task or args.family}_{args.length}"
    outfile = outdir / f"{cell}_{args.reader_prompt}{shard_tag}.jsonl"
    cfgfile = outdir / f"{cell}_{args.reader_prompt}{shard_tag}.config.json"

    index_dir = Path(args.index_dir) / f"{args.family}"
    index_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    it = _FAMILY_ITER[args.family](
        args, tokenizer, device, use_chat, enable_thinking, gen_boundary_ids)
    scorer = _FAMILY_SCORE[args.family]

    records = []
    index_sizes = []
    t_start = time.time()
    for pos, s in enumerate(it):
        # dense chunking IDENTICAL to qcmem_generate: tokens.split(chunk_size),
        # context = chunks[:-1], query chunk = chunks[-1].
        tokens = s.input_ids[0]
        chunks = list(tokens.split(args.chunk_size))
        context_chunks = chunks[:-1]
        n_ctx = len(context_chunks)
        # decode each context chunk to text for the (text-space) dense retriever.
        ctx_texts = [tokenizer.decode(c.tolist(), skip_special_tokens=True)
                     for c in context_chunks]
        sel_idx, scores, lat_ms, index_bytes = retriever.select_topk(
            ctx_texts, s.query_text, args.topk)
        index_sizes.append(index_bytes)

        # recall: gold support chunk in the dense top-k pack? (answer-independent)
        gold_set = _gold_chunk_set(s.input_ids, s.gold_probes, tokenizer,
                                   args.chunk_size)
        if gold_set is None:
            recall_hit = None                    # unlocatable -> excluded
        else:
            gold_in_ctx = {c for c in gold_set if 0 <= c < n_ctx}
            recall_hit = int(bool(gold_in_ctx & set(sel_idx))) \
                if gold_in_ctx else None

        # reader: feed dense-selected indices as oracle needle_chunk_set -> the
        # oracle branch packs EXACTLY those chunks and write_chunks writes them to
        # CoMem's depth-12 h12 (resume_j=12, LoRA enabled).
        bare_q_ids = tokenizer.encode(s.query_text or "", add_special_tokens=False)
        gen_stats = {}
        try:
            output = qcmem_generate(
                qc=qc, tokenizer=tokenizer, input_ids=s.input_ids,
                chunk_size=args.chunk_size, max_new_tokens=args.max_new_tokens,
                selector="oracle", topk=args.topk,
                sink_tokens=args.sink_tokens,
                needle_chunk_set=set(sel_idx), bare_question_ids=bare_q_ids,
                no_retrieval=False, stats=gen_stats,
                gen_boundary_ids=gen_boundary_ids)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            output = "[OOM]"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        sc = scorer(output, s.score_ctx)
        rec = {
            "id": s.sid,
            "family": args.family, "task": args.task, "length": args.length,
            "query_text": s.query_text,
            "n_context_chunks": n_ctx,
            "dense_sel_idx": sel_idx,
            "gold_chunk_set": (sorted(gold_set) if gold_set else None),
            "recall_hit": recall_hit,
            "read_len": gen_stats.get("read_len"),
            "n_selected_chunks": gen_stats.get("n_selected_chunks"),
            "retrieval_latency_ms": lat_ms,
            "index_bytes": index_bytes,
            "output": output,
            "input_ids_sha256": _sha256_str(",".join(map(str, tokens.tolist()))),
            "pack_sel_sha256": _sha256_str(",".join(map(str, sel_idx))),
        }
        rec.update(sc)
        records.append(rec)

        with open(outfile, "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(outfile, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # per-cell index manifest (sizes; embeddings rebuilt per query, size reported).
    mean_bytes = (sum(index_sizes) / len(index_sizes)) if index_sizes else 0
    with open(index_dir / f"{cell}_{args.reader_prompt}{shard_tag}.index.json",
              "w") as f:
        json.dump({
            "cell": cell, "shard": shard_tag or "single",
            "n_samples": len(records),
            "embed_dim": retriever.hidden,
            "dtype": args.dtype, "dtype_bytes": retriever.dtype_bytes,
            "metric": "cosine", "index_type": provenance["index_type"],
            "index_bytes_mean": round(mean_bytes, 1),
            "index_bytes_min": min(index_sizes) if index_sizes else 0,
            "index_bytes_max": max(index_sizes) if index_sizes else 0,
        }, f, indent=2)

    cfg = {
        "status": "completed", "family": args.family, "task": args.task,
        "length": args.length, "n": len(records),
        "n_requested": args.limit,
        "arm": "comem_dense_selector (#144)",
        "sharding": {"num_shards": args.num_shards,
                     "shard_index": args.shard_index},
        "reader": reader_cfg,
        "retriever_provenance": provenance,
        "recall_definition": ("gold support-span chunk (family oracle locator) in "
                              "dense top-k pack; decided INDEPENDENTLY of answer; "
                              "unlocatable gold -> excluded from recall denom"),
        "pairing": ("same seed/shard/chunk_size as flagship iter_bm25 CoMem; "
                    "input_ids_sha256 + pack_sel_sha256 recorded per example"),
        "elapsed_seconds": round(time.time() - t_start, 2),
    }
    with open(cfgfile, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[#144] wrote {len(records)} records -> {outfile}")
    print(f"[#144] cell config -> {cfgfile}")


# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(
        description="Paper A #144 CoMem dense-selector swap (single-variable)")
    p.add_argument("--mode", choices=["run", "aggregate", "provenance"],
                   default="run")
    p.add_argument("--family", choices=list(_FAMILY_ITER),
                   help="benchmark family (run mode)")
    p.add_argument("--task", type=str, default=None,
                   help="babilong qa5 ; ruler niah_single_2/niah_multikey_1/"
                        "variable_tracking ; unused for longeval/locomo")
    p.add_argument("--length", type=str, default="8k")
    p.add_argument("--model_path", type=str, default="models/Qwen3-8b-local")
    p.add_argument("--retriever_path", type=str,
                   default="models/bge-large-en-v1.5")
    p.add_argument("--lora_adapter", type=str, default=FLAGSHIP_LORA,
                   help="flagship CoMem LoRA (MUST be non-empty; sha-gated).")
    p.add_argument("--resume_j", type=int, default=12,
                   help="MUST be 12 — #144 reader is the flagship CoMem "
                        "mid-depth resume reader.")
    p.add_argument("--topk", type=int, default=12)
    p.add_argument("--chunk_size", type=int, default=512)
    p.add_argument("--sink_tokens", type=str, default="bos",
                   choices=["bos", "none"])
    p.add_argument("--max_new_tokens", type=int, default=48)
    p.add_argument("--reader_prompt", choices=["plain", "native"], default="plain",
                   help="plain = unified no-chat main protocol (chat_template off, "
                        "the config#2口径). native = chat-template variant.")
    p.add_argument("--enable_thinking", action="store_true", default=False)
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn_impl", type=str, default="sdpa")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--dataset_name", type=str, default="RMT-team/babilong")
    p.add_argument("--locomo_data", type=str, default="locomo/data/locomo10.json")
    p.add_argument("--locomo_categories", type=str, default=None)
    p.add_argument("--output_dir", type=str,
                   default="bench_results/dense_selector")
    p.add_argument("--index_dir", type=str,
                   default="retrieval_results/dense_selector")
    p.add_argument("--allow_retriever_sha_mismatch", action="store_true",
                   default=False,
                   help="bypass the BGE weight sha fail-closed gate (audit only).")
    p.add_argument("--require_family", nargs="+", default=None,
                   help="aggregate all-tasks guard: 'family:task1,task2' specs.")
    return p


def main():
    args = build_parser().parse_args()
    if args.mode == "run":
        if not args.family:
            build_parser().error("--family is required in run mode")
        if args.num_shards < 1:
            build_parser().error("--num_shards must be >= 1")
        if not (0 <= args.shard_index < args.num_shards):
            build_parser().error("--shard_index out of range")
        run_cell(args)
    elif args.mode == "aggregate":
        aggregate(args)      # P1.9's aggregate verbatim (selector-agnostic)
    elif args.mode == "provenance":
        # delegate to P1.9's BGE weight-sha provenance check (identical retriever).
        weight = os.path.join(
            args.retriever_path if os.path.isabs(args.retriever_path)
            else os.path.join(PROJECT_ROOT, args.retriever_path),
            "model.safetensors")
        sha = p19._sha256_file(weight) if os.path.exists(weight) else None
        ok = (sha == EXPECTED_BGE_SHA256)
        print(json.dumps({
            "retriever_path": args.retriever_path,
            "weight_sha256": sha,
            "expected_sha256": EXPECTED_BGE_SHA256,
            "revision": EXPECTED_BGE_REVISION,
            "sha_ok": ok,
        }, indent=2))
        sys.exit(0 if ok else 6)


if __name__ == "__main__":
    main()
