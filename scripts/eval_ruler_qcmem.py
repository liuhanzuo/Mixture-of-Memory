#!/usr/bin/env python
"""QCMem mid-depth resume — RULER long-context eval driver.

Validates the QCMem write/read resume primitive
(``src/memory/qcmem/qcmem_model.py``) on the *real long-document* RULER tasks
(NIAH needle-in-a-haystack + variable_tracking), the generalisation companion to
``scripts/eval_qcmem_babilong.py`` (which runs the same primitive on BABILong).

Design: this is a thin composition of two existing, unmodified drivers —

  RULER task framework  (imported from ``scripts/eval_ruler_mem_space.py``):
    * ``_build_sample``          — RULER-faithful (context, answers, gold_needle)
                                   generation, sized to ~target tokens (NIAH
                                   single/multikey on noise|PG19-prose haystack,
                                   variable_tracking on noise haystack).
    * ``_make_vt_icl``           — the fixed VT in-context worked example.
    * ``_string_match_all_one``  — RULER ``string_match_all`` recall scoring
                                   (fraction of reference strings that appear as
                                   a case-insensitive substring of the output).
    * ``_LENGTH_TOKENS``         — length-bucket -> target token budget.

  QCMem forward path (imported from ``scripts/eval_qcmem_babilong.py``):
    * ``qcmem_generate``         — chunk the prompt -> write_chunk each chunk to
                                   depth j -> selector picks topk context chunks
                                   -> read (pack [sink; selected h_j; query h_j],
                                   resume layers[j:]) -> greedy decode.
    * ``run_self_test``          — j=0 correctness gate (QCMem read == full
                                   forward, fp32 max|logit diff| < 1e-4).
    * ``QCMemModel``             — the write/read orchestrator (read-only import).
    * ``harness``                — ``_locate_needle_chunks`` (oracle) +
                                   ``_write_results_csv`` (QUOTE_ALL CSV writer).

The RULER sample RNG + shard filter are replicated bit-for-bit from
``eval_ruler_mem_space.main`` (build sample ``i`` with a deterministic
per-``(task,length,i)`` seed, then keep only ``i % num_shards == shard_index``),
so shards share one sample set and the needle format / scoring口径 are identical
to the mem_space RULER numbers — only the model forward differs (QCMem instead of
mem_space streaming / base full-attention).

Selectors: bm25 / recency / reader_attn behave exactly as in the BABILong driver.
``oracle`` is supported for the NIAH tasks (the queried needle sentence is known
at generation time, so we locate its document-absolute chunk index with
``harness._locate_needle_chunks``); it is NOT supported for variable_tracking
(no single gold-needle span — the answer is a set of variable names scattered
across the chain), where it transparently degrades to recency.

Usage (Direction-A QCMem-distill LoRA on Qwen3-8B):
    python scripts/eval_ruler_qcmem.py \
        --model_path /apdcephfs_wzc1/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b \
        --resume_j 12 --lora_adapter outputs/qcmem_distill_qwen_j12_r32_4k/final \
        --selector bm25 --topk 12 --sink_tokens bos \
        --ruler_tasks niah_single niah_multi vt --lengths 4k 8k 16k 32k \
        --limit 50 --output_name ruler_qcmem_j12 --results_folder ruler_results/qcmem_j12

Usage (funnel-Qwen continued-pretrain arm — pretrain-vs-vanilla head-to-head):
    python scripts/eval_ruler_qcmem.py \
        --model_path /apdcephfs_wzc1/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b \
        --bottleneck_ckpt outputs/qwenbott_funnel_L12_d512/final.pt --resume_j 13 \
        --selector bm25 --topk 12 --sink_tokens bos \
        --ruler_tasks niah_single niah_multi --lengths 8k 16k 32k \
        --limit 50 --output_name ruler_qcmem_funnelL12 --results_folder ruler_results/qcmem_funnelL12
"""
from __future__ import annotations

import argparse
import json
import os
import random
import socket
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJECT_ROOT,
           os.path.join(PROJECT_ROOT, "scripts"),
           os.path.join(PROJECT_ROOT, "third_party", "babilong-pkg")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

# RULER task framework (generation + scoring) — reused verbatim, unmodified.
import scripts.eval_ruler_mem_space as ruler  # noqa: E402

# QCMem forward path — reused verbatim, unmodified. Importing this module runs
# its explicit-file-path load of the babilong harness ("_qcmem_harness"); we
# reach that harness (needle locator + CSV writer) through ``qcb.harness``.
import scripts.eval_qcmem_babilong as qcb  # noqa: E402

QCMemModel = qcb.QCMemModel
qcmem_generate = qcb.qcmem_generate
run_self_test = qcb.run_self_test


# --------------------------------------------------------------------------- #
# RULER task-name aliases -> canonical eval_ruler_mem_space task ids.
# --------------------------------------------------------------------------- #
# The canonical RULER task ids are the long form (niah_single_1 = noise haystack,
# niah_single_2 = PG19-prose haystack, niah_multikey_1 = prose + 4 distractor
# keys, variable_tracking = 4-hop chain). We accept both the long form AND short
# friendly aliases so "niah_single / niah_multi / vt" work on the CLI.
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
        f"unknown ruler task {name!r}; expected one of "
        f"{sorted(_CANONICAL_TASKS)} or aliases {sorted(_TASK_ALIAS)}"
    )


def _resolve_selector(selector: str, task: str) -> str:
    """Map the CLI --selector to the concrete per-task selector.

    Default sentinel 'auto' = the data-validated per-task routing (8B RULER
    n=500): variable_tracking uses FIXED ``iter_bm25`` (multi-hop BFS on the
    literal VAR chain; adaptive's confidence early-stop killed the chain →
    31/25/22 vs iter_bm25's 97/97/98 on 8k/16k/32k), while all niah_* single-
    shot tasks use plain ``bm25`` (adaptive 91/70/32 vs bm25 91/91/92 on
    niah_multikey). Any explicit --selector X overrides and applies X to ALL
    tasks (for controls)."""
    if selector != "auto":
        return selector
    if task == "variable_tracking":
        return "iter_bm25"
    return "bm25"


def _bare_question(prompt: str) -> str:
    """Extract the trailing question line (used as the bm25 lexical query).

    RULER's NIAH / VT templates always place a ``\\n`` immediately before the
    question ("...\\n{context}\\nWhat is the special magic number for {key}...?"
    / "...\\n{context}\\nQuestion: Find all variables assigned the value
    {value}..."). Because the question follows the whole context, the LAST
    newline in the rendered prompt is exactly that boundary, so everything after
    it is the question + answer-prefix — which contains the queried key (NIAH) /
    value (VT), the terms bm25 should retrieve on. Robust to the noise haystack's
    own many internal newlines (they all precede the question)."""
    return prompt[prompt.rfind("\n") + 1:].strip()


def _oracle_needle_chunks(input_ids, gold_needle, answers, tokenizer, chunk_size):
    """Document-absolute chunk index set containing the queried needle, for the
    NIAH oracle selector. Try the full gold-needle sentence first (most
    distinctive), then fall back to each answer value. Returns None if nothing
    can be located (caller's selector then degrades to recency)."""
    for probe in ([gold_needle] if gold_needle else []) + list(answers or []):
        if not probe:
            continue
        chunks = qcb.harness._locate_needle_chunks(
            input_ids, probe, tokenizer, chunk_size)
        if chunks:
            return chunks
    return None


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="QCMem mid-depth resume — RULER (NIAH/VT) eval driver"
    )
    # --- CLI aligned with scripts/eval_qcmem_babilong.py ---
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to plain backbone weights (Qwen3-8B / Llama-3-8B).")
    parser.add_argument("--resume_j", type=int, default=12,
                        help="Layer split index j (0=RAG upper bound, L=closed-book).")
    parser.add_argument("--top_prepay_b", type=int, default=0,
                        help="Direction-B top-prepay: run the top b layers "
                             "query-local at read (0=exact connective resume).")
    parser.add_argument("--reuse_kv_blockdiag", action="store_true", default=False,
                        help="Ablation arm (ii): resume layers[j:] over the SAME "
                             "pack/positions/depth but with a BLOCK-DIAGONAL "
                             "attention mask (sink global; each context chunk "
                             "within-block only, query-blind; query reads sink+all "
                             "chunks+itself). Isolates the value of cross-chunk + "
                             "query attention vs. per-chunk query-blind KV reuse. "
                             "Only valid with --top_prepay_b 0.")
    parser.add_argument("--lora_adapter", type=str, default="",
                        help="Optional path to a trained QCMem-distill LoRA "
                             "adapter dir (Direction A).")
    parser.add_argument("--bottleneck_ckpt", type=str, default="",
                        help="Optional path to a continued-pretrain funnel-Qwen "
                             "checkpoint (a *.pt with a full 'model_state' + an "
                             "arch_meta.json next to it, produced by "
                             "scripts/train_qwen_bottleneck_continued.py). When "
                             "set, the backbone is rebuilt as 'stock Qwen3-8B "
                             "(--model_path) + a BottleneckLayer funnel injected "
                             "on the OUTPUT of decoder layer bottleneck_layer', "
                             "then load_state_dict(model_state) — this is the "
                             "'funnel-Qwen + QCMem' arm of the pretrain-vs-vanilla "
                             "head-to-head. RECOMMENDED --resume_j == "
                             "bottleneck_layer+1 (=13 for the default L12 ckpt) so "
                             "QCMem caches the COMPRESSED funnel output h_j' "
                             "(write runs layers[0:resume_j], which then includes "
                             "the funnel layer). Mutually exclusive with "
                             "--lora_adapter (that is the stock-Qwen LoRA arm).")
    parser.add_argument("--baseline", type=str, default="none",
                        choices=["none", "kvdirect", "hcache"],
                        help="Mechanism-level head-to-head baseline (2026-07-08). "
                             "'none' = normal QCMem (retrieval topk + resume_j). "
                             "'kvdirect' (2603.19664 'The Residual Stream Is All "
                             "You Need') = FULL-DEPTH recompute (forces resume_j=0) "
                             "+ NO retrieval (packs every context chunk) + no LoRA "
                             "(training-free) — read grows O(context). 'hcache' "
                             "(2410.05004) = mid-layer recompute (keeps --resume_j) "
                             "+ NO retrieval (packs every context chunk) + no LoRA "
                             "(post-hoc, no training) — read grows O(context). Both "
                             "isolate QCMem's two primitives: retrieval (fixed read) "
                             "and layer-partial recompute.")
    parser.add_argument("--selector", type=str, default="auto",
                        choices=["auto", "bm25", "recency", "oracle", "reader_attn",
                                 "iter_reader_attn", "iter_bm25",
                                 "iter_bm25_adaptive"],
                        help="Chunk selector for the read pack. Default 'auto' is "
                             "the DATA-VALIDATED per-task routing (8B RULER n=500): "
                             "variable_tracking -> FIXED 'iter_bm25' (multi-hop BFS "
                             "on the literal VAR chain, iter_hop_topk=4), all niah_* "
                             "-> 'bm25' (single-shot lexical). This replaces the old "
                             "single universal 'iter_bm25_adaptive' default, whose "
                             "confidence early-stop killed the VT chain (31/25/22 vs "
                             "iter_bm25 97/97/98 on 8k/16k/32k) and hurt "
                             "niah_multikey (91/70/32 vs bm25 91/91/92). Pass an "
                             "explicit --selector (bm25 / iter_bm25 / "
                             "iter_bm25_adaptive / reader_attn / oracle / ...) to "
                             "override it on ALL tasks for controls. "
                             "oracle is NIAH-only "
                             "(degrades to recency on variable_tracking). "
                             "iter_reader_attn iterates reader_attn as a multi-hop "
                             "BFS over the cached h_j (query -> found-chunk -> ...) "
                             "to follow the vt reference chain; forward-free. "
                             "iter_bm25 is the same multi-hop BFS but with pure "
                             "lexical BM25 as the hop signal (round 1 == single-shot "
                             "bm25, later rounds re-query with the previous picks' "
                             "token text) — best for vt where the chain links are "
                             "LITERAL VAR names; pure CPU, forward-free. "
                             "iter_bm25_adaptive is iter_bm25 with a confidence-based "
                             "adaptive stop (no fixed topk budget): stop when a hop's "
                             "best score drops below --iter_conf_ratio x the round-1 "
                             "best or --iter_max_chunks is hit, so short chains don't "
                             "hard-fill low-score noise chunks into the read.")
    parser.add_argument("--iter_rounds", type=int, default=0,
                        help="iter_reader_attn / iter_bm25: #BFS hop rounds (<=0 -> "
                             "ceil(topk/iter_hop_topk)).")
    parser.add_argument("--iter_hop_topk", type=int, default=4,
                        help="iter_reader_attn / iter_bm25 / iter_bm25_adaptive: "
                             "chunks added per BFS round.")
    parser.add_argument("--iter_score", type=str, default="meanpool",
                        choices=["meanpool", "maxsim"],
                        help="iter_reader_attn scoring: meanpool (mean-pool cosine, "
                             "== reader_attn) or maxsim (token-level late "
                             "interaction, dilution-free). Both forward-free.")
    parser.add_argument("--iter_conf_ratio", type=float, default=0.3,
                        help="iter_bm25_adaptive: stop a hop when its best BM25 score "
                             "falls below this ratio x the round-1 best score.")
    parser.add_argument("--iter_max_chunks", type=int, default=64,
                        help="iter_bm25_adaptive: hard cap on accumulated chunks.")
    parser.add_argument("--topk", type=int, default=12,
                        help="Number of context chunks to pack into the read.")
    parser.add_argument("--sink_tokens", type=str, default="bos",
                        choices=["bos", "none"],
                        help="Attention-sink anchor at packed position 0.")
    parser.add_argument("--results_folder", type=str, default="./ruler_results")
    parser.add_argument("--output_name", type=str, required=True)
    parser.add_argument("--chunk_size", type=int, default=512,
                        help="QCMem chunk size (matches eval_qcmem_babilong; the "
                             "prompt is split into chunk_size-token segments).")
    parser.add_argument("--max_new_tokens", type=int, default=48,
                        help="Greedy decode budget (RULER uses 48; VT is bumped "
                             "to >=60 to fit the variable list, as in "
                             "eval_ruler_mem_space).")
    parser.add_argument("--limit", type=int, default=50,
                        help="Samples per (task,length) cell (RULER's num_samples).")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn_impl", type=str, default="sdpa")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base RNG seed (matches eval_ruler_mem_space so the "
                             "sample set is comparable to the mem_space run).")
    # --- RULER-specific ---
    parser.add_argument("--ruler_tasks", type=str, nargs="+",
                        default=["niah_single", "niah_multi", "vt"],
                        help="RULER tasks / aliases: niah_single(_1/_2), "
                             "niah_multi(key_1), vt(variable_tracking).")
    parser.add_argument("--lengths", type=str, nargs="+",
                        default=["4k", "8k", "16k", "32k"])
    parser.add_argument("--self_test", action="store_true", default=False,
                        help="Run the shared QCMem j=0 correctness gate (read == "
                             "full forward, fp32 < 1e-4) and exit.")
    parser.add_argument("--use_chat_template", action="store_true", default=False,
                        help="Wrap the model INPUT in the tokenizer chat template "
                             "(mirrors eval_qcmem_babilong). bare_q / bare_q_ids "
                             "stay derived from the raw prompt.")
    parser.add_argument("--enable_thinking", action="store_true", default=False,
                        help="When --use_chat_template is set, keep Qwen3 thinking "
                             "ON. Default OFF -> enable_thinking=False.")
    args = parser.parse_args()

    if args.num_shards < 1:
        parser.error("--num_shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        parser.error(f"--shard_index must be in [0, {args.num_shards})")

    # --- head-to-head baseline resolution (mechanism-level, 2026-07-08) --------
    # A baseline is expressed as a re-parameterisation of the QCMem primitives, so
    # the ONLY thing that differs vs. QCMem is the specific primitive under test
    # (same backbone / same RULER sample set / same scoring).
    #   kvdirect (2603.19664) : full-depth recompute -> force resume_j=0;
    #                           no retrieval (pack all chunks); training-free (drop
    #                           any --lora_adapter).
    #   hcache   (2410.05004) : mid-layer recompute -> keep --resume_j as given;
    #                           no retrieval (pack all chunks); post-hoc (drop LoRA).
    # QCMem ('none') keeps retrieval (selector+topk), the given resume_j, and LoRA.
    no_retrieval = (args.baseline != "none")
    if args.bottleneck_ckpt and args.lora_adapter:
        parser.error("--bottleneck_ckpt (funnel-Qwen arm) and --lora_adapter "
                     "(stock-Qwen LoRA arm) are mutually exclusive; pick one.")
    if args.baseline == "kvdirect":
        if args.resume_j != 0:
            print(f"[QCMem-RULER] baseline=kvdirect -> forcing resume_j "
                  f"{args.resume_j} -> 0 (full-depth K/V recompute).")
        args.resume_j = 0
        if args.lora_adapter:
            print("[QCMem-RULER] baseline=kvdirect is training-free -> ignoring "
                  f"--lora_adapter {args.lora_adapter!r}.")
            args.lora_adapter = ""
    elif args.baseline == "hcache":
        if args.lora_adapter:
            print("[QCMem-RULER] baseline=hcache is post-hoc (no training) -> "
                  f"ignoring --lora_adapter {args.lora_adapter!r}.")
            args.lora_adapter = ""
    if no_retrieval and args.reuse_kv_blockdiag:
        parser.error("--reuse_kv_blockdiag is a QCMem ablation and is incompatible "
                     "with --baseline (kvdirect/hcache pack all chunks with the "
                     "standard causal read).")

    tasks = [_resolve_task(t) for t in args.ruler_tasks]

    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16,
             "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    if args.self_test:
        dtype = torch.float32  # tight <1e-4 gate needs fp32 (bf16 ~1e-2 roundoff)

    print(f"[QCMem-RULER] model_path={args.model_path}")
    print(f"[QCMem-RULER] baseline={args.baseline} "
          f"(no_retrieval={no_retrieval}) "
          f"resume_j={args.resume_j} top_prepay_b={args.top_prepay_b} "
          f"reuse_kv_blockdiag={args.reuse_kv_blockdiag} "
          f"selector={args.selector} topk={args.topk} sink={args.sink_tokens} "
          f"chunk_size={args.chunk_size} dtype={dtype} attn_impl={args.attn_impl}")
    print(f"[QCMem-RULER] tasks={tasks} lengths={args.lengths} limit={args.limit}")

    # local_files_only=True: offline nodes otherwise treat a local dir path as an
    # HF repo_id and error ("Repo id must be in the form ...").
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
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
    # PeftModel.base_model.model is the underlying CausalLM QCMemModel reads off.
    if args.lora_adapter:
        from peft import PeftModel
        print(f"[QCMem-RULER] loading LoRA adapter: {args.lora_adapter}")
        peft_model = PeftModel.from_pretrained(model, args.lora_adapter).eval()
        model = peft_model.base_model.model

    # Funnel-Qwen arm: rebuild "stock Qwen + mid-layer BottleneckLayer funnel"
    # exactly as continued-pretrain saved it, then load the full state_dict. We
    # REUSE ``inject_bottleneck`` from the train script so the wrapper structure
    # (layers[j] -> BottleneckLayer(.inner/.down/.up)) is guaranteed identical to
    # the one that produced ``model_state`` (which keys off .inner/.down/.up).
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
        print(f"[QCMem-RULER] funnel-Qwen: arch_meta {meta_path} -> "
              f"bottleneck_layer={b_layer} bottleneck_dim={b_dim} "
              f"num_hidden_layers={meta.get('num_hidden_layers')}")
        if meta.get("model_path") and os.path.abspath(meta["model_path"]) != \
                os.path.abspath(args.model_path):
            print(f"[QCMem-RULER][WARN] --model_path {args.model_path!r} != "
                  f"arch_meta model_path {meta['model_path']!r}; injecting funnel "
                  f"onto the --model_path backbone regardless.")
        # Inject the funnel onto the OUTPUT of decoder layer b_layer, in-place.
        inject_bottleneck(model, b_layer, b_dim, dtype)
        # Load the continued-pretrain weights (funnel + co-adapted upper stack).
        ck = torch.load(args.bottleneck_ckpt, map_location="cpu")
        state = ck.get("model_state", ck)
        missing, unexpected = model.load_state_dict(state, strict=False)
        # Every param we saved should map back; only buffers (rotary inv_freq etc.)
        # may legitimately differ. Fail loudly on any missing/unexpected WEIGHT.
        bad_missing = [k for k in missing if "inv_freq" not in k]
        if bad_missing or unexpected:
            print(f"[QCMem-RULER][WARN] load_state_dict missing={bad_missing[:8]}"
                  f"{'...' if len(bad_missing) > 8 else ''} "
                  f"unexpected={unexpected[:8]}"
                  f"{'...' if len(unexpected) > 8 else ''}")
        model = model.to(device).eval()
        step = ck.get("step")
        print(f"[QCMem-RULER] funnel-Qwen loaded from {args.bottleneck_ckpt} "
              f"(step={step}). RECOMMENDED --resume_j == bottleneck_layer+1 "
              f"(={b_layer + 1}); you passed --resume_j {args.resume_j}.")
        if args.resume_j != b_layer + 1:
            print(f"[QCMem-RULER][NOTE] --resume_j {args.resume_j} != "
                  f"bottleneck_layer+1 ({b_layer + 1}): QCMem will cache h_j at a "
                  f"depth that does NOT coincide with the funnel output — usually "
                  f"you want resume_j={b_layer + 1} so the cached h_j' is the "
                  f"compressed representation.")

    if args.self_test:
        ok = run_self_test(model, tokenizer, device, args.chunk_size)
        sys.exit(0 if ok else 1)

    qc = QCMemModel(model, resume_j=args.resume_j, top_prepay_b=args.top_prepay_b,
                    block_diagonal=args.reuse_kv_blockdiag)

    # Chat assistant generation prefix (no-think when enable_thinking is False),
    # appended at the QCMem query boundary so decoding resumes after the closed
    # <think></think> block regardless of chunk_size. Computed once
    # (content-independent). None when --use_chat_template is off -> byte-identical.
    gen_boundary_ids = None
    if args.use_chat_template:
        gen_boundary_ids = qcb._chat_generation_boundary_ids(
            tokenizer, args.enable_thinking)
        print(f"[QCMem-RULER] chat generation boundary ids ({len(gen_boundary_ids or [])} tok): "
              f"{tokenizer.decode(gen_boundary_ids) if gen_boundary_ids else None!r}")

    outdir = Path(args.results_folder) / args.output_name
    outdir.mkdir(parents=True, exist_ok=True)
    sharded = args.num_shards > 1
    shard_tag = f"_shard{args.shard_index}of{args.num_shards}" if sharded else ""

    summary: dict = {}
    for task in tqdm(tasks, desc="tasks"):
        summary[task] = {}
        sel = _resolve_selector(args.selector, task)
        for length in tqdm(args.lengths, desc="lengths", leave=False):
            cell_started = time.time()
            if length not in ruler._LENGTH_TOKENS:
                print(f"[WARN] unknown length {length}, skipping")
                continue
            target_tokens = ruler._LENGTH_TOKENS[length]
            print(f"[QCMem-RULER] {task}/{length}: selector={sel}")
            # Deterministic per-(task,length) RNG so shards share the sample set
            # (identical construction to eval_ruler_mem_space.main).
            base_seed = args.seed + (hash((task, length)) % 100000)

            # Fixed in-context example for VT (shared across all samples).
            vt_icl = None
            if task == "variable_tracking":
                vt_icl = ruler._make_vt_icl(random.Random(base_seed + 777), 4)

            sample_indices = set(
                list(range(args.limit))[args.shard_index::args.num_shards]
            )
            if sharded:
                print(f"[QCMem-RULER] {task}/{length} shard "
                      f"{args.shard_index}/{args.num_shards}: "
                      f"{len(sample_indices)} of {args.limit} samples")

            df = pd.DataFrame({"target": [], "output": [], "question": [],
                               "recall": []})
            recall_sum = 0.0
            total = 0
            n_tok_seen = 0
            read_len_sum = 0
            read_len_last = 0
            oom_count = 0
            mnt = args.max_new_tokens if task != "variable_tracking" \
                else max(args.max_new_tokens, 60)

            for i in tqdm(range(args.limit), desc=f"{task}/{length}", leave=False):
                # Build EVERY sample (fixed per-i seed) so shard sample sets align,
                # then process only this shard's indices.
                rng = random.Random(base_seed * 1000 + i)
                prompt, answers, gold_needle = ruler._build_sample(
                    task, target_tokens, tokenizer, rng, vt_icl)
                if i not in sample_indices:
                    continue

                # bare_q / bare_q_ids MUST come from the ORIGINAL raw prompt — the
                # chat template would corrupt the _bare_question last-line extraction.
                bare_q = _bare_question(prompt)
                bare_q_ids = tokenizer.encode(bare_q, add_special_tokens=False)

                # Only the model INPUT is (optionally) chat-templated. The
                # assistant generation prefix (no-think block) is NOT baked into
                # the chunked input (add_generation_prompt=False); it is appended
                # at the QCMem query boundary via gen_boundary_ids so the closed
                # <think></think> always lands at the generation tail regardless
                # of chunk_size (see qcmem_generate). If the boundary delta could
                # not be extracted, fall back to baking it into the input.
                model_prompt = prompt
                if args.use_chat_template:
                    messages = [{"role": "user", "content": prompt}]
                    add_gen = gen_boundary_ids is None
                    try:
                        model_prompt = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=add_gen,
                            enable_thinking=args.enable_thinking)
                    except TypeError:  # non-Qwen3 tokenizer
                        model_prompt = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=add_gen)

                ids = tokenizer.encode(model_prompt, add_special_tokens=True,
                                       return_tensors="pt")
                if isinstance(ids, list):
                    ids = torch.tensor([ids], dtype=torch.long)
                input_ids = ids.to(device)
                n_tok_seen = int(input_ids.shape[1])

                # oracle needle chunks (NIAH only; VT has no single gold span).
                needle_set = None
                if sel == "oracle":
                    needle_set = _oracle_needle_chunks(
                        input_ids, gold_needle, answers,
                        tokenizer, args.chunk_size)

                try:
                    gen_stats: dict = {}
                    output = qcmem_generate(
                        qc=qc, tokenizer=tokenizer, input_ids=input_ids,
                        chunk_size=args.chunk_size, max_new_tokens=mnt,
                        selector=sel, topk=args.topk,
                        sink_tokens=args.sink_tokens,
                        needle_chunk_set=needle_set, bare_question_ids=bare_q_ids,
                        no_retrieval=no_retrieval, stats=gen_stats,
                        iter_rounds=args.iter_rounds,
                        iter_hop_topk=args.iter_hop_topk,
                        iter_score=args.iter_score,
                        iter_conf_ratio=args.iter_conf_ratio,
                        iter_max_chunks=args.iter_max_chunks,
                        gen_boundary_ids=gen_boundary_ids,
                    )
                    if "read_len" in gen_stats:
                        read_len_last = int(gen_stats["read_len"])
                        read_len_sum += read_len_last
                except RuntimeError as e:
                    if "out of memory" not in str(e).lower():
                        raise
                    output = "[OOM]"
                    oom_count += 1
                    print(f"[OOM] i={i} task={task} length={length}: {e}",
                          flush=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                rec = ruler._string_match_all_one(output, answers)
                recall_sum += rec
                total += 1
                df.loc[len(df)] = [" | ".join(answers), output, bare_q, rec]
                if len(df) % 10 == 0:
                    qcb.harness._write_results_csv(
                        df, outdir / f"{task}_{length}{shard_tag}.csv")

            score = (recall_sum / total * 100.0) if total else 0.0
            avg_read_len = round(read_len_sum / total, 1) if total else 0
            summary[task][length] = {
                "score": round(score, 2), "n": total,
                "approx_tokens": n_tok_seen,
                # Packed read length (sink + selected chunk hiddens + query) — the
                # efficiency-table quantity. CONSTANT for QCMem (fixed topk),
                # grows O(context) for the kvdirect/hcache baselines (all chunks).
                "avg_read_len": avg_read_len,
                "last_read_len": read_len_last,
            }
            outfile = outdir / f"{task}_{length}{shard_tag}.csv"
            qcb.harness._write_results_csv(df, outfile)
            cfg_file = outdir / f"{task}_{length}{shard_tag}.json"
            json.dump(
                {
                    "status": "completed" if oom_count == 0 else "failed",
                    "task": task, "length": length,
                    "n_requested": args.limit,
                    "sharding": {"num_shards": args.num_shards,
                                 "shard_index": args.shard_index},
                    "summary": summary[task][length],
                    "score": summary[task][length]["score"],
                    "oom_count": oom_count,
                    "elapsed_seconds": round(time.time() - cell_started, 3),
                    "runtime": {
                        "node": socket.gethostname(),
                        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                        "device": args.device,
                        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
                        "seed": args.seed,
                        "dtype": args.dtype,
                        "attn_implementation": args.attn_impl,
                    },
                    "chat_template": bool(args.use_chat_template),
                    "enable_thinking": bool(args.enable_thinking),
                    "scoring": "scripts.eval_ruler_mem_space._string_match_all_one",
                    "baseline": args.baseline,
                    "no_retrieval": bool(no_retrieval),
                    "qcmem": {
                        "resume_j": args.resume_j,
                        "top_prepay_b": args.top_prepay_b,
                        "reuse_kv_blockdiag": bool(args.reuse_kv_blockdiag),
                        "selector": (None if no_retrieval else sel),
                        "topk": (None if no_retrieval else args.topk),
                        "iter": (
                            {"rounds": args.iter_rounds,
                             "hop_topk": args.iter_hop_topk,
                             "score": args.iter_score}
                            if (not no_retrieval and sel == "iter_reader_attn")
                            else {"rounds": args.iter_rounds,
                                  "hop_topk": args.iter_hop_topk}
                            if (not no_retrieval and sel == "iter_bm25")
                            else {"hop_topk": args.iter_hop_topk,
                                  "conf_ratio": args.iter_conf_ratio,
                                  "max_chunks": args.iter_max_chunks}
                            if (not no_retrieval and sel == "iter_bm25_adaptive")
                            else None
                        ),
                        "sink_tokens": args.sink_tokens, "num_layers": L,
                        "lora_adapter": args.lora_adapter or None,
                        "chunk_size": args.chunk_size,
                    },
                    "model": {"model_path": args.model_path,
                              "num_hidden_layers": L,
                              "bottleneck_ckpt": args.bottleneck_ckpt or None},
                    "zero_training_no_adapter": bool(
                        args.baseline == "none" and not args.lora_adapter
                        and not args.bottleneck_ckpt
                    ),
                },
                open(cfg_file, "w"), indent=2,
            )
            print(f"[QCMem-RULER] {task}/{length}: recall={score:.2f} "
                  f"({total} samples, ~{n_tok_seen} tok, "
                  f"read_len~{avg_read_len}) -> {outfile}")

    print("\n[QCMem-RULER] SUMMARY")
    for task in summary:
        row = "  ".join(
            f"{ln}={summary[task][ln]['score']:.1f}" for ln in summary[task])
        print(f"  {task:>18}: {row}")
    with open(outdir / f"_summary{shard_tag}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[QCMem-RULER] Evaluation complete!")


if __name__ == "__main__":
    main()
