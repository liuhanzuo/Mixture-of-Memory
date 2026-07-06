#!/usr/bin/env python
"""QCMem mid-depth resume — zero-training BABILong j-sweep driver.

Evaluates the QCMem write/read primitive (``src/memory/qcmem/qcmem_model.py``)
on BABILong with a *plain* Llama-3-8B backbone (NO training, NO mem_space patch).
The question this answers: does the QCMem claim "small resume-depth j already
collapses on a real Llama backbone" hold for our model? j=0 is the RAG upper
bound (selective full re-forward); j=L is closed-book.

Reused verbatim from ``scripts/run_babilong_mem_space.py`` (imported, not copied):
  * ``load_babilong_dataset``          — offline Arrow-cache dataset loading
  * ``_bm25_scores``                   — pure-CPU lexical BM25 (no forward)
  * ``_locate_needle_chunks`` / ``_find_subsequence_ids`` — oracle needle locator
  * ``_write_results_csv`` / ``_sanitize_output`` — CSV writer (QUOTE_ALL)
Prompt formatting uses babilong's ``get_formatted_input`` + ``DEFAULT_PROMPTS``.
Scoring is left to ``scripts/score_nested_babilong.py`` (official
``babilong.metrics.compare_answers``); this driver only writes the CSVs in the
same nested layout so the standard scorer + 2-group task-pool merge apply.

--------------------------------------------------------------------------- #
CORRECTNESS GATE — --self_test (run this FIRST, must pass before any eval)
--------------------------------------------------------------------------- #
At j=0 the QCMem write/read packing path MUST reproduce a stock
``model(input_ids=packed)`` forward to floating-point tolerance (fp32 max
|logit diff| < 1e-4), because j=0 write == embed and j=0 read == "resume from
layer 0 on the packed embeddings with contiguous positions" == the standard
forward on the concatenated token ids (RoPE lives inside the layers, embeddings
are position-free). This mirrors the max-diff=0 result of
``scripts/qcmem_resume_primitive_check.py`` on the tiny model, scaled to the
real backbone. If self_test fails, the implementation is wrong — do NOT eval.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (PROJECT_ROOT,
          os.path.join(PROJECT_ROOT, "scripts"),
          os.path.join(PROJECT_ROOT, "third_party", "babilong-pkg")):
    if p not in sys.path:
        sys.path.insert(0, p)

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from babilong.prompts import DEFAULT_PROMPTS, DEFAULT_TEMPLATE, get_formatted_input  # noqa: E402

# Reuse the harness helpers (importing the module defines mem_space classes but
# does NOT patch any model — QCMem never calls apply_mem_space_to_model).
# Load it by EXPLICIT file path: a stale duplicate ``run_babilong_mem_space.py``
# may sit at the project root and shadow ``scripts/run_babilong_mem_space.py`` on
# sys.path, so we bind the canonical scripts copy directly.
import importlib.util as _ilu  # noqa: E402

_harness_path = os.path.join(PROJECT_ROOT, "scripts", "run_babilong_mem_space.py")
_spec = _ilu.spec_from_file_location("_qcmem_harness", _harness_path)
harness = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(harness)

from src.memory.qcmem import QCMemModel  # noqa: E402


# --------------------------------------------------------------------------- #
# chunk selection
#   recency / bm25 / oracle : lexical / positional, no model forward.
#   reader_attn             : semantic, scores the ALREADY-computed depth-j
#                             ``write_chunk`` hiddens (bottom-j only, NOT a full
#                             model re-forward — so QCMem's compute saving holds).
# --------------------------------------------------------------------------- #
def _reader_attn_scores(context_hj, query_hj):
    """Salience of each context chunk to the query, from cached depth-``j`` hidden
    states (``h_j`` == the ``write_chunk`` output; NO extra model forward).

    Score = cosine similarity between the mean-pooled (over the token axis) query
    ``h_j`` vector and each context chunk's mean-pooled ``h_j`` vector. Mean-pool
    collapses ``[1, T, d] -> [d]`` so variable chunk lengths compare cleanly, and
    cosine normalises away per-chunk hidden-norm scale differences.

    Returns ``list[float]`` aligned with ``context_hj`` (higher == more salient).
    """
    # query vector: mean over tokens, then L2-normalise (fp32 for a stable dot).
    q_vec = query_hj.float().mean(dim=1).squeeze(0)          # [d]
    q_vec = q_vec / (q_vec.norm() + 1e-8)
    scores = []
    for h in context_hj:
        if h is None or h.shape[1] == 0:
            scores.append(float("-inf"))
            continue
        c_vec = h.float().mean(dim=1).squeeze(0)             # [d]
        c_vec = c_vec / (c_vec.norm() + 1e-8)
        scores.append(float(torch.dot(q_vec, c_vec).item()))  # cosine similarity
    return scores


def _select_context_chunk_indices(
    selector: str,
    context_chunks,        # list[LongTensor] == chunks[:-1] (doc order)
    query_ids,             # list[int] bare-question token ids
    topk: int,
    needle_chunk_set,      # set[int] doc-absolute chunk indices (oracle) or None
    context_hj=None,       # list[Tensor [1,T,d]] cached h_j  (reader_attn only)
    query_hj=None,         # Tensor [1,T,d] query chunk h_j   (reader_attn only)
):
    """Return a sorted list of context-chunk indices (into ``context_chunks``)
    to pack into the read, chosen by the requested selector.

    * ``recency``     — the last ``topk`` context chunks.
    * ``bm25``        — the ``topk`` context chunks with the highest lexical BM25
                        overlap with the bare question (pure CPU, no forward).
    * ``oracle``      — the context chunks that CONTAIN the gold answer (upper
                        bound). Falls back to ``recency`` if the needle can't be
                        located.
    * ``reader_attn`` — the ``topk`` context chunks whose cached depth-``j`` hidden
                        ``h_j`` is most salient to the query ``h_j`` (mean-pool
                        cosine; see :func:`_reader_attn_scores`). Consumes the
                        ALREADY-computed ``write_chunk`` outputs, so it adds NO
                        model forward beyond the bottom-j writes QCMem already
                        does. Falls back to ``recency`` if the caller did not
                        supply ``context_hj`` / ``query_hj``.

    ``context_hj`` / ``query_hj`` are used only by ``reader_attn``; the other
    selectors ignore them (kwargs default to ``None`` for backward compatibility).
    """
    n_ctx = len(context_chunks)
    if n_ctx == 0:
        return []
    k = max(0, int(topk))

    if selector == "recency":
        if k <= 0:
            return []
        return list(range(max(0, n_ctx - k), n_ctx))

    if selector == "oracle":
        if needle_chunk_set:
            # needle_chunk_set holds DOCUMENT-ABSOLUTE chunk indices; context
            # chunks are doc-absolute 0..n_ctx-1 (query is the last, index n_ctx).
            sel = sorted(c for c in needle_chunk_set if 0 <= c < n_ctx)
            if sel:
                return sel
        # fall back to recency when the needle can't be located
        if k <= 0:
            return []
        return list(range(max(0, n_ctx - k), n_ctx))

    if selector == "bm25":
        if k <= 0:
            return []
        docs = [c.tolist() for c in context_chunks]
        scores = harness._bm25_scores(docs, list(query_ids))
        if not scores:
            return list(range(max(0, n_ctx - k), n_ctx))
        order = sorted(range(n_ctx), key=lambda i: scores[i], reverse=True)
        return sorted(order[:k])

    if selector == "reader_attn":
        if k <= 0:
            return []
        # Needs the cached h_j of every context chunk + the query chunk. If the
        # caller didn't supply them, degrade gracefully to recency.
        if not context_hj or query_hj is None or len(context_hj) != n_ctx:
            return list(range(max(0, n_ctx - k), n_ctx))
        scores = _reader_attn_scores(context_hj, query_hj)
        order = sorted(range(n_ctx), key=lambda i: scores[i], reverse=True)
        return sorted(order[:k])

    raise ValueError(f"unknown selector {selector!r}")


# --------------------------------------------------------------------------- #
# per-sample QCMem generation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def qcmem_generate(
    qc: QCMemModel,
    tokenizer,
    input_ids: torch.Tensor,      # [1, L] full formatted sample (BOS-prefixed)
    chunk_size: int,
    max_new_tokens: int,
    selector: str,
    topk: int,
    sink_tokens: str,             # "bos" | "none"
    needle_chunk_set=None,        # set[int] for oracle; else None
    bare_question_ids=None,       # list[int] for bm25 query
) -> str:
    device = qc.device
    tokens = input_ids[0]
    chunks = list(tokens.split(chunk_size))
    n_chunks = len(chunks)

    context_chunks = chunks[:-1]      # everything before the question chunk
    query_chunk = chunks[-1]

    # ---- sink: BOS's depth-j hidden (attention-sink anchor at packed pos 0) ----
    sink_hj = None
    if sink_tokens == "bos":
        bos_id = tokenizer.bos_token_id
        if bos_id is None:
            bos_id = int(tokens[0].item())
        sink_hj = qc.write_chunk([bos_id])

    # ---- select which context chunks to pack ----
    # reader_attn scores each chunk by its cached depth-j hidden (query h_j vs
    # each context chunk h_j). Those hiddens are the bottom-j ``write_chunk``
    # outputs QCMem needs anyway, so we compute them up front and REUSE the
    # selected ones below — nothing is written (or full-forwarded) twice.
    context_hj = None
    query_hj_for_sel = None
    if selector == "reader_attn":
        context_hj = [qc.write_chunk(c) for c in context_chunks]
        query_hj_for_sel = qc.write_chunk(query_chunk)

    sel_idx = _select_context_chunk_indices(
        selector, context_chunks, bare_question_ids or [], topk, needle_chunk_set,
        context_hj=context_hj, query_hj=query_hj_for_sel,
    )

    # ---- write (encode to depth j) the selected context chunks ONCE ----
    if context_hj is not None:
        # reader_attn already wrote every context chunk: reuse, do not re-write.
        selected_hj = [context_hj[i] for i in sel_idx]
    else:
        selected_hj = [qc.write_chunk(context_chunks[i]) for i in sel_idx]

    # ---- greedy decode: only the growing query chunk is re-encoded per step ----
    query_ids = query_chunk.tolist()
    eos_id = tokenizer.eos_token_id
    generated = []
    for step in range(max_new_tokens):
        q_hj = qc.write_chunk(query_ids)
        logits = qc.read(sink_hj, selected_hj, q_hj)   # [1, |H|, V]
        next_logits = logits[0, -1].float()
        if step == 0 and eos_id is not None:
            next_logits[eos_id] = float("-inf")
        next_tok = int(next_logits.argmax().item())
        if eos_id is not None and next_tok == eos_id and step > 0:
            break
        generated.append(next_tok)
        query_ids = query_ids + [next_tok]

    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# --------------------------------------------------------------------------- #
# self-test (correctness gate)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_self_test(model, tokenizer, device, chunk_size: int) -> bool:
    """Verify QCMem write/read at j=0 == stock full forward (fp32 < 1e-4).

    Builds a packed token sequence [sink(BOS) ; c1 ; c2 ; c3 ; query], runs it
    through QCMem's write_chunk (each chunk to depth 0) + read, and compares the
    packed logits against ``model(input_ids=packed)``. Because j=0 write is a
    bare embedding lookup and j=0 read resumes layers[0:] over the concatenated
    embeddings with contiguous positions, the two MUST match to fp-tolerance.

    Also checks resume_forward_ids (single-sequence split-at-0) as a second,
    stricter mirror of the primitive check.
    """
    print("=" * 72)
    print("QCMem self-test on real backbone (j=0 packing == full forward)")
    print("=" * 72)
    qc0 = QCMemModel(model, resume_j=0)

    torch.manual_seed(0)
    V = int(model.config.vocab_size)
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1

    # Build sink + 3 context chunks + a query chunk (small so it runs on 1 card).
    def rand_ids(n):
        return torch.randint(0, V, (1, n), device=device)

    sink_ids = torch.tensor([[bos_id]], device=device)
    c1, c2, c3 = rand_ids(37), rand_ids(29), rand_ids(41)
    q = rand_ids(23)
    packed_ids = torch.cat([sink_ids, c1, c2, c3, q], dim=1)

    # Reference: stock full forward on the concatenated ids.
    ref = qc0.full_forward_logits(packed_ids)                 # [1, |H|, V]

    # (A) write/read packing path at j=0.
    sink_hj = qc0.write_chunk(sink_ids)
    ctx_hj = [qc0.write_chunk(c) for c in (c1, c2, c3)]
    q_hj = qc0.write_chunk(q)
    out_pack = qc0.read(sink_hj, ctx_hj, q_hj)                # [1, |H|, V]
    diff_pack = (out_pack.float() - ref.float()).abs().max().item()

    # (B) single-sequence split-at-0 resume.
    out_resume = qc0.resume_forward_ids(packed_ids)
    diff_resume = (out_resume.float() - ref.float()).abs().max().item()

    # (C) sanity: a NON-zero j on a single sequence must ALSO equal full forward
    # (resume primitive holds at every j when the whole seq is one chunk).
    L = int(model.config.num_hidden_layers)
    diffs_j = {}
    for j in (1, L // 2, L):
        qcj = QCMemModel(model, resume_j=j)
        outj = qcj.resume_forward_ids(packed_ids)
        diffs_j[j] = (outj.float() - ref.float()).abs().max().item()

    tol = 1e-4
    print(f"  model dtype: {next(model.parameters()).dtype}, layers L={L}, "
          f"packed_len={packed_ids.shape[1]}")
    print(f"  (A) write/read packing (j=0) max|logit diff| = {diff_pack:.3e}  "
          f"{'PASS' if diff_pack < tol else 'FAIL'}")
    print(f"  (B) resume_forward_ids (j=0) max|logit diff| = {diff_resume:.3e}  "
          f"{'PASS' if diff_resume < tol else 'FAIL'}")
    for j, d in diffs_j.items():
        print(f"  (C) resume_forward_ids single-seq (j={j:>2}) max|diff| = {d:.3e}  "
              f"{'PASS' if d < tol else 'FAIL'}")
    ok = (diff_pack < tol and diff_resume < tol
          and all(d < tol for d in diffs_j.values()))
    print("-" * 72)
    print(f"SELF-TEST: {'ALL PASS — QCMem read == full forward at j=0 (impl correct)' if ok else 'FAILURE — DO NOT EVAL'}")
    print("=" * 72)
    return ok


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="QCMem mid-depth resume — zero-training BABILong j-sweep"
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to plain Llama-3-8B weights.")
    parser.add_argument("--resume_j", type=int, default=6,
                        help="Layer split index j (0=RAG upper bound, L=closed-book).")
    parser.add_argument("--top_prepay_b", type=int, default=0,
                        help="Direction-B top-prepay: run the top b layers "
                             "query-local at read (0=exact connective resume).")
    parser.add_argument("--lora_adapter", type=str, default="",
                        help="Optional path to a trained QCMem-distill LoRA "
                             "adapter dir (Direction A). Loaded onto the frozen "
                             "backbone before building the QCMem orchestrator.")
    parser.add_argument("--selector", type=str, default="bm25",
                        choices=["bm25", "recency", "oracle", "reader_attn"],
                        help="Chunk selector for the read pack. reader_attn scores "
                             "chunks by query-h_j vs chunk-h_j cosine (reuses the "
                             "cached write_chunk hiddens, no extra forward).")
    parser.add_argument("--topk", type=int, default=4,
                        help="Number of context chunks to pack into the read.")
    parser.add_argument("--sink_tokens", type=str, default="bos",
                        choices=["bos", "none"],
                        help="Attention-sink anchor at packed position 0.")
    parser.add_argument("--results_folder", type=str, default="./babilong_results")
    parser.add_argument("--output_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="RMT-team/babilong")
    parser.add_argument("--tasks", type=str, nargs="+",
                        default=["qa1", "qa2", "qa5"])
    parser.add_argument("--lengths", type=str, nargs="+",
                        default=["0k", "1k", "2k", "4k", "8k", "16k"])
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn_impl", type=str, default="sdpa")
    parser.add_argument("--use_instruction", action="store_true", default=True)
    parser.add_argument("--use_examples", action="store_true", default=True)
    parser.add_argument("--use_post_prompt", action="store_true", default=True)
    parser.add_argument("--use_chat_template", action="store_true", default=False)
    parser.add_argument("--self_test", action="store_true", default=False,
                        help="Run the j=0 correctness gate on the real backbone "
                             "and exit. Forces fp32 for a tight tolerance.")
    args = parser.parse_args()

    if args.num_shards < 1:
        parser.error("--num_shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        parser.error(f"--shard_index must be in [0, {args.num_shards})")

    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16,
             "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    # self_test needs fp32 to hit the <1e-4 gate (bf16 has ~1e-2 roundoff).
    if args.self_test:
        dtype = torch.float32

    print(f"[QCMem-BABILong] model_path={args.model_path}")
    print(f"[QCMem-BABILong] resume_j={args.resume_j} selector={args.selector} "
          f"topk={args.topk} sink={args.sink_tokens} chunk_size={args.chunk_size} "
          f"dtype={dtype} attn_impl={args.attn_impl}")

    # local_files_only=True: when the node is offline (e.g. .52 H20), transformers
    # otherwise falls back to treating a local dir path as an HF repo_id and errors
    # with "Repo id must be in the form ...". Force pure local resolution.
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

    # Direction A: load a trained QCMem-distill LoRA adapter onto the backbone.
    # QCMemModel reads .model.layers / .lm_head etc. off whatever object we hand
    # it; PeftModel.base_model.model is the underlying CausalLM exposing that
    # structure, and the LoRA-wrapped Linear submodules apply their delta when
    # called directly by _run_layers (verified 2026-07-05).
    if args.lora_adapter:
        from peft import PeftModel
        print(f"[QCMem-BABILong] loading LoRA adapter: {args.lora_adapter}")
        peft_model = PeftModel.from_pretrained(model, args.lora_adapter).eval()
        model = peft_model.base_model.model

    if args.self_test:
        ok = run_self_test(model, tokenizer, device, args.chunk_size)
        sys.exit(0 if ok else 1)

    qc = QCMemModel(model, resume_j=args.resume_j, top_prepay_b=args.top_prepay_b)

    for task in tqdm(args.tasks, desc="tasks"):
        if task not in DEFAULT_PROMPTS:
            print(f"[WARNING] Task {task} not in DEFAULT_PROMPTS, skipping.")
            continue
        prompt_cfg = {
            "instruction": DEFAULT_PROMPTS[task]["instruction"] if args.use_instruction else "",
            "examples":    DEFAULT_PROMPTS[task]["examples"]    if args.use_examples    else "",
            "post_prompt": DEFAULT_PROMPTS[task]["post_prompt"] if args.use_post_prompt else "",
            "template":    DEFAULT_TEMPLATE,
            "chat_template": args.use_chat_template,
            "system_prompt": "",
        }
        prompt_name = "_".join(
            [f"{k}_yes" if prompt_cfg[k] else f"{k}_no"
             for k in prompt_cfg if k != "template"]
        )

        for split_name in tqdm(args.lengths, desc="lengths", leave=False):
            print(f"\n[QCMem-BABILong] task={task}, length={split_name}")
            try:
                data = harness.load_babilong_dataset(args.dataset_name, split_name)
                task_data = data[task]
            except Exception as e:
                print(f"[ERROR] Failed to load {args.dataset_name}/{split_name}/{task}: {e}")
                continue

            outdir = Path(args.results_folder) / args.output_name
            outdir.mkdir(parents=True, exist_ok=True)
            sharded = args.num_shards > 1
            shard_tag = f"_shard{args.shard_index}of{args.num_shards}" if sharded else ""
            outfile = outdir / f"{task}_{split_name}_{prompt_name}{shard_tag}.csv"
            cfg_file = outdir / f"{task}_{split_name}_{prompt_name}{shard_tag}.json"

            json.dump(
                {
                    "prompt": prompt_cfg,
                    "generate_kwargs": {
                        "max_new_tokens": args.max_new_tokens,
                        "do_sample": False, "num_beams": 1,
                    },
                    # QCMem free-selector reads re-forward RAW TOKENS of selected
                    # chunks -> a THEORETICAL/oracle-ish upper bound (esp. selector
                    # =oracle). Kept for parity with the reforward-guard tag.
                    "theoretical_upper_bound": bool(args.selector == "oracle"),
                    "qcmem": {
                        "resume_j": args.resume_j,
                        "top_prepay_b": args.top_prepay_b,
                        "selector": args.selector,
                        "topk": args.topk,
                        "sink_tokens": args.sink_tokens,
                        "num_layers": L,
                        "lora_adapter": args.lora_adapter or None,
                    },
                    "model": {
                        "model_path": args.model_path,
                        "chunk_size": args.chunk_size,
                    },
                },
                open(cfg_file, "w"), indent=4,
            )

            df = pd.DataFrame({"target": [], "output": [], "question": []})

            num_samples = len(task_data)
            if args.limit > 0:
                num_samples = min(num_samples, args.limit)
            sample_indices = list(range(num_samples))[args.shard_index::args.num_shards]
            if sharded:
                print(f"[QCMem-BABILong] shard {args.shard_index}/{args.num_shards}: "
                      f"{len(sample_indices)} of {num_samples} samples")

            for idx in tqdm(sample_indices, desc=f"{task}/{split_name}", leave=False):
                sample = task_data[idx]
                target = sample["target"]
                question = sample["question"]
                input_text = get_formatted_input(
                    sample["input"], sample["question"],
                    prompt_cfg["examples"], prompt_cfg["instruction"],
                    prompt_cfg["post_prompt"], template=prompt_cfg["template"],
                )
                if args.use_chat_template:
                    messages = [{"role": "user", "content": input_text}]
                    input_text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                ids = tokenizer.encode(input_text, add_special_tokens=True, return_tensors="pt")
                if isinstance(ids, list):
                    ids = torch.tensor([ids], dtype=torch.long)
                input_ids = ids.to(device)

                # oracle needle chunks (doc-absolute) for selector=oracle
                needle_set = None
                if args.selector == "oracle":
                    needle_set = harness._locate_needle_chunks(
                        input_ids, target, tokenizer, args.chunk_size
                    )
                bare_q_ids = tokenizer.encode(
                    (question or "").strip(), add_special_tokens=False
                )

                try:
                    output = qcmem_generate(
                        qc=qc, tokenizer=tokenizer, input_ids=input_ids,
                        chunk_size=args.chunk_size, max_new_tokens=args.max_new_tokens,
                        selector=args.selector, topk=args.topk,
                        sink_tokens=args.sink_tokens,
                        needle_chunk_set=needle_set, bare_question_ids=bare_q_ids,
                    )
                except RuntimeError as e:
                    if "out of memory" not in str(e).lower():
                        raise
                    output = "[OOM]"
                    print(f"[OOM] idx={idx} task={task} length={split_name}: {e}", flush=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                df.loc[len(df)] = [target, output, question]
                if len(df) % 10 == 0 or idx == sample_indices[-1]:
                    harness._write_results_csv(df, outfile)

            harness._write_results_csv(df, outfile)
            print(f"[QCMem-BABILong] Saved {len(df)} results to {outfile}")

    print("\n[QCMem-BABILong] Evaluation complete!")


if __name__ == "__main__":
    main()
