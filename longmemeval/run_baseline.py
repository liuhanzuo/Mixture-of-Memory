"""LongMemEval system-memory baseline — single CLI entrypoint.

Run a round-level RAG memory baseline over the official LongMemEval data and
emit (1) a submission JSONL for the official GPT-4o judge and (2) a cheap
API-free retrieval recall@k report.

Examples
--------
Synthetic smoke (no external data, BM25-only, stub reader)::

    python -m longmemeval.run_baseline --self_test

Real data, BM25 retriever, stub reader (recall diagnostic only)::

    python -m longmemeval.run_baseline \
        --data data/longmemeval/longmemeval_s.json \
        --retriever bm25 --top_k 10 --reader stub \
        --out outputs/longmemeval/bm25_stub.jsonl

Union (BM25 + bge-m3 embeddings) with GPT-4o reader::

    export LONGMEMEVAL_READER_API_KEY=...   # never hardcode
    python -m longmemeval.run_baseline \
        --data data/longmemeval/longmemeval_s.json \
        --retriever union --embed_model models/bge-m3 \
        --top_k 10 --evidence_token_budget 4000 \
        --reader openai --reader_model gpt-4o \
        --out outputs/longmemeval/union_gpt4o.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from .backends import (
    Evidence,
    IdentityReranker,
    KeywordOverlapReranker,
    MoMModelReranker,
    MoMSlotReranker,
    RoundFlatRetriever,
    TemporalAwareReranker,
)
from .data import LongMemEvalExample, Round, load_longmemeval
from .reader import build_reader
from .scoring import aggregate_recall, recall_at_k, write_submission


# Rough word->token proxy. The official data is in English; ~1.3 tokens/word.
_TOKENS_PER_WORD = 1.3


def _apply_token_budget(evidence: List[Evidence], budget: int) -> List[Evidence]:
    """Trim evidence list so total approx tokens <= budget (keep order)."""
    if not budget or budget <= 0:
        return evidence
    out: List[Evidence] = []
    used = 0
    for ev in evidence:
        approx = int(len(ev.text.split()) * _TOKENS_PER_WORD)
        if out and used + approx > budget:
            break
        used += approx
        out.append(ev)
    return out


def _build_reranker(name: str, args=None):
    if name == "none":
        return IdentityReranker()
    if name == "keyword":
        return KeywordOverlapReranker()
    if name == "temporal":
        return TemporalAwareReranker()
    if name == "mom_stub":
        return MoMSlotReranker()
    if name == "mom_slot":
        if args is None:
            raise ValueError("mom_slot reranker requires CLI args (checkpoint/config)")
        if not args.mom_checkpoint or not args.mom_adapter_config:
            raise ValueError(
                "--reranker mom_slot requires --mom_checkpoint and "
                "--mom_adapter_config"
            )
        return MoMModelReranker(
            checkpoint=args.mom_checkpoint,
            adapter_config=args.mom_adapter_config,
            model_path=args.mom_model_path,
            device=args.reranker_device,
            fusion_weight=args.mom_fusion_weight,
            chunk_size=args.mom_chunk_size,
        )
    raise ValueError(f"unknown reranker: {name}")


def _build_retriever(args) -> RoundFlatRetriever:
    return RoundFlatRetriever(
        mode=args.retriever,
        embed_model_path=args.embed_model,
        device=args.device,
        reranker=_build_reranker(args.reranker, args),
        candidate_multiplier=args.candidate_multiplier,
    )


def _run(examples: List[LongMemEvalExample], args) -> dict:
    reader = build_reader(args.reader, model=args.reader_model)
    retriever = _build_retriever(args)

    submission = []
    recalls = []
    per_type_recall = {}

    for ex in examples:
        retriever.index_example(ex)
        evidence = retriever.query(ex.question, date=ex.question_date, top_k=args.top_k)
        evidence = _apply_token_budget(evidence, args.evidence_token_budget)

        rec = recall_at_k(evidence, ex.answer_session_ids, ex.question_id)
        recalls.append(rec)
        per_type_recall.setdefault(ex.question_type, []).append(rec)

        hypothesis = reader.answer(
            ex.question, ex.question_date, evidence, token_budget=0
        )
        submission.append({"question_id": ex.question_id, "hypothesis": hypothesis})

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        write_submission(args.out, submission)

    report = {
        "n_questions": len(examples),
        "retriever_mode_requested": args.retriever,
        "retriever_mode_effective": retriever.effective_mode,
        "degraded": retriever.degraded,
        "degraded_reason": retriever.degraded_reason,
        "reader": args.reader,
        "reranker": args.reranker,
        "top_k": args.top_k,
        "evidence_token_budget": args.evidence_token_budget,
        "overall_recall": aggregate_recall(recalls),
        "recall_by_type": {
            t: aggregate_recall(rs) for t, rs in per_type_recall.items()
        },
        "submission_path": args.out,
    }
    return report


def _make_synthetic() -> List[LongMemEvalExample]:
    """Two fake questions to prove the pipeline end-to-end (no external data)."""
    def sess(turns):
        return [{"role": r, "content": c} for r, c in turns]

    ex1 = LongMemEvalExample(
        question_id="synthetic_1",
        question_type="single-session-user",
        question="What kind of pet did I say I adopted?",
        answer="a golden retriever puppy",
        question_date="2024-05-01",
        haystack_sessions=[
            sess([("user", "I love hiking in the mountains."),
                  ("assistant", "Hiking is great exercise!")]),
            sess([("user", "I just adopted a golden retriever puppy named Max."),
                  ("assistant", "Congratulations on adopting Max the golden retriever!")]),
            sess([("user", "What's a good recipe for pasta?"),
                  ("assistant", "Try a simple garlic and olive oil pasta.")]),
        ],
        haystack_dates=["2024-01-10", "2024-02-15", "2024-03-20"],
        haystack_session_ids=["s1", "s2", "s3"],
        answer_session_ids=["s2"],
    )
    ex2 = LongMemEvalExample(
        question_id="synthetic_2",
        question_type="temporal-reasoning",
        question="Which city did I move to most recently?",
        answer="Seattle",
        question_date="2024-05-01",
        haystack_sessions=[
            sess([("user", "I moved to Boston for a new job."),
                  ("assistant", "Boston is a wonderful city!")]),
            sess([("user", "I just relocated to Seattle last week."),
                  ("assistant", "Hope you enjoy Seattle!")]),
        ],
        haystack_dates=["2023-06-01", "2024-04-20"],
        haystack_session_ids=["a1", "a2"],
        answer_session_ids=["a2"],
    )
    return [ex1, ex2]


def _self_test_reranker_probe(examples: List[LongMemEvalExample]) -> dict:
    """Show that temporal reranking changes top-1 recall on synthetic data."""
    ex = next(e for e in examples if e.question_id == "synthetic_2")
    out = {}
    for name in ("none", "temporal", "mom_stub"):
        retriever = RoundFlatRetriever(
            mode="bm25",
            reranker=_build_reranker(name),
            candidate_multiplier=4,
        )
        retriever.index_example(ex)
        evidence = retriever.query(ex.question, date=ex.question_date, top_k=1)
        rec = recall_at_k(evidence, ex.answer_session_ids, ex.question_id)
        out[name] = {
            "top1_session": evidence[0].session_id if evidence else None,
            "top1_score": evidence[0].score if evidence else None,
            "any_hit_recall": rec.any_hit,
        }
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m longmemeval.run_baseline",
        description="LongMemEval system-memory RAG baseline (Track B).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", type=str, default=None,
                   help="Path to official LongMemEval JSON (e.g. longmemeval_s.json).")
    p.add_argument("--limit", type=int, default=None,
                   help="Only evaluate the first N questions.")
    p.add_argument("--retriever", type=str, default="bm25",
                   choices=["bm25", "embedding", "union"],
                   help="First-stage retrieval over rounds.")
    p.add_argument("--reranker", type=str, default="none",
                   choices=["none", "keyword", "temporal", "mom_stub", "mom_slot"],
                   help="Second-stage reranker over the first-stage candidate pool. "
                        "'mom_slot' = real mem_space model-backed reranker.")
    p.add_argument("--embed_model", type=str, default="models/bge-m3",
                   help="Local HF / sentence-transformers embedding model "
                        "(used by embedding/union; degrades to BM25 if unloadable).")
    p.add_argument("--device", type=str, default=None,
                   help="Device for the embedding encoder (cuda/cpu; auto if unset).")
    p.add_argument("--top_k", type=int, default=10,
                   help="Number of evidence rounds passed to the reader.")
    p.add_argument("--candidate_multiplier", type=int, default=4,
                   help="First-stage pool size = top_k * this (before rerank).")
    p.add_argument("--evidence_token_budget", type=int, default=0,
                   help="Approx token cap on total evidence (0 = no cap).")
    p.add_argument("--reader", type=str, default="stub",
                   choices=["stub", "openai", "local", "hf", "local_hf"],
                   help="Reader LLM. 'stub' = no API (recall diagnostic); "
                        "'local'/'hf'/'local_hf' = local HF causal LM.")
    p.add_argument("--reader_model", type=str, default=None,
                   help="Reader model id (e.g. gpt-4o). Defaults via env.")
    p.add_argument("--out", type=str, default=None,
                   help="Submission JSONL output path ({question_id, hypothesis}).")
    p.add_argument("--report", type=str, default=None,
                   help="Optional path to write the JSON metrics report.")
    p.add_argument("--self_test", action="store_true",
                   help="Run an internal synthetic smoke (no data/API needed).")
    # -- MoM model-backed reranker (--reranker mom_slot) ------------------- #
    p.add_argument("--mom_checkpoint", type=str, default=None,
                   help="mem_space adapter .pt checkpoint for --reranker mom_slot.")
    p.add_argument("--mom_adapter_config", type=str, default=None,
                   help="adapter_config.json describing the MemorySpaceConfig "
                        "for --reranker mom_slot.")
    p.add_argument("--mom_model_path", type=str, default="models/Meta-Llama-3-8B",
                   help="Base Llama model dir for the mem_space reranker backbone.")
    p.add_argument("--mom_fusion_weight", type=float, default=0.5,
                   help="Convex fusion weight w for the mom_slot reranker: "
                        "final = w*mom_cosine + (1-w)*bm25 (both min-max "
                        "normalized). 1.0 = pure MoM, 0.0 = BM25 order.")
    p.add_argument("--mom_chunk_size", type=int, default=512,
                   help="Chunk size for streaming text through the mem_space "
                        "memory bank during reranking (match training seq_len).")
    p.add_argument("--reranker_device", type=str, default="cuda:0",
                   help="Device for the mom_slot mem_space reranker model.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.self_test:
        examples = _make_synthetic()
        # Force a no-dependency configuration for the smoke.
        args.retriever = "bm25"
        args.reader = "stub"
        if not args.out:
            args.out = "outputs/longmemeval/self_test.jsonl"
    else:
        if not args.data:
            print("ERROR: --data is required (or use --self_test).", file=sys.stderr)
            return 2
        examples = load_longmemeval(args.data, limit=args.limit)

    report = _run(examples, args)
    if args.self_test:
        report["self_test_reranker_probe"] = _self_test_reranker_probe(examples)
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
