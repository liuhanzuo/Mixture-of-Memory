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

B08 leg-1's three arms (single variable = context composition; retrieval,
reader weights, notes text and decode settings identical across arms). Arm 2
GENERATES the notes cache; arms 1 and 3 consume it read-only so the notes text
is provably the same string::

    COMMON="--data data/longmemeval/longmemeval_s.json --retriever bm25 \
        --reranker none --top_k 10 --evidence_token_budget 4000 \
        --reader local_hf --question_types knowledge-update,single-session-assistant \
        --expect_n 134"

    # arm A-notes+raw  (writes outputs/b08_leg1/notes.jsonl)
    python -m longmemeval.run_baseline $COMMON --compressor self_notes \
        --reader_evidence_mode notes_plus_evidence \
        --notes_cache outputs/b08_leg1/notes.jsonl \
        --context_log outputs/b08_leg1/A-notes+raw/context.jsonl \
        --out outputs/b08_leg1/A-notes+raw/submission.jsonl

    # arm A-notes-only (raw WITHHELD; reuses the same notes verbatim)
    python -m longmemeval.run_baseline $COMMON --compressor self_notes \
        --reader_evidence_mode notes_only \
        --notes_cache outputs/b08_leg1/notes.jsonl --notes_cache_readonly \
        --context_log outputs/b08_leg1/A-notes-only/context.jsonl \
        --out outputs/b08_leg1/A-notes-only/submission.jsonl

    # arm A-raw
    python -m longmemeval.run_baseline $COMMON --compressor none \
        --context_log outputs/b08_leg1/A-raw/context.jsonl \
        --out outputs/b08_leg1/A-raw/submission.jsonl
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
    MoMReadoutReranker,
    MoMSlotReranker,
    RoundFlatRetriever,
    TemporalAwareReranker,
)
from .compressor import build_compressor
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
    if name == "mom_readout":
        if args is None:
            raise ValueError("mom_readout reranker requires CLI args (checkpoint/config)")
        if not args.mom_checkpoint or not args.mom_adapter_config:
            raise ValueError(
                "--reranker mom_readout requires --mom_checkpoint and "
                "--mom_adapter_config"
            )
        return MoMReadoutReranker(
            checkpoint=args.mom_checkpoint,
            adapter_config=args.mom_adapter_config,
            model_path=args.mom_model_path,
            device=args.reranker_device,
            fusion_weight=args.mom_fusion_weight,
            chunk_size=args.mom_chunk_size,
            slot_mean_mix=args.mom_slot_mean_mix,
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


#: B08 leg-1 arm names. The arm is a function of (compressor, evidence mode),
#: so it is derived once and written into every artifact -- an arm label must be
#: recoverable from the file, never from shell history.
_ARM_NAMES = {
    ("none", "notes_plus_evidence"): "A-raw",
    ("none", "notes_only"): "A-raw",          # notes are "" -> composition unchanged
    ("none", "evidence_only"): "A-raw",
    ("self_notes", "notes_plus_evidence"): "A-notes+raw",
    ("self_notes", "notes_only"): "A-notes-only",
    ("self_notes", "evidence_only"): "A-raw-with-notes-cost",
}


def _arm_name(compressor: str, evidence_mode: str) -> str:
    return _ARM_NAMES.get(
        ((compressor or "none").lower(), evidence_mode),
        f"{compressor}:{evidence_mode}",
    )


def _compress(compressor, ex: LongMemEvalExample, evidence: List[Evidence]) -> str:
    """Call ``compress``, passing ``question_id`` when the compressor takes it.

    ``SelfNotesCompressor`` keys its notes cache on ``question_id`` so both
    notes arms read byte-identical notes text; the older ``Compressor`` ABC
    signature has no such parameter, so this stays backward compatible.
    """
    try:
        return compressor.compress(
            ex.question, ex.question_date, evidence, question_id=ex.question_id
        )
    except TypeError:
        return compressor.compress(ex.question, ex.question_date, evidence)


def _append_context_record(path: str, rec: dict) -> None:
    """Append one per-item context record (JSONL).

    The ``U`` metric is defined against *that arm's own context*, so scoring it
    requires the exact context each item was answered from -- all of them, not a
    3-example debugging sample. This file is the provenance the scorer reads.
    """
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _select_question_types(
    examples: List[LongMemEvalExample],
    question_types: Optional[str],
    expect_n: Optional[int],
) -> List[LongMemEvalExample]:
    """Filter to a question-type stratum and ASSERT the survivor count.

    ``--limit`` cannot express B08's stratum: it takes a PREFIX, and the
    retrieval-closed stratum (``knowledge-update`` + ``single-session-assistant``)
    is a SUFFIX of the load order. Selecting by type is also robust to file
    ordering, which a hardcoded slice would not be.

    ``--expect_n`` is the mechanical form of the pre-registration's
    "assert ``n_scored == expected`` per cell, not just check for NaNs": it makes
    a silently-wrong stratum fail at INPUT time instead of reaching the scorer.
    """
    if question_types:
        want = {t.strip() for t in question_types.split(",") if t.strip()}
        unknown = want - {e.question_type for e in examples}
        if unknown:
            raise SystemExit(
                f"--question_types names {sorted(unknown)}, absent from the data "
                f"(present: {sorted({e.question_type for e in examples})})"
            )
        examples = [e for e in examples if e.question_type in want]
    if expect_n is not None and len(examples) != expect_n:
        raise SystemExit(
            f"stratum size {len(examples)} != --expect_n {expect_n} "
            f"(question_types={question_types!r}). Refusing to run: a "
            "silently-wrong cell must not reach the scorer."
        )
    return examples


def _run(examples: List[LongMemEvalExample], args) -> dict:
    reader = build_reader(args.reader, model=args.reader_model)
    retriever = _build_retriever(args)
    # The compressor may need the reader itself (--compressor self_notes shares
    # the reader's loaded weights; see compressor.build_compressor).
    compressor = build_compressor(args.compressor, args, reader=reader)
    evidence_mode = getattr(args, "reader_evidence_mode", "notes_plus_evidence")
    notes_label = getattr(compressor, "label", "MoM")

    submission = []
    recalls = []
    per_type_recall = {}
    notes_examples = []

    for ex in examples:
        retriever.index_example(ex)
        evidence = retriever.query(ex.question, date=ex.question_date, top_k=args.top_k)
        evidence = _apply_token_budget(evidence, args.evidence_token_budget)

        rec = recall_at_k(evidence, ex.answer_session_ids, ex.question_id)
        recalls.append(rec)
        per_type_recall.setdefault(ex.question_type, []).append(rec)

        # Compressor path: produce a compact, question-conditioned "notes"
        # synopsis of the retrieved rounds. What the reader then SEES is
        # selected by --reader_evidence_mode -- this is the B08 leg-1 single
        # variable (context composition), with retrieval frozen above.
        # IdentityCompressor returns "" -> evidence unchanged (baseline).
        reader_evidence = evidence
        notes = _compress(compressor, ex, evidence)
        if notes:
            notes_block = Evidence(
                round_id=f"{ex.question_id}_{notes_label.lower()}_notes",
                session_id=f"{notes_label}-NOTES",
                session_date=ex.question_date,
                text=f"{notes_label} NOTES: {notes}",
                score=float("inf"),
            )
            if evidence_mode == "notes_only":
                reader_evidence = [notes_block]
            elif evidence_mode == "evidence_only":
                reader_evidence = list(evidence)
            else:  # notes_plus_evidence -- the pre-2026-08-16 default
                reader_evidence = [notes_block] + list(evidence)
            if len(notes_examples) < 3:
                notes_examples.append(
                    {"question_id": ex.question_id, "question": ex.question, "notes": notes}
                )

        hypothesis = reader.answer(
            ex.question, ex.question_date, reader_evidence, token_budget=0
        )
        submission.append({"question_id": ex.question_id, "hypothesis": hypothesis})
        if args.context_log:
            _append_context_record(
                args.context_log,
                {
                    "question_id": ex.question_id,
                    "question_type": ex.question_type,
                    "question": ex.question,
                    "arm": _arm_name(args.compressor, evidence_mode),
                    "compressor": args.compressor,
                    "reader_evidence_mode": evidence_mode,
                    "notes": notes,
                    "context_blocks": [
                        {"round_id": ev.round_id, "session_id": ev.session_id,
                         "session_date": ev.session_date, "text": ev.text}
                        for ev in reader_evidence
                    ],
                    "hypothesis": hypothesis,
                },
            )

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
        "compressor": args.compressor,
        "reader_evidence_mode": evidence_mode,
        "arm": _arm_name(args.compressor, evidence_mode),
        "notes_label": notes_label,
        "top_k": args.top_k,
        "evidence_token_budget": args.evidence_token_budget,
        "overall_recall": aggregate_recall(recalls),
        "recall_by_type": {
            t: aggregate_recall(rs) for t, rs in per_type_recall.items()
        },
        "submission_path": args.out,
        "context_log_path": args.context_log,
    }
    if hasattr(compressor, "stats"):
        report["compressor_stats"] = compressor.stats
    if notes_examples:
        report["notes_examples"] = notes_examples
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
                   help="Only evaluate the first N questions (a PREFIX; use "
                        "--question_types to select a stratum).")
    p.add_argument("--question_types", type=str, default=None,
                   help="Comma-separated question_type allow-list, e.g. "
                        "'knowledge-update,single-session-assistant' (B08's "
                        "retrieval-closed stratum). Applied after loading; "
                        "unlike --limit this is not order-dependent.")
    p.add_argument("--expect_n", type=int, default=None,
                   help="Assert the number of selected questions equals this, "
                        "else exit non-zero BEFORE any model runs. Pre-registered "
                        "read-out guard (B08: --expect_n 134).")
    p.add_argument("--retriever", type=str, default="bm25",
                   choices=["bm25", "embedding", "union"],
                   help="First-stage retrieval over rounds.")
    p.add_argument("--reranker", type=str, default="none",
                   choices=["none", "keyword", "temporal", "mom_stub", "mom_slot",
                            "mom_readout"],
                   help="Second-stage reranker over the first-stage candidate pool. "
                        "'mom_slot' = real mem_space model-backed reranker "
                        "(last-hidden cosine); 'mom_readout' = mem_space "
                        "slot-memory cosine reranker (scores question vs the "
                        "round-populated slot bank).")
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
    p.add_argument("--mom_slot_mean_mix", type=float, default=0.0,
                   help="For --reranker mom_readout: blend of mean-cosine into "
                        "the per-round slot relevance score. 0.0 = pure max "
                        "cosine over slots (does this round have any slot that "
                        "matches the question?); 1.0 = pure mean cosine.")
    p.add_argument("--reranker_device", type=str, default="cuda:0",
                   help="Device for the mom_slot mem_space reranker model.")
    # -- notes compressors (--compressor self_notes / mom_notes) ---------- #
    p.add_argument("--compressor", type=str, default="none",
                   choices=["none", "self_notes", "mom_notes"],
                   help="Evidence compressor. 'none' = identity (baseline); "
                        "'self_notes' = the READER'S OWN model generates a "
                        "question-conditioned notes block from the same "
                        "retrieved evidence (B08 leg-1); "
                        "'mom_notes' = mem_space model streams the retrieved "
                        "rounds and generates a short question-conditioned "
                        "notes block PREPENDED to the raw evidence.")
    p.add_argument("--reader_evidence_mode", type=str,
                   default="notes_plus_evidence",
                   choices=["notes_plus_evidence", "notes_only", "evidence_only"],
                   help="What occupies the reader's context when the compressor "
                        "returns a non-empty notes string. 'notes_plus_evidence' "
                        "(default) = notes block PREPENDED to raw evidence (the "
                        "pre-2026-08-16 behaviour, unchanged); 'notes_only' = "
                        "notes block ONLY, raw evidence WITHHELD (B08's "
                        "A-notes-only arm); 'evidence_only' = ignore the notes "
                        "but still pay their generation cost (a cost control, "
                        "not a quality arm).")
    p.add_argument("--notes_cache", type=str, default=None,
                   help="JSONL cache of {question_id, notes} for "
                        "--compressor self_notes. Generate ONCE, then point BOTH "
                        "notes arms at the same file so they see byte-identical "
                        "notes text (part of the frozen single variable).")
    p.add_argument("--notes_cache_readonly", action="store_true",
                   help="With --notes_cache: FAIL instead of generating when a "
                        "question is missing from the cache. Use for the second "
                        "and third arms so notes can never be silently "
                        "regenerated (which would unfreeze the single variable).")
    p.add_argument("--context_log", type=str, default=None,
                   help="JSONL path recording, per item, the EXACT context "
                        "blocks the reader saw plus the hypothesis. Required by "
                        "the U (unsupported-claim) scorer, which is defined "
                        "against 'that arm's own context'.")
    p.add_argument("--compressor_checkpoint", type=str, default=None,
                   help="mem_space adapter .pt checkpoint for --compressor mom_notes.")
    p.add_argument("--compressor_adapter_config", type=str, default=None,
                   help="adapter_config.json (MemorySpaceConfig) for "
                        "--compressor mom_notes.")
    p.add_argument("--compressor_device", type=str, default="cuda:0",
                   help="Device for the mom_notes mem_space compressor model.")
    p.add_argument("--compressor_max_new_tokens", type=int, default=128,
                   help="Max tokens to generate for the MoM notes synopsis.")
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
        examples = _select_question_types(
            examples, args.question_types, args.expect_n
        )

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
