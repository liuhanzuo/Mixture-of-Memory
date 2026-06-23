"""Scoring: submission format + retrieval recall@k.

Two outputs:

  1. Submission JSONL of ``{"question_id": ..., "hypothesis": ...}`` — the
     format consumed by LongMemEval's official GPT-4o auto-evaluator
     (``evaluate_qa.py`` in the upstream repo). This harness does NOT bundle
     the GPT-4o judge (it needs an API key); it produces the file the judge
     expects.

  2. Retrieval recall@k against ``answer_session_ids`` — a cheap, API-free
     diagnostic for the retrieval stage:
       * session-level recall: did the top-k evidence include at least one
         round from a gold answer session?
       * also reports recall over all gold sessions (fraction covered).

This lets us tune the retriever (bm25 / embedding / union, top_k) without
calling any reader LLM.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Sequence

from .backends import Evidence


def write_submission(path: str, records: Sequence[Dict[str, str]]) -> None:
    """Write JSONL submission: one {question_id, hypothesis} per line."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(
                json.dumps(
                    {
                        "question_id": rec["question_id"],
                        "hypothesis": rec.get("hypothesis", ""),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


@dataclass
class RecallResult:
    question_id: str
    any_hit: bool          # >=1 gold session present in top-k
    covered: float         # fraction of gold sessions present in top-k
    n_gold: int
    n_retrieved_sessions: int


def recall_at_k(
    evidence: List[Evidence],
    answer_session_ids: Sequence[str],
    question_id: str = "",
) -> RecallResult:
    """Compute session-level recall of retrieved evidence vs gold sessions."""
    gold = set(answer_session_ids)
    retrieved_sessions = {e.session_id for e in evidence}
    if not gold:
        # Abstention / no-evidence questions: recall is undefined; report 1.0
        # so they don't drag down the retrieval metric.
        return RecallResult(question_id, True, 1.0, 0, len(retrieved_sessions))
    hits = gold & retrieved_sessions
    return RecallResult(
        question_id=question_id,
        any_hit=len(hits) > 0,
        covered=len(hits) / len(gold),
        n_gold=len(gold),
        n_retrieved_sessions=len(retrieved_sessions),
    )


def aggregate_recall(results: Sequence[RecallResult]) -> Dict[str, float]:
    """Aggregate per-question recall into corpus-level metrics."""
    if not results:
        return {"n": 0, "any_hit_recall": 0.0, "mean_covered": 0.0}
    scored = [r for r in results if r.n_gold > 0]
    n = len(scored)
    if n == 0:
        return {"n": 0, "any_hit_recall": 0.0, "mean_covered": 0.0}
    any_hit = sum(1 for r in scored if r.any_hit) / n
    mean_cov = sum(r.covered for r in scored) / n
    return {
        "n": n,
        "any_hit_recall": any_hit,
        "mean_covered": mean_cov,
    }
