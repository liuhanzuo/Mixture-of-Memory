"""LongMemEval system-memory evaluation harness (Track B).

A modular, official-paper-compatible RAG memory baseline for the
LongMemEval benchmark (Wu et al. 2024, arXiv:2410.10813,
https://github.com/xiaowu0162/LongMemEval).

This package provides a *baseline* retrieval-augmented memory pipeline
(no Mixture-of-Memory yet). It is intentionally structured so that MoM
variants can later be plugged in at well-defined extension points:

  - ``MemoryBackend``  : abstract memory store (insert / query)
  - ``Reranker``       : optional second-stage scorer over candidate evidence
  - ``Reader``         : LLM that answers from retrieved evidence

Planned MoM extension points (NOT implemented here):
  * MoM-reranker     : a learned Reranker that reorders RoundFlatRetriever
                       candidates using the MoM memory representation.
  * MoM-summary-key  : a MemoryBackend whose per-round keys are MoM gist
                       summaries instead of raw text BM25/embedding keys.
  * MoM-compressor   : a MemoryBackend that compresses sessions into a
                       fixed-size MoM buffer and queries that buffer.
  * MoM-hybrid       : RoundFlatRetriever recall + MoM-reranker precision.
"""

from .data import LongMemEvalExample, load_longmemeval, iter_rounds, Round
from .backends import (
    Evidence,
    MemoryBackend,
    Reranker,
    IdentityReranker,
    RoundFlatRetriever,
)
from .reader import Reader, StubReader, OpenAIChatReader, build_reader
from .scoring import (
    write_submission,
    recall_at_k,
    aggregate_recall,
)

__all__ = [
    "LongMemEvalExample",
    "load_longmemeval",
    "iter_rounds",
    "Round",
    "Evidence",
    "MemoryBackend",
    "Reranker",
    "IdentityReranker",
    "RoundFlatRetriever",
    "Reader",
    "StubReader",
    "OpenAIChatReader",
    "build_reader",
    "write_submission",
    "recall_at_k",
    "aggregate_recall",
]
