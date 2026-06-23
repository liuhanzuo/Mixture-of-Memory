"""Memory backends and retrieval for the LongMemEval baseline.

This module defines the abstract memory interface and a concrete
``RoundFlatRetriever`` baseline. The retriever indexes ROUND-level units
(user+assistant pairs) with:

  * BM25 (pure-python Okapi BM25; no ``rank_bm25`` dependency required)
  * dense embedding similarity (sentence-transformers if available, else a
    local HuggingFace encoder such as ``models/bge-m3`` via ``transformers``;
    the embedding backend is OPTIONAL and degrades to BM25-only if no encoder
    can be loaded).

Extension points for future Mixture-of-Memory work:

  * ``MemoryBackend``  - subclass to implement MoM-summary-key (keys are MoM
    gist summaries) or MoM-compressor (sessions compressed into a fixed-size
    MoM buffer that ``query`` reads from).
  * ``Reranker``       - subclass to implement MoM-reranker (reorder candidate
    Evidence using the MoM memory representation). Plug into RoundFlatRetriever
    via the ``reranker`` argument for an MoM-hybrid (sparse/dense recall +
    MoM precision) baseline.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Sequence

from .data import LongMemEvalExample, Round, iter_rounds


@dataclass
class Evidence:
    """A retrieved memory unit returned to the reader."""

    round_id: str
    session_id: str
    session_date: str
    text: str
    score: float

    def as_block(self, idx: int) -> str:
        """Render as a structured evidence block for the reader prompt."""
        date = self.session_date or "unknown-date"
        return (
            f"[Evidence {idx} | session={self.session_id} | date={date}]\n"
            f"{self.text}"
        )


# --------------------------------------------------------------------------- #
# Tokenization (shared by BM25; deliberately simple + dependency-free)
# --------------------------------------------------------------------------- #
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def simple_tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


# --------------------------------------------------------------------------- #
# Abstract interfaces
# --------------------------------------------------------------------------- #
class MemoryBackend(ABC):
    """Abstract long-term memory store.

    A backend ingests an interactive history (``insert``) and answers a
    question by returning bounded evidence (``query``). Concrete MoM variants
    will subclass this:

      * MoM-summary-key : ``insert`` stores MoM gist summaries as retrieval keys
      * MoM-compressor  : ``insert`` compresses into a fixed-size MoM buffer;
                          ``query`` reads from that buffer.
    """

    @abstractmethod
    def insert(self, unit) -> None:
        """Add one memory unit (a :class:`Round` for the baseline)."""

    @abstractmethod
    def query(self, question: str, date: str = "", top_k: int = 10) -> List[Evidence]:
        """Return up to ``top_k`` evidence units for the question."""

    def reset(self) -> None:
        """Clear all inserted units (per-question harnesses re-index)."""
        raise NotImplementedError


class Reranker(ABC):
    """Second-stage scorer over candidate evidence.

    Baseline ships :class:`IdentityReranker`. The planned MoM-reranker will
    reorder candidates using the MoM memory representation (e.g. attention
    over a fixed-size memory buffer keyed by the question).
    """

    @abstractmethod
    def rerank(
        self, question: str, date: str, candidates: List[Evidence], top_k: int
    ) -> List[Evidence]:
        ...


class IdentityReranker(Reranker):
    """No-op reranker: keeps first-stage order, truncates to ``top_k``."""

    def rerank(
        self, question: str, date: str, candidates: List[Evidence], top_k: int
    ) -> List[Evidence]:
        return candidates[:top_k]


# --------------------------------------------------------------------------- #
# Pure-python Okapi BM25
# --------------------------------------------------------------------------- #
class _BM25:
    def __init__(self, corpus_tokens: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_tokens = [list(t) for t in corpus_tokens]
        self.n_docs = len(self.corpus_tokens)
        self.doc_len = [len(d) for d in self.corpus_tokens]
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0
        self.doc_freqs: List[Counter] = [Counter(d) for d in self.corpus_tokens]
        df: Counter = Counter()
        for df_doc in self.doc_freqs:
            for term in df_doc:
                df[term] += 1
        # Okapi BM25 idf with +1 smoothing (always positive).
        self.idf = {
            term: math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))
            for term, n in df.items()
        }

    def scores(self, query_tokens: Sequence[str]):
        scores = [0.0] * self.n_docs
        for term in query_tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, freq_counter in enumerate(self.doc_freqs):
                f = freq_counter.get(term, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (
                    1 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1.0)
                )
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        return scores


# --------------------------------------------------------------------------- #
# Optional dense embedding encoder
# --------------------------------------------------------------------------- #
class _Embedder:
    """Lazy dense encoder. Tries sentence-transformers, then a local HF model.

    Returns None-capable: if no encoder can be loaded the caller must fall
    back to BM25-only. Embeddings are L2-normalized so dot == cosine.
    """

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None, batch_size: int = 32):
        self.model_path = model_path
        self.batch_size = batch_size
        self._backend = None  # "st" | "hf"
        self._st_model = None
        self._hf_tok = None
        self._hf_model = None
        self._device = device
        self._init(model_path, device)

    def _init(self, model_path, device):
        # 1) sentence-transformers (preferred if installed)
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._st_model = SentenceTransformer(model_path or "BAAI/bge-m3", device=device)
            self._backend = "st"
            return
        except Exception:
            pass

        # 2) raw transformers encoder (e.g. local models/bge-m3)
        try:
            import torch  # noqa: F401
            from transformers import AutoModel, AutoTokenizer  # type: ignore

            if not model_path:
                raise RuntimeError("no embedding model_path provided for HF fallback")
            self._hf_tok = AutoTokenizer.from_pretrained(model_path)
            self._hf_model = AutoModel.from_pretrained(model_path)
            self._hf_model.eval()
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self._device = device
            self._hf_model.to(device)
            self._backend = "hf"
            return
        except Exception as e:
            self._backend = None
            self._init_error = repr(e)

    @property
    def available(self) -> bool:
        return self._backend is not None

    def encode(self, texts: List[str]):
        import numpy as np

        if not self.available:
            raise RuntimeError("embedding backend unavailable")

        if self._backend == "st":
            emb = self._st_model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return np.asarray(emb, dtype="float32")

        # HF mean-pooling + L2 norm
        import torch

        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            enc = self._hf_tok(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                model_out = self._hf_model(**enc)
                last = model_out.last_hidden_state  # (B, T, H)
                mask = enc["attention_mask"].unsqueeze(-1).float()
                summed = (last * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                mean = summed / counts
                mean = torch.nn.functional.normalize(mean, p=2, dim=1)
                out.append(mean.cpu().float().numpy())
        return np.concatenate(out, axis=0)


# --------------------------------------------------------------------------- #
# Baseline retriever
# --------------------------------------------------------------------------- #
class RoundFlatRetriever(MemoryBackend):
    """Flat round-level retriever with BM25 / embedding / union scoring.

    Args:
        mode: "bm25" | "embedding" | "union" (reciprocal-rank fusion of both).
        embed_model_path: local HF / sentence-transformers model. If the
            embedding backend can't be loaded, embedding/union modes
            transparently degrade to BM25-only (``degraded`` is set True).
        reranker: optional second-stage :class:`Reranker` (MoM-reranker hook).
        rrf_k: constant for reciprocal-rank fusion in union mode.
        candidate_multiplier: first-stage pool size = top_k * this, before
            reranking.
    """

    def __init__(
        self,
        mode: str = "bm25",
        embed_model_path: Optional[str] = None,
        device: Optional[str] = None,
        reranker: Optional[Reranker] = None,
        rrf_k: int = 60,
        candidate_multiplier: int = 4,
    ):
        if mode not in ("bm25", "embedding", "union"):
            raise ValueError(f"unknown retriever mode: {mode}")
        self.mode = mode
        self.embed_model_path = embed_model_path
        self.device = device
        self.reranker = reranker or IdentityReranker()
        self.rrf_k = rrf_k
        self.candidate_multiplier = candidate_multiplier

        self.degraded = False
        self.degraded_reason = ""

        self._rounds: List[Round] = []
        self._bm25: Optional[_BM25] = None
        self._embedder: Optional[_Embedder] = None
        self._round_embeddings = None  # np.ndarray (N, H)

        if mode in ("embedding", "union"):
            self._embedder = _Embedder(embed_model_path, device=device)
            if not self._embedder.available:
                self.degraded = True
                self.degraded_reason = (
                    "embedding backend unavailable (no sentence-transformers / "
                    "could not load HF encoder); falling back to BM25-only"
                )

    @property
    def effective_mode(self) -> str:
        if self.degraded and self.mode in ("embedding", "union"):
            return "bm25"
        return self.mode

    # -- ingestion ---------------------------------------------------------- #
    def reset(self) -> None:
        self._rounds = []
        self._bm25 = None
        self._round_embeddings = None

    def insert(self, unit: Round) -> None:
        self._rounds.append(unit)
        # Indexes are (re)built lazily in ``_build``.
        self._bm25 = None
        self._round_embeddings = None

    def index_example(self, example: LongMemEvalExample) -> int:
        """Convenience: reset and insert all rounds of one example."""
        self.reset()
        n = 0
        for r in iter_rounds(example):
            self.insert(r)
            n += 1
        self._build()
        return n

    def _build(self) -> None:
        texts = [r.text for r in self._rounds]
        if self.effective_mode in ("bm25", "union") and self._bm25 is None:
            self._bm25 = _BM25([simple_tokenize(t) for t in texts])
        if self.effective_mode in ("embedding", "union") and self._round_embeddings is None:
            if self._embedder is not None and self._embedder.available:
                if texts:
                    self._round_embeddings = self._embedder.encode(texts)

    # -- retrieval ---------------------------------------------------------- #
    def _bm25_ranking(self, question: str) -> List[int]:
        if self._bm25 is None:
            return []
        scores = self._bm25.scores(simple_tokenize(question))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return order

    def _embed_ranking(self, question: str) -> List[int]:
        if self._round_embeddings is None or self._embedder is None:
            return []
        import numpy as np

        q = self._embedder.encode([question])[0]
        sims = self._round_embeddings @ q
        order = list(np.argsort(-sims))
        return [int(i) for i in order]

    def query(self, question: str, date: str = "", top_k: int = 10) -> List[Evidence]:
        if not self._rounds:
            return []
        self._build()
        pool = max(top_k * self.candidate_multiplier, top_k)
        mode = self.effective_mode

        if mode == "bm25":
            scores = self._bm25.scores(simple_tokenize(question)) if self._bm25 else []
            order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            ranked = [(i, float(scores[i])) for i in order[:pool]]
        elif mode == "embedding":
            import numpy as np

            q = self._embedder.encode([question])[0]
            sims = self._round_embeddings @ q
            order = list(np.argsort(-sims))[:pool]
            ranked = [(int(i), float(sims[i])) for i in order]
        else:  # union via reciprocal-rank fusion
            bm_order = self._bm25_ranking(question)
            em_order = self._embed_ranking(question)
            rrf = {}
            for rank, i in enumerate(bm_order):
                rrf[i] = rrf.get(i, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            for rank, i in enumerate(em_order):
                rrf[i] = rrf.get(i, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            order = sorted(rrf, key=lambda i: rrf[i], reverse=True)
            ranked = [(i, rrf[i]) for i in order[:pool]]

        candidates = [
            Evidence(
                round_id=self._rounds[i].round_id,
                session_id=self._rounds[i].session_id,
                session_date=self._rounds[i].session_date,
                text=self._rounds[i].text,
                score=score,
            )
            for i, score in ranked
        ]
        return self.reranker.rerank(question, date, candidates, top_k)
