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
from dataclasses import dataclass, replace
from datetime import date
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


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "did", "do", "for", "i",
    "in", "is", "it", "me", "my", "of", "on", "or", "the", "to", "was",
    "were", "what", "when", "where", "which", "who", "why", "with",
}
_RECENCY_CUES = {
    "current", "currently", "last", "latest", "newest", "now", "recent",
    "recently", "today", "updated",
}
_EARLIEST_CUES = {"earliest", "first", "initial", "initially", "oldest", "original"}


def _content_tokens(text: str) -> List[str]:
    return [t for t in simple_tokenize(text) if t not in _STOPWORDS]


def _safe_date_ordinal(value: str) -> Optional[int]:
    match = re.search(r"\d{4}-\d{2}-\d{2}", value or "")
    if not match:
        return None
    try:
        return date.fromisoformat(match.group(0)).toordinal()
    except ValueError:
        return None


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


class KeywordOverlapReranker(Reranker):
    """Cheap lexical second-stage reranker.

    This is intentionally dependency-free: it boosts candidate rounds that cover
    rare-ish content words from the question while retaining a small normalized
    first-stage score contribution for tie-breaking.
    """

    def __init__(self, keyword_weight: float = 1.0, base_weight: float = 0.15):
        self.keyword_weight = keyword_weight
        self.base_weight = base_weight

    @staticmethod
    def _normalized_base_scores(candidates: List[Evidence]) -> List[float]:
        if not candidates:
            return []
        scores = [ev.score for ev in candidates]
        lo, hi = min(scores), max(scores)
        if hi <= lo:
            return [1.0] * len(scores)
        return [(s - lo) / (hi - lo) for s in scores]

    def _keyword_score(self, query_terms: Sequence[str], evidence: Evidence) -> float:
        if not query_terms:
            return 0.0
        text_terms = set(_content_tokens(evidence.text))
        if not text_terms:
            return 0.0
        overlap = sum(1 for term in query_terms if term in text_terms)
        return overlap / max(len(set(query_terms)), 1)

    def rerank(
        self, question: str, date: str, candidates: List[Evidence], top_k: int
    ) -> List[Evidence]:
        query_terms = _content_tokens(question)
        base_scores = self._normalized_base_scores(candidates)
        scored = []
        for rank, ev in enumerate(candidates):
            score = (
                self.keyword_weight * self._keyword_score(query_terms, ev)
                + self.base_weight * base_scores[rank]
            )
            scored.append((score, -rank, replace(ev, score=score)))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [ev for _, _, ev in scored[:top_k]]


class TemporalAwareReranker(KeywordOverlapReranker):
    """Keyword reranker with a temporal prior for recency/earliest questions."""

    def __init__(
        self,
        keyword_weight: float = 0.5,
        base_weight: float = 0.1,
        temporal_weight: float = 1.0,
    ):
        super().__init__(keyword_weight=keyword_weight, base_weight=base_weight)
        self.temporal_weight = temporal_weight

    def _temporal_direction(self, question: str) -> int:
        terms = set(simple_tokenize(question))
        if terms & _RECENCY_CUES:
            return 1
        if terms & _EARLIEST_CUES:
            return -1
        return 0

    def rerank(
        self, question: str, date: str, candidates: List[Evidence], top_k: int
    ) -> List[Evidence]:
        direction = self._temporal_direction(question)
        if direction == 0:
            return super().rerank(question, date, candidates, top_k)

        query_terms = _content_tokens(question)
        base_scores = self._normalized_base_scores(candidates)
        ordinals = [_safe_date_ordinal(ev.session_date) for ev in candidates]
        known = [o for o in ordinals if o is not None]
        lo, hi = (min(known), max(known)) if known else (0, 0)
        span = max(hi - lo, 1)

        scored = []
        for rank, ev in enumerate(candidates):
            ordinal = ordinals[rank]
            temporal = 0.0
            if ordinal is not None:
                temporal = (ordinal - lo) / span
                if direction < 0:
                    temporal = 1.0 - temporal
            score = (
                self.keyword_weight * self._keyword_score(query_terms, ev)
                + self.base_weight * base_scores[rank]
                + self.temporal_weight * temporal
            )
            scored.append((score, -rank, replace(ev, score=score)))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [ev for _, _, ev in scored[:top_k]]


class MoMSlotReranker(KeywordOverlapReranker):
    """Placeholder adapter for a future MoM slot/query-similarity reranker.

    The intended production path is:
      1. Encode each candidate round into MoM summary/slot keys during indexing.
      2. Encode the LongMemEval question into a MoM query representation.
      3. Rerank candidates by slot-query similarity, optionally fused with the
         first-stage BM25/embedding score and temporal priors.

    This stub deliberately does not load a GPU model or checkpoint. For recall
    sweeps today it behaves as a keyword-overlap proxy while preserving the same
    Reranker seam that real MoM inference can replace later. The real,
    model-backed reranker is :class:`MoMModelReranker` below (``--reranker
    mom_slot``); this stub remains the cheap, dependency-free proxy
    (``--reranker mom_stub``).
    """

    pass


class MoMModelReranker(Reranker):
    """Real Mixture-of-Memory (mem_space) precision reranker.

    This is the first *model-backed* MoM reranker for the LongMemEval Track B
    harness. It does NOT do recall: it only re-orders an existing first-stage
    (BM25) candidate pool, testing whether the mem_space memory representation
    can rank true-evidence rounds above lexical false positives that BM25
    surfaces.

    Scoring (v1, defensible + genuinely model-backed)
    -------------------------------------------------
    The mem_space model is a Llama-3-8B backbone patched with
    ``MemorySpaceLayer``s: every decoder forward reads from / writes to a
    fixed-size slot memory bank, so the model's last-layer hidden state already
    reflects the memory readout (not just raw self-attention). We use this
    patched model as a *contextual encoder*:

      * ``_encode(text)`` resets the memory bank, streams the text through the
        bank chunk-by-chunk (``chunk_size``), and mean-pools the final forward's
        last-layer hidden state over the real (non-pad) tokens, L2-normalized.
      * The question is encoded once; each candidate round is encoded the same
        way. The MoM relevance score is the cosine similarity between the
        question vector and the candidate-round vector.

    This is the "last-hidden-state cosine" v1 explicitly sanctioned by the task:
    it is simple, runs on 1 GPU at batch_size 1, and genuinely exercises the
    mem_space model (memory layers are active in every forward), rather than
    falling back to BM25 / keyword overlap.

    Fusion with BM25
    ----------------
    Candidates arrive pre-sorted by the first-stage BM25 score. We fuse the
    (min-max normalized) MoM cosine score with the (min-max normalized) BM25
    score via a convex combination controlled by ``fusion_weight`` w:

        final = w * mom_norm + (1 - w) * bm25_norm

    ``w = 1.0`` -> pure MoM reranking; ``w = 0.0`` -> identity (BM25 order). The
    default ``0.5`` blends both. Ties break toward the original BM25 rank.
    """

    def __init__(
        self,
        checkpoint: str,
        adapter_config: str,
        model_path: str = "models/Meta-Llama-3-8B",
        device: str = "cuda:0",
        fusion_weight: float = 0.5,
        chunk_size: int = 512,
        dtype: str = "bfloat16",
        max_text_tokens: int = 2048,
    ):
        if not (0.0 <= fusion_weight <= 1.0):
            raise ValueError(f"fusion_weight must be in [0, 1]; got {fusion_weight}")
        self.checkpoint = checkpoint
        self.adapter_config = adapter_config
        self.model_path = model_path
        self.device_str = device
        self.fusion_weight = fusion_weight
        self.chunk_size = chunk_size
        self.dtype_str = dtype
        self.max_text_tokens = max_text_tokens

        # Heavy imports + model load happen once, at construction.
        import json

        import torch  # noqa: F401
        from transformers import AutoTokenizer

        # Reuse the BABILong mem_space loader so checkpoint/config handling stays
        # in one place (DDP-prefix stripping, RoPE fp32 snapshot, Fix J, etc.).
        from scripts.run_babilong_mem_space import (
            build_mem_space_config,
            generate_with_mem_space,  # noqa: F401  (kept for parity/debug)
            load_mem_space_model,
            _freeze_banks,
            _reset_banks,
            _reset_l2,
            _unfreeze_banks,
        )

        self._torch = torch
        self._reset_banks = _reset_banks
        self._reset_l2 = _reset_l2
        self._freeze_banks = _freeze_banks
        self._unfreeze_banks = _unfreeze_banks

        self._device = torch.device(device)
        self._dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[dtype]

        with open(adapter_config, "r") as f:
            adapter_cfg = json.load(f)
        mem_config = build_mem_space_config(adapter_cfg)
        # L3 token-recon head sizes pos_queries to chunk_size at train time;
        # mirror it here so the loaded ckpt's shapes line up.
        mem_config.l3_recon_max_positions = chunk_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = load_mem_space_model(
            model_path=model_path,
            checkpoint_path=checkpoint,
            mem_config=mem_config,
            device=self._device,
            dtype=self._dtype,
        )
        self.model.eval()
        self.loaded = True  # truthy marker so the CLI can assert non-degraded

    # -- mem_space contextual encoding ------------------------------------- #
    def _encode(self, text: str):
        """Stream ``text`` through the memory bank and return an L2-normalized
        mean-pooled last-hidden-state vector (numpy float32, shape (H,))."""
        torch = self._torch
        ids = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_text_tokens,
            return_tensors="pt",
        )["input_ids"][0].to(self._device)
        if ids.numel() == 0:
            ids = self.tokenizer(" ", return_tensors="pt")["input_ids"][0].to(self._device)

        self._reset_banks(self.model)
        self._reset_l2(self.model)

        chunks = list(ids.split(self.chunk_size))
        with torch.no_grad():
            # Stream all-but-last chunk so the bank accumulates the full text.
            if len(chunks) > 1:
                for chunk in chunks[:-1]:
                    self.model(input_ids=chunk.unsqueeze(0), use_cache=False)
            # Freeze the bank during the readout forward so it isn't polluted.
            self._freeze_banks(self.model)
            try:
                last = chunks[-1].unsqueeze(0)
                out = self.model(
                    input_ids=last, use_cache=False, output_hidden_states=True
                )
                hidden = out.hidden_states[-1][0]  # (T, H)
                vec = hidden.mean(dim=0)            # (H,)
                vec = torch.nn.functional.normalize(vec.float(), p=2, dim=0)
            finally:
                self._unfreeze_banks(self.model)
        return vec.detach().cpu().numpy()

    @staticmethod
    def _minmax(values: List[float]) -> List[float]:
        if not values:
            return []
        lo, hi = min(values), max(values)
        if hi <= lo:
            return [1.0] * len(values)
        return [(v - lo) / (hi - lo) for v in values]

    def rerank(
        self, question: str, date: str, candidates: List[Evidence], top_k: int
    ) -> List[Evidence]:
        if not candidates:
            return []
        import numpy as np

        q_vec = self._encode(question)
        mom_scores = []
        for ev in candidates:
            c_vec = self._encode(ev.text)
            mom_scores.append(float(np.dot(q_vec, c_vec)))

        bm25_scores = [ev.score for ev in candidates]
        mom_norm = self._minmax(mom_scores)
        bm25_norm = self._minmax(bm25_scores)
        w = self.fusion_weight

        scored = []
        for rank, ev in enumerate(candidates):
            fused = w * mom_norm[rank] + (1.0 - w) * bm25_norm[rank]
            scored.append((fused, -rank, replace(ev, score=fused)))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [ev for _, _, ev in scored[:top_k]]


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
