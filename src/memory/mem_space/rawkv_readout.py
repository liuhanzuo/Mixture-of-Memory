"""Raw-KV Readout — Method A (per-chunk raw-KV + emergent trainable gist-key
soft attention). docs/RAWKV_READOUT_PROPOSAL.md §2 Method A. 2026-06-19.

This is the structural successor to the eval-time ``use_inattn_kv`` probe
(``inattn_kv.py``). That probe proved the in-attention raw-KV concat mechanism
is in-graph and multi-layer capable, but its RETRIEVAL is non-differentiable:
stored tokens are scored by the ``TopKSelector`` routing-q under ``no_grad`` +
hard top-k, so NO gradient ever reaches a selection scorer (the same dead-
retriever trap landmark-s5 flagged). Method A replaces that with an EMERGENT,
TRAINABLE gist-key soft attention (the Landmark mechanism), removing the trained
``TopKSelector`` from the read path entirely.

Two pieces:

* :class:`RawKVReadoutStore` — pure runtime per-sequence state (NOT an
  nn.Module; DDP ignores it). As each chunk streams through the write-owner
  layer it appends that chunk's UNCOMPRESSED token hidden states (detached) +
  source positions + a per-chunk *gist source* (the mean-pooled chunk hidden,
  detached). Content is stored raw; nothing is compressed.

* :class:`GistReadout` — a tiny TRAINABLE nn.Module shared across the readout
  layers (registered once on the model root, like ``l3_pool``). It owns the
  query / key projections of the gist scoring space. At read time it recomputes
  the per-chunk gist KEY from the (detached) stored gist source via its trainable
  ``key_proj`` and scores it against the current query via ``query_proj``; the
  resulting per-chunk soft-selection weight is returned as an additive LOG-BIAS
  on the retrieved KV columns (Landmark §4b: cross-block weight = token-score ×
  landmark-score, expressed in additive log space inside the ONE softmax).

The gist projection therefore sits IN the loss graph and gets gradient every
time the read fires — the query carries grad, the projections are trainable, and
the bias flows through the native attention softmax into the loss. The stored
gist source is detached (recomputing the key through ``key_proj`` is the trick
the L3 pool uses to keep a projection's gradient path clean under the streamed-
context + detach training regime).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RawKVReadoutStore:
    """Per-sequence raw-KV + per-chunk gist-source store. NOT an nn.Module.

    All tensors are detached on write (pure runtime state, mirrors
    ``ChunkMemoryBank`` / the rawkv channel on ``MemoryBank``). Reset at the
    document / rollout boundary.

    State (lazily materialised on the first append; None when empty):
        token_hidden : [B, M, d]    detached raw token hidden states
        token_pos    : [B, M]  long source in-chunk RoPE positions
        token_chunk  : [B, M]  long which chunk each token belongs to (0..C-1)
        gist_src     : [B, C, d]    detached per-chunk pooled hidden (gist source)
    M grows by the chunk length each append; C grows by 1.
    """

    def __init__(self) -> None:
        self.token_hidden: Optional[torch.Tensor] = None
        self.token_pos: Optional[torch.Tensor] = None
        self.token_chunk: Optional[torch.Tensor] = None
        self.gist_src: Optional[torch.Tensor] = None
        self.n_chunks: int = 0

    # ------------------------------------------------------------------ #
    # WRITE
    # ------------------------------------------------------------------ #
    def append_chunk(
        self,
        token_hidden: torch.Tensor,
        token_pos: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
        pool: str = "mean",
    ) -> None:
        """Append one chunk's raw token hidden states + derive its gist source.

        Args:
            token_hidden: [B, T, d] this chunk's UNCOMPRESSED token hidden
                states (the pre-LN layer input — same representation
                ``build_retrieved_kv`` expects). Detached on write.
            token_pos: [B, T] long source in-chunk positions (RoPE phase). When
                None defaults to ``0..T-1``.
            token_mask: optional [B, T] bool — True == real token (False ==
                pad). When provided the gist source mean-pools over real tokens
                only and pad tokens are still stored (their score will be low)
                — kept simple for the MVP.
            pool: per-chunk gist-source pooling. "mean" (original; dilutes a
                small needle in a large chunk) or "max" (element-wise max over
                chunk tokens; the salient token survives → anti-dilution, H1-fix
                2026-06-20).
        """
        if token_hidden.dim() != 3:
            return
        with torch.no_grad():
            B, T, d = token_hidden.shape
            dev = token_hidden.device
            _h = token_hidden.detach()
            if token_pos is not None:
                _p = token_pos.to(device=dev, dtype=torch.long)
                if _p.dim() == 1:
                    _p = _p.unsqueeze(0).expand(B, -1)
            else:
                _p = (
                    torch.arange(T, device=dev, dtype=torch.long)
                    .unsqueeze(0)
                    .expand(B, -1)
                )
            _cid = torch.full(
                (B, T), self.n_chunks, device=dev, dtype=torch.long
            )
            # Per-chunk gist source, detached. mean (default) or max pooling.
            if pool == "max":
                if token_mask is not None and token_mask.shape == (B, T):
                    # Mask pads to -inf so they never win the max.
                    _neg = torch.finfo(_h.dtype).min
                    _mh = _h.masked_fill(
                        ~token_mask.to(device=dev).unsqueeze(-1), _neg
                    )
                    _gsrc = _mh.max(dim=1).values                # [B, d]
                else:
                    _gsrc = _h.max(dim=1).values                 # [B, d]
            else:
                if token_mask is not None and token_mask.shape == (B, T):
                    _m = token_mask.to(device=dev, dtype=_h.dtype).unsqueeze(-1)
                    _denom = _m.sum(dim=1).clamp_min(1.0)
                    _gsrc = (_h * _m).sum(dim=1) / _denom        # [B, d]
                else:
                    _gsrc = _h.mean(dim=1)                       # [B, d]
            _gsrc = _gsrc.unsqueeze(1)                           # [B, 1, d]

            if (
                self.token_hidden is None
                or self.token_hidden.shape[0] != B
                or self.token_hidden.shape[-1] != d
            ):
                self.token_hidden = _h
                self.token_pos = _p
                self.token_chunk = _cid
                self.gist_src = _gsrc
                self.n_chunks = 1
            else:
                self.token_hidden = torch.cat([self.token_hidden, _h], dim=1)
                self.token_pos = torch.cat([self.token_pos, _p], dim=1)
                self.token_chunk = torch.cat([self.token_chunk, _cid], dim=1)
                self.gist_src = torch.cat([self.gist_src, _gsrc], dim=1)
                self.n_chunks += 1

    def size(self) -> int:
        return 0 if self.token_hidden is None else int(self.token_hidden.shape[1])

    def reset(self) -> None:
        self.token_hidden = None
        self.token_pos = None
        self.token_chunk = None
        self.gist_src = None
        self.n_chunks = 0


class GistReadout(nn.Module):
    """Trainable gist-key soft-attention scorer (Landmark-style, emergent).

    Shared singleton (one per model, registered on the root like ``l3_pool``).
    Owns the gist scoring space projections. NO STE, NO load-balance, NO separate
    per-slot selector head — just two small trainable projections + a soft
    (optionally soft-top-k) attention over the per-chunk gist keys.

    The scoring is intentionally minimal so the only thing learned is *which
    chunk is relevant to the current query* — the content comes from the raw KV
    (re-projected through the reader's own native k/v_proj), not from here.
    """

    def __init__(self, d_model: int, gist_dim: int = 128,
                 inwindow_summary: bool = False) -> None:
        super().__init__()
        self.d_model = d_model
        self.gist_dim = gist_dim
        self.query_proj = nn.Linear(d_model, gist_dim, bias=False)
        self.key_proj = nn.Linear(d_model, gist_dim, bias=False)
        # Small-random init: the read path is active (produces a non-uniform
        # selection) from step 0 so gradient flows immediately, but not so large
        # it swamps the native softmax before the scorer learns.
        nn.init.normal_(self.query_proj.weight, std=0.02)
        nn.init.normal_(self.key_proj.weight, std=0.02)
        self._scale = gist_dim ** -0.5
        # (B in-window summary, 2026-06-20) The "summarizer": maps a per-sub-block
        # pooled hidden to a SUMMARY hidden that the reader's own k_proj/v_proj
        # then project into the attention space (summary_key=k_proj(summary_proj(
        # pool)), summary_val=v_proj(pool) — value uses raw pool, Landmark-faithful:
        # the landmark token's value goes through the standard v_proj, the summary
        # capacity lives in the hidden/key). summary_proj is the ONLY new trainable
        # param; it learns to compress a sub-block into a selectable summary via the
        # in-window bottleneck objective (dense gradient from the target chunk's LM
        # loss; gradient source 2). Allocated ONLY when inwindow_summary is on so
        # the off path keeps a byte-identical state_dict (no extra params).
        self.inwindow_summary = bool(inwindow_summary)
        if self.inwindow_summary:
            self.summary_proj = nn.Linear(d_model, d_model, bias=False)
            # init near identity so the un-trained summary ≈ mean-pool (a sane
            # starting summary) and gradient can shape it from there.
            nn.init.eye_(self.summary_proj.weight)
        else:
            self.summary_proj = None
        # Diagnostics (refreshed each retrieve; layer/smoke reads these).
        self._last_n_chunks: int = 0
        self._last_R: int = 0
        self._last_weight_max: float = 0.0
        self._last_weight_entropy: float = 0.0

    def summarize(self, sub_block_hidden: torch.Tensor) -> torch.Tensor:
        """Per-sub-block SUMMARY hidden for the in-window bottleneck (B4).

        Args:
            sub_block_hidden: [..., n_sub, d_model] the per-sub-block POOLED hidden
                (mean over the sub-block's tokens). Grad-bearing when computed in
                the target chunk's forward (gradient source 2 flows here).
        Returns:
            [..., n_sub, d_model] summary hidden. The caller (inattn_kv bottleneck)
            applies the reader's k_proj to get the summary KEY and the reader's
            v_proj to the RAW pool to get the summary VALUE (Landmark-faithful).
        When inwindow_summary is off, returns the input unchanged (no-op).
        """
        if self.summary_proj is None:
            return sub_block_hidden
        return self.summary_proj(sub_block_hidden)


    def retrieve(
        self,
        query_hidden: torch.Tensor,
        store: "RawKVReadoutStore",
        topk_chunks: int = 0,
        temperature: float = 1.0,
        disable_col_bias: bool = False,
        keep_set_override: Optional[torch.Tensor] = None,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Differentiable gist soft-attention retrieval.

        Args:
            query_hidden: [B, Tq, d] current chunk's query hidden states
                (grad-bearing — the layer input at the readout layer).
            store: the per-sequence :class:`RawKVReadoutStore`.
            topk_chunks: soft-top-k — keep this many highest-scoring chunks per
                query token; 0 or >= n_chunks → keep all (pure soft attention).
                The WEIGHTS on the kept chunks stay differentiable.
            temperature: softmax temperature for the gist selection.

        Returns:
            (retrieved_hidden, retrieved_pos, col_bias) or None when the store
            is empty / shape-mismatched.
              retrieved_hidden : [B, R, d]   detached raw token hidden (content)
              retrieved_pos    : [B, R]  long source positions
              col_bias         : [B, Tq, R] additive LOG-weight bias per
                  (query-token, retrieved-token) — = log(soft-selection weight of
                  the chunk that retrieved token belongs to). Grad-bearing.
            R = total tokens of the kept chunks.
        """
        if (
            store.token_hidden is None
            or store.gist_src is None
            or query_hidden.dim() != 3
        ):
            return None
        B, M, d = store.token_hidden.shape
        if query_hidden.shape[0] != B or M == 0:
            return None
        C = store.gist_src.shape[1]
        if C == 0:
            return None

        dev = query_hidden.device
        # Recompute gist KEY from the DETACHED stored source via trainable proj
        # (keeps key_proj in the grad graph). [B, C, gist_dim].
        gkey = self.key_proj(store.gist_src.to(dev, dtype=query_hidden.dtype))
        gq = self.query_proj(query_hidden)                       # [B, Tq, gist]
        # Per-(query token, chunk) score. [B, Tq, C].
        score = torch.einsum("bqg,bcg->bqc", gq, gkey) * self._scale
        score = score / max(temperature, 1e-6)

        # Soft-top-k: choose the kept chunk set by the per-query MAX score (so a
        # single kept set is shared across query tokens → bounded R), but keep
        # the WEIGHTS over the kept chunks fully differentiable (softmax of the
        # per-query-token scores restricted to the kept set).
        keep_all = topk_chunks <= 0 or topk_chunks >= C
        if keep_set_override is not None and keep_set_override.numel() > 0:
            # Explicit kept-chunk indices chosen by the CALLER (e.g. reader-attn
            # salience or oracle), bypassing the gist-salience selection. HARD
            # isolation: only these chunks' tokens enter attention (the others
            # are physically gathered out below), so there is NO dilution from
            # the excluded chunks. This is the keep_set_mode={reader_attn,oracle}
            # path (2026-06-20 dilution fix).
            kept = keep_set_override.to(dev).long().clamp_(0, C - 1)
            kept = torch.unique(kept)
            kept, _ = torch.sort(kept)
        elif keep_all:
            kept = torch.arange(C, device=dev)
        else:
            chunk_sal = score.max(dim=1).values                  # [B, C]
            # Use the batch-mean salience so the kept set is consistent across
            # the batch (the store layout is shared per-sequence; for B>1 this is
            # an MVP simplification, fine for the architecture validation).
            sal = chunk_sal.mean(dim=0)                           # [C]
            kept = torch.topk(sal, k=topk_chunks, dim=0).indices  # [k]
            kept, _ = torch.sort(kept)

        score_k = score[:, :, kept]                              # [B, Tq, Ck]
        w_k = torch.softmax(score_k, dim=-1)                     # differentiable
        logw_k = torch.log(w_k.clamp_min(1e-9))                  # [B, Tq, Ck]

        # Gather the tokens of the kept chunks. token_chunk: [B, M].
        kept_set = kept.to(dev)
        # Build a [B, M] bool mask of tokens whose chunk is kept.
        tok_kept = (
            store.token_chunk.to(dev).unsqueeze(-1) == kept_set.view(1, 1, -1)
        ).any(dim=-1)                                            # [B, M]
        # For a shared store the per-sequence chunk layout is identical, so the
        # kept-token count is the same across batch; gather per-batch to be safe.
        # MVP: assume uniform chunk length → identical counts; take row 0's count.
        idx_list = tok_kept[0].nonzero(as_tuple=False).squeeze(-1)  # [R]
        if idx_list.numel() == 0:
            return None
        R = int(idx_list.numel())
        gather_idx = idx_list.view(1, R).expand(B, R)            # [B, R]
        ret_h = store.token_hidden.gather(
            1, gather_idx.unsqueeze(-1).expand(B, R, d)
        )                                                        # [B, R, d]
        ret_pos = store.token_pos.gather(1, gather_idx)          # [B, R]
        ret_chunk = store.token_chunk.gather(1, gather_idx)      # [B, R]

        # Map each retrieved token's chunk id → its column index within `kept`,
        # then look up that chunk's per-query-token log weight.
        # kept is sorted; build a dense [C] -> kept-column map.
        col_of_chunk = torch.full((C,), -1, device=dev, dtype=torch.long)
        col_of_chunk[kept_set] = torch.arange(
            kept_set.numel(), device=dev, dtype=torch.long
        )
        ret_col = col_of_chunk[ret_chunk]                        # [B, R] in 0..Ck-1
        # col_bias[b, q, r] = logw_k[b, q, ret_col[b, r]].
        Tq = query_hidden.shape[1]
        ret_col_exp = ret_col.unsqueeze(1).expand(B, Tq, R)      # [B, Tq, R]
        col_bias = torch.gather(logw_k, 2, ret_col_exp)          # [B, Tq, R]

        # ABLATION (2026-06-20, go/no-go): zero the trained selection bias so the
        # reader attends the kept raw-KV columns through its OWN native q·k only
        # (no trained gist weight in the softmax). Tests whether the reader's
        # native attention over raw-KV (probed at 55% needle precision) is itself
        # the retriever — i.e. the trained scorer is unnecessary. With
        # topk_chunks>=C this keeps ALL chunks → pure reader attention over all
        # historical raw-KV. col_bias is detached-zero so no grad path either.
        if disable_col_bias:
            col_bias = torch.zeros_like(col_bias)

        # Diagnostics.
        with torch.no_grad():
            self._last_n_chunks = int(C)
            self._last_R = int(R)
            _wmean = w_k.mean(dim=(0, 1))
            self._last_weight_max = float(w_k.max().item())
            _p = _wmean.clamp_min(1e-9)
            self._last_weight_entropy = float(-(_p * _p.log()).sum().item())

        return ret_h.detach(), ret_pos, col_bias
