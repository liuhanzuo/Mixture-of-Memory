"""QCMem mid-depth resume orchestrator over a plain Llama backbone.

See ``src/memory/qcmem/__init__.py`` for the high-level contract. This file
holds the ``QCMemModel`` class that implements the write/read primitive proven
by ``scripts/qcmem_resume_primitive_check.py``, scaled from the tiny random
model to a real Llama-3-8B.

Design notes
------------
* We never patch or mutate the backbone. We only *read* off it:
  ``model.model.embed_tokens``, ``model.model.layers``, ``model.model.norm``,
  ``model.model.rotary_emb`` and ``model.lm_head``. So this can coexist with the
  mem_space patch machinery without interference (we simply don't use it).
* WRITE runs bottom ``j`` layers over ONE chunk with a chunk-local causal mask
  and RoPE positions ``0:T`` (each chunk is contextualised in isolation).
* READ concatenates the cached depth-``j`` hidden states of the sink + selected
  chunks + query into a single sequence, assigns FRESH contiguous RoPE positions
  ``0:|H|``, applies a standard causal mask, and resumes ``layers[j:] -> norm
  -> lm_head``.
* ``j == 0`` degenerates to "resume from layer 0 on the packed embeddings with
  contiguous positions", which IS a standard forward on the concatenated token
  ids — i.e. selective full re-forward (the RAG upper bound). ``j == L`` is the
  closed-book endpoint (chunks never attend to each other or the query at any
  layer; read only re-normalises + projects the stacked depth-L hiddens).
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from transformers.masking_utils import create_causal_mask


class QCMemModel:
    """Mid-depth resume memory over a stock ``LlamaForCausalLM``.

    Parameters
    ----------
    model:
        A loaded ``LlamaForCausalLM`` (or any model exposing the same
        ``.model.embed_tokens / .model.layers / .model.norm / .model.rotary_emb``
        and ``.lm_head`` structure). The backbone is used read-only and is NOT
        patched.
    resume_j:
        Bottom prepay depth ``a`` = layer index at which the forward is split.
        Valid range ``[0, num_hidden_layers]``. ``j=0`` -> selective full
        re-forward (RAG upper bound); ``j=L`` -> closed-book endpoint. The bottom
        ``layers[0:a]`` are run chunk-local at WRITE (exact prepay); the read
        resumes from layer ``a``.
    top_prepay_b:
        Number of TOP layers ``b`` to run *query-local* at READ instead of over
        the full packed sequence (a non-contiguous "front + back" resume variant,
        2026-07-05). Valid range ``[0, num_hidden_layers - resume_j]``.

        * ``b == 0`` (default) -> EXACT connective resume: ``read`` runs
          ``layers[a:L]`` over the whole packed sequence and returns logits at
          every packed position. Reproduces a stock full forward at ``a=0`` to
          fp-tolerance (this is the self-test gate).
        * ``b > 0`` -> APPROXIMATE top-prepay: the middle band ``layers[a:L-b]``
          is recomputed over the FULL packed sequence (query-aware, exact), then
          only the QUERY tail (last ``T_q`` positions) is pushed through the top
          band ``layers[L-b:L]`` with fresh contiguous positions and a causal
          mask over ``T_q``. This saves running the top ``b`` layers over the
          (long) context, at the cost of the query's top-band attention no longer
          seeing the context tokens.

          Rationale / caveat: an EXACT top prepay is impossible — the top band's
          input ``h_{L-b}`` is produced by the query-conditioned middle band, so
          it cannot be materialised chunk-local before the query is present (a
          chunk-local ``h_{L-b}`` diverges from the query-aware one by cosine
          ~0.87-0.92 / relative-L2 4-11x on Qwen3-8B, measured 2026-07-05).
          The variant above is the tractable, no-KV-injection approximation that
          leans on the hypothesis that the top layers are query-blind
          "output/format" layers; whether it holds is exactly the ablation
          question. Because ``b > 0`` only reshapes the ``read`` path over a
          MULTI-chunk pack, ``resume_forward_ids`` (single contiguous sequence)
          stays exact for any ``(a, b)`` and is used by the self-test to validate
          the layer-slicing plumbing.
    """

    def __init__(
        self,
        model: nn.Module,
        resume_j: int,
        top_prepay_b: int = 0,
        block_diagonal: bool = False,
    ):
        self.model = model
        inner = getattr(model, "model", model)
        self.inner = inner
        self.embed_tokens = inner.embed_tokens
        self.layers = inner.layers
        self.norm = inner.norm
        self.rotary_emb = inner.rotary_emb
        self.lm_head = model.lm_head
        self.config = inner.config
        self.num_layers = int(self.config.num_hidden_layers)

        if not (0 <= int(resume_j) <= self.num_layers):
            raise ValueError(
                f"resume_j must be in [0, {self.num_layers}]; got {resume_j}"
            )
        self.resume_j = int(resume_j)

        if not (0 <= int(top_prepay_b) <= self.num_layers - self.resume_j):
            raise ValueError(
                f"top_prepay_b must be in [0, {self.num_layers - self.resume_j}] "
                f"(num_layers - resume_j); got {top_prepay_b}"
            )
        self.top_prepay_b = int(top_prepay_b)
        # Boundary between the recomputed middle band and the query-local top band.
        self.mid_end = self.num_layers - self.top_prepay_b  # == L-b

        # --- ablation flag: block-diagonal read attention (2026-07-07) ---------
        # When True, ``read_core`` (with the default ``top_prepay_b == 0``) resumes
        # ``layers[a:L]`` over the SAME packed sequence but with a custom 4D
        # block-diagonal attention mask instead of the standard causal one:
        #   * the sink is globally visible (every position may attend to it);
        #   * each selected context chunk attends ONLY within its own block
        #     (no cross-chunk attention, and it does NOT attend to the query) —
        #     i.e. each chunk is contextualised as if prefilled in isolation and
        #     its KV reused (query-blind, chunk-isolated);
        #   * the query segment attends to the sink + ALL context chunks + itself
        #     (causal within the query), i.e. it reads over the reused KV.
        # The ONLY variable vs. the standard read is attention connectivity: RoPE
        # positions, sink, retrieved chunks and the split depth ``j`` are identical,
        # so ``(i) standard`` vs ``(ii) block_diagonal`` isolates the value of
        # cross-chunk + query attention (vs. per-chunk query-blind KV reuse).
        self.block_diagonal = bool(block_diagonal)

        # Optional gradient checkpointing on the (grad-bearing) read layer loop.
        # Only consulted by ``read_core`` when grad is enabled; write/eval paths
        # run under ``no_grad`` so checkpointing is a no-op there.
        self.grad_checkpoint = False

        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.hidden_size = int(self.config.hidden_size)

    # ------------------------------------------------------------------ #
    # low-level helpers
    # ------------------------------------------------------------------ #
    def _as_ids(self, token_ids) -> torch.Tensor:
        """Coerce a chunk's token ids to a [1, T] LongTensor on the model device."""
        if not torch.is_tensor(token_ids):
            token_ids = torch.tensor(token_ids, dtype=torch.long)
        ids = token_ids.to(self.device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        if ids.dim() != 2 or ids.shape[0] != 1:
            raise ValueError(
                f"expected token_ids of shape [T] or [1, T]; got {tuple(ids.shape)}"
            )
        return ids.long()

    def _make_mask_and_rope(self, hidden_like: torch.Tensor, positions: torch.Tensor):
        """Build the causal mask + RoPE (cos, sin) exactly the way
        ``LlamaModel.forward`` would for a sequence with the given ``positions``.

        ``hidden_like`` is only used for its shape/dtype/device (create_causal_mask
        and rotary_emb both take a tensor purely as a shape/device/dtype proxy).
        ``positions`` is a [1, S] LongTensor of RoPE position ids.
        """
        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=hidden_like,
            attention_mask=None,
            past_key_values=None,
            position_ids=positions,
        )
        position_embeddings = self.rotary_emb(hidden_like, position_ids=positions)
        return causal_mask, position_embeddings

    def _make_block_diagonal_mask_and_rope(
        self,
        hidden_like: torch.Tensor,
        positions: torch.Tensor,
        seg_lens: Sequence[tuple],
    ):
        """Build a ``[1, 1, H, H]`` BLOCK-DIAGONAL attention mask + RoPE.

        ``seg_lens`` is the packed layout as an ordered list of ``(kind, length)``
        tuples with ``kind`` in ``{"sink", "chunk", "query"}`` (exactly one
        ``"query"``, last; at most one ``"sink"``, first if present; the rest
        ``"chunk"`` in doc order). The connectivity (row ``i`` attends to col
        ``k`` iff ``mask[i, k]``) is::

            allow = is_sink_col
                    OR same_block(i, k)
                    OR (is_query_row(i) AND is_context_chunk_col(k))
            keep  = allow AND (k <= i)          # still causal (block-diag ⊆ causal)

        i.e. the sink is globally visible; each context chunk attends only to the
        sink + causally within its own block (NO cross-chunk, NO query — a
        query-blind, chunk-isolated KV prefill that is then reused); the query
        attends to the sink + every context chunk + causally within itself.

        Because the packed RoPE positions are the SAME contiguous ``0:H`` as the
        standard read, within-chunk relative positions equal an isolated prefill,
        and the ONLY difference vs. the exact causal read is attention
        connectivity — which is exactly the ablation variable. With a SINGLE
        context chunk the mask is identical to the standard causal mask (there is
        no other chunk to hide and the query already sees the one chunk), so the
        block-diagonal read degenerates to the standard read (self-test gate).

        The returned mask matches the dtype/format the backbone's attention impl
        expects, inferred from a reference ``create_causal_mask`` output.
        """
        H = int(positions.shape[1])
        device = hidden_like.device

        # Per-position block id: sink -> -1, context chunk c -> c (0-indexed),
        # query -> num_chunks. (No sink => no position carries -1.)
        block = torch.empty(H, dtype=torch.long, device=device)
        num_chunks = sum(1 for kind, _ in seg_lens if kind == "chunk")
        pos = 0
        chunk_c = 0
        for kind, length in seg_lens:
            length = int(length)
            if length <= 0:
                continue
            if kind == "sink":
                block[pos:pos + length] = -1
            elif kind == "chunk":
                block[pos:pos + length] = chunk_c
                chunk_c += 1
            elif kind == "query":
                block[pos:pos + length] = num_chunks
            else:  # pragma: no cover - guarded by callers
                raise ValueError(f"unknown segment kind {kind!r}")
            pos += length
        if pos != H:  # pragma: no cover - guarded by callers
            raise ValueError(f"seg_lens sum {pos} != packed length {H}")

        row_block = block.view(H, 1)          # [H, 1]
        col_block = block.view(1, H)          # [1, H]
        is_sink_col = (col_block == -1)                                  # [1, H]
        is_ctx_chunk_col = (col_block >= 0) & (col_block < num_chunks)   # [1, H]
        is_query_row = (row_block == num_chunks)                         # [H, 1]
        same_block = (row_block == col_block)                            # [H, H]

        allow = is_sink_col | same_block | (is_query_row & is_ctx_chunk_col)
        row_idx = torch.arange(H, device=device).view(H, 1)
        col_idx = torch.arange(H, device=device).view(1, H)
        causal = col_idx <= row_idx
        keep = allow & causal                                           # [H, H] bool
        keep = keep.view(1, 1, H, H)

        # Match the format the attention impl expects. For SDPA the causal mask is
        # a bool "keep" tensor (True == attend), so pass our bool mask straight
        # through. For an additive float mask (eager) convert keep -> {0, min}.
        ref = create_causal_mask(
            config=self.config,
            inputs_embeds=hidden_like,
            attention_mask=None,
            past_key_values=None,
            position_ids=positions,
        )
        if isinstance(ref, torch.Tensor) and ref.dtype != torch.bool:
            min_val = torch.finfo(ref.dtype).min
            mask = torch.zeros(1, 1, H, H, dtype=ref.dtype, device=device)
            mask = mask.masked_fill(~keep, min_val)
        else:
            # bool "keep" mask (SDPA), or ref is None (pure-causal skip) — either
            # way an explicit bool mask is the correct, always-valid form here.
            mask = keep

        position_embeddings = self.rotary_emb(hidden_like, position_ids=positions)
        return mask, position_embeddings

    @staticmethod
    def _layer_out_hidden(out):
        """Coerce a decoder layer's return to the residual-stream hidden tensor.

        In-tree transformers >=5.x decoder layers (dense AND MoE — Llama, Qwen3,
        Qwen3-MoE, Qwen2-MoE, Mixtral, DeepSeek-V2/V3, OLMoE, Phi-MoE, DBRX,
        GLM4-MoE, Hunyuan-V1-MoE, …) all return a BARE ``hidden_states`` tensor;
        the MoE router logits are no longer part of the layer return, so the
        residual stream QCMem caches (``h_j``) is identical in shape/semantics to
        the dense case (expert-aggregated output == the layer's hidden output).

        Older / custom ``trust_remote_code`` modeling (e.g. the Hunyuan Hy3
        ``hy_v3`` custom decoder, or legacy Mixtral-style layers) may instead
        return a ``tuple`` whose FIRST element is the hidden state (optionally
        followed by attn weights / router logits / present-KV). We defensively
        unwrap that first element so QCMem works on both conventions without
        touching the dense fast-path (a bare tensor passes straight through).
        """
        if torch.is_tensor(out):
            return out
        if isinstance(out, (tuple, list)):
            return out[0]
        # BaseModelOutput-like object exposing .last_hidden_state / .hidden_states
        for attr in ("last_hidden_state", "hidden_states"):
            val = getattr(out, attr, None)
            if torch.is_tensor(val):
                return val
        raise TypeError(
            f"decoder layer returned unsupported type {type(out)!r}; expected a "
            "tensor or a tuple whose first element is the hidden state"
        )

    def _run_layers(
        self,
        hidden: torch.Tensor,
        layer_slice: slice,
        causal_mask,
        positions: torch.Tensor,
        position_embeddings,
    ) -> torch.Tensor:
        """Run ``self.layers[layer_slice]`` on ``hidden`` with the given mask/RoPE.

        Works uniformly for dense and MoE backbones: the per-layer call is the
        standard ``layer(hidden, attention_mask=, position_ids=,
        position_embeddings=, use_cache=False)`` interface, and MoE routing lives
        entirely inside ``layer.mlp`` (the sparse block), which is position-blind
        (it routes each token by its own hidden vector, independent of sequence
        position or which other tokens are packed alongside it). So a chunk-local
        WRITE routes each token exactly as the full-context forward would, and the
        cached depth-``j`` hidden reproduces bit-for-bit — no MoE-specific plumbing
        is needed here beyond tolerating a tuple layer-return (see
        :meth:`_layer_out_hidden`)."""
        use_ckpt = (
            self.grad_checkpoint
            and torch.is_grad_enabled()
            and hidden.requires_grad
        )
        for layer in self.layers[layer_slice]:
            if use_ckpt:
                out = torch.utils.checkpoint.checkpoint(
                    lambda h, _l=layer: _l(
                        h,
                        attention_mask=causal_mask,
                        position_ids=positions,
                        position_embeddings=position_embeddings,
                        use_cache=False,
                    ),
                    hidden,
                    use_reentrant=False,
                )
            else:
                out = layer(
                    hidden,
                    attention_mask=causal_mask,
                    position_ids=positions,
                    position_embeddings=position_embeddings,
                    use_cache=False,
                )
            hidden = self._layer_out_hidden(out)
        return hidden

    # ------------------------------------------------------------------ #
    # WRITE side: embed + layers[0:j] over one chunk (chunk-local)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def write_chunk(self, token_ids) -> torch.Tensor:
        """Encode ONE chunk in isolation to depth ``j``.

        Runs ``embed_tokens`` + ``layers[0:j]`` with a chunk-local causal mask and
        RoPE positions ``0:T``. Returns the depth-``j`` hidden state ``h_j`` of
        shape ``[1, T, d]`` (this is what QCMem caches per chunk). The bottom-j
        KV is discarded; only ``h_j`` survives.
        """
        ids = self._as_ids(token_ids)
        T = ids.shape[1]
        inputs_embeds = self.embed_tokens(ids)
        positions = torch.arange(T, device=self.device).unsqueeze(0)
        causal_mask, position_embeddings = self._make_mask_and_rope(
            inputs_embeds, positions
        )
        h_j = self._run_layers(
            inputs_embeds, slice(0, self.resume_j),
            causal_mask, positions, position_embeddings,
        )
        return h_j  # [1, T, d]

    # ------------------------------------------------------------------ #
    # READ side: pack cached h_j pieces + resume layers[j:] -> logits
    # ------------------------------------------------------------------ #
    def read_core(
        self,
        sink_hj: Optional[torch.Tensor],
        selected_hj_list: Sequence[torch.Tensor],
        query_hj: torch.Tensor,
        logits_tail: Optional[int] = None,
    ) -> torch.Tensor:
        """Grad-bearing read: resume from layer ``a`` over the packed sequence.

        This is the computational core of :meth:`read` *without* the
        ``no_grad`` guard, so the distillation trainer can run it inside its own
        autograd context (the student's trainable LoRA lives in ``layers[a:]``,
        which this method executes). Callers who only want inference should use
        :meth:`read`, which wraps this in ``torch.no_grad()``.

        Behaviour depends on ``self.top_prepay_b``:

        * ``b == 0`` — EXACT connective resume. Pack
          ``[sink ; ctx... ; query]`` into ``[1, |H|, d]`` with fresh contiguous
          RoPE positions and a causal mask over ``|H|``, run ``layers[a:L]``,
          norm + lm_head. Returns logits for EVERY packed position ``[1, |H|, V]``.

          If ``self.block_diagonal`` is set, the SAME pack / positions / depth are
          used but the causal mask is replaced by a block-diagonal one (sink
          global; each context chunk within-block only; query sees sink + all
          chunks + itself). This is the "reuse per-chunk query-blind KV" ablation
          arm; see :meth:`_make_block_diagonal_mask_and_rope`. Only valid for
          ``b == 0``.

        * ``b > 0`` — APPROXIMATE top-prepay (front + back caching). Run the
          middle band ``layers[a : L-b]`` over the full packed sequence
          (query-aware), then slice off the QUERY tail (last ``T_q`` positions,
          where ``T_q == query_hj.shape[1]``) and run ONLY that tail through the
          top band ``layers[L-b : L]`` with fresh contiguous positions ``0:T_q``
          and a causal mask over ``T_q``, then norm + lm_head. Returns logits for
          the QUERY tail only ``[1, T_q, V]`` (the read tail is all a caller ever
          decodes from, so this is sufficient for generation and distillation).

        ``logits_tail`` (optional): if set to ``n > 0``, apply ``norm + lm_head``
        to only the LAST ``n`` positions of the resumed hidden state and return
        ``[1, n, V]``. This avoids materialising the full ``[1, |H|, V]`` logit
        tensor (Qwen ``V=151936`` -> ~1.2 GB per 2k-pack in bf16) when the caller
        (e.g. the distillation trainer) only needs the query-segment logits. The
        pre-``lm_head`` layer stack is unchanged, so the returned tail is
        numerically identical to slicing the full output.
        """
        pieces: List[torch.Tensor] = []
        seg_lens: List[tuple] = []
        if sink_hj is not None:
            pieces.append(sink_hj)
            seg_lens.append(("sink", int(sink_hj.shape[1])))
        for h in selected_hj_list:
            if h is not None and h.shape[1] > 0:
                pieces.append(h)
                seg_lens.append(("chunk", int(h.shape[1])))
        pieces.append(query_hj)
        seg_lens.append(("query", int(query_hj.shape[1])))

        packed = torch.cat(pieces, dim=1)  # [1, |H|, d]
        H = packed.shape[1]
        positions = torch.arange(H, device=self.device).unsqueeze(0)
        if self.block_diagonal:
            if self.top_prepay_b != 0:
                raise ValueError(
                    "block_diagonal read is only defined for top_prepay_b == 0 "
                    f"(the exact-depth resume path); got b={self.top_prepay_b}"
                )
            causal_mask, position_embeddings = self._make_block_diagonal_mask_and_rope(
                packed, positions, seg_lens
            )
        else:
            causal_mask, position_embeddings = self._make_mask_and_rope(packed, positions)

        if self.top_prepay_b == 0:
            # Exact resume over the whole packed sequence.
            hidden = self._run_layers(
                packed, slice(self.resume_j, self.num_layers),
                causal_mask, positions, position_embeddings,
            )
            if logits_tail is not None and logits_tail > 0:
                hidden = hidden[:, -int(logits_tail):, :]
            hidden = self.norm(hidden)
            return self.lm_head(hidden)  # [1, |H| or tail, V]

        # --- b > 0: recompute middle band (query-aware) over full pack ---
        mid = self._run_layers(
            packed, slice(self.resume_j, self.mid_end),
            causal_mask, positions, position_embeddings,
        )  # [1, |H|, d]  == h_{L-b} for every packed position (query-conditioned)

        # --- top band runs QUERY-LOCAL: only the query tail, fresh positions ---
        T_q = int(query_hj.shape[1])
        top_in = mid[:, -T_q:, :]  # [1, T_q, d]
        top_pos = torch.arange(T_q, device=self.device).unsqueeze(0)
        top_mask, top_pe = self._make_mask_and_rope(top_in, top_pos)
        top_out = self._run_layers(
            top_in, slice(self.mid_end, self.num_layers),
            top_mask, top_pos, top_pe,
        )  # [1, T_q, d]
        if logits_tail is not None and logits_tail > 0:
            top_out = top_out[:, -int(logits_tail):, :]
        top_out = self.norm(top_out)
        return self.lm_head(top_out)  # [1, T_q or tail, V]

    @torch.no_grad()
    def read(
        self,
        sink_hj: Optional[torch.Tensor],
        selected_hj_list: Sequence[torch.Tensor],
        query_hj: torch.Tensor,
    ) -> torch.Tensor:
        """Inference read (``no_grad`` wrapper around :meth:`read_core`).

        Packs ``[sink_hj ; selected_hj_list[0] ; ... ; query_hj]`` along the time
        axis and resumes ``layers[a:]``. See :meth:`read_core` for the exact vs
        top-prepay (``b>0``) semantics and output shapes.

        Parameters
        ----------
        sink_hj:
            Depth-``a`` hidden of the attention-sink token(s) (e.g. BOS), prepended
            at packed position 0. ``None`` to omit the sink.
        selected_hj_list:
            Ordered list of the selected context chunks' cached ``h_j`` tensors,
            each ``[1, T_c, d]``.
        query_hj:
            The query chunk's cached ``h_j`` (``[1, T_q, d]``), appended last so the
            final logits are read at its tail.

        Returns
        -------
        logits: ``[1, |H|, V]`` when ``top_prepay_b == 0`` else ``[1, T_q, V]``.
        """
        return self.read_core(sink_hj, selected_hj_list, query_hj)

    # ------------------------------------------------------------------ #
    # convenience: full split forward on a single packed token sequence
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def resume_forward_ids(self, token_ids) -> torch.Tensor:
        """Split-at-j forward over a SINGLE contiguous token sequence.

        Equivalent to ``write_chunk`` (0:j) immediately followed by ``read`` with
        no sink and no separate query (the whole sequence IS the query). Used by
        the self-test: at ``j=0`` this must equal the stock ``model(input_ids)``
        forward to floating-point tolerance, mirroring the primitive check.
        """
        ids = self._as_ids(token_ids)
        T = ids.shape[1]
        inputs_embeds = self.embed_tokens(ids)
        positions = torch.arange(T, device=self.device).unsqueeze(0)
        causal_mask, position_embeddings = self._make_mask_and_rope(
            inputs_embeds, positions
        )
        hidden = self._run_layers(
            inputs_embeds, slice(0, self.resume_j),
            causal_mask, positions, position_embeddings,
        )
        hidden = self._run_layers(
            hidden, slice(self.resume_j, self.num_layers),
            causal_mask, positions, position_embeddings,
        )
        hidden = self.norm(hidden)
        return self.lm_head(hidden)

    @torch.no_grad()
    def full_forward_logits(self, token_ids) -> torch.Tensor:
        """Stock ``model(input_ids)`` logits — the self-test reference."""
        ids = self._as_ids(token_ids)
        return self.model(input_ids=ids, use_cache=False).logits
