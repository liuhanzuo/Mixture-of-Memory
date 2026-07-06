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
        Layer index ``j`` at which the forward is split. Valid range
        ``[0, num_hidden_layers]``. ``j=0`` -> selective full re-forward (RAG
        upper bound); ``j=L`` -> closed-book endpoint.
    """

    def __init__(self, model: nn.Module, resume_j: int):
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

    def _run_layers(
        self,
        hidden: torch.Tensor,
        layer_slice: slice,
        causal_mask,
        positions: torch.Tensor,
        position_embeddings,
    ) -> torch.Tensor:
        """Run ``self.layers[layer_slice]`` on ``hidden`` with the given mask/RoPE."""
        for layer in self.layers[layer_slice]:
            hidden = layer(
                hidden,
                attention_mask=causal_mask,
                position_ids=positions,
                position_embeddings=position_embeddings,
                use_cache=False,
            )
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
    @torch.no_grad()
    def read(
        self,
        sink_hj: Optional[torch.Tensor],
        selected_hj_list: Sequence[torch.Tensor],
        query_hj: torch.Tensor,
    ) -> torch.Tensor:
        """Resume the forward from layer ``j`` over the packed memory sequence.

        Packs ``[sink_hj ; selected_hj_list[0] ; ... ; query_hj]`` along the time
        axis into a single ``[1, |H|, d]`` sequence, assigns FRESH contiguous RoPE
        positions ``0:|H|``, applies a standard causal mask, then runs
        ``layers[j:] -> norm -> lm_head``.

        Parameters
        ----------
        sink_hj:
            Depth-``j`` hidden of the attention-sink token(s) (e.g. BOS), prepended
            at packed position 0. ``None`` to omit the sink.
        selected_hj_list:
            Ordered list of the selected context chunks' cached ``h_j`` tensors,
            each ``[1, T_c, d]``.
        query_hj:
            The query chunk's cached ``h_j`` (``[1, T_q, d]``), appended last so the
            final logits are read at its tail.

        Returns
        -------
        logits: ``[1, |H|, V]``
        """
        pieces: List[torch.Tensor] = []
        if sink_hj is not None:
            pieces.append(sink_hj)
        for h in selected_hj_list:
            if h is not None and h.shape[1] > 0:
                pieces.append(h)
        pieces.append(query_hj)

        packed = torch.cat(pieces, dim=1)  # [1, |H|, d]
        H = packed.shape[1]
        positions = torch.arange(H, device=self.device).unsqueeze(0)
        causal_mask, position_embeddings = self._make_mask_and_rope(packed, positions)

        hidden = self._run_layers(
            packed, slice(self.resume_j, self.num_layers),
            causal_mask, positions, position_embeddings,
        )
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)
        return logits  # [1, |H|, V]

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
