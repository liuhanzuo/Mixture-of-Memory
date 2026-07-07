"""PyramidMemory — MemoryLLM (mid tier) + QCMem raw-hidden (base tier) merge.

Skeleton for Pyramid v1. See ``versions/pyramid_v1_memoryllm_qcmem_merge.md`` for
the full architecture, the P1 (single-injection MVP) vs P2 (faithful dual-cadence)
read variants, initialization and known issues.

Structure is complete; the load-bearing read math (P2's mixed far/near attention)
is left as ``NotImplementedError`` until the mask/RoPE consistency gate (Known
issue #2/#3 in the design doc) is validated. The write side and P1 read (which
reduce to already-proven MemoryLLM.inject_memory + QCMem read_core primitives) are
sketched concretely so the class is runnable end-to-end once P2 lands.

Backbone contract
-----------------
Unlike ``QCMemModel`` (which wraps a *stock HF* Llama: model-level ``rotary_emb``,
decoder layers taking ``position_embeddings``), PyramidMemory drives the **ported
MemoryLLM** backbone (custom tf-4.43-era Llama: per-attention ``rotary_emb``,
decoder layers taking ``cache_position`` + ``prefix_token_length``). Both tiers
therefore share the SAME ``mllm.model.layers`` — we never mix two Llama impls.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn


class PyramidMemory:
    """Two-tier (raw + compressed) long-context memory over one MemoryLLM backbone.

    Parameters
    ----------
    mllm:
        A loaded ported ``MemoryLLM`` (``src.memory.memoryllm_ported.MemoryLLM``).
        Exposes ``.model.layers`` (32 ``LlamaDecoderLayer``), ``.model.embed_tokens``,
        ``.model.norm``, ``.lm_head``, the trained pool ``.memory`` and the helpers
        ``inject_memory`` / ``cat_memory_and_hiddens`` / ``update_memory_with_delta_memory``.
        Used read-only for the layer stack; ``.memory`` IS mutated by writes
        (that is MemoryLLM's design) so callers must ``reset_memory`` per document.
    resume_j:
        Split depth ``a`` shared by both tiers. ``layers[0:a]`` run chunk-local for
        the raw near tier (QCMem write); read resumes ``layers[a:L]`` over the
        near pack + (P2) per-layer far pool prefix. ``[0, L]``; ``a=0`` self-tests
        to the RAG upper bound.
    near_chunks_k:
        ``K`` — number of recent chunks kept RAW (base tier). The rest of the
        context is compressed into the MemoryLLM pool (mid tier).
    variant:
        ``"P2"`` (default, faithful dual-cadence: far pool re-prefixed every layer,
        near pack injected once) or ``"P1"`` (single-injection MVP: one depth-``a``
        pool slice prepended as ordinary tokens, reuses a QCMem-style resume).
    far_tokens_m:
        ``M_far`` — number of pool tokens per layer to use at read. ``None`` =
        full ``num_blocks*num_tokens`` (12800). Smaller subsamples the pool for
        speed at some recall cost.
    """

    def __init__(
        self,
        mllm: nn.Module,
        resume_j: int,
        near_chunks_k: int = 4,
        variant: str = "P2",
        far_tokens_m: Optional[int] = None,
    ):
        self.mllm = mllm
        inner = getattr(mllm, "model", mllm)
        self.inner = inner
        self.embed_tokens = inner.embed_tokens
        self.layers = inner.layers
        self.norm = inner.norm
        self.lm_head = mllm.lm_head
        self.config = inner.config
        self.num_layers = int(self.config.num_hidden_layers)          # 32
        self.d = int(self.config.hidden_size)                         # 4096

        if not (0 <= int(resume_j) <= self.num_layers):
            raise ValueError(f"resume_j must be in [0, {self.num_layers}]; got {resume_j}")
        self.resume_j = int(resume_j)

        if variant not in ("P1", "P2"):
            raise ValueError(f"variant must be 'P1' or 'P2'; got {variant!r}")
        self.variant = variant

        self.near_chunks_k = int(near_chunks_k)
        # Full pool width per layer = num_blocks * num_tokens (== 12800 for the ckpt).
        self.pool_width = int(mllm.num_blocks) * int(mllm.num_tokens)
        self.far_tokens_m = self.pool_width if far_tokens_m is None else int(far_tokens_m)

        self.device = next(mllm.parameters()).device
        self.dtype = next(mllm.parameters()).dtype

        # Snapshot the trained init pool so reset_memory() can restore it per doc.
        self._pool_init = mllm.memory.detach().clone()
        self._init_flag = int(mllm.initialized)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _as_ids(self, token_ids) -> torch.Tensor:
        """Coerce a chunk's token ids to a [1, T] LongTensor on the model device."""
        if not torch.is_tensor(token_ids):
            token_ids = torch.tensor(token_ids, dtype=torch.long)
        ids = token_ids.to(self.device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        if ids.dim() != 2 or ids.shape[0] != 1:
            raise ValueError(f"expected [T] or [1, T]; got {tuple(ids.shape)}")
        return ids.long()

    def reset_memory(self) -> None:
        """Restore the MemoryLLM pool to its trained init (call before each doc).

        ``inject_memory`` mutates ``mllm.memory`` in place (FIFO drop+append), so
        without this the far context of one eval sample bleeds into the next
        (Known issue #4). Cheap: an in-place copy of the [32,12800,4096] pool.
        """
        with torch.no_grad():
            self.mllm.memory.data.copy_(self._pool_init.to(self.mllm.memory.device))
            self.mllm.initialized.fill_(self._init_flag)

    # ------------------------------------------------------------------ #
    # WRITE — mid tier (compressed far) + base tier (raw near)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def write_far(self, far_segment_ids: Sequence) -> None:
        """MID tier: FIFO-inject one far-context segment into the MemoryLLM pool.

        Thin wrapper over ``mllm.inject_memory(..., update_memory=True)`` — runs
        the full ported MemoryLLM on the segment and drops/append its delta into
        ``mllm.memory``. Segment length must be > 16 tokens (MemoryLLM hard min).
        """
        ids = self._as_ids(far_segment_ids)
        self.mllm.inject_memory(ids, update_memory=True)

    @torch.no_grad()
    def write_near_chunk(self, token_ids) -> torch.Tensor:
        """BASE tier: chunk-local embed + ``layers[0:a]`` -> depth-``a`` hidden.

        The QCMem raw-hidden write, expressed on the MemoryLLM backbone's layer
        API (``prefix_token_length=0`` -> a plain chunk-local forward with no pool
        prefix because we run each layer directly, not via MemoryLLM.forward).
        Returns ``h_a`` of shape ``[1, T, d]``.

        NOTE (Known issue #1/#2): building the chunk-local causal mask + RoPE
        ``position_ids 0:T`` for the *ported* layer API (which expects
        ``cache_position`` and computes cos/sin inside each attention via its own
        ``rotary_emb``) is the remaining plumbing. Left NotImplemented until the
        mask/RoPE gate is written and diffed against a reference forward.
        """
        raise NotImplementedError(
            "write_near_chunk: implement chunk-local layers[0:a] forward on the "
            "ported MemoryLLM layer API (cache_position + per-attention rotary_emb). "
            "Gate at a=0 against embed()==h_a. See design doc Known issue #1/#2."
        )

    # ------------------------------------------------------------------ #
    # READ — resume layers[a:L] over near pack + far pool
    # ------------------------------------------------------------------ #
    def read_core(
        self,
        sink_ha: Optional[torch.Tensor],
        near_ha_list: Sequence[torch.Tensor],
        query_ha: torch.Tensor,
        logits_tail: Optional[int] = None,
    ) -> torch.Tensor:
        """Resume ``layers[a:L]`` over the near pack, using the far pool per the
        chosen variant. Returns logits ``[1, N, V]`` (or the last ``logits_tail``).

        * P1 — prepend one depth-``a`` pool slice as ordinary tokens and resume
          once (reduces to a QCMem-style ``read_core`` with an extra far block).
        * P2 — re-prefix ``memory[idx]`` at every resumed layer with
          ``prefix_token_length`` so the near pack attends to the full per-layer
          pool (faithful MemoryLLM cadence). Requires the mixed far/near mask.
        """
        raise NotImplementedError(
            f"read_core[{self.variant}]: implement the resume over layers[a:L]. "
            "P1 = QCMem read_core + far_ha block (reuse the proven packing); "
            "P2 = per-layer pool prefix via prefix_token_length + a mixed 4D mask "
            "(far fully visible to near; near causal among itself). Self-test: "
            "P2 with far_tokens_m=0 must equal the P1/QCMem near-only read. "
            "See design doc Architecture + Known issue #2/#3."
        )

    @torch.no_grad()
    def read(
        self,
        sink_ha: Optional[torch.Tensor],
        near_ha_list: Sequence[torch.Tensor],
        query_ha: torch.Tensor,
    ) -> torch.Tensor:
        """Inference read (``no_grad`` wrapper around :meth:`read_core`)."""
        return self.read_core(sink_ha, near_ha_list, query_ha)

    # ------------------------------------------------------------------ #
    # end-to-end convenience
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate_from_context(
        self,
        input_ids: torch.Tensor,      # [1, L] full formatted sample (BOS-prefixed)
        chunk_size: int,
        max_new_tokens: int,
        far_group_tokens: Optional[int] = None,
        sink_bos: bool = True,
    ) -> List[int]:
        """Full pyramid decode: split -> write both tiers -> greedy read loop.

        Skeleton wiring (calls the write/read primitives above). Mirrors
        ``scripts/eval_qcmem_babilong.py::qcmem_generate`` but with the far tier
        added. Decodes via a manual greedy loop (NOT ``mllm.generate`` — broken on
        the port under tf-5.5.4, Known issue #5).
        """
        raise NotImplementedError(
            "generate_from_context: wire write_far (far chunks) + write_near_chunk "
            "(recent K chunks) then a manual greedy read loop over read(). Depends "
            "on write_near_chunk + read_core landing first."
        )
