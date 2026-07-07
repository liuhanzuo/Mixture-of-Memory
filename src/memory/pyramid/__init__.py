"""Pyramid memory — two-tier long-context compression (v1).

Combines two of the three pyramid tiers over ONE shared Llama-3-8B backbone:

* **Base tier (raw hidden, QCMem).** Recent ``K`` chunks are kept as *raw*
  depth-``a`` hidden states (``write_chunk`` = embed + ``layers[0:a]`` chunk-local).
  Precise, no compression. See ``src/memory/qcmem/qcmem_model.py``.
* **Mid tier (MemoryLLM pool).** Distant context is compressed into MemoryLLM's
  trained per-layer memory pool ``memory[L, num_blocks*num_tokens, d]`` via
  ``inject_memory``; at read the pool is a per-layer KV prefix. See
  ``src/memory/memoryllm_ported/``.

Top tier (slot bank) is NOT part of v1.

Design doc: ``versions/pyramid_v1_memoryllm_qcmem_merge.md`` (read it first — it
carries the forward-pass pseudo-code, the P1/P2 variants, and the known issues).
"""

from .pyramid_model import PyramidMemory

__all__ = ["PyramidMemory"]
