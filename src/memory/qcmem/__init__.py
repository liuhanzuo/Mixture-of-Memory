"""QCMem — mid-depth resume memory (query-conditioned chunk memory).

A self-contained orchestrator that splits a *plain* (un-patched) Llama backbone
at an arbitrary layer ``j``:

  * WRITE  (per chunk, chunk-local):  embed -> layers[0:j] with a chunk-local
    causal mask + RoPE positions 0:T -> cache the depth-``j`` hidden state h_j.
  * READ   (per query):  pack [h_j^sink; h_j^c1; ...; h_j^ck; h_j^q] into a
    single [1,|H|,d] sequence with FRESH contiguous RoPE positions 0:|H| and a
    standard causal mask over |H|, then resume layers[j:] -> norm -> lm_head.

This module does NOT depend on the mem_space patch and never mutates the model
in place — it only reads layers/embeddings/norm/lm_head off a stock
LlamaForCausalLM. j=0 is the RAG upper bound (selective full re-forward); j=L is
the closed-book endpoint.

The load-bearing correctness claim (resume-from-layer-j == full forward, bitwise
close) is proven by ``scripts/qcmem_resume_primitive_check.py``.
"""

from .qcmem_model import QCMemModel

__all__ = ["QCMemModel"]
