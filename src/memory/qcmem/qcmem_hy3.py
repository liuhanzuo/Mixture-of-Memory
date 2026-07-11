"""QCMem depth-partitioned retrieval read-out on the Tencent Hunyuan **Hy3**
(``hy_v3``) 80-layer MoE backbone, sharded across multiple GPUs (2026-07-11).

Why a separate file
-------------------
The stock :class:`~src.memory.qcmem.qcmem_model.QCMemModel` assumes the whole
backbone lives on ONE device (it creates ids / RoPE positions on ``self.device``
and runs ``layers[...]`` + ``norm`` + ``lm_head`` there). Hy3 is a 597 GB bf16
model that only fits when ``device_map="auto"`` shards its 80 ``HYV3DecoderLayer``
blocks across the 8 local L20A GPUs — so the depth-partitioned WRITE/READ loop
must hop the residual-stream hidden (and the causal mask + RoPE cos/sin) onto
whichever GPU the *next* layer sits on. This module is that thin, device-aware
subclass; it reuses ALL of the parent's packing / masking / read semantics and
ONLY overrides the pieces that touch a device:

  * ``_run_layers`` — before every ``HYV3DecoderLayer`` call, move
    ``(hidden, attention_mask, position_ids, position_embeddings)`` to that
    layer's parameter device (looked up once from ``hf_device_map`` / the layer's
    own params). A no-op when the model is single-device (CPU tiny self-test).
  * ``norm`` / ``lm_head`` are wrapped as device-moving callables so the parent's
    ``self.norm(hidden)`` / ``self.lm_head(hidden)`` land the hidden on the right
    GPU without duplicating ``read_core`` / ``resume_forward_ids``.

The parent's assumptions that MADE this trivial (verified against the HF
``modeling_hy_v3.py`` source, tf 5.13.1):

  * ``HYV3DecoderLayer.forward(hidden_states, attention_mask=, position_ids=,
    past_key_values=, use_cache=, position_embeddings=)`` — the EXACT kwargs the
    parent ``_run_layers`` already passes, and it returns a BARE hidden tensor
    (standard residual stream ``h = residual + block(h)``).
  * ``model.model.{embed_tokens, layers, norm, rotary_emb}`` + ``model.lm_head``
    + ``model.config`` are all present (checked on a meta-device build).
  * ``HYV3RotaryEmbedding.forward(x, position_ids) -> (cos, sin)`` — same
    interface as Llama's ``rotary_emb``; it internally does ``inv_freq.to(x.device)``
    so cos/sin land on ``x``'s device (we compute them once on the embed device).
  * The MoE router (``HYV3TopKRouter``) is a pure function of the token hidden
    (``gate(hidden)``), position-blind — so chunk-local WRITE routes each token
    exactly as the full-context forward would, and cached ``h_j`` is reproducible
    (this is claim (B), validated by the self-test).

j-semantics are identical to the parent: ``j=0`` == selective full re-forward
(RAG upper bound; self-test gate), ``j=L`` == closed-book endpoint. Layer 0 of
Hy3 is a dense ``HYV3MLP`` and layers 1..79 are sparse ``HYV3MoE`` — the
depth-partition crosses that boundary transparently.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .qcmem_model import QCMemModel


def _module_device(module: nn.Module) -> torch.device:
    """Device of a module's first parameter, falling back to its first buffer.

    ``HYV3RotaryEmbedding`` has no parameters (only the ``inv_freq`` buffer), and a
    module could in principle be on meta; we return a concrete device or ``cpu``.
    """
    for p in module.parameters(recurse=True):
        return p.device
    for b in module.buffers(recurse=True):
        return b.device
    return torch.device("cpu")


def _to_device(obj, device: torch.device):
    """Move a tensor / tuple / list / dict of tensors (or ``None``) to ``device``.

    Handles the exact shapes QCMem passes through ``_run_layers``: the residual
    hidden (tensor), ``position_ids`` (tensor), the causal mask (tensor / ``None`` /
    bool tensor / additive float tensor), and ``position_embeddings`` (a
    ``(cos, sin)`` tuple). Non-tensor leaves pass through untouched.
    """
    if obj is None:
        return None
    if torch.is_tensor(obj):
        return obj.to(device) if obj.device != device else obj
    if isinstance(obj, tuple):
        return tuple(_to_device(o, device) for o in obj)
    if isinstance(obj, list):
        return [_to_device(o, device) for o in obj]
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    return obj


class _DeviceMovingCall:
    """Wrap a module so ``wrapper(x, ...)`` first moves ``x`` (and any extra tensor
    args/kwargs) to the module's own device, then calls it. Used for ``norm`` and
    ``lm_head`` so the parent's ``self.norm(hidden)`` / ``self.lm_head(hidden)``
    work across the shard boundary without reimplementing the read path.
    """

    def __init__(self, module: nn.Module):
        self.module = module
        self.device = _module_device(module)

    def __call__(self, x, *args, **kwargs):
        x = _to_device(x, self.device)
        args = tuple(_to_device(a, self.device) for a in args)
        kwargs = {k: _to_device(v, self.device) for k, v in kwargs.items()}
        return self.module(x, *args, **kwargs)


class QCMemHy3Model(QCMemModel):
    """Multi-GPU (``device_map``-sharded) QCMem over a stock ``HYV3ForCausalLM``.

    Drop-in for :class:`QCMemModel` — same ``write_chunk`` / ``read`` / ``read_core``
    / ``resume_forward_ids`` / ``full_forward_logits`` API and the same ``resume_j``
    / ``top_prepay_b`` / ``block_diagonal`` semantics. The ONLY behavioural
    difference is that every layer / norm / lm_head call moves its inputs to the
    correct GPU first, so the model may be sharded by ``device_map="auto"``.
    """

    def __init__(
        self,
        model: nn.Module,
        resume_j: int,
        top_prepay_b: int = 0,
        block_diagonal: bool = False,
    ):
        super().__init__(model, resume_j, top_prepay_b, block_diagonal)

        # Inputs (ids / RoPE positions / mask / cos-sin) are created on the embed
        # device; layer 0 lives here too under a contiguous device_map. The parent
        # set self.device = next(model.parameters()).device which for device_map
        # is the first-param device (== embed device) — keep it, but pin it to the
        # embed module explicitly so a non-cuda:0 embed is handled correctly.
        self.embed_device = _module_device(self.embed_tokens)
        self.device = self.embed_device

        # Per-layer execution device, resolved once (cheap lookup in _run_layers).
        self._layer_devices = [_module_device(layer) for layer in self.layers]

        # Wrap norm + lm_head so the parent's read tail lands hidden on their GPU.
        self.norm = _DeviceMovingCall(self.norm)
        self.lm_head = _DeviceMovingCall(self.lm_head)

        # For reporting / debugging.
        self.hf_device_map = getattr(model, "hf_device_map", None)
        self.is_sharded = len({str(d) for d in self._layer_devices}) > 1

    # ------------------------------------------------------------------ #
    # device-aware layer loop (the only override that runs the backbone)
    # ------------------------------------------------------------------ #
    def _run_layers(
        self,
        hidden: torch.Tensor,
        layer_slice: slice,
        causal_mask,
        positions: torch.Tensor,
        position_embeddings,
    ) -> torch.Tensor:
        """Run ``self.layers[layer_slice]`` moving ``(hidden, mask, position_ids,
        cos/sin)`` onto each layer's device before the call.

        Identical control flow to :meth:`QCMemModel._run_layers` (same kwargs, same
        gradient-checkpointing gate, same ``_layer_out_hidden`` unwrap) with a
        per-layer device hop inserted. On a single-device model every move is a
        no-op, so this is a strict superset of the parent behaviour.
        """
        use_ckpt = (
            self.grad_checkpoint
            and torch.is_grad_enabled()
            and hidden.requires_grad
        )
        start = 0 if layer_slice.start is None else layer_slice.start
        for offset, layer in enumerate(self.layers[layer_slice]):
            dev = self._layer_devices[start + offset]
            hidden = _to_device(hidden, dev)
            mask_d = _to_device(causal_mask, dev)
            pos_d = _to_device(positions, dev)
            pe_d = _to_device(position_embeddings, dev)
            if use_ckpt:
                out = torch.utils.checkpoint.checkpoint(
                    lambda h, _l=layer, _m=mask_d, _p=pos_d, _pe=pe_d: _l(
                        h,
                        attention_mask=_m,
                        position_ids=_p,
                        position_embeddings=_pe,
                        use_cache=False,
                    ),
                    hidden,
                    use_reentrant=False,
                )
            else:
                out = layer(
                    hidden,
                    attention_mask=mask_d,
                    position_ids=pos_d,
                    position_embeddings=pe_d,
                    use_cache=False,
                )
            hidden = self._layer_out_hidden(out)
        return hidden


# --------------------------------------------------------------------------- #
# loader
# --------------------------------------------------------------------------- #
def load_hy3_qcmem(
    model_path: str,
    resume_j: int,
    top_prepay_b: int = 0,
    block_diagonal: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
    attn_implementation: str = "sdpa",
    max_memory: Optional[dict] = None,
) -> QCMemHy3Model:
    """Load a sharded Hy3 (``HYV3ForCausalLM``) and wrap it in :class:`QCMemHy3Model`.

    Uses native ``transformers`` support for ``hy_v3`` (>= 5.13.1) — NO
    ``trust_remote_code`` — with ``device_map="auto"`` to split the 80 decoder
    layers across the visible GPUs. ``_no_split_modules=["HYV3DecoderLayer"]``
    guarantees each layer stays whole on one device (required for the per-layer
    hop in :meth:`QCMemHy3Model._run_layers`).
    """
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
        low_cpu_mem_usage=True,
        local_files_only=True,
        max_memory=max_memory,
    ).eval()
    return QCMemHy3Model(
        model,
        resume_j=resume_j,
        top_prepay_b=top_prepay_b,
        block_diagonal=block_diagonal,
    )
