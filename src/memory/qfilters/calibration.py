"""Q-Filters offline calibration.

Runs a forward pass on a small calibration corpus, captures the post-RoPE
query tensor Q_l for every attention layer `l`, and returns the top-`rank`
right-singular vectors of each per-head Q matrix. These filters identify
the dominant directions of the head's query space; projecting keys onto
them approximates the head's future attention scores.

API:
    filters = compute_filters(model, calib_loader, rank)
    # filters: dict[int, Tensor[num_heads, head_dim, rank]]
    torch.save(filters, "outputs/qfilters_baseline/filters.pt")
    filters = torch.load("outputs/qfilters_baseline/filters.pt")
"""
from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class _QCapture:
    """Forward-pre-hook bank that captures the post-RoPE Q of every LlamaAttention."""

    def __init__(self) -> None:
        self.buffers: Dict[int, List[torch.Tensor]] = {}
        self.handles: list = []

    # We hook q_proj forward (pre-RoPE). Post-RoPE Q differs only by position-
    # dependent rotation, which, averaged across many positions on a
    # long calibration corpus, leaves the dominant subspace unchanged (rotations
    # are orthogonal). The reference impl captures post-RoPE Q for exactness;
    # we use post-RoPE below too, via a forward hook on the whole attention
    # module's RoPE application. See _attach_post_rope_hook.
    def capture(self, layer_idx: int, t: torch.Tensor) -> None:
        # t: [B, H, T, D]  — cast to fp32 and move to CPU to bound memory.
        flat = t.detach().to(torch.float32).reshape(-1, t.shape[-2], t.shape[-1])
        # [B*H, T, D] collapsed per layer; split by head below.
        # Keep on CPU to survive multi-chunk calibration on a single GPU.
        self.buffers.setdefault(layer_idx, []).append(flat.cpu())

    def clear(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


def _collect_attn_layers(model: nn.Module):
    """Find all LlamaAttention-like modules and return (layer_idx, module)."""
    pairs = []
    # Standard Llama model path: model.model.layers[i].self_attn
    root = getattr(model, "model", model)
    layers = getattr(root, "layers", None)
    if layers is None:
        raise RuntimeError(
            "Could not locate model.model.layers; only Llama-family models supported."
        )
    for i, block in enumerate(layers):
        attn = getattr(block, "self_attn", None)
        if attn is None:
            continue
        pairs.append((i, attn))
    return pairs


def _attach_post_rope_hooks(model: nn.Module, cap: _QCapture) -> None:
    """Capture post-RoPE Q by wrapping the attention forward.

    Strategy: we use a *module forward hook* on each attention layer's q_proj
    to get pre-RoPE Q as a cheap fallback, AND a pre-forward hook on the
    attention module to capture the post-RoPE Q once the outer forward has
    applied RoPE. The cleanest cross-version approach is to monkey-patch each
    LlamaAttention.forward to stash `query_states` post-RoPE into `cap`.
    """
    import types

    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    for layer_idx, attn in _collect_attn_layers(model):

        def _make_forward(orig_forward, idx, module):
            def forward(self, hidden_states, position_embeddings=None,
                        attention_mask=None, past_key_values=None, **kwargs):
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, self.head_dim)
                query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                if position_embeddings is not None:
                    cos, sin = position_embeddings
                    query_states, key_states = apply_rotary_pos_emb(
                        query_states, key_states, cos, sin
                    )
                # <-- capture post-RoPE Q
                cap.capture(idx, query_states)
                # Restore call to the ORIGINAL forward so downstream code path
                # (attention impl, output proj) is untouched. We do it by
                # calling the original forward a second time -- slightly
                # wasteful, but calibration is tiny (few chunks).
                return orig_forward(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    **kwargs,
                )
            return forward

        orig = attn.forward
        attn._qfilters_orig_forward = orig
        attn.forward = types.MethodType(_make_forward(orig, layer_idx, attn), attn)
        cap.handles.append(
            # lambda-style "unhook": restore orig forward
            _UnhookForward(attn, orig)
        )


class _UnhookForward:
    def __init__(self, module, orig):
        self.module = module
        self.orig = orig

    def remove(self) -> None:
        self.module.forward = self.orig
        if hasattr(self.module, "_qfilters_orig_forward"):
            delattr(self.module, "_qfilters_orig_forward")


@torch.no_grad()
def compute_filters(
    model: nn.Module,
    calib_loader: Iterable,
    rank: int,
    num_kv_heads: Optional[int] = None,
    device: Optional[torch.device] = None,
    max_tokens_per_layer: int = 1_000_000,
) -> Dict[int, torch.Tensor]:
    """Run calibration and return per-layer Q-filters.

    Args:
        model: LlamaForCausalLM (eval mode, already on device).
        calib_loader: iterable of dicts with "input_ids" (Long tensor).
        rank: number of right-singular vectors to keep per head.
        num_kv_heads: if given, collapse the Q tensor from `num_attention_heads`
            down to `num_kv_heads` by averaging within each GQA group before
            SVD. This lets the filters be indexed by the cache's kv-head axis
            (which is what the compression path sees). If None, keep
            num_attention_heads.
        device: where the model lives.
        max_tokens_per_layer: cap on the number of (position, query) samples
            stacked per layer before SVD. Keeps memory bounded.

    Returns:
        dict layer_idx -> Tensor of shape [heads_out, head_dim, rank] (float32, CPU).
    """
    cap = _QCapture()
    _attach_post_rope_hooks(model, cap)
    try:
        model.eval()
        if device is None:
            device = next(model.parameters()).device
        n_chunks = 0
        for batch in calib_loader:
            input_ids = batch["input_ids"]
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            input_ids = input_ids.to(device)
            model(input_ids=input_ids)
            n_chunks += 1
        logger.info("calibration forward: %d chunks consumed", n_chunks)
    finally:
        cap.clear()

    # Now compute SVD per layer, per head.
    filters: Dict[int, torch.Tensor] = {}
    for layer_idx, chunks in cap.buffers.items():
        # chunks: list of [B*H, T, D] fp32 tensors (CPU)
        stacked = torch.cat([c for c in chunks], dim=1)      # along T: [B*H, T_total, D]
        bh, t_total, d = stacked.shape
        # Reconstruct (B, H, T, D) by assuming a single batch per capture; with
        # B=1 per chunk (our calib loader enforces this) we have bh == H.
        # If calib uses B>1 we would need to bookkeep, but we do not.
        num_heads = bh  # equals config.num_attention_heads
        q_per_head = stacked                                # [H, T_total, D]

        # Cap tokens to keep SVD memory bounded.
        if t_total > max_tokens_per_layer:
            stride = t_total // max_tokens_per_layer + 1
            q_per_head = q_per_head[:, ::stride, :]
        q_per_head = q_per_head.contiguous()

        # Optionally collapse to kv-head granularity.
        # GQA correctness: Q-Filters score KEYS, which HF's DynamicCache stores
        # per KV-head (shape [B, num_kv_heads, T, D]). So the filter tensor must
        # be keyed by KV-head index too. On Llama-2 (no GQA) the two counts are
        # equal and this branch is a no-op. On Llama-3.0-8B (32 Q heads, 8 KV
        # heads, group=4) we average Q within each group before SVD so the
        # filter subspace reflects the aggregate query geometry that every Q
        # head attending through that KV head will project from.
        if num_kv_heads is not None and num_kv_heads != num_heads:
            assert num_heads % num_kv_heads == 0, (
                f"num_attention_heads ({num_heads}) not divisible by "
                f"num_kv_heads ({num_kv_heads})"
            )
            group = num_heads // num_kv_heads
            q_per_head = q_per_head.view(num_kv_heads, group, -1, d).mean(dim=1)

        heads_out = q_per_head.shape[0]
        # SVD each head's [T, D] matrix -> top-`rank` right-singular vectors (D x R).
        out = torch.empty(heads_out, d, rank, dtype=torch.float32)

        # Issue #110 fix (2026-04-26): rank<=2 uses exact SVD to avoid
        # sign/direction ambiguity in torch.svd_lowrank(niter=2) that
        # caused kv=256 rank=1 Llama-2 PPL spread 161/752/788 across
        # identical configs. Per-head D=128 -> exact SVD is cheap; we
        # batch across heads on GPU so that calibration-on-rank-0 does
        # not exceed the NCCL barrier timeout that the other ranks hold.
        if rank <= 2:
            svd_device = device if device is not None else torch.device("cpu")
            # q_per_head: [H, T, D] fp32 on CPU. Batched SVD on GPU.
            q_dev = q_per_head.to(svd_device, dtype=torch.float32, non_blocking=True)
            try:
                U, S, Vh = torch.linalg.svd(q_dev, full_matrices=False)
                # Vh: [H, min(T,D), D] -> we want top-rank right-singular vectors:
                # top row(s) of Vh give them, transposed to [H, D, rank].
                v_all = Vh[:, :rank, :].mH.contiguous().to("cpu")  # [H, D, rank]
            except Exception as e:
                logger.warning(
                    "batched GPU SVD failed (%s); falling back to per-head CPU SVD", e,
                )
                v_all = torch.empty(heads_out, d, rank, dtype=torch.float32)
                for h in range(heads_out):
                    U, S, Vh = torch.linalg.svd(q_per_head[h], full_matrices=False)
                    v_all[h] = Vh.mH[:, :rank]
            v_all = torch.nan_to_num(v_all, nan=0.0, posinf=0.0, neginf=0.0)
            # Pad if short (shouldn't happen at rank<=2 unless T<rank)
            if v_all.shape[-1] < rank:
                pad = torch.zeros(heads_out, d, rank - v_all.shape[-1])
                v_all = torch.cat([v_all, pad], dim=-1)
            out.copy_(v_all)
        else:
            for h in range(heads_out):
                mat = q_per_head[h]                              # [T, D]
                try:
                    # svd_lowrank: returns U [T,R], S [R], V [D,R]
                    # niter raised 2 -> 7 for better subspace convergence.
                    _, _, V = torch.svd_lowrank(mat, q=rank, niter=7)
                    v = V
                    if v.shape[-1] < rank:
                        # pad with zeros to enforce fixed rank
                        pad = torch.zeros(d, rank - v.shape[-1])
                        v = torch.cat([v, pad], dim=-1)
                except Exception as e:
                    logger.warning(
                        "svd failed at layer %d head %d (%s); "
                        "falling back to torch.linalg.svd", layer_idx, h, e,
                    )
                    U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
                    v = Vh.mH[:, :rank]
                # Numerical hygiene.
                v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                out[h] = v
        filters[layer_idx] = out
        logger.info(
            "layer %d: filters tensor shape=%s, nnz rows=%d / %d",
            layer_idx, tuple(out.shape),
            int((out.abs().sum(dim=(1, 2)) > 0).sum().item()),
            heads_out,
        )

    return filters
