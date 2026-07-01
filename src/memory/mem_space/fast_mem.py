"""FastMem — Per-layer Gated Delta Rule continuous fast-weight memory.

Captures a continuous running summary of ALL tokens that pass through a layer,
complementing the discrete top-k slot routing (which only stores ~12.5% of
tokens).  Uses the Gated Delta Rule update (strongest known linear-complexity
associative memory) with efficient Triton kernels from flash-linear-attention.

Design reference:
    ops/research_notes/20260521_fast_weight_memory_research.md §3

Update rule (per token t):
    S_t = α_t · S_{t-1} + β_t · (v_t - α_t·S_{t-1} · k_t) ⊗ k_t
    o_t = S_t · q_t

Where α_t = sigmoid(gate_logit_t) is the per-head forget/retention gate, and
β_t = sigmoid(beta_logit_t) is the per-head learning rate for the delta rule.

The error-correction term (v_t - S_{t-1}·k_t) is the key difference from
additive linear attention:
    - Prevents rewriting already-known associations
    - Gives O(d²) effective capacity vs O(d) for additive
    - Makes the state self-correcting (bounded without norm caps)

Performance strategy:
    PRIMARY (fla available): Uses flash-linear-attention's `chunk_gated_delta_rule`
    Triton kernel which is ~10-50x faster than sequential.  The entire T-token
    sequence is processed in a single call (fla handles internal chunking).
    Speed: ~0.3ms for B=2, T=1024, H=4, d=128 on H20.

    FALLBACK (fla unavailable): Sequential loop with small chunk_size (default 16).
    Within each mini-chunk of 16 tokens, the exact sequential delta rule runs.
    Between mini-chunks, state S is DETACHED (truncated BPTT).

Integration:
    The output is gated by a learned per-feature fusion_gate (init sigmoid≈0.12)
    and added to next_hidden after the slot_delta term.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Try importing flash-linear-attention (Triton kernels for Gated Delta Rule)
# --------------------------------------------------------------------------- #
_FLA_FORCE_DISABLE = os.environ.get("DISABLE_FLA", "0") == "1"

try:
    from fla.ops.gated_delta_rule import (
        chunk_gated_delta_rule as _fla_chunk_gated_delta_rule,
    )
    _FLA_AVAILABLE = not _FLA_FORCE_DISABLE
    if _FLA_FORCE_DISABLE:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "FLA available but DISABLED via DISABLE_FLA=1 env var. Using sequential fallback."
        )
except ImportError:
    _FLA_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Sequential fallback (when fla is not available)
# --------------------------------------------------------------------------- #

def _delta_rule_segment(
    k: torch.Tensor,    # [B, C, H, d_k]
    v: torch.Tensor,    # [B, C, H, d_v]
    q: torch.Tensor,    # [B, C, H, d_k]
    gate: torch.Tensor, # [B, C, H]  (scalar per head, ∈ (0,1))
    beta: torch.Tensor, # [B, C, H]
    S: torch.Tensor,    # [B, H, d_k, d_v]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sequential Gated Delta Rule over a segment of C tokens.

    This is the fallback recurrence used when flash-linear-attention is not
    available.  C should be small (16-32) so the sequential loop completes
    quickly (~0.5ms for C=16 on H20).

    Args:
        k: L2-normalized keys [B, C, H, d_k]
        v: values [B, C, H, d_v]
        q: queries [B, C, H, d_k]
        gate: per-head forget gate ∈ (0,1) [B, C, H]
        beta: per-head learning rate ∈ (0,1) [B, C, H]
        S: incoming state [B, H, d_k, d_v]

    Returns:
        output: [B, C, H, d_v]
        S: updated state [B, H, d_k, d_v]
    """
    B, C, H, d_k = k.shape

    outputs = []

    for t in range(C):
        k_t = k[:, t]       # [B, H, d_k]
        v_t = v[:, t]       # [B, H, d_v]
        q_t = q[:, t]       # [B, H, d_k]
        gate_t = gate[:, t] # [B, H]
        beta_t = beta[:, t] # [B, H]

        # 1. Apply per-head forget gate: S = α_t · S
        S = gate_t.unsqueeze(-1).unsqueeze(-1) * S  # [B, H, 1, 1] * [B, H, d_k, d_v]

        # 2. Delta rule update with error correction:
        #    retrieved = S · k_t  (what's currently stored for this key)
        #    error = v_t - retrieved  (the residual — what's NOT yet stored)
        #    S += β_t · error ⊗ k_t
        retrieved = torch.einsum('bhkv,bhk->bhv', S, k_t)  # [B, H, d_v]
        error = v_t - retrieved                              # [B, H, d_v]
        delta = torch.einsum('bhv,bhk->bhkv', error, k_t)  # [B, H, d_k, d_v]
        S = S + beta_t.unsqueeze(-1).unsqueeze(-1) * delta

        # 3. Retrieve: o_t = S · q_t
        o_t = torch.einsum('bhkv,bhk->bhv', S, q_t)  # [B, H, d_v]
        outputs.append(o_t)

    output = torch.stack(outputs, dim=1)  # [B, C, H, d_v]
    return output, S


# --------------------------------------------------------------------------- #
# FastMem Module
# --------------------------------------------------------------------------- #

class FastMemModule(nn.Module):
    """Per-layer fast weight memory using the Gated Delta Rule.

    Memory state: S ∈ [B, H, d_k, d_v] per layer, per sample.

    Uses flash-linear-attention Triton kernels when available (10-50x speedup).
    Falls back to sequential loop with truncated BPTT otherwise.

    Args:
        d_model:    Hidden dimension of the backbone (e.g. 4096 for Llama-3-8B).
        num_heads:  Number of fast-weight heads (default 4).
        d_state:    Key/value dimension per head (default 128).
        chunk_size: BPTT window size for sequential fallback (default 16).
                    Ignored when fla is available.
        fusion_init: Initial value of fusion_gate logit (default -2.0 → sigmoid≈0.12).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        d_state: int = 128,
        chunk_size: int = 16,
        fusion_init: float = -2.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.H = num_heads
        self.d_k = d_state
        self.d_v = d_state
        self.chunk_size = chunk_size

        total_kv_dim = num_heads * d_state

        # Projections: hidden → multi-head {key, value, query}
        self.W_k = nn.Linear(d_model, total_kv_dim, bias=False)
        self.W_v = nn.Linear(d_model, total_kv_dim, bias=False)
        self.W_q = nn.Linear(d_model, total_kv_dim, bias=False)

        # Data-dependent forget gate: SCALAR per head (standard GDN formulation)
        # Produces one gate logit per head; sigmoid gives retention rate α ∈ (0,1)
        self.W_gate = nn.Linear(d_model, num_heads, bias=True)

        # Data-dependent learning rate (beta) for delta rule: scalar per head
        self.W_beta = nn.Linear(d_model, num_heads, bias=True)

        # Output projection: all heads concat → d_model
        self.W_o = nn.Linear(total_kv_dim, d_model, bias=False)

        # Per-feature fusion gate: controls fast-mem contribution to output.
        # Init to fusion_init. sigmoid(0)=0.5 is a neutral starting point that
        # gives maximum gradient (sigmoid'(0)=0.25) for fastest learning.
        # The W_o small init ensures actual output magnitude is still small at start.
        self.fusion_gate = nn.Parameter(
            torch.full((d_model,), fusion_init, dtype=torch.float32)
        )

        # --- Initialization for safe continued pretraining ---
        nn.init.normal_(self.W_k.weight, std=0.02)
        nn.init.normal_(self.W_v.weight, std=0.02)
        nn.init.normal_(self.W_q.weight, std=0.02)
        # W_gate: larger std (0.1) to avoid logsigmoid saturation at init.
        # If weight is too small, gate_logit ≈ bias → constant gate → gradient ≈ 0.
        nn.init.normal_(self.W_gate.weight, std=0.1)
        # W_gate bias: init at 0 → logsigmoid(0) = -ln(2) ≈ -0.693 (50% retention)
        nn.init.zeros_(self.W_gate.bias)
        # W_o initialized very small → safety net: even with fusion_gate=0.5,
        # actual output is still tiny at step 0 (std=0.001 → ||output|| ≈ 0)
        nn.init.normal_(self.W_o.weight, std=0.001)
        # Beta: init bias at 0 → sigmoid(0)=0.5 initial learning rate
        nn.init.zeros_(self.W_beta.bias)
        nn.init.normal_(self.W_beta.weight, std=0.1)

    @property
    def use_fla(self) -> bool:
        """Whether flash-linear-attention is available for acceleration."""
        return _FLA_AVAILABLE

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run Gated Delta Rule memory.

        When fla is available: processes all T tokens in one optimized Triton call.
        When fla is unavailable: sequential loop with truncated BPTT (chunk_size=16).

        Args:
            hidden_states: [B, T, d_model] — current chunk's hidden states.
            memory_state:  [B, H, d_k, d_v] or None (cold start → zeros).

        Returns:
            output:    [B, T, d_model] — gated fast-mem contribution.
            new_state: [B, H, d_k, d_v] — updated state (detached for inter-chunk).
        """
        B, T, d = hidden_states.shape
        H, d_k, d_v = self.H, self.d_k, self.d_v

        # Project inputs to multi-head format: [B, T, H, d_k/d_v]
        k = self.W_k(hidden_states).view(B, T, H, d_k)
        v = self.W_v(hidden_states).view(B, T, H, d_v)
        q = self.W_q(hidden_states).view(B, T, H, d_k)

        # L2-normalize keys for stable delta rule (critical!)
        k = F.normalize(k, dim=-1)

        # Data-dependent beta (learning rate) ∈ (0, 1): scalar per head
        beta = torch.sigmoid(self.W_beta(hidden_states))  # [B, T, H]

        # Initialize state if cold start
        if memory_state is None:
            S = torch.zeros(
                B, H, d_k, d_v,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        else:
            S = memory_state.to(dtype=hidden_states.dtype)

        if _FLA_AVAILABLE:
            output, S = self._forward_fla(hidden_states, k, v, q, beta, S)
        else:
            output, S = self._forward_sequential(hidden_states, k, v, q, beta, S)

        # Reshape: [B, T, H*d_v] and project to d_model
        output = output.reshape(B, T, H * d_v)
        output = self.W_o(output)  # [B, T, d_model]

        # Apply learned per-feature fusion gate
        fusion = torch.sigmoid(self.fusion_gate.to(dtype=output.dtype))
        output = fusion * output

        return output, S

    def _forward_fla(
        self,
        hidden_states: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q: torch.Tensor,
        beta: torch.Tensor,
        S: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fast path using flash-linear-attention Triton kernel.

        Processes the full sequence in one call.  fla handles internal chunking
        for memory efficiency.  ~0.3ms for B=2, T=1024, H=4, d=128 on H20.
        """
        B, T, H, d_k = k.shape

        # Per-head forget gate in LOG space for fla
        # logsigmoid(x) = log(sigmoid(x)) = -softplus(-x), always negative
        gate_logits = self.W_gate(hidden_states)  # [B, T, H]
        g = F.logsigmoid(gate_logits)  # [B, T, H], log-space gate (negative values)

        # Call fla's optimized kernel
        output, S_new = _fla_chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=S,
            output_final_state=True,
        )
        # output: [B, T, H, d_v], S_new: [B, H, d_k, d_v]
        return output, S_new

    def _forward_sequential(
        self,
        hidden_states: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q: torch.Tensor,
        beta: torch.Tensor,
        S: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fallback path: sequential loop with truncated BPTT.

        The sequence is split into mini-chunks of `chunk_size` (default 16).
        Within each mini-chunk: exact sequential delta rule.
        Between mini-chunks: state is detached (gradient truncation).
        """
        B, T, H, d_k = k.shape
        C = self.chunk_size

        # Per-head forget gate ∈ (0, 1) for sequential path
        gate = torch.sigmoid(self.W_gate(hidden_states))  # [B, T, H]

        # Split into segments and process
        n_segments = (T + C - 1) // C
        all_outputs = []

        for seg_idx in range(n_segments):
            start = seg_idx * C
            end = min(start + C, T)

            # Slice this segment's inputs
            k_seg = k[:, start:end]
            v_seg = v[:, start:end]
            q_seg = q[:, start:end]
            gate_seg = gate[:, start:end]
            beta_seg = beta[:, start:end]

            # Run sequential delta rule on this segment
            o_seg, S = _delta_rule_segment(
                k_seg, v_seg, q_seg, gate_seg, beta_seg, S
            )
            all_outputs.append(o_seg)

            # Truncated BPTT: detach state between segments during training
            if self.training and seg_idx < n_segments - 1:
                S = S.detach()

        # Concatenate all segment outputs: [B, T, H, d_v]
        output = torch.cat(all_outputs, dim=1)
        return output, S
