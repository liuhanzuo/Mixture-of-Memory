"""FastMem — Per-layer Gated Delta Rule continuous fast-weight memory.

Captures a continuous running summary of ALL tokens that pass through a layer,
complementing the discrete top-k slot routing (which only stores ~12.5% of
tokens).  Uses the Gated Delta Rule update (strongest known linear-complexity
associative memory) with truncated-BPTT sequential processing.

Design reference:
    ops/research_notes/20260521_fast_weight_memory_research.md §3

Update rule (per token t):
    S_t = gate_t ⊙ S_{t-1} + β_t · (v_t - S_{t-1} · k_t) ⊗ k_t
    o_t = S_t · q_t

The error-correction term (v_t - S_{t-1}·k_t) is the key difference from
additive linear attention:
    - Prevents rewriting already-known associations
    - Gives O(d²) effective capacity vs O(d) for additive
    - Makes the state self-correcting (bounded without norm caps)

Performance strategy:
    The 1024-token sequence is split into sub-segments of `chunk_size` (default
    64).  Within each segment, the exact sequential delta rule runs (64
    iterations of einsum — fast enough).  Between segments, state S is
    DETACHED from the computation graph (truncated BPTT).  This means:
    - Forward: EXACT (correct delta rule, full state propagation)
    - Backward: gradient flows through at most `chunk_size` steps
    - Memory: O(chunk_size) intermediate states per segment

    With chunk_size=64 and 1024 tokens: 16 segments, each with 64-step BPTT.
    The projections (W_k, W_v, W_q, W_gate, W_beta, W_o) get gradient from ALL
    tokens (just with truncated state gradient).

Integration:
    The output is gated by a learned per-feature fusion_gate (init sigmoid≈0.12)
    and added to next_hidden after the slot_delta term.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _delta_rule_segment(
    k: torch.Tensor,    # [B, C, H, d_k]
    v: torch.Tensor,    # [B, C, H, d_v]
    q: torch.Tensor,    # [B, C, H, d_k]
    gate: torch.Tensor, # [B, C, H, d_k]
    beta: torch.Tensor, # [B, C, H]
    S: torch.Tensor,    # [B, H, d_k, d_v]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sequential Gated Delta Rule over a segment of C tokens.

    This is the core recurrence.  C should be moderate (32-128) so the
    sequential loop is fast and the backward pass (autograd through C steps)
    doesn't blow up memory.

    Args:
        k: L2-normalized keys [B, C, H, d_k]
        v: values [B, C, H, d_v]
        q: queries [B, C, H, d_k]
        gate: per-feature forget gate ∈ (0,1) [B, C, H, d_k]
        beta: per-head learning rate ∈ (0,1) [B, C, H]
        S: incoming state [B, H, d_k, d_v]

    Returns:
        output: [B, C, H, d_v]
        S: updated state [B, H, d_k, d_v]
    """
    B, C, H, d_k = k.shape
    d_v = v.shape[-1]

    outputs = []

    for t in range(C):
        k_t = k[:, t]       # [B, H, d_k]
        v_t = v[:, t]       # [B, H, d_v]
        q_t = q[:, t]       # [B, H, d_k]
        gate_t = gate[:, t] # [B, H, d_k]
        beta_t = beta[:, t] # [B, H]

        # 1. Apply per-feature forget gate: S = gate_t ⊙ S
        S = gate_t.unsqueeze(-1) * S  # [B, H, d_k, d_v]

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


class FastMemModule(nn.Module):
    """Per-layer fast weight memory using the Gated Delta Rule.

    Memory state: S ∈ [B, H, d_k, d_v] per layer, per sample.

    Args:
        d_model:    Hidden dimension of the backbone (e.g. 4096 for Llama-3-8B).
        num_heads:  Number of fast-weight heads (default 4).
        d_state:    Key/value dimension per head (default 128).
        chunk_size: BPTT window size — gradient flows through this many tokens
                    before being truncated.  Default 64 gives a good speed/gradient
                    trade-off.  Forward is always exact regardless of this value.
        fusion_init: Initial value of fusion_gate logit (default -2.0 → sigmoid≈0.12).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        d_state: int = 128,
        chunk_size: int = 64,
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

        # Data-dependent forget gate: per-head, per-key-feature
        self.W_gate = nn.Linear(d_model, total_kv_dim, bias=False)

        # Data-dependent learning rate (beta) for delta rule: scalar per head
        self.W_beta = nn.Linear(d_model, num_heads, bias=True)

        # Output projection: all heads concat → d_model
        self.W_o = nn.Linear(total_kv_dim, d_model, bias=False)

        # Per-feature fusion gate: controls fast-mem contribution to output
        # Init to fusion_init so sigmoid(fusion_init)≈0.12 at start
        self.fusion_gate = nn.Parameter(
            torch.full((d_model,), fusion_init, dtype=torch.float32)
        )

        # --- Initialization for safe continued pretraining ---
        # Small std so outputs are near-zero at init (combined with fusion_gate)
        nn.init.normal_(self.W_k.weight, std=0.02)
        nn.init.normal_(self.W_v.weight, std=0.02)
        nn.init.normal_(self.W_q.weight, std=0.02)
        nn.init.normal_(self.W_gate.weight, std=0.02)
        # W_o initialized very small → even with non-trivial S·q, output ≈ 0
        nn.init.normal_(self.W_o.weight, std=0.01)
        # Beta bias: init near 0 → sigmoid(0)=0.5 initial beta (moderate update rate)
        nn.init.zeros_(self.W_beta.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run Gated Delta Rule with truncated-BPTT chunking.

        Forward pass is EXACT (correct delta rule, state propagates through all
        T tokens).  Backward pass is truncated: gradients flow through at most
        `chunk_size` sequential steps, then S is detached.

        Args:
            hidden_states: [B, T, d_model] — current chunk's hidden states.
            memory_state:  [B, H, d_k, d_v] or None (cold start → zeros).

        Returns:
            output:    [B, T, d_model] — gated fast-mem contribution.
            new_state: [B, H, d_k, d_v] — updated state (detached for inter-chunk).
        """
        B, T, d = hidden_states.shape
        H, d_k, d_v = self.H, self.d_k, self.d_v
        C = self.chunk_size

        # Project inputs to multi-head format
        k = self.W_k(hidden_states).view(B, T, H, d_k)   # [B, T, H, d_k]
        v = self.W_v(hidden_states).view(B, T, H, d_v)   # [B, T, H, d_v]
        q = self.W_q(hidden_states).view(B, T, H, d_k)   # [B, T, H, d_k]

        # L2-normalize keys for stable delta rule (critical!)
        k = F.normalize(k, dim=-1)

        # Data-dependent forget gate ∈ (0, 1): per-head, per-feature
        gate = torch.sigmoid(
            self.W_gate(hidden_states).view(B, T, H, d_k)
        )  # [B, T, H, d_k]

        # Data-dependent delta rule learning rate ∈ (0, 1): scalar per head
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

        # Truncated-BPTT: split into segments of chunk_size, detach S between
        # segments.  Forward is exact; backward is truncated.
        n_segments = (T + C - 1) // C  # ceiling division
        all_outputs = []

        for seg_idx in range(n_segments):
            start = seg_idx * C
            end = min(start + C, T)

            # Slice this segment's inputs
            k_seg = k[:, start:end]       # [B, seg_len, H, d_k]
            v_seg = v[:, start:end]       # [B, seg_len, H, d_v]
            q_seg = q[:, start:end]       # [B, seg_len, H, d_k]
            gate_seg = gate[:, start:end] # [B, seg_len, H, d_k]
            beta_seg = beta[:, start:end] # [B, seg_len, H]

            # Run sequential delta rule on this segment
            o_seg, S = _delta_rule_segment(k_seg, v_seg, q_seg, gate_seg, beta_seg, S)
            all_outputs.append(o_seg)  # [B, seg_len, H, d_v]

            # Truncated BPTT: detach state between segments during training
            # (forward value is preserved exactly, only gradient is truncated)
            if self.training and seg_idx < n_segments - 1:
                S = S.detach()

        # Concatenate all segment outputs
        output = torch.cat(all_outputs, dim=1)  # [B, T, H, d_v]

        # Reshape: [B, T, H*d_v] and project to d_model
        output = output.reshape(B, T, H * d_v)
        output = self.W_o(output)  # [B, T, d_model]

        # Apply learned per-feature fusion gate
        fusion = torch.sigmoid(self.fusion_gate.to(dtype=output.dtype))
        output = fusion * output

        return output, S
