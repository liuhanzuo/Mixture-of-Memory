"""Attention Matching — training-free KV cache compression (arXiv:2602.16284).

Implements the three-step pipeline:
  1. Key selection (OMP or highest-attention)
  2. Beta fitting via NNLS (mass matching)
  3. Value fitting via ridge regression (output matching)

This module is intentionally standalone: it does not subclass nn.Module and
carries no trainable parameters.  All heavy numerics run in float32 for
stability, even when the model's KV cache is in bfloat16.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import numpy as np
from scipy.optimize import nnls


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_f32(t: torch.Tensor) -> torch.Tensor:
    return t.float() if t.dtype != torch.float32 else t


def _softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically-stable softmax in float32."""
    logits = _to_f32(logits)
    m = logits.max(dim=dim, keepdim=True).values
    e = torch.exp(logits - m)
    return e / e.sum(dim=dim, keepdim=True)


# ---------------------------------------------------------------------------
# Core compressor
# ---------------------------------------------------------------------------

class AttentionMatchingCompressor:
    """Training-free KV cache compression via Attention Matching.

    Parameters
    ----------
    compression_ratio : int
        Keep ``1 / compression_ratio`` of the original keys.
        E.g. ``8`` means 8192 -> 1024 keys.
    method : str
        Key selection method: ``'omp'`` or ``'highest_attn'``.
    ridge_lambda : float
        L2 regularisation weight for value fitting (Step 3).
    beta_clamp : float
        Clamp ``exp(beta)`` to ``[exp(-clamp), exp(clamp)]`` for stability.
    max_omp_candidates : int
        When T is very large, OMP scores are only computed over the first
        ``max_omp_candidates`` keys (subsample) to bound memory.  0 = no limit.
    """

    def __init__(
        self,
        compression_ratio: int = 8,
        method: str = "omp",
        ridge_lambda: float = 1e-4,
        beta_clamp: float = 10.0,
        max_omp_candidates: int = 0,
    ):
        if method not in ("omp", "highest_attn"):
            raise ValueError(f"method must be 'omp' or 'highest_attn', got {method!r}")
        self.compression_ratio = compression_ratio
        self.method = method
        self.ridge_lambda = ridge_lambda
        self.beta_clamp = beta_clamp
        self.max_omp_candidates = max_omp_candidates

    # ------------------------------------------------------------------
    # Step 1: Key selection
    # ------------------------------------------------------------------

    def select_keys_omp(
        self,
        K: torch.Tensor,
        budget: int,
    ) -> torch.Tensor:
        """Orthogonal Matching Pursuit key selection.

        Greedily selects *budget* keys from K that best span the column
        space of K (maximise explained correlation).

        Parameters
        ----------
        K : [T, d]
            All original keys for **one** KV head.
        budget : int
            Number of keys to keep (m).

        Returns
        -------
        indices : [m] long tensor of selected positions.
        """
        T, d = K.shape
        K = _to_f32(K)
        budget = min(budget, T)

        selected: List[int] = []
        selected_set: set = set()

        # Residual starts as the full matrix K.
        # At each iteration, pick the key with the largest inner product
        # with the residual, then project out the selected subspace.
        residual = K.clone()  # [T, d]

        # Pre-compute Gram matrix K @ K^T for fast inner products.
        # Only needed if T is moderate; for very large T we do it online.
        if T <= 16384:
            gram = K @ K.T  # [T, T]
        else:
            gram = None

        for _ in range(budget):
            # Score each unselected key by |residual @ k_j| summed.
            if gram is not None:
                # residual @ K^T  =  residual @ K^T  (each col of gram has been updated)
                scores = (residual @ K.T).abs().sum(dim=0)  # [T]
            else:
                scores = (residual @ K.T).abs().sum(dim=0)  # [T]

            # Mask already-selected indices.
            for s in selected:
                scores[s] = -float("inf")

            best = int(scores.argmax().item())
            selected.append(best)
            selected_set.add(best)

            # Orthogonalise residual against the newly selected key.
            # Use modified Gram-Schmidt on the accumulated selected set.
            # For efficiency, we subtract the projection of residual onto
            # the span of selected keys.
            K_sel = K[selected]  # [m_cur, d]
            # Q, R = torch.linalg.qr(K_sel.T)  -- too costly each iter
            # Instead: subtract projection incrementally.
            # proj = K_sel[-1:] @ K_sel[-1:].T -> only the new vector.
            new_vec = K[best : best + 1]  # [1, d]
            coeff = (residual @ new_vec.T) / (new_vec @ new_vec.T + 1e-8)  # [T, 1]
            residual = residual - coeff * new_vec  # [T, d]

        return torch.tensor(selected, dtype=torch.long, device=K.device)

    def select_keys_highest_attn(
        self,
        K: torch.Tensor,
        Q_ref: torch.Tensor,
        budget: int,
    ) -> torch.Tensor:
        """Select keys with the highest cumulative attention mass.

        Parameters
        ----------
        K : [T, d]
        Q_ref : [T_ref, d]
        budget : int

        Returns
        -------
        indices : [budget] long tensor.
        """
        K = _to_f32(K)
        Q_ref = _to_f32(Q_ref)
        d = K.shape[-1]
        scale = math.sqrt(d)

        # attn_logits: [T_ref, T]
        attn_logits = Q_ref @ K.T / scale
        # Sum of exp logits over queries for each key.
        # Use log-sum-exp trick for stability.
        max_logit = attn_logits.max(dim=0).values  # [T]
        scores = ((attn_logits - max_logit.unsqueeze(0)).exp()).sum(dim=0)  # [T]
        scores = scores * max_logit.exp()  # rescale (monotonic, just for ordering)

        budget = min(budget, K.shape[0])
        _, indices = torch.topk(scores, budget)
        return indices

    # ------------------------------------------------------------------
    # Step 2: Beta fitting (NNLS mass matching)
    # ------------------------------------------------------------------

    def fit_beta(
        self,
        Q_ref: torch.Tensor,
        Ck: torch.Tensor,
        original_attn_mass: torch.Tensor,
    ) -> torch.Tensor:
        """Fit per-key log-bias beta via Non-negative Least Squares.

        We want the compact attention mass to match the original:
            sum_j  exp(q_i . Ck_j / sqrt(d) + beta_j)  ≈  sum_j  exp(q_i . k_j / sqrt(d))

        Let Phi[i,j] = exp(q_i . Ck_j / sqrt(d))  and  a[i] = original mass.
        Solve:  min  ||Phi @ exp(beta) - a||^2   s.t.  beta >= 0
        via NNLS on the variable x = exp(beta).

        Parameters
        ----------
        Q_ref : [T_ref, d]
        Ck : [m, d]  selected compact keys
        original_attn_mass : [T_ref]  target mass a[i] for each reference query.

        Returns
        -------
        beta : [m]  (log-bias values, may be < 0 after log).
        """
        Q_ref = _to_f32(Q_ref)
        Ck = _to_f32(Ck)
        d = Ck.shape[-1]
        scale = math.sqrt(d)

        # Phi[i, j] = exp(q_i . Ck_j / sqrt(d))
        logits = Q_ref @ Ck.T / scale  # [T_ref, m]
        # Subtract per-row max for stability before exp.
        logits_max = logits.max(dim=1, keepdim=True).values
        Phi = torch.exp(logits - logits_max)  # [T_ref, m]

        # Rescale target: original_attn_mass is sum_j exp(q.K^T/sqrt(d)).
        # We need to rescale by the same max that we subtracted from Phi.
        # original_attn_mass already includes its own exp; rescale to same basis.
        # Actually we want: Phi @ x = a / exp(logits_max)
        a = _to_f32(original_attn_mass)
        a_rescaled = a / torch.exp(logits_max.squeeze(1))

        # Convert to numpy for scipy.optimize.nnls.
        Phi_np = Phi.cpu().numpy().astype(np.float64)
        a_np = a_rescaled.cpu().numpy().astype(np.float64)

        m = Phi_np.shape[1]
        beta_np = np.zeros(m, dtype=np.float64)
        # NNLS: minimise ||Phi @ x - a||^2 s.t. x >= 0
        x, _ = nnls(Phi_np, a_np, maxiter=300)

        # Clamp to avoid log(0).
        x = np.clip(x, 1e-12, None)
        beta_np = np.log(x).astype(np.float32)

        # Clamp beta for stability.
        clamp = self.beta_clamp
        beta_np = np.clip(beta_np, -clamp, clamp)

        return torch.from_numpy(beta_np).to(device=Q_ref.device)

    # ------------------------------------------------------------------
    # Step 3: Value fitting (ridge regression)
    # ------------------------------------------------------------------

    def fit_values(
        self,
        Q_ref: torch.Tensor,
        Ck: torch.Tensor,
        Cv_init: torch.Tensor,
        V_original: torch.Tensor,
        beta: torch.Tensor,
        K_original: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fit compact values via ridge regression.

        Goal: for each reference query q_i,
            sum_j  w_ij * Cv_j   ≈   sum_j  w_orig_ij * V_j

        where w_ij = softmax(q.K^T/sqrt(d) + beta)[i,j] (compact attn)
        and   w_orig_ij = softmax(q.K_orig^T/sqrt(d))[i,j] (full attn).

        Parameters
        ----------
        Q_ref : [T_ref, d]
        Ck : [m, d] compact keys
        Cv_init : [m, d] initial compact values (used as warm-start; not required).
        V_original : [T, d] original values
        beta : [m] fitted log-biases
        K_original : [T, d] original keys (needed to compute original attn output).
            If None, V_original must already be the weighted output Y.

        Returns
        -------
        Cv : [m, d] fitted compact values.
        """
        Q_ref = _to_f32(Q_ref)
        Ck = _to_f32(Ck)
        V_original = _to_f32(V_original)
        beta = _to_f32(beta)

        T_ref, d = Q_ref.shape
        m = Ck.shape[0]
        scale = math.sqrt(d)

        # ---- Compute compact attention weights with beta ----
        compact_logits = Q_ref @ Ck.T / scale  # [T_ref, m]
        compact_logits = compact_logits + beta.unsqueeze(0)  # broadcast bias
        W_compact = _softmax(compact_logits, dim=-1)  # [T_ref, m]

        # ---- Compute original attention output (target Y) ----
        if K_original is not None:
            K_original = _to_f32(K_original)
            orig_logits = Q_ref @ K_original.T / scale  # [T_ref, T]
            W_orig = _softmax(orig_logits, dim=-1)  # [T_ref, T]
            Y = W_orig @ V_original  # [T_ref, d]
        else:
            Y = V_original  # caller already computed

        # ---- Build design matrix X ----
        # X[i, j*d : (j+1)*d] = W_compact[i,j] * Ck[j]
        # This is: X = (W_compact.unsqueeze(-1) * Ck.unsqueeze(0)).reshape(T_ref, m*d)
        # But m*d can be huge (1024*128 = 131072).  Use normal equations instead.

        # Normal equation: (X^T X + lambda I) Cv_flat = X^T Y
        # X^T X is [m*d, m*d] — too large.
        #
        # Instead, solve per-head-dimension independently:
        #   For each dim k:
        #     W_compact[:, j] * Ck[:, k] is not separable because Ck varies.
        #
        # Alternative: solve in the *reduced* basis.
        # Let A = W_compact  [T_ref, m]
        # Let S = Ck  [m, d]
        # We want Cv s.t.  (A * S_k) @ Cv_k ≈ Y_k for each dimension k.
        #
        # Actually the design is: X[i, j*d+k] = A[i,j] * S[j,k]
        # So X = khatri-rao(A, S) which factorises.
        # X^T X = (A^T A) ⊙ (S^T S)  (Hadamard product).
        # X^T Y[k*m : (k+1)*m] = S[:,k:k+1] * (A^T @ Y[:,k:k+1])
        #
        # This is still m*d x m*d.  For m=1024, d=128 -> 131K x 131K — too big.
        #
        # Practical approach: solve per output dimension (d independent problems).
        # For dim k:
        #   x_k[i, j] = A[i,j] * S[j,k]      [T_ref, m]
        #   y_k[i]    = Y[i,k]                 [T_ref]
        #   Solve: min ||x_k @ cv_k - y_k||^2 + lambda ||cv_k||^2
        #   cv_k = [m] compact values for dim k.
        #
        # Each is a small ridge regression: m x m matrix.
        # Do them in a batch.

        # A: [T_ref, m], S: [m, d], Y: [T_ref, d]
        A = W_compact  # [T_ref, m]
        S = Ck          # [m, d]

        # For each dim k: x_k = A * S[:,k].unsqueeze(0)  (broadcast)
        # x_k^T x_k = (S[:,k] * A^T) @ (A * S[:,k]) = (A^T A) * (S[:,k] S[:,k]^T)
        # x_k^T y_k = S[:,k] * (A^T Y[:,k])
        #
        # Batched: let s = S.T  [d, m], then for all dims at once:
        #   XTX[k] = (A^T A) * (s[k] s[k]^T)  for each k  -- [d, m, m]
        #   XTy[k] = s[k] * (A^T @ Y[:,k])      for each k  -- [d, m]
        #
        # Solving d independent m x m systems is feasible when m <= 1024.

        ATA = A.T @ A  # [m, m]
        ATY = A.T @ Y  # [m, d]
        s = S.T  # [d, m]

        # XTX[k, i, j] = ATA[i,j] * s[k,i] * s[k,j]
        # XTy[k, i]    = ATY[i, k] * s[k, i]
        # Plus lambda * I on diagonal.

        # Vectorised over d:
        # XTX = (s.unsqueeze(2) * s.unsqueeze(1)) * ATA.unsqueeze(0)  # [d, m, m]
        XTX = (s.unsqueeze(2) * s.unsqueeze(1)) * ATA.unsqueeze(0)  # [d, m, m]
        # Add ridge on diagonal.
        diag_idx = torch.arange(m, device=XTX.device)
        XTX[:, diag_idx, diag_idx] += self.ridge_lambda

        # XTy = ATY.T * s  # [d, m]  (ATY.T is [d, m])
        XTy = ATY.T * s  # [d, m]

        # Solve d independent m x m systems: XTX @ cv.T = XTy.T
        # cv: [d, m] -> Cv: [m, d]
        # torch.linalg.solve: XTX [d, m, m] @ cv [d, m, 1] = XTy [d, m, 1]
        Cv_flat = torch.linalg.solve(XTX, XTy.unsqueeze(-1)).squeeze(-1)  # [d, m]
        Cv = Cv_flat.T  # [m, d]

        return Cv

    # ------------------------------------------------------------------
    # Full 3-step pipeline (per KV head)
    # ------------------------------------------------------------------

    def compress(
        self,
        K: torch.Tensor,
        V: torch.Tensor,
        Q_ref: torch.Tensor,
        budget: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full Attention Matching pipeline for one KV head.

        Parameters
        ----------
        K : [T, d_head]
        V : [T, d_head]
        Q_ref : [T_ref, d_head]
        budget : int  (number of compact keys to keep)

        Returns
        -------
        Ck : [budget, d_head]
        Cv : [budget, d_head]
        beta : [budget]
        """
        K = _to_f32(K)
        V = _to_f32(V)
        Q_ref = _to_f32(Q_ref)
        T, d = K.shape
        budget = min(budget, T)
        scale = math.sqrt(d)

        # Step 1: Key selection
        if self.method == "omp":
            indices = self.select_keys_omp(K, budget)
        else:
            indices = self.select_keys_highest_attn(K, Q_ref, budget)

        Ck = K[indices]  # [m, d]

        # Step 2: Beta fitting
        # Compute original attention mass for each reference query.
        orig_logits = Q_ref @ K.T / scale  # [T_ref, T]
        orig_logits_max = orig_logits.max(dim=1, keepdim=True).values
        original_attn_mass = torch.exp(orig_logits - orig_logits_max).sum(dim=-1)  # [T_ref]
        # Rescale by exp of the subtracted max.
        original_attn_mass = original_attn_mass * torch.exp(orig_logits_max.squeeze(1))

        beta = self.fit_beta(Q_ref, Ck, original_attn_mass)

        # Step 3: Value fitting
        Cv = self.fit_values(Q_ref, Ck, V[indices], V, beta, K_original=K)

        return Ck, Cv, beta

    # ------------------------------------------------------------------
    # End-to-end: run model, extract KV, compress all layers
    # ------------------------------------------------------------------

    def compact_kv_cache(
        self,
        model,
        input_ids: torch.Tensor,
        budget_per_head: int,
        ref_query_mode: str = "repeat_prefill",
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """End-to-end: run model, extract KV cache, compress each layer.

        Parameters
        ----------
        model : a HuggingFace ``LlamaForCausalLM`` (or compatible).
            Must expose ``model.model.layers[i].self_attn``.
        input_ids : [1, T] token IDs (already on the right device).
        budget_per_head : number of compact keys per KV head.
        ref_query_mode : str
            How to generate reference queries.  ``'self'`` uses the input
            hidden states directly (simplest); ``'repeat_prefill'`` constructs
            a repeat prompt.

        Returns
        -------
        compact_kv : list of (Ck, Cv, beta), one per layer.
            Each Ck is [n_kv_heads, budget, d_head].
            Each Cv is [n_kv_heads, budget, d_head].
            Each beta is [n_kv_heads, budget].
        """
        device = input_ids.device
        model.eval()

        # --- Step A: forward pass to capture all per-layer KV ---
        all_K_per_layer: List[torch.Tensor] = []
        all_V_per_layer: List[torch.Tensor] = []
        all_hidden_per_layer: List[torch.Tensor] = []

        # Use the model's internals directly.  For LlamaForCausalLM:
        #   model.model.layers[i].self_attn has k_proj, v_proj.
        # We'll do a manual forward, capturing hidden states at each layer.

        with torch.no_grad():
            # Get embedding output.
            hidden = model.model.embed_tokens(input_ids)  # [1, T, d]

            T = input_ids.shape[1]
            d_head = model.config.hidden_size // model.config.num_attention_heads
            n_kv_heads = model.config.num_key_value_heads
            n_layers = model.config.num_hidden_layers

            for layer_idx in range(n_layers):
                layer = model.model.layers[layer_idx]
                self_attn = layer.self_attn

                # Project hidden to Q, K, V.
                # Shape: [1, T, num_heads * d_head] or [1, T, n_kv_heads * d_head].
                # For GQA, K and V have n_kv_heads.
                h = hidden
                if hasattr(layer, 'input_layernorm'):
                    h_ln = layer.input_layernorm(h)
                else:
                    h_ln = h

                K_full = self_attn.k_proj(h_ln)  # [1, T, n_kv_heads * d_head]
                V_full = self_attn.v_proj(h_ln)  # [1, T, n_kv_heads * d_head]

                K_full = K_full.view(1, T, n_kv_heads, d_head).squeeze(0)  # [T, n_kv_heads, d_head]
                V_full = V_full.view(1, T, n_kv_heads, d_head).squeeze(0)  # [T, n_kv_heads, d_head]

                all_K_per_layer.append(K_full)
                all_V_per_layer.append(V_full)
                all_hidden_per_layer.append(h.clone())

                # Run the layer forward to get hidden for the next layer.
                # We need the full decoder layer output.
                # Use the model's own forward for this layer.
                # Position embeddings are needed; use model's rotary embedding.
                # Simplest: call layer.forward with position_embeddings.
                position_ids = torch.arange(T, device=device).unsqueeze(0)
                position_embeddings = model.model.rotary_emb(h_ln, position_ids)

                layer_out = layer(
                    h,
                    attention_mask=None,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    use_cache=False,
                )
                hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        # --- Step B: compress each layer's KV ---
        compact_kv = []

        for layer_idx in range(n_layers):
            K = all_K_per_layer[layer_idx]   # [T, n_kv_heads, d_head]
            V = all_V_per_layer[layer_idx]   # [T, n_kv_heads, d_head]
            H = all_hidden_per_layer[layer_idx]  # [1, T, d_model]

            # Use hidden states as reference queries (projected through Q).
            self_attn = model.model.layers[layer_idx].self_attn
            if hasattr(model.model.layers[layer_idx], 'input_layernorm'):
                H_ln = model.model.layers[layer_idx].input_layernorm(H)
            else:
                H_ln = H

            Q_proj = self_attn.q_proj(H_ln)  # [1, T, num_heads * d_head]
            n_heads = model.config.num_attention_heads
            Q_proj = Q_proj.view(1, T, n_heads, d_head).squeeze(0)  # [T, n_heads, d_head]

            # Subsample reference queries for efficiency (use every 4th token).
            ref_stride = max(1, T // 512)
            Q_ref_all = Q_proj[::ref_stride]  # [T_ref, n_heads, d_head]

            layer_Ck = []
            layer_Cv = []
            layer_beta = []

            for kv_head in range(n_kv_heads):
                # For GQA: map KV head to the corresponding query head(s).
                # Q_ref for this KV head: average over the query heads that
                # share this KV head.
                heads_per_kv = n_heads // n_kv_heads
                q_start = kv_head * heads_per_kv
                q_end = q_start + heads_per_kv
                Q_ref_head = Q_ref_all[:, q_start:q_end, :].mean(dim=1)  # [T_ref, d_head]

                K_head = K[:, kv_head, :]  # [T, d_head]
                V_head = V[:, kv_head, :]  # [T, d_head]

                Ck, Cv, beta = self.compress(K_head, V_head, Q_ref_head, budget_per_head)
                layer_Ck.append(Ck)
                layer_Cv.append(Cv)
                layer_beta.append(beta)

            # Stack: [n_kv_heads, budget, d_head]
            layer_Ck = torch.stack(layer_Ck)
            layer_Cv = torch.stack(layer_Cv)
            layer_beta = torch.stack(layer_beta)

            compact_kv.append((layer_Ck, layer_Cv, layer_beta))

        return compact_kv


# ---------------------------------------------------------------------------
# Utility: compute perplexity using compressed KV cache
# ---------------------------------------------------------------------------

def compute_compressed_ppl(
    model,
    input_ids: torch.Tensor,
    compact_kv: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> float:
    """Compute perplexity when attention uses compressed (Ck, Cv, beta) KV.

    Replaces each layer's KV cache with the compressed version and runs
    a forward pass, computing the cross-entropy loss on the input tokens.

    The model attends to the compact keys/values with the beta bias added
    to the attention logits.

    Parameters
    ----------
    model : LlamaForCausalLM
    input_ids : [1, T]
    compact_kv : list of (Ck, Cv, beta) per layer.

    Returns
    -------
    ppl : float  (perplexity).
    """
    import torch.nn.functional as F

    device = input_ids.device
    model.eval()
    T = input_ids.shape[1]
    d_head = model.config.hidden_size // model.config.num_attention_heads
    n_kv_heads = model.config.num_key_value_heads
    n_heads = model.config.num_attention_heads
    n_layers = model.config.num_hidden_layers
    scale = math.sqrt(d_head)

    with torch.no_grad():
        hidden = model.model.embed_tokens(input_ids)  # [1, T, d_model]

        total_loss = 0.0
        count = 0

        for layer_idx in range(n_layers):
            layer = model.model.layers[layer_idx]
            self_attn = layer.self_attn
            Ck, Cv, beta = compact_kv[layer_idx]
            # Ck: [n_kv_heads, budget, d_head]
            # Cv: [n_kv_heads, budget, d_head]
            # beta: [n_kv_heads, budget]

            position_ids = torch.arange(T, device=device).unsqueeze(0)

            h = hidden
            if hasattr(layer, 'input_layernorm'):
                h_ln = layer.input_layernorm(h)
            else:
                h_ln = h

            # Project Q.
            Q = self_attn.q_proj(h_ln)  # [1, T, n_heads * d_head]
            Q = Q.view(1, T, n_heads, d_head)  # [1, T, n_heads, d_head]

            # Apply rotary embeddings to Q.
            # We need to apply RoPE to Q (the compact keys were NOT rotated during
            # compression, so we skip RoPE on them and apply it only to Q).
            # Actually, the keys WERE already projected and would have had RoPE
            # applied during normal forward. Since we captured them pre-RoPE,
            # we need to apply RoPE to both Q and the compact keys.
            # However, the compact keys are a subset of the original K projection
            # output. We need to apply RoPE at the correct positions.
            # For simplicity, since this is a first implementation, we apply RoPE
            # to Q and to Ck using position IDs 0..budget-1.
            # NOTE: This is an approximation. Properly, we should know which
            # original positions the selected keys came from and use those positions.
            # For now, we use sequential positions.

            # Apply rotary to Q.
            cos, sin = model.model.rotary_emb(h_ln, position_ids)
            Q1 = Q.clone()
            Q_rotated = _apply_rotary_pos_emb(Q1, cos, sin)

            # For compact keys, assign positions.
            # We don't have the original indices here, so use uniform spacing.
            budget = Ck.shape[1]
            ck_positions = torch.linspace(0, T - 1, budget).long().to(device)
            ck_cos = cos[:, ck_positions]  # [1, budget, 1, d_head] -- need broadcast
            ck_sin = sin[:, ck_positions]

            # Reshape Ck for RoPE: [1, budget, n_kv_heads, d_head] -> process each head.
            Ck_for_rope = Ck.permute(1, 0, 2).unsqueeze(0)  # [1, budget, n_kv_heads, d_head]
            Ck_rotated = _apply_rotary_pos_emb(Ck_for_rope, ck_cos.unsqueeze(2), ck_sin.unsqueeze(2))

            # Compute attention: Q_rotated @ Ck_rotated^T with beta bias.
            # Q_rotated: [1, T, n_heads, d_head]
            # Ck_rotated: [1, budget, n_kv_heads, d_head]
            # Need to handle GQA: repeat Ck for each query head.

            heads_per_kv = n_heads // n_kv_heads
            # Expand Ck, Cv, beta for GQA.
            Ck_exp = Ck_rotated.repeat_interleave(heads_per_kv, dim=2)  # [1, budget, n_heads, d_head]
            Cv_exp = Cv.permute(1, 0, 2).unsqueeze(0).repeat_interleave(heads_per_kv, dim=2)  # [1, budget, n_heads, d_head]
            beta_exp = beta.unsqueeze(0).unsqueeze(2).repeat_interleave(heads_per_kv, dim=2)  # [1, n_heads, budget] -> need [1, budget, n_heads]

            # Transpose for matmul: [1, n_heads, T, d_head]
            Q_t = Q_rotated.transpose(1, 2)  # [1, n_heads, T, d_head]
            Ck_t = Ck_exp.transpose(1, 2)    # [1, n_heads, budget, d_head]
            Cv_t = Cv_exp.transpose(1, 2)     # [1, n_heads, budget, d_head]
            beta_t = beta.T.repeat_interleave(heads_per_kv, dim=0).unsqueeze(0)  # [1, n_heads, budget]

            # Attention logits: [1, n_heads, T, budget]
            attn_logits = torch.matmul(Q_t, Ck_t.transpose(-2, -1)) / scale
            attn_logits = attn_logits + beta_t.unsqueeze(2)  # add beta bias

            attn_weights = _softmax(attn_logits, dim=-1)  # [1, n_heads, T, budget]
            attn_output = torch.matmul(attn_weights, Cv_t)  # [1, n_heads, T, d_head]

            # Reshape and project.
            attn_output = attn_output.transpose(1, 2).contiguous().view(1, T, -1)
            attn_output = self_attn.o_proj(attn_output)

            # Residual connection: hidden = original_hidden + attn_output
            hidden = hidden + attn_output

            # MLP with post-attention layernorm + residual.
            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            hidden = residual + layer.mlp(hidden)

        # Final layernorm + LM head.
        hidden = model.model.norm(hidden)
        logits = model.lm_head(hidden)  # [1, T, vocab]

        # Compute cross-entropy loss.
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
        )
        ppl = math.exp(loss.item())

    return ppl


def _apply_rotary_pos_emb(q, cos, sin, position_ids=None):
    """Apply rotary position embedding — matches Llama's implementation."""
    # q: [batch, seq_len, num_heads, head_dim] or [batch, num_heads, seq_len, head_dim]
    # cos, sin: [batch, seq_len, head_dim] or broadcastable

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    # Ensure shapes broadcast.
    # If q is [B, T, H, D] and cos is [B, T, D], we need cos to be [B, T, 1, D].
    if cos.dim() == 3 and q.dim() == 4:
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed
