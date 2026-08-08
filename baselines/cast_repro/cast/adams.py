"""AdamS: the CAST sparsity-inducing optimizer (Algorithm 1, Eq. 7-8).

Paper: arXiv:2509.25996v1, Algorithm 1 + Eq. (8).

Per-parameter update, for every scalar theta in the CAST scope
(Alg. 1 lines 10-21; line numbers cited inline):

    g_t       = dL/dtheta                                        (line 10)
    mu_t      = b1*mu_{t-1} + (1-b1)*g_t                         (line 11)
    alpha_t   = t / T                                            (line 12)
    if m_t == 0:  mu~_t = (1-alpha_t)*mu_t + alpha_t*lam*sign(theta_{t-1})  (line 14)
    else:         mu~_t = mu_t                                   (line 16)
    v_t       = b2*v_{t-1} + (1-b2)*mu~_t^2                       (line 18)  <-- mu~, not g
    mu^_t     = mu~_t / (1-b1^t)                                 (line 19)
    v^_t      = v_t  / (1-b2^t)                                  (line 20)
    theta_t   = theta_{t-1} - gamma_t * mu^_t / (sqrt(v^_t)+eps)  (line 21)

Three details that are easy to get wrong, and how we resolve them:

1. ``v_t`` is built from ``mu~_t`` for *all* parameters in scope, not only the
   masked ones.  Alg. 1 line 18 sits *outside* the if/else of lines 13-17, and
   for kept weights line 16 defines mu~ = mu, so "v from mu~" and "v from mu"
   coincide there.  Sec. IV-A3 confirms the intent: "we apply the decay to the
   first-order momentum and use the resulting sum to compute the second-order
   moment".  Note this still differs from textbook Adam, which uses g^2, not
   mu^2 -- that difference is deliberate and is CAST's third modification.
   [paper_explicit]

2. ``alpha_t`` indexing.  Alg. 1 loops ``t = 0..T-1`` (line 5) so alpha_0 = 0
   and the very first update is pure Adam with no decay.  Alg. 2 loops
   ``t = 1..T``.  We follow Alg. 1 because it is the one that defines AdamS
   itself, and because alpha_0 = 0 makes lines 12-14 consistent.  Concretely,
   with PyTorch's 1-based ``state['step']``, alpha = (step-1)/T.
   [paper_explicit, ambiguity documented in SPEC.md]

3. ``mu_t`` keeps accumulating the *raw* gradient (line 11 is unconditional);
   the decay only ever enters ``mu~_t``, which is discarded after the step.
   This is exactly the "decoupling" of Sec. IV-A3 -- the decay direction must
   not be polluted by momentum history, otherwise it lags when theta crosses
   zero.  Do not write the decayed value back into exp_avg.
   [paper_explicit]

Numerics: the optimizer requires fp32 parameters.  lambda = 4e-7 (Table XI) is
far below bf16's ~0.4% relative resolution near typical weight magnitudes, so a
bf16 master weight silently rounds the entire decay signal away.  ``AdamS``
raises if handed a non-fp32 in-scope parameter.  Use bf16 only for autocast
activations.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional

import torch
from torch.optim import Optimizer


class MaskCoverageError(RuntimeError):
    """Raised when an in-scope weight did not receive the AdamS decay path.

    This exists because the previous reproduction *silently* degraded to vanilla
    Adam under FSDP (audit doc section 4.1) and burned 7.86B tokens before anyone
    noticed.  Coverage below 100% is now a crash, never a warning.
    """


class AdamS(Optimizer):
    """Adam with CAST's selective, proportional, decoupled L1 decay.

    Args:
        params: parameter groups.  Set ``cast_scope=True`` on a group to enable
            AdamS; those parameters must each carry a ``cast_mask`` attribute
            (a same-shape binary tensor).  Groups without the flag get plain
            Adam (used for ``cast_scale``, norms, embeddings, lm_head).
        total_steps: T in alpha_t = t/T.  Must be the *planned* total, since the
            decay schedule is calibrated against it.
        l1_decay: lambda.  4e-7 for LLaMA per Table XI.
        expected_scope_elements: if given, the number of in-scope elements is
            asserted against it on the first step (3_238_002_688 for LLaMA2-7B
            in-block 2:4 -- exactly half of the 6,476,005,376 in-block linear
            parameters).
        expected_scope_tensors: likewise for the tensor count (224 for LLaMA2-7B).
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 2e-5,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        total_steps: int = 7500,
        l1_decay: float = 4e-7,
        expected_scope_elements: Optional[int] = None,
        expected_scope_tensors: Optional[int] = None,
        require_fp32: bool = True,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"invalid lr: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"invalid betas: {betas}")
        if total_steps < 1:
            raise ValueError(f"total_steps must be >= 1, got {total_steps}")
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, cast_scope=False
        )
        super().__init__(params, defaults)
        self.total_steps = int(total_steps)
        self.l1_decay = float(l1_decay)
        self.expected_scope_elements = expected_scope_elements
        self.expected_scope_tensors = expected_scope_tensors
        self.require_fp32 = bool(require_fp32)
        #: filled in by every step(); read by the training loop for logging.
        self.last_stats: Dict[str, Any] = {}
        self._checked_static_scope = False

    # -- alpha_t (Alg. 1 line 12) -------------------------------------------
    def alpha_for_step(self, step_one_based: int) -> float:
        """alpha_t = t/T with Alg. 1's t = 0..T-1 convention.

        ``step_one_based`` is PyTorch's post-increment ``state['step']``, so the
        first call passes 1 and must yield alpha = 0.
        """
        t = max(0, int(step_one_based) - 1)
        return min(1.0, float(t) / float(self.total_steps))

    @torch.no_grad()
    def step(self, closure=None):  # noqa: C901 - single hot loop, kept flat on purpose
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        scope_tensors = 0
        scope_tensors_with_mask = 0
        scope_elements = 0
        decayed_elements = 0
        alpha_t = 0.0

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]
            in_scope = bool(group.get("cast_scope", False))

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamS does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                if wd != 0.0:
                    # Not used in the CAST recipe (Table XI lists no weight decay);
                    # kept for completeness, AdamW-style decoupled form.
                    p.mul_(1.0 - lr * wd)

                # Alg. 1 line 11: momentum always tracks the RAW gradient.
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                if in_scope:
                    scope_tensors += 1
                    scope_elements += p.numel()

                    if self.require_fp32 and p.dtype != torch.float32:
                        raise MaskCoverageError(
                            f"[AdamS] in-scope parameter has dtype {p.dtype}, expected float32. "
                            f"lambda={self.l1_decay:g} is below bf16/fp16 resolution and would be "
                            "rounded away; keep fp32 master weights and use autocast for activations."
                        )

                    mask = getattr(p, "cast_mask", None)
                    if mask is None:
                        raise MaskCoverageError(
                            "[AdamS] in-scope parameter has no `cast_mask` attribute. "
                            "Every CAST weight must carry an element-aligned binary mask."
                        )
                    if mask.shape != p.shape:
                        raise MaskCoverageError(
                            f"[AdamS] mask/weight shape mismatch: weight={tuple(p.shape)} "
                            f"mask={tuple(mask.shape)}. Under FSDP the two are sliced from "
                            "different FlatParameter offsets and are NOT element-aligned -- "
                            "that was the root cause of the failed 7.86B-token run. Use DDP."
                        )
                    if mask.device != p.device:
                        raise MaskCoverageError(
                            f"[AdamS] mask on {mask.device} but weight on {p.device}"
                        )
                    scope_tensors_with_mask += 1

                    # Alg. 1 lines 12-16.
                    alpha_t = self.alpha_for_step(step)
                    keep = mask if mask.dtype == torch.bool else mask > 0.5
                    decayed_elements += int((~keep).sum().item())

                    # mu~ = (1-alpha)*mu + alpha*lam*sign(theta)  on masked entries,
                    #       mu                                    on kept entries.
                    # torch.sign gives sign(0) = 0, so a weight sitting exactly at
                    # zero receives no decay -- correct, it is already pruned.
                    decayed = exp_avg.mul(1.0 - alpha_t).add_(
                        torch.sign(p), alpha=alpha_t * self.l1_decay
                    )
                    mu_tilde = torch.where(keep, exp_avg, decayed)
                else:
                    mu_tilde = exp_avg

                # Alg. 1 line 18: second moment from mu~ (NOT from g).
                exp_avg_sq.mul_(beta2).addcmul_(mu_tilde, mu_tilde, value=1.0 - beta2)

                # Alg. 1 lines 19-21.
                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                p.addcdiv_(mu_tilde, denom, value=-lr / bias_correction1)

        # ---- hard runtime assertions ----
        if scope_tensors != scope_tensors_with_mask:
            raise MaskCoverageError(
                f"[AdamS] only {scope_tensors_with_mask}/{scope_tensors} in-scope tensors got "
                "an aligned mask. AdamS coverage must be 100%; refusing to continue."
            )
        if scope_tensors and not self._checked_static_scope:
            if self.expected_scope_tensors is not None and scope_tensors != self.expected_scope_tensors:
                raise MaskCoverageError(
                    f"[AdamS] in-scope tensor count {scope_tensors} != expected "
                    f"{self.expected_scope_tensors}"
                )
            if (
                self.expected_scope_elements is not None
                and scope_elements != self.expected_scope_elements
            ):
                raise MaskCoverageError(
                    f"[AdamS] in-scope element count {scope_elements:,} != expected "
                    f"{self.expected_scope_elements:,}"
                )
            self._checked_static_scope = True

        # For exact 2:4, half of every in-scope tensor must be masked.
        if scope_elements:
            expected_decayed = scope_elements // 2
            if decayed_elements != expected_decayed:
                raise MaskCoverageError(
                    f"[AdamS] decayed_elements={decayed_elements:,} but exact 2:4 requires "
                    f"{expected_decayed:,} (= half of {scope_elements:,}). The mask is not a "
                    "valid 2:4 pattern."
                )

        self.last_stats = {
            "cast_tensors": scope_tensors,
            "cast_tensors_aligned": scope_tensors_with_mask,
            "cast_elements": scope_elements,
            "decayed_elements": decayed_elements,
            "alpha_t": alpha_t,
            "coverage": 1.0 if scope_tensors == scope_tensors_with_mask else 0.0,
        }
        return loss


def build_param_groups(
    model,
    lr: float,
    weight_decay: float = 0.0,
) -> list:
    """Split parameters into the CAST scope and everything else.

    In-scope: the ``weight`` of every ``CastSparseLinear`` (tagged with
    ``cast_in_scope``).  Everything else -- ``cast_scale``, RMSNorm weights,
    ``embed_tokens``, ``lm_head`` -- gets plain Adam.  The paper does not give
    separate learning rates, so all groups share ``lr`` (Table XI lists a single
    LR of 2e-5 for LLaMA).  [implementation_choice: which tensors are in scope]
    """
    scope, other = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (scope if getattr(p, "cast_in_scope", False) else other).append(p)
    groups = []
    if scope:
        groups.append({"params": scope, "cast_scope": True, "lr": lr, "weight_decay": weight_decay})
    if other:
        groups.append({"params": other, "cast_scope": False, "lr": lr, "weight_decay": weight_decay})
    return groups
