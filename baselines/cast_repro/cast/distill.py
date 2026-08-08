"""CAST knowledge distillation loss (Eq. 13).

Paper: arXiv:2509.25996v1 Sec. IV-C, Eq. (13):

    L_kl = D_KL(P_t || P_s) = sum_x P_t(x) log( P_t(x) / P_s(x) )
    L    = eta * L_kl + (1 - eta) * L_ce

with eta = 1/3 for LLaMA (Table XI, "Kl Coefficient").  The teacher is the
frozen dense model itself ("dense model as a self-teacher", Sec. IV-C).

TEMPERATURE -- documented divergence from the reference code.  Eq. (13) has no
temperature: the KL is taken between the raw softmax distributions, i.e. T = 1.
The same group's AST code (baselines/ast_official_clean/sparse_modeling.py:240)
hardcodes ``temperature=2`` and multiplies the result by ``T**2``.  We default to
the paper-literal ``T = 1`` and expose ``temperature`` so the AST-style variant
can be run as an explicitly-named ablation.  Do not change the default silently:
with the convex combination below, T also rescales the effective LR on the KL
term, so T=2 (x4 from the T^2 factor) is not a neutral choice.
[paper_explicit for T=1; ast_code_inferred for T=2]

NORMALISATION.  Eq. (13) is a convex combination, ``eta*KL + (1-eta)*CE``, and
both terms are per-token means.  This matters for the learning rate: Table XI's
LR of 2e-5 is only meaningful if the total loss has the same scale as a plain LM
loss.  An "un-normalised" form ``CE + eta'*KL`` would need eta' = eta/(1-eta) and
an LR rescaled by (1-eta) to be equivalent -- see ``convex_to_unnormalised``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def kl_divergence_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
    ignore_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean-per-token forward KL D_KL(P_teacher || P_student), Eq. (13).

    Args:
        student_logits: (..., V)
        teacher_logits: (..., V), treated as a constant (detached).
        temperature: T. 1.0 = paper-literal. When T != 1 the loss is multiplied
            by T**2, matching the AST reference implementation.
        ignore_mask: optional bool tensor broadcastable to the token dims, True
            where the token should be *excluded*.

    Reduction is a mean over tokens and a sum over the vocabulary, which is what
    Eq. (13) computes (the sum over x is the vocabulary sum; the 1/(B*S) factor
    in the AST paper's Eq. 4 is the token mean).
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"logit shape mismatch: student={tuple(student_logits.shape)} "
            f"teacher={tuple(teacher_logits.shape)}"
        )
    t = float(temperature)
    student_logp = F.log_softmax(student_logits.float() / t, dim=-1)
    teacher_logp = F.log_softmax(teacher_logits.float().detach() / t, dim=-1)

    # sum_x P_t * (log P_t - log P_s), summed over vocab -> per-token KL.
    per_token = F.kl_div(
        student_logp, teacher_logp, log_target=True, reduction="none"
    ).sum(dim=-1)

    if ignore_mask is not None:
        valid = ~ignore_mask
        denom = valid.sum().clamp(min=1)
        loss = (per_token * valid).sum() / denom
    else:
        loss = per_token.mean()

    if t != 1.0:
        loss = loss * (t * t)
    return loss


def cast_loss(
    student_logits: torch.Tensor,
    teacher_logits: Optional[torch.Tensor],
    labels: torch.Tensor,
    eta: float = 1.0 / 3.0,
    temperature: float = 1.0,
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, dict]:
    """L = eta*L_kl + (1-eta)*L_ce, Eq. (13).

    ``labels`` are already shifted by the caller (next-token targets aligned to
    ``student_logits``).  Returns (total_loss, components) where components holds
    detached floats for logging.

    ``teacher_logits=None`` or ``eta=0`` degenerates to pure CE (the eta=0 arm of
    the Fig. 6 ablation).
    """
    if not 0.0 <= eta <= 1.0:
        raise ValueError(f"eta must be in [0,1], got {eta}")

    ce = F.cross_entropy(
        student_logits.float().reshape(-1, student_logits.size(-1)),
        labels.reshape(-1),
        ignore_index=ignore_index,
    )

    if teacher_logits is None or eta == 0.0:
        return ce, {"loss": float(ce.detach()), "ce": float(ce.detach()), "kl": 0.0, "eta": eta}

    kl = kl_divergence_loss(
        student_logits,
        teacher_logits,
        temperature=temperature,
        ignore_mask=(labels == ignore_index),
    )
    total = eta * kl + (1.0 - eta) * ce
    return total, {
        "loss": float(total.detach()),
        "ce": float(ce.detach()),
        "kl": float(kl.detach()),
        "eta": eta,
    }


def convex_to_unnormalised(eta: float, lr: float) -> Tuple[float, float]:
    """Equivalence between ``eta*KL + (1-eta)*CE`` and ``CE + eta'*KL``.

    ``(1-eta)*(CE + eta/(1-eta) * KL)`` is the convex form, so the un-normalised
    objective with ``eta' = eta/(1-eta)`` produces identical *directions* but
    gradients larger by 1/(1-eta); to match, its LR must be ``lr*(1-eta)``.
    For eta=1/3: eta' = 0.5 and lr' = lr*2/3.  Recorded so the LR in Table XI is
    unambiguous -- we use the convex form, so lr=2e-5 is used verbatim.
    """
    if not 0.0 <= eta < 1.0:
        raise ValueError(f"eta must be in [0,1) for this conversion, got {eta}")
    return eta / (1.0 - eta), lr * (1.0 - eta)
