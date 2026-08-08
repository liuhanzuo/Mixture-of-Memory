"""CAST SparseLinear: dense forward + learnable group scaling + N:M magnitude mask.

Paper: CAST: Continuous and Differentiable Semi-Structured Sparsity-Aware Training
       for Large Language Models (arXiv:2509.25996v1)

Source-of-truth mapping (see ../SPEC.md for the full table):

  * Eq. (3)/(4)  -- contiguous groups of M along the *input* (column) axis,
                    each group keeps exactly N nonzeros.       [paper_explicit]
  * Eq. (6)      -- mask = 1 where |W| >= xi, xi = the 2nd largest absolute
                    value inside the 4-element group.          [paper_explicit]
  * Eq. (11)/(12)-- learnable scaling A^k of shape (R_k, n), n=2 for LLaMA;
                    initialised to all-ones (Alg. 2 line 4).   [paper_explicit]
  * Sec. IV / Fig. 2 -- forward stays DENSE for the whole run; the binary mask is
                    only consumed by the optimizer, never by the forward.
                                                               [paper_explicit]
  * Alg. 2 line 20/21 -- finalisation = hard prune with M_T, then fold the
                    scaling module into the weights.           [paper_explicit]

IMPORTANT (why plain 2-D tensors matter): the previous reproduction attempt ran
under FSDP FULL_SHARD, which splits `weight` and `mask` into *different* slices
of a FlatParameter.  AdamS then silently fell back to vanilla Adam whenever the
two shards disagreed, so the selective L1 decay never ran on most tensors
(audit doc section 4.1).  Everything here assumes plain DDP: `weight` and
`mask` are full, identically-shaped, element-aligned 2-D tensors on the same
device.  `assert_mask_alignment()` enforces that at runtime.
"""

from __future__ import annotations

from typing import Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Eq. (6): magnitude-based N:M mask
# ---------------------------------------------------------------------------
@torch.no_grad()
def nm_magnitude_mask(weight: torch.Tensor, n: int = 2, m: int = 4) -> torch.Tensor:
    """Binary N:M mask from weight magnitude, per Eq. (6).

    Groups are contiguous along the last (input/column) axis, per Eq. (3).
    Within each group of ``m`` elements the top-``n`` by absolute value get 1.

    Eq. (6) states the threshold xi is "the second largest absolute value within
    the 4 elements", and uses ``>=`` for keep.  With exact ties (e.g. two equal
    magnitudes at the boundary) a literal ``|w| >= xi`` comparison can keep more
    than ``n`` elements and break the hard 2:4 constraint that Eq. (4) requires.
    We therefore use ``topk``, which keeps exactly ``n`` and resolves ties by
    index.  This is the only tie-breaking-safe reading of Eq. (4) + Eq. (6).

    Returns a ``torch.bool`` tensor: the paper's Remark in Sec. IV-A budgets the
    mask at "1/32" of the optimizer state, i.e. one bit per parameter, so a
    float mask would be 32x over budget (26 GB/rank for LLaMA2-7B).
    """
    if weight.dim() != 2:
        raise ValueError(f"expected 2-D weight, got shape {tuple(weight.shape)}")
    rows, cols = weight.shape
    if cols % m != 0:
        raise ValueError(f"in_features ({cols}) must be divisible by M ({m})")

    groups = cols // m
    mag = weight.detach().abs().reshape(rows, groups, m)
    keep = mag.topk(k=n, dim=-1, largest=True).indices
    mask = torch.zeros(mag.shape, dtype=torch.bool, device=weight.device)
    mask.scatter_(-1, keep, True)
    return mask.reshape(rows, cols)


class CastSparseLinear(nn.Linear):
    """nn.Linear + (a) a 2:4 magnitude mask buffer and (b) CAST weight scaling.

    Forward is dense (Sec. IV, Fig. 2 right).  The mask is metadata for AdamS.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        n: int = 2,
        m: int = 4,
        scale_groups: int = 2,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        if in_features % m != 0:
            raise ValueError(f"in_features ({in_features}) must be divisible by M ({m})")
        if scale_groups <= 0 or in_features % scale_groups != 0:
            raise ValueError(
                f"scale_groups ({scale_groups}) must be > 0 and divide in_features ({in_features})"
            )
        self.nm_n = int(n)
        self.nm_m = int(m)
        self.scale_groups = int(scale_groups)

        # Binary mask. Registered as a *buffer*, not a Parameter: it must never
        # receive gradients, and under DDP buffers are broadcast from rank 0 so
        # every rank agrees. Same shape as weight => element-aligned by
        # construction. dtype=bool keeps it at 1 byte/param (Sec. IV-A Remark
        # budgets 1 bit; torch has no bit-packed tensor, 1 byte is the floor).
        self.register_buffer(
            "mask",
            torch.ones(out_features, in_features, device=self.weight.device, dtype=torch.bool),
        )

        # Eq. (11)/(12) + Alg. 2 line 4: A^k in R^{R_k x n}, initialised to 1.
        self.cast_scale = nn.Parameter(
            torch.ones(out_features, self.scale_groups, device=self.weight.device, dtype=self.weight.dtype)
        )

        # Tags consumed by AdamS to identify CAST-scope weights. Attached to the
        # Parameter object so the optimizer can find the mask without needing a
        # module reference (mirrors how the AST/CAST reference code does it).
        self.weight.cast_in_scope = True
        self.weight.cast_mask = self.mask

    # -- Eq. (12): scale, keeping the sparsity pattern intact -----------------
    def scaled_weight(self) -> torch.Tensor:
        """W_scale: element (r, c) multiplied by A[r, c // (C/n)]  (Eq. 12).

        Implemented as view(R, n, C/n) * A.unsqueeze(-1) -> view(R, C), i.e. the
        scaling axis partitions each row into ``n`` contiguous column blocks.
        Because it is a pure element-wise multiply, zeros stay zero, so the 2:4
        pattern survives folding (Sec. IV-B).
        """
        r, c = self.out_features, self.in_features
        n = self.scale_groups
        return (self.weight.view(r, n, c // n) * self.cast_scale.unsqueeze(-1)).view(r, c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # DENSE forward. The mask is deliberately absent here (Sec. IV, Fig. 2).
        return F.linear(x, self.scaled_weight(), self.bias)

    # -- Eq. (6) refresh -----------------------------------------------------
    @torch.no_grad()
    def refresh_mask(self) -> int:
        """Recompute the mask from current |W|; returns the number of flips."""
        new_mask = nm_magnitude_mask(self.weight, self.nm_n, self.nm_m)
        flips = int((self.mask ^ new_mask).sum().item())
        self.mask.copy_(new_mask)
        self.weight.cast_mask = self.mask
        return flips

    # -- Alg. 2 lines 20-21: finalise ---------------------------------------
    @torch.no_grad()
    def finalize(self) -> None:
        """Prune with M_T, then fold the scaling module into the weight.

        Order follows Alg. 2 exactly: line 20 prunes, line 21 folds.  Folding is
        element-wise so it cannot resurrect a pruned entry; the result is an
        exact N:M sparse plain weight and ``cast_scale`` becomes all-ones (a
        no-op) so the module is numerically a bare nn.Linear.
        """
        folded = self.scaled_weight() * self.mask.to(self.weight.dtype)
        self.weight.copy_(folded)
        self.cast_scale.fill_(1.0)

    # -- runtime invariants --------------------------------------------------
    def assert_mask_alignment(self) -> None:
        w, msk = self.weight, self.mask
        if msk.shape != w.shape:
            raise RuntimeError(
                f"[CAST] mask/weight shape mismatch: weight={tuple(w.shape)} mask={tuple(msk.shape)}. "
                "This is the FSDP sharding bug from the previous run; use plain DDP."
            )
        if msk.device != w.device:
            raise RuntimeError(f"[CAST] mask/weight device mismatch: {msk.device} vs {w.device}")
        if getattr(w, "cast_mask", None) is not msk:
            raise RuntimeError("[CAST] weight.cast_mask is not the module's mask buffer")

    def exact_nm_violations(self) -> int:
        """Number of 4-element groups in the *weight* that do not have exactly N nonzeros."""
        r, c, m = self.out_features, self.in_features, self.nm_m
        nz = (self.weight.detach() != 0).reshape(r, c // m, m).sum(-1)
        return int((nz != self.nm_n).sum().item())


# ---------------------------------------------------------------------------
# model surgery
# ---------------------------------------------------------------------------
#: Names of the 7 in-block projections of a LLaMA decoder layer.
LLAMA_INBLOCK_PROJECTIONS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def convert_llama_to_cast(
    model: nn.Module,
    n: int = 2,
    m: int = 4,
    scale_groups: int = 2,
    target_names: Tuple[str, ...] = LLAMA_INBLOCK_PROJECTIONS,
) -> List[str]:
    """Replace the in-block projections with CastSparseLinear, in place.

    Only the 7 x n_layers projections inside the decoder blocks are sparsified.
    ``embed_tokens`` and ``lm_head`` stay dense -- this is an implementation
    choice (the paper says "L linear layers", Sec. III, without enumerating
    them); it is what every N:M LLM-pruning baseline does (Wanda / SparseGPT /
    MaskLLM all skip embeddings and the LM head) and it is what makes the
    224-tensor / 3,238,002,688-element accounting come out exact for LLaMA2-7B.

    Returns the list of fully-qualified names that were converted.
    """
    converted: List[str] = []
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if child_name not in target_names or not isinstance(child, nn.Linear):
                continue
            if isinstance(child, CastSparseLinear):
                continue
            new = CastSparseLinear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                n=n,
                m=m,
                scale_groups=scale_groups,
                device=child.weight.device,
                dtype=child.weight.dtype,
            )
            with torch.no_grad():
                new.weight.copy_(child.weight)
                if child.bias is not None:
                    new.bias.copy_(child.bias)
            setattr(parent, child_name, new)
            converted.append(f"{parent_name}.{child_name}" if parent_name else child_name)
    return converted


def cast_modules(model: nn.Module) -> Iterator[Tuple[str, CastSparseLinear]]:
    for name, mod in model.named_modules():
        if isinstance(mod, CastSparseLinear):
            yield name, mod


@torch.no_grad()
def refresh_all_masks(model: nn.Module) -> Tuple[int, int]:
    """Refresh every CAST mask (Eq. 6). Returns (n_modules, total_flips)."""
    n_mod = 0
    flips = 0
    for _, mod in cast_modules(model):
        flips += mod.refresh_mask()
        n_mod += 1
    return n_mod, flips


@torch.no_grad()
def finalize_all(model: nn.Module) -> int:
    n_mod = 0
    for _, mod in cast_modules(model):
        mod.finalize()
        n_mod += 1
    return n_mod


def cast_scope_stats(model: nn.Module) -> dict:
    """Static accounting of the CAST scope. Used for the hard runtime assertion."""
    n_tensors = 0
    n_elements = 0
    n_masked = 0
    for _, mod in cast_modules(model):
        n_tensors += 1
        n_elements += mod.weight.numel()
        n_masked += int((~mod.mask).sum().item())
    return {
        "cast_tensors": n_tensors,
        "cast_elements": n_elements,
        "cast_masked_elements": n_masked,
    }
