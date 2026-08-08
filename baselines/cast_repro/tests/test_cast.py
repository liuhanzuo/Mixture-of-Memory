"""Unit tests for the CAST reproduction. Pure torch, CPU, no transformers needed.

Run:  python -m pytest baselines/cast_repro/tests -v
  or: python baselines/cast_repro/tests/test_cast.py     (prints a summary)

Every test targets one of the six root causes in
Mixture-of-Memory/SparseForge_Data/docs/CAST_REPRODUCTION_AUDIT.md, or one of
the ambiguities resolved in SPEC.md.
"""

from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cast import (  # noqa: E402
    AdamS,
    CastSparseLinear,
    MaskCoverageError,
    build_param_groups,
    cast_loss,
    convert_llama_to_cast,
    convex_to_unnormalised,
    kl_divergence_loss,
    nm_magnitude_mask,
    refresh_all_masks,
)


# ---------------------------------------------------------------------------
# 1. Eq. (6): hand-computed 2:4 mask
# ---------------------------------------------------------------------------
def test_nm_mask_hand_computed():
    # Two rows, 8 columns => 4 groups of 4. Keep top-2 |.| per group.
    w = torch.tensor(
        [
            # group0: |.| = 1,2,3,4 -> keep cols 2,3   group1: 9,7,5,3 -> keep 4,5
            [1.0, -2.0, 3.0, -4.0, 9.0, -7.0, 5.0, 3.0],
            # group0: 0.5,0.4,0.3,0.2 -> keep 0,1      group1: -8,1,-6,2 -> keep 8? cols 4,6
            [0.5, -0.4, 0.3, -0.2, -8.0, 1.0, -6.0, 2.0],
        ]
    )
    mask = nm_magnitude_mask(w, n=2, m=4)
    expected = torch.tensor(
        [
            [0, 0, 1, 1, 1, 1, 0, 0],
            [1, 1, 0, 0, 1, 0, 1, 0],
        ],
        dtype=torch.bool,
    )
    assert mask.dtype == torch.bool
    assert torch.equal(mask, expected), f"got\n{mask.int()}\nexpected\n{expected.int()}"
    # exactly 2 kept in each of the 4 groups of 4
    per_group = mask.reshape(2, 2, 4).sum(-1)
    assert torch.equal(per_group, torch.full((2, 2), 2)), per_group
    return "2:4 mask matches hand computation; exactly N per group"


def test_nm_mask_ties_still_exact():
    # All four magnitudes equal: a literal `|w| >= xi` would keep all 4.
    w = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
    mask = nm_magnitude_mask(w, n=2, m=4)
    assert int(mask.sum()) == 2, f"tie case kept {int(mask.sum())} elements, expected exactly 2"
    return "exact ties still yield exactly 2 kept (topk, not >= xi)"


# ---------------------------------------------------------------------------
# 2. AdamS decay is applied to masked positions only
# ---------------------------------------------------------------------------
def _one_layer(bias=False, in_f=8, out_f=2, dtype=torch.float32):
    torch.manual_seed(0)
    lin = CastSparseLinear(in_f, out_f, bias=bias, scale_groups=2, dtype=dtype)
    return lin


def test_adams_decays_masked_not_kept():
    """lambda*sign(theta) must move masked weights toward zero and leave kept
    weights driven only by the gradient."""
    lin = _one_layer()
    with torch.no_grad():
        lin.weight.copy_(
            torch.tensor([[0.4, 0.3, 0.2, 0.1, 0.4, 0.3, 0.2, 0.1],
                          [0.4, 0.3, 0.2, 0.1, 0.4, 0.3, 0.2, 0.1]])
        )
    lin.refresh_mask()
    # cols 0,1 and 4,5 kept; cols 2,3 and 6,7 masked
    assert torch.equal(
        lin.mask[0], torch.tensor([1, 1, 0, 0, 1, 1, 0, 0], dtype=torch.bool)
    )

    lam = 1e-2  # exaggerated so the effect is visible in one step
    opt = AdamS(
        build_param_groups(lin, lr=1e-3),
        lr=1e-3,
        total_steps=10,
        l1_decay=lam,
        require_fp32=True,
    )
    # Zero gradient => the ONLY force is the L1 decay.
    lin.weight.grad = torch.zeros_like(lin.weight)
    lin.cast_scale.grad = torch.zeros_like(lin.cast_scale)

    # step 1 -> alpha_0 = 0 -> pure Adam, no decay at all, and grad is 0 => no move
    before = lin.weight.detach().clone()
    opt.step()
    assert torch.allclose(lin.weight, before, atol=1e-9), "alpha_0 must be 0 (no decay on step 1)"
    a0 = opt.last_stats["alpha_t"]

    # advance a few steps so alpha > 0
    for _ in range(4):
        lin.weight.grad = torch.zeros_like(lin.weight)
        lin.cast_scale.grad = torch.zeros_like(lin.cast_scale)
        opt.step()

    w = lin.weight.detach()
    keep_moved = (w[:, [0, 1, 4, 5]] - before[:, [0, 1, 4, 5]]).abs().max().item()
    masked_delta = (w[:, [2, 3, 6, 7]] - before[:, [2, 3, 6, 7]])
    assert keep_moved < 1e-9, f"kept weights moved by {keep_moved} with zero grad; must not decay"
    assert (masked_delta < 0).all(), f"masked positive weights must shrink, got {masked_delta}"
    return (
        f"alpha_0={a0:.1f} (no first-step decay); after 5 steps kept |delta|={keep_moved:.2e}, "
        f"masked mean delta={masked_delta.mean().item():+.3e} (toward zero)"
    )


def test_adams_sign_at_zero_is_zero():
    """torch.sign(0)=0, so a weight exactly at zero receives no decay kick."""
    lin = _one_layer()
    with torch.no_grad():
        lin.weight.zero_()
        lin.weight[:, 0] = 1.0  # keep col 0/1 non-degenerate
        lin.weight[:, 4] = 1.0
    lin.refresh_mask()
    opt = AdamS(build_param_groups(lin, lr=1e-3), lr=1e-3, total_steps=2, l1_decay=1.0)
    for _ in range(2):
        lin.weight.grad = torch.zeros_like(lin.weight)
        lin.cast_scale.grad = torch.zeros_like(lin.cast_scale)
        opt.step()
    zeros_stay = lin.weight.detach()[:, 3]  # a masked, exactly-zero column
    assert torch.allclose(zeros_stay, torch.zeros_like(zeros_stay), atol=1e-12), zeros_stay
    return "sign(0)=0: weights already at zero stay exactly zero"


def test_adams_alpha_ramps_linearly_zero_to_one():
    lin = _one_layer()
    lin.refresh_mask()
    T = 8
    opt = AdamS(build_param_groups(lin, lr=0.0), lr=0.0, total_steps=T, l1_decay=0.0)
    seen = []
    for _ in range(T + 2):
        lin.weight.grad = torch.zeros_like(lin.weight)
        lin.cast_scale.grad = torch.zeros_like(lin.cast_scale)
        opt.step()
        seen.append(round(opt.last_stats["alpha_t"], 6))
    expected = [round(min(1.0, i / T), 6) for i in range(T + 2)]
    assert seen == expected, f"alpha schedule {seen} != {expected}"
    assert seen[0] == 0.0 and seen[T] == 1.0
    return f"alpha_t = t/T from {seen[0]} to {seen[-1]} (clamped at 1.0), T={T}"


def test_second_moment_uses_mu_tilde():
    """Alg. 1 line 18 builds v from mu~, so a masked weight with ZERO gradient
    still accumulates a non-zero second moment (from the decay term alone)."""
    lin = _one_layer()
    with torch.no_grad():
        lin.weight.copy_(torch.full_like(lin.weight, 0.5))
        lin.weight[:, [0, 4]] = 1.0  # make the keep/mask split deterministic
    lin.refresh_mask()
    opt = AdamS(build_param_groups(lin, lr=1e-3), lr=1e-3, total_steps=4, l1_decay=1e-1)
    for _ in range(3):
        lin.weight.grad = torch.zeros_like(lin.weight)
        lin.cast_scale.grad = torch.zeros_like(lin.cast_scale)
        opt.step()
    v = opt.state[lin.weight]["exp_avg_sq"]
    masked_v = v[~lin.mask]
    kept_v = v[lin.mask]
    assert (masked_v > 0).all(), "v must be > 0 on masked entries (fed by lambda*sign)"
    assert torch.allclose(kept_v, torch.zeros_like(kept_v)), "v must stay 0 on kept entries w/ zero grad"
    return (
        f"v from mu~: masked v mean={masked_v.mean():.3e} > 0, kept v={kept_v.max():.1e} "
        "(vanilla Adam would give 0 everywhere)"
    )


# ---------------------------------------------------------------------------
# 3. Coverage assertions must fire (the silent-degradation bug)
# ---------------------------------------------------------------------------
def test_missing_mask_raises():
    lin = _one_layer()
    lin.refresh_mask()
    del lin.weight.cast_mask  # simulate the FSDP fallback
    opt = AdamS(build_param_groups(lin, lr=1e-3), lr=1e-3, total_steps=4, l1_decay=1e-3)
    lin.weight.grad = torch.zeros_like(lin.weight)
    lin.cast_scale.grad = torch.zeros_like(lin.cast_scale)
    try:
        opt.step()
    except MaskCoverageError as e:
        return f"missing mask raises MaskCoverageError: {str(e)[:70]}..."
    raise AssertionError("AdamS silently accepted an in-scope weight with no mask")


def test_shape_mismatch_raises():
    """The exact FSDP failure mode: mask numel != weight numel."""
    lin = _one_layer()
    lin.refresh_mask()
    lin.weight.cast_mask = torch.ones(3, dtype=torch.bool)  # bogus shard
    opt = AdamS(build_param_groups(lin, lr=1e-3), lr=1e-3, total_steps=4, l1_decay=1e-3)
    lin.weight.grad = torch.zeros_like(lin.weight)
    lin.cast_scale.grad = torch.zeros_like(lin.cast_scale)
    try:
        opt.step()
    except MaskCoverageError as e:
        assert "FSDP" in str(e)
        return "mask/weight shape mismatch raises (no silent Adam fallback)"
    raise AssertionError("AdamS silently accepted a misaligned mask shard")


def test_expected_element_count_enforced():
    lin = _one_layer()
    lin.refresh_mask()
    opt = AdamS(
        build_param_groups(lin, lr=1e-3),
        lr=1e-3,
        total_steps=4,
        l1_decay=1e-3,
        expected_scope_elements=999,  # wrong on purpose
    )
    lin.weight.grad = torch.zeros_like(lin.weight)
    lin.cast_scale.grad = torch.zeros_like(lin.cast_scale)
    try:
        opt.step()
    except MaskCoverageError as e:
        assert "element count" in str(e)
        return "wrong expected_scope_elements raises before any training happens"
    raise AssertionError("element-count assertion did not fire")


def test_non_fp32_in_scope_raises():
    lin = _one_layer(dtype=torch.bfloat16)
    lin.refresh_mask()
    opt = AdamS(build_param_groups(lin, lr=1e-3), lr=1e-3, total_steps=4, l1_decay=4e-7)
    lin.weight.grad = torch.zeros_like(lin.weight)
    lin.cast_scale.grad = torch.zeros_like(lin.cast_scale)
    try:
        opt.step()
    except MaskCoverageError as e:
        assert "float32" in str(e)
        return "bf16 in-scope weight is rejected up front"
    raise AssertionError("bf16 master weight was accepted")


# ---------------------------------------------------------------------------
# 4. bf16 eats lambda=4e-7, fp32 does not
# ---------------------------------------------------------------------------
def test_bf16_swallows_lambda_fp32_does_not():
    """The reason fp32 master weights are mandatory.

    Repeatedly subtract the paper's lambda=4e-7 from a typical weight magnitude
    and check whether the value actually moves.
    """
    lam = 4e-7
    w0 = 0.02  # typical LLaMA2-7B kept-weight magnitude (audit doc section 5)
    n_steps = 100

    fp32 = torch.tensor([w0], dtype=torch.float32)
    for _ in range(n_steps):
        fp32 = fp32 - lam
    bf16 = torch.tensor([w0], dtype=torch.bfloat16)
    for _ in range(n_steps):
        bf16 = bf16 - torch.tensor([lam], dtype=torch.bfloat16)

    fp32_moved = abs(fp32.item() - w0)
    bf16_moved = abs(bf16.float().item() - torch.tensor([w0], dtype=torch.bfloat16).float().item())
    expected = lam * n_steps

    assert fp32_moved > 0.5 * expected, f"fp32 lost the signal: moved {fp32_moved:.3e}"
    assert bf16_moved == 0.0, f"bf16 unexpectedly moved by {bf16_moved:.3e}"
    return (
        f"after {n_steps} decays of lambda={lam:g}: fp32 moved {fp32_moved:.3e} "
        f"(expected {expected:.3e}), bf16 moved {bf16_moved:.1f} -> signal fully lost"
    )


# ---------------------------------------------------------------------------
# 5. Weight scaling: folding preserves the 2:4 pattern
# ---------------------------------------------------------------------------
def test_scale_groups_axis_semantics():
    """A[r, c // (C/n)] -- verify the scaling axis matches Eq. (12)."""
    lin = CastSparseLinear(8, 2, bias=False, scale_groups=2)
    with torch.no_grad():
        lin.weight.copy_(torch.ones_like(lin.weight))
        lin.cast_scale.copy_(torch.tensor([[2.0, 3.0], [5.0, 7.0]]))
    sw = lin.scaled_weight()
    expected = torch.tensor(
        [[2.0] * 4 + [3.0] * 4, [5.0] * 4 + [7.0] * 4]
    )
    assert torch.equal(sw, expected), f"got\n{sw}\nexpected\n{expected}"
    return "A[k] axis: element (r,c) scaled by A[r, c//(C/n)] (contiguous column blocks)"


def test_finalize_exact_2of4_and_scale_folded():
    torch.manual_seed(1)
    lin = CastSparseLinear(16, 4, bias=False, scale_groups=2)
    with torch.no_grad():
        lin.weight.copy_(torch.randn_like(lin.weight))
        lin.cast_scale.copy_(torch.rand_like(lin.cast_scale) + 1.5)
    lin.refresh_mask()
    pre_scaled = lin.scaled_weight().detach().clone()
    lin.finalize()

    # exactly 2 nonzeros per group of 4
    nz = (lin.weight.detach() != 0).reshape(4, 4, 4).sum(-1)
    assert (nz == 2).all(), f"non-exact 2:4 after finalize: group nnz = {nz.tolist()}"
    assert lin.exact_nm_violations() == 0
    # kept entries equal the folded pre-finalize values
    kept = lin.mask
    assert torch.allclose(lin.weight.detach()[kept], pre_scaled[kept], atol=1e-6)
    assert torch.allclose(lin.cast_scale.detach(), torch.ones_like(lin.cast_scale))
    # forward now equals a plain sparse linear
    x = torch.randn(3, 16)
    assert torch.allclose(lin(x), torch.nn.functional.linear(x, lin.weight), atol=1e-6)
    return (
        f"finalize: every one of {nz.numel()} groups has exactly 2 nonzeros; "
        "scale folded into W and reset to 1"
    )


def test_forward_is_dense():
    """The mask must NOT appear in the forward pass (Sec. IV / Fig. 2)."""
    lin = CastSparseLinear(8, 2, bias=False, scale_groups=2)
    with torch.no_grad():
        lin.weight.copy_(torch.ones_like(lin.weight))
    lin.refresh_mask()
    x = torch.ones(1, 8)
    out = lin(x)
    # dense forward with all-ones weight and unit scale => sum of 8 inputs
    assert torch.allclose(out, torch.full_like(out, 8.0)), out
    return "forward is dense (masked weights still contribute): output=8.0 for 8 unit inputs"


# ---------------------------------------------------------------------------
# 6. Mask refresh timing: every T1 steps, BEFORE optimizer.step()
# ---------------------------------------------------------------------------
class _Tiny(nn.Module):
    """Mimics the LLaMA structure closely enough for convert_llama_to_cast."""

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.down_proj = nn.Linear(8, 8, bias=False)
        self.other = nn.Linear(8, 8, bias=False)  # must stay dense


def test_conversion_scope():
    m = _Tiny()
    names = convert_llama_to_cast(m, scale_groups=2)
    assert sorted(names) == ["down_proj", "q_proj"], names
    assert isinstance(m.q_proj, CastSparseLinear) and isinstance(m.down_proj, CastSparseLinear)
    assert not isinstance(m.other, CastSparseLinear), "non-target linear was converted"
    assert getattr(m.q_proj.weight, "cast_in_scope", False)
    assert not getattr(m.other.weight, "cast_in_scope", False)
    return f"converted exactly {names}; 'other' left dense and out of scope"


def test_mask_refresh_before_step_and_every_T1():
    """Regression for audit section 4.5: the refresh must happen at the top of
    step t, so the first AdamS step already sees a real 2:4 mask, not all-ones."""
    m = _Tiny()
    convert_llama_to_cast(m, scale_groups=2)
    opt = AdamS(build_param_groups(m, lr=1e-3), lr=1e-3, total_steps=100, l1_decay=1e-3)

    T1 = 10
    refreshed_at = []
    all_ones_at_first_step = None
    for step in range(0, 25):
        if step % T1 == 0:
            refresh_all_masks(m)
            refreshed_at.append(step)
        if step == 0:
            frac = float(m.q_proj.mask.float().mean())
            all_ones_at_first_step = frac == 1.0
        for p in m.parameters():
            p.grad = torch.zeros_like(p)
        opt.step()  # would raise if the mask were not a valid 2:4 pattern

    assert refreshed_at == [0, 10, 20], refreshed_at
    assert all_ones_at_first_step is False, "step 0 still used an all-ones mask"
    return (
        f"refreshed at steps {refreshed_at} (T1={T1}); step-0 mask already 2:4 "
        "(50% kept), and AdamS's exact-2:4 assertion held on every step"
    )


# ---------------------------------------------------------------------------
# 7. Distillation loss (Eq. 13)
# ---------------------------------------------------------------------------
def test_kl_zero_when_identical():
    torch.manual_seed(3)
    logits = torch.randn(2, 5, 11)
    kl = kl_divergence_loss(logits, logits.clone(), temperature=1.0)
    assert abs(float(kl)) < 1e-6, kl
    return f"D_KL(P||P) = {float(kl):.2e}"


def test_cast_loss_is_convex_combination():
    torch.manual_seed(4)
    s = torch.randn(2, 4, 7, requires_grad=True)
    t = torch.randn(2, 4, 7)
    labels = torch.randint(0, 7, (2, 4))
    eta = 1.0 / 3.0
    total, comp = cast_loss(s, t, labels, eta=eta, temperature=1.0)
    manual = eta * comp["kl"] + (1 - eta) * comp["ce"]
    assert abs(float(total.detach()) - manual) < 1e-5, (float(total.detach()), manual)
    # eta=0 must be exactly CE
    only_ce, c2 = cast_loss(s, t, labels, eta=0.0)
    assert abs(float(only_ce.detach()) - c2["ce"]) < 1e-6
    return (
        f"L = {eta:.3f}*KL + {1-eta:.3f}*CE checks out "
        f"(ce={comp['ce']:.4f}, kl={comp['kl']:.4f}, total={float(total.detach()):.4f}); eta=0 -> pure CE"
    )


def test_temperature_default_is_paper_literal():
    import inspect

    sig = inspect.signature(kl_divergence_loss)
    assert sig.parameters["temperature"].default == 1.0, "default T must be 1 (Eq. 13 has no T)"
    sig2 = inspect.signature(cast_loss)
    assert sig2.parameters["temperature"].default == 1.0
    assert sig2.parameters["eta"].default == 1.0 / 3.0, "eta default must be 1/3 (Table XI)"
    # AST-style T=2 must be a different number, i.e. the choice is not cosmetic
    torch.manual_seed(5)
    s, t = torch.randn(1, 3, 9), torch.randn(1, 3, 9)
    k1 = float(kl_divergence_loss(s, t, temperature=1.0))
    k2 = float(kl_divergence_loss(s, t, temperature=2.0))
    assert abs(k1 - k2) > 1e-3, (k1, k2)
    return f"default T=1.0 (paper), eta=1/3; KL@T=1={k1:.4f} vs AST-style T=2 -> {k2:.4f}"


def test_convex_unnormalised_equivalence():
    eta_p, lr_p = convex_to_unnormalised(1.0 / 3.0, 2e-5)
    assert abs(eta_p - 0.5) < 1e-9 and abs(lr_p - 2e-5 * 2 / 3) < 1e-12
    return f"eta=1/3 convex  <=>  eta'={eta_p:.3f} un-normalised with lr'={lr_p:.4e}"


# ---------------------------------------------------------------------------
# 8. End-to-end: masked weights actually converge to zero
# ---------------------------------------------------------------------------
def test_end_to_end_masked_weights_go_to_zero():
    """Miniature CAST run: does the audit's failure metric actually improve?

    Audit section 5 measured masked/kept magnitude ratio = 0.294 and only 21.5%
    of masked weights below 1e-4 -- i.e. the decay never ran.  Here a short but
    complete run must drive that ratio to ~0.

    NOTE the cosine LR decay is load-bearing, not cosmetic.  See
    ``test_terminal_magnitude_is_set_by_final_lr``: because Adam normalises the
    update, masked weights converge to an oscillation of amplitude ~lr, so the
    *final* LR sets the residual floor.  A constant LR leaves them parked at
    ~lr, never at zero.  The real recipe decays 2e-5 -> 2e-6.

    The LR here is scaled so that the alpha-weighted decay budget
    ``sum_t lr_t * alpha_t`` is ~4x the initial mean |W|, matching the headroom
    the paper recipe has for LLaMA2-7B (4.32x, see tools/decay_budget.py).  With
    a starved budget the masked weights provably cannot reach zero no matter how
    correct the optimizer is -- at 2x headroom this same toy leaves a heavy tail
    (p99 |masked| = 1.1e-2) and the final prune still costs 5% output error,
    which is a scaled-down version of the failed run's collapse.  Mean-based
    metrics hide that tail, so this test also asserts on the MAX.
    """
    torch.manual_seed(7)
    model = nn.Sequential()
    layer = CastSparseLinear(32, 16, bias=False, scale_groups=2)
    layer.weight.cast_in_scope = True
    model.add_module("q_proj", layer)

    T = 600
    lam = 1e-2
    lr0, min_lr = 4e-3, 4e-5
    opt = AdamS(build_param_groups(model, lr=lr0), lr=lr0, total_steps=T, l1_decay=lam)
    x = torch.randn(64, 32)
    target = torch.randn(64, 16)

    for step in range(T):
        if step % 10 == 0:
            refresh_all_masks(model)
        lr_t = min_lr + 0.5 * (lr0 - min_lr) * (1 + math.cos(math.pi * step / T))
        for g in opt.param_groups:
            g["lr"] = lr_t
        out = model(x)
        loss = ((out - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    w = layer.weight.detach().abs()
    masked = w[~layer.mask]
    kept = w[layer.mask]
    ratio = float(masked.mean() / kept.mean())
    frac_tiny = float((masked < 1e-4).float().mean())
    masked_max = float(masked.max())
    assert ratio < 0.01, f"masked/kept ratio {ratio:.4f} not << the broken run's 0.294"
    assert frac_tiny > 0.99, f"only {frac_tiny:.1%} of masked weights below 1e-4"
    assert masked_max < 1e-3, f"tail survived: max |masked| = {masked_max:.2e}"

    # hard prune must now be ~free
    dense_out = model(x)
    layer.finalize()
    sparse_out = model(x)
    rel = float((sparse_out - dense_out).norm() / dense_out.norm())
    assert layer.exact_nm_violations() == 0
    assert rel < 0.005, f"pruning changed the output by {rel:.3%}"
    return (
        f"after {T} steps (lr {lr0:.0e}->{min_lr:.0e}, 4.2x decay headroom): masked/kept "
        f"ratio={ratio:.6f} (broken run: 0.294), {frac_tiny:.1%} of masked < 1e-4 "
        f"(broken run: 21.5%), max |masked|={masked_max:.1e}, hard-prune delta={rel:.4%}"
    )


def test_terminal_magnitude_is_set_by_final_lr():
    """Mechanistic check, and the reason a decaying LR is mandatory.

    With zero gradient and alpha -> 1, mu~ = alpha*lam*sign(w) and
    v = (alpha*lam)^2, so mu^/sqrt(v^) -> sign(w) and the step size saturates at
    ~lr regardless of lambda.  Masked weights therefore cannot settle below
    O(lr); lambda controls how fast the decay dominates the gradient, not the
    final floor.  Consequence for the 7500-step run: min_lr must be small enough
    (2e-6) that O(min_lr) is a negligible residual.
    """
    lam, lr, T = 1e-2, 1e-3, 200
    torch.manual_seed(0)
    lin = CastSparseLinear(8, 1, bias=False, scale_groups=2)
    with torch.no_grad():
        lin.weight.fill_(0.5)
    lin.refresh_mask()
    opt = AdamS(build_param_groups(lin, lr=lr), lr=lr, total_steps=T, l1_decay=lam)
    step_sizes = []
    for _ in range(T):
        lin.weight.grad = torch.zeros_like(lin.weight)
        lin.cast_scale.grad = torch.zeros_like(lin.cast_scale)
        before = lin.weight.detach().clone()
        opt.step()
        step_sizes.append(float((lin.weight.detach() - before)[~lin.mask].abs().mean()))
    late = step_sizes[-1] / lr
    assert 0.5 < late < 3.0, f"late step size / lr = {late:.3f}, expected O(1)"
    return (
        f"late per-step |dw| = {late:.2f} x lr (independent of lambda) -> the residual "
        "floor is O(final lr), so the LR schedule must decay for masked weights to vanish"
    )


# ---------------------------------------------------------------------------
def main():
    tests = [
        test_nm_mask_hand_computed,
        test_nm_mask_ties_still_exact,
        test_adams_decays_masked_not_kept,
        test_adams_sign_at_zero_is_zero,
        test_adams_alpha_ramps_linearly_zero_to_one,
        test_second_moment_uses_mu_tilde,
        test_missing_mask_raises,
        test_shape_mismatch_raises,
        test_expected_element_count_enforced,
        test_non_fp32_in_scope_raises,
        test_bf16_swallows_lambda_fp32_does_not,
        test_scale_groups_axis_semantics,
        test_finalize_exact_2of4_and_scale_folded,
        test_forward_is_dense,
        test_conversion_scope,
        test_mask_refresh_before_step_and_every_T1,
        test_kl_zero_when_identical,
        test_cast_loss_is_convex_combination,
        test_temperature_default_is_paper_literal,
        test_convex_unnormalised_equivalence,
        test_end_to_end_masked_weights_go_to_zero,
        test_terminal_magnitude_is_set_by_final_lr,
    ]
    failed = 0
    for fn in tests:
        try:
            msg = fn()
            print(f"PASS  {fn.__name__}\n        {msg}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {fn.__name__}\n        {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
