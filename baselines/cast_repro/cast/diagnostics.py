"""Sparsification diagnostics -- reproduces the audit's failure metrics.

Mixture-of-Memory/SparseForge_Data/docs/CAST_REPRODUCTION_AUDIT.md section 5
diagnosed the broken run by sampling the 224 in-block linear layers of the
pre-finalization checkpoint:

    masked/pruned weight mean magnitude = 0.00669
    kept   weight mean magnitude        = 0.02259
    mean magnitude ratio               = 0.294     <-- should be ~0
    median magnitude ratio             = 0.345
    21.5% of masked weights < 1e-4               <-- should be ~100%
    27.3% of masked weights < 1e-3

    per-projection ratio: q 0.210  v 0.128  k 0.342  o 0.344
                          gate 0.346  up 0.345  down 0.344

Those numbers are the signature of AdamS never running: only q/v happened to
align under FSDP.  ``magnitude_report`` recomputes the same statistics so a new
checkpoint can be checked against the same yardstick.

Acceptance targets for a correct run (see also Sec. IV-A: "masked weights are
reduced to negligible magnitudes by the end", and Appendix C: the sparse weight
ratio S_t reaches 1):

    ratio_mean            << 0.294, target < 0.01
    frac_below_1e4        >> 21.5%, target > 95%
    max_masked_magnitude  small -- the mean hides a heavy tail, so this is the
                          honest indicator (a 2x-underbudgeted toy run still
                          shows ratio 0.004 while max |masked| = 7e-2)
    sparse_weight_ratio   -> 1.0   (Appendix C definition, S_t)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch


def _proj_kind(name: str) -> str:
    for k in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        if name.endswith(k):
            return k
    return "other"


@torch.no_grad()
def magnitude_report(model, thresholds=(1e-4, 1e-3), max_modules: Optional[int] = None) -> Dict:
    """Masked-vs-kept magnitude statistics, overall and per projection type.

    Works on a live model or one loaded from a pre-finalization checkpoint (it
    reads ``weight`` and the ``mask`` buffer, so it must be run BEFORE
    ``finalize()`` -- afterwards the masked entries are exactly zero by
    construction and the metric is vacuous).
    """
    from .sparse_linear import cast_modules

    per_kind: Dict[str, Dict[str, float]] = {}
    tot_masked_abs = 0.0
    tot_kept_abs = 0.0
    n_masked = 0
    n_kept = 0
    below = {t: 0 for t in thresholds}
    max_masked = 0.0
    ratios: List[float] = []
    n_mod = 0

    for name, mod in cast_modules(model):
        if max_modules is not None and n_mod >= max_modules:
            break
        n_mod += 1
        w = mod.weight.detach().abs().float()
        keep = mod.mask if mod.mask.dtype == torch.bool else mod.mask > 0.5
        masked_vals = w[~keep]
        kept_vals = w[keep]
        if masked_vals.numel() == 0 or kept_vals.numel() == 0:
            continue

        m_sum = float(masked_vals.sum())
        k_sum = float(kept_vals.sum())
        tot_masked_abs += m_sum
        tot_kept_abs += k_sum
        n_masked += masked_vals.numel()
        n_kept += kept_vals.numel()
        max_masked = max(max_masked, float(masked_vals.max()))
        for t in thresholds:
            below[t] += int((masked_vals < t).sum())

        m_mean = m_sum / masked_vals.numel()
        k_mean = k_sum / kept_vals.numel()
        r = m_mean / k_mean if k_mean > 0 else float("nan")
        ratios.append(r)

        kind = _proj_kind(name)
        d = per_kind.setdefault(kind, {"masked_abs": 0.0, "kept_abs": 0.0, "n_masked": 0, "n_kept": 0})
        d["masked_abs"] += m_sum
        d["kept_abs"] += k_sum
        d["n_masked"] += masked_vals.numel()
        d["n_kept"] += kept_vals.numel()

    masked_mean = tot_masked_abs / max(1, n_masked)
    kept_mean = tot_kept_abs / max(1, n_kept)
    ratios_t = torch.tensor(ratios) if ratios else torch.tensor([float("nan")])

    by_projection = {}
    for k, d in sorted(per_kind.items()):
        mm = d["masked_abs"] / max(1, d["n_masked"])
        km = d["kept_abs"] / max(1, d["n_kept"])
        by_projection[k] = round(mm / km, 6) if km > 0 else None

    # Appendix C: S_t = ||W (*) M||_1 / ||W||_1 -> 1 when the decay worked.
    total_abs = tot_masked_abs + tot_kept_abs
    sparse_weight_ratio = tot_kept_abs / total_abs if total_abs > 0 else float("nan")

    summary = {
        "modules": n_mod,
        "masked_mean_magnitude": round(masked_mean, 8),
        "kept_mean_magnitude": round(kept_mean, 8),
        "ratio_mean": round(masked_mean / kept_mean, 6) if kept_mean > 0 else None,
        "ratio_median_over_layers": round(float(ratios_t.median()), 6),
        "max_masked_magnitude": round(max_masked, 8),
        "sparse_weight_ratio": round(sparse_weight_ratio, 6),
    }
    for t in thresholds:
        summary[f"frac_masked_below_{t:g}"] = round(below[t] / max(1, n_masked), 6)

    # explicit side-by-side with the broken run
    summary["BROKEN_RUN_ratio_mean"] = 0.294
    summary["BROKEN_RUN_frac_below_1e-4"] = 0.215
    verdict = "UNKNOWN"
    rm = summary["ratio_mean"]
    fb = summary.get("frac_masked_below_0.0001")
    if rm is not None and fb is not None:
        if rm < 0.01 and fb > 0.95:
            verdict = "OK: masked weights collapsed; hard prune should be ~free"
        elif rm < 0.05:
            verdict = "MARGINAL: decayed but a tail survives; check max_masked_magnitude"
        else:
            verdict = "BAD: looks like the broken run -- AdamS is probably not running"
    summary["verdict"] = verdict

    return {"summary": summary, "by_projection": by_projection}


@torch.no_grad()
def exactness_report(model) -> Dict:
    """Verify the exact N:M pattern of the *weights* (run AFTER finalize())."""
    from .sparse_linear import cast_modules

    viol = 0
    mods = 0
    zero_frac_num = 0
    zero_frac_den = 0
    for _, mod in cast_modules(model):
        mods += 1
        viol += mod.exact_nm_violations()
        w = mod.weight.detach()
        zero_frac_num += int((w == 0).sum())
        zero_frac_den += w.numel()
    return {
        "modules": mods,
        "groups_violating_exact_nm": viol,
        "zero_fraction": round(zero_frac_num / max(1, zero_frac_den), 6),
        "exact": viol == 0,
    }
