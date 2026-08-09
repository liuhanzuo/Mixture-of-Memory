"""CAST reproduction (arXiv:2509.25996v1). See ../SPEC.md for the source-level
mapping of every hyperparameter and implementation choice."""

from .adams import AdamS, MaskCoverageError, build_param_groups
from .checkpoint import (
    CKPT_FORMAT,
    RESUME_CRITICAL_ARGS,
    ResumeMismatchError,
    assert_optimizer_state_restored,
    check_resume_args,
    checkpoint_size_bytes,
    find_latest_checkpoint,
    load_training_state,
    prune_old_checkpoints,
    save_training_state,
)
from .diagnostics import exactness_report, magnitude_report
from .distill import cast_loss, convex_to_unnormalised, kl_divergence_loss
from .sparse_linear import (
    LLAMA_INBLOCK_PROJECTIONS,
    CastSparseLinear,
    cast_modules,
    cast_scope_stats,
    convert_llama_to_cast,
    finalize_all,
    nm_magnitude_mask,
    refresh_all_masks,
)

#: LLaMA2-7B, 2:4 over the 224 in-block projections.
#: 32 layers x (4*4096*4096 + 3*4096*11008) = 6,476,005,376 params; half masked.
LLAMA2_7B_CAST_TENSORS = 224
LLAMA2_7B_CAST_ELEMENTS = 6_476_005_376
LLAMA2_7B_DECAYED_ELEMENTS = 3_238_002_688

__all__ = [
    "AdamS",
    "MaskCoverageError",
    "build_param_groups",
    "ResumeMismatchError",
    "RESUME_CRITICAL_ARGS",
    "CKPT_FORMAT",
    "check_resume_args",
    "save_training_state",
    "load_training_state",
    "assert_optimizer_state_restored",
    "checkpoint_size_bytes",
    "find_latest_checkpoint",
    "prune_old_checkpoints",
    "magnitude_report",
    "exactness_report",
    "cast_loss",
    "kl_divergence_loss",
    "convex_to_unnormalised",
    "CastSparseLinear",
    "convert_llama_to_cast",
    "cast_modules",
    "cast_scope_stats",
    "refresh_all_masks",
    "finalize_all",
    "nm_magnitude_mask",
    "LLAMA_INBLOCK_PROJECTIONS",
    "LLAMA2_7B_CAST_TENSORS",
    "LLAMA2_7B_CAST_ELEMENTS",
    "LLAMA2_7B_DECAYED_ELEMENTS",
]
