"""Q-Filters KV-cache compression.

Training-free KV-cache compression via Q-geometry filters
(Godey et al., 2025 — arXiv:2503.02812).

Public API:
    QFiltersConfig(kv_budget, filter_rank, recent_window, calibration_chunks)
    QFiltersCache(filters, config)                   — DynamicCache that prunes on update
    patch_model(model, filters, config)              — monkeypatch LlamaAttention.forward
    compute_filters(model, calib_loader, rank)       — offline calibration (SVD of Q)
    compress_kv(queries_proj, filters, keys, values, budget, recent_window)
"""
from .calibration import compute_filters
from .compression import compress_kv, score_keys
from .layer import (
    QFiltersAttention,
    QFiltersCache,
    QFiltersConfig,
    patch_model,
)

__all__ = [
    "QFiltersAttention",
    "QFiltersCache",
    "QFiltersConfig",
    "compress_kv",
    "compute_filters",
    "patch_model",
    "score_keys",
]
