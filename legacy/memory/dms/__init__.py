"""Dynamic Memory Sparsification (DMS) for KV cache compression.

Based on: Łańcucki et al., "Inference-Time Hyper-Scaling with KV Cache Compression"
(arXiv:2506.05345), NVIDIA.

DMS retrofitting: per-token, per-layer, per-head binary eviction decisions
using Gumbel-Sigmoid relaxation. Compatible with Llama-2-7B and Qwen3-8B (GQA).
"""

from .dms_attention import DMSAttentionWrapper, apply_dms_to_model
from .dms_decision_head import DMSDecisionHead, gumbel_sigmoid
from .dms_training import DMSLoss, CompressionScheduler

__all__ = [
    "DMSAttentionWrapper",
    "apply_dms_to_model",
    "DMSDecisionHead",
    "gumbel_sigmoid",
    "DMSLoss",
    "CompressionScheduler",
]
