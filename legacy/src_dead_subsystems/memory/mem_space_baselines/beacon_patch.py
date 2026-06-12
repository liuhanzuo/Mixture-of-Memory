"""Beacon Patch — apply Activation Beacon compression to a HuggingFace Llama model.

This is independent from the existing patch.py (slot memory). It wraps the model
with BeaconModel which manages per-layer beacon projections and streaming
interval-based compression.

Usage:
    from src.memory.mem_space.beacon_patch import apply_beacon_to_model
    model, beacon_model = apply_beacon_to_model(base_model, n_beacon=4, interval_size=512)
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from .beacon import BeaconLayer, BeaconModel


def apply_beacon_to_model(
    model: nn.Module,
    n_beacon: int = 4,
    interval_size: int = 512,
) -> Tuple[nn.Module, BeaconModel]:
    """Wrap a HuggingFace LlamaForCausalLM with beacon compression.

    Args:
        model: A HuggingFace LlamaForCausalLM (or compatible) model.
        n_beacon: Number of beacon tokens per interval boundary.
        interval_size: Base interval size in tokens (overridden by
            compression_ratio at training time).

    Returns:
        (original_model, beacon_model) where beacon_model wraps the
        original and manages beacon projections + streaming forward.
        The original model's parameters are frozen; only beacon params
        are trainable.
    """
    # Validate model structure
    root = getattr(model, "model", model)
    layers = getattr(root, "layers", None)
    if layers is None:
        raise RuntimeError(
            "apply_beacon_to_model: could not locate `model.model.layers`; "
            "only Llama-family architectures are supported."
        )

    # Freeze all base model parameters
    for p in model.parameters():
        p.requires_grad = False

    # Create BeaconModel wrapper
    beacon_model = BeaconModel(
        base_model=model,
        n_beacon=n_beacon,
        interval_size=interval_size,
    )

    # Ensure only beacon params are trainable
    for p in beacon_model.get_trainable_params():
        p.requires_grad = True

    return model, beacon_model


def count_beacon_params(beacon_model: BeaconModel) -> dict:
    """Count trainable vs total parameters in the beacon model.

    Returns:
        dict with 'trainable', 'total', 'ratio' keys.
    """
    trainable = sum(p.numel() for p in beacon_model.get_trainable_params())
    total = sum(p.numel() for p in beacon_model.base_model.parameters())
    return {
        "trainable": trainable,
        "total": total,
        "beacon_ratio": trainable / max(total, 1),
    }


def freeze_base_unfreeze_beacon(beacon_model: BeaconModel) -> None:
    """Utility to re-freeze base and unfreeze beacon after loading checkpoint."""
    for p in beacon_model.base_model.parameters():
        p.requires_grad = False
    for p in beacon_model.get_trainable_params():
        p.requires_grad = True
