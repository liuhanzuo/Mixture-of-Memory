#!/usr/bin/env python3
"""Diagnose slot diversity for Experiment E (Contrastive) checkpoint.

Purpose: Verify or refute the hypothesis that all slot contents are identical
(mean-pool broadcast) after training. Loads the step_4000 checkpoint, runs a
single forward pass, and computes pairwise cosine similarity and L2 distance
between all 64 slots at selected layers.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/diagnose_slot_diversity.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

# Setup project path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import LlamaForCausalLM


def main():
    # ---- Config matching Experiment E (launch_experiment_e_contrastive.sh) ----
    model_path = "/apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b"
    ckpt_path = "outputs/experiment_e_contrastive/step_4000.pt"
    data_path = "data/wikitext_chunks_llama3_4096.npy"
    log_path = "logs/slot_diversity_diagnosis.txt"

    num_slots = 64
    seq_len = 4096
    dtype = torch.bfloat16
    device = torch.device("cuda:0")
    layers_to_inspect = [0, 8, 16, 24, 31]

    os.makedirs("logs", exist_ok=True)

    # Collect output lines for both stdout and file
    output_lines = []

    def log(msg: str):
        print(msg)
        output_lines.append(msg)

    log("=" * 70)
    log("Slot Diversity Diagnosis — Experiment E (Contrastive), step_4000")
    log("=" * 70)

    # ---- Load base model ----
    t0 = time.time()
    log(f"\n[1/4] Loading base model from {model_path} ...")
    base_model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map={"": device},
    )
    log(f"       Base model loaded in {time.time() - t0:.1f}s")

    # ---- Build CrossAttentionMemoryModel ----
    # Import after path setup
    from scripts.train_cross_attn_memory import CrossAttentionMemoryModel

    log("\n[2/4] Building CrossAttentionMemoryModel (slot_forward, strided init) ...")
    cm_model = CrossAttentionMemoryModel(
        base_model=base_model,
        num_slots=num_slots,
        top_k=8,
        full_finetune=True,
        use_memory=True,
        use_cross_attn_memory=True,
        gradient_checkpointing=False,  # Not needed for inference
        cross_attn_dropout=0.0,
        residual_scale=0.01,
        swa_window=0,
        write_lr=0.1,
        slot_forward=True,
        slot_isolated=False,
        memory_init="strided",
        recon_loss_weight=0.0,
        cross_chunk_propagation=False,
    ).to(device).to(dtype)

    # ---- Load checkpoint ----
    log(f"\n[3/4] Loading checkpoint from {ckpt_path} ...")
    t1 = time.time()
    ckpt = torch.load(ckpt_path, map_location=device)
    cm_model.load_state_dict(ckpt['model_state_dict'])
    log(f"       Checkpoint loaded in {time.time() - t1:.1f}s (step={ckpt.get('global_step', '?')})")

    cm_model.eval()

    # ---- Load data ----
    log(f"\n[4/4] Loading data from {data_path} ...")
    all_chunks = np.load(data_path, mmap_mode="r")
    # Take first chunk as our batch (B=1, T=seq_len)
    input_ids_np = all_chunks[0][:seq_len].astype(np.int64)
    input_ids = torch.tensor(input_ids_np, dtype=torch.long, device=device).unsqueeze(0)  # [1, 4096]
    log(f"       Input shape: {input_ids.shape}")

    # ---- Forward pass ----
    log("\nRunning forward pass (slot_forward mode) ...")
    t2 = time.time()
    with torch.no_grad():
        result = cm_model._forward_slot_forward(input_ids, labels=None)
    log(f"Forward pass completed in {time.time() - t2:.1f}s")

    # ---- Analyze slot diversity ----
    log("\n" + "=" * 70)
    log("SLOT DIVERSITY ANALYSIS")
    log("=" * 70)
    log(f"{'Layer':>6} | {'Cos Mean':>9} {'Cos Min':>9} {'Cos Max':>9} | {'L2 Mean':>9} {'L2 Min':>9} {'L2 Max':>9}")
    log("-" * 70)

    all_cos_means = []
    all_l2_means = []

    for layer_idx in layers_to_inspect:
        slots = cm_model.slot_values[layer_idx]  # [1, 64, D]
        assert slots is not None, f"slot_values[{layer_idx}] is None!"
        slots = slots[0]  # [64, D]

        S = slots.shape[0]
        D = slots.shape[1]

        # Pairwise cosine similarity (64x64)
        slots_norm = F.normalize(slots, p=2, dim=-1)  # [64, D]
        cos_matrix = slots_norm @ slots_norm.T  # [64, 64]

        # Exclude diagonal
        mask = ~torch.eye(S, dtype=torch.bool, device=device)
        cos_off_diag = cos_matrix[mask]

        cos_mean = cos_off_diag.mean().item()
        cos_min = cos_off_diag.min().item()
        cos_max = cos_off_diag.max().item()

        # Pairwise L2 distance
        # Use broadcasting: [64,1,D] - [1,64,D] -> [64,64,D] -> norm -> [64,64]
        l2_matrix = torch.cdist(slots.unsqueeze(0), slots.unsqueeze(0), p=2)[0]  # [64, 64]
        l2_off_diag = l2_matrix[mask]

        l2_mean = l2_off_diag.mean().item()
        l2_min = l2_off_diag.min().item()
        l2_max = l2_off_diag.max().item()

        log(f"{layer_idx:>6} | {cos_mean:>9.4f} {cos_min:>9.4f} {cos_max:>9.4f} | {l2_mean:>9.4f} {l2_min:>9.4f} {l2_max:>9.4f}")

        all_cos_means.append(cos_mean)
        all_l2_means.append(l2_mean)

        # Print first and last slot's first 10 dims
        s0 = slots[0, :10].tolist()
        s63 = slots[63, :10].tolist()
        log(f"       slot[0][:10]  = [{', '.join(f'{v:.4f}' for v in s0)}]")
        log(f"       slot[63][:10] = [{', '.join(f'{v:.4f}' for v in s63)}]")
        log("")

    # ---- Conclusion ----
    log("=" * 70)
    log("CONCLUSION")
    log("=" * 70)

    avg_cos = sum(all_cos_means) / len(all_cos_means)
    avg_l2 = sum(all_l2_means) / len(all_l2_means)

    log(f"Average pairwise cosine similarity across inspected layers: {avg_cos:.4f}")
    log(f"Average pairwise L2 distance across inspected layers: {avg_l2:.4f}")
    log("")

    if avg_cos > 0.99:
        log(">>> HYPOTHESIS CONFIRMED: Slots are nearly identical (cos > 0.99).")
        log("    This strongly suggests a mean-pool broadcast effect —")
        log("    all slots converge to the same representation.")
    elif avg_cos > 0.95:
        log(">>> HYPOTHESIS PARTIALLY CONFIRMED: Slots are very similar (cos 0.95-0.99).")
        log("    There is minimal diversity; likely limited differentiation.")
    elif avg_cos > 0.7:
        log(">>> MIXED EVIDENCE: Moderate similarity (cos 0.7-0.95).")
        log("    Slots share common structure but have some differentiation.")
    else:
        log(">>> HYPOTHESIS REFUTED: Slots show significant diversity (cos < 0.7).")
        log("    The broadcast/mean-pool collapse is NOT happening.")

    if avg_l2 < 0.01:
        log("    L2 distance < 0.01 further confirms near-identity.")
    elif avg_l2 < 1.0:
        log(f"    L2 distance = {avg_l2:.4f} indicates moderate but limited spread.")
    else:
        log(f"    L2 distance = {avg_l2:.4f} indicates substantial differentiation.")

    log("\n" + "=" * 70)

    # Write to file
    with open(log_path, "w") as f:
        f.write("\n".join(output_lines) + "\n")
    print(f"\nResults written to {log_path}")


if __name__ == "__main__":
    main()
