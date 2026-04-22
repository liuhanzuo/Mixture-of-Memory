#!/usr/bin/env python3
"""Evaluate gate statistics after Phase 1 training.

This script analyzes gate statistics from a trained checkpoint and compares
against the Phase 0 (v5) baseline to verify that gate initialization is working.

Usage:
    python scripts/eval_phase1_gate.py --checkpoint_path outputs/phase1_gate_init_v6/checkpoint-300
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.sparse import SparseMemoryModel
from transformers import AutoConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate gate statistics after Phase 1 training")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to checkpoint directory")
    parser.add_argument("--model_name_or_path", type=str,
                        default="models/Llama--Llama2-7b",
                        help="Base model path (for loading SparseMemoryModel)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run evaluation on")
    return parser.parse_args()


def compute_gate_statistics(gated_attn, device, hidden_dim=4096, num_heads=32, batch_size=1, seq_len=512):
    """Compute gate statistics from a sample forward pass."""

    # Create dummy input
    dummy_hidden = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=torch.bfloat16)

    # Compute gate values
    gate_input = dummy_hidden
    gate = torch.sigmoid(gated_attn.gate_proj(gate_input))  # [B, T, 1]

    # Statistics
    gate_mean = gate.mean().item()
    gate_std = gate.std().item()
    gate_min = gate.min().item()
    gate_max = gate.max().item()
    gate_median = gate.median().item()

    # Compute gradient at current activation
    # σ'(x) = σ(x) * (1 - σ(x))
    gate_grad = gate * (1 - gate)
    grad_mean = gate_grad.mean().item()

    return {
        'mean': gate_mean,
        'std': gate_std,
        'min': gate_min,
        'max': gate_max,
        'median': gate_median,
        'grad_mean': grad_mean,
    }


def evaluate_checkpoint(args):
    """Load checkpoint and evaluate gate statistics."""

    print(f"Loading checkpoint: {args.checkpoint_path}")
    print(f"Base model: {args.model_name_or_path}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load base model
    base_model = torch.load(
        Path(args.checkpoint_path) / "pytorch_model.bin",
        map_location=device,
    )

    # TODO: For now, we'll just load a fresh model and analyze init
    # In production, load the actual checkpoint weights

    # Load config
    config = AutoConfig.from_pretrained(args.model_name_or_path)

    print("\n" + "="*60)
    print("Phase 0 (v5) Baseline (gate_bias_init=2.0):")
    print("="*60)
    print("  gate_mean: 0.876 ± 0.0004")
    print("  Problem: Saturated (sigmoid(2.0) ≈ 0.88)")
    print("  Sigmoid gradient: ~0.105 (vs 0.25 at 0.5)")
    print("  Result: ~2.4x gradient reduction, amplified to ~4000x")
    print("  All layers: identical alpha ~0.92 (no selectivity)")

    print("\n" + "="*60)
    print("Phase 1 Expected (gate_bias_init=0.0):")
    print("="*60)
    print("  Initial sigmoid: 0.5 (max gradient)")
    print("  Initial sigmoid gradient: 0.25 (maximum)")
    print("  Expected gate_mean at step 100: ~0.4-0.6")
    print("  Expected gate_std at step 100: >0.01 (layers learning)")
    print("  Expected alpha values: varying across layers")

    print("\n" + "="*60)
    print("Success Criteria:")
    print("="*60)
    print("  ✅ gate_mean != 0.876 (not saturated)")
    print("  ✅ gate_std > 0.05 (layers learning different selectivity)")
    print("  ✅ gate_grad_mean > 0.15 (gradient not diluted)")
    print("  ✅ Alpha values vary across layers (not all ~0.92)")

    print("\n" + "="*60)
    print("Next Steps:")
    print("="*60)
    print("  1. Run training: bash scripts/_launch_phase1_v6.sh")
    print("  2. Monitor logs: Check gate_mean/gate_std at step 100, 300")
    print("  3. If successful: Proceed to Phase 2 (negative gate bias)")
    print("  4. If failed: Investigate EMA dynamics, top-k retrieval")


def parse_trainer_log(log_path):
    """Parse trainer log to extract gate statistics."""

    print(f"\nParsing trainer log: {log_path}")

    gate_stats = []
    with open(log_path, 'r') as f:
        for line in f:
            try:
                log_entry = json.loads(line)
                if 'gate_mean' in log_entry:
                    gate_stats.append({
                        'step': log_entry.get('step', -1),
                        'gate_mean': log_entry.get('gate_mean'),
                        'gate_std': log_entry.get('gate_std'),
                        'loss': log_entry.get('loss'),
                    })
            except json.JSONDecodeError:
                continue

    if gate_stats:
        print("\nGate Statistics from Log:")
        print("-" * 60)
        for stat in gate_stats[-10:]:  # Show last 10 entries
            print(f"Step {stat['step']:4d}: "
                  f"gate_mean={stat['gate_mean']:.4f}, "
                  f"gate_std={stat['gate_std']:.4f}, "
                  f"loss={stat['loss']:.4f}")

        # Analyze trends
        print("\nAnalysis:")
        print("-" * 60)
        final = gate_stats[-1]
        print(f"Final gate_mean: {final['gate_mean']:.4f}")
        print(f"Final gate_std: {final['gate_std']:.4f}")

        if final['gate_std'] > 0.05:
            print("✅ gate_std > 0.05: Layers are learning different selectivity!")
        else:
            print("❌ gate_std < 0.05: Layers may still be uniform")

        if abs(final['gate_mean'] - 0.876) > 0.1:
            print("✅ gate_mean != 0.876: Not saturated like v5!")
        else:
            print("❌ gate_mean ≈ 0.876: Still saturated like v5")

    else:
        print("No gate statistics found in log. Check logging configuration.")


def main():
    args = parse_args()

    # Evaluate checkpoint
    evaluate_checkpoint(args)

    # Also check for trainer log
    checkpoint_path = Path(args.checkpoint_path)
    log_path = checkpoint_path / "trainer_log.jsonl"
    if log_path.exists():
        parse_trainer_log(log_path)
    else:
        print(f"\nTrainer log not found at {log_path}")
        print("Logs may be in a different location or not yet generated.")


if __name__ == "__main__":
    main()
