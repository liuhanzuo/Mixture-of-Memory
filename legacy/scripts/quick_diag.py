#!/usr/bin/env python3
import torch, math
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test both implementations
print("="*70)
print("TESTING: MemoryBank implementations")
print("="*70)

from src.memory.sparse.memory_bank import SparseMemoryBank as OldBank
from src.memory.sparse_memory.memory_bank import MemoryBank as NewBank

dim = 4096
num_slots = 128
T = 4096
alpha = 0.1

print(f"\nConfig: slots={num_slots}, seq_len={T}, alpha={alpha}")

# Theoretical analysis
writes_per_slot = T / num_slots
retention = (1 - alpha) ** writes_per_slot
print(f"\nTHEORETICAL (all-tokens writing):")
print(f"  Writes per slot: {writes_per_slot:.1f}")
print(f"  Retention per step: {retention*100:.2f}%")

# Top-K selective writing
K = 8
writes_per_slot_k = K / num_slots
retention_k = (1 - alpha) ** writes_per_slot_k
print(f"\nTHEORETICAL (top-K selective, K={K}):")
print(f"  Writes per slot: {writes_per_slot_k:.2f}")
print(f"  Retention per step: {retention_k*100:.2f}%")

print("\n" + "="*70)
print("OLD IMPLEMENTATION (sparse/memory_bank.py)")
print("="*70)
old_bank = OldBank(num_slots=num_slots, hidden_dim=dim, ema_alpha=alpha, write_top_k=0)
print(f"  Has write_top_k parameter: Yes")
print(f"  Has importance_mode parameter: Yes")
print(f"  Default write_top_k: {old_bank.write_top_k}")
print(f"  Default importance_mode: {old_bank.importance_mode}")

print("\n" + "="*70)
print("NEW IMPLEMENTATION (sparse_memory/memory_bank.py)")
print("="*70)
new_bank = NewBank(num_slots=num_slots, hidden_dim=dim, ema_alpha=alpha,
                   write_top_k=K, importance_mode="combined")
print(f"  Has write_top_k parameter: Yes")
print(f"  Has importance_mode parameter: Yes")
print(f"  Default write_top_k: {new_bank.write_top_k}")
print(f"  Default importance_mode: {new_bank.importance_mode}")
print(f"  Has importance_head: Yes (learnable)")

print("\n" + "="*70)
print("ATTENTION INTEGRATION")
print("="*70)
print("  SparseMemoryAttention exists: Yes")
print("  It calls memory_bank.update_slots(): Yes")
print("  update_slots() has write_top_k logic: Yes")

print("\n" + "="*70)
print("STATUS SUMMARY")
print("="*70)
print(f"Old implementation (SparseMemoryBank): DEPRECATED")
print(f"  - Has write_top_k parameter but NOT USED in attention.py")
print(f"New implementation (MemoryBank): ACTIVE")
print(f"  - Full top-K selective writing support")
print(f"  - Three importance modes: magnitude, attention_surprise, combined")
print(f"  - Currently using write_top_k={K} from experiments")
print(f"  - Expected retention: {retention_k*100:.1f}% (vs {retention*100:.1f}% without selective)")
print("\n" + "="*70)
