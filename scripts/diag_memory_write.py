"""Quick diagnosis of memory bank write behavior.

Checks memory diversity and retention without needing complex hooks.
"""
import os, sys, json, math, torch, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def main():
    num_slots = 128
    top_k = 8
    sliding_window = 256
    ema_alpha = 0.1
    seq_length = 4096

    from src.memory.sparse_memory.memory_bank import MemoryBank

    print("=" * 70)
    print("DIAGNOSIS: Memory Bank Write Behavior")
    print("=" * 70)
    print(f"Config: slots={num_slots}, top_k={top_k}, window={sliding_window}, alpha={ema_alpha}, seq_len={seq_length}")
    print()

    # === THEORETICAL ANALYSIS ===
    print("--- THEORETICAL ---")
    T = seq_length
    writes_per_slot = T / num_slots
    retention_per_step = (1 - ema_alpha) ** writes_per_slot
    print(f"Tokens per step: {T}")
    print(f"Num slots: {num_slots}")
    print(f"Writes per slot per step: {T}/{num_slots} = {writes_per_slot:.1f}")
    print(f"Old content retention per step: (1-{ema_alpha})^{writes_per_slot:.1f} = {retention_per_step:.6f} ({retention_per_step*100:.2f}%)")
    print(f"→ Memory is {(1-retention_per_step)*100:.1f}% overwritten with current segment each step")
    print(f"→ Memory is essentially a compressed copy of the CURRENT window")
    print()

    # How many slots needed for 50% retention?
    max_writes_for_50pct = math.log(0.5) / math.log(1 - ema_alpha)
    slots_for_50pct = math.ceil(T / max_writes_for_50pct)
    print(f"For 50% retention: need writes/slot < {max_writes_for_50pct:.1f} → need > {slots_for_50pct} slots")
    print(f"For 80% retention: need writes/slot < {math.log(0.8)/math.log(1-ema_alpha):.1f} → need > {math.ceil(T/(math.log(0.8)/math.log(1-ema_alpha)))} slots")

    # With selective write (only top-K tokens)
    print(f"\nWith SELECTIVE write (top-{top_k} tokens only):")
    selective_writes = top_k / num_slots
    selective_retention = (1 - ema_alpha) ** selective_writes
    print(f"  Writes per slot: {top_k}/{num_slots} = {selective_writes:.2f}")
    print(f"  Retention per step: {selective_retention*100:.2f}%")
    print(f"  → Memory would actually preserve historical information!")

    # === EMPIRICAL: Simulate circular buffer ===
    print("\n--- EMPIRICAL SIMULATION ---")
    
    # Simulate: create random "hidden states" for 3 consecutive segments
    torch.manual_seed(42)
    dim = 4096
    
    bank = MemoryBank(num_slots=num_slots, hidden_dim=dim, ema_alpha=ema_alpha, dtype=torch.float32)
    bank.reset(batch_size=1)
    
    # Segment 1: fill memory
    h1 = torch.randn(seq_length, dim)
    bank.write(h1, batch_idx=0)
    mem_after_1 = bank.memory[0].detach().clone()
    
    # Segment 2: different data
    h2 = torch.randn(seq_length, dim)
    bank.write(h2, batch_idx=0)
    mem_after_2 = bank.memory[0].detach().clone()
    
    # Segment 3: different data
    h3 = torch.randn(seq_length, dim)
    bank.write(h3, batch_idx=0)
    mem_after_3 = bank.memory[0].detach().clone()
    
    # Measure retention
    def avg_cos_sim(a, b):
        a_n = torch.nn.functional.normalize(a, dim=-1)
        b_n = torch.nn.functional.normalize(b, dim=-1)
        return (a_n * b_n).sum(dim=-1).mean().item()
    
    ret_1_to_2 = avg_cos_sim(mem_after_1, mem_after_2)
    ret_1_to_3 = avg_cos_sim(mem_after_1, mem_after_3)
    ret_2_to_3 = avg_cos_sim(mem_after_2, mem_after_3)
    
    print(f"After 3 consecutive segments with random data:")
    print(f"  Cosine sim(seg1→seg2): {ret_1_to_2:.4f}")
    print(f"  Cosine sim(seg1→seg3): {ret_1_to_3:.4f}")
    print(f"  Cosine sim(seg2→seg3): {ret_2_to_3:.4f}")
    print(f"  (random baseline ≈ 0.0)")
    
    # Intra-memory diversity
    m_norm = torch.nn.functional.normalize(mem_after_3, dim=-1)
    sim_matrix = m_norm @ m_norm.T
    N = sim_matrix.shape[0]
    mask = ~torch.eye(N, dtype=torch.bool)
    print(f"\n  Intra-memory diversity after segment 3:")
    print(f"    Avg pairwise cosine sim: {sim_matrix[mask].mean().item():.4f}")
    print(f"    (closer to 0 = more diverse)")

    # === COMPARE: Selective write simulation ===
    print("\n--- SELECTIVE WRITE SIMULATION (top-K only) ---")
    
    bank2 = MemoryBank(num_slots=num_slots, hidden_dim=dim, ema_alpha=ema_alpha, dtype=torch.float32)
    bank2.reset(batch_size=1)
    
    # Write only top-K tokens per step (simulate by writing K non-overlapping tokens)
    K = top_k
    
    h1 = torch.randn(seq_length, dim)
    # Select top-K "important" tokens (by norm, as a simple proxy)
    norms1 = h1.norm(dim=-1)
    topk_idx1 = norms1.topk(K).indices
    topk_h1 = h1[topk_idx1]
    bank2.write(topk_h1, batch_idx=0)
    mem2_after_1 = bank2.memory[0].detach().clone()
    
    h2 = torch.randn(seq_length, dim)
    norms2 = h2.norm(dim=-1)
    topk_idx2 = norms2.topk(K).indices
    topk_h2 = h2[topk_idx2]
    bank2.write(topk_h2, batch_idx=0)
    mem2_after_2 = bank2.memory[0].detach().clone()
    
    h3 = torch.randn(seq_length, dim)
    norms3 = h3.norm(dim=-1)
    topk_idx3 = norms3.topk(K).indices
    topk_h3 = h3[topk_idx3]
    bank2.write(topk_h3, batch_idx=0)
    mem2_after_3 = bank2.memory[0].detach().clone()
    
    sel_ret_1_3 = avg_cos_sim(mem2_after_1, mem2_after_3)
    sel_ret_2_3 = avg_cos_sim(mem2_after_2, mem2_after_3)
    
    print(f"Writing only top-{K} tokens per step:")
    print(f"  Cosine sim(seg1→seg3): {sel_ret_1_3:.4f}")
    print(f"  Cosine sim(seg2→seg3): {sel_ret_2_3:.4f}")
    
    m2_norm = torch.nn.functional.normalize(mem2_after_3, dim=-1)
    sim2 = m2_norm @ m2_norm.T
    mask2 = ~torch.eye(m2_norm.shape[0], dtype=torch.bool)
    print(f"  Intra-memory diversity: avg pairwise cos sim = {sim2[mask2].mean().item():.4f}")
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(f"Circular buffer: writes ALL {T} tokens → each slot overwritten {writes_per_slot:.0f}x")
    print(f"  → {retention_per_step*100:.1f}% old info retained per step → memory ≈ current window")
    print(f"\nSelective write: writes only top-{K} tokens → each slot overwritten {K/num_slots:.2f}x")
    print(f"  → {(1-ema_alpha)**(K/num_slots)*100:.1f}% old info retained per step → memory preserves history")
    print(f"\nFIX: Change MemoryBank.write() from 'write all tokens' to 'write top-K important tokens'")
    print(f"  This is a ONE-LINE change: write(hidden_states[important_indices]) instead of write(hidden_states)")
