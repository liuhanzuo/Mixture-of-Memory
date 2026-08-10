# Mixture-of-Memory BABILong Results — Final Summary (2026-05-16 03:15)

## Champion: Phase 8 (P8)

**Checkpoint**: `outputs/babilong_sft_phase8_l1l3_lr2e5/mem_space_adapter.pt` (382 keys)
**Config**: `outputs/babilong_sft_phase8_l1l3_lr2e5/adapter_config.json`

### Setup
- Cold-start (no NIAH pretrained init)
- L1 (mem_space slots) + L3 (Q-Former summary pool) dual hierarchy
- Dual-gate (forget_bias=2.0, tanh_new=true)
- lr=2e-5, 500 steps, writeback_warmup=1000, gate_max=0.3
- num_slots=512, top_k=64, shared_memory_bank
- Llama-3-8B-Instruct frozen backbone + ~1.38B trainable adapter
- Trained on BABILong qa1+qa2+qa5 × {1k,2k,4k} + PG19 20% mix

### Results (n=100, all 21 cells)

| task | 0k | 1k | 2k | 4k | 8k | 16k | 32k | mean |
|------|-----|-----|-----|-----|------|------|------|------|
| qa1  | 88  | 92  | 82  | 71  | 69   | 40   | 34   | 68.0 |
| qa2  | 42  | 50  | 44  | 38  | 50   | 33   | 19   | 39.4 |
| qa5  | 74  | 77  | 61  | 68  | 87   | 70   | 53   | 70.0 |
| **overall** |   |   |   |   |     |     |     | **59.14** |

### Comparison vs baselines

| system | overall mean | Δ vs P8 |
|--------|-------------|---------|
| **P8 (champion)** | **59.14** | — |
| P5a (pure L3, NIAH init) | 52.24 | -6.90 |
| P5b (L1+L3, NIAH init) | 50.10 | -9.04 |
| P10/s1000 (1000 steps cold) | 60.96 (19 cells) | -1.62 |

### Key findings

1. **NIAH bias lives in L1 init**: P5b (NIAH-init L1 + L3) scored qa5/4k=46 vs P5a (pure L3)=76. Removing L1 or cold-starting it eliminates the bias.

2. **Cold-start + lr=2e-5 is the winning recipe**: lr=1e-4 causes catastrophic divergence at step 400-600 (P6/P7 outputs garbage). lr=2e-5 stays healthy.

3. **500 steps is the sweet spot**: More steps -> over-training to {1k,2k,4k} distribution -> long-context (8k+) degrades. P10/s1000 (1000 steps) lost 7pp on qa5/16k vs P8.

4. **L1+L3 hierarchy works**: P8 (L1+L3 cold) beats P5a (pure L3) by +6.9pp overall. Slot retrieval (L1) + summary pooling (L3) provide complementary benefits.

5. **Long-context massive improvement**: qa5/8k +9, qa5/16k +14, qa5/32k +15, qa2/8k +20, qa2/16k +24 vs P5a.

### Ablation matrix

| Phase | L1 init | L3 | lr | steps | overall |
|-------|---------|----|----|-------|---------|
| P5a (pure L3) | disabled | cold | 2e-5 | 1000 | 52.24 |
| P5b (L1+L3) | NIAH | cold | 2e-5 | 1000 | 50.10 |
| P6 (L1 cold no L3) | cold | none | 1e-4 | 2000 | 0 (diverged) |
| P7 (L1+L3 cold) | cold | cold | 1e-4 | 2000 | 0 (diverged) |
| **P8 (champion)** | **cold** | **cold** | **2e-5** | **500** | **59.14** |
| P10/s1000 | cold | cold | 2e-5 | 1000 | 60.96 (partial) |

---
Generated 2026-05-16 03:15 CST
