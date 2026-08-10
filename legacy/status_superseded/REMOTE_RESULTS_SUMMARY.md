# Remote L20A Training Results Summary

**Collected: 2026-04-21 07:23 CST**

All 4 nodes finished training successfully. No errors or crashes.

---

## Node 1: 28.89.17.143 (b200-1) — `sparse_memory_pg19_l0`

| Item | Value |
|---|---|
| Run | `rmt_v10_l0_20260420_110124` |
| Status | ✅ Training complete |
| Epochs | 5 (452 batches/epoch, 282 steps total) |
| Final train loss | ~2.24 |
| Eval @ step 100 | loss=2.5859, ppl=13.28 |
| Eval @ step 200 | loss=2.2871, ppl=9.85 |
| Checkpoint | `checkpoint_step200/` + `final/` (LoRA merged) |
| LoRA merge | ✅ "Merging LoRA adapter into base model..." |

---

## Node 2: 28.89.17.144 (b200-2) — `sparse_memory_pg19_l1`

| Item | Value |
|---|---|
| Run | `rmt_v10_l0l1_20260420_110141` |
| Status | ✅ Training complete |
| Epochs | 5 (452 batches/epoch, 282 steps total) |
| Final train loss | ~2.27 |
| Eval @ step 100 | loss=2.6055, ppl=13.54 |
| Eval @ step 200 | loss=2.2607, ppl=9.59 |
| Checkpoint | `checkpoint_step200/` + `final/` (LoRA merged) |
| LoRA merge | ✅ "Merging LoRA adapter into base model..." |

---

## Node 3: 28.89.17.85 (b200-3) — `sparse_memory_pg19_l2`

| Item | Value |
|---|---|
| Run | `sparse_memory_pg19_l2_20260421_025910` (two runs: l2 and l0l2) |
| Status | ✅ Training complete |
| Epochs | 5 (452 batches/epoch, 282 steps total) |
| Final train loss (l2) | ~2.68 |
| Eval @ step 200 (l2) | loss=3.0254, ppl=20.60 |
| Final train loss (l0l2) | ~2.68 |
| Eval @ step 200 (l0l2) | loss=3.0039, ppl=20.16 |
| Checkpoint | `checkpoint_step100/`, `checkpoint_step200/`, `final/` |
| LoRA merge | ✅ Both runs merged successfully |

---

## Node 4: 28.89.19.134 (b200-4) — `rmt_v10_l0l1l2`

| Item | Value |
|---|---|
| Run | `rmt_v10_l0l1l2_20260420_110216` |
| Status | ✅ Training complete |
| Epochs | 5 (452 batches/epoch, 282 steps total) |
| Final train loss | ~2.34 |
| Eval @ step 100 | loss=2.8066, ppl=16.55 |
| Eval @ step 200 | loss=2.3682, ppl=10.68 |
| Checkpoint | `checkpoint_step200/` + `final/` (LoRA merged) |
| LoRA merge | ✅ "Merging LoRA adapter into base model..." |

---

## Comparison Table

| Node | Task | Eval Step 200 Loss | Eval Step 200 PPL | LoRA Merged |
|---|---|---|---|---|
| b200-1 (l0) | l0 only | **2.2871** | **9.85** | ✅ |
| b200-2 (l1) | l0+l1 | **2.2607** | **9.59** | ✅ |
| b200-3 (l2) | l2 only | 3.0254 | 20.60 | ✅ |
| b200-4 (l0l1l2) | l0+l1+l2 | **2.3682** | **10.68** | ✅ |

## Key Observations

1. **All runs completed successfully** — no crashes, no NaN, no errors.
2. **l0 and l0l1 converge best** — PPL ~9.6-9.9 at eval step 200.
3. **l2 alone is notably worse** — PPL ~20.6, roughly 2x higher loss. The sparse memory at layer 2 is harder to train.
4. **l0l1l2 (b200-4) is between** — PPL 10.68, slightly worse than l0/l0l1 but much better than l2 alone.
5. **b200-3 also ran l0l2** — loss 3.0039 ppl 20.16, similar to l2-only, suggesting l2 is the bottleneck.
6. **All checkpoints saved and LoRA merged** into final models under each run's `final/` directory.

## NIH Eval (b200-4, l0l1l2)

Partial eval results visible in `eval_nih_l0l1l2.log`:
- 2048 tokens: 67-70% recall at some positions
- 4096 tokens: **100% recall** at 30%-90% positions

## Files on Remote Nodes

All final models contain: `model.safetensors`, `rmt_memory.pt`, `config.json`, tokenizer files.
