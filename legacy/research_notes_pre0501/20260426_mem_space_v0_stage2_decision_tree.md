# Memory-Space v0 — Stage-2 decision tree (post Tier-3 cure)

**Written**: 2026-04-26 19:45, awaiting b200-2 held-out verdict (subagent a8717def76272b32c).
**Purpose**: Collapse the post-held-out decision into one turn. Each branch below
is a concrete, parameterized next action; main picks exactly one based on the
held-out PPL returned by Thread A.

---

## Reference anchors

| Quantity | Value | Source |
|---|---|---|
| Bypass-parity ceiling (skip=40000, 8G×200) | **PPL = 16.50** | gpu_runs.jsonl row (2026-04-26 early) |
| Tier-3 full train==eval (skip=0, 8G×200) | **PPL = 1.5751** | gpu_runs.jsonl row 237 (19:12) |
| Tier-3 smoke train==eval (skip=0, 1G×10) | **PPL = 3.01** | gpu_runs.jsonl (19:00) |
| Unit-test bypass parity (cpu/fp32) | **rel_l2 = 0.000000** | test_mem_space_bypass_parity.py |
| Predicted trainable params post freeze | **540–580 M** | Tier-3 researcher note |
| Actual trainable params | **570.43 M** | Tier-3 full run |

The train==eval PPL 1.5751 is **below** the bypass-parity ceiling 16.50, which is
only possible if the model has learned to memorize the eval chunks — exactly
what `--skip_chunks=200` tests for.

---

## Decision tree (branch on held-out PPL P*)

### Branch 1 — PASS. `P* < 10`

**Interpretation**: The Tier-3-cured wrapper has a real generalization edge.
Joint-attn with frozen `hidden_to_slot` is a lexically-grounded sparse
read-from-slots signal, and the output-side gate `α` has picked up a real
signal without overfitting.

**Action (Stage-2a — unfreeze `hidden_to_slot`, re-evaluate)**:
1. Flip `MemorySpaceConfig`: add `hidden_to_slot_frozen=False` code path,
   re-train on 200 chunks (same skip=0 setup to match 1.5751 baseline).
2. If PPL improves (< 1.5751 train==eval), re-run held-out `skip=200`.
3. Additionally, launch Stage-2b in parallel on b200-4:
   - Scale `num_slots` 512 → 1024, `top_k` 64 → 128, same 1G×10 smoke.

### Branch 2 — MODERATE. `10 ≤ P* < 20`

**Interpretation**: Cure removed the phantom-logit perturbation at step 0
(that was the 62.64 step-1 bug), but the 200 trained steps do not generalize
beyond the training slice. Slot contribution pathway is either too narrow or
doesn't carry lexical signal — α has been scaled by the optimizer to near-zero.

**Action (Stage-2c — diagnose α and the slot signal)**:
1. Add an eval-time hook to dump `tanh(α)` per layer on the held-out slice.
2. If `tanh(α)` ≈ 0 everywhere → the optimizer is effectively disabling slots.
   Remedy: unfreeze `hidden_to_slot` (Branch 1 action), OR add a small auxiliary
   loss that penalizes `α → 0`.
3. If `tanh(α)` is non-trivial but PPL still doesn't improve → the slot
   *contents* aren't informative. Remedy: unfreeze `hidden_to_slot` (same fix).

### Branch 3 — FAIL-SOFT. `20 ≤ P* < 100`

**Interpretation**: Overfitting strong. The 1.5751 was essentially a memorized
eval; without overlap, the wrapper is worse than bypass (16.5). The output-side
gate α trained to a value that *helps on the training slice but hurts
elsewhere*.

**Action (Stage-2d — reduce overfit, longer seq, writeback-BPTT)**:
1. Extend `seq_len` 4096 → 8192 to dilute the single-chunk memorization.
2. Activate writeback-BPTT (gate warmup + longer warmup steps).
3. If still no win: declare "Tier-3 cure works structurally (no phantom logits)
   but there's no useful signal to pass through the gate" and pivot to unfreeze
   `hidden_to_slot`.

### Branch 4 — FAIL-HARD. `P* ≥ 100`

**Interpretation**: CLAUDE.md red line "PPL > 100 → /researcher; don't touch
hyperparameters". This would mean the cure is not actually at bypass parity on
held-out data — maybe our eval evaluator has a training-specific side effect
(e.g., `_reset_banks` leaks state across chunks differently under disjoint
slices).

**Action**:
1. Run skip=200 with `--max_train_steps=0` (pure zero-shot eval, no gradient
   whatsoever) on 1G×10 smoke first. If that also > 100, Tier-3 cure does not
   transfer across slices → /researcher Tier-4.
2. Otherwise: training is actively corrupting held-out performance. /researcher
   on overfitting mechanism.

### Branch 5 — NAN / CRASH

**Action**: Rerun once. If deterministic, escalate to /researcher with the full
training-log tail and dump of α evolution across chunks.

---

## Follow-up parallelism (regardless of branch)

Free nodes (b200-1, b200-4, local H20) should not idle while decisions are made.
Queued independent work:

- **b200-1**: Issue #110 phase-transition investigation (kv ≥ 192 rank=1
  PPL 479.26 → 610.87 deterministic-but-unexplained); dispatch /researcher.
- **b200-4**: Q-Filters `filter_rank` sweep at kv=256, recent=192 (the U-shape
  minimum) — cross-check the 147.04 point's stability.
- **local H20**: Bypass-parity regression suite (re-run unit test every
  time the `mem_space` layer is edited).

---

## Ingest template (for when Thread A returns)

Append to `status/gpu_runs.jsonl`:

```json
{"ts":"2026-04-26T<HH:MM>:00+08:00","thread":"A","sweep":"mem_space_v0_tier3_heldout_llama3","node":"b200-2","model":"Llama-3-8B","config":{"num_slots":512,"top_k":64,"max_chunks":200,"seq_len":4096,"skip_chunks":200,"train_steps":0,"lr":1e-3,"dtype":"bfloat16"},"ppl":<P*>,"step1_ppl":<x>,"n_trainable_M":570.43,"n_nonfinite":<n>,"wallclock_s":<t>,"branch":"<1|2|3|4|5>","verdict":"<PASS|MODERATE|FAIL_SOFT|FAIL_HARD|CRASH>","decision":"<one of: unfreeze_hidden_to_slot / scale_N_k / longer_seq / writeback_bptt / researcher_tier4>"}
```

Append to `status/ACTIVE_SWEEPS.jsonl`:

```json
{"ts":"...","sweep":"mem_space_v0_tier3_heldout_llama3","thread":"A","status":"completed","node":"b200-2","ppl":<P*>,"verdict":"<...>","next_dispatch":"<Stage-2 sweep_id or 'researcher'>"}
```

Write-overwrite `status/TRAINER_ACTIVE.md` with updated cluster snapshot and
next-action queue.
