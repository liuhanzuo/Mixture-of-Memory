# TRAINER_ACTIVE.md — Active Training Runs

## 2026-05-08 15:27 — Middle-Layer Memory A/B Test Running

### Experiment H (local b200-1) — RUNNING
- **Architecture**: MemLong-style middle-layer memory
- **Write layer**: 16 (mid of 32)
- **Read layers**: {18, 22, 26, 30}
- **Started**: 15:16 | Current step: ~10 / 5000 (14 s/step)
- **Baseline PPL**: vanilla=9.48, memory=9.69 (ratio=1.022)
- **GPU**: 8× L20A @ 96 GiB / 99% util
- **Log**: logs/experiment_h_middle_20260508_1512.log
- **First eval**: step 200 (~16:05)

### Experiment H2 (remote b200-3) — RUNNING
- **Architecture**: MemLong-style, deeper middle-layer
- **Write layer**: 20 (~L*5/8)
- **Read layers**: {22, 25, 28, 31}
- **Started**: 15:27 | Current step: < step 1 (loading model)
- **Purpose**: A/B test sensitivity of write_layer position. If H2 > H → "more middle" ≠ "deeper high-level semantics".
- **Log**: logs/experiment_h2_deeper_20260508_1527.log
- **First eval**: step 200 (~16:20)

### Still running from before (needs verification):
- b200-2 (.144): Experiment D (cross_chunk_propagation) — may have diverged, PENDING CHECK
- b200-4 (.134): Experiment F residual (42 GiB small, likely stale — PENDING CLEANUP)

### Node Availability

| Node | Status |
|------|--------|
| b200-1 (local) | **Experiment H: L16 middle-layer** |
| b200-2 | Experiment D (running, needs recheck) |
| b200-3 | **Experiment H2: L20 middle-layer (ablation)** |
| b200-4 | Stale/partial — needs cleanup |
