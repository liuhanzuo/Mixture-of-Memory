# TRAINER_ACTIVE.md — Active Training Runs

## 2026-05-08 15:17 — Experiment H (MemLong-style middle-layer) Launched

### Experiment H: Middle-Layer Memory — RUNNING (local b200-1)
- **Architecture**: write memory at L16, read at {18, 22, 26, 30}, other layers vanilla
- **Rationale**: MemLong Table 3 showed full-layer retrieval (our current design) is strictly worse than 4 retrieval layers. Test hypothesis that signal dilution is a root cause of NIAH=0%.
- **Config**: seq_len=4096, num_slots=64, memory_init=strided, lr=5e-6, 5000 steps, niah_mix=0.30
- **Baseline PPL**: vanilla=9.48, memory=9.69 (ratio=1.022 at step 0) — memory layers slightly shift logits, expected
- **GPU**: 8× L20A @ 66 GiB / 97-99% util
- **Commit**: d10b8de
- **Log**: logs/experiment_h_middle_20260508_1512.log
- **First eval**: step 200 (~30 min)

### Node Availability

| Node | Status |
|------|--------|
| b200-1 (local) | **Experiment H: middle-layer memory** |
| b200-2 | Previous: Experiment D (may still be running per TRAINER_ACTIVE staleness — needs SSH check) |
| b200-3 | Previous: Experiment E (may still be running) |
| b200-4 | Previous: Experiment F (may still be running) |
