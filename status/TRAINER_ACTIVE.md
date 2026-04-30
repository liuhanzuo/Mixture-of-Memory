# TRAINER_ACTIVE.md — Active Training Runs

## 2026-04-30 ~11:06 — fix_y_ablation CONCLUDED: ALL ARMS FAILED fwd=2000 criterion

> ❌ fix_x3_ablation KILLED 2026-04-30 10:31 CST — Arms A/B/C all collapsed. Root cause: VQ-EMA dead slot revival missing.
> ✅ Fix Y researcher + coder COMPLETED 2026-04-30 10:31 — dead slot revival + QUERY_DIAG diagnostics.
> 🚀 fix_y_ablation LAUNCHED 2026-04-30 10:33–10:40 CST — b200-1(Y1), b200-2(Y2), b200-3(Y3).
> ❌ fix_y_ablation_node0 (Y1, b200-1) KILLED 2026-04-30 10:47 — top1_sim<0.005 at fwd=700. No InfoNCE.
> ❌ fix_y_ablation_node2 (Y3, b200-3) KILLED 2026-04-30 10:47 — top1_sim=0.003387 at fwd=400. No InfoNCE.
> ❌ fix_y_ablation_node1 (Y2, b200-2) KILLED 2026-04-30 11:06 — fwd=2000 FAILED: top1_sim=0.002991 (need >0.010). InfoNCE only delayed collapse ~350 fwd. dead_revived=0 throughout (revival never triggered because uniform routing keeps all ema_cluster_count above threshold). ALL 3 NODES NOW IDLE.

## ⚠️ CRITICAL RESEARCH FINDING

**Fix Y (dead slot revival) is INSUFFICIENT for sustained routing selectivity.**

Collapse trajectory of Y2 (best arm, full Fix Y):
| fwd   | top1_sim | pairwise_cos | dead_revived |
|-------|----------|--------------|--------------|
| 50    | 0.644531 | (low)        | —            |
| 1000  | 0.042236 | 0.2461       | 0            |
| 1350  | 0.003250 | —            | 0            |
| 2000  | 0.002991 | —            | 0            |
| 2250  | 0.002884 | 0.5703       | 0            |

**Root cause hypothesis**: Revival never fires because uniform routing gives ALL slots ema_cluster_count above 0.5 threshold. Dead slot revival only helps when slots are literally never selected — not when they're selected uniformly. The primary issue is winner-takes-all collapse where popular keys converge, not truly dead slots. InfoNCE (qa=0.05) only delays convergence by ~350 fwd steps.

**OPEN QUESTION**: What prevents uniform key convergence when query distribution itself is near-uniform (or low-rank)?

## Active Experiments

| Node   | IP             | Experiment              | Status          | Notes |
|--------|----------------|-------------------------|-----------------|-------|
| b200-1 | 28.89.17.143   | (idle)                  | **IDLE**        | Y1 killed fwd=700 |
| b200-2 | 28.89.17.144   | (idle)                  | **IDLE**        | Y2 killed fwd=2250 — FAILED fwd=2000 criterion |
| b200-3 | 28.89.17.85    | (idle)                  | **IDLE**        | Y3 killed fwd=400 |
| b200-4 | 28.89.19.134   | (retired)               | **IDLE**        | 4th NaN incident, not reusing |

## Pending

- ALL NODES IDLE — awaiting researcher Fix Z analysis
- Researcher dispatched 2026-04-30 11:06 to analyze Y2 collapse mechanics

## Red lines

1. TRAINER_ACTIVE.md: Write-only ✓
2. No hyperparameter edits without approval ✓
3. gpu_runs.jsonl append-only ✓
4. 1 active 8-GPU run per node max ✓
5. Main agent does NOT write code ✓ (dispatched to coder subagent)
6. Significant bug → autonomous kill + researcher + coder + restart ✓ (Red Line #7)
