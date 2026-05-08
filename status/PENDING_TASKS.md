# PENDING_TASKS.md — Pending Tasks

**Heartbeat must check this file every inspection. If pending tasks exist, heartbeat cannot just report HEARTBEAT_OK.**

## Active Tasks

- Experiment H (local b200-1): middle-layer memory, write=L16, read={18,22,26,30}, running.
- Experiment H2 (remote b200-3): middle-layer memory, write=L20, read={22,25,28,31}, running.

## Pending Tasks

### [PENDING] sync_heartbeat_tracking_sources — Heartbeat misses experiments not in remote_experiments.json
- priority: medium
- auto_launch: false (needs design discussion)
- description: |
    2026-05-08 15:31 the independent heartbeat reported "b200-3 unknown process"
    even though TRAINER_ACTIVE.md had the full H2 entry at 15:28:37 (commit f4ad2e2).
    Root cause: the heartbeat reads only configs/remote_experiments.json (stale at
    2026-04-30 fix_y_ablation state), not TRAINER_ACTIVE.md. Fixed for this
    experiment via commit d18965f, but the discrepancy is systemic.

    Fix options:
    1. Modify /heartbeat skill to cross-check TRAINER_ACTIVE.md in addition to remote_experiments.json.
    2. Add a pre-experiment-launch hook that updates both files atomically.
    3. Add a periodic sync script that derives remote_experiments.json from TRAINER_ACTIVE.md.

    Until fixed: agents launching remote experiments MUST update BOTH files.

### [PENDING] evaluate_middle_layer_hypothesis — H vs H2 first NIAH eval at step 200
- priority: **high**
- auto_launch: true (heartbeat should inspect at step 200)
- description: |
    Experiment H step 200 eval ~16:05. Experiment H2 ~16:20.
    Decision tree:
    - NIAH > 0% on either arm → middle-layer hypothesis confirmed. Continue to step 1000.
    - NIAH = 0% on both + PPL ratio < 1.1 → add P1 (freeze writer) on top of P0.
    - PPL ratio > 1.2 → read_layers selection wrong. Sweep {14,18,22,26} / {16,20,24,28} / {8,16,24,30}.
    - If H2 (L20) > H (L16) → "deeper middle" beats "strict L/2", consider sweeping L18/L22.

### [PENDING] analyze_slot_forward_finetune — Fine-tune test completing soon
- priority: **high**
- auto_launch: true
- description: |
    Slot-forward fine-tune test (b200-1) reaching step 500/500 ~17:50.
    Current trajectory: ratio 750→3.83→1.33→1.20→1.18
    NIAH still 0/57 throughout.
    After completion: analyze final eval, compare with pretrain approach on b200-2.

### [PENDING] monitor_pretrain_slot_forward — Pretrain experiment early phase
- priority: **high**
- auto_launch: true
- description: |
    Pretrain slot-forward (b200-2) step ~10/2000, Phase 1 (backbone frozen).
    Step-0 PPL=1258 (vs 7115 w/o learnable emb).
    Watch for: PPL convergence speed, Phase 1→2 transition at step 1000.
    First eval at step 200 (~18:30).

### [PENDING] git_push_pat_fix — GitHub PAT missing workflow scope
- priority: medium
- auto_launch: false (needs user action)
- description: |
    git push fails: "refusing to allow a Personal Access Token to create or update workflow ... without `workflow` scope"
    User needs to update PAT at GitHub Settings → Developer Settings → Personal Access Tokens.

### [OBSOLETE] post_contrastive_analysis — V3 experiments killed (NIAH 0/486)
- All 4 V3 arms killed 2026-05-06 ~15:00. Pivoted to slot-forward architecture.

### [OBSOLETE] direction_decision — Resolved: pivoted to slot-forward / joint attention
- User confirmed slot-forward (joint attention) as new direction 2026-05-06 ~15:00.

## Node Availability (2026-05-06 17:35 updated)

| Node | Status | ETA Free |
|------|--------|----------|
| b200-1 (local) | Slot-forward fine-tune (step 400/500) | ~17:50 |
| b200-2 | Pretrain slot-forward (step 10/2000, Phase 1) | ~5-6 hrs |
| b200-3 | **FREE** | — |
| b200-4 | **FREE** | — |
