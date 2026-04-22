# Active Training Run

## NO LOCAL TRAINING ACTIVE

**Last Update**: 2026-04-22 12:47

### Current State
- **Local GPUs**: IDLE (0% utilization, 0 MiB used)
- **Local training processes**: NONE
- **Last training killed**: 2026-04-22 11:53 (sparse_memory_concat_fusion_v1_fixed at step ~1150/5000)

### Previous Run (Killed)
- **Experiment**: sparse_memory_concat_fusion_v1_fixed
- **Reason**: Killed due to conflicting research recommendations
  - RESEARCH_REPORT.md (07:11): STOP sparse memory, pivot to Selective Context
  - Researcher brief (09:10): Continue with proper scaling
- **Final status**: Step ~1150/5000, ~23% complete
- **Issue**: Shared tensors bug was fixed and validated, but run was stopped due to direction uncertainty

### Remote Cluster Status
See `configs/remote_experiments.json` for full status.
- Cluster 1 (nodes 0-3): 3 completed, 1 error (SSH issue)
- Cluster 2 (nodes 4-7): Idle, pending experiments (trainer currently checking status)

### Pending Decision
- **Conflict**: Pivot to Selective Context/CCM/LM2 vs continue sparse memory with proper scaling
- **Awaiting**: Researcher assessment to resolve direction uncertainty

### Status Summary
- ❌ No local training running
- ⏳ Trainer subagent checking remote cluster status
- ⏳ Researcher subagent assessing research direction
- ❓ Research direction pending resolution
