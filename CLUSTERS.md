# CLUSTERS.md — Compute Resource & Experiment Registry

All training experiments MUST run on remote B200 nodes. Local GPU is for debugging only.

**`configs/remote_experiments.json` is the CANONICAL experiment table.**
Every heartbeat MUST check this file. Every launch MUST update this file.
See [Experiment Tracking Protocol](#experiment-tracking-protocol) below.

---

## Remote Nodes (B200 Cluster)

### node0: 28.89.17.143 (PRIMARY)

| Item | Value |
|------|-------|
| **Host** | `28.89.17.143` |
| **SSH Port** | `36000` |
| **User** | `root` |
| **Password** | File: `configs/password.txt` (shared via CephFS) |
| **GPU** | 8× NVIDIA B200 (displayed as L20A in nvidia-smi), 180GB each |
| **CPU** | 256 cores |
| **RAM** | 2.0 TB |
| **Disk** | 28 TB (overlay, 26TB free) |
| **OS** | TencentOS Server 3.2 |
| **Python** | 3.11.13 |
| **PyTorch** | 2.9.1+cu128 |
| **CUDA** | 12.8 |
| **Transformers** | 5.5.4 |
| **Triton** | 3.5.1 |
| **Flash Attention** | ❌ NOT installed |
| **DeepSpeed** | ❌ NOT installed |
| **DDP** | ✅ torchrun supported |
| **FSDP** | ✅ supported |
| **Project Path** | `/root/Mixture-of-Memory` |

**All 4 nodes share identical hardware and software.** Common specs listed above.

| Node | IP | SSH Port | Role |
|------|----|----------|------|
| node0 | 28.89.17.143 | 36000 | Primary (models/data here) |
| node1 | 28.89.17.144 | 36000 | Experiment |
| node2 | 28.89.17.85 | 36000 | Experiment |
| node3 | 28.89.19.134 | 36000 | Experiment |

### Remote GPU Notes
- GPUs are **B200** hardware, but nvidia-smi reports them as "L20A"
- Treat as B200 for all training configurations (dtype, memory planning, compute capability)
- No flash_attn → use PyTorch native `F.scaled_dot_product_attention` (SDPA) as fallback
- No DeepSpeed → use DDP or FSDP
- No hugepages configured
- Container environment (overlay filesystem)
- **Only node0 has models and pre-tokenized data.** Other nodes must rsync from node0.

### Remote Conventions
- Training logs: `/root/Mixture-of-Memory/logs/<experiment_name>.log`
- Outputs: `/root/Mixture-of-Memory/outputs/<experiment_name>/`
- Data: `/root/Mixture-of-Memory/data/`
- Models: `/root/Mixture-of-Memory/models/`
- Launch via `nohup` + background `&`, capture PID for monitoring

### SSH Access — Cluster 1 (node0-node3)
```bash
SSH_PASS="/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/configs/password.txt"
SSH_OPTS="-o PubkeyAuthentication=no -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -p 36000"
sshpass -f "$SSH_PASS" ssh $SSH_OPTS root@<IP>
```

---

## Cluster 2: node4-node7 (B200) — NEW

### node4: 28.89.18.19 (CLUSTER2 PRIMARY)

| Item | Value |
|------|-------|
| **Host** | `28.89.18.19` |
| **SSH Port** | `36000` |
| **User** | `root` |
| **Password** | File: `configs/password_cluster2.txt` |
| **GPU** | 8× NVIDIA B200 (displayed as L20A), 180GB each |
| **CPU** | 256 cores |
| **RAM** | 2.0 TB |
| **Disk** | 28 TB (27TB free) |
| **OS** | TencentOS Server 3.2 |
| **Python** | 3.11.13 |
| **PyTorch** | 2.8.0+cu128 |
| **CUDA** | 12.8 |
| **Flash Attention** | ❌ NOT installed |
| **DDP** | ✅ torchrun supported |
| **Project Path** | `/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/` |

**All 4 nodes share identical hardware and software.**

| Node | IP | SSH Port | Role |
|------|----|----------|------|
| node4 | 28.89.18.19 | 36000 | Primary (models/data here) |
| node5 | 28.89.17.189 | 36000 | Experiment |
| node6 | 28.89.20.126 | 36000 | Experiment |
| node7 | 28.89.18.40 | 36000 | Experiment |

### Cluster 2 Remote Conventions
- Project path: `/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/`
- Training logs: `/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/logs/<experiment_name>.log`
- Outputs: `/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/outputs/<experiment_name>/`
- Data: `/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/data/`
- Models: `/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/models/`
- Shared CephFS across all 4 nodes

### Cluster 2 Rsync
**Cluster 2 uses different CephFS mount (`wzc1` vs local `zwfy6`), so rsync is REQUIRED.**
```bash
SSH_PASS="/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/configs/password_cluster2.txt"
SSH_OPTS="-o PubkeyAuthentication=no -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -p 36000"
SRC="/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/"
DST="/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/"
# Sync to all cluster2 nodes (CephFS NOT shared between zwfy6 and wzc1):
sshpass -f "$SSH_PASS" rsync -avz --timeout=60 -e "ssh $SSH_OPTS" "$SRC" root@28.89.18.19:"$DST"
sshpass -f "$SSH_PASS" rsync -avz --timeout=60 -e "ssh $SSH_OPTS" "$SRC" root@28.89.17.189:"$DST"
sshpass -f "$SSH_PASS" rsync -avz --timeout=60 -e "ssh $SSH_OPTS" "$SRC" root@28.89.20.126:"$DST"
sshpass -f "$SSH_PASS" rsync -avz --timeout=60 -e "ssh $SSH_OPTS" "$SRC" root@28.89.18.40:"$DST"
```

### SSH Access — Cluster 2 (node4-node7)
```bash
SSH_PASS="/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/configs/password_cluster2.txt"
SSH_OPTS="-o PubkeyAuthentication=no -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -p 36000"
sshpass -f "$SSH_PASS" ssh $SSH_OPTS root@<IP>
```

---

## Local: TENCENT64 (DEBUG ONLY)

| Item | Value |
|------|-------|
| **Host** | TENCENT64.site (this machine) |
| **GPU** | 8× NVIDIA H20, 98GB each |
| **CPU** | 384 cores |
| **RAM** | 2.2 TB |
| **Disk** | CephFS (shared, can be slow for sharded model loading) |
| **Project Path** | `/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/` |

### Local Usage Policy
- **DO** run: smoke tests, single-GPU debugging, quick inference, file editing
- **DO NOT** run: full training, multi-GPU experiments, heavy benchmarking
- CephFS model loading is slow (291 shards × 8 GPU = 40+ min)

---

## Launch Policy

1. **Training experiments** → on remote B200 nodes
2. **Evaluation** (PPL, benchmarks) → on remote B200 nodes
3. **Data preprocessing** (tokenization etc) → on remote B200 nodes
4. **Code editing / debugging** → locally (fast CephFS write, no SSH latency)
5. **Smoke tests** (< 100 steps) → locally acceptable
6. **Model checkpoint loading** → on node0 (local disk, fast)

---

## Experiment Tracking Protocol

### Canonical File
**`configs/remote_experiments.json`** — the single source of truth for ALL experiments on ALL nodes.

### Rules
1. **Before launching ANY experiment**: update `remote_experiments.json` with experiment config and `status: "launching"`
2. **After launch verified** (process alive, GPU active): change status to `"running"`
3. **Every heartbeat**: trainer MUST read this file, SSH to each node, verify actual state matches `status`
4. **State mismatch** (e.g. status=`running` but process dead/GPU idle): trainer updates to `status: "error"`, records last log lines, reports to main
5. **Experiment completed**: update status to `"completed"`, record key results (final loss, PPL, etc.)
6. **Experiment killed**: update status to `"killed"`, record reason and who authorized it
7. **NEVER launch without updating this file first**
8. **NEVER overwrite a running experiment's slot without main approval**

### Required Fields Per Node
```json
{
  "experiment": "human-readable-name",
n  "description": "what hypothesis this tests",
  "script": "scripts/train_xxx.py",
  "command": "full torchrun command string",
  "status": "launching|running|completed|error|killed|idle",
  "launched": "2026-04-21T14:00:00",
  "pid": 12345,
  "output_dir": "outputs/name",
  "log_file": "logs/name.log",
  "base_model": "models/Model-Name",
  "key_config": {"lr": "2e-5", "batch_size": "2", "sliding_window": "256"},
  "result": null,
  "last_verified": "2026-04-21T14:30:00"
}
```

### Heartbeat Check Procedure (for trainer)

On every heartbeat activation, trainer MUST:

1. **Read** `configs/remote_experiments.json`
2. **For each node** with `status="running"` or `status="launching"`:
   a. SSH to the node
   b. Run `nvidia-smi` — check GPU utilization and memory
   c. Run `ps aux | grep train` — check process is alive
   d. Run `tail -5 <log_file>` — check for errors in logs
   e. **If process alive + GPU active → confirm `status: "running"`, update `last_verified`**
   f. **If process dead or GPU idle → update to `status: "error"`, capture last log lines, report to main**
3. **For each node** with `status="error"`:
   - Report details to main for decision (relaunch/kill/ignore)
4. **For each node** with `status="idle"`:
   - Do nothing (main decides what to launch next)
5. **After all checks**: rsync updated `remote_experiments.json` back to local

### Local Sync
- `remote_experiments.json` lives on **node0**: `/root/Mixture-of-Memory/configs/remote_experiments.json`
- Local copy at: `configs/remote_experiments.json` (synced via CephFS)
- Trainer reads from node0, writes to node0
- Main reads from local CephFS copy
