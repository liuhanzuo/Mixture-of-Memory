# /train — 标准化实验启动

启动训练实验，自动记录日志、分配节点、写入跟踪文件。

## 调用方式

```
/train <experiment_name> --node <node> --convergence '<json>' -- <training_args>
/train <experiment_name> --matrix '<json_matrix>' --nodes <node_list> -- <shared_training_args>
```

### 示例

单实验：
```
/train infini_attn_v4 --node b200-1 --convergence '{"max_steps":5000,"kill_if_ppl_above":100}' -- --use_infini_attention --infini_beta_init -1.0 --lr 1e-3 --max_steps 5000
```

批量消融（自动分配到空闲节点）：
```
/train beta_init_sweep --matrix '{"infini_beta_init": [-2.0, -1.0, 0.0]}' --nodes b200-2,b200-3,b200-4 -- --use_infini_attention --infini_beta_lr 0.1 --lr 1e-3 --max_steps 5000
```

---

## 执行步骤

### Step 0: 读取当前状态（必须）

```
Read: status/RUNNING_EXPERIMENTS.json（不存在则初始化空）
Read: status/PENDING_TASKS.md
```

### Step 1: 验证节点可用性

已知节点列表：
| 节点 | IP | GPU |
|------|-----|-----|
| b200-1 | 28.89.17.143 | 8×L20A |
| b200-2 | 28.89.17.144 | 8×L20A |
| b200-3 | 28.89.17.85 | 8×L20A |
| b200-4 | 28.89.19.134 | 8×L20A |

- 如果指定了 `--node`：检查该节点不在 RUNNING_EXPERIMENTS.json 的 running 实验中，然后 SSH 验证 GPU 空闲
- 如果未指定 `--node`：从 b200-1 到 b200-4 中找第一个空闲节点
- 如果是 `--matrix` 模式：为每个矩阵值分配一个空闲节点（需要 N 个空闲节点，N=矩阵值数量）
- SSH 检查命令：`sshpass -f configs/password.txt ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@<IP> "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1"`

### Step 2: 生成实验 ID 和路径

```bash
TIMESTAMP=$(date -u "+%Y%m%d_%H%M")
```

- **单实验**: experiment_id = `<experiment_name>_<TIMESTAMP>`
- **消融**: experiment_id = `<experiment_name>_<arm_value>_<TIMESTAMP>`（每个 arm 一个 ID）

路径规则：
- **Log**: `logs/<experiment_id>.log`
- **Output**: `outputs/<experiment_name>/`（同组实验共享 output 目录前缀，消融用 `outputs/<experiment_name>_<arm>/`）
- **Port**: b200-1=29501, b200-2=29502, b200-3=29503, b200-4=29504

### Step 3: 构建训练命令

#### 本地节点 (b200-1)

```bash
nohup torchrun --nproc_per_node=8 --master_port=<PORT> \
  scripts/train_mem_space_pg19.py \
    --model /apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b \
    --data /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/data/pg19_chunks_llama3.npy \
    --output_dir outputs/<experiment_name> \
    <training_args> \
  > logs/<experiment_id>.log 2>&1 &
```

#### 远程节点 (b200-2/3/4)

```bash
sshpass -f configs/password.txt ssh -o StrictHostKeyChecking=no root@<IP> \
  "cd /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory && \
   source /opt/conda/etc/profile.d/conda.sh && conda activate torch-base && \
   nohup torchrun --nproc_per_node=8 --master_port=<PORT> \
     scripts/train_mem_space_pg19.py \
       --model /apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b \
       --data /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/data/pg19_chunks_llama3.npy \
       --output_dir outputs/<experiment_name> \
       --max_chunks 500 --skip_chunks 0 --seq_len 4096 --batch_size 1 \
       --num_slots 128 --top_k 64 \
       <training_args> \
   > logs/<experiment_id>.log 2>&1 & \
   echo PID:\$!"
```

### Step 4: 验证启动

等待 30 秒，然后：
- SSH 到节点执行 `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader`
- 检查 log 文件是否存在且有内容（`wc -l logs/<experiment_id>.log`）
- 如果验证失败：在 RUNNING_EXPERIMENTS.json 中标记 `launch_failed`

### Step 5: 写入跟踪记录

#### 5a. 更新 RUNNING_EXPERIMENTS.json（Read→modify→Write）

```json
{
  "_last_updated": "<now ISO8601>",
  "experiments": {
    "<experiment_id>": {
      "experiment_id": "<experiment_id>",
      "experiment_name": "<experiment_name>",
      "status": "running",
      "node": "b200-N",
      "node_ip": "x.x.x.x",
      "n_gpus": 8,
      "port": 2950N,
      "log_path": "logs/<experiment_id>.log",
      "output_dir": "outputs/<experiment_name>/",
      "launched_at": "<now ISO8601>",
      "launched_by": "train_command",
      "config": {
        "use_infini_attention": true,
        "infini_beta_init": -1.0,
        "lr": 1e-3,
        "max_steps": 5000,
        "batch_size": 1,
        "seq_len": 4096
      },
      "convergence_criteria": {
        "max_steps": 5000,
        "kill_if_ppl_above": 100,
        "kill_if_stalled_for_steps": 200
      },
      "progress": {
        "last_checked_at": "<now ISO8601>",
        "current_step": 0,
        "max_steps": 5000,
        "latest_ppl": null,
        "latest_loss": null,
        "health": "launching",
        "diagnostics": {}
      },
      "ablation_group": "<group_name> or null",
      "related_experiments": [],
      "on_complete": [
        "analyze_results",
        "update_PENDING_TASKS"
      ]
    }
  }
}
```

#### 5b. 追加 gpu_runs.jsonl

```json
{
  "timestamp": "<now ISO8601>",
  "experiment_id": "<experiment_id>",
  "experiment_name": "<experiment_name>",
  "node": "b200-N",
  "ip": "x.x.x.x",
  "n_gpus": 8,
  "status": "running",
  "log": "logs/<experiment_id>.log",
  "output": "outputs/<experiment_name>/",
  "config": { ... }
}
```

#### 5c. 覆写 TRAINER_ACTIVE.md

```markdown
# TRAINER_ACTIVE.md — Active Training Runs

## <timestamp> — <experiment_name> (RUNNING)
| Node | GPUs | Config | Status |
|------|------|--------|--------|
| b200-N | 8×L20A | <key config> | **RUNNING** |

Log: logs/<experiment_id>.log
Output: outputs/<experiment_name>/

### Idle Nodes
- (列出不在 RUNNING_EXPERIMENTS.json 中的节点)
```

#### 5d. 追加 TRAINER_ACTIVITY.jsonl

```json
{
  "timestamp": "<now ISO8601>",
  "event": "train_launch",
  "experiment_id": "<experiment_id>",
  "node": "b200-N",
  "conclusion": "OK",
  "note": "Launched <experiment_name> on b200-N"
}
```

#### 5e. 更新 PENDING_TASKS.md

如果此实验对应 PENDING_TASKS.md 中的一个 `[PENDING]` 任务，将其改为 `[RUNNING]` 并填写 experiment_id 和 node。

### Step 6: 输出启动报告

```
## /train 启动报告

| 项目 | 值 |
|------|-----|
| Experiment ID | <experiment_id> |
| Node | b200-N |
| Log | logs/<experiment_id>.log |
| Output | outputs/<experiment_name>/ |
| Config | <key params> |
| Convergence | max_steps=5000, kill_if_ppl_above=100 |

状态: **RUNNING** (已验证 GPU 加载)
```

---

## 消融矩阵模式

当使用 `--matrix` 时：

1. 解析 JSON 矩阵，例如 `{"infini_beta_init": [-2.0, -1.0, 0.0]}`
2. 为每个值生成一个实验：
   - `beta_init_sweep_n2.0_20260501_1030`
   - `beta_init_sweep_n1.0_20260501_1030`
   - `beta_init_sweep_0.0_20260501_1030`
3. 分配到 `--nodes` 指定的节点（或自动选择空闲节点）
4. 所有实验共享 `ablation_group: "beta_init_sweep_20260501"`
5. `related_experiments` 列出同组其他实验 ID
6. 并行启动所有实验

---

## 收敛标准 (convergence_criteria)

| 字段 | 默认值 | 说明 |
|------|--------|------|
| max_steps | 从训练参数推断 | 训练最大步数 |
| kill_if_ppl_above | 100 | PPL 超过此值立即 kill |
| kill_if_stalled_for_steps | 200 | PPL 连续 N 步无改善则 kill |
| expected_final_ppl_range | null | 完成后期望的 PPL 范围 |
| intermediate_checkpoints | [1000, 2000] | 在这些步数检查是否达标 |
| checkpoint_criteria | null | 每个检查点的通过条件 |

未指定时使用默认值。heartbeat 在每次巡检时检查这些标准。
