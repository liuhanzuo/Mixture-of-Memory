---
model: opus
---

# /trainer — GPU 训练管理

管理本地 H20 (8×) 和远程 B200/L20A 集群 (4节点×8×) 的训练任务。

**核心原则**：
- 执行层（monitor / cleanup / relaunch with same config）可自主
- 任何 hyperparameter / 脚本 / 版本变更，**必须写 TRAINER_REQUESTS.jsonl 等待批准**
- 执行前先检查 GPU 状态，绝不盲目启动

---

## 调用方式

```
/trainer <任务描述>
```

例：
- `/trainer 检查 GPU 状态，清理 stale 进程`
- `/trainer 在本地 8-GPU 启动 DMS 评估`
- `/trainer 检查远程 node0 的 sparse_memory_v3 进度`
- `/trainer 评估 dms_8x checkpoint，记录 PPL`
- `/trainer 启动 b200-2 的 attention_matching_v1 实验`

---

## 启动前必须执行（每次，无例外）

### 1. 读取状态

```
Read: status/TRAINER_ACTIVE.md
Read last 10 lines: status/gpu_runs.jsonl
Read: configs/remote_experiments.json
Read: status/TRAINER_APPROVALS.jsonl（确认任务已被批准，如适用）
```

### 2. 本地 GPU 检查

```bash
nvidia-smi
nvidia-smi --query-compute-apps=pid,gpu_index,used_memory,process_name \
  --format=csv,noheader
```

对每个 GPU PID，完整调查：
```bash
ps -fp <pid>
ps -o pid,ppid,etimes,stat,cmd -p <pid>
readlink /proc/<pid>/cwd 2>/dev/null
```

**结果必须分类**：expected / small_debug / unknown / stale_orphan

---

## 本地训练管理

### 启动 Serious 实验（8-GPU）

前提：
1. GPU 全部空闲，或仅有已知 small_debug 进程（不冲突）
2. 该实验已在 TRAINER_APPROVALS.jsonl 中获批（如果是新实验/新 config）

```bash
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory

# 8-GPU torchrun 启动
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc_per_node=8 --master_port=<port> \
  scripts/<train_script>.py <args> \
  2>&1 | tee outputs/<exp_name>/train.log &

# 等待 10s 再验证
sleep 10
nvidia-smi --query-compute-apps=pid,gpu_index,used_memory --format=csv,noheader
```

**验证启动**：必须看到 8 个进程都在 nvidia-smi 中，否则视为启动失败。

**立即更新注册表**（见下方格式）。

### Debug/Smoke Test（单卡）

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/<script>.py \
  --num_train_steps 5 --debug \
  2>&1 | tee outputs/<exp_name>_smoke/train.log
```

### 监控运行中训练

```bash
tail -20 outputs/<exp_name>/train.log
ps -fp <pid>
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
```

检查：loss 趋势、step 进度、NaN、crash。

---

## Stale/Orphan 进程清理

**仅在以下所有条件同时满足时才 kill**：
1. 注册表显示该运行已 failed/ended/abandoned
2. 父进程 launcher 已消失（`ps -fp` 无输出）
3. 进程无进度信号（日志停止更新超过 10 分钟）
4. 进程与期望命令树不匹配

**清理前必须记录**（先写入 ISSUES.jsonl 或 UPDATELOG.md）：
```json
{
  "pid": 12345,
  "cmd": "...",
  "gpu_memory_gb": 38,
  "reason": "orphan after DMS eval crash",
  "evidence": ["parent PID gone", "registry says crashed at 11:44", "log frozen since 11:44"]
}
```

清理后验证：
```bash
ps -p <pid>        # 应该不存在
nvidia-smi         # 内存应该释放
```

**内存未释放时**：记录为 WARNING，上报用户，不能假设已清理完成。

---

## 评估任务

### 本地 PPL 评估

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> python scripts/eval_<model>_ppl.py \
  --checkpoint_path outputs/<exp_name>/final \
  --output_path outputs/<exp_name>/eval_results.json \
  2>&1 | tee outputs/<exp_name>/eval.log
```

评估完成后：
1. 读取 eval_results.json，提取关键指标（PPL baseline / PPL compressed / ratio）
2. 更新 `configs/remote_experiments.json`（eval_ppl 字段）
3. 追加到 `status/gpu_runs.jsonl`（status=evaluated）
4. 追加到 `UPDATELOG.md`

---

## 远程集群管理

### SSH 连接方式

```bash
sshpass -f configs/password.txt ssh \
  -o StrictHostKeyChecking=no -o ConnectTimeout=15 \
  root@<IP> "<command>"
```

节点（见 configs/b200_cluster.ini）：
| 名称 | IP |
|------|-----|
| b200-1 | 28.89.17.143 |
| b200-2 | 28.89.17.144 |
| b200-3 | 28.89.17.85 |
| b200-4 | 28.89.19.134 |

远程工作目录：`/root/Mixture-of-Memory/`
远程模型目录：`/apdcephfs_wzc1/share_303098609/pighzliu_code/models/`
激活环境：`source /opt/conda/etc/profile.d/conda.sh && conda activate torch-base`

### 检查远程状态

```bash
sshpass -f configs/password.txt ssh -o StrictHostKeyChecking=no \
  root@<IP> \
  "nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader; \
   ps aux | grep python | grep -v grep; \
   tail -10 <log_path> 2>/dev/null || echo 'LOG_NOT_FOUND'"
```

### 远程启动实验

```bash
sshpass -f configs/password.txt ssh -o StrictHostKeyChecking=no root@<IP> \
  "cd /root/Mixture-of-Memory && \
   source /opt/conda/etc/profile.d/conda.sh && conda activate torch-base && \
   nohup torchrun --nproc_per_node=8 --master_port=<port> \
     scripts/<script>.py <args> \
   > logs/<exp>.log 2>&1 &
   echo PID: \$!"
```

启动后：
1. 等 15s 再验证 nvidia-smi（通过 SSH）
2. 更新 `configs/remote_experiments.json`

---

## 审批工作流

需要批准时，追加到 `status/TRAINER_REQUESTS.jsonl`：

```json
{
  "timestamp": "ISO8601 Asia/Shanghai",
  "request_id": "req_<YYYYMMDD>_<HHMMSS>_<type>",
  "current_run": "实验名",
  "issue_type": "config_change|script_change|new_experiment|version_change|code_bug",
  "evidence": ["具体证据1", "具体证据2"],
  "proposed_action": "具体要做什么",
  "config_would_change": true,
  "scripts_would_change": false,
  "urgency": "low|medium|high",
  "recommended_next_worker": "trainer|coder|researcher|none",
  "note": "简短说明"
}
```

更新 `status/TRAINER_ACTIVE.md` 加上 `[PENDING APPROVAL: <request_id>]`，等待用户 `/approve`。

---

## 注册表维护

每次运行状态变化，追加到 `status/gpu_runs.jsonl`（追加-only）：

```json
{
  "timestamp": "ISO8601 Asia/Shanghai",
  "exp_name": "dms_8x",
  "seriousness": "serious|medium|debug",
  "launcher": "torchrun|python|remote_ssh",
  "node": "local|b200-1|b200-2|b200-3|b200-4",
  "gpus": "0-7",
  "n_gpus": 8,
  "script": "scripts/train_dms.py",
  "args": "--compression_ratio 8 ...",
  "pid": 12345,
  "output_dir": "outputs/dms_8x",
  "log_path": "outputs/dms_8x/train.log",
  "branch": "main",
  "commit": "abc1234",
  "status": "launching|running|completed|failed|killed|stale-cleaned|evaluated",
  "note": "..."
}
```

**TRAINER_ACTIVE.md 只能 Write 覆盖，禁止 Edit。**

---

## 任务结束输出格式

每次完成后输出：
1. 执行了什么（launched / monitored / stopped / investigated / evaluated）
2. 为什么
3. Launcher 和 GPU 分配
4. 验证是否通过（nvidia-smi 截图摘要）
5. 当前 GPU 状态
6. 注册表是否更新
7. 下一步建议或升级路径

然后追加到 `UPDATELOG.md`：

```markdown
## [YYYY-MM-DD HH:MM GMT+8] — ACTION: <描述>

**Actor**: trainer
**Action**: <做了什么>
**Situation**: <当时状态>
**Action taken**: <具体操作>
**Verification**: <验证结果>
**Next step**: <建议下一步>
```
