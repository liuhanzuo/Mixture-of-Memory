---
model: glm-5.1
---

---
model: glm-5.1
---

# /heartbeat — 自动巡检 + 工作路由

每次执行严格按本手册操作。时区：**Asia/Shanghai (GMT+8)**。

---

## Step 0：时间戳与陈旧指令检测（每次必做）

1. 用 `date "+%Y-%m-%d %H:%M:%S %Z"` 取当前本地时间（GMT+8）。
2. 回复**第一行**必须写：`## HEARTBEAT [YYYY-MM-DD HH:MM GMT+8]`（精确到分钟）。
3. **陈旧指令跳过规则**：cron 是链式的，如果上一条 `/heartbeat` 执行耗时 > 15 分钟，
   下一条会排在当前时间之后立即触发，此时它实际上是 "被延迟的 15 分钟前那一拍"。
   - 读 `status/TRAINER_ACTIVITY.jsonl` 最后一条 `event=heartbeat` 记录的 timestamp。
   - 如果 `(current_time - last_heartbeat_timestamp) < 10 min`，说明刚刚才跑过一次，
     本拍是陈旧触发 → **跳过本次**，只追加一行 `TRAINER_ACTIVITY.jsonl` 标记
     `{event: heartbeat_skip, reason: stale_chain_trigger, age_sec: N}`，然后结束。
   - 否则正常进入 Step 1。

---

## Step 1：读取上下文（必须第一步）

```
Read: status/AUTONOMOUS_MODE.md         ← 新增：检查是否启用全自动
Read: status/TRAINER_ACTIVE.md
Read: HEARTBEAT.md
Read: status/TRAINER_REQUESTS.jsonl（全部）
Read: status/TRAINER_APPROVALS.jsonl（全部）
Read last 5 lines: status/TRAINER_ACTIVITY.jsonl
Read last 10 lines: status/AUTO_CHAIN.jsonl  ← 新增：自动链状态
Read: configs/remote_experiments.json
```

对比 REQUESTS 和 APPROVALS，找出所有 request_id 尚无 APPROVALS 匹配的条目 → 待审批列表。

**AUTONOMOUS_MODE=ENABLED 时**（见该文件白名单）：
- 匹配白名单的待审批请求 → 本次 heartbeat 内 **直接自动 /approve**（追加 APPROVALS.jsonl + UPDATELOG.md），然后按 Step 6 自动链调度下一环节
- 非白名单 → 仍列给用户

---

## Step 2：本地 GPU 状态检查

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
```

```bash
nvidia-smi --query-compute-apps=pid,gpu_index,used_memory,process_name \
  --format=csv,noheader
```

**对每个 GPU 占用 PID：**
```bash
ps -fp <pid>
ps -o pid,ppid,etimes,stat,cmd -p <pid>
readlink /proc/<pid>/cwd 2>/dev/null
```

与 `status/gpu_runs.jsonl` 对比，**给每个进程分类**：
- `expected` — 与注册表中 running 条目匹配
- `small_debug` — 单卡 debug / eval，已知且不影响
- `unknown` — 无法确认来源，需调查
- `stale_orphan` — 父进程已消失 + 注册表显示该运行已结束/放弃

---

## Step 3：活跃训练日志检查

如果 TRAINER_ACTIVE.md 有 status=running 的训练：

```bash
tail -20 <log_path>
```

判断健康状态：
- ✅ 健康：loss 在下降，无 NaN，step 在递增
- ⚠️ 警告：loss 停滞超过 50 steps，或 grad_norm 骤变
- ❌ 崩溃：Python traceback、CUDA error、NCCL error、进程已退出

---

## Step 4：远程集群检查

遍历 `configs/remote_experiments.json`，**对 status=running 的每个节点**：

```bash
sshpass -f configs/password.txt ssh \
  -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
  root@<IP> \
  "nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader 2>/dev/null; \
   tail -5 <log_path> 2>/dev/null || echo 'LOG_NOT_FOUND'"
```

**SSH 失败**：最多重试 2 次，失败记为 `ssh_timeout`，不立即升级。

**判断每个节点状态**：
- `running_healthy` — 进程存在，log 有新内容，loss 正常
- `running_stalled` — 进程存在，log 无更新 or loss 停滞
- `completed` — 进程已退出，log 显示正常结束（loss 收敛）
- `crashed` — 进程已退出，log 有 error / traceback
- `ssh_timeout` — 连接失败

节点状态变化时（running → completed/crashed），**立即用 Write 更新 `configs/remote_experiments.json`**。

---

## Step 5：待审批请求处理

如有未审批请求，在回复中**逐条列出**：

```
### 待审批：<request_id> [urgency: high]
- 实验：<current_run>
- 问题：<issue_type>
- 证据：<evidence 摘要>
- 建议方案：<proposed_action>
→ 需要你运行 /approve <request_id>
```

---

## Step 6：问题判断与自主路由

### 一切正常 → HEARTBEAT_OK

条件：
- 所有 GPU 进程 = expected 或 small_debug
- 无待审批请求
- 活跃训练健康（或无训练）
- 无 running 远程节点出现异常

回复末尾写 `HEARTBEAT_OK`，附简短状态表格。

---

### 小问题 → 自主修复（无需用户确认）

| 情况 | 动作 |
|------|------|
| TRAINER_ACTIVE.md 内容过时但进程健康 | Write 覆盖更新 |
| 远程节点 running→completed/crashed，json 未更新 | Write 更新 remote_experiments.json |
| gpu_runs.jsonl 条目为 launching，进程已健康运行 | 追加 updated 条目（原条目不动） |
| 单卡 debug 进程占 1 个 GPU | 记录，不干涉 |

---

### 中等问题 → 调查 + 提醒用户 + 建议派发 subagent

| 情况 | 行动 |
|------|------|
| unknown GPU 进程 | 完整调查（ps tree, cwd, cmdline），写 ISSUES.jsonl，回复中描述 |
| 远程节点 training stalled >50 steps | SSH 深度检查，写 ISSUES.jsonl，**建议 `/trainer 检查 <node>`** |
| 实验刚 completed，PPL 未记录 | **建议 `/trainer 评估 <exp_name>`** |
| GPU 全部空闲 + 有批准的实验待运行 | **建议 `/trainer 启动 <approved_exp>`** |

---

### 大问题 → 升级，不自主处理

列出完整证据，等用户决策：
- 活跃训练 crash（traceback / OOM / NCCL error）
- GPU 内存泄漏（进程已退出，内存仍占用）
- stale/orphan 进程需要 kill → 列证据，让用户 confirm
- 远程所有节点实验全部失败
- 发现代码 bug 导致训练结果无效

---

### 研究进展触发 → 建议 /researcher

当发现以下情况时，**建议运行 `/researcher`**：
- 实验结果出来了（新 PPL），需要分析和下一步建议
- 某个方向连续失败（>= 3次），需要重新调研
- `status/ISSUES.jsonl` 有 `type=research_decision` 的 open issue
- 距上次 researcher 报告 > 48 小时且 GPU 空闲

---

### 自动链调度（AUTONOMOUS_MODE=ENABLED）

每次 heartbeat，读取 `status/AUTO_CHAIN.jsonl` 最后一条关于每个活跃 request_id 的记录：

| 上一阶段 `stage` | 本次 heartbeat 自动执行 |
|------------------|------------------------|
| `approved` (new_experiment) | 通过 Agent 工具异步派发 `/researcher` 任务；完成后回写 `stage=researcher_done` |
| `researcher_done` | 派发 `/coder` 实现模块+脚本，必须 smoke pass；完成后 `stage=coder_done` |
| `coder_done` | 派发 `/trainer smoke`（单卡 10 step/10 chunk）；通过后 `stage=trainer_smoke_done` |
| `trainer_smoke_done` | 派发 `/trainer full`（目标节点全量）；结果出后 `stage=trainer_full_done` |
| `trainer_full_done` | 派发 `/researcher` 做结果分析；完成后 `stage=chain_complete` |
| `chain_complete` | 不再处理，留作归档 |

**异步原则**：
- 每个 subagent 用 `run_in_background=true` 派发，避免 heartbeat 超时
- subagent 完成时，它负责在 `AUTO_CHAIN.jsonl` 追加自己的完成记录
- 下一次 heartbeat 看到新 stage，接力下一步

**失败处理**：
- subagent 报错（stage=<stage>_failed）→ 追加 ISSUES.jsonl，停止该链，不自动重试
- 连续 2 次 heartbeat 无进展（same stage）→ 升级给用户

---

### 自动链调度（AUTONOMOUS_MODE=ENABLED）

每次 heartbeat，读取 `status/AUTO_CHAIN.jsonl` 最后一条关于每个活跃 request_id 的记录：

| 上一阶段 `stage` | 本次 heartbeat 自动执行 |
|------------------|------------------------|
| `approved` (new_experiment) | 通过 Agent 工具异步派发 `/researcher` 任务；完成后回写 `stage=researcher_done` |
| `researcher_done` | 派发 `/coder` 实现模块+脚本，必须 smoke pass；完成后 `stage=coder_done` |
| `coder_done` | 派发 `/trainer smoke`（单卡 10 step/10 chunk）；通过后 `stage=trainer_smoke_done` |
| `trainer_smoke_done` | 派发 `/trainer full`（目标节点全量）；结果出后 `stage=trainer_full_done` |
| `trainer_full_done` | 派发 `/researcher` 做结果分析；完成后 `stage=chain_complete` |
| `chain_complete` | 不再处理，留作归档 |

**异步原则**：
- 每个 subagent 用 `run_in_background=true` 派发，避免 heartbeat 超时
- subagent 完成时，它负责在 `AUTO_CHAIN.jsonl` 追加自己的完成记录
- 下一次 heartbeat 看到新 stage，接力下一步

**失败处理**：
- subagent 报错（stage=<stage>_failed）→ 追加 ISSUES.jsonl，停止该链，不自动重试
- 连续 2 次 heartbeat 无进展（same stage）→ 升级给用户

---

## Step 7：写入活动日志（每次必须）

追加一条到 `status/TRAINER_ACTIVITY.jsonl`：

```json
{
  "timestamp": "ISO8601 Asia/Shanghai",
  "event": "heartbeat",
  "trigger": "cron|manual",
  "gpu_state": "idle|training|mixed|leaked|unknown",
  "local_pids": ["<pid>:<classification>", ...],
  "remote_status": {"node0": "running_healthy", "node1": "completed"},
  "action_taken": "none|registry_update|issue_created|escalate",
  "pending_requests": 0,
  "issues_found": ["描述"],
  "conclusion": "OK|WARNING|CRITICAL",
  "note": "简短说明"
}
```

---

## Step 8：输出格式

```
## HEARTBEAT [YYYY-MM-DD HH:MM GMT+8]

### 本地 GPU
| GPU | 显存占用 | 利用率 | 进程 | 分类 |
|-----|---------|--------|------|------|
| 0   | 38 GB / 97.8 GB | 85% | PID 3842460 (torchrun) | expected |
| ... |

### 远程集群
| 节点 | IP | 实验 | 状态 | 最新 log |
|------|----|------|------|---------|
| node0 | 28.89.17.143 | llama_baseline | completed | — |

### 活跃训练
- <实验名>: step X/Y, loss Z → 健康/警告/崩溃
  （无活跃训练：无）

### 待审批请求
- <request_id>: <摘要> → /approve <id>
  （无：无）

### 发现的问题
- [按 severity 列出，无则省略本节]

### 建议行动
- [具体建议，如 /trainer 评估 dms_8x，或 /researcher 分析结果]

### 结论
HEARTBEAT_OK
或
⚠️ WARNING: [描述]
或
❌ CRITICAL: [描述，需要立即处理]
```

---

## Red Lines（永不违反，AUTONOMOUS_MODE 也不例外）

- ❌ 不能 kill GPU 进程，除非用户明确授权
- ❌ 不能修改训练脚本的 hyperparameters（只能按已 approved 的 request 照原样执行）
- ❌ 不能假设 unknown = stale（先调查）
- ❌ TRAINER_ACTIVE.md 只能 Write 覆盖，绝不能 Edit
- ❌ gpu_runs.jsonl 只追加，不修改历史记录
- ✅ **可以跨节点并行跑多个 8-GPU serious 实验**（2026-04-26 用户确认：多节点文件系统共享但 GPU 独立，不构成资源冲突）。启动前置条件：`nvidia-smi` 确认目标节点无 running 实验，或运行中的是僵尸 / 无主进程 / 可归属为 stale。单节点内仍不能叠两个 8-GPU。
- ❌ 不能绕过 smoke test 启动 full training

**AUTONOMOUS_MODE 下**可以自主做的（已显式授权）：
- ✅ 自动批准白名单内的 request（见 AUTONOMOUS_MODE.md）
- ✅ 自动启动 approved 实验（单实验；8-GPU 或 smoke）
- ✅ 自动派发 /researcher /coder /trainer subagent
- ✅ 自动更新 remote_experiments.json 的元数据
