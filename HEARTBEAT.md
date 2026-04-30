# HEARTBEAT.md — 自动监控操作手册

每 20 分钟触发一次。目标：监控实验状态，发现问题，路由工作，**不自主决策重大变更**。

---

## 每次 heartbeat 必须执行的检查

### Step 1: GPU 状态检查（本地）

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,compute_mode \
  --format=csv,noheader
nvidia-smi --query-compute-apps=pid,gpu_index,used_memory,process_name \
  --format=csv,noheader
```

对每个 GPU 占用 PID，检查：
- `ps -fp <pid>` — 进程命令
- `readlink /proc/<pid>/cwd` — 工作目录
- 是否出现在 `status/gpu_runs.jsonl`
- 是否匹配 `status/TRAINER_ACTIVE.md`

分类：`expected` / `small_debug` / `unknown` / `stale_orphan`

### Step 2: 远程集群检查

遍历 `configs/remote_experiments.json` 中 status=running 的节点：

```bash
sshpass -f configs/password.txt ssh -o StrictHostKeyChecking=no \
  -o ConnectTimeout=10 root@<IP> \
  "nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader; \
   tail -5 <log_path> 2>/dev/null"
```

检查是否还在运行，loss 是否正常。

### Step 3: 待审批请求检查

读取 `status/TRAINER_REQUESTS.jsonl`，找出所有没有对应 APPROVALS 条目的请求（匹配 request_id）。

如果有待审批请求：
- 在回复中明确列出，要求用户确认
- **不要自主批准或拒绝**

### Step 4: 活跃训练监控

如果 `status/TRAINER_ACTIVE.md` 显示有活跃训练：
- 检查进程是否还在
- 读取最新日志（最后 10 行）
- 判断是否健康：loss 下降、无 NaN、无 crash

### Step 5: ISSUES.jsonl 检查

读取 `status/ISSUES.jsonl`，找 status=open 的问题，判断是否需要立即处理。

---

## 判断和行动规则

### 一切正常 → HEARTBEAT_OK

如果：
- 所有 GPU 占用都是 expected
- 无待审批请求
- 活跃训练健康
- 无 open issues

回复：`HEARTBEAT_OK`，可附简短状态摘要。

### 小问题 → 自主修复

可以自主处理：
- 注册表过时（registry stale），但进程明显健康
- TRAINER_ACTIVE.md 需要刷新（用 write，不用 edit）
- 小型 debug 进程占用 1 个 GPU，不影响计划工作

### 中等问题 → 调查 + 提醒用户

- 未知进程，但不明显是 stale
- 远程节点 SSH 超时（先尝试 3 次）
- 训练进展异常（loss 停滞超过 50 steps）

操作：调查，写入 ISSUES.jsonl，在回复中明确描述。

### 大问题 → 升级给用户

必须升级（不自主决策）：
- 活跃训练 crash
- stale/orphan 进程需要 kill（记录完整证据再提出）
- GPU 内存泄漏
- 远程节点所有实验都失败

---

## TRAINER_ACTIVE.md 更新规则

**永远只用 write，禁止 edit**（曾导致 gateway 崩溃）。

流程：
1. Read 当前内容
2. 在内存中修改需要更新的字段
3. Write 整个文件

---

## 活动日志规则

当 heartbeat 发现有意义的状态变化或采取了动作，追加一条 JSON 到 `status/TRAINER_ACTIVITY.jsonl`：

```json
{
  "timestamp": "ISO8601",
  "trigger": "heartbeat",
  "gpu_state": "idle|training|mixed|unknown",
  "local_pids": [],
  "remote_status": {},
  "action_taken": "none|cleanup|escalate|...",
  "pending_requests": [],
  "issues_found": [],
  "note": "简短说明"
}
```

---

## 不允许的行为（Red Lines）

- ❌ 不能 kill GPU 进程，除非用户明确授权
- ❌ 不能自主启动新的 serious 训练（8-GPU）
- ❌ 不能修改训练脚本或 hyperparameters
- ❌ 不能自主批准 TRAINER_REQUESTS
- ❌ 不能假设 unknown = stale（先调查）
- ❌ 不能重复执行失败的操作（循环 retry）

---

## 回复格式

```
## HEARTBEAT [timestamp]

### 本地 GPU
- GPU 0-7: [状态]
- 活跃进程: [PID 列表或 none]

### 远程集群
- node0 (28.89.17.143): [状态]
- node1-3: [状态]

### 活跃训练
- [实验名称]: step X/Y, loss Z, 健康/异常

### 待审批请求
- [request_id]: [内容] → 需要你决定

### 问题
- [如有]

### 结论
HEARTBEAT_OK  或  [需要关注的事项]
```
