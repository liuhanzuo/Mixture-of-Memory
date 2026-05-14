# HEARTBEAT.md — 自动监控操作手册

每 20 分钟触发一次。目标：监控实验状态，发现问题，**像 main agent 一样自主行动**。

---

## ⚡ 执行计划书（2026-05-11 起）

**首要任务：对照 `status/H_V2_PLAN.md` 推进 H-series v2 训练和基线复现。**

每次 heartbeat 必须先读取 `CODEBUDDY.md` 和 `status/H_V2_PLAN.md`，理解当前阶段和下一步。计划书里明确列出：
- 每个节点当前跑什么
- 每个任务的完成标志
- 完成后自动触发的下一步
- 哪些操作无需审批（auto_launch=true）

**`status/H_V2_PLAN.md` 现在视为常态化 / 持续维护的 plan 文件，不是一次性草案。**
- heartbeat 在处理完问题、推进任务、确认新状态后，**可以直接更新 `H_V2_PLAN.md`**，把它当作当前执行面的主 plan 文档持续维护
- 允许更新的内容包括：节点映射、正在运行/已完成状态、next action、H20 eval 进度、以及新的执行决议

**不读 H_V2_PLAN.md 的 heartbeat = 无效 heartbeat。**

---

## ⚡ 架构说明（必读）

**Heartbeat session 本身就是 main agent，能力完全一致。**

- Heartbeat 由 CronCreate 触发，启动一个新的 Claude Code agent session
- 这个 session 自动读取 CODEBUDDY.md（通过 system prompt 注入），拥有所有工具：Bash、Read、Write、Edit、**Agent**
- **没有一个"独立的 main agent"在后台等待唤醒** — 每次对话（包括用户对话）都是独立 session
- Heartbeat 发现问题 → 直接用 `Agent` tool 派 researcher/coder subagent → subagent 完成后返回结果 → heartbeat 继续执行

### 派 researcher subagent

```python
Agent(
    subagent_type="general-purpose",
    description="分析 chunk_isolation 实验结果",
    prompt="""你是 Mixture-of-Memory 项目的 researcher subagent。
    工作目录：/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/
    请先读取 CODEBUDDY.md 了解项目背景，然后分析以下实验数据...
    [具体数据和问题]
    
    输出格式：
    - 根因分析
    - 建议方案
    - confidence: high/medium/low
    """
)
```

### 派 coder subagent

```python
Agent(
    subagent_type="general-purpose",
    model="reasoning",
    description="修复 cross_attn_memory 中的 bug",
    prompt="""你是 Mixture-of-Memory 项目的 coder subagent。
    工作目录：/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/
    请先读取 CODEBUDDY.md 了解规范，然后修改以下代码...
    [具体修改要求：文件路径、修改内容、不能碰哪些文件]
    
    完成后报告：修改了哪些文件、做了什么改动
    """
)
```

**注意**：subagent 没有本次 heartbeat 的上下文，prompt 必须自包含（包含所有必要信息）。

---

## ⚡ 遇到问题时的决策原则

**发现任何问题或不确定如何处理时，读取 `CODEBUDDY.md`，按里面的规则决策和操作。**

CODEBUDDY.md 包含完整的：
- 自主派发规则（哪些操作不需要用户审批）
- 多节点并行调度准则
- 实验生命周期自动化流程
- Red Lines（不允许的操作）
- Subagent 使用准则（何时派 researcher / coder）

Heartbeat 拥有和 main agent 完全相同的权限和工具，**CODEBUDDY.md 里允许 main agent 做的事，heartbeat 全部可以做**。遇到边界情况，以 CODEBUDDY.md 为准，不要保守地等待用户。

### 整改优先原则（2026-05-11 用户指令）

**Heartbeat 的默认目标不是“汇报问题”，而是“发现问题后立即整改并形成闭环”。**

**标准闭环：**
1. 发现问题
2. 派 researcher 分析（**optional**，仅当根因不清或需要判断下一步时）
3. 派 coder 修复（**optional**，仅当需要改代码时）
4. **继续训练 / 继续调度 / 继续执行 pending 任务**

- **默认动作 = 诊断 → 整改 → 复查 → 记录 → 继续训练**，不是只描述现象
- 只有以下三类情况允许“只汇报不整改”：
  1. 需要用户审批的高风险/不可逆操作
  2. 证据不足，继续动作可能误伤
  3. 外部资源不可用（如账号权限、机器已被平台回收）
- 如果只是 “某服务器 SSH 失败 / 某实验异常 / 某节点空闲” → **不够**；必须继续执行对应整改动作
- researcher 不是必派：如果问题已经明显（如 stale pid、状态文件过期、GPU 空闲但有 auto_launch 任务），heartbeat 直接整改
- coder 不是必派：如果不需要改代码（如重启训练、迁移节点、刷新状态、kill orphan、重调度任务），heartbeat 直接执行
- 一旦 researcher / coder 给出高置信结论，heartbeat **同一轮内**继续推进到启动/恢复训练，不能停在“已分析”或“已修复待后续”
- 典型整改动作包括：
  - retry SSH / 交叉验证节点是否存活
  - kill stale/orphan 进程
  - 刷新 `TRAINER_ACTIVE.md` / `remote_experiments.json` / `PENDING_TASKS.md` / `H_V2_PLAN.md`
  - researcher 分析根因后，直接派 coder 修复
  - 修复完成后直接重启实验或把任务迁移到空闲节点
  - 如果节点连续 3 次 heartbeat 无法访问，标记 `node_revoked` 并重新调度任务
- Heartbeat 输出里必须明确写出 **action_taken**，不能只有 “发现了什么问题”

---

## 每次 heartbeat 必须执行的检查

### Step 0: 待完成任务检查（PENDING_TASKS.md）

**这是最重要的步骤，必须在所有其他步骤之前执行。**

读取 `status/PENDING_TASKS.md`，检查：
- 是否有 `[PENDING]` 或 `[RUNNING]` 状态的任务
- `[RUNNING]` 任务：检查对应的训练 log，判断进度和健康状态
- `[PENDING]` 任务：如果 GPU 空闲，应该执行（除非需要用户确认）
- 如果有待完成任务且 GPU 全部空闲 → **不能报 HEARTBEAT_OK**，必须报告"有待完成任务未执行"

**重要规则**：
- 发现问题时（训练完成、训练异常、GPU 空闲但有 pending 任务）→ 必须采取行动或明确告知用户
- **绝对不允许** GPU 全部空闲 + 有 pending 任务时只输出 "HEARTBEAT_OK"
- 每次 heartbeat 必须根据当前状态更新 PENDING_TASKS.md

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

遍历 `configs/remote_experiments.json` 中 status=running 的节点。**集群分为三类，密码文件不同**：

| 集群 | IP 列表 | 密码文件 | CEPH 共享 | 备注 |
|------|---------|----------|-----------|------|
| **b200-1..4 (原始)** | 28.89.17.143, .144, 28.89.17.85, 28.89.19.134 | `configs/password.txt` | `share_303098609` (项目主目录) | 稳定，主训练资源 |
| **b200-5..8 (replacement B200)** | 28.89.18.252, 28.89.20.82, 28.89.20.27, 28.89.18.19 | `configs/password_b200_ephemeral.txt` | `share_303098609` (与主项目同一 share) | 当前 replacement B200 节点；密码文件已更新，可直接用于 heartbeat SSH |
| **h20-1..4 (H20)** | 28.58.244.13, 28.85.54.125, 28.59.5.176, 28.83.52.26 | `configs/password_h20.txt` | `zwfy6/share_304376610` | 8x H20 (97.8 GB)，VRAM 是 B200 一半 |

每类节点的 SSH 命令模板：

```bash
# 原始 B200
sshpass -f configs/password.txt ssh -o StrictHostKeyChecking=no \
  -o ConnectTimeout=10 root@<IP> \
  "nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader; \
   tail -5 <log_path> 2>/dev/null"

# ephemeral B200（注意 -o PreferredAuthentications=password 避免 pubkey 卡死）
sshpass -f configs/password_b200_ephemeral.txt ssh -o StrictHostKeyChecking=no \
  -o ConnectTimeout=10 -o PreferredAuthentications=password root@<IP> "<cmd>"

# 检查 ephemeral 节点是否仍存活（先 ping 再 ssh）
sshpass -f configs/password_b200_ephemeral.txt ssh -o StrictHostKeyChecking=no \
  -o ConnectTimeout=10 -o PreferredAuthentications=password root@<IP> "echo alive" 2>&1 \
  | grep -q alive && echo OK || echo DEAD
```

**replacement B200 / H20 节点 SSH 失败时**：不要立即升级。在 `status/TRAINER_ACTIVITY.jsonl` 标记 `ssh_timeout`，连续 3 次（约 60 分钟）失败才视为节点不可用，再更新状态。

**replacement B200 节点路径**：项目根 = `/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/`，与主项目完全共享，不需要 rsync。

**H20 节点路径**：项目根 = `/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/`，仍与主项目不同步；把任务派到 H20 时必须确认脚本/数据路径兼容。

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

### Step 6: 主动调研与代码审查（每次 heartbeat 必须执行）

**不需要有问题才调研。训练正在跑、一切正常时，同样应该主动派 researcher 和 coder。**

#### 6a. 主动文献/方案调研（每次 heartbeat 考虑）

读取 `RESEARCH_LITERATURE.md` 末尾，判断距离上次调研过了多久。如果超过 **2 小时**（约 6 次 heartbeat），派 researcher 做一次主动调研：

```python
Agent(
    subagent_type="general-purpose",
    run_in_background=True,   # 训练在跑时可以并行，不阻塞 heartbeat
    prompt="""你是 Mixture-of-Memory 项目的 researcher。
    工作目录：/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/
    请先读取 CODEBUDDY.md 了解项目背景和当前方向，再读取 RESEARCH_LITERATURE.md 了解已调研内容。
    
    当前正在运行的实验：[heartbeat 在这里填入当前实验摘要]
    
    请围绕当前实验方向，调研以下内容（选择最相关的 1-2 个方向）：
    - 同类方法的最新实现技巧（如 memory slot 更新策略、routing 设计）
    - 有没有论文解决了我们正在遇到的问题（如 cross-chunk 信息传递）
    - 有没有我们可以借鉴的 initialization / training trick
    - 当前代码实现与论文原版是否有偏差
    
    输出：
    1. 调研发现（具体论文/代码片段/技巧）
    2. 对当前实验的建议（如果有）
    3. 是否发现值得立即实施的改进（confidence: high/medium/low）
    
    完成后 append 到 RESEARCH_LITERATURE.md（格式：## [日期] [主题]）
    append 到 status/RESEARCHER_REPORTS.jsonl
    """
)
```

#### 6b. 主动代码审查（每次 heartbeat 考虑）

读取 `status/RESEARCHER_REPORTS.jsonl` 末尾，判断距离上次代码审查过了多久。如果超过 **4 小时**（约 12 次 heartbeat），或者刚完成一个重要代码改动，派 coder 做代码审查：

```python
Agent(
    subagent_type="general-purpose",
    run_in_background=True,
    prompt="""你是 Mixture-of-Memory 项目的 coder，负责代码审查。
    工作目录：/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/
    请先读取 CODEBUDDY.md 了解项目规范。
    
    当前核心代码文件：[heartbeat 在这里填入当前实验使用的主要脚本]
    
    请检查以下内容：
    1. 是否有明显的 bug（数值精度、tensor shape、梯度截断等）
    2. memory 读写逻辑是否与设计文档（versions/ 下）一致
    3. eval 逻辑是否正确（vanilla_ppl 和 memory_ppl 计算是否对称）
    4. 是否有可能导致 ratio 虚假改善的 bug（如 eval 时 memory 影响了 vanilla 的 ppl 计算）
    
    输出：
    - 发现的问题（如有）：文件名 + 行号 + 问题描述
    - 建议的修复（如有）
    - 如果发现 critical bug → 直接修复（不需要等待审批）
    
    完成后 append 到 status/ISSUES.jsonl（如有发现）
    """
)
```

#### 6c. 处理 CODE_CLEANUP_SUGGESTIONS.md（每周 CI 更新后检查）

检查文件是否在过去 7 天内被 CI 更新：
```bash
git log --since="7 days ago" --oneline -- CODE_CLEANUP_SUGGESTIONS.md | head -3
```

如果有更新，读取文档并按以下规则处理：

**confidence: high 的建议 → 自主执行：**
```bash
# 1. 先 grep 验证文件未被 import
grep -r "$(basename <filepath> .py)" src/ scripts/ tests/ --include="*.py" | grep -v "__pycache__"
grep -r "$(basename <filepath> .py)" src/ --include="__init__.py"
```
- 如果 **无 import** → 派 coder 执行删除或移入 `legacy/`：
```python
Agent(
    subagent_type="general-purpose",
    prompt="""你是 Mixture-of-Memory 项目的 coder。
工作目录：/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/
任务：根据 CODE_CLEANUP_SUGGESTIONS.md 中 confidence: high 的建议执行代码清理。
规则：
1. 读取 CODE_CLEANUP_SUGGESTIONS.md，找出 confidence: high 的条目
2. 对每个候选文件，先 grep 验证无 import：
   grep -r "<module_name>" src/ scripts/ tests/ --include="*.py" | grep -v __pycache__
3. 无 import → 删除（如果是纯死代码）或移入 legacy/（如果有历史参考价值）
4. 如果移入 legacy/，更新或创建 legacy/README.md，说明文件的历史作用
5. git add <修改的文件> && git commit -m "chore: cleanup dead code per CODE_CLEANUP_SUGGESTIONS.md"
6. 调用 /gitpush skill 完成推送（/gitpush 会派 subagent 审核后再 push）
   - 或者手动流程：先派 general-purpose subagent 审核 git diff，APPROVED 后再 push
   - export http_proxy=http://star-proxy.oa.com:3128 && git push origin main
7. 报告：删除了哪些文件，移动了哪些文件，commit hash 是多少
绝对不能碰：src/memory/cross_attn/, configs/, status/, CODEBUDDY.md, HEARTBEAT.md, CODE_CLEANUP_SUGGESTIONS.md
"""
)
```
- 如果**有 import** → 降级为 medium 处理

**confidence: medium 的建议 → 写入 PENDING_TASKS.md：**
```markdown
### [PENDING] code_cleanup_<YYYY-MM-DD> — 代码清理（需确认）
- priority: low
- auto_launch: false (需要用户确认)
- description: |
    CODE_CLEANUP_SUGGESTIONS.md 中有 medium confidence 建议需要用户确认后执行。
    候选文件：<列出文件名和理由>
    请查看 CODE_CLEANUP_SUGGESTIONS.md 后决定是否执行。
```

**confidence: low 的建议 → 忽略，不操作**

#### 6d. 触发条件汇总

| 触发条件 | 动作 |
|---------|------|
| 距上次文献调研 > 2h | 派 researcher（后台，不阻塞） |
| 距上次代码审查 > 4h | 派 coder 审查（后台，不阻塞） |
| 实验出现新结果（eval 完成） | 立即派 researcher 分析（前台，等结果） |
| 训练完成 | 立即派 researcher 分析结论 + 决定下一步 |
| 发现 bug 或异常 | 立即派 researcher 根因分析（前台） + coder 修复 |
| 方向停滞（连续 3 个实验无改善） | 派 researcher 做系统性文献调研，寻找新方向 |
| CODE_CLEANUP_SUGGESTIONS.md 过去 7 天内有更新 | 检查 high/medium 建议，按 6c 规则处理 |

---

## 判断和行动规则

### 一切正常 → 仍然主动调研

如果：
- 所有 GPU 占用都是 expected
- 无待审批请求
- 活跃训练健康
- 无 open issues
- **PENDING_TASKS.md 中无未完成任务或所有任务都在执行中**

**不是直接 HEARTBEAT_OK。还需执行 Step 6：检查是否该派 researcher 调研或 coder 审查代码。**

只有 Step 6 也不需要触发时，才输出 `HEARTBEAT_OK`。

**如果 PENDING_TASKS.md 有未执行任务且 GPU 空闲 → 不能报 HEARTBEAT_OK，必须 WARNING + 执行任务。**

### 小问题 → 自主修复

可以自主处理：
- 注册表过时（registry stale），但进程明显健康
- TRAINER_ACTIVE.md 需要刷新（用 write，不用 edit）
- 小型 debug 进程占用 1 个 GPU，不影响计划工作

### 中等问题 → 自主调查 + 派 researcher

- 未知进程，但不明显是 stale
- 远程节点 SSH 超时（先尝试 3 次）
- 训练进展异常（loss 停滞超过 50 steps）
- 实验完成，需要分析结果、决定下一步

**操作（按顺序）**：
1. 调查：读日志、SSH 检查、收集诊断信息
2. 写入 ISSUES.jsonl
3. 如果根因已经明显 → 直接整改；如果根因不清 → **派 researcher subagent**（用 Agent tool，见上方 "架构说明" 的模板）
4. 如果整改需要改代码 → 派 coder；如果不需要改代码 → 直接改参数 / 迁移节点 / 重启任务
5. **同一轮 heartbeat 内继续训练或继续调度**，不要停在分析阶段
6. 在 PENDING_TASKS.md 记录分析结论和下一步

### 大问题 → 自主处理（含 kill + 修复 + 重启）

- 活跃训练 crash
- stale/orphan 进程（需要 kill）
- GPU 内存泄漏
- 远程节点所有实验都失败
- 显著 bug（训练进程运行但核心功能完全失效）

**操作（按顺序）**：
1. 收集充分证据（≥3 个诊断点）
2. 写入 ISSUES.jsonl（包含完整证据）
3. 如果是显著 bug / stale 进程 → 自主 kill 问题进程
4. 如果根因不清 → **派 researcher 分析根因**（run_in_background=false，等结果）；如果根因已清楚可跳过 researcher
5. 如果需要改代码 → **派 coder 修复**（run_in_background=false，等完成）；如果不需要改代码可跳过 coder
6. 整改完成后 → **立即 SSH 启动/恢复训练，或把任务迁移到空闲节点继续跑**
7. 更新 TRAINER_ACTIVE.md（write 覆盖）、gpu_runs.jsonl（append）、UPDATELOG.md（append）
8. 复查新进程 / 新日志，确认 heartbeat 真正完成闭环
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

## 实验生命周期自动化（2026-05-03 用户指令）

**核心原则：实验完成不是终点，是下一步的起点。Heartbeat 必须形成闭环。**

### 实验完成后的强制流程

当 heartbeat 发现训练完成时，**必须在同一个 heartbeat 内**执行以下流程：

1. **记录结果** — 更新 TRAINER_ACTIVE.md、gpu_runs.jsonl、PENDING_TASKS.md
2. **分析结果** — 读取最终 eval 指标，判断实验结论（成功/失败/需改进）
3. **决定下一步** — 基于结果，在 PENDING_TASKS.md 写入新任务
4. **立即执行** — 如果有空闲 GPU + 新任务不需要用户审批 → **立刻启动，不等下一个 heartbeat**

### 什么算 "可自主启动"（无需用户审批）

以下**全部可自主启动**：
- 同一算法的 hyperparameter sweep（改 lr/num_slots/top_k/chunks_per_doc 等）
- 同一算法在不同数据集/序列长度上的评估
- 已有算法的 bug fix 或稳定性改进
- 已有实验的 checkpoint 分析或对比评估
- 文献调研、结果分析、代码改进
- **方向切换 — 前提：`/researcher` 分析后建议切换，且报告标注 confidence: high/very_high**
  - 例：v4 slot memory 收敛后，researcher 分析认为 KV cache quantization 更有前景 → 可自主切换
  - 例：某个方向连续 3 个实验都失败，researcher 建议换方向 → 可自主切换
  - researcher 必须给出：切换理由、新方向预期优势、具体实验计划

以下**仍需用户审批**：
- researcher 没有确认的方向切换（自己觉得"应该换"不行）
- 涉及大规模算力/数据变更（如从 8B 切到 70B 模型）

**判断规则：如果能用现有代码+改参数直接跑 → 自主启动。如果需要写新代码 → 先派 coder，代码完成后自主启动。如果 researcher 确认换方向 → 自主执行新方向。**

### 多实验并行调度规则（确保不浪费 GPU）

**核心原则：4 个 B200 节点 + 1 个本地 8×H20 = 5 组 GPU。任何时刻空闲节点 ≥ 1 且有可做的事 = 浪费。**

Heartbeat 必须维护一个 **"节点占用表"**，每次检查时：

| 节点 | 当前实验 | 预计完成时间 |
|------|----------|------------|
| b200-1 | ... | ... |
| ... | ... | ... |

#### 调度算法（每次 heartbeat 执行）

1. **扫描空闲节点**：哪些节点的 GPU utilization = 0%
2. **扫描可执行任务**：PENDING_TASKS.md 中 `auto_launch: true` 的任务
3. **匹配规则**：
   - 每个空闲节点分配一个任务（不同节点可跑不同实验）
   - 优先级排序：high > medium > low
   - 如果任务数 < 空闲节点数 → 派 researcher 分析当前结果，产出新的实验想法，填充任务队列
   - 如果任务数 > 空闲节点数 → 按优先级排，剩余任务等下一个节点空闲
4. **代码依赖处理**：
   - 任务不需要新代码 → 直接 SSH 启动
   - 任务需要新代码 → 派 coder 在后台写，节点先分配给其他不需要代码的任务
   - coder 完成后，下一次 heartbeat 在空闲节点启动
5. **实验完成时**：
   - 某个节点训练完成 → **立即**分析结果、决定下一步、在该节点或其他空闲节点启动
   - 不要等所有节点都完成再统一分析

#### 典型并行场景

**场景 A：4-arm ablation 正在跑，1 个节点先完成**
- 分析该 arm 结果
- 如果结果有启发（如 "slots=8 最优"）→ 在该空闲节点启动 follow-up 实验（如 slots=4 vs slots=8 精细对比）
- 不等其他 3 个 arm 完成

**场景 B：所有实验完成，4 个节点全空**
- 派 researcher 分析所有结果，产出下一步建议
- 同时：如果 researcher 建议有 high confidence → 直接启动新方向
- 如果没有明确下一步 → 在 1 个节点跑 eval/baseline 对比，另 1 个节点跑 checkpoint 分析，不浪费

**场景 C：实验方向需要代码改动**
- 派 coder 写代码（后台）
- 剩余空闲节点跑其他不需要代码的任务（如已有 checkpoint 的评估）
- 代码写完后立即在空闲节点启动新实验

### 新任务需要新代码时的处理

如果分析后决定的下一步需要新代码（如新的 eval 脚本）：
1. 在 PENDING_TASKS.md 写入任务，标记 `needs_code: true`
2. **立即派 `/coder`** subagent 实现代码
3. coder 完成后，**立即在空闲节点启动实验**
4. 全程不停顿、不等用户

### 空闲 GPU 处理规则（强化版）

**heartbeat 发现任何 GPU 空闲时的强制动作：**

1. 检查 PENDING_TASKS.md 是否有 `[PENDING]` + `auto_launch: true` 的任务
2. 如果有 → 分配到空闲节点（不需要代码的直接启动，需要的先派 coder）
3. 如果没有 pending 任务 → 基于最近实验结果分析下一步，写入新任务，启动
4. 如果没有明确下一步 → 派 researcher 调研，产出新实验计划
5. **绝对不允许**：有空闲 GPU + 无事可做 → 直接报 HEARTBEAT_OK

**PENDING_TASKS.md 任务必须包含 `auto_launch` 字段：**
- `auto_launch: true` → heartbeat 发现空闲 GPU 时自动启动
- `auto_launch: false` → 需要用户确认后才启动（仅用于高风险/高成本操作）

---

## 不允许的行为（Red Lines）

- ❌ 不能 kill GPU 进程，除非用户明确授权或 Red Line #7 规定的显著 bug
- ❌ 不能自主启动全新方向的训练（非 ablation/fix 延伸）
- ❌ 不能修改训练脚本或 hyperparameters（但 researcher 确认的更改除外）
- ❌ 不能自主批准 TRAINER_REQUESTS
- ❌ 不能假设 unknown = stale（先调查）
- ❌ 不能重复执行失败的操作（循环 retry）
- ❌ **不能在实验完成后等用户回复才决定下一步**（2026-05-03 新增）
- ❌ **不能在 GPU 空闲 + 有 pending 任务时问用户 "要我做什么"**（2026-05-03 新增）

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
