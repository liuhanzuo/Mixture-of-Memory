---
model: claude-opus-4.7-1m
---

# /heartbeat — 闭环巡检与推进

每 20 分钟执行一次。**你是 main agent**，权限和能力完全相同（CODEBUDDY.md 里允许 main 做的事，你都可以做，无需用户审批）。

时区：**Asia/Shanghai (GMT+8)**。

---

## 闭环（每次 heartbeat 必须走完）

```
[1] 查看 GPU 状态 & plan 目录
        ↓
[2] 处理错误（坏进程 / 卡死 / 训练崩溃）
        ↓
[3] 按 plan 推进（PENDING_TASKS / H_V2_PLAN）
        ↓
[4] 该开跑就开跑（GPU 空闲 + 有 auto_launch=true 任务 → 立刻派 worker）
        ↓
[5] 记录下一步到 plan（更新 PENDING_TASKS.md，写明白下一拍要做什么）
```

**绝对禁止**：GPU 全空 + 有 PENDING 任务时只输出 `HEARTBEAT_OK`。这是空转，浪费一拍。

---

## Step 1：查看 GPU 状态 & plan 目录

```bash
date "+%Y-%m-%d %H:%M:%S %Z"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
ps aux | grep -E "python|nohup" | grep -v grep
```

读以下文件（必读）：
- `status/PENDING_TASKS.md` — 任务看板
- `status/TRAINER_ACTIVE.md` — 当前活跃训练
- `status/H_V2_PLAN.md`（如存在）— 当前计划
- `status/ISSUES.jsonl` 末尾 5 行 — 未解决的问题

**检查第二节点（28.59.80.196）GPU 状态**：
```bash
PASS=$(cat configs/password_h20_nodes.txt)
expect -c "
set timeout 15
spawn ssh -o StrictHostKeyChecking=no -o PreferredAuthentications=password -o ConnectTimeout=10 root@28.59.80.196 \"nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader\"
expect \"password:\"
send \"$PASS\\r\"
expect eof
"
```
（password 在 `configs/password_h20_nodes.txt`，已 gitignore；末尾逗号是密码一部分）

回复**第一行**必须是：`## HEARTBEAT [YYYY-MM-DD HH:MM GMT+8]`

---

## Step 2：处理错误

对每个发现的错误**立即修**，不要只记录：

| 现象 | 行动（不需要审批） |
|------|-------------------|
| GPU 上有 `unknown` 或 `stale_orphan` 进程 | `kill <pid>` 并记录原因 |
| 训练 log 显示 PPL > 100 | kill 训练，派 researcher 分析根因 |
| 训练 log 连续 2 次 heartbeat 无新 step | kill stalled |
| ckpt 写不出来 / 磁盘满 | `df -h` 排查，清理无用 outputs/ |
| ssh 不通某节点 | 重试 1 次；不行就在 PENDING_TASKS 标记 unreachable |

修完了 append 一行到 `status/UPDATELOG.md`。

---

## Step 3：按 plan 推进

读 `status/PENDING_TASKS.md`，对每条任务判断：

- **`[RUNNING]` + 进程还活着**：检查健康度（loss 下降？步数推进？），健康就跳过
- **`[RUNNING]` + 进程死了**：标记为 `[FAILED]`，写明死亡原因，根据情况派 coder/researcher 或重启
- **`[PENDING]` + auto_launch=true + GPU 有空闲**：**立即启动**（Step 4）
- **`[PENDING]` + auto_launch=false**：列入"等用户确认"清单，不动

---

## Step 4：该开跑就开跑

启动方式（按任务大小选）：

### 单 GPU eval / 短 inference（< 30 GB VRAM）
直接 `nohup ... &` + `tee` 到 `logs/<exp>_<ts>.log`，不派 subagent

### 多 GPU 训练 / 长跑（> 1 h）
派 background subagent：
```python
Agent(
    subagent_type="general-purpose",
    model="reasoning",
    description="<short>",
    prompt="<self-contained>",  # 必须包含工作目录、命令行、约束、输出要求
    run_in_background=True
)
```

启动后**立即** append 到：
- `status/gpu_runs.jsonl`：`{"ts": ..., "node": "h20-1", "exp": "...", "commit_hash": "...", "status": "running"}`
- `status/TRAINER_ACTIVE.md`（Write 覆盖，不要 Edit）

---

## Step 5：记录下一步到 plan

更新 `status/PENDING_TASKS.md`：
- 这一拍启动的任务 → 标 `[RUNNING]`
- 这一拍完成的任务 → 移到 `## [DONE]` 区（带完成时间）
- 新发现的工作 → 添 `[PENDING]` 条目（标 auto_launch true/false）
- **下一拍 heartbeat 要做什么**：在文件顶部 `## Next heartbeat actions` 区写明白

如果 `H_V2_PLAN.md` 等其他 plan 文件需要更新（节点映射、执行决议），一并更新。

最后追加一行到 `status/TRAINER_ACTIVITY.jsonl`：
```json
{"ts": "<iso>", "event": "heartbeat", "actions": [...], "next": "..."}
```

---

## 报告格式（heartbeat 的最终输出）

```
## HEARTBEAT [2026-MM-DD HH:MM GMT+8]

### Step 1: 状态
- 本机 GPU: GPU 1 has process X (job Y, healthy, step 234/500)
- 第二节点 GPU: all idle
- Pending tasks: 3 ([PENDING] auto_launch=true × 1, [PENDING] auto_launch=false × 2)

### Step 2: 错误处理
- (none) 或 killed PID 12345 stale orphan

### Step 3: 推进决策
- Task A is healthy, no action
- Task B [PENDING auto_launch=true] → launching now

### Step 4: 启动
- Launched: <Agent ID 或 nohup PID>, log path

### Step 5: Plan 更新
- PENDING_TASKS.md 写入 next heartbeat 要做的事
- TRAINER_ACTIVITY.jsonl appended
```

---

## 自主授权速查（无需用户审批）

按 CODEBUDDY.md 的标准授权清单：
- 派 researcher / coder subagent
- researcher confidence:high 的代码/参数改动 → 直接执行
- kill 不健康训练（PPL > 100、stalled 2 拍、crash）
- 在空闲节点启动 PENDING auto_launch=true 任务
- ablation / fix 延伸训练（同算法改参数、同代码新数据、ckpt eval）
- 实验完成后自动决定下一步并立即执行
- git commit / git push（带 subagent 审核）

仍需用户审批：全新方向训练（非延伸）、架构重大重构、kill 健康训练。

---

## 资源备忘

**当前可用集群（2026-05-14 起，旧的 b200-1..8 全部已不在）**：
- 本机 H20：`29.162.227.178`，8× H20 (97.8 GiB / 卡)
- 第二节点 H20：`28.59.80.196`，8× H20
- SSH 密码（两节点相同）：见 `configs/password_h20_nodes.txt`（gitignored，末尾逗号是密码一部分）
- 共享文件系统：`/apdcephfs_zwfy6/share_303098609/`（两节点都看到同一份）
- 工作目录：`/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory/`
- conda env：`/opt/conda/envs/torch-base`（torch 2.8 + transformers 5.8.1 + accelerate + peft）
- HF proxy：`http_proxy=http://star-proxy.oa.com:3128`
- 模型：`models/{Meta-Llama-3-8B,Meta-Llama-3-8B-Instruct,Llama-3.2-1B-Instruct,Qwen2-7B-Instruct,Beacon-Qwen2-7B}`
- BABILong package：`third_party/babilong-pkg/`，需要 `PYTHONPATH=$(pwd)/third_party/babilong-pkg:$PYTHONPATH`

旧的 `b200-1..8` 集群、`/apdcephfs_wzc1/...`、`share_304376610` 路径**全部失效**，遇到引用这些的脚本要把路径改成上面的。
