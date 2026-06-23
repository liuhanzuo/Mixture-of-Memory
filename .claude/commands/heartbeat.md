---
model: claude-opus-4-7
---

# /heartbeat — 闭环巡检与推进

每 40 分钟执行一次。**你是 main agent**，权限和能力完全相同（CODEBUDDY.md 里允许 main 做的事，你都可以做，无需用户审批）。

时区：**Asia/Shanghai (GMT+8)**。

---

## 闭环（每次 heartbeat 必须走完）

```
[1] 查看训练状态
        ↓
[2] 处理错误（坏进程 / 卡死 / 训练崩溃）
        ↓
[3] 按 plan 推进（GPU 空闲 + 有 auto_launch=true 任务 → 立刻启动）
        ↓
[4] 记录状态到 TRAINER_ACTIVITY.jsonl
```

---

## Step 1：检查状态

对每个节点检查训练 log 最新 step：

**本机 H20**：
```bash
grep "\[step" logs/prepend_all_fastmem_10k_h20.log 2>/dev/null | tail -2
```

**远程 H20**（28.59.80.196）：
```bash
sshpass -f configs/password_h20_new.txt ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o PreferredAuthentications=password root@28.59.80.196 "grep '\[step' /apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory/logs/prepend_all_fastmem_10k_remote_h20.log 2>/dev/null | tail -2"
```

**H800 Node 1**（30.203.132.121）：
```bash
sshpass -f configs/password_h800_new.txt ssh -o StrictHostKeyChecking=no -o ConnectTimeout=60 -o PreferredAuthentications=password root@30.203.132.121 "grep '\[step' /apdcephfs_jn2/share_304376610/pighzliu_code/Mixture-of-Memory/logs/prepend_all_10k_node1.log 2>/dev/null | tail -2"
```

**H800 Node 2**（30.203.138.209）：
```bash
sshpass -f configs/password_h800_new.txt ssh -o StrictHostKeyChecking=no -o ConnectTimeout=60 -o PreferredAuthentications=password root@30.203.138.209 "grep '\[step' /apdcephfs_jn2/share_304376610/pighzliu_code/Mixture-of-Memory/logs/prepend_all_10k_node2.log 2>/dev/null | tail -2"
```

也检查 eval 结果：
```bash
grep "eval step" logs/prepend_all_fastmem_10k_h20.log 2>/dev/null | tail -1
```

回复**第一行**必须是：`## HEARTBEAT [YYYY-MM-DD HH:MM GMT+8]`

---

## Step 2：处理错误

对每个发现的错误**立即修**：

| 现象 | 行动 |
|------|------|
| 训练 log 连续 2 次 heartbeat 无新 step | kill stalled，重启 |
| 训练崩溃（log 有 Error/Traceback） | 诊断并重启 |
| GPU 空闲 + PENDING_TASKS 有 auto_launch=true | 立即启动 |
| GPU 空闲 + 据当前结论能推断有价值的下一步 | **直接派 coder 写代码 + 启动新实验**（2026-06-21 用户授权，无需 needs_code 审批）|
| 训练完成（step 达到 total_steps） | 收集结果，决定下一步，立即启动延伸实验 |

### ★写代码 + 启动新实验授权（2026-06-21 用户指令）

**heartbeat 可自主写代码 + 启动新实验，无需 emit needs_code 等审批。** GPU 空闲时即使没有现成 PENDING 任务，只要能据当前结论推断有价值的下一步，就直接派 `/coder`（或 general-purpose，`model="reasoning"`，`run_in_background=true`）写代码 + 跑多卡实验，把空闲卡用起来。代码改动遵守 CODEBUDDY.md（committer=LiuHanzuo，不加 AI 署名，`git add <具体文件>`）。完成后落账 + emit train_done。**仅重大方向切换/架构重构仍 emit needs_code 等主会话。绝不允许有空闲卡却只报 HEARTBEAT_OK 不行动。**

---

## Step 2.5：检查 GPU 监控前端（2026-06-20 新增，每次 heartbeat 必查）

监控前端是给用户实时看三节点显存/利用率/功耗+当前运行任务的网页，跑在**本机 8088 端口**。

**必须用 Bash `run_in_background=true` 启动**（不能用 setsid/nohup 脱离后台——codebuddy sandbox 会把脱离的进程隔离到独立 network namespace，导致端口在主环境不可达）。

检查命令（主环境 curl，不是 ss——ss 看不到跨 netns）：
```bash
curl -s -m 8 -o /dev/null -w "%{http_code}" http://127.0.0.1:8088/api/data
```
- 返回 `200` → 前端正常，无需操作。
- 返回 `000` / 非 200 / 超时 → 前端挂了，重启：
  1. 先清理残留进程（注意 grep pattern 别匹配到当前 shell 命令本身）：
     ```bash
     ps -eo pid,cmd | grep "monitor/gpu_monitor_server.py --port" | grep -v grep | grep -v "ps -eo" | awk '{print $1}' | while read pid; do kill -9 "$pid" 2>/dev/null; done
     ```
  2. 用 Bash 工具的 `run_in_background=true` 重新启动（**关键：用这个，不要 setsid/nohup**）：
     ```
     cd <PROJECT_ROOT> && .venv/bin/python -u monitor/gpu_monitor_server.py --port 8088 --interval 5
     ```
  3. 等约 50s（首轮采集含 .196/B200 的 ssh,较慢）后再 curl 确认 200 + `/api/data` 的 tasks 字段有当前 run。
- 报告里加一行：`- Monitor: http200 OK (local/.196/B200 tasks shown)` 或 `restarted`。

---

## Step 3：报告格式

```
## HEARTBEAT [YYYY-MM-DD HH:MM GMT+8]
- H20 local: step X/10000, lm=Y, healthy
- H20 remote: step X/10000, lm=Y, healthy
- H800 node1: step X/10000, lm=Y, healthy
- H800 node2: step X/10000, lm=Y, healthy
- Action: none / <描述操作>
```

最后追加一行到 `status/TRAINER_ACTIVITY.jsonl`。

---

## 资源备忘

**当前可用集群（2026-06-21 更新，覆盖旧 H800 记录——H800 已下线）**：
- 本机 H20：8× H20 (97.8 GiB/卡)，直接本地访问，`.venv/bin/python`
- 第二节点 H20：`28.59.80.196`，8× H20，密码 `configs/password_diskA.txt`（与本机共享盘A FS）
- 回归 H20：`28.48.7.53` / `28.58.245.174`，各 8× H20，密码 `configs/password_h20_returned.txt`，挂盘B `/apdcephfs_zwfy6/share_304376610/`，项目 `.venv/bin/python` 可用
- B200 `28.89.16.18:36000`：8× L20A (183 GiB/卡)，密码 `configs/password_b200_new.txt`，端口 36000
- B200 `28.88.184.53`：8× L20A (183 GiB/卡)，密码 `configs/password_b200_53.txt`，22 端口（2026-06-21 新增，与 .18 共享 wzc1 盘）
- B200 用 `.venv/bin/python`(torch2.10 支持 L20A)；⚠️ L20A 跑不了 faithful Landmark(torch2.1)，但能跑 mem_space 训练/eval
- SSH 命令：`sshpass -f <password_file> ssh -o StrictHostKeyChecking=no -o ConnectTimeout=12 -o PreferredAuthentications=password root@<IP> "<cmd>"`（.18 加 `-p 36000`）
- 盘A FS（本机 + .196）：`/apdcephfs_zwfy6/share_303098609/`
- 盘B FS（回归 H20 + .76/.249）：`/apdcephfs_zwfy6/share_304376610/`
- wzc1 FS（两台 B200 共享）：`/apdcephfs_wzc1/share_304376610/`
- 工作目录（盘A H20）：`/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory/`；（盘B H20）：`/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/`；（B200）`/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/`
- 工作目录（H800）：`/apdcephfs_jn2/share_304376610/pighzliu_code/Mixture-of-Memory/`
- Python（本机）：`.venv/bin/python`
- Python（远程 H20 / H800）：`/opt/conda/envs/torch-base/bin/python`

**当前活跃训练**：
- 本机 H20: `prepend_all_fastmem_10k_h20` (seed=42, 10k steps)
- 远程 H20: `prepend_all_fastmem_10k_remote_h20` (seed=44, 10k steps)
- H800 Node 1: `prepend_all_10k_node1` (seed=42, 10k steps, wandb=xx72k497)
- H800 Node 2: `prepend_all_10k_node2` (seed=44, 10k steps, wandb=9oljda98)
