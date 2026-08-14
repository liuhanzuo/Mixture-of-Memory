---
model: opus
---

# /heartbeat — 闭环巡检与推进

每 40 分钟执行一次。**你是 main agent**，权限和能力完全相同（CODEBUDDY.md 里允许 main 做的事，你都可以做，无需用户审批）。

时区：**Asia/Shanghai (GMT+8)**。

---

## 闭环（每次 heartbeat 必须走完）

```
[1] 查 GPU（先） + 查训练 log（后）—— 两者对照
        ↓
[2] 处理错误（坏进程 / 卡死 / 训练崩溃 / 抢卡）
        ↓
[3] 按 plan 推进（GPU 空闲 + 有 auto_launch=true 任务 → 立刻启动）
        ↓
[4] 记录状态到 TRAINER_ACTIVITY.jsonl
```

---

## Step 0：先读台账，再动手

```bash
head -40 status/GPU_STATUS.md      # 哪节点哪张卡应该在跑什么
grep -c PENDING status/PENDING_TASKS.md
```
台账说在跑、实测空闲 = 任务已完成或已崩 → 补卡 + 更新台账。
**⚠️ 台账可能过期（本文件自己就过期了 2 个月）。冲突时以实测为准，然后立刻改台账。**

---

## Step 1a：GPU 实测（★ 每个节点都要，不可跳过）

**判据「GPU 空闲」必须由实测支撑，不能靠 log 推断。** 下面这条同时拿到 显存 / 利用率 / 谁在占卡：

```bash
# 本机（LOCAL == 28.89.19.21）
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
```

```bash
# 远程 4 台：.212 / .73 / .82 / .104（一律省略 -p，全局 ssh_config 已设 Port 36000）
for spec in "28.89.18.212 configs/password_b200_18212.txt 212" \
            "28.85.35.73  configs/password_h20_853573.txt 73" \
            "28.82.250.82 configs/password_h20_82250.txt  82" \
            "28.83.24.104 configs/password_h20_24104.txt  104"; do
  set -- $spec; printf "=== .%s ===\n" "$3"
  timeout 70 sshpass -f "$2" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=12 \
    -o PreferredAuthentications=password root@"$1" \
    'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader' 2>&1 | tail -8
done
```

### ★★ 三条实测铁律（每条都是踩过坑写下来的）

1. **瞬时 0% util ≠ 空闲。** 分阶段 eval driver 在每次换模型加载时都读 0%，8B ckpt 能读 0% 长达数分钟。
   **判活看 artifact 的 mtime / log 有没有推进，或连采 ≥3 次**（2026-08-14 我两次把 model-load 间隙误判为 stall）。
   反过来 **100% util 也 ≠ 在跑对的东西**（2026-08-12 连报 14 轮「5/5 busy」，其实在跑降级任务）。

2. **一张卡可能有两个主人。** `--query-gpu` 只给总显存，看不出共享。**必须用 `--query-compute-apps` 把占用拆到 PID**
   —— 2026-08-14 就是靠这个才发现 union-9 eval 和 noslorb 训练挤在 LOCAL 0-3 卡（142-150 GB / 183 GB）。
   > ⚠️ **PID 只能在本机映射。** 远程 `--query-compute-apps` 返回的是 **host namespace PID**，在容器内 `ps` 查不到
   > （实测 .104 报 163739，`ps` 里根本没有；真实 PID 是 3343485）。**远程要认任务就用 `pgrep -af <脚本名>`。**

3. **加卡不减每卡显存。** 项目训练是 plain DDP（不是 FSDP），只 all-reduce 梯度、不 shard 参数/optimizer。
   OOM 时先算 per-rank 静态显存，别指望加节点能救。

4. **显存突然大跌 + util 仍 100% ≠ 崩了，通常是到点做 eval。** 2026-08-14 实测 `.212` 从 114 GB 掉到
   21 GB（−82%），一眼看像 OOM 后残留；实际是它跑到 iter 6600 的 eval 里程碑，`eval_ppl` 释放了训练
   activation。**先看 log 有没有 `evaluating` / `saving checkpoint` / `eval_ppl`，再判生死**，
   别拿一个显存数字下结论。

---

## Step 1b：训练 log（对照 GPU 实测）

**⚠️ 不要硬编码 log 文件名——run 换了名字，写死的路径只会静默返回空。每次先按 mtime 找最新的：**

```bash
ls -t logs/*.log | head -5                       # 本机最新 log
L=$(ls -t logs/<本轮 run 关键字>*.log | head -1)  # 例: sparseforge_tm_*RESUME*
echo "mtime=$(stat -c %y "$L")"                  # ★ mtime 比内容更早暴露 stall
tail -c 4000 "$L" | tr '\r' '\n' | grep -aoE "loss=[0-9.]+|[0-9]+/[0-9]+" | tail -3
grep -aiE "Traceback|out of memory|ChildFailedError|NaN" "$L" | grep -viE "No NaN" | head -3
```

**速率一律用 elapsed/iter 自己算，不要信 tqdm 的瞬时 `s/it`：** 隔 ≥120 s 采两次 iter 号，`Δt/Δiter`。
**且必须提取单个数值**——2026-08-14 我的 grep 匹配到两行，shell 代入多行值，脚本打印出
「0.0 s/it → 没有变慢」，而同一份输出里的原始 iter 号是 60.0 s/it。**脚本的 VERDICT 行在输入畸形时不是证据。**

当前 5 节点的 log 位置（会变，以 mtime 为准）：

| 节点 | 盘 | log 路径 |
|---|---|---|
| LOCAL (=`.21`) | wzc1 | `logs/*.log`（本地直读） |
| `.212` | wzc1（**与 LOCAL 同一物理盘，无需 scp**） | 同上路径，ssh 过去读 |
| `.73`/`.82`/`.104` | zwfy6（**独立 checkout**） | `/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/logs/` |

回复**第一行**必须是：`## HEARTBEAT [YYYY-MM-DD HH:MM GMT+8]`


---

## Step 2：处理错误

对每个发现的错误**立即修**：

| 现象 | 行动 |
|------|------|
| 训练 log 连续 2 次 heartbeat 无新 step **且 mtime 也没动** | kill stalled，重启 |
| 训练崩溃（log 有 Error/Traceback/OOM） | 诊断并重启 |
| GPU 空闲（见下方定义）+ PENDING_TASKS 有 auto_launch=true | 立即启动 |
| GPU 空闲 + 据当前结论能推断有价值的下一步 | **直接派 coder 写代码 + 启动新实验**（2026-06-21 用户授权，无需 needs_code 审批）|
| 训练完成（step 达到 total_steps） | 收集结果，决定下一步，立即启动延伸实验 |
| **同一张卡出现 ≥2 个 compute-app PID（抢卡）** | 见下方「抢卡处置」——**不要条件反射去 kill** |

### 「GPU 空闲」的判据（本文件此前只有这个词、没有定义）

一张卡算空闲，必须**同时**满足：`memory.used` 接近 0、`utilization.gpu` 为 0、
**且该卡在 `--query-compute-apps` 里没有 PID**。三者缺一不可 —— 一个刚被 kill 的进程会留下
`0% util` 但显存未释放，而一个 model-load 中的进程会占着显存却报 0% util。

**但「有空闲卡」≠「必须马上塞任务」。** 先问三个问题，顺序不能颠倒：

1. **paperC / proposal 有没有待跑的？** 判据是「它们是不是真的没活了」，**不是「哪台卡空了」**
   （2026-08-12 我按「卡空了」把降级的 Paper B resume 塞进去，同期 proposal 的下一棒就在手上没接）。
2. **这些卡是不是对的架构？** LOCAL/`.212` = **sm_100**（B200），`.73/.82/.104` = **sm_90**（H20）。
   同口径续跑 / 同 harness 复现**必须同架构**，否则数字不可比。
   > 2026-08-14 我把 16 张 H20 记成「给 union-9 agent 预留」，而那个任务的归档 arm 是 sm_100 上打的，
   > 那些卡**从一开始就不合法**。预留一个架构不对的节点等于没预留。
3. **有没有别的 agent 已经拿了这些卡？** 派活前先看有没有在跑的 node-owning agent，
   两个 agent 同占一节点会 OOM（2026-08-08 毁了 4/5 rung）。

### 抢卡处置（≥2 个 PID 共占一卡）

**先量，再决定，不要默认 kill。** 步骤：

1. **量拖慢**：对被影响的训练取两次 iter 号（间隔 ≥120 s）算 `Δt/Δiter`，与它自己的单跑基线比。
2. **量剩余代价**：看抢卡方还剩多久（它自己的 progress 条），算「拖慢 × 剩余时间」的**上界**。
3. **判谁会先 OOM**：还在增长分配的那一方是受害者。稳态训练 + `expandable_segments:True` 通常不是它。
4. **看崩了赔多少**：`eval_interval` / `save_every` 决定最坏损失。若 ckpt 密且 resume 路径已验证，容忍度大得多。

若上界小（例如 <0.5 h）且抢卡方即将结束 → **让它跑完**，在报告里写明实测倍数，
**不要声称「共享是免费的」**。若上界大或它刚开始 → kill 抢卡方（**用 PID + `setsid` + `kill -9`，绝不 `pkill -f`**，
且 kill 前 `grep -v eval` 排除 eval 进程，否则会误杀正在收尾的 eval 丢数字）。


### ★写代码 + 启动新实验授权（2026-06-21 用户指令）

**heartbeat 可自主写代码 + 启动新实验，无需 emit needs_code 等审批。** GPU 空闲时即使没有现成 PENDING 任务，只要能据当前结论推断有价值的下一步，就直接派 `/coder`（或 general-purpose，`model="opus"`，`run_in_background=true`）写代码 + 跑多卡实验，把空闲卡用起来。代码改动遵守 CODEBUDDY.md（committer=LiuHanzuo，不加 AI 署名，`git add <具体文件>`）。完成后落账 + emit train_done。**仅重大方向切换/架构重构仍 emit needs_code 等主会话。绝不允许有空闲卡却只报 HEARTBEAT_OK 不行动。**

---

## Step 2.5：检查 GPU 监控前端（2026-06-20 新增，每次 heartbeat 必查）

监控前端是给用户实时看 **5 节点**显存/利用率/功耗+当前运行任务的网页，跑在**本机 8088 端口**。

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
  3. 等约 50s（首轮采集要 ssh 4 台远程，较慢）后再 curl 确认 200，并确认 `/api/data` 的 **`latest`** 键里 5 个节点各 8 卡。
     > ⚠️ GPU 数据在 `d['latest']`，**不在 `d['nodes']`**（后者只有 id+label）。
     > 2026-08-14 我按 `nodes` 解析，得到「5 节点全部 gpus=0」，差点把一个健康的 server 报成故障。
- 报告里加一行：`- Monitor: http200 OK` 或 `restarted`。
- ⚠️ 监控 server 靠 `sshpass` 拉远程数据。**`sshpass` 缺失时它会静默把远程节点报成空**，不是节点真的空。
- ⚠️ `NODES` 表在 `monitor/gpu_monitor_server.py`，**它也会过期**：2026-08-14 它还把 `.21` 当远程节点，
  于是 monitor ssh 到自己、把同一节点的 8 卡数了两遍，而真正新增的 `.212` 没有面板。改集群后要同步改它。

---

## Step 3：报告格式

**每个节点一行，必须含「GPU 实测」和「进度」两类信息**——只报 step 不报 GPU，等于没做 Step 1a：

```
## HEARTBEAT [YYYY-MM-DD HH:MM GMT+8]
- LOCAL(=.21): <run> step X/Y, loss=Z, 8x<mem>GB @<util>%, healthy
- .212:        <run> step X/Y, loss=Z, 8x<mem>GB @<util>%, healthy
- .73:         idle 8x0 MiB  /  <run> ...
- .82:         idle 8x0 MiB  /  <run> ...
- .104:        <run> step X/Y, ppl=Z, 8x<mem>GB @<util>%, healthy
- Monitor: http200 OK / restarted
- Action: none / <描述操作>
```

若某卡被多方共占，必须显式写出来（例如 `GPUs 0-3 shared with <什么>, measured 1.32x slowdown`），
**不能只报总显存把共享藏起来**。

速率若与该 run 自己的基线偏离 >10%，报出实测值 + 基线 + 倍数，不要只说「healthy」。

最后追加一行到 `status/TRAINER_ACTIVITY.jsonl`，并在启动/kill 任务时同步更新 `status/GPU_STATUS.md`。

---

## 资源备忘

> ⚠️ **2026-08-14 重写。** 此前这一节列的是 `28.59.80.196` / 两台 H800 / `28.89.16.18` / `28.88.184.53` /
> `28.48.7.53` —— **6 个 IP 全部已不在集群**，且 Step 1 点名的 4 个 `prepend_all_*.log`
> **全部不存在于盘上**。写死的节点表和 log 名会静默返回空，读起来像「一切正常」。
> **每次发现本节与实测不符，就地改本文件。**

**当前 5 节点 = 40 卡，⚠️ 分属两个物理盘：**

| 节点 | 硬件 | 盘 | 密码文件 | Python |
|---|---|---|---|---|
| **LOCAL** = `28.89.19.21` | 8×**B200** sm_100 183GB | wzc1 | —（本地） | `/opt/conda/envs/torch-base/bin/python` |
| **`.212`** = `28.89.18.212` | 8×**B200** sm_100 192GB | wzc1（**与 LOCAL 同物理盘**） | `configs/password_b200_18212.txt` | 同上 |
| **`.73`** = `28.85.35.73` | 8×H20 sm_90 97.8GB | zwfy6 | `configs/password_h20_853573.txt` | 同上 |
| **`.82`** = `28.82.250.82` | 8×H20 sm_90 | zwfy6 | `configs/password_h20_82250.txt` | 同上 |
| **`.104`** = `28.83.24.104` | 8×H20 sm_90 | zwfy6 | `configs/password_h20_24104.txt` | 同上 |

- **`nvidia-smi` 把 LOCAL/`.212` 的 name 显示成 `NVIDIA L20A`，那只是显示 bug，真实硬件是 B200。**
  判代际看 `capability`（sm_100）+ SM 数（148），不看 name。
- **SSH：一律省略 `-p`**（全局 `ssh_config` 已设 `Port 36000`）。写 `-p 22` 会 `Permission denied`。
  ```bash
  sshpass -f <pwfile> ssh -o StrictHostKeyChecking=no -o ConnectTimeout=12 \
    -o PreferredAuthentications=password root@<IP> "<cmd>"
  ```
  > ⚠️ `sshpass` 曾被节点重启抹掉（2026-08-14）。若 `which sshpass` 为空：`yum install -y sshpass`。
- **两盘不共享**：wzc1 = LOCAL + `.212`（这两台之间**无需 scp**）；zwfy6 = `.73/.82/.104`（**独立 checkout，commit 常落后**）。
  跨盘一律 `scp -O` + 核 hash。**「文件不存在」在两个盘都搜过前不成立。**
- 工作目录：wzc1 `/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/`；
  zwfy6 `/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/`
- ⚠️ **节点重启会重置 conda env**（2026-08-14 实测被剥到只剩 torch+numpy）。跑训练前先验 import 链，别等烧了 GPU 才发现。
- ⚠️ **`29.162.226.120`（dllm）已归还，绝不连。**

**当前活跃训练**：**不要信本节的静态列表**——用 `head -40 status/GPU_STATUS.md` +
`ls -t logs/*.log | head` 现查。这里只记长期基线：`.104` 的 paperC heal 是 5.74 s/step、
LOCAL/`.212` 的 SparseForge 臂是 45-48 s/it（单跑）。

