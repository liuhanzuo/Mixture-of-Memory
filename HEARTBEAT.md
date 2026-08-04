# HEARTBEAT.md — 自动监控操作手册

每 30 分钟触发一次。目标：监控实验状态，发现问题，**像 main agent 一样自主行动**。

---

## ⚡⚡ 当前阶段 + 节点 roster（2026-07-13 更新，最高优先；★改这里不用重启 cron★）

> **用户 2026-07-13 指令**：heartbeat cron 只指向本文件；易变的节点清单/待办写在这里，改动无需重启 cron 任务。**每轮 heartbeat 第一件事读本块 + `status/SESSION_HANDOFF.md` + `status/QCMEM_AUTONOMOUS_AGENDA.md`。**

### ★ 当前节点 roster（权威，QCMem = 5 节点 40 卡；2026-08-04 更新）
sshpass 前先 `export PATH=/opt/conda/bin:$PATH`。**全部共享同一 wzc1 项目盘 `/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/`**（本机与 .252 是 B200，真共享该盘、互相无需 rsync；.73/.82/.104 是 H20，也在该路径下跑训练/eval）。

| # | 节点 | IP:端口 | 硬件 | 密码文件 | Python |
|---|------|---------|------|---------|--------|
| 1 | 本机 (local, wzc1) | 本地直连 | 8×L20A（B200 级，183GB/卡） | — | `.venv/bin/python`（torch2.10+cu128，支持 L20A sm_100） |
| 2 | .252 | `28.89.19.252` :22 | 8×B200 | configs/password_b200_19252.txt | `.venv/bin/python` |
| 3 | .73 | `28.85.35.73` :36000 | 8×H20（97.8GB） | configs/password_h20_853573.txt | `/opt/conda/envs/torch-base/bin/python` |
| 4 | .82 | `28.82.250.82` :36000 | 8×H20 | configs/password_h20_82250.txt | `/opt/conda/envs/torch-base/bin/python` |
| 5 | .104 | `28.83.24.104` :36000 | 8×H20 | configs/password_h20_24104.txt | `/opt/conda/envs/torch-base/bin/python` |

- **SSH 通式**：`sshpass -f <密码文件> ssh -o StrictHostKeyChecking=no -o ConnectTimeout=12 -o PreferredAuthentications=password [-p 36000] root@<IP>`（H20 三台 .73/.82/.104 加 `-p 36000`；.252 走默认 22 端口）。密码只见对应 `configs/password*.txt`（含末尾逗号是密码的一部分，用 `sshpass -f`，不要手写展开）。
- ⚠️ **SSH recipe（2026-07-13 修，重要）**：本会话 conda 装过 texlive → shell snapshot 带进了 `LD_LIBRARY_PATH=/opt/conda/lib`，会让系统 `ssh` 加载 conda libcrypto 与 `/usr/lib64/libk5crypto.so.3` 冲突（`undefined symbol EVP_KDF_ctrl`）。**正确连法：`unset LD_LIBRARY_PATH` + 全路径 `/usr/bin/ssh` + sshpass 用 `/opt/conda/bin/sshpass`，且不要 `export PATH=/opt/conda/bin`。** 例：`unset LD_LIBRARY_PATH; /opt/conda/bin/sshpass -f configs/password_h20_853573.txt /usr/bin/ssh -o StrictHostKeyChecking=no -o PreferredAuthentications=password root@28.85.35.73 '<cmd>'`
- ⚠️ **★29.162.226.120 dllm 已归还，绝不连。** 环境 reset 后 `/etc/ssh/ssh_config` 全局 `Port 36000`：连 36000 端口节点直接连（不加 -p）；连 22 端口节点才要 `-F /dev/null`。

### ★ 两条铁律（最高优先）
1. **GPU 绝不空转**：判空卡用 `nvidia-smi -i K --query-compute-apps=pid | wc -l`（数进程，**非显存**——MoE load / model inject 期 GPU 0GB 但进程活，避误判堆叠）。发现空卡**立即填**（推进 TaskList column 或 `status/QCMEM_AUTONOMOUS_AGENDA.md` §1 四方向）。
2. **每结论查全证据 + 官方判分**（babilong=`TASK_LABELS`+`compare_answers` **禁 re.search**；RULER=`string_match`）+ 真实 `date` + util 低先查功耗/log 增长再判卡死。

### ★★★ EVAL 统一协议（2026-07-17 用户指令，最高优先，覆盖旧的 per-task selector）
- **所有 QCMem eval（RULER / BABILong / LongBench / LoCoMo / vs-Dense，所有 scale、所有 j、所有 task）统一用 `selector=iter_bm25`。** 不再用 bm25 单遍或 per-task 混选。启动脚本传 `SELECTOR=iter_bm25`（taskpool）或 `--selector iter_bm25`（单 cell）。
- **所有 benchmark + 所有 baseline（含 MemoryLLM / HCache / KV-Direct / Dense）都用同一配置测（2026-07-17 用户加强），保证可比：**
  - **chat template + no-think**：`--use_chat_template`，enable_thinking 默认 False。QCMem 生成边界 no-think 前缀已由 `c056a6d` 修好。
  - **QCMem selector = iter_bm25**（见上）。
  - **HCache / KV-Direct / Dense** 走 `eval_ruler_qcmem.py --baseline ...` / `eval_qcmem_babilong.py`（已有 chat 旗标）→ 直接传 `--use_chat_template` 重跑。
  - **⚠️ MemoryLLM = `YuWangX/memoryllm-8b-chat` = Llama-based**：无 thinking 模式（enable_thinking 仅 Qwen3）→ no-think 天然满足；无 bm25 selector（内部 stateful memory）→ iter_bm25 不适用。「同配置」对它 = 用 chat template（它本就是 -chat 模型）。且需专用 env（`../MemoryLLM-source` + `external/memoryllm_venv`/ported transformers），只在特定节点能跑 → 单独 track。
- 之前用 bm25 或非-chat 跑的结果**作废，需重跑**（含 8B-adapter BABILong bm25=62.2 / iter_bm25=57.1）。

### 当前在跑（每轮核对实测，随状态改这里）（2026-08-04 更新）

**★★ 当前主线（2026-07-16 用户定）：Paper A(QCMem) 全 scale benchmark 已基本完成并 push（MoM `8176949` + COMem 论文 `196d4de`）；主力转 Paper B = OLMo-2 base 剪层-heal（纠正原 instruct-continue-train 错误）。★live 状态以 `status/SESSION_HANDOFF.md` 顶部快照为准，本块只给方向骨架。**

- **当前节点分配（示例，实测为准；live 详情见 `status/SESSION_HANDOFF.md`）**：
  - 本机 (8×L20A)：armA。
  - .252 (8×B200)：armB。
  - .73 + .82（16×H20 多机 DDP，TCP over bond1，IB 挂时 `NCCL_IB_DISABLE=1`）：P1.3。
  - .104 (8×H20)：P1.10。
- **ckpt 轮转**：cron `4ec42903`（:47）清 wzc1 旧 OLMo ckpt 防盘满（≠heartbeat，勿删）。
- **★论文方向**：单一 insight「前几层已压缩语义」→ Paper A=QCMem(已 benchmark+push)、Paper B=OLMo-2 剪层-heal(在跑)、Paper C=蒸馏。
- **QCMem 收尾**：LongMemEval/∞Bench/HELMET 待接 API-judge harness（用户 2026-07-16 定暂不评）；LoCoMo 报 F1。

### 红线：所有训练 `--babilong_mix_fraction 0`；泄漏 ckpt（b50/b100/P2/c1024/旧 b25）完全不碰、不引用其分数。

### 🚀 效率三条铁律（2026-06-28 用户指令）

1. **有需要探索的方向 → 直接派 Workflow 多 agent 并行探索，提高效率。** 不要一个个串行派 subagent。需要分解问题、覆盖多个子方向、或对比多方案时，用 Workflow 编排（fan-out 调研 → 验证 → 综合）。单点查证才用单个 Agent。

2. **空闲节点要善用，但讲轻重缓急（不是机械填满每张卡）。** 每轮巡查显式列出 5 个节点占用（`[IP] 忙X/8 跑什么`），然后按优先级判断空卡怎么用：
   - **有明确待跑任务时（最优先）**：新 ckpt 落盘要 eval、缺的长度档/step、待验证的 probe → 用空卡立刻跑。
   - **没有待跑任务但有空节点时**：空闲是用来往前推的，几个用法（自己判断哪个最值）：
     - (a) **robustness 保证**：多 seed / 更大样本量复核已得结论、回归测已修的 bug、长档/边界条件压测、ckpt 完整性校验，避免结论建在脆弱数据上。
     - (b) **开新实验 / 探索主方向**：派 Workflow 多 agent 并行分析「当前主方向下一步该做什么、哪里能改进、有什么新实验值得开」，产出设计后直接执行。
     - (c) 写/改代码推进（如实现待验证的机制、修工具链坑）。
   - **不必为填而填**：不要塞低价值的零碎任务（如已知结论的冗余档）只为占满。判断这张卡这一轮拿来做什么最值。
   - **跨盘约束**：A ckpt 在 diskB；diskA 节点（本机/.196）需 ckpt 传到 diskA 共享路径；L20A（wzc1）需单独 scp（含 adapter_config，曾漏传）。
   - status 里写清：5 个 IP 各在跑什么 / 空的为什么空（无任务则说明在推进什么探索）。

3. **关键实验 / 实验不多时 → 多节点并行加速（如 2 IP 16 卡）。** 当某个实验是关键判据、或当前待跑实验很少（多数节点空）时，不要让单个实验只占一台机慢慢跑——把它拆到多个节点并行（eval 按 shard 分到 2+ 节点；训练用多机 DDP，CODEBUDDY.md:221 的 `torchrun --nnodes N --rdzv_backend c10d` 配方，2 节点 16 卡可将训练提速近 2×）。判断点：实验关键且慢 + 有空闲节点 = 并行。注意跨盘 ckpt/数据可达性（同盘节点优先并行，跨盘需先同步）。

   **★★ 少实验就合成多节点（2026-07-01 用户指令，强化上条）：当【待跑训练 ≤ 3 个】时，不要一实验一节点各自慢跑，而是把【同盘的两个节点合成一个 16 卡节点】做多机 DDP 加速单个实验。**
   - **同 wzc1 盘可合成 16 卡（合并前提=共享盘）**：当前 5 节点全部共享 wzc1 盘 → 任两台都能合成 16 卡多机 DDP（代码/ckpt/数据免同步）。常用组合：H20 三台（.73/.82/.104）任取两台，或本机 + .252 两台 B200 级。
   - **配方**（现成 2node 脚本参考 `scripts/launch_landmark_S2_dolmino_2node.sh` + `run_landmark_S2_node.sh`）：两节点各跑一次 `torchrun --nnodes 2 --node_rank {0/1} --nproc_per_node 8 --rdzv_backend c10d --rdzv_endpoint <MASTER内网IP>:<PORT>`，master 用内网 IP，NCCL 注意 bond1 + IB disabled（见 run_landmark_S2_node.sh）。
   - **决策**：≤3 训练 + 有同盘空节点 → 合成 16 卡跑最关键那个（训练提速近 2×，或让慢的 16k eval 分片到 16 卡）。>3 实验或都是独立小实验 → 一实验一节点铺开。判断"这一轮 16 卡合起来加速一个，还是分开跑多个"哪个总产出高。

---

## ⚡ 执行计划书（2026-05-11 起，已被上方「当前阶段」取代，保留作参考）

**~~首要任务：对照 `status/H_V2_PLAN.md` 推进 H-series v2 训练和基线复现。~~（过时）**

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

> **当前权威集群 = 顶部「★ 当前节点 roster」的 5 节点 40 卡**（本机 + .252 两台 B200 级 + .73/.82/.104 三台 H20），全部共享 wzc1 项目盘 `/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/`，代码/ckpt/数据免同步。
>
> 旧的「盘A/盘B 双 ceph」「b200-1..4 / b200-5..8 / h20-1..4 / 回归 H20」拓扑已全部回收，**不要再 ssh 探测任何不在 roster 里的 IP**（含所有 `30.203.*` H800、`28.59.80.196`、`28.49.*`、`28.48.7.53`、`28.58.245.174`、`28.89.16/17/18.*`、`28.89.19.134`、`28.89.20.*` 等）。**★29.162.226.120 dllm 已归还，绝不连。**

远程 4 台节点 SSH 检查模板（密码文件见 roster；H20 三台 .73/.82/.104 加 `-p 36000`，.252 走默认 22 端口）：

```bash
# H20（.73 / .82 / .104，端口 36000）
sshpass -f configs/password_h20_853573.txt ssh -o StrictHostKeyChecking=no \
  -o ConnectTimeout=12 -o PreferredAuthentications=password -p 36000 root@28.85.35.73 \
  "nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader; \
   tail -5 <log_path> 2>/dev/null"

# .252（B200，默认 22 端口）
sshpass -f configs/password_b200_19252.txt ssh -o StrictHostKeyChecking=no \
  -o ConnectTimeout=12 -o PreferredAuthentications=password root@28.89.19.252 "<cmd>"
```

（SSH recipe 坑：conda texlive 污染 `LD_LIBRARY_PATH` 导致 ssh libcrypto 冲突时，见 roster 里的 `unset LD_LIBRARY_PATH` + `/usr/bin/ssh` + `/opt/conda/bin/sshpass` 修法。）

**节点 SSH 失败时**：不要立即升级。在 `status/TRAINER_ACTIVITY.jsonl` 标记 `ssh_timeout`，连续 3 次（约 60 分钟）失败才视为节点不可用，再更新状态。

**所有节点共享 wzc1 项目盘**：项目根 = `/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/`，代码/模型/数据免同步。

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

## 告警上报通道（2026-06-09 用户指令）

**独立 heartbeat 进程（`scripts/heartbeat_daemon.sh` → `scripts/heartbeat_cron.sh` → fresh `codebuddy --print "/heartbeat"`）是孤立的：零对话历史、token 有界、很省钱，但发现重要事件时没有通道通知主会话。**

为此新增一个 alert 上报通道：heartbeat **常规巡检保持静默**（只写 `status/TRAINER_ACTIVITY.jsonl` 流水），**只有下面三类事件**才往 `status/HEARTBEAT_ALERTS.jsonl` 追加一条结构化告警。主会话用独立 cron probe 读取 `ack:false` 的行决定是否唤醒处理。

### 何时写 alert（且仅这三类）

| event_class | severity | 触发条件 | detail 内容 |
|-------------|----------|---------|------------|
| `train_done` | `info` | 某 run 到达 total_steps 完成 / 出 final ckpt | exp 名、ckpt 路径、最终指标、建议动作（如"可起 eval"）|
| `train_anomaly` | `critical` | run 崩溃 / 连续两次巡检无新 step（卡死）/ loss NaN / PPL>100 等 | 诊断证据 + heartbeat 已采取的自主动作（如已 kill 并重启 / 已迁移节点）|
| `needs_code` | `warning` | heartbeat 判断需写新代码/改配置，但**超出 heartbeat 自主权限**、需主会话决策 | 需要做什么、为什么超出 heartbeat 自主权限 |

**常规健康巡检（GPU 正常训练 / 空闲已被自主调度 / stale 状态文件已自主刷新 / researcher 已确认的延伸实验已自主启动）→ 不写 alert，只写 TRAINER_ACTIVITY.jsonl 流水。** 这是省钱的关键：alert 是给主会话 probe 的稀缺唤醒信号，不是流水账。

### 去重规则（必须）

`id` 是去重键，**必须是稳定 key**，使同一事件被多次 heartbeat 观测时只产生一条 alert：
- `train_done:<exp_name>:<step>`
- `train_anomaly:<exp_name>:<step_or_symptom>`（symptom 如 `nan` / `stall` / `crash`）
- `needs_code:<short_slug>`

写入前先 grep 该 `id` 是否已在文件里，已存在则跳过。**推荐用辅助脚本 `scripts/hb_emit_alert.sh`，它自动做 fixed-string 去重 + 安全 JSON 转义**：

```bash
scripts/hb_emit_alert.sh \
  --event-class train_done \
  --severity   info \
  --id         "train_done:dolmino_bugfix_slotq_t2h:2000" \
  --summary    "dolmino_bugfix_slotq_t2h 训练完成 (step 2000)" \
  --detail     "ckpt=outputs/dolmino_bugfix_slotq_t2h/final; lm=2.31; 建议起离线 BABILong eval" \
  --node       local
```

脚本退出码：`0` = 已写入或因重复跳过（都算成功），`2` = 参数错误。`bash -n` 通过，去重已自测。

### alert JSONL 格式 spec（`status/HEARTBEAT_ALERTS.jsonl`，append-only，每行一个 JSON）

```json
{"ts":"2026-06-09T15:00:00+08:00","id":"train_done:<exp>:<step>","severity":"info|warning|critical","event_class":"train_done|train_anomaly|needs_code","node":"<node/run>","summary":"<一行人类可读>","detail":"<run/step/loss/log路径/建议动作>","ack":false}
```

### ack 字段读写约定（写给 main reader）

- **`ack:false` 由写入端（heartbeat / `hb_emit_alert.sh`）写**：每条新 alert 恒为 `false`，表示"未被主会话处理"。
- **`ack:true` 由读取端（主会话 / probe reader）写**：主会话 probe 读到 `ack:false` 行、处理完后把它翻转为 `true`。
- **heartbeat 永不写 `ack:true`，也不修改任何已有行**——只 append 新行。主会话 reader 负责 ack 翻转（read→modify→write 整个文件，或按 id 定位行更新）。
- 文件若不存在，`hb_emit_alert.sh` 首次调用会自动 `touch` 创建（已初始化为空文件）。

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

**核心原则：当前 5 节点 40 卡（本机 + .252 + .73/.82/.104）。任何时刻空闲节点 ≥ 1 且有可做的事 = 浪费。**

Heartbeat 必须维护一个 **"节点占用表"**，每次检查时：

| 节点 | 当前实验 | 预计完成时间 |
|------|----------|------------|
| 本机 | ... | ... |
| .252 | ... | ... |

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
- .252 (28.89.19.252): [状态]
- .73/.82/.104 (H20): [状态]

### 活跃训练
- [实验名称]: step X/Y, loss Z, 健康/异常

### 待审批请求
- [request_id]: [内容] → 需要你决定

### 问题
- [如有]

### 结论
HEARTBEAT_OK  或  [需要关注的事项]
```
