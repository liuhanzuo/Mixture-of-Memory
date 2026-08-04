# Mixture-of-Memory — CodeBuddy Code 工作手册

## ⚡ 启动 / compact 后第一件事（2026-06-11 用户指令，最高优先级）

**每次新会话启动、或上下文被 compact / 丢失后，第一件事必须读 `status/SESSION_HANDOFF.md`** —— 它是当前研究状态的交接文档（一句话现状 + 核心认知 + 在跑实验 + 待办 + 运维坑）。读完它 + `status/RUN_REGISTRY.md` §3/§4 + `status/TRAINER_ACTIVITY.jsonl` 尾部，就能无缝接上之前在干什么，不要重走已证伪的方向。

**维护职责**：main agent 每当「方向切换 / 出新结论 / 在跑实验有重大变化」时，必须覆盖更新 `status/SESSION_HANDOFF.md` 的「当前快照」区（保持精简，旧结论沉淀到 RUN_REGISTRY），确保它始终反映最新状态——这样下一次 compact 后的 agent 才能接得上。

## 语言规则（2026-05-19 用户指令）

**只使用中文和英文交流，禁止使用韩语或其他语言。**

## 🖥️ 当前 GPU 集群（2026-08-04 更新，权威，覆盖旧记录）

**当前只有 5 个节点 = 40 卡，但 ⚠️ 分属【两个物理盘】，不是"全部共享 wzc1"。**

> **★★ 2026-08-04 实测纠正（旧文档「5 台全部共享 wzc1、互相无需 rsync」是错的，已让多个 agent 白跑，务必按本条执行）：**
> - **wzc1 盘 = 本机 LOCAL + .252**（两台 B200 级，真共享 `/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/`，互相无需 rsync）。
> - **zwfy6 盘 = .73 / .82 / .104**（三台 H20，真实 root = `/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/`）。它是**另一份独立 checkout、commit 常落后**（实测 `2d98c5a`，且非 local HEAD 的祖先）。
> - **陷阱 1（.73）**：`/apdcephfs_wzc1` 在 .73 上是**指向 zwfy6 的 symlink** —— wzc1 路径字符串"看着能用"，但物理盘不同于 LOCAL/.252，**写进去 LOCAL 看不到**。.73 的 PROJECT_ROOT 应写 zwfy6 路径。
> - **陷阱 2（.82）**：`/apdcephfs_wzc1` 在 .82 上**根本不存在**。
> - **跨盘搬运一律 `scp -O`**（.82 的 sftp subsystem 已坏，普通 `scp` 报 `subsystem request failed`），搬完核 md5/sha256。
> - **推论**：wzc1-only 的新脚本/新 ckpt 必须显式 `scp -O` 到 zwfy6 才能在三台 H20 上跑；「同盘合 16 卡多机 DDP」只在**同盘内**成立（LOCAL+.252，或 .73/.82/.104 任两台），**不可跨盘合并**。
> - **软件差异**：三台 H20 的 `.venv/bin/python` 已坏 → 用 `/opt/conda/envs/torch-base/bin/python`；**LOCAL 的 `.venv` 现也已无 torch**（2026-08-04 实测），同样改用 conda。**.82 未装 `bitsandbytes`** → `OPT=bnb8bit` 在 .82 不可用。

| # | 节点 | IP:端口 | 硬件 | 密码文件 | Python |
|---|------|---------|------|---------|--------|
| 1 | 本机 (local, wzc1) | 本地直连 | 8×L20A（B200 级，183GB/卡） | — | `.venv/bin/python`（torch2.10+cu128，支持 L20A sm_100） |
| 2 | .252 | `28.89.19.252` :22 | 8×B200 | `configs/password_b200_19252.txt` | `.venv/bin/python` |
| 3 | .73 | `28.85.35.73` :36000 | 8×H20（97.8GB） | `configs/password_h20_853573.txt` | `/opt/conda/envs/torch-base/bin/python` |
| 4 | .82 | `28.82.250.82` :36000 | 8×H20 | `configs/password_h20_82250.txt` | `/opt/conda/envs/torch-base/bin/python` |
| 5 | .104 | `28.83.24.104` :36000 | 8×H20 | `configs/password_h20_24104.txt` | `/opt/conda/envs/torch-base/bin/python` |

- **SSH 通式**：`sshpass -f <密码文件> ssh -o StrictHostKeyChecking=no -o ConnectTimeout=12 -o PreferredAuthentications=password [-p 36000] root@<IP>`（H20 三台 .73/.82/.104 加 `-p 36000`；.252 走默认 22 端口）。
- **⚠️ 两个物理盘，非全共享**：**wzc1** = LOCAL + .252（`/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/`）；**zwfy6** = .73/.82/.104（`/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/`，独立 checkout、commit 常落后）。**.73 上 `/apdcephfs_wzc1` 是指向 zwfy6 的 symlink；.82 上该路径不存在。** 跨盘一律 `scp -O` + 核 md5。详见顶部纠正条。
- **密码只见对应 `configs/password*.txt` 文件**（含末尾逗号是密码的一部分，用 `sshpass -f`，不要 `tr -d` 或手写展开）。
- ⚠️ **dllm 节点 `29.162.226.120` 已归还，绝不连。**
- ⚠️ **内联 BABILong eval 会导致 NCCL 崩溃**（2026-06-02 实测）：`quick_eval_babilong` 在 DDP 循环里做变长 greedy generation 会让各 rank desync → ALLREDUCE 等满 30min watchdog timeout → 整个 job SIGABRT。**训练时务必 `--eval_interval 0`**（launch 脚本已默认 `EVAL_INTERVAL=0`），eval 改为离线单独跑 checkpoint。

## Wandb 配置（2026-05-25）

```bash
export WANDB_API_KEY="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"
```

训练脚本默认 `--wandb_project mixture-of-memory`，启动训练时务必设置此环境变量。

## 外网代理配置（2026-05-18 用户提供）

访问 HuggingFace / arxiv / GitHub 等外网时，必须设置以下代理：

```bash
export http_proxy="http://hy-proxy.woa.com:3128"
export https_proxy="http://hy-proxy.woa.com:3128"
export all_proxy="http://hy-proxy.woa.com:3128"
export no_proxy="mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,localhost,127.0.0.1,.oa.com,.woa.com,.local"
```

> 注：`no_proxy` 已精简，完整版见 CODEBUDDY.md 历史。下载 HF 模型时也需要设置这些环境变量。

---

## Benchmark 结果汇总（2026-05-18）

**`status/BENCHMARK_RESULTS.md`** 是所有实验结果的永久记录文件，包含：
- 我们自己的实验结果（P8、v2、v2-base、plain baseline 等）
- 其他论文的参考数字（LM2、BABILong paper 等）
- 正在进行的实验（v6、v7、MemoryLLM eval）

**格式**: 10 task (qa1-qa10) × 7 lengths (0k-32k) × n=100 samples，babilong.metrics 口径

**heartbeat / 新实验完成后必须更新此文件**，把新结果追加进去，方便横向对比。

### ★ 标准 eval 方式（2026-06-13 用户指定，权威，覆盖旧的 LPT 静态预分配调度）

**所有 BABILong 离线 eval 必须用 `scripts/_eval_taskpool_2group.sh`（2-组 task-pool 动态调度）。**

调度范式:
- 8 GPU 分 **2 组**:GROUP0=GPU0-3,GROUP1=GPU4-7。
- 一个「任务」= (ckpt, task, length),如 qa1×16k,共 100 样本。
- 一个任务在**一组内 4 卡各跑 25 样本**(`--num_shards 4 --shard_index {0..3}`,样本 `[i::4]` 均分)。
- `{qa1,qa2,qa5} × {0k,1k,2k,4k,8k,16k,32k}`(× 多 ckpt)= 21+ 任务进**一个共享 pool**;哪组空闲就 `flock` 原子 pop 下一个 append 给它 → 动态负载均衡,最大化吞吐(长档 32k 慢任务不会拖死整组)。
- `score_nested_babilong.py` 把 4 个 `_shard{i}of4` CSV 求和合并回单 cell。

用法:
```bash
RUN_PREFIX=expXXX CKPT_FILES="path/step500.pt path/final.pt" \
CK_NAMES="expXXX_step500 expXXX_step1000" \
ADAPTER_CONFIG=outputs/expXXX/adapter_config.json \
[EXTRA_ARGS="--swa_eval_chunks 6"] \
PROJECT_ROOT=<节点root> PYTHON_BIN=<节点.venv或conda> \
setsid nohup bash scripts/_eval_taskpool_2group.sh >logs/...sched.out 2>&1 &
```
- diskB(.76/.249/B200)PYBIN 用 `.venv`,diskA(本机/.196)用 conda 或 .venv;PROJECT_ROOT 指本节点 root。
- 旧的 per-GPU LPT 静态预分配调度器(`_expR1c*_eval_sched.sh` 等)**已弃用**——它把任务静态切到 8 卡,长档 shard 会让某些卡空转。新 task-pool 动态补任务,无空转。

### `status/RUN_REGISTRY.md`（2026-06-05 新增）

**`status/RUN_REGISTRY.md` 是 mem_space 系列每个训练 run 的「配置 + 离线 BABILong 结果」横向对照总账。**

- 职责：记录每个 run 的关键超参（chunk_size、slot_dim、num_slots、route_aux、读路径、steps、节点、状态）+ 同口径 BABILong eval 结果（n=100，qa1/qa2/qa5 × 0k-32k，babilong.metrics）。
- 与 `BENCHMARK_RESULTS.md` 分工：BENCHMARK_RESULTS 含外部论文数字、是大杂烩；**RUN_REGISTRY 只记我们自己的 mem_space run，强调配置可复现 + 严格同口径横向对照**。
- **每启动一个新 run / 每跑完一次 eval，必须在 RUN_REGISTRY.md 追加或更新对应行**，便于快速回答"X 配置 vs Y 配置在 BABILong 上差多少"。

---

## 自主派发规则（2026-04-29 用户指令，2026-04-30 更新）

**派发 `/researcher` 和 `/coder` subagent 不需要用户审批，可随时自主执行。**

- heartbeat 发现需要调研或代码修改时，直接派 subagent，无需询问用户
- 这包括但不限于：Fix 方案实现、bug 修复、诊断代码添加、文献调研

**2026-04-30 用户新增授权：hyperparameter 更改 + researcher 确认的更改，无需用户审批，可自动执行。**

- `/researcher` 分析后给出的 hyperparameter 建议（如 temperature、learning rate、skrl_weight 等）→ **直接自动执行，无需等待用户确认**
- researcher 报告中明确标注 `confidence: high / very_high` 的代码或配置更改 → **直接自动执行**
- **仍需用户审批的情况**：启动全新方向的训练实验（非 ablation/fix 延伸）；涉及模型架构的重大重构

---

## 待完成任务系统（2026-05-01 用户指令）

**`status/PENDING_TASKS.md` 是任务看板，heartbeat 必须每次检查。**

规则：
1. 发现问题（训练完成、训练异常、新 insight）→ 立即写入 PENDING_TASKS.md
2. 待执行任务写入 `[PENDING]` 状态；正在执行写入 `[RUNNING]` 状态；完成移入 `[DONE]` 区
3. **heartbeat 如果发现 GPU 空闲 + 有 `[PENDING]` 任务 → 必须在同一 heartbeat 内执行任务**
4. 每次实验完成后，必须分析结果、写入下一步任务、**并立即启动（如果是 ablation/fix 延伸）**
5. **绝对不允许** 连续多次 heartbeat 报 "全部空闲, HEARTBEAT_OK" 而不采取行动
6. 当发现分析/调研/修改需求时，在 PENDING_TASKS.md 记录，确保不会遗忘
7. **每个 PENDING 任务必须标注 `auto_launch: true/false`**。`true` 的任务 heartbeat 必须自动启动
8. **实验完成后不能等用户回复。** 分析 → 决定下一步 → 立即执行，形成闭环（详见 HEARTBEAT.md "实验生命周期自动化"章节）

### 模型配置
- 主模型: GLM-5.1
- Heartbeat 模型: 同主模型，使用 GLM5.1->GLM5-turbo->GLM5->GLM4.7 fallback 顺序

### 多节点并行消融 (strongly encouraged)

遇到问题时，系统性地思考多个解决方案，在不同节点上并行验证：
- 例：beta gate 停滞 → 同时跑 beta_init={-2.0, -1.0, 0.0} 三个 arm
- 例：PPL 不理想 → 同时跑 {lr=1e-3, lr=1e-4, lr=5e-4} 三个 arm
- 使用 `/train` 的 `--matrix` 参数自动展开消融矩阵
- 所有消融结果汇总后，派 /researcher 分析哪个最优

### Heartbeat 自主决策规则

Heartbeat 可以自主做的（无需用户确认）：
- Kill PPL > 100 的训练
- Kill 连续 stalled 的训练（2次 heartbeat 确认）
- 清理已 crash 的训练
- 在空闲节点启动 PENDING_TASKS.md 中 auto_launch=true 的任务
- 派 /researcher 分析结果
- 派 /coder 修复 bug
- **启动 ablation/fix 延伸实验**（同算法改参数、同代码新数据、checkpoint 评估等）
- **实验完成后自动决定下一步并立即执行**（分析→写任务→启动，形成闭环）

Heartbeat 不能自主做的：
- Kill healthy 的训练
- 修改 hyperparameters 或代码（但 researcher 确认 + confidence: high 的除外）

**方向切换规则**：`/researcher` 分析后建议切换方向，且报告标注 confidence: high/very_high → heartbeat 可以自主执行新方向，无需用户审批。没有 researcher 确认的方向切换仍需用户审批。

**绝对不允许** GPU 全部空闲 + 有 pending 任务时只输出 HEARTBEAT_OK。

---

## Git 提交规则

**禁止在 commit message 中加 `Co-Authored-By: Claude` 或任何 Anthropic/Claude 相关的 trailer。**
git commit 只包含实际修改内容的描述，不附加任何 AI 署名行。

## 多节点 GPU 利用规则（2026-04-27 用户指令）

**当只有一个训练任务时，必须尽量使用多个节点来最大化效率。**

- 当前有 5 个节点 = 40 卡（本机 + .252 两台 B200 级 + .73/.82/.104 三台 H20），闲置节点是浪费
- 单任务训练时，优先考虑 **多机多卡 DDP**（torchrun `--nnodes N --rdzv_backend c10d --rdzv_endpoint <master_ip>:29500`）
- 若多机配置复杂（数据 sharding / 脚本不支持），退而求其次使用 **gradient accumulation** 提升单节点有效 batch size，目标 GPU mem bandwidth ≥ 70%
- **GPU mem bandwidth < 50% 视为欠载**，必须在 heartbeat 报告中标注 WARNING 并提出扩展方案
- kill 当前低效 run 并以更优配置重启是被允许的，不需要额外审批（用户指令优先）

### ★★ 少实验就合成多节点 16 卡（2026-07-01 用户指令，强化）

**当【待跑训练 ≤ 3 个】时，不要一实验一节点各自慢跑，而是把【同盘的两个节点合成一个 16 卡节点】做多机 DDP 加速单个关键实验。**

- **合成 16 卡只能【同盘内】（合并前提=共享 FS）**：⚠️ 5 节点**分属两盘**，**不可跨盘合并**。合法组合：**LOCAL + .252**（wzc1），或 **.73/.82/.104 任取两台**（zwfy6）。
- **配方**：现成 2-node 脚本 `scripts/launch_landmark_S2_dolmino_2node.sh` + `scripts/run_landmark_S2_node.sh`；两节点各跑一次 `torchrun --nnodes 2 --node_rank {0/1} --nproc_per_node 8 --rdzv_backend c10d --rdzv_endpoint <MASTER内网IP>:<PORT>`。NCCL 注意 bond1 + IB disabled（见 run_landmark_S2_node.sh verified recipe）。
- **决策准则**：≤3 训练 + 有同盘空节点 → 合成 16 卡跑最关键那个（训练提速近 2×；慢的 16k eval 也可分片到 16 卡）。>3 独立实验 → 一实验一节点铺开。每轮判断"16 卡合起来加速一个 vs 分开跑多个"哪个总产出高。

## 项目概述

研究方向：**固定大小 memory buffer 压缩长上下文**。
核心问题：如何让 7B/8B LLM 在有限 KV budget 下处理超长序列。

代码根目录：`/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/`

### 已完成的工作
- Sparse Memory (MAG): EMA-based memory bank，128/256 slots，PPL 退化 ~20%，**已放弃**
- Selective Context (token pruning): PPL 退化 500-5000%，**已放弃**
- RMT v3-v10: 训练后 generation 退化（重复 pattern），**已放弃**
- DMS 8x: 800 steps 完成，checkpoint 在 `outputs/dms_8x/final`，待评估
- Slot Memory Stage 2: 最后已知健康（2026-04-19），checkpoint 待确认

### 当前方向（2026-04-23 pivot）
根据 researcher 报告，下一步重点：
1. **Attention Matching** (arXiv:2602.16284) — 50x 无训练 latent-space 压缩
2. **4-bit KV Quantization** — 快速基线
3. **Heavy Hitter** (Cold-Compress) — token-space 最强基线

---

## Agent 架构

本项目使用 CodeBuddy Code slash commands 实现多 agent 协作，对应 OpenClaw 的 trainer/researcher/coder 角色。

### Slash Commands

| 命令 | 角色 | 触发方式 |
|------|------|---------|
| `/heartbeat` | 主监控 | 每 20 分钟自动 (CronCreate) 或手动 |
| `/trainer` | GPU 训练管理 | 手动派发，管理本地 H20 + 远程 B200 |
| `/researcher` | 文献调研 + 实验分析 | 手动派发或 heartbeat 触发 |
| `/coder` | 代码实现 + 修 bug | 手动派发，使用 opus 模型 |
| `/status` | 快速状态总览 | 随时手动 |
| `/approve` | 批准 trainer 请求 | 手动，响应 trainer 的 REQUESTS |

---

## 关键文件路径

### 状态文件（机器可读）
```
status/GPU_STATUS.md           — ★两节点 GPU 实时台账（2026-07-10 用户指令）
status/gpu_runs.jsonl          — 训练运行历史（append-only）
status/RUNNING_EXPERIMENTS.json — 运行中实验索引（Read→modify→Write，非 append-only）
status/TRAINER_ACTIVE.md       — 当前活跃训练（write 覆盖，禁止 edit）
status/TRAINER_REQUESTS.jsonl  — trainer 发给 main 的审批请求
status/TRAINER_APPROVALS.jsonl — main 的审批决定
status/RESEARCHER_REPORTS.jsonl — researcher 的研究结论
status/ISSUES.jsonl            — 问题追踪
status/PENDING_TASKS.md        — 待完成任务看板（heartbeat 必检）
```

### ★ GPU_STATUS.md 维护规则（2026-07-10 用户指令）
**每次启动或 kill GPU 任务时，必须同步更新 `status/GPU_STATUS.md`**（哪节点哪张卡跑什么、起始时间、预计时长）。
- heartbeat 每轮：先读 GPU_STATUS.md → 对照 `nvidia-smi` 实测 → 若"台账说在跑但实际空闲"=任务已完成/崩溃 → 立即补卡（铁律1）+ 更新台账。
- 目的：避免反复"查 GPU→发现空转→补卡"的低效；台账=两节点单一事实来源。
- ⚠️ 教训：babilong 低档(0k-8k)/RULER 短档极快(几分钟)跑完即空转——别用短任务填卡，优先耐跑任务(LoCoMo/长档/训练)，短任务成批投。


### 日志文件（人类可读）
```
UPDATELOG.md                   — 所有重大操作的时序日志
RESEARCH_LITERATURE.md         — 文献整理（持续更新）
HEARTBEAT.md                   — heartbeat 操作手册
ops/research_notes/            — researcher 每次研究的详细笔记
```

### 代码结构
```
src/memory/
  sparse_memory/     — 旧方案（MAG）
  sparse/            — 新 sparse 实现
  slot_memory/       — Slot Memory 压缩
  slot/              — slot compressor
  dms/               — Dynamic Memory Sparsification
  rmt/               — Recurrent Memory Transformer
  l1/ l2/ l3/ mag/  — 分层记忆模块
scripts/             — 训练/评估脚本
configs/
  b200_cluster.ini   — 远程节点 SSH 配置
  remote_experiments.json — 远程实验状态追踪
```

---

## 计算资源

**当前集群（2026-08-04 更新）：5 个节点 = 40 卡，⚠️ 分属【两个物理盘】——wzc1（LOCAL + .252）与 zwfy6（.73/.82/.104），跨盘需 `scp -O`，不可跨盘合成多机 DDP。详见顶部「🖥️ 当前 GPU 集群」表与其纠正条。**

| # | 节点 | 硬件 | Python |
|---|------|------|--------|
| 1 | 本机 (local, wzc1) | 8×L20A（B200 级，183GB/卡） | `.venv/bin/python` |
| 2 | .252 (`28.89.19.252` :22) | 8×B200 | `.venv/bin/python` |
| 3 | .73 (`28.85.35.73` :36000) | 8×H20（97.8GB） | `/opt/conda/envs/torch-base/bin/python` |
| 4 | .82 (`28.82.250.82` :36000) | 8×H20 | `/opt/conda/envs/torch-base/bin/python` |
| 5 | .104 (`28.83.24.104` :36000) | 8×H20 | `/opt/conda/envs/torch-base/bin/python` |

- 本机 + .252 是 B200 级（183GB/卡），显存大，适合重型 8B+memory 训练；.73/.82/.104 是 H20（97.8GB），1B 训练无压力，8B+memory 需 gradient_checkpointing。
- SSH / 密码文件见顶部「🖥️ 当前 GPU 集群」表与 SSH 通式（H20 三台加 `-p 36000`；.252 走默认 22 端口）。
- ⚠️ **dllm 节点 `29.162.226.120` 已归还，绝不连。**

### 集群间分配指南

| 任务类型 | 推荐节点 |
|---------|----------|
| 主训练 / baseline / eval / inference | 任一节点（5 台共享 wzc1 盘） |
| 重型 8B+memory 训练 | **本机 或 .252**（B200 级 183 GiB 空间更大） |
| 多机 16 卡加速单个关键实验 | H20 三台（.73/.82/.104）任取两台，或本机 + .252 |

---

## 关键规则（Red Lines）

4. **新实验启动前必须确认 GPU 状态**（nvidia-smi 验证，无 orphan 进程）
5. **同一节点同时只跑一个 serious 实验**（8-GPU）；**不同节点可以并行跑不同实验**
6. **heartbeat 发现问题后可以自动派 `/researcher` 和 `/coder` 解决，无需用户审批**（2026-04-28 用户授权）
   - 发现代码 bug / 训练异常 / 检索退化等问题 → 直接派 `/researcher` 分析根因，派 `/coder` 修复
   - 自动派发的 subagent 完成后，main 必须落账（ISSUES.jsonl / UPDATELOG.md / TRAINER_ACTIVITY.jsonl）
   - **2026-04-30 更新**：hyperparameter 更改 + researcher 确认的更改 → **无需用户审批，自动执行**（见"自主派发规则"）
7. **发现显著 bug 时，可自主 kill 运行中实验 + 调研 + 启动新实验，无需用户审批**（2026-04-28 用户授权）
   - 判断标准："显著 bug" = 训练进程运行但核心功能完全失效（如 K_sel routing 永久退化、NIAH gradient 被截断、loss 指标计算错误等），且有充分证据（≥3 个 QUERY_DIAG 点或等价诊断）
   - 流程：(1) kill 相关进程并记录原因 → (2) 派 `/researcher` 分析根因 + 提出修复方案 → (3) 派 `/coder` 实现修复 → (4) 重新启动实验
   - 全程落账：ISSUES.jsonl、UPDATELOG.md、TRAINER_ACTIVE.md、TRAINER_ACTIVITY.jsonl、gpu_runs.jsonl
   - 不属于显著 bug 的情况：训练收敛慢、PPL 不如预期、hyperparameter 不理想——这些是实验结果，不是 bug: 派 `/researcher` 分析根因 + 提出修复方案, (如果需要的话派 `/coder` 实现修复, 重新启动实验)

---

## 并行 GPU 利用准则 (2026-04-26 user directive)

用户明确要求:**"我们的卡很多, 我希望你可以以最大的效率利用他们"**。

- 当前有 5 个节点 = 40 卡（本机 + .252 两台 B200 级 + .73/.82/.104 三台 H20）
- **不同节点可以并行跑不同实验**(red line #5 说的是同一节点不能双开)
- 当一类工作(例如 memory 架构实现)在推进时,**旧线索(WikiText rank sweep 等)不能停**
- 如果架构训练需要 1 个 8-GPU 节点,就让其余节点继续跑 baseline / eval / sweep
- ~~Q-Filters checklist~~ 已废弃 (2026-05-10 用户授权), `src/memory/qfilters/` 移入 `legacy/`
- **每个训练开一个后台 subagent 跑,不要阻塞 main**,main 继续调度其他工作
- subagent 返回的信息 main 必须处理(检查结果、决定下一步、落账)

### 跳过单卡 smoke + 最大化 batch size (2026-04-26 user directive)

用户明确要求:**"更改之后不需要单卡 smoke. 直接多卡运行. 另外记得要选取好 batch size 最大化显卡效率."**

- **代码改动后直接派 8-GPU 多卡 run,不再做 1-GPU × 10 chunks 的 smoke 步**
  - 之前的 smoke-before-full 约定作废(仅限"代码改动后"场景);如果是跨节点拓扑 / 数据分片这类纯运行时风险,仍可自行判断是否跑 tiny sanity
  - NaN / OOM / 炸显存等失败模式直接在多卡 full run 上暴露;失败时按红线 #3 append correction 行到 gpu_runs.jsonl 并查因
- **派发训练前必须算 batch size**:参考该训练脚本的显存占用经验值或前一个成功 run,把 `--batch_size` / `--per_device_batch_size` 调到 B200 (183 GiB) 或 H20 (97.8 GiB) 快被吃满为止
  - 目标:单卡显存占用 ≥ 80% (剩 20% 给 activation peak + NCCL buffer)
  - 如果该脚本不支持 batch 维度(例如 pg19 chunk 是样本粒度且 seq_len 已经占满显存),在 trainer commentary 里明确说"batch size 已受 seq_len 限制,无调节空间"
- **TRAINER_ACTIVE.md / gpu_runs.jsonl 必须记录 batch_size 字段**(新增字段,和 seq_len / num_slots 并列),便于复盘显存利用率

## 并行多算法实验规则（2026-05-02 用户指令）

**当同时推进多个算法方向时，必须遵守以下隔离规则，确保不互相干扰。**

### 代码隔离
- 每个算法使用**独立的配置参数名**（如 `--use_cross_attn_memory` vs `--use_infini_attention` vs `--use_attention_matching`）
- 共享底层基础设施（config.py、layer.py 的框架代码），但核心算法逻辑放在**不同文件或不同类**中
- coder subagent 修改代码前**必须确认当前没有其他算法在同文件修改**——如果冲突，先在 PENDING_TASKS.md 登记，等前一个 coder 完成
- 每个算法的**启动脚本独立**：`scripts/launch_<algorithm>.sh`

### 运行隔离
- 每个算法的 **output_dir 独立**：`outputs/<algorithm_name>/`
- 每个算法的 **log 文件独立**：`logs/<algorithm_name>_*.log`
- 同一节点同时只跑**一个算法**（8 GPU 全占）
- 节点分配记录在 `status/TRAINER_ACTIVE.md`

### Coder 并行派发
- 派发多个 coder subagent 时，每个 coder 的 prompt 必须：
  1. 明确说明**只修改哪些文件/类**
  2. 说明**不能碰哪些文件**（被其他 coder 占用）
  3. 提供**独立的启动脚本路径**
  4. 使用 `isolation: "worktree"` 或确保不修改共享代码
- 如果两个 coder 需要修改同一个文件（如 config.py），**串行执行**或由 main 合并更改

### 创新要求（用户 2026-05-02 指令）
- 复现已有工作（如 Attention Matching）时，必须**加入自己的创新点**
- 可以融合项目中已有的 memory slots 检索机制
- 每个算法方向的版本描述写入 `versions/vN_<描述>.md`

---

## PPL 级别洞察 — "PPL 显显著高 = 语言模型被污染" (2026-04-26 user insight)

**PPL 显著偏高(例如 167 / 752 / 5102)意味着不只是 memory 检索做得差,而是已经污染了整个语言模型的正常输出。**

| PPL 级别 | 诊断 | 行动 |
|---|---|---|
| < 10 | memory/压缩"基本可用",差异来自 retrieval 质量 | 可以调 hyperparam |
| 10-100 | 有明显问题但模型还在做 LM | 可能是 rank/budget 选得差 |
| **> 100** | **模型已经不会说话了**,attention 分布被破坏 | **先不要调 hyperparam**,排查 KV/RoPE/calibration/mask bug |
| > 1000 | 近乎随机输出,整段 logits 被污染 | 从最基础的单元测试开始排查 |

**操作准则**:
- PPL > 100 时,**先不要调 hyperparameter**
- 如果暂时说不出原因,**派 /researcher 去调查**(而不是继续调参)
- researcher 报告根因后再决定是修 bug 还是调参

### 当前未解 outlier
- **kv=256 rank=1 Llama-2 PPL=752**(其他同 rank 点 119-146 范围): 按上面准则这是"模型被污染"级别,应该去查 calibration 或 indexing,而不是当作"rank=1 在 kv=256 表现差"。

---

## 模型配置建议（2026-05-11）

**推荐组合：**
- **日常主模型**：`gpt-5.4`
- **长上下文主模型**：`claude-sonnet-4.6-1m`
- **coder subagent**：`claude-opus-4.7`（注意是 `4.7`，不是旧的 `4-7`）
- **轻量调研 / 快速任务**：`claude-sonnet-4.6`

**临时切换（日常最常用）：**
```bash
/model gpt-5.4
```

**长文档 / 大仓库切换：**
```bash
/model claude-sonnet-4.6-1m
```

**永久配置（推荐写进 shell rc，如 `~/.bashrc`）：**
```bash
export CODEBUDDY_MODEL=gpt-5.4
export CODEBUDDY_BIG_SLOW_MODEL=claude-opus-4.7
export CODEBUDDY_CODE_SUBAGENT_MODEL=claude-opus-4.7
export CODEBUDDY_SMALL_FAST_MODEL=claude-sonnet-4.6
```

**含义：**
- `CODEBUDDY_MODEL`：主会话默认模型
- `CODEBUDDY_BIG_SLOW_MODEL`：复杂推理 / `model="reasoning"` 默认走的模型
- `CODEBUDDY_CODE_SUBAGENT_MODEL`：子 agent 默认模型
- `CODEBUDDY_SMALL_FAST_MODEL`：轻量 / `model="lite"` 默认走的模型

> 依据 CodeBuddy Code 内置文档 `models.md`：`CODEBUDDY_CODE_SUBAGENT_MODEL` 决定子代理默认模型；`CODEBUDDY_BIG_SLOW_MODEL` / `CODEBUDDY_SMALL_FAST_MODEL` 决定 `reasoning` / `lite` 变体映射。

## Subagent 使用准则

> **⚠️ 严格禁止：把 Opus 4.7 写成旧错名 `claude-opus-4-7`。**
> **Agent tool 调用时优先用 `model="reasoning"`（映射到 Opus 4.7），不要写失效 alias。**
> **轻量调研/搜索再使用 `lite`。**

**写代码规则（2026-05-08 用户指令，2026-05-12 修正）**:
- **优先派 subagent 改代码**：超过 1 个文件、或单文件超过 5 行、或涉及架构/新功能 → 优先通过 Agent tool 派 coder
- **尽量用 Opus 4.7**：coder / repro / 复杂修复任务优先 `model="reasoning"`
- **错误根因已确认**：之前失败是因为把外部模型名写成了 `claude-opus-4-7`，正确配置名是 `claude-opus-4.7`
- **默认稳定用法**：代码任务显式写 `model="reasoning"`，并把环境里的 reasoning/subagent 路由指到 `claude-opus-4.7`
- **轻量任务**：调研/搜索/枚举可用 `model="lite"`
- **唯一例外 1**：修改在 **1 个文件、5 行以内** 的简单 bug fix → main 可以直接改
- **唯一例外 2**：如果当前主会话模型本身已足够且直接修改更快，可以直接改，不必为了形式强行派 agent
- **标准调用方式**：
```python
Agent(
    subagent_type="general-purpose",
    model="reasoning",
    description="实现/修复 <功能描述>",
    prompt="工作目录：/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/\n请先读取 CODEBUDDY.md...\n[详细修改要求，包含文件路径、代码片段]",
    run_in_background=True
)
```
- `reasoning` 路由当前应映射到 Opus 4.7；轻量调研再用 `lite`

**训练派 subagent 的阈值**:
- **每个 8-GPU 训练都派一个后台 subagent**(run_in_background=true)
- Main 继续做其他事情(写代码、调度其他节点、整理结果)
- subagent 完成时 main 收到通知,**必须**处理返回信息:检查结果合理性、落账 gpu_runs.jsonl / ACTIVE_SWEEPS.jsonl / TRAINER_ACTIVITY.jsonl、决定下一步

**调研派 subagent**:
- 文献调研、架构分析、跨 30+ 文件的 codebase 审计 → 派 researcher/general-purpose subagent
- 单文件或简单 grep → main 自己做

---

## 状态文件更新(operational hygiene)

| 事件 | 文件 | 动作 |
|---|---|---|
| 派发训练 | `status/ACTIVE_SWEEPS.jsonl` | append `{status:running}` 行 |
| 派发训练 | `status/TRAINER_ACTIVE.md` | **Write** 覆盖(禁止 Edit),记录所有活跃 run |
| 训练完成 | `status/ACTIVE_SWEEPS.jsonl` | append `{status:completed}` 行 + results |
| 训练完成 | `status/gpu_runs.jsonl` | append 每个 run 一行 |
| heartbeat | `status/TRAINER_ACTIVITY.jsonl` | append 心跳行 |
| 研究结论 | `status/RESEARCHER_REPORTS.jsonl` | append 报告摘要 |
| 重大操作 | `UPDATELOG.md` | append 人类可读记录 |

append-only 文件写错了**不要 edit**,追加一条 correction 行:
```json
{"ts":"...","sweep":"...","correction":"prior entry at 14:08 had wrong node .104; actual local","...":...}
```

---

## 版本管理规则（2026-04-30 用户指令）

**每次架构修改都要在 `versions/` 文件夹里创建一个版本描述文件。**

- 文件名格式：`versions/vN_<简短描述>.md`（如 `v1_top_k_routing.md`, `v2_cross_attention.md`）
- 内容必须包含：
  1. **Architecture**: 算法的完整 forward pass 伪代码
  2. **Initialization**: 每个关键参数的初始化值和理由
  3. **Relationship to prior work**: 与已知工作的区别（MemoryLLM、Block Recurrent Transformer、Infini-attention 等）
  4. **Known issues**: 当前版本的已知问题

---

## 关键路径速查

- 本地代码根:`/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/`
- **远程 canonical workdir**:`/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/`(wzc1 跨节点共享,5 节点都 cd 到这里,训练/eval/ckpt/数据免同步)
- SSH 通式见顶部「🖥️ 当前 GPU 集群」表:`sshpass -f <configs/password*.txt> ssh -o StrictHostKeyChecking=no -o PreferredAuthentications=password [-p 36000] root@<IP>`
- 当前 5 节点:本机(local) + .252(`28.89.19.252`) 两台 B200 级 + .73(`28.85.35.73`)/.82(`28.82.250.82`)/.104(`28.83.24.104`) 三台 H20(端口 36000)

---

## 启动时必读

每次新会话启动时,请读取:
1. `status/TRAINER_ACTIVE.md` — 当前有无运行中的训练
2. `UPDATELOG.md` 最后 30 行 — 最近发生了什么
3. `status/TRAINER_REQUESTS.jsonl` — 有无待审批请求
4. `AGENTS.md`(如存在) — 更详细的操作手册和历史教训

如果这是 heartbeat 触发的会话,直接执行 `/heartbeat` 的检查流程。

### ⚡ 关机恢复（2026-06-09，重要）

**本机 `/root/.codebuddy/` 和 `/root/.claude/` 在关机后会被 reset**（跨会话 memory、config、session 历史、tasks/teams 全丢）。已全量备份到项目盘 `cc_state/`（随盘持久，不丢）：
- `cc_state/codebuddy_full/`（~1.2G，`/root/.codebuddy` 全量快照）
- `cc_state/claude_full/`（~42M，`/root/.claude` 全量快照）
- `cc_state/memory/` + `cc_state/config/`（核心精简副本，便于人工查看）
- `cc_state/restore_cc_state.sh` — 一键全量恢复脚本

**新机器/关机重启后，第一件事在项目根执行：**
```bash
bash cc_state/restore_cc_state.sh   # 把 .codebuddy + .claude 拷回 /root/，恢复全部 cc 状态
```

⚠️ `cc_state/` 是**手动快照**，不会自动更新。**关机前**若想保留最新 memory/session，重跑一次备份：
```bash
cp -a /root/.codebuddy/. cc_state/codebuddy_full/ && cp -a /root/.claude/. cc_state/claude_full/
```
即使忘了备份，项目状态（`CODEBUDDY.md` / `status/` / `versions/` / `HEARTBEAT.md` / `.codebuddy/commands/`）随项目盘持久，不依赖 `cc_state/`——所以核心研究上下文永不丢，`cc_state/` 只是额外恢复 cc 自身的记忆/历史。

---

## Git 提交规范（2026-05-05，2026-05-08 更新）

**所有代码修改都必须有对应的 git commit，以便 CI 流程正常工作并保持实验可复现性。**

### Committer 设置
```
git config --global user.name "LiuHanzuo"
git config --global user.email "lhz24@mails.tsinghua.edu.cn"
```
**所有 commit 的 author 必须是 "LiuHanzuo"。**

### Coder 完成代码修改后必须：
1. `git add <具体文件名>` — 不用 `git add .` 或 `git add -A`（防止意外提交密码/权重）
2. `git commit -m "<type>: <简短描述>"` — type: feat / fix / refactor / chore / experiment
3. 如果配套有训练启动，将 commit_hash 记录到 `gpu_runs.jsonl`

### commit_hash 字段（gpu_runs.jsonl）
每条训练记录必须包含 `"commit_hash": "<7位hash>"` 字段：
```bash
git rev-parse --short HEAD  # 获取 7 位 hash
```

示例 gpu_runs.jsonl 条目：
```json
{"ts": "2026-05-05T12:00:00", "node": "local", "exp": "chunk_isolation_arm1", "commit_hash": "a1b2c3d", "seq_len": 256, "status": "running"}
```

### Heartbeat 对 git 的职责
- 每次 heartbeat 开始时执行 `git status` — 如果有未提交的修改（非训练日志类文件），记录为 WARNING
- 可以自主 `git commit`（提交代码改动）
- 在报告 HEARTBEAT_OK 前确认 git 状态干净或已知原因

### Git Push 规则（2026-05-05 用户授权）

**Agent（coder/heartbeat）在以下情况下可自主 `git push`，无需用户确认：**

| 情况 | Push 目标 | 条件 |
|------|-----------|------|
| 实验有进展（ratio 改善、新功能可用、bug 修复完成） | `main` 分支 | 无 `*.pt` / `password.txt`，非 force push，审核通过 |
| 实验无明显进展但代码值得保存 | 新 feature 分支（如 `archive/<exp_name>-<date>`） | 同上，方便以后回溯 |
| CI chore / 状态文件更新 | `main` 分支 | 同上 |

**Push 流程（必须按顺序）：**

1. **安全检查**（可自己执行）：
```bash
git diff --name-only HEAD  # 确认无 *.pt / *.bin / password.txt
git log --oneline -3       # 确认 commit 内容合理
```

2. **派 subagent 审核（必须，不可跳过）**：
   - 派一个 `general-purpose` subagent，prompt 包含：待推送的 `git log` + `git diff --stat`
   - subagent 执行 `git diff origin/main..HEAD` 检查改动，给出 APPROVED / REJECTED 结论
   - 只有 **APPROVED** 才能继续推送；REJECTED 则停止，写入 PENDING_TASKS.md

3. **通过 star-proxy 推送**（审核通过后）：
```bash
export http_proxy=http://star-proxy.oa.com:3128
export https_proxy=http://star-proxy.oa.com:3128
git push origin main   # 或 archive/... 分支
```

> 推荐直接调用 `/gitpush` skill，它包含完整的审核→推送流程。

**分支命名规范（无进展时）：**
```bash
git checkout -b archive/<exp_name>-$(date +%Y%m%d)
git push origin archive/<exp_name>-$(date +%Y%m%d)
```

### 禁止
- `git commit` 中加 `Co-Authored-By: Claude` 或任何 Anthropic/AI 署名
- `git add .` 或 `git add -A`（可能包含 configs/password.txt、*.pt 等敏感/大文件）
- `git commit --amend` 在已 push 的 commit 上（会破坏 CI 历史）
- `git push --force` 或 `git push -f`（任何情况下均禁止）
- Push 包含 `*.pt`、`*.bin`、`configs/password.txt` 的 commit

---

## CODE_CLEANUP_SUGGESTIONS.md 使用规范（2026-05-05）

**`CODE_CLEANUP_SUGGESTIONS.md` 由 CI 每周自动生成（`.github/workflows/ci_cleanup_suggestions.yml`）。**

### 文档职责
- 汇总项目中可能已经无用的代码（死代码、对应已放弃方向的模块）
- 指出可以**删除**的文件（高置信度：对应明确放弃的方向，60 天无 git 活动）
- 指出可以**移入 `legacy/` 文件夹**的文件（有历史参考价值但不活跃）
- 指出文件内部可以简化的代码段
- 列出近期最活跃的文件，帮助 agent 了解当前工作焦点

### Agent 自主执行规则
| confidence 级别 | Agent 行为 |
|----------------|-----------|
| `high` | heartbeat/coder 可自主删除或移入 `legacy/`，**无需用户确认** |
| `medium` | 写入 `PENDING_TASKS.md` 为 `[PENDING]` 任务，`auto_launch=false`，等用户确认 |
| `low` | 仅供参考，不做任何操作 |

### 执行前必须验证（无论 confidence 级别）
```bash
# 1. 检查文件是否被 import
grep -r "$(basename <filepath> .py)" src/ scripts/ tests/ --include="*.py" | grep -v "__pycache__"
# 2. 检查 __init__.py 中是否有引用
grep -r "$(basename <filepath> .py)" src/ --include="__init__.py"
```
- 如果有 import → **强制降级为 medium**，写入 PENDING_TASKS.md，不自主删除
- 确认无 import 后才执行，并立即 git commit

### legacy/ 文件夹规范
- `legacy/` 中的文件不参与任何 import（根目录 `__init__.py` 不引用）
- 移入时同时更新或创建 `legacy/README.md`，说明各文件的历史作用
- coder 移入后执行 `git commit -m "chore: move <files> to legacy/ - no longer active"`

### 触发条件
- 每周一 02:00 UTC 自动运行
- main 分支中 `src/` 或 `scripts/` 有改动时自动运行
- GitHub Actions 页面手动触发 `workflow_dispatch`
- 需要在 GitHub repo Settings → Secrets 添加 `ANTHROPIC_API_KEY`
