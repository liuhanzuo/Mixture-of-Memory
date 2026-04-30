# Mixture-of-Memory — Claude Code 工作手册

## 自主派发规则（2026-04-29 用户指令，2026-04-30 更新）

**派发 `/researcher` 和 `/coder` subagent 不需要用户审批，可随时自主执行。**

- heartbeat 发现需要调研或代码修改时，直接派 subagent，无需询问用户
- 这包括但不限于：Fix 方案实现、bug 修复、诊断代码添加、文献调研

**2026-04-30 用户新增授权：hyperparameter 更改 + researcher 确认的更改，无需用户审批，可自动执行。**

- `/researcher` 分析后给出的 hyperparameter 建议（如 temperature、learning rate、skrl_weight 等）→ **直接自动执行，无需等待用户确认**
- researcher 报告中明确标注 `confidence: high / very_high` 的代码或配置更改 → **直接自动执行**
- **仍需用户审批的情况**：启动全新方向的训练实验（非 ablation/fix 延伸）；涉及模型架构的重大重构

---

## Git 提交规则

**禁止在 commit message 中加 `Co-Authored-By: Claude` 或任何 Anthropic/Claude 相关的 trailer。**
用户明确要求（2026-04-27）：git commit 只包含实际修改内容的描述，不附加任何 AI 署名行。

## 多节点 GPU 利用规则（2026-04-27 用户指令）

**当只有一个训练任务时，必须尽量使用多个节点来最大化效率。**

- 有 4 个 B200 节点（b200-1/2/3/4）+ 本地 8×H20，闲置节点是浪费
- 单任务训练时，优先考虑 **多机多卡 DDP**（torchrun `--nnodes N --rdzv_backend c10d --rdzv_endpoint <master_ip>:29500`）
- 若多机配置复杂（数据 sharding / 脚本不支持），退而求其次使用 **gradient accumulation** 提升单节点有效 batch size，目标 GPU mem bandwidth ≥ 70%
- **GPU mem bandwidth < 50% 视为欠载**，必须在 heartbeat 报告中标注 WARNING 并提出扩展方案
- kill 当前低效 run 并以更优配置重启是被允许的，不需要额外审批（用户指令优先）

## 项目概述

研究方向：**固定大小 memory buffer 压缩长上下文**。
核心问题：如何让 7B/8B LLM 在有限 KV budget 下处理超长序列。

代码根目录：`/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/`

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

本项目使用 Claude Code slash commands 实现多 agent 协作，对应 OpenClaw 的 trainer/researcher/coder 角色。

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
status/gpu_runs.jsonl          — 训练运行历史（append-only）
status/TRAINER_ACTIVE.md       — 当前活跃训练（write 覆盖，禁止 edit）
status/TRAINER_REQUESTS.jsonl  — trainer 发给 main 的审批请求
status/TRAINER_APPROVALS.jsonl — main 的审批决定
status/RESEARCHER_REPORTS.jsonl — researcher 的研究结论
status/ISSUES.jsonl            — 问题追踪
```

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

### 本地
- 8× H20 (97.8 GiB each)，当前状态见 `nvidia-smi`
- 节点：本机

### 远程 B200/L20A 集群
- 4 节点，每节点 8× L20A (183 GiB)
- IPs: 28.89.17.143 / .144 / 28.89.17.85 / 28.89.19.134
- SSH: `sshpass -f configs/password.txt ssh -o StrictHostKeyChecking=no root@<IP>`
- 远程工作目录：`/root/Mixture-of-Memory/`
- 远程模型：`/apdcephfs_wzc1/share_303098609/pighzliu_code/models/`

---

## 关键规则（Red Lines）

1. **TRAINER_ACTIVE.md 只能 write 覆盖，禁止 edit**（曾导致 gateway 崩溃）
2. **trainer 不能自主修改 hyperparameters/脚本版本**，必须写 TRAINER_REQUESTS.jsonl 等待批准
3. **gpu_runs.jsonl / ACTIVE_SWEEPS.jsonl 只追加，不修改历史**（错了就追加 correction 行，不要 edit）
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
   - **不属于显著 bug 的情况（仍需用户审批）**：训练收敛慢、PPL 不如预期、hyperparameter 不理想——这些是实验结果，不是 bug

---

## 并行 GPU 利用准则 (2026-04-26 user directive)

用户明确要求:**"我们的卡很多, 我希望你可以以最大的效率利用他们"**。

- 有 4 个 B200 节点 + 1 个本地 8× H20 节点
- **不同节点可以并行跑不同实验**(red line #5 说的是同一节点不能双开)
- 当一类工作(例如 memory 架构实现)在推进时,**旧线索(Q-Filters checklist、WikiText rank sweep 等)不能停**
- 如果架构训练需要 1 个 8-GPU 节点,就让剩下 3 个节点继续跑 Q-Filters / baseline / eval
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

---

## PPL 级别洞察 — "PPL 显著高 = 语言模型被污染" (2026-04-26 user insight)

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

## Subagent 使用准则

> **⚠️ 严格禁止：Main agent（Sonnet）自己改代码。**
> Main agent 使用的是 Claude Sonnet 模型，代码质量和推理能力不如 Opus。
> **所有代码修改（包括 bug fix、新功能、脚本调整）必须派 `/coder` subagent 执行。**
> `/coder` 使用 Claude Opus 模型，专门负责代码实现。
> Main 只做调度、分析、状态记录，不写代码。

**写代码派 subagent 的阈值**:
- 代码量 > ~200 行 或 涉及 ≥ 3 个新文件 → **派 coder subagent**(Agent tool, subagent_type=general-purpose)
- 代码量 < 100 行 且 在 1-2 个文件内 → **main 自己写**
- 介于之间:看 main 当前有没有其他阻塞任务;有就派 subagent

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
{"ts":"...","sweep":"...","correction":"prior entry at 14:08 had wrong node b200-3; actual b200-1","...":...}
```

---

## 关键路径速查

- 本地代码根:`/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/`
- **远程 canonical workdir**:`/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/`(wzc1 跨节点共享,训练脚本都 cd 到这里)
- SSH:`sshpass -f configs/password.txt ssh -o StrictHostKeyChecking=no root@<IP>`
- IPs:b200-1 .143, b200-2 .144, b200-3 .85, b200-4 .134

---

## 启动时必读

每次新会话启动时,请读取:
1. `status/TRAINER_ACTIVE.md` — 当前有无运行中的训练
2. `UPDATELOG.md` 最后 30 行 — 最近发生了什么
3. `status/TRAINER_REQUESTS.jsonl` — 有无待审批请求
4. `AGENTS.md`(如存在) — 更详细的操作手册和历史教训

如果这是 heartbeat 触发的会话,直接执行 `/heartbeat` 的检查流程。
