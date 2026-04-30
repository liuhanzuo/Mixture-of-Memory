# AGENTS.md — Mixture-of-Memory Operating Notes

本文件记录 main agent 在本项目中必须持守的工作准则。由 main agent 维护,每次重要教训发生后追加。

---

## 1. Red Lines(CLAUDE.md 已声明,此处加粗)

1. **TRAINER_ACTIVE.md 只能 Write 覆盖,禁止 Edit**(曾导致 gateway 崩溃)
2. **trainer 不能自主修改 hyperparameters/脚本版本**,必须写 TRAINER_REQUESTS.jsonl 等待批准
3. **gpu_runs.jsonl / ACTIVE_SWEEPS.jsonl 只追加,不修改历史**(错了就写一条 correction 行,不要 edit)
4. **新实验启动前必须确认 GPU 状态**(nvidia-smi 验证,无 orphan 进程)
5. **同一节点同时只跑一个 serious 实验**(8-GPU);多节点可并行

## 2. 并行 GPU 利用准则 (2026-04-26 user directive)

用户明确要求:**"我们的卡很多, 我希望你可以以最大的效率利用他们"**。

- 有 4 个 B200 节点 + 1 个本地 8× H20 节点
- **不同节点可以并行跑不同实验**(red line #5 说的是同一节点不能双开)
- 当一类工作(例如 memory 架构实现)在推进时,**旧线索(Q-Filters checklist、WikiText rank sweep 等)不能停**
- 如果架构训练需要 1 个 8-GPU 节点,就让剩下 3 个节点继续跑 Q-Filters / baseline / eval
- **每个训练开一个后台 subagent 跑,不要阻塞 main**,main 继续调度其他工作
- subagent 返回的信息 main 必须处理(检查结果、决定下一步、落账)

## 3. "PPL 显著高 = 语言模型被污染,不止是记忆问题" (2026-04-26 user insight)

用户提醒:**PPL 显著偏高(例如 PPL=167, 752, 5102 这种数量级)意味着不只是 memory 检索做得差,而是已经污染了整个语言模型的正常输出。**

应该形成的洞察:
- PPL < 10 → memory 或压缩方案"基本可用",差异来自 retrieval 质量
- PPL 10-100 → 有明显问题但模型还在做 LM,可能是 rank/budget 选得差
- PPL > 100 → **模型已经不会说话了**,attention 分布被破坏,KV 残缺到 logits 失真;此时继续调 retrieval 没意义,要先排查是否:
  - KV cache 被写坏(indexing bug, shape mismatch)
  - RoPE 位置乱掉(sub-window 位置 bug,Patch-A 修的就是这个)
  - calibration 数值异常(filters 里有 NaN / 极端值)
  - attention mask 漏/错
- PPL > 1000 → **近乎随机输出**,通常意味着整段 logits 被污染,需要从最基础的单元测试开始排查

**操作准则**:
- 看到 PPL >100 的结果,**先不要去调 hyperparameter**
- 如果暂时说不出原因,**派 /researcher 去调查**(而不是继续调参)
- researcher 报告根因后再决定是修 bug 还是调参

### 当前未解 outlier
- **kv=256 rank=1 Llama-2 PPL=752**(其他同 rank 点 119-146 范围): task #110 正在待处理。按上面准则这是"模型被污染"级别,不应该被当作"rank=1 在 kv=256 表现差",要去查 calibration 或 indexing。

## 4. Subagent 使用准则

**写代码派 subagent 的阈值**:
- 代码量 > ~200 行 或 涉及 ≥ 3 个新文件 → **派 coder subagent**(用 Agent tool,subagent_type=general-purpose 或 code-specific)
- 代码量 < 100 行 且 在 1-2 个文件内 → **main 自己写**
- 介于之间:看 main 当前有没有其他阻塞任务;有就派 subagent

**训练派 subagent 的阈值**:
- **每个 8-GPU 训练都派一个后台 subagent**(run_in_background=true)
- Main 继续做其他事情(写代码、调度其他节点、整理结果)
- subagent 完成时 main 收到通知,**必须**处理返回信息:检查结果合理性、落账 gpu_runs.jsonl / ACTIVE_SWEEPS.jsonl / TRAINER_ACTIVITY.jsonl、决定下一步

**调研派 subagent**:
- 文献调研、架构分析、跨 30+ 文件的 codebase 审计 → 派 researcher/general-purpose subagent
- 单文件或简单 grep → main 自己做

## 5. 状态文件更新(operational hygiene)

每个训练启动、完成、心跳时都要同步更新:

| 事件 | 文件 | 动作 |
|---|---|---|
| 派发训练 | `status/ACTIVE_SWEEPS.jsonl` | append `{status:running}` 行 |
| 派发训练 | `status/TRAINER_ACTIVE.md` | **Write** 覆盖(禁止 Edit),记录所有活跃 run |
| 训练完成 | `status/ACTIVE_SWEEPS.jsonl` | append `{status:completed}` 行 + results |
| 训练完成 | `status/gpu_runs.jsonl` | append 每个 run 一行 |
| heartbeat | `status/TRAINER_ACTIVITY.jsonl` | append 心跳行 |
| 研究结论 | `status/RESEARCHER_REPORTS.jsonl` | append 报告摘要 |
| 重大操作 | `UPDATELOG.md` | append 人类可读记录 |

## 6. 修正已追加但写错的条目

append-only 文件写错了**不要 edit**,而是追加一条 correction 行:
```json
{"ts":"...","sweep":"...","correction":"prior entry at 14:08 had wrong node b200-3; actual b200-1","...":...}
```

## 7. 关键路径速查

- 代码根:`/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/`
- 远程 canonical workdir:`/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/`(wzc1 跨节点共享)
- SSH:`sshpass -f configs/password.txt ssh -o StrictHostKeyChecking=no root@<IP>`
- IPs:b200-1 .143, b200-2 .144, b200-3 .85, b200-4 .134
