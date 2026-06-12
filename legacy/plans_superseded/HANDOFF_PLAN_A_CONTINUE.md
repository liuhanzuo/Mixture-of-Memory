# Handoff: 方案A 完成后续工作

**写入时间**: 2026-05-11 00:05 GMT+8
**写入人**: 上一个 Claude Code session（即将关闭/被杀）
**给**: 下一个 Claude Code CLI session

---

## 1. 你是谁，要做什么

你是 Mixture-of-Memory 项目的 main agent。上一个 session 完成了「方案A」(扩展 BABILong watchdog + 验证 H13/H14 NIAH accuracy)，结论是 **CRITICAL: H 系列全部 0%**。你需要接续：(1) 落账上一 session 留下的状态变更；(2) 派 /researcher 分析根因；(3) 决定是否切换方向。

**项目根**: `/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/`
**主要 CLAUDE.md**: 同根目录，必读。

---

## 2. 方案A 关键结论（必读）

| Exp | Step | PPL ratio | niah_loss (train) | BABILong qa1 | qa2 | qa5 | avg |
|-----|------|-----------|-------------------|--------------|-----|-----|-----|
| H13_isolate | 2500 | ~1.035 | 0.580 | 0/30 (1k/2k/4k/8k) | 0/30 | 0/30 | **0%** |
| H14_isolate_aggr | 1500 | ~1.013 (RECORD) | 11.64 | 0/30 (1k/2k/4k/8k) | 0/30 | 0/30 | **0%** |

**含义**: PPL/NIAH trade-off 假说被实证否决。training niah_loss 跨度 0.58→11.64 但 BABILong accuracy 都是 0%。这不是 trade-off 边界，是 **整个 H 系列 cross-attn memory 在 bAbI 风格 retrieval 任务上无法 generalize**。

完整证据：
- `status/babilong_realtime.jsonl`（在 h20-3 上：`/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/status/babilong_realtime.jsonl`，11 行 — H13 4 个 length + H14 6 条带重复）
- `status/ISSUES.jsonl` 最后一条 `id: issue_planA_niah_zero` (severity: CRITICAL)
- `UPDATELOG.md` 最后一条 `2026-05-11 00:02 GMT+8 — 方案A 完成`

---

## 3. h20-3 还在跑的进程（需要 babysit）

通过 SSH 连：`sshpass -f /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/configs/password_h20.txt ssh -o StrictHostKeyChecking=no -o PreferredAuthentications=password -o ConnectTimeout=15 root@28.49.38.97`

**活进程**（截至 00:04）:
- `PID 36687` — `babilong_ckpt_watchdog.py` daemon，启动时间 23:50。**当前卡在 H10 历史 ckpt rsync**（cluster 1 → h20-3，每个 50GB ~40 min）。会自然推进，但有 8-12 个 H10/H12/H11_v2/H9 老 ckpt 排队，全部 rsync 完要数小时。
  - log: `/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/logs/watchdog_20260510_2353.log`
  - state: `/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/.watchdog_state.json`
- `PID 36000` + `PID 37726` — 两个 sshpass+rsync 子进程（H10/step_500 + H10/step_1000 从 b200-2 28.89.17.144）

**日志文件**: `/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/logs/`
- `watchdog_20260510_2353.log` — daemon 主循环（最新）
- `babilong_eval_*.log` — 自动 eval 输出
- `manual_H14_1500_*_g[4567].log` — 手动 launch 的 H14 4 个 GPU eval 输出

**H14 eval 完成状态**: 4 个 manual H14 eval 全部完成（PIDs 39814/40234/40846/41458，已退出）。结果在 babilong_realtime.jsonl。

---

## 4. 上一 session 已完成的落账

| 文件 | 改动 |
|------|------|
| `scripts/babilong_ckpt_watchdog.py` | 已 commit `d4f5723` — 扩展 cluster-2 + 加 H13_isolate + H14_isolate_aggr |
| `status/ISSUES.jsonl` | append `issue_planA_niah_zero` (CRITICAL, open) |
| `UPDATELOG.md` | append 方案A 完成报告 |
| h20-3 上的 `scripts/eval_cross_attn_babilong.py` | rsync 修过的版本（修了 LEN_MAP `int("1k")` ValueError） |

**未落账，需要你做**:
- `status/TRAINER_ACTIVE.md` — 加入「方案A 完成」块 + H13/H14 BABILong 0% 实测数据。**注意 RED LINE**: 只能 Write 覆盖，**不能 Edit**。最新版本是另一拍 heartbeat 改的（截至 23:45），有 H14@1800 ratio 1.0127、H13@3000 ratio 1.0316 等新评估点。直接 Read 当前最新内容 → 加上方案A 块 → Write 覆盖。
- `status/TRAINER_ACTIVITY.jsonl` — append 一条 heartbeat / event 记录方案A 结果

---

## 5. 立即行动建议（按优先级）

### Priority 1: 落账 TRAINER_ACTIVE.md
```
Read /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/status/TRAINER_ACTIVE.md
→ 在 Updated 时间戳下加方案A 章节 (BABILong 0% 数据)
→ Write 覆盖
```

### Priority 2: 派 /researcher 分析根因（无需用户审批，CLAUDE.md 授权）
prompt 模板（给 researcher subagent，run_in_background=true）:
```
你是 Mixture-of-Memory 项目的 researcher。

紧急任务（CRITICAL）：分析为什么 H 系列 cross-attn memory 在 BABILong 上 qa1/qa2/qa5 全部 0/30。

证据（status/ISSUES.jsonl `issue_planA_niah_zero`）:
- H13_isolate@2500 PPL 1.035, niah_loss 0.580 → BABILong 1k/2k/4k/8k 全 0/30
- H14_isolate_aggr@1500 PPL 1.013, niah_loss 11.64 → BABILong 1k/2k/4k/8k 全 0/30
- H10/H11_v2/H12 历史 ckpt 在 babilong_realtime.jsonl 也都 0% (除 H10@500 step 8k qa5 = 2/30 = 6.7%, 噪声水平)

研究问题:
1. 训练 NIAH 任务（合成 needle: "the magic password is XYZZ123"）vs BABILong qa（真实 bAbI: "Where is John?"）的 prompt 格式 / answer format 差异
2. chunk_size 4096 下 cross-attn memory write/read 是否真的传递了 needle 信息？读 src/memory/slot_memory/* 找证据
3. niah_loss = next-token CE 对预测 needle 而不是真正 retrieval — 这是不是根本失配？
4. 找 baseline Llama-3-8B（无 memory）的 BABILong qa1/qa2/qa5 准确率作对照（已存在的 baseline 在 babilong_baselines/* 或 RESEARCHER_REPORTS.jsonl 里）

输出: 写入 ops/research_notes/2026-05-11_planA_niah_zero_diagnosis.md，并 append RESEARCHER_REPORTS.jsonl 一行。

可参考 versions/v* 文件了解架构演进，src/memory/* 是当前实现。confidence 务必明确（high/medium/low）。

不要派子 agent 写代码，只做调研和分析。
```

### Priority 3: 决定方向（researcher 报告完成后）

预期 researcher 报告会暴露以下方向之一:
- **A**: 训练任务和 BABILong eval format 失配 → 直接换训练任务（用真实 bAbI / HotpotQA / RACE）
- **B**: cross-attn memory 在 chunk 边界丢失信息 → 改架构（fixed memory bank + recurrent state）
- **C**: niah_loss != retrieval — needle prediction 不等同 retrieval → 改 loss function

每个方向都涉及 multi-file 改动 → **必须派 opus** 实现，不要 main 自己改。

---

## 6. 当前 8 个活跃训练（背景）

读最新 `status/TRAINER_ACTIVE.md` 获取实时步数。截至 23:45（上一拍 heartbeat）:

| Exp | Node | 当前步 | PPL ratio | NIAH 状态 |
|-----|------|--------|-----------|-----------|
| H14_isolate_aggr | b200-8 (cluster 2) | ~1880 | 1.0127@1800 RECORD | **方案A 实测 0%** |
| H13_isolate | b200-1 | ~3050 | 1.0316@3000 | **方案A 实测 0%** |
| H10 | b200-2 | ~3400+ | 1.045@3200 | 历史 0% |
| H12 | b200-4 | ~3260+ | 1.055@3200 | 历史 0% |
| H11v2 | b200-3 | ~2870 | 1.058@2600 plateau | 历史 0% |
| H13_slot128 | b200-5 | ~1820 | 1.056@1800 ABORT WARN | 未测 |
| H13b_slot256 | b200-6 | ~1810 | 1.062@1800 ABORT WARN | 未测 |
| H14_base_unfrozen | h20-4 | ~820 | 1.109@800 ABORT WARN | 未测 |

**核心矛盾**: 8 个实验都在烧 GPU 但 BABILong NIAH 全部 0%。需要决策是否 kill 全部转向新方向。但**不要自主 kill healthy 训练** — 这是 user 决策。

---

## 7. 集群速查（CLAUDE.md 完整版）

| 集群 | 节点 | 密码文件 | 远程 share | 备注 |
|------|------|---------|----------|------|
| 1 (原始 B200, 稳定) | b200-1..4 (28.89.17.143/.144/.85, 28.89.19.134) | `configs/password.txt` | `/apdcephfs_wzc1/share_303098609/...` | 主训练 |
| 2 (Ephemeral B200) | b200-5..8 (28.89.18.132/.190, 28.89.20.82, 28.88.184.252) | `configs/password_b200_ephemeral.txt` | `/apdcephfs_wzc1/share_304376610/...` | 可能随时被回收 |
| 3 (H20) | h20-1..4 (28.48.2.147, 28.49.48.243, 28.49.38.97, 28.58.246.254) | `configs/password_h20.txt` | `/apdcephfs_zwfy6/share_304376610/...` | watchdog 在 h20-3 |

SSH 模板（cluster 2/3 需 PreferredAuthentications=password）:
```bash
sshpass -f configs/password_h20.txt ssh -o StrictHostKeyChecking=no -o PreferredAuthentications=password -o ConnectTimeout=15 root@28.49.38.97
```

---

## 8. 上一 session 工具调用记录（参考）

- 派了 1 个 opus 后台 agent 改 watchdog（agent ID `adcc83dfe71836202`，commit `d4f5723`） — 已完成
- 启了 2 个本地后台 rsync（task_id `bs0ty5j7p` H13, `bb324035e` H14） — 已完成
- 启了 1 个本地后台 daemon launch（task_id `b1xcm6hws`, `bh6xn1fpb`） — 已完成
- 4 个 H14 manual eval（h20-3 PIDs 39814/40234/40846/41458） — 已完成

所有 background tasks 都已退出（exit 0）。h20-3 daemon (PID 36687) 仍在跑，占着低优先级 H10 历史 rsync — 可让它继续，也可 `pkill -f babilong_ckpt_watchdog` 停掉以省带宽。

---

## 9. 你的第一条 reply 应该是

1. Read `HANDOFF_PLAN_A_CONTINUE.md`（这个文档）
2. Read `status/TRAINER_ACTIVE.md` 看最新状态
3. Read `status/ISSUES.jsonl` 末尾几行确认 `issue_planA_niah_zero` 在
4. **执行 Priority 1**: Update TRAINER_ACTIVE.md 加方案A 块
5. **执行 Priority 2**: 派 researcher subagent (run_in_background=true) 分析 NIAH=0 根因
6. 然后报告给用户：「方案A 落账完成 + researcher 已派发」

不要重新跑方案A，不要重新启 daemon，不要再 rsync ckpt。所有数据已经收齐。

---

**Good luck. — 上一 session @ 2026-05-11 00:05 GMT+8**
