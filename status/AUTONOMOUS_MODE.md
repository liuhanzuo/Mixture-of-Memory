# AUTONOMOUS_MODE

**Status**: ENABLED
**Since**: 2026-04-24 17:00 GMT+8
**Last whitelist update**: 2026-04-26 12:02 GMT+8
**Authorized by**: user direct instruction ("我想让你有一个全自动的工作流(i.e.完全不询问我)" + 2026-04-26 追加 "这些我都授权. 我的意思是晚上我不在的时候你不需要经过我的授权就可以干这些事情" + 2026-04-26 12:02 追加 "加入白名单: 1.仅仅添加代码, 不更改 2.做research 3.数据同步 4. 跑实验")

---

## 启用时的自动化动作（heartbeat 发现即执行，不询问用户）

### 自动批准的 request 类型
- `new_experiment` 目标在 CLAUDE.md 已列为研究方向
  - P0: Attention Matching (arXiv:2602.16284)
  - P1: 4-bit KV Quantization
  - P2: Heavy Hitter (Cold-Compress)
  - P3: ARMT
  - **P4: Q-Filters (arXiv:2503.02812)** — 新增 2026-04-26；已有 Patch A 基础设施（`src/memory/qfilters/`, `scripts/eval_qfilters.py`），涵盖 kv_budget / recent_window / filter_rank sweep 续作
- `bug_fix` 且 evidence 包含完整 traceback + 单点修复建议
- `eval_only` 任何已完成训练的 checkpoint
- `config_refresh` 针对 remote_experiments.json 的元数据更新（不改超参）
- **`doc_only` 任何纯文档/分析产出（2026-04-26 追加）**
  - `/researcher` 分析类任务（retraction addendum、postmortem、sweep 分析笔记、文献综述）
  - `ops/research_notes/*.md` 写作，不涉及 GPU
  - 无超参变更、无代码变更、无新训练启动
- **`tech_debt` 对已标记 bug 的历史脚本修复（2026-04-26 追加）**
  - 已在 `status/ISSUES.jsonl` 或 TaskCreate 列出且指定文件 & 修复模式（例：Task #63 double-shift 修复）
  - 只修 bug，不动算法/超参
  - 每个脚本必须附最小 smoke test 或 grep 验证
- **`sweep_followup` 已完成 sweep 的直接后续（2026-04-26 追加）**
  - 白名单候选：`pg19 kv_budget curve`、`filter_rank sweep`、与已跑 sweep 同 seq/chunk/filter_rank 的 op-point 扩展
  - 必须复用已有 calibration filters 缓存（不重跑 calibration）
  - 单 8-GPU 实验，节点可用时立即启动
- **`code_additive` 只新增代码、不改已有代码（2026-04-26 12:02 追加）**
  - 允许：新增文件 / 新增函数 / 新增脚本 / 新增驱动（含 2-D sweep 驱动、fine-sweep 驱动）
  - 禁止：任何 Edit 已有文件中已有行（删除/重写已有行即算"更改"）
  - 新增必须不破坏已有 smoke 回归（不动 import graph 关键节点）
  - 与 `tech_debt` 正交：`tech_debt` 是修 bug（改已有代码），本类是纯新增
- **`research` 任何文献调研、实验分析、机制推测（2026-04-26 12:02 追加）**
  - `/researcher` 读文献、跑 WebSearch/WebFetch、写 analysis note、产 hypothesis 一律自动
  - 等价于 `doc_only` 的 superset：允许代码片段引用、Jupyter-style 计算、数值重验算
  - 仍禁止：直接启动 GPU 训练 / 直接改 `src/` 或 `scripts/` 中的非笔记类文件
- **`data_sync` 跨节点代码/数据同步（2026-04-26 12:02 追加）**
  - rsync（带 `--exclude='models/' --exclude='outputs/'` 等）、scp、filer 镜像、`/apdcephfs_wzc1` ↔ `/apdcephfs_zwfy6` 同步
  - 必须带 `--partial`（可恢复）；跨节点前先验证 ssh 活性
  - 不允许 `--delete` 之外的破坏性选项（如 `--remove-source-files`）
  - 后台 rsync 必须记录 PID + 日志路径到 `logs/`
- **`run_experiment` 跑实验（2026-04-26 12:02 追加，广义 sweep_followup + new_experiment 合并）**
  - 允许：已 smoke 过的脚本在任意 idle 节点启动 full run（单 8-GPU 约束仍保留）
  - 允许：新 driver 脚本（由 `code_additive` 产出）的 smoke + full
  - 仍禁止：并行多 8-GPU、覆盖 checkpoint、改已 approved 实验的超参

### 自动拒绝
- `hyperparameter_change` 对正在运行的实验（除非 OOM / NaN / NCCL 崩溃）
- 会覆盖现有 checkpoint 的动作
- 在同一节点上叠第二个 8-GPU 实验（跨节点并行已授权）

### 自动升级到用户（不自主处理）
- 活跃训练 Python traceback / OOM / NCCL error
- GPU 显存泄漏（进程已退出但显存仍占用）
- 所有远程节点同时失败
- 代码 bug 导致多轮实验结果无效
- 磁盘即将写满（< 50 GB 可用）
- 本地 GPU 上出现 unknown 进程且无法归因

---

## 自动工作流链（heartbeat 发现 approved request 后依次调度）

```
approved request
  ├─ new_experiment (新方向，无代码)
  │    → /researcher 产出 brief
  │    → /coder 实现模块+脚本 (必须 smoke pass)
  │    → /trainer 跑 smoke → full eval
  │
  ├─ bug_fix / tech_debt
  │    → /coder 修复（必须带 smoke test 或 grep 验证）
  │    → /trainer 重启 or eval（如需）
  │
  ├─ eval_only / sweep_followup
  │    → /trainer 直接评估（复用 filter cache）
  │
  └─ doc_only
       → /researcher 写 brief/analysis/retraction（无 GPU）
```

每一步完成后：
1. 写 `status/TRAINER_ACTIVITY.jsonl` 一条记录
2. 如果是链的中间环节，在 `status/AUTO_CHAIN.jsonl` 追加 `{request_id, stage, next_action}` 让下次 heartbeat 接力
3. 如果整条链完成（例如 eval 出 PPL），追加 `RESEARCHER_REPORTS.jsonl` 触发下一次 /researcher 分析

---

## 红线（即使自动化也不违反）

- ❌ 不能 kill GPU 进程（永远先升级）
- ❌ 不能绕过 smoke test 启动 full training
- ❌ 不能并行跑两个 8-GPU serious 实验
- ❌ 不能自主修改已 approved 实验的超参
- ❌ 不能删除任何 checkpoint 或 log
- ❌ TRAINER_ACTIVE.md 只能 Write 覆盖，禁止 Edit

---

## 关闭方式

用户说 "关闭自动化" 或 "disable autonomous"，或把本文件 Status 改为 DISABLED。
