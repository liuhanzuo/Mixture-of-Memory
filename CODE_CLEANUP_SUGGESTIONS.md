# CODE_CLEANUP_SUGGESTIONS.md
最近一次手动审计: 2026-05-10 (用户指示 "把没用的文件可以删一删")
上次自动生成: 2026-05-05 (CI 模板，未填充)

## A. 已执行的清理（2026-05-10 — 已 commit）

下列文件已 `git mv` 到 `legacy/`（详见 `legacy/README.md`），对应 CLAUDE.md 中
明确标注**已放弃**的方向：

| 类别 | 路径 | 已放弃方向 |
|------|------|-----------|
| 模块 | `src/memory/sparse_memory/` → `legacy/memory/sparse_memory/` | Sparse Memory (MAG-EMA) |
| 模块 | `src/memory/sparse/` → `legacy/memory/sparse/` | Sparse Memory v3 |
| 模块 | `src/memory/dms/` → `legacy/memory/dms/` | Dynamic Memory Sparsification |
| 模块 | `src/memory/rmt/` → `legacy/memory/rmt/` | RMT v3-v10 |
| 模块 | `src/memory/selective_context.py` → `legacy/memory/selective_context.py` | Selective Context |
| 脚本 | `scripts/train_*sparse*`, `scripts/train_mag*`, `scripts/train_mac*`, `scripts/train_dms*`, `scripts/train_rmt_v*`, `scripts/train_rmt_pg19.py`, `scripts/train_rmt_original.py`, `scripts/train_rmt.py` 及对应 `eval_*` / `debug_*` / `test_*` / shell wrappers | 同上 |
| 测试 | `tests/test_arch_a_fusion.py`, `tests/test_sparse_memory_smoke.py`, `tests/test_compressed_memory.py`, `tests/test_e2e_compressed_memory.py`, `tests/test_dms_qwen3.py` | 同上 |
| 顶层 | `launch_2048_v4.sh`, `run_eval.sh`, `run_eval_v9.sh`, `run_eval_fixed.sh`, `run_locomo_v4*.sh`, `run_nih_smoke.sh`, `run_nih_zh.sh`, `run_v8_nih.sh`, `run_nih_extended*.sh`, `run_ppl_v{4,10}.sh`, `run_32k_eval.sh` | 同上 |
| 删除 | 顶层 0 字节垃圾文件 `npm`, `thu-ai-assistant@1.0.0`, `ts-node` (untracked，直接 `rm`) | — |
| 删除 | `nohup.out`, `nohup_v3_8gpu.out` (untracked stale logs) | — |

**验证已做的 grep**：移动前 grep 全仓代码（`scripts/`, `src/`, `tests/`），确认仅
来自彼此的脚本/模块互相 import；移动后 `python -c "from src.memory.mem_space..."`
等活跃 import 仍然成功。

## B. MEDIUM confidence — 需用户决定后再清理

| 文件路径 | 理由 | 风险 |
|---------|------|------|
| `src/memory/mag/` (整个目录) | MAG-Gate 实现。仍被 `src/memory/scheduler.py` 的 `init_mag` 路径（lazy-import）和 `src/backbone/swa_model.py` 引用。但 `enable_mag=False` 是默认配置，scheduler 整体最近 60 天无 commit (上次 2026-03-22)。 | 删除前需先决定是否一并 retire 三级记忆 (l1/l2/l3 + scheduler) |
| `src/memory/{l1,l2,l3}/`, `src/memory/scheduler.py`, `src/memory/state.py` | 原 MoM 三级记忆栈。最后 commit 2026-03-22。仅由 `src/agents/` (memory_agent / turn_processor) 和 `src/training/train_l{2,3}.py`、`tests/test_l{1,2,3}.py`、`tests/test_scheduler.py` 引用 — 这些自身也都 60 天以上没改动。 | 当前所有训练脚本（H 系列）只用 `src/memory/mem_space/`，所以这一栈是死代码，但删除会 break tests。建议先 retire 这些 tests 再 retire 模块 |
| `src/agents/` (memory_agent.py, turn_processor.py, session_runner.py) + `tests/test_agent_smoke.py` + `scripts/run_chat.py`, `scripts/run_eval.py`, `scripts/run_ablation.py` | 同上，配套的 agent 框架 | 同上 |
| `src/training/train_l{2,3}.py`, `src/training/train_gate.py` + `scripts/train_l{2,3}.py` | 同上 | 同上 |
| `src/memory/qfilters/` + `scripts/eval_qfilters*.py` + `scripts/_issue110_smoke_calibration.py` | Q-Filters 相关。Q-Filters 仍是 CLAUDE.md "未停的旧线索"。最后 commit 2026-04-30。 | **目前不应删** — CLAUDE.md 明确说 "Q-Filters checklist 不能停" |
| `scripts/run_train_v8_*` 类未在 A 中处理但相关的旧 sweep 脚本 | 已在 A 中处理 | — |

**建议处理流程**：上面这些条目用户确认后写入 `PENDING_TASKS.md` 为 `[PENDING]`、
`auto_launch=false` 任务，再分批 retire。

## C. LOW confidence / 不动

| 文件路径 | 理由 |
|---------|------|
| `src/memory/slot/`, `src/memory/slot_memory/` | 仍然被 `eval_slot_memory.py` 等使用，CLAUDE.md 明确标注 active |
| `src/memory/rmt_slot/` | **2026-05-08 新增**，最活跃方向之一 |
| `src/memory/mem_space/` | H 系列训练核心 |
| `outputs/dms_8x/` | CLAUDE.md "待评估"，**绝不能动** |
| `tests/test_l{1,2,3}.py`, `tests/test_scheduler.py`, `tests/test_agent_smoke.py` | 见 B；当前 retire 三级记忆栈才能一起删 |
| `tests/test_bypass_*`, `tests/test_mem_space_smoke.py`, `tests/test_wrapper_internal_parity.py` | 与活跃路径相关，保留 |

## D. 活跃文件摘要 (近 7 天 git 改动)

| 文件 | 最近 commit | 角色 |
|------|------------|------|
| `scripts/train_cross_attn_memory.py` | 2026-05-10 | H 系列主训练脚本 |
| `src/memory/mem_space/selector.py` | 2026-05-10 | CrossAttentionMemoryV2 |
| `scripts/train_rmt_slot.py` | 2026-05-08 | RMT-Slot hybrid (新方向) |
| `src/memory/rmt_slot/rmt_slot_model.py` | 2026-05-08 | RMTSlotModel |
| `scripts/launch_experiment_h*.sh` | 2026-05-09/10 | H1-H13 ablation 启动器 |
| `test_h9_grad_flow.py` | 2026-05-10 | H9 梯度回归测试 |
| `test_contrastive_capture.py` | 2026-05-10 | 对比损失监控 |
| `versions/v3_rmt_slot.md` | 2026-05-10 | 架构版本笔记 |

## 说明

- confidence: high → heartbeat/coder 可自主执行删除或移入 `legacy/`（执行前必须 `grep` 验证无 import）
- confidence: medium → 写入 `PENDING_TASKS.md` 为 `[PENDING]` 任务，auto_launch=false，等用户确认
- confidence: low → 仅供参考，不做任何操作
- **绝对规则：移动/删除前必须确认文件未被任何 import 或 `__init__.py` 引用**
- 本文档由 `.github/workflows/ci_cleanup_suggestions.yml` 每周一自动生成（也可手动触发 `workflow_dispatch`），亦支持 main agent 手动审计后覆盖。
- 触发条件：每周 Monday 02:00 UTC，或 main 分支中 `src/` 或 `scripts/` 有改动时
