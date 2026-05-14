# PENDING_TASKS.md — Pending Tasks

**Heartbeat must check this file every inspection. If pending tasks exist, heartbeat cannot just report HEARTBEAT_OK.**

## Active Tasks

- [RUNNING] H-series v2 / baseline execution
  - owner: heartbeat/main
  - auto_launch: true
  - status: 用户于 2026-05-12 澄清：**暂时不在的是 replacement B200 `b200-5..8`，稳定 B200 `b200-1..4` 还在**。当前 heartbeat 再次复查确认：b200-1 的 H-v2 A、b200-3 的 H-v2 D、b200-4 的 H-v2 B 三条 Phase 2 训练继续健康推进，并已在 `segments=8` 分别推进到约 `step=340/320/360`。相反，b200-2 上的 `H-v2 C Phase 1` 仍保持已停止状态；旧日志尾部继续显示它在更晚窗口（约 `epoch 0.03569`）仍持续 `eval_loss=nan`，因此“不要直接重启 C”这一结论不变。replacement B200 仍为 `b200-5/6: Permission denied`、`b200-7/8: Connection refused`，继续视为 unavailable / not ready。H20 侧当前 4 节点都可达且空闲；已知结果不变：`h20-1` 的 `MPlus-8B-smoke-grid-20260513` 仍是 6 个 `qa1 × length` 全部退化为 `!!!!!!!!!!!!!!!!!!!!`，`h20-2/3/4` 的 `RMT PPL / NIH / memory` retry3 仍全部结束且根因确认为 `legacy/scripts/eval_rmt.py` 与 v10 checkpoint schema 不兼容，而不是 checkpoint 本身损坏。本轮 heartbeat 已进一步派出更聚焦的后台 follow-up：一条研究 RMT 的 v10-aware / official eval path，一条审计 MPlus live import / RoPE 修复路径。
  - next_action: 继续监控 b200-1/3/4 上的 H-v2 A / D / B 训练健康度；b200-2 在查清 `third_party/associative-recurrent-memory-transformer/run_finetuning_lm_rmt_hf.py` 的 eval path / labels / metric 之前，**不要直接重启 H-v2 C**。H20 侧优先级维持不变但状态更明确：1) 先修复 `scripts/eval_baseline_babilong.py` / `MemoryLLM-source` 的 MPlus punctuation-only generation；2) 把 RMT H20 eval 从 `legacy/scripts/eval_rmt.py` 切到 v10-aware / official path；3) 若后续 ARMT checkpoint 在 H20 shared mount 就绪，再接 `ARMT qa1 smoke → full eval`。H-v2 训练侧仍保留 researcher 建议的 D multi-task Phase 2 作为下一条高价值 follow-up；待当前占用节点允许启动 follow-up 时，直接沿用已修补 prompt parity + cross-segment BPTT 的 Phase 2 路径。

## Pending Tasks

### [DONE] Start Phase 1 Memory Paper Reproduction
- priority: high
- auto_launch: true  (user approved on 2026-05-11)
- description: |
    已完成 Phase 1 关键动作：MemoryLLM BABILong eval 完成；ARMT full PG19 已在 b200-5/6/7/8 完成；HMT 正在 b200-2 跑；H-series v2 A/B/D 已启动。
    后续推进转入当前 RUNNING 任务 “H-series v2 / baseline execution”。

### [PENDING] resolve issue_planA_niah_zero
- priority: medium
- auto_launch: false
- description: |
    CRITICAL: Llama-3-8B base scores 0-1% on BABILong natively.
    Root cause: base model (not Instruct) cannot follow BABILong prompts.
    Resolution: Switch to Instruct base model for future experiments.
    Direction decided on 2026-05-11: proceed with Phase 1 reproduction first, and use checkpoint-only validation to close this issue.

### [PENDING] cleanup pighzliu_code unused files
- priority: medium
- auto_launch: false
- description: |
    Audit /apdcephfs_wzc1/share_303098609/pighzliu_code and summarize files/directories not needed for current reproduction/training work.
    Produce a delete plan with clear categories: safe-to-delete, keep, and needs-confirmation.
    Delete only clearly irrelevant or user-approved files/folders; avoid deleting models, datasets, active logs, checkpoints, or running experiment outputs without explicit confirmation.

### [PENDING] repair replacement_b200_mom_env
- priority: high
- auto_launch: false
- description: |
    Current heartbeat probe shows b200-5..8 are alive and idle, and b200-5 has now verified that the correct project root is under `/apdcephfs_wzc1/share_304376610/...` with a working project `.venv` (`torch/transformers/accelerate/fla/CUDA` all import and a CUDA tensor smoke passes).
    However, the repaired ARMT relaunch on b200-5 (`logs/armt_pg19_full_b2005_20260512_2303.log`) still crashes during the initial evaluation phase inside `third_party/associative-recurrent-memory-transformer/modeling_amt/language_modeling.py` with `AttributeError: 'NoneType' object has no attribute 'size'` while processing `past_key_values`.
    Before using replacement B200 for H-v2 C / ARMT / RMT follow-up work, finish the remaining runtime repair:
    1. keep launcher path detection on `share_304376610`,
    2. keep using a Python env with `fla` available,
    3. fix the ARMT runtime bug around `past_key_values` / evaluation on that pool,
    4. only then retry C/ARMT on replacement B200.
    Until this is repaired, replacement B200 should be treated as reachable-but-not-ready for Mixture-of-Memory auto launches.

## Key Discoveries (H-Series Complete)

### H-Series Winner: H14_isolate_aggr (ratio=1.0112@4600 RECORD)
- isolate_write_layer + aggressive NIAH (mix=0.50, lambda=2.0) + cross_attn_lr_factor=10
- Best PPL ratio achieved in entire H-series

### 方案A: BABILong 0% Root Cause
- Llama-3-8B base cannot do BABILong (0-1% natively)
- NIAH training (single-fact recall) does not teach multi-entity relational reasoning
- niah_loss is NOT a retrieval metric (H13=0.58 vs H14=11.64, both 0% BABILong)
