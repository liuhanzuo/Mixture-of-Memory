# PENDING_TASKS.md — Task Board
## Updated 2026-05-16 21:30 CST

---

## [RUNNING] P11 8B FSDP 训练
- Node: Local H20 × 8
- Step: ~2190/5000 (44%), ETA ~05:00 CST 2026-05-17
- auto_launch: N/A (already running)
- Next: eval on BABILong qa1/qa2/qa5 × 7 lengths when complete

## [RUNNING] 1B v4 L1+L2+L3 FSDP 训练
- Node: 28.59.80.196 × 8
- Step: ~2010/5000 (40%), ETA ~03:30 CST 2026-05-17
- auto_launch: N/A (already running)
- Next: eval + compare vs v2 (L1+L3 only, mean=37.29) to measure L2 contribution

---

## [PENDING] P11 8B eval（训练完成后）
- 触发条件: P11 training reaches step 5000 / outputs final ckpt
- 任务: 在空闲节点跑 BABILong eval qa1/qa2/qa5 × {0k,1k,2k,4k,8k,16k,32k}
- 比较目标: vs 8B P8 (mean=59.14), vs LM2 paper 8B vanilla
- auto_launch: true
- 节点: 28.59.80.196 (if 1B v4 done by then) or any free node

## [PENDING] 1B v4 eval（训练完成后）
- 触发条件: 1B v4 training reaches step 5000
- 任务: eval qa1/qa2/qa5 × 7 lengths
- 比较目标: vs v2 (mean=37.29, L1+L3 only) — does L2 help?
- auto_launch: true

## [PENDING] v3 short-context fix 训练（1B）
- 代码已就绪: `scripts/launch_phase1b_v3_shortfix.sh` (commit f9e3fa7)
- 内容: length-weighted sampler (75% short) + skip_mem_when_short
- 目标: 修复 qa5 短上下文 -20.5pp 问题
- 节点: 需要空闲 H20 节点（目前两台都被占）
- auto_launch: true（有空闲节点时）
- 前置: 等 1B v4 或 P11 完成后用其节点

## [PENDING] 更新 status/H_V2_PLAN.md
- 内容: 用当前 Phase-1B 实验状态替换旧的 H-v2 内容
- auto_launch: false（低优先级文档整理）

## [PENDING] Git push（5 commits ahead of origin）
- 待 push commits: a6dcda3 / f9e3fa7 / c05b1c9 / 9b53360 / 0349264
- 内容: FSDP fixes, v3 short-fix, v4 launch script, L2Compressor fix
- 流程: git diff 安全检查 → subagent 审核 → push via star-proxy
- auto_launch: false（需用户授权 or 等训练稳定后）

---

## [DONE]
- 1B v1 训练 (500 steps) + eval ✅
- 1B v2 训练 (10k steps) + multi-ckpt eval ✅
- v2 multi-ckpt scoring (step1000/5000/final × qa1/qa2/qa5) ✅
- P11 8B OOM 根因分析 (cuBLAS workspace fragmentation) ✅
- FSDP migration (Coder-28, 3 commits) ✅
- 1B v4 launch script + FSDP fix (Coder general-purpose-1) ✅
- L2Compressor FSDP forward() fix ✅
