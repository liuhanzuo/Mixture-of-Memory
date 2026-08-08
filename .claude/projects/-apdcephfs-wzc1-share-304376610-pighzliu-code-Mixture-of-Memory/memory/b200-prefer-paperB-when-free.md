---
name: b200-prefer-paperb-when-free
description: "用户2026-08-02指令:B200(wzc1,~5-9x H20)收尾任务跑完空出后,优先把TODOList待跑任务(尤其Paper B训练)放B200跑;不打断/不浪费当前在跑的"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
  modified: 2026-08-02T10:58:53.892Z
---

用户 2026-08-02 指令（回答迁移策略提问）：**"等跑完现在的实验迁移过去，不要浪费 GPU 资源，也就是在 todolist 里面的任务尽量放到快的 B200 平台上跑。"**

**Why:** B200（wzc1 盘，8×L20A 183GB）实测比 H20 快 ~5–9×（训练 ~1s/step vs H20 ~9.5s/step；OLMo-2 7B 200k step ≈ 2.3 天 vs H20 ~22 天）。时间紧，用户要最大化吞吐。

**How to apply:**
- **不打断、不浪费**：当前在 B200 上跑的（LOCAL SnapKV yarn、.21 PyramidKV yarn、P2.4 eval 等）跑完再腾挪；在 H20 正常跑的（如 P0.5 两 arm）不为迁移而打断——它们在跑不算浪费。
- **B200 空出 → 优先接 TODOList 待跑 GPU 任务**，尤其 Paper B 训练（如 #123 PaperB P2.4 general-SFT pipeline data→pre→sft→post、任何新 heal/depth arm）。到时按 TODOList 优先级选具体任务。
- **迁移前置**：wzc1 与 diskB 不共享盘——迁 Paper B 任务需确认 wzc1 有其训练数据/ckpt。已知 wzc1 已有 `data/dolmino_now15b.npy`(62GB) + `../models/OLMo-2-1124-7B`（P0.5/keep-ladder 系列可直接在 B200 跑）。ckpt 若在 diskB 需先 scp 到 wzc1。
- 与 [[h20-paperA-over-paperB-priority]] 协同：H20 仍 PaperA-first；B200 空出则优先 Paper B 加速（PaperA GPU 项已基本收尾）。
- 参见 [[dllm-autonomous-phase-transition]]：阶段结果出+清晰下一步可自主启动。
