---
name: h20-eval-b200-train-split
description: "用户2026-08-02指令:H20优先跑eval,B200优先跑长程训练;分工提高整体吞吐"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
  modified: 2026-08-02T11:56:42.092Z
---

用户 2026-08-02 两条连续指令，合并为节点分工规则：
- **"H20 优先用来跑 eval"**
- **"B200 最好跑那些长程的训练任务，这样效率最高"**

**Why:** B200(wzc1 8×L20A) 训练比 H20 快 ~5.4×（P0.5 实测 1.77s/step vs H20 9.5s/step），长程训练放 B200 收益最大；H20 相对慢但足够跑推理/eval（RULER/LoCoMo/MMLU-content/timing 等）。分工 = B200 满载长训练 + H20 满载 eval，整体吞吐最高、不浪费。

**How to apply:**
- **长程训练（200k-step heal / depth-sweep / P0.5 结构隔离等）→ 优先派 B200**（LOCAL wzc1 + .21 wzc1）。空 B200 优先接 Paper B 长训练（见 [[b200-prefer-paperB-when-free]]）。
- **eval / 推理 / timing bench / MMLU-content 扫描 → 优先派 H20**（.73/.82/.104 diskB）。空 H20 优先接待跑 eval 项。
- 与 [[h20-paperA-over-paperB-priority]] 协同：H20 跑 eval 时仍 PaperA-first（PaperA eval 项 > PaperB eval 项）；PaperA eval 排空后 H20 接 PaperB eval（如 P0.6 MMLU-content）。
- 迁移/kill 只在「当前任务跑完 + 有更优节点」时做，不打断在跑的健康任务。
