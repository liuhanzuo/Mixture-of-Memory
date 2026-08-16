---
name: dllm-autonomous-phase-transition
description: dllm 项目授权规则 — 阶段性结果出来后可以自主决定并启动下一步方向，不必等用户拍板
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 6c395da6-15fa-436e-b529-3d4585cc5de2
---

**用户明确授权（2026-07-10）**：当 dllm 项目里出现"某个实验阶段性结束、但存在清晰的可继续方向"这种局面时（如 exp41 GSM8K a/b/c 判决跑完出了决定性结果），**heartbeat/patrol 可以自主决定下一步并直接启动，不需要等用户回复确认**，目标是把 GPU 利用效率拉满。

**背景**：exp41 完成后我给用户列了 5 个候选方向（泛化检验/argmax-commit ablation/wall-clock测速/接回命题C训练线/更新论文骨架），用户选了"泛化检验 + argmax ablation 并行跑"，同时明确说以后遇到这种情况不用等他决策。

**How to apply**：
- 判断"可以自主决定"的边界：延伸方向必须是在**已验证方法/已有代码基础设施上的自然延伸**（换个任务域重测同一判决、对已有开关做 ablation、增大 n、复现同一实验换 checkpoint 等）——这类不需要用户批准。
- 仍需要用户批准的：架构性改动、需要新采购/新资源、彻底脱离当前研究问题的全新方向。
- 执行前仍要走安全流程：单卡 smoke 验证代码能跑通再上全量（[[dllm-h20-node]] 里定的规矩），落账 TRAINER_ACTIVITY.jsonl 说明"为什么选这个方向 + 怎么autonomous decide的"。
- 多个候选方向都低风险时，优先并行起（不同 GPU 分组），而不是排队串行跑，见 [[dllm-h20-node]] 的节点资源就是本机 8×H20，可以拆 4+4 或更细粒度并行。

**关联**：[[dllm-h20-node]]（节点连接信息 + 边界）
