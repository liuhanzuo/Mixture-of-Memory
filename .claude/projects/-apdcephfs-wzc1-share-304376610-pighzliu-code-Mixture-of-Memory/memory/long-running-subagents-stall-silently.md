---
name: long-running-subagents-stall-silently
description: "★2026-08-08 一天内三次 opus/reasoning 长时运行的 workflow lane/agent 卡 85-100 分钟不释放且 status 仍 \"running\"; 拆小 + 短超时 + 早期结构化输出, 别把关键合成堆在最末尾"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

**opus / reasoning 模型下运行 > ~60 分钟的 subagent 或 workflow lane 有相当概率静默 stuck**：transcript 停止更新、无新 tool 调用、TaskOutput 仍返回 `running`。今天一天内三次同样模式：

- `af7aa54f`（single agent，3 任务串行）: 85 min 卡住，最后 tool 调用后无 event，只交付了 45 KB 转录
- `wcd4z00jb`（proposal-triage workflow，5 lane）: 5 lane 都停在同一分钟，report 未产出
- `wvb0oo7sq`（proposal-plan-and-tidy workflow，5 lane）: 5 lane 各写 200-580 KB 但**都没走到 StructuredOutput**，100 min 无新活动

**Why:** 具体原因不确定（模型端超时？转发端断开？agent runtime bug？），但代价对称 —— 大 workflow 死了会丢掉几十万 token 的中间产出。

**How to apply:**

1. **单个 subagent 尽量 < 45 min**。把长任务拆成"研究 → 计划 → 执行"三段，各自独立派单，MAIN 在中间落地一次。
2. **workflow 的 lane 也别太重**。Scan phase 让每个 lane 早期就 `agent(..., schema=X)` 交付一次 structured findings，别把关键合成都堆在 lane 尾部。
3. **不要把"写文件"这一步押在 workflow 最末 phase**。让每个 lane 自己写自己的中间产物到 status/，末尾合成 agent 只做汇总，即使合成挂了也有 lane-level 输出可捞。
4. **判断死活的启发式**：`stat` transcript file 的 mtime + `TaskOutput status`。若 status=running 但 transcript > 45 min 无更新 → 90% 已挂，杀掉不用等。
5. **MAIN 自己能做的（引用检查、读 md、跑单元测试、简单 pip install）就自己做**，不派 subagent。用户明确抱怨过 token 浪费，反复重派大 workflow 更浪费。

相关：[[subagent-prompt-must-state-gpu-budget]]、[[reassign-node-revoke-old-owner-first]]
