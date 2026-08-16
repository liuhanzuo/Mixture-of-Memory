---
name: continue-agent-with-sendmessage-not-agent
description: "★2026-08-11 同一轮内犯两次: 要接续一个在跑的 background agent 必须用 SendMessage(to=agentId), 用 Agent() 会另起一个新 agent; 两次共烧 342k token 且第二次撞上 edit war 风险"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

要给一个**已在运行的** background agent 追加信息，工具是 **`SendMessage(to='<agentId>')`**，不是 `Agent(...)`。`Agent()` 永远**新起**一个 agent，它不继承那个 agent 的上下文，也不知道任务已经有人在做。

**2026-08-11 我在同一轮对话里犯了两次**：
1. 第一次连 prompt 都没填，直接提交了 `Agent(prompt="placeholder")` → 新 agent `ac1258…` 白跑，**111,078 token**。
2. 被自己抓到后，我写好了正确的正文，却**又一次调了 `Agent` 而不是 `SendMessage`** → 新 agent `a54f10…`，**230,991 token**，且它抵达时发现 task #249 已有 agent 在写文件（相差 12 秒），只能 stand down。合计 **342k token**。

**为什么危险，不只是浪费**：第二个 agent 差点和 #249 的正主同时写 `proposal/` 下重叠目录 —— 这正是 [[reassign-node-revoke-old-owner-first]] 里那场 30 分钟编辑战的成因。它是靠自己检测到重复派发才避开的，不是靠我。

**How to apply**：调 `Agent` 之前先问一句「这是**新任务**，还是**接续一个还活着的**任务？」
- 接续 → `SendMessage(to=agentId)`。agentId 在派发时的返回里，也可以从 `/tasks` 找。
- 新任务 → `Agent(...)`，且**提交前确认 prompt 正文已写完**。

**副产品（别浪费掉）**：那两个误派 agent 各自独立复核了集群状态，其中第二个纠正了我一个实质错误 —— 我以为 `.82` 的 seed45 因 `max_steps=300000` 不会自停、需要人工 kill；实际 `scripts/_run_a03_dataorder_repl.sh:74-85` 有自带 watcher，每 60s 轮询 `step220000.pt`，命中即 `kill -TERM` + `kill -9`，且因为它以**checkpoint 存在**为触发条件，ckpt 必然已落盘。**教训叠加：说「这个 job 需要人工干预」之前，先读它的 wrapper 脚本，别只看 trainer 的 `--max_steps`。**

相关：[[long-running-subagents-stall-silently]]、[[subagent-prompt-must-state-gpu-budget]]、[[reassign-node-revoke-old-owner-first]]。
