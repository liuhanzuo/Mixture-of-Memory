---
name: subagent-prompt-must-state-gpu-budget
description: "★用户2026-08-08: 派任何涉及训练/eval的subagent, prompt必须显式写明「能否用GPU + 哪些节点可用 + 哪些禁碰 + 用前须nvidia-smi自查」; 否则agent会抢占别人的卡"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

派 subagent 做任何可能占 GPU 的活（训练、eval、smoke、benchmark）时，prompt 里**必须**包含一段明确的 GPU 预算声明：

1. **能不能用 GPU**（本任务是纯 CPU / 允许短 smoke / 允许长训练）
2. **哪些节点可用**（写清 IP + 卡型 + 密码文件路径）
3. **哪些节点禁碰**（写清正在跑什么，别 kill）
4. **用之前必须自己 `nvidia-smi` 确认空闲**，不许照抄 prompt 里的台账
5. 若要 kill 别人的任务，必须先回报 MAIN，不许自行决定

**Why:** 2026-08-08 我同时派了两个 agent 到 .73 —— 一个起 keep8 resume（14:32，占 76.6 GiB/卡），另一个 5 分钟后往同节点投 bs16 eval，直接 OOM，5 个 rung 只成活 1 个。两个 agent 各自都没错，错在我派单时没写互斥约束，也没告诉任何一方"这台机器还有别人"。用户随后明确要求：「后面每个训练派 subagent 的时候都要说清楚是否可用 GPU，可用的话可以用哪些节点上的」。

**How to apply:** 在 subagent prompt 里固定加一节「## GPU 预算」，例如：

```
## GPU 预算（必须遵守）
- 本任务：允许 GPU，但仅限 ≤50 步 smoke
- 可用：.21 (28.89.19.21, 8×L20A 183GB, configs/password_b200_19021.txt) —— 已腾空
- 禁碰：LOCAL / .73 / .82 / .104 —— 全部在跑 Paper B resume，kill 会毁掉忠实 resume
- 用前必须自己 nvidia-smi 确认目标节点 0% / 0 MiB，不许照抄本 prompt 的台账
- 需要更多卡时回报 MAIN，不许自行 kill 任何进程
```

相关：[[kill-remote-gpu-job-by-pid-not-pkill]]、[[kill-hung-train-must-exclude-eval]]、[[cluster-two-disks-not-shared]]
