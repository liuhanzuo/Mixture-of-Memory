---
name: long-agents-must-write-incrementally-not-at-the-end
description: "paperC stats lane 三次失败: 两次 workflow lane 卡死, 一次 54 次工具调用 50 分钟后死于 API 错误, 全部零产出; 派长任务必须要求首几次调用就建文件并逐步 append"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

**长任务 agent 必须被明确要求「增量写盘」。默认行为是「做完再写」，而做不完就等于什么都没做。**

**Why:** 2026-08-16 paperC round_04 的 stats lane **失败三次**：

1. workflow lane 卡死（`no progress for 180000ms` × 6 次尝试）
2. 同上，第二个 lane
3. 单独派 agent：**54 次工具调用、50 分钟**，然后 `API Error: API returned an empty or malformed response (HTTP 200)` —— **盘上零产出**

同批的另外两个 lane（soundness / adversary）跑了 28 和 42 分钟成功交付。所以不是任务本身不可完成，而是**长任务的失败概率会累积，而"做完再写"把全部产出押在最后一刻**。

同一天已有的相关但不同的教训：[[agent-output-must-be-persisted-to-the-consumers-file]] 讲的是 **workflow 的 return 值没落盘**（跑完了但值丢了）；这一条讲的是 **agent 没跑完就死，中途结论从未落盘**。前者是交付路径错，后者是**交付时机**错。

**How to apply:**
- 派任何预期 >20 分钟的 agent，prompt 里必须写死三条：
  1. **头几次工具调用内**就创建输出文件，含 header + `## WORK IN PROGRESS` 标记；
  2. **每完成一项核实就 append**（重算了什么、是否吻合、当前判断）；
  3. 只在最后重构成要求的 schema。
- 明确告诉它：**「一份写下了五个已核实数字的残缺产物，远胜于零」**，以及「只存在于你 context 里的发现不存在」。
- 加一句**预算取舍指令**：「宁可少做几项但都写下来，也不要做一个你可能活不到交付的穷尽计划」—— 否则 agent 会先规划 15 项检查，死在第 8 项。
- 重派时**把前次失败如实告诉它**（几次调用、多久、什么错误、零产出）。它会据此调整策略，我实测这比只说「请增量写盘」有效。
- MAIN 侧：任务失败后**先 `ls` 输出文件**再决定是否重派 —— 有残件就接着做，没有才从头。

见 [[long-running-subagents-stall-silently]]、[[agent-output-must-be-persisted-to-the-consumers-file]]。
