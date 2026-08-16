---
name: retained-means-the-null-is-retained
description: "我把 reviewer 的「olmo2/keep14 is RETAINED」读成「结论保住了」, 于是虚构出两个 reviewer 结论相反的争议; 多重检验里 retained = 零假设保住 = 该格没通过校正"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

**多重检验语境里 "the cell is RETAINED" 指的是【零假设被保留】，即该格 *没有* 通过校正。** 不是「这个发现保住了」。

**Why:** 2026-08-16 paperC round_04：R-claude-stats 的 V13 和独立 meta-review 的 §3.7 都写了 BH over 27 cells → 拒 6 个，`olmo2/keep14` (p=0.0172) **RETAINED**。我把它读成「keep14 留下来了 = 支持 bidirectionality」，而 meta 说它撑不住 claim，于是我在 handoff 里写下：

> "Two reviewers, opposite conclusions, same 27 numbers. 我不从摘要裁决，需要没有立场的第三方重算。"

**根本没有争议。** 两份评审字面一致，我自己重算（tie-aware BH step-up）也一致：BH 拒 6，`olmo2/keep14` q=0.0663 不过。我凭一个反向的词义虚构出一场分歧，还差点为它派一个 adjudicator agent。

**同一次核实里我发现了真正该修的东西（而它被那场假争议挡住了）**：两位 reviewer 的**摘要句都错**——都写「neither trace-signal cell survives」，但 BH 下 `qwen3/k14` (p=0.0066, rank 6, 阈值 0.011111, **q=0.0297**) **是过的**；只有 Bonferroni 下两个都不过。两人各自的逐 rank 枚举都写着 rank 6 拒，**自相矛盾于自己的摘要**。照抄任一句进论文就会印一句假话。

**How to apply:**
- 见到 `retained` / `rejected`，先确认主语是**零假设**还是**结论**。统计语境里 reject = 有信号，retain = 没信号，**与日常语感相反**。
- **声称两个 reviewer 冲突之前，把两边的原句并排读一遍。** 我是从自己的摘要里读出的冲突，而摘要是我写的 —— 冲突在我的转述里，不在证据里。
- **裁决工作的第一步是重算，不是找第三方。** 算术是机械的、可核的；我先算就会立刻发现无争议，省掉一次 agent 派发。
- **reviewer 的摘要句和它自己的明细表可能不一致；明细优先。** 逐条枚举是它真算过的，摘要是它事后概括的。这与 [[read-what-the-consumer-reads-not-the-bare-key]] 同源：读真正承载事实的那一层。
- 两个独立 reviewer 犯**同一个**精度错误，不代表那是对的 —— 更可能是同一个直觉捷径（"borderline 的都活不下来"）。

见 [[reviewer-observation-right-attribution-may-be-mine]]、[[state-direction-only-for-rows-you-computed]]。
