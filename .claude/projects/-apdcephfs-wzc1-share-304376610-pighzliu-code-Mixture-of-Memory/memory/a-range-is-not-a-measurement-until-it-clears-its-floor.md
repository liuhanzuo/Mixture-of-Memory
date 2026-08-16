---
name: a-range-is-not-a-measurement-until-it-clears-its-floor
description: "★★两个 range 的比值在两者都没超过各自 noise floor 时是 undefined 而非'方向'; 且 E[range of k]/σ 的常数与 k 强相关(k=2:1.1284 / k=3:1.6925687506432689 / k=8:2.8472), 用错 k 会让 floor 偏低 40.6% 并翻转布尔判定"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

**2026-08-13 一天之内因为这一条废掉两个任务（A04 内），两次都是「算术完全正确但结论不成立」。**

## 规则

**一个 range / margin / 差值，在它超过自己的 noise floor 之前，不是一次测量。**
不能拿它算比值、不能读它的符号、不能说它「方向相反」。它是 undefined。

反例（真实发生）：within-arm LR 任务算出 cluster1 的 range 是 cluster2 的 **0.2576×**，
我读成「比值 < 1，所以 H_LR 被证伪」。实际上 cluster1 在**四个轴上全部低于自己的 floor**
（0.44× / 0.20× / 0.29× / 0.17×）→ **分子本身不是测量**，比值 undefined。
verdict 从 `REFUTED` 改成 `UNRESOLVED_SUBNOISE`，我的两次「已证伪」声明全部撤回。

复算证明我的**算术是对的**：`reproduction_vs_archive` 8/8 cell 与 archive 逐位相同。
**错的不是数字，是我没有 gate。**

## noise 常数是 k 的函数，不是通用值

`E[range of k iid N(0,σ)] / σ`：

| k | 常数 | 来源 |
|---|---|---|
| 2 | **1.1284** | `2/√π` |
| 3 | **1.6925687506432689** | `3/√π`，闭式 |
| 8 | **≈2.8472** | Monte Carlo，无闭式 |

**拿 k=3 的 1.6926 去 gate 8 个点，floor 会低 40.6%**，足以把 `False` 翻成 `True`。
我曾一整天把 1.6926 当成「那个 noise gate」到处用。**用几个点就查几的常数，并在报告里写明 k。**

另一个同类陷阱：**range 会随点数增长**（8 点的 range 天然大于 3 点的），
所以**不同 k 的 range 之间不能直接比大小**，必须各自除以自己的 c_k 之后再比。

## How to apply

- 任何 `ratio = range_A / range_B` 之前，**先分别 gate A 和 B**；任一未过 → 结论是
  `UNRESOLVED_SUBNOISE`，**不是** 「A 小于 B」。
- prereg 里就把 c_k 表和「用哪个 k」写死（`A04_SHALLOW_RUNG_LADDER_PREREG.md` §4.2 是范本，
  它还明文写了「用 k=3 的值去 gate k=8 会让 floor 低 40.6%」作为守卫）。
- 报「X 比 Y 大/小」时，同时报 **两者各自的 floor 与倍数**，不只报比值。
- 训练 ppl / loss **不是** decision axis，不能拿来代替 gate 过的 margin。
- 与 [[one-sample-is-not-a-trend-or-state]] 是同一族错误的两个面：那条讲**时间维度**
  （一次采样 ≠ 趋势/状态），这条讲**幅度维度**（一个差值 ≠ 一次测量）。
  另见 [[numpy-version-split-breaks-cross-node-bootstrap]]（同一分析换节点会漂 5e-3 pp，
  比某些硬校验阈值还松）。
