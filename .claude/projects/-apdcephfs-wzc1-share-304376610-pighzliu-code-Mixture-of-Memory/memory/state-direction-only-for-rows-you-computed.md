---
name: state-direction-only-for-rows-you-computed
description: "我从 MMLU-Pro 一个方向推出「修正不朝有利方向动任何一行」, 但 n_opt>k 的项方向相反(ARC 实测 -17σ/-23σ); 且 P(X>=stored_floor) 因 6dp 舍入排除了观测本身, 把 p 从 0.083 压到 0.078"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

**只为你真正算过的那些行陈述方向。** 我把 MMLU-Pro 的一个方向一般化成了全表结论，而它在另一类项上是**反的**。

**Why:** 2026-08-16 我修 paperC 的 legality-aware null 时写下：

> "the fix is monotone … **no row moves in the paper favour**；legality-aware null 抬高 E[max] → 抬高 p → 只可能把行推得更深进 noise 桶"

**这是假的。** 我从 MMLU-Pro 的情形（多余项 `n_opt < k`，少选项 → 抬高 E[max] → 抬高 p）推广到全部。但 `n_opt > k` 的项**方向相反**：概率质量被分给一个几乎处处非法的字母，它基本不可能赢得 max → **压低 E[max] → 压低 p** → 把行推向"survives"，即**对作者有利**。

我自己用独立 toy 验了机制（n=2000、nominal k=4、扫 n_opt=5 的比例）：

| frac(n_opt=5) | E[max] | q95 |
|---|---|---|
| 0.00 | 0.261571 | 0.272000 |
| 0.05 | **0.259010** | 0.269500 |

ARC 两行实测 dp = **−0.0025 (−17σ)** 和 **−0.0063 (−23σ)**（8 seed × 1e6 draws），是真效应不是 MC 噪声。**结论仍然成立**（两行都不接近 0.05），但那句 blanket claim 不能印。

**同一轮还漏了第二个独立缺陷（不是我发现的）：floor 的舍入约定。** floor 精确值 `1403/12032 = 0.116605718085`，以 6 位小数存成 `0.116606` 后**更大**（差 2.819e-07）。于是 `P(max ≥ 0.116606)` 要求计数 ≥ **1404**，**观测到的 1403 被排除在它自己的尾事件之外**。单侧 p 值 `P(X ≥ x_obs)` **必须包含 x_obs**。我报的 `p=0.078295` 偏低，正确是 **0.083**；影响 9 行里的 5 行，方向同样偏向"survives"。

**两个错误都让我自己的论证看起来比实际更整齐** —— 这是最该警惕的错误方向。

**How to apply:**
- 声明「修正是单调的 / 不会朝有利方向动」之前，**对每一类项分别算符号**。我手上已经有能跑的 sampler，只是没跑。参数化的量（这里是 `n_opt` 相对 `k`）**跨越阈值时符号会翻**。
- **存储精度会改变离散分布的尾事件成员资格。** 比较 `P(X ≥ threshold)` 时用**精确有理数**（`Fraction(1403,12032)`）而不是四舍五入的小数；断言 `x_obs` 满足该事件。
- **派 agent 时说「verify，不要 repeat」**。这次是 agent 的 sign self-test 在我的 claim 失败时**拒绝写出输出**，错误才浮出来。一个被要求「应用 MAIN 的修正」的 agent 会把两个错误一起写进论文。
- 被 subagent 纠正时，**独立验证机制再接受** —— 不能因为它在纠正我就直接采信，那和因为它支持我就采信是同一种失职。

见 [[a-null-outside-the-legal-support-is-not-a-null]]、[[one-sample-is-not-a-trend-or-state]]、[[a-range-is-not-a-measurement-until-it-clears-its-floor]]。
