---
name: a-null-outside-the-legal-support-is-not-a-null
description: "paperC 旗舰行的 balanced null 对 12032 题全在 k=10 上抽 gold, 但 2051 题不足 10 选项 → 该标签分配不可实现; 修正后 p 从 0.0 变 0.078 不过 .05, 三个 construct 变两个"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

**一个 null 如果能生成不可能出现的数据，它不是那个 construct 的 null。**「用 construct 自己的 (n,k)」听起来已经够谨慎了，但 **k 恒定是一个额外假设**，而它常常是假的。

**Why:** 2026-08-16，paperC round_04 六个 codex reviewer 里有四个独立指出旗舰 MMLU-Pro 校准的 null 不合法。我自己复核后确认成立：

- 论文的 null 对全部 n=12032 题在全部 **k=10** 个字母上均匀抽 gold letter。
- 但 `n_opt` 直方图是 `{3:21, 4:606, 5:52, 6:93, 7:158, 8:320, 9:801, 10:9981}` —— **2051/12032 (17.05%) 的题不足 10 个选项**，它们的 gold 不可能是 J。该 null 落在**任何合法标签分配的支撑集之外**。
- 正确做法：**保留观测到的 n_opt 直方图**（它是题集属性不是随机量），每题在**它自己的 n_opt 个合法字母**上抽。

| | E[max] | q95 | p |
|---|---|---|---|
| legality-blind k=10（论文） | 0.104457 | 0.107048 | **0.0** |
| legality-aware（正确） | 0.113877 | 0.117188 | **0.078295** |

**判决从 `above balanced null (p<1e-5)` 翻成 `inside estimator noise`**，abstract 的「八个 construct 里有**三个**」变成**两个**，而丢掉的恰是最大 n、用来压 power wall 的那一个。

**诊断线索早就在证据文件里**：同一个 JSON 自己记了 `n_opt_is_not_constant: True` 和 `chance_mean_1_over_nopt=0.110877`（因为 n_opt 可变才需要这个字段），却仍在 null 里用恒定 k=10。**文件自己承认了可变性，代码却没用上** —— 自相矛盾的字段是"这里有 bug"的信号，跟 [[read-env-not-source-defaults-for-running-procs]] 同一族。另一个免费线索：ARC-Easy/ARC-Challenge 的 chance 写成 0.250161 / 0.250156 而非精确 0.25 —— **不精确的 chance 就是 n_opt 不恒定的自供**。

**How to apply:**
- 写任何 permutation / balanced / bootstrap null 前，先问：**这个 null 能生成的每一个样本，都是真实数据可能长成的样子吗？** 不能，就换 null。
- **保留数据的结构约束**（每题选项数、每组样本量、分层权重），只随机化你声称在检验的那一维。
- 修正的**方向性**必须报告：这里 k 恒定的行（MMLU 4-way / BoolQ 2-way）p 完全不变，可变 k 的行只会被推得更深进 noise 桶 —— 所以**没有一行朝作者有利的方向动**。能证明这一点会把「我们犯了错」变成「我们的规则连自己都管」，是加分项，而这正是这篇论文自己在讲的道理。
- 复核这类事**必须两个独立实现 + 多种子 + 一次大抽样**（我用 stratified multinomial 和 per-item categorical，三种子 + 1e6 次，E[max] 一致到 <1e-4，q95 完全相同）。单实现单种子的 p 不足以推翻已发表的表格行。
- ⚠️ **floor 本身不变** —— 它是 gold label 的经验属性，与 null 无关。要显式验证下游那些用 observed floor 的数字（14/15、3/12 vs 1/12）**不依赖** null 的 E[max]，别顺手把它们一起改了。

见 [[a-range-is-not-a-measurement-until-it-clears-its-floor]]、[[selftest-over-invented-inputs-proves-nothing-about-the-pipeline]]。
