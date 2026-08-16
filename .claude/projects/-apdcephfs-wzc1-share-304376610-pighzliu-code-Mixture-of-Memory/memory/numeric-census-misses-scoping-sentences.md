---
name: numeric-census-misses-scoping-sentences
description: "paperC C9 重构时一句「That is a loophole-closing construction, not a new statistic」被静默删掉而 numeric-token census 报 0 loss —— 划界句不含数字, 必须另做逐句 verbatim 比对"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

**`numeric-token census` 通过 ≠ 内容没丢。** 论文里最要紧的一类句子恰恰**不含数字**。

**Why:** 2026-08-16 实施 paperC 的 C9 页数方案时，一次「收紧文字」的改写静默删掉了两句：
- **"That is a loophole-closing construction, not a new statistic"** —— 这是论文**给自己贡献划界**的那句话；
- **"as a measured property … not as an identity"** —— 把一个结论限定为实测而非恒等式。

两句都不含数字，所以 census 报 **0 loss / 0 decreased**，看起来完全无害。是一次**逐句 verbatim 比对**才抓到的。
删掉前者会让论文听起来在宣称一个新统计量（overclaim）；删掉后者会把实测性质说成恒等式 —— 正是本项目反复踩过的
「k=3 时 Spearman ρ=+1 是算术恒等式不是证据」那类错误的镜像。两句已复原到附录。

**How to apply:**
- 任何为压页数/精简而做的重写，**census 之外必须再做一次逐句 verbatim diff**（旧文件 vs 新文件+附录的并集），
  专盯这三类无数字句：**贡献划界**（"we do not claim…" / "not a new X"）、**认识论限定**（"measured property, not an identity"、
  "lower bound"、"under assumption Y"）、**否证条件**。
- 反向也成立：这类句子被删是**审稿人最会抓的 overclaim 来源**，比丢一个小数点更致命。
- 相关但独立的一条：同轮还发现**页数缺口可能是结构性的**——C9 除了搬内容，还把
  `\subsection{Auditing the audit}` 降级为 `\paragraph{}`；没做这一步时，无论怎么收紧文字都到不了 8.857。
  **改写前先做 parity 测试**（把参考方案的原样 stub 代进自己的树重建），能立刻区分「文字太松」还是「结构没照做」。

见 [[reporting-a-gap-is-not-closing-it]]、[[a-range-is-not-a-measurement-until-it-clears-its-floor]]、[[repo-checkers-are-writers-not-probes]]。
