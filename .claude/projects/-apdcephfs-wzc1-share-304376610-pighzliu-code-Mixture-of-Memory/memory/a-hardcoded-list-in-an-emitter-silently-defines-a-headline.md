---
name: a-hardcoded-list-in-an-emitter-silently-defines-a-headline
description: "paperC 的 rollup.damaged_rungs=[keep12,keep10,keep8] 同时驱动 14/15 和 0/15 两个 headline; 补回 manuscript 自己声明的两个 arm 后 0/15 变 9/25, 而 near-unanimity 正是它用来替代 multiplicity correction 的东西"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

**证据 emitter 里的一个硬编码列表会静默定义论文的 headline 分母，而没有任何检查断言它与正文声称的集合一致。** 这个模式 2026-08-16 一天内出现**三次**。

**Why:** paperC 的 `evidence/mmlu_scale_power/mmlu_pro_power_nulls_v2.json` 里 `rollup.olmo2_7b.damaged_rungs` 就是一行 `[keep12, keep10, keep8]`。它同时驱动**两个** headline：

| | 论文报的 | 补回 manuscript 自己列为 damaged 的 arm 后 |
|---|---|---|
| MMLU-Pro | **14/15** at-or-below floor | shortgpt16 = **+3.6735 pp, p=0.0001, "above the floor"** |
| off-MMLU | **0/15** above floor | **9/25** |

`shortgpt16` 在 **5/5** 个小 benchmark 上显著清过 floor（+11.208 到 +49.663 pp，全 p=0.0001）；`keep14` 在 **4/5** 上（+6.470 到 +18.687 pp）。而被计入的三个 arm 全在 floor ±5 pp 内、14/15 n.s.。**这不是边缘重分类。**

`04_experiments.tex:8` 明确把这两个 arm 列为 OLMo-2 的 arm；`09a_relocated.tex:24` 承诺被排除者要「named at the point of exclusion together with its own floor delta」；`04_experiments.tex:23` 声明计数「cannot be inflated by selecting on outcome」。**三条都被这一行硬编码违反了。**

**为什么它比同轮的 null 错误严重**：null 错误只把一个表行从「above balanced null」改成「inside noise」，代价是摘要里一个词。这条打的是 **aggregate 的 near-unanimity**，而 `09a_relocated.tex:26` **明确把 near-unanimity 当作 family-wise multiplicity correction 的替代品** —— 替代品本身塌了。

**同一天的另两次**：(1) `k=10` 的 balanced null 与同文件的 `n_opt_is_not_constant: True` 并存（见 [[a-null-outside-the-legal-support-is-not-a-null]]）；(2) 我的 freeze 调用手写 `--evidence` 只打包 2/24（见 [[reviewer-observation-right-attribution-may-be-mine]]）。**共同结构：一个不会被校验的字面量决定了一个被广泛引用的量。**

**How to apply:**
- **任何被正文引用的分母/计数，必须有 gate 断言它 == 正文声明的集合**，且基数 == arms × benchmarks。不一致退出非 0。一条断言能同时抓住 14/15 和 0/15 两个实例。
- 看到 `damaged_rungs` / `arms` / `cells` 这类**列表字面量**在 emitter 里，就问：**正文有没有在别处独立地声明过这个集合？两处一致吗？** 论文里写「the designation is fixed by construction」时，代码里的那行才是真正的 construction。
- **发现一处立刻 grep 同族**：同一个文件里 `llama2_7b` / `llama3_8b` / `qwen3_8b_base` 的 `damaged_rungs` 也要核。见 [[fix-the-class-not-the-instance]]。
- ⚠️ **这类发现不一定是坏消息。** shortgpt16 保留 16/32 层、keep14 保留 14/32，都比 keep8/10/12 多 —— **「保留更多 stack 的 arm 清过 floor、更少的没清」是单调的，它证明 floor test 追踪 capability 而不是对什么都判失败**，比一个无解释的 0/15 更有说服力。先算单调性再决定怎么写；一条缺陷可能能改写成正面结果。**但不许为了好看而选披露方式。**
- **主动披露严格优于被 reviewer 挖出来** —— 这次就是被挖出来的。

见 [[read-what-the-consumer-reads-not-the-bare-key]]、[[selftest-over-invented-inputs-proves-nothing-about-the-pipeline]]。
