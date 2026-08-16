---
name: an-informational-nonzero-rc-hides-real-defects
description: "paperC 的 gate_exact_floor_tail 文档写「rc=1 是 informational 不算失败」, 于是它连续多轮打印 5 个 hazard 行而正文写「其余七行不受影响」无人对账; 非零 rc 必须要么是失败要么是 0"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

**把某个 gate 的非零退出码定义成「仅供参考、不算失败」，等于让它在 sweep 里永久隐身。非零 rc 要么代表失败，要么就该是 0；「informational」这一档不存在。**

**Why:** 2026-08-16 我跑 paperC 全量 gate sweep，`gate_exact_floor_tail.py` 返回 **rc=1**。
它的 docstring 明确写着：

> `Exit codes: 0 = no row carries the hazard. 1 = at least one row does (informational --
> this is expected and is NOT a failure by itself; ...)`

于是我（和之前每一轮）都把它当成「已知的、预期的」跳过了。可它每一次都在打印 **5 of 9 rows have
stored floor > exact rational**，点名 MMLU-Pro(×2)、**MMLU**、PIQA、**BoolQ**；而 `03b_nulls.tex:39`
同时写着「**the other seven rows are unaffected because their stored floors round down**」——
用 `Fraction` 独立重算后确认：MMLU 的 0.268908 > 3776/14042、BoolQ 的 0.621713 > 2033/3270，
**四个 construct / 五行都是向上取整**，干净的只有 OpenBookQA(恰好相等) + ARC-Easy + ARC-Challenge + CommonsenseQA。

正文**机制说错、结论说对**（MMLU/BoolQ 在 p<1e-5，差一个 count 到不了判定）—— 这正是 reviewer 能抓、
作者无法解释的那种错。**gate 一直知道，只是它的 rc 被我自己定义成了噪声。**

**How to apply:**
- **不要给 gate 设「informational 非零」档。** 想表达「有需要注意的行但不是缺陷」，就 **rc=0 + 打印**；
  真正的缺陷才非零。我这次的修法：gate 现在把**测出的 hazard 行集合与正文的说法对账**，
  矛盾或漏名 → **rc=2**；只有集合一致时才回到 rc=1。
- **gate 打印的清单必须有人跟正文逐项对账**，不能只看 rc。清单类输出（哪些行/哪些 cell/哪些 arm）
  是最容易「打印了但没人读」的东西 —— 与 [[a-green-checker-covers-only-what-it-targets]] 同族：
  那条是绿色掩盖未覆盖的类，这条是**非零被定义成绿色**。
- 改完必须跑双向控制：干净树 rc=1 且无 contradiction；把旧的错句子改回去 → rc=2 且诊断点名。
  控制反了比没 gate 更糟（见 [[rank-local-counters-and-gated-postfix-fake-failures]]）。
- **我自己的修正也要过同一把尺。** 第一版我写「Four of the nine rows」——四个 **construct** 配九个
  **row**，混了单位；`tab_nulls` 九行八 construct（MMLU-Pro 占两行）这个坑
  [[read-what-the-consumer-reads-not-the-bare-key]] 家族里已经记过一次，我又踩了。
  报计数必须说清单位是 row 还是 construct。
