---
name: blindness-check-must-grep-writer-steering-not-just-panel-words
description: "paperC round 00-02 把作者侧 EVIDENCE_PACK.md 打进盲审包(含「attacking it would be an easy referee kill」); A-07 的词表全是审稿过程词, 泄漏行一个都不含, 且它只在恰好无该文件的 round_03 上验过"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

**盲审检查必须同时 grep 两类泄漏：(i) 暴露评审过程的词，(ii) 暴露作者隐藏计划的祈使句。第二类才是真正泄出去的，而且危害更大。**

**Why:** 2026-08-16 实测，paperC 的 round_00/01/02 盲审包里都有 `evidence/EVIDENCE_PACK.md`，md5 `4d0013b52eed`、77766 B，**与作者侧 `tcodex_out/` 那份逐字节相同**。它包含：

- line 649 `DO NOT claim differential learning rates on either side.` —— 作者知道有缺陷并指示别提
- line 690 `attacking it would be an easy referee kill` —— **直接告诉 reviewer 往哪打、以及作者打算怎么绕**
- line 758 `the corresponding limitation L1 in the writer prompt` —— 暴露 writer prompt 存在
- line 548 `MUST NOT be claimed (each was claimed at some point ... and then killed)` —— 完整自我证伪史 + 战术

**round 00–02 的分数作废**，不得进任何 median / ceiling / trajectory / 轮间改善。

**那份"已验证零泄漏"的断言（A-07）怎么放过的 —— 三个独立失效，每个都值得记：**

1. **词表族错了。** 八个 grep 词全是**审稿过程**词汇（`reviewer` / `verdict` / `overall_score` / `panel median`…），一个**writer-steering 祈使句**都没有。泄漏行 line 649 对 `reviewer` 和 `verdict` 命中数都是 **0**。更阴的是：那些词**确实命中了**（`verdict` 16 次、`reviewer` 2 次），而检查把每个命中都正确地解释为「论文自己的词汇（结果表有 Verdict 列）」—— 于是**它靠正确地排除错误的命中，产出了绿色结论**。
2. **只在恰好无风险的那个实例上验过。** A-07 标题写的是 round_03，而 round_03 打包了 **0 个** evidence 文件 —— 该 vector 在那轮**根本不存在**，所以 grep 空洞地为真。它据此写下关于 `freeze_round.py`「by rule 排除 `tcodex_out/`」的**类级结论**，而这在三轮前就已为假。
3. **排除规则存在但没作用在关键路径上。** `tcodex_out/` 确实在排除列表里；v1 只把列表应用到 manuscript 闭包，`--evidence` 循环完全绕过。**规则正确、未被执行。**

A-07 的 "Residual risk" 段**方向写反了**：它精准点名了这个文件，然后从一个未核实的前提推理 —— 「reviewer 读不到它（已被排除）」。reviewer **读得到**，因为它被打包了。

**How to apply:**
- 盲审 gate 的 pattern 必须含 **`DO NOT` / `MUST NOT` / `referee kill` / `writer prompt` / `easy .* kill`** 这类**大小写敏感**的祈使句（`we do not claim` 是论文正当措辞，`DO NOT claim` 是 steering；靠大小写区分）。
- **在有该 vector 的实例上验证**，不是在恰好没有的那个上。「grep 返回空」要先证明**被检查的东西真的在树里**，否则是空洞真。见 [[selftest-over-invented-inputs-proves-nothing-about-the-pipeline]]。
- 排除列表要**在打包完成后对最终产物再扫一遍并 assert**，不能只在某一条输入路径上过滤。
- **命中后判"这是正当词汇"要逐条记录判据** —— 那正是本例产生假绿的地方。
- 一个 agent 在做别的事时发现比自己 brief 更大的问题、如实上报且不自行改动任何 reviewer 分数 —— **这是正确处理**，要在落账里明确 credit。

见 [[a-hardcoded-list-in-an-emitter-silently-defines-a-headline]]、[[reviewer-observation-right-attribution-may-be-mine]]、[[repo-checkers-are-writers-not-probes]]。
