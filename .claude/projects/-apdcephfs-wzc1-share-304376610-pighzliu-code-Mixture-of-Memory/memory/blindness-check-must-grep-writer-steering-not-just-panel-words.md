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

---

## ★ 2026-08-17：同一个病的第三种外衣 —— 词表查得对，但**换成转述就漏**

今天给 `refreeze_from_manuscript.py` 补了 fail-closed blindness screen（它此前**一个屏蔽都没有**），
词表是 `reviewer|referee|rebuttal|round_0N|NEEDS_REVISION|meta-review|blind review`。
它确实抓住了真泄漏（`s2_02` 6 行、`s2_03` 2 行，含 `"reviewer_verdict": "NEEDS_REVISION"`）。

**但它挡不住转述。** 反证来自去归属工作本身：agent 写的第一版替换措辞是

> `an audit found, **in several independent readings**, that ...`

**这仍然告诉盲审「有好几个人独立读过」**，而我的词表**放行**。实测同样放行的还有：

| 句子 | 泄漏了什么 |
|---|---|
| `four of six independent readers flagged this` | **轮次的人数 + 票数分布** |
| `multiple independent assessors agreed` | 存在评审团 |
| `three of the five reports raised it` | 人数 + 票数 |

**这和 round 00-02 那次是同一个病**：我查的是「评审过程的词汇」，不是「有没有泄漏这件事」。
**「N 个中的 M 个」这种计数构造本身就是泄漏，不管它用什么名词。**

### 修法：加第二个 pattern 查「形状」而不是「词汇」

`_BLIND_SHAPE`：`<N> of <the>? <N> <人称名词>` 以及 `several|multiple|<N> independent <人称名词>`，
名词表**故意收窄**（readers/readings/assessors/evaluators/respondents/panellists/reports/
critiques/referees/opinions），这样普通研究文句不会误报。

**19 个控制**：5 个转述全抓；9 个普通句全放行 —— 包括必须活下来的**真 paperC 正文**
`13 of 27 clear the floor by more than 0.2992 pp`、`0/60 near-unanimous vs 3/12 minority`、
`15 of 15 cells`、`we independently verified`；回归：两个泄漏文件仍 6/2 命中，两个 `_shippable` 仍 0，
19 个 cited path 无新误报。commit `56ea891`。

### How to apply（补强本文件开头那条）

- **写屏蔽器时，词表只是第一层。** 第二层必须问：「**不用这些词**，能不能说出同一件事？」
  能，就补一个查**构造**的 pattern。
- **让别人（agent）去改措辞时，不能只靠屏蔽器验收** —— 屏蔽器的盲区正好是措辞自由度最大的地方。
  这次是 agent 自己 grep 了屏蔽器覆盖不到的归属词才发现，值得记：**它报告了自己的失败**。
- **同族**：[[a-green-checker-covers-only-what-it-targets]]（绿灯只覆盖它瞄准的东西）、
  [[selftest-over-invented-inputs-proves-nothing-about-the-pipeline]]。
  共同点：**通过 ≠ 安全，只等于「我列举的那些没中」。**

## ★★ 2026-08-17 第四种外衣：我为消除**假阳性**而加的边界，制造了一个**假阴性**的洞

`_BLIND_FATAL` 里我写 `(?<![A-Za-z])(?:reviewer|referee|rebuttal)(?![A-Za-z])`。
那个**尾部** `(?![A-Za-z])` 是为了让 `refereed`（"a refereed venue"）不误报 —— 它确实做到了。
**但 `s` 也是 `[A-Za-z]`，所以它把所有复数形式一并排除了。** 实测（我自己复跑确认，不是转述）：

| 句子 | 我的 pattern |
|---|---|
| `the reviewer said` | BLOCK ✅ |
| **`two reviewers asked`** | **PASS ❌** |
| **`the reviewers disagreed`** | **PASS ❌** |
| **`flagged by referees`** | **PASS ❌** |
| **`our rebuttals addressed it`** | **PASS ❌** |

**而复数恰恰是评审语境最自然的说法**（"the reviewers", "two reviewers"）。
`_BLIND_SHAPE` 也盖不住它（它只管 N-of-M 和 "N independent <名词>"）。
我那 19 个控制里**一个复数都没有** —— 因为我在测「refereed 会不会误报」，
没测「reviewers 会不会漏报」。**为修假阳性而加的约束，必须重测假阴性一侧。**

修法（agent 发现并修，commit `4e3c512`；我复跑验证 6/6 复数 block、7/7 look-alike 仍 pass）：
尾部不能简单排除所有字母，要允许 `s`/`es` 等屈折后缀。

### How to apply
- **给屏蔽器加「排除」时，正例一侧要重跑全套**，尤其**屈折变化**（复数/过去式/动名词）。
  经验规则：安全词表加 negative lookahead 后，**至少测该词的复数**。
- **控制集要按「两侧」设计**：假阳性一侧（refereed / interviewer / 13 of 27 cells）
  和假阴性一侧（reviewer**s** / referee**s** / rebuttal**s**）。只测一侧的控制会给出虚假的安全感。
- **附带发现（结构性，未修）**：`refreeze_from_manuscript.py` 的 `manuscript/` 与 `MANIFEST.json`
  是在两道 screen **之后**才写的（`man = DEST/"manuscript"` 在 :421，screen 在 :364/:401），
  所以**它们在构造上不被屏蔽**。今天实测 4 处命中都无害（ICLR 模板自带的
  `Paper under double-blind review` 署名行、manifest 自指的 `round_dir`、
  一处 `Seven of the ten cells reading` 被 `_BLIND_SHAPE` 误判），
  但**正文才是真泄漏最致命的地方，而它恰好没被看**。
