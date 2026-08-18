# [SUPERSEDED] Elastic Scaffold-Coder 论文 Proposal

> **STATUS: SUPERSEDED — 2026-08-08**
>
> 本文档只保留为原始设计和撤回过程的历史记录，不再代表当前论文主张、
> 实验优先级或 GPU 启动依据。
>
> 当前方向以：
>
> - `DLLM_SALVAGE_ROADMAP_20260808.md`
> - `proposal/active/A01-execution-grounded-typed-repair/PROPOSAL.md`
>
> 为准；完整结果更正与 retraction 时间线见
> `DLLM_RESULTS_20260807.md`。
>
> 已失效或停止的部分包括：competitive full-program generation、
> cross-family low-cost Pareto、shared-head `[expand]/[delete]` 长度控制，
> 以及在当前弱 Scaffold checkpoint 上优先推进 Adaptive runtime 的最小论文
> 计划。静态 capacity 对同一 checkpoint failure shape 的影响仍可作为内部
> 系统观察，但不能作为跨家族 headline。

## 0. 工作标题

英文标题候选：

> **Elastic Scaffolds for Diffusion Code Models: Capacity-on-Demand Structured Decoding**

备选标题：

> **Executable Scaffolds for Diffusion Code Models: Capacity, Termination, and Structural Revision**

中文标题：

> **面向扩散代码模型的弹性可执行脚手架：按需结构容量、终止与结构化修订**

项目简称：

```text
Elastic Scaffold-Coder
```

---

## 1. 一句话论文主张

> 结构化 dLLM 的性能不仅由模型 checkpoint 决定，也由 executable
> runtime 提供的结构容量和终止策略决定；固定的小容量 runtime 会形成
> “简单任务极快成功、困难任务运行到上限”的长尾，而按需扩容可以在
> 保留低成功样本 NFE 的同时降低 step-limit failure，并改善全任务的
> 功能正确率—计算量 Pareto frontier。

如果后续结构化修订实验成功，可将主张扩展为：

> 可执行脚手架还提供了天然的程序级修订单元：通过 verifier 定位错误
> 子树并执行合法的 forward collapse，可以比整程序重启或平面 token
> remask 更高效地进行功能修复。

---

## 2. 研究背景与动机

### 2.1 dLLM 代码生成面临的三个特殊问题

与自然语言相比，完整程序生成同时要求模型决定：

1. **内容：** 变量、表达式、API 和算法；
2. **拓扑：** 函数、循环、条件和嵌套关系；
3. **长度：** 程序行数、statement 数和表达式 token 数。

普通 masked diffusion LM 将这些决策全部放在固定的平面 token
canvas 中。已有工作通常从以下方向介入：

- 重新设计 masking/noising schedule；
- 调整 token unmask 顺序；
- 使用 CFG/grammar constraint；
- 通过 expand/delete 支持可变长度；
- 允许 token remasking 或 verifier-guided revision。

这些方法很少把：

```text
结构表示
动态长度
终止
局部修订
```

统一到同一个可执行状态空间中。

### 2.2 Scaffold-Coder 已实现的核心机制

我们已经实现：

- typed structural meta-token；
- line-level 和 token-level 两种 mask 粒度；
- deterministic template expansion；
- line-slot body planning；
- rule-emitted indentation、colon 和 newline；
- typed vocabulary constraints；
- `[expand]/[delete]` 弹性编辑；
- forward collapse / reverse expansion；
- training-consistent subtree backtracking；
- full-sequence bidirectional model calls。

这些机制使 dLLM 的中间状态不只是 token 序列，而是一个可执行、可渲染、
可扩容、可折叠的程序树。

---

## 3. 关键观察：固定 runtime 容量产生非单调的质量—成本曲线

### 3.1 我们的 Stage-1 Scaffold

| Benchmark | Plus pass@1 | Generation failure | 成功样本平均 NFE |
|---|---:|---:|---:|
| HumanEval+ | 18.29% | 9.15% | 58.79 |
| MBPP+ | 32.01% | 9.26% | 47.77 |

将失败样本按 512 NFE 计入后，全任务平均 NFE 约为：

```text
HumanEval ≈ 100
MBPP      ≈ 91
```

### 3.2 同一 checkpoint 的 64-task 容量筛选

固定 `global_step_4465`、固定 64 个按 canonical depth/length 分层选择的
HumanEval+ 任务，只改变 runtime 容量：

| Runtime | HE+ | Failure | 成功平均 NFE | 主要终止原因 |
|---|---:|---:|---:|---|
| Tiny | 0.00% | 34.38% | 15.60 | depth capacity exhausted |
| Small | 3.13% | 37.50% | 44.03 | depth capacity exhausted |
| Medium | 10.94% | 0.00% | 69.55 | resolved |
| Large | 10.94% | 10.94% | 62.61 | model-call budget |

Medium 与 Large 的 HE+ 相同，但 Medium 没有 generation failure，且按当前
统计的全任务平均 NFE 更低。这说明“容量越大越好”并不成立：容量不足会
造成表达失败，而过大的可编辑空间又可能形成 expand/delete 或未决 mask
长尾。

### 3.3 当前证据边界

该筛选是同 checkpoint、同任务、同模型调用预算下的纯 runtime 对照，
因此可以将 failure 形态的变化归因于 runtime 配置。不过它仍然只是
64-task screening，且 Tiny/Small 的早停失败成本在第一版汇总中被统一按
512 NFE 计入。完成 partial-cost instrumentation 和全量 HumanEval+/MBPP+
复现前，不能声称：

```text
Medium 在所有 benchmark 上都严格支配其他固定容量
```

目前可以说的是：

> 静态结构容量是 Scaffold decoding 的一阶变量。过小容量导致明确的
> depth exhaustion；过大容量则可能引入 model-call 长尾；存在一个
> 中间容量区间，在当前筛选集上同时改善功能结果、终止可靠性和成本。

这一非单调现象是本 proposal 的直接实验出发点。

---

## 4. 核心研究问题

### RQ1：固定 runtime 容量如何影响质量—计算曲线？

研究以下容量参数：

- initial root/body slots；
- header/statement 初始 token slots；
- max tree depth；
- max lines per body；
- max total lines；
- max tokens per hole；
- module-level expand；
- expansion budget。

关注：

- pass@1；
- parseability；
- generation failure；
- 成功样本 NFE；
- 全任务 NFE；
- cumulative model tokens；
- wall-clock。

### RQ2：是否存在 failure-heavy 的小容量 regime？

验证以下假设：

> 小容量 runtime 的成功任务 NFE 很低，但 failure rate 随任务长度和
> 深度快速升高；只报告 successful-only NFE 会系统性夸大其效率。

### RQ3：按需扩容是否能支配固定容量？

设计 adaptive runtime：

```text
从小容量开始
→ 监测 capacity pressure
→ 只在需要时增加 slot/depth/token budget
→ 保留简单任务的低 NFE
→ 避免困难任务跑满 step limit
```

目标是相对：

- Fixed-Small：大幅降低 failure；
- Fixed-Large：降低平均 canvas / cumulative tokens；
- 同时保持或提高 pass@1。

### RQ4：终止策略如何影响长尾？

比较：

- 固定 hard step limit；
- repeated-state cycle suppression；
- no-progress patience；
- capacity expansion；
- forced delete/pass；
- confidence/entropy plateau termination；
- verifier-assisted termination。

### RQ5：结构化 revision 能否把结构优势转化为语义收益？

在失败或低置信度 subtree 上：

```text
定位错误 subtree
→ collapse subtree to one mask
→ 保留其他上下文
→ 局部重生成
```

比较：

- full restart；
- random span remask；
- lowest-confidence token remask；
- AST subtree collapse；
- verifier-localized subtree collapse。

### RQ6：runtime 效应是否跨 checkpoint 泛化？

当前先在 `global_step_4465` 上完成完整容量实验，再训练一个低学习率、
一轮的同 schema hierarchical checkpoint：

| Checkpoint | Small | Medium | Large | Adaptive |
|---|---:|---:|---:|
| Stage-1 `global_step_4465` | 可立即运行 | 可立即运行 | 可立即运行 | 实现后运行 |
| Low-LR 1 epoch | 训练后运行 | 训练后运行 | 训练后运行 | 训练后运行 |

两者共享 tokenizer、结构 token 和 runtime schema，因此可以做真正的
checkpoint × runtime 交叉实验。

---

## 5. 方法设计

## 5.1 Fixed capacity family

### Fixed-Tiny

故意设置为强约束的低容量端点：

```text
initial_root_slots        2
initial_body_slots        1–2
initial_statement_masks   1
header_masks              1
max_tree_depth            1
max_lines_per_body        2
max_total_lines           16
max_tokens_per_hole       2
module_expand             false
```

该端点用于测量低成本快速完成与结构表达失败之间的 trade-off，不作为
推荐配置。

### Fixed-Small

```text
depth              2
lines/body         4
total lines        32
tokens/hole        8
module expand      false/on 两个版本
```

### Fixed-Medium

```text
depth              4
lines/body         16
total lines        64
tokens/hole        32
module expand      on
```

### Fixed-Large

当前 Stage-1 默认：

```text
max_tree_depth       16
max_lines_per_body   128
max_tokens_per_hole  512
module expand        on
```

---

## 5.2 Capacity pressure instrumentation

每个样本记录：

- `line_capacity_hits`；
- `token_capacity_hits`；
- `depth_capacity_hits`；
- `total_line_capacity_hits`；
- `module_expand_suppressed`；
- `expand_budget_hits`；
- `edit_cycle_breaks`；
- `repeated_canvas_count`；
- `no_progress_calls`；
- unresolved masks 数量曲线；
- canvas length 曲线；
- tree depth 曲线；
- line count 曲线；
- 每个 subtree 的 confidence/entropy；
- 最终 failure reason。

这一部分是 adaptive runtime 的基础，也是解释 failure-heavy tail 的关键
证据。

---

## 5.3 Adaptive capacity policy

### 初始容量

从 Fixed-Tiny 或 Fixed-Small 开始。

### 扩容触发信号

触发任意一项时扩大相应局部容量：

1. 模型预测 `[expand]`，但当前 line/token cap 阻止执行；
2. 同一 canvas 重复出现；
3. 连续 `p` 次模型调用 unresolved mask 数未下降；
4. 当前 subtree 的平均 entropy 高于阈值；
5. 当前 body 所有 slots 已占用且仍预测结构/expand；
6. verifier 显示程序缺少要求的函数或 return；
7. 到达深度上限，但模型持续预测 compound construct。

### 扩容动作

优先局部扩容，而不是全局直接切换到 Large：

```text
line slots       ×2
token slots      ×2
depth cap        +1
total line cap   +8 / +16
module expand    on-demand enable
```

### 扩容预算

限制：

- 每个 subtree 最大扩容次数；
- 全程序最大扩容次数；
- 最大 cumulative model tokens；
- 最大 wall-clock；
- 最大 tree/canvas hard cap。

---

## 5.4 Termination policy

终止输出必须区分：

```text
successfully resolved
capacity exhausted
cycle detected
no progress
model-call budget
invalid final render
```

建议策略：

### T0：Hard limit

现有 baseline。

### T1：Cycle-aware

现有 repeated-state suppression。

### T2：Capacity-before-termination

检测 no progress 后，先扩容一次；扩容仍无进展才终止。

### T3：Graceful forced resolution

在只剩 optional slot 时强制 delete；空 required body 渲染 `pass`。

### T4：Verifier-assisted

若程序已 parse 且满足入口函数、签名和 return 等静态条件，则允许提前
终止；否则触发局部扩容。

---

## 5.5 Structural revision

第一阶段不把 revision 作为主贡献门槛，避免 paper scope 过大。

若 adaptive capacity 已经建立稳定优势，再加入：

```text
test/static verifier
→ subtree localization
→ training-consistent collapse
→ local regeneration
```

Revision 必须报告：

- 新增 pass 数；
- regression 数；
- extra NFE；
- extra cumulative tokens；
- 被删除的 subtree 大小；
- 定位 precision；
- repair success rate。

---

## 6. 实验设计

## 6.1 Benchmark

### 第一阶段

- HumanEval / HumanEval+；
- MBPP / MBPP+。

### 第二阶段

- BigCodeBench；
- LiveCodeBench post-cutoff；
- HumanEval-Infilling multi-line；
- 自建 deep-control-flow slice；
- code repair / failing-test repair benchmark。

### 为什么需要复杂度 slice

容量效应预计与以下因素强相关：

- canonical solution 行数；
- compound depth；
- token 长度；
- function 数；
- loop/if 数；
- import/helper 需求；
- 生成树实际深度；
- 实际 slot 使用量。

---

## 6.2 Baseline

必须包含：

1. Dream-Coder 64 / 128 / 512 NFE；
2. Plain SFT 64 / 128 / 512 NFE；
3. Scaffold Fixed-Large；
4. Scaffold Fixed-Tiny；
5. Scaffold Fixed-Small / Medium；
6. Adaptive Scaffold；
7. CFG/grammar-only baseline，如可实现；
8. full restart / token remask baseline，用于 revision。

---

## 6.3 统一成本口径

所有表格同时报告：

| 指标 | 原因 |
|---|---|
| pass@1 | 功能结果 |
| parseability | 结构结果 |
| generation failure | 终止可靠性 |
| successful-only NFE | 简单成功样本成本 |
| all-task mean NFE | failure-adjusted 成本 |
| median / P90 NFE | 长尾 |
| rule-only steps | runtime 开销 |
| cumulative model tokens | 不同 canvas 长度下的可比成本 |
| wall-clock | 实际性能 |
| peak memory | 系统成本 |
| max canvas / tree depth / lines | 容量使用情况 |

论文中不能只报 successful-only mean NFE。

---

## 6.4 关键消融

### E1：容量曲线

```text
Tiny → Small → Medium → Large
```

### E2：单因素消融

固定其他变量，分别扫：

- depth；
- line cap；
- token cap；
- total lines；
- module expand。

### E3：Adaptive vs fixed

比较：

- Fixed-Tiny；
- Fixed-Large；
- Adaptive。

### E4：终止策略

比较 T0–T4。

### E5：复杂度分层

按 canonical depth、长度和 runtime pressure 分层。

### E6：Checkpoint 泛化

至少增加一个同 schema checkpoint：

- 当前 Stage-1；
- 低学习率一轮 hierarchical checkpoint，或不同 seed。

### E7：Revision

仅在 adaptive 主线通过后运行。

---

## 7. Reviewer 会觉得有趣的结果

## 7.1 最理想的核心图

横轴：

```text
all-task cumulative model tokens 或 wall-clock
```

纵轴：

```text
Plus pass@1
```

点的颜色：

```text
generation failure rate
```

点的大小：

```text
parseability
```

若 Adaptive 同时位于 Fixed-Tiny 和 Fixed-Large 的 Pareto frontier 外侧，
这是最有说服力的 headline。

## 7.2 Failure-heavy tail 图

绘制每任务 NFE 分布：

```text
成功任务低 NFE 峰
+
step-limit 失败任务高 NFE 峰
```

比较 Tiny、Large 和 Adaptive。

若 Adaptive 能显著减少 510/511 NFE 的失败峰，同时保留低 NFE 成功峰，
reviewer 会认为这是一个清晰且普适的系统发现。

## 7.3 Capacity pressure 与任务复杂度

展示：

- canonical depth 与 depth-cap hit；
- solution length 与 line-cap hit；
- capacity hit 与 failure probability；
- cycle/no-progress 与 step-limit failure；
- adaptive expansion 次数与最终 pass。

## 7.4 同 checkpoint 的纯 runtime 结果

最关键的因果证据必须来自同一个 checkpoint：

| Checkpoint | Tiny | Small | Medium | Large | Adaptive |
|---|---:|---:|---:|---:|---:|
| Stage-1 | ... | ... | ... | 18.29/32.01 | ... |

这比两个独立实现之间直接比较更可信。

---

## 8. 主要贡献

如果实验成功，论文贡献可写为：

1. **Executable scaffold state**
   - 将 dLLM 状态推广为可增长、可渲染的程序树。

2. **Failure-adjusted analysis**
   - 揭示静态小容量 runtime 的成功任务低 NFE 与失败任务高 NFE 双峰
     现象。

3. **Capacity-on-demand decoding**
   - 以局部 pressure signal 动态扩大结构和 token 容量。

4. **Termination-aware structured decoding**
   - 联合处理 cycle、no-progress、capacity exhaustion 和合法强制收尾。

5. **可选的 training-consistent revision**
   - 使用 forward-reachable subtree collapse 进行局部修订。

---

## 9. Related Work 与当前新颖性边界

> 本节按 2026-08-08 的文献与仓库结果更新。它说明旧 proposal 为什么不再是
> 当前主线；正式 typed-repair proposal 的完整来源审计见
> `proposal/active/A01-execution-grounded-typed-repair/SOURCES.md`。

### 9.1 Code-specialized dLLM

Dream-Coder（arXiv `2509.01142`）和 DiffuCoder（`2506.20639`）已经覆盖
code-specialized masked diffusion、any-order generation、generation-order
分析与 diffusion-native RL。DreamOn（`2602.01326`）进一步覆盖 variable-
length code infilling 和 expand/contract 状态。

因此旧 proposal 不能再把以下内容当作单独新意：

- code dLLM；
- any-order infilling；
- generation-order control；
- expand/delete 式可变长度。

### 9.2 Dynamic structure 与 grammar constraints

CFG-constrained diffusion decoding（`2508.10111`）以及 EPIC
（`2606.00722`）已经研究如何让并行 diffusion 输出满足 context-free
grammar。它们主要解决 partial sequence 是否仍可完成为合法字符串。

旧 Scaffold runtime 的差异原本是：

```text
typed recursive tree
deterministic multi-line rendering
capacity/termination telemetry
subtree collapse
```

但仓库结果表明，结构合法性没有自动转化为程序语义；因此这个差异只能说明
runtime 资产，而不能支持 full-generation superiority。

### 9.3 Flexible length 与 dynamic decoding

DreamOn 已直接覆盖 code infilling 的 variable-length canvas。Dystruct
（`2605.09820`）、DiSE（`2603.02760`）以及 stability/adaptive decoding
工作也在研究动态长度、结构和停止。

因此：

- “何时扩容”本身已不是空白；
- Adaptive capacity 必须相对最佳 fixed point 实测节省至少 20% 成本且不损失
  超过 1pp 质量，才可能保留；
- 当前弱 checkpoint 上不应优先做 Adaptive。

### 9.4 Remasking、editing 与 self-correction

Targeted Remasking（`2605.26436`）、SCOPE+D3IM（`2606.01026`）、
MRP（`2605.18817`）、Edit-Based Refinement（`2605.09603`）、
Multi-Block Editing（`2607.22663`）和 Speculative Correction
（`2608.02625`）已覆盖：

```text
token-to-mask
visible-token revision
insert/delete/replace
local/global reopening
draft-then-refine
```

Detect-Remask-Repair（`2606.12807`）还显式研究 localized repair 与
preservation trade-off。

因此旧 §5.5 的“subtree collapse 后局部重生成”不能仅凭 remask/revision
成为新意。当前可辩护的空缺已经收窄为：

> execution-grounded、typed-AST/subtree-localized、显式测量外围代码
> preservation，并与 strong AR/FIM 和 full regeneration 比较的程序修复。

### 9.5 Program repair 与 localization

Graph2Diff、Beep、DEAR、AutoCodeRover、CodePilot、SHERLOC、Loc2Repair、
SiblingRepair 与 MultiFixer 已研究 AST/graph diff、fine-grained fix
localization、SBFL、execution-guided search 和 multi-hunk repair。

所以不能主张：

- 首次 AST/tree repair；
- 首次 execution-guided repair；
- 首次 localization→repair decomposition；
- 首次 multi-site repair。

当前 typed repair 的可能区别是把已定位 subtree 返回 masked-diffusion 的原生
状态，并把外围代码 bit-exact preservation 设为 primary metric。

### 9.6 Evaluation protocol

CaRE（`2607.24763`）和 `Diffusion Language Models: An Experimental
Analysis`（`2606.19475`）已经指出 dLLM 结果高度依赖 sampler、compute
budget 和 inference protocol。当前仓库还实测：

- HumanEval+ sampler plausible spread 26.8pp；
- cross-benchmark ranking transfer 很弱；
- NFE 不能作为 AR/diffusion/Scaffold 的公平跨家族成本；
- success-conditioned cost 会漏掉最贵失败。

因此任何未来方法比较必须先冻结 sampler，并同时报告 token/context/wall-time
成本，而不是继续使用旧 proposal 的 NFE-centric framing。

### 9.7 旧 proposal 的最终定位

本 proposal 最多保留两类价值：

1. static capacity 会改变同一弱 checkpoint 的 failure shape；
2. typed runtime、stable anchors、C1/C2/C3 primitive 和 telemetry 是可复用
   工程资产。

它不再支持：

- competitive full-program generation；
- absolute low-cost frontier；
- shared-head edit-token length controller；
- Adaptive-first 论文路线。

---

## 10. 风险与止损标准

### 语义质量目标

> **历史段落说明：** 以下是旧 proposal 的 gate 设计，不是当前启动计划。
> 其中 baseline 数字已被后续 sampler audit 更新，训练路线也已由
> `proposal/active/A01-execution-grounded-typed-repair/PROPOSAL.md`
> 取代。

Runtime 优化不能替代 checkpoint 语义能力。旧 proposal 当时使用的
Dream-Coder Instruct 目标线是：

```text
HumanEval+ = 50.00%  # historical protocol
MBPP+      = 65.08%  # historical protocol
```

后续固定仓库 run 在另一 sampler 协议下达到约 70.7% / 68.0%，进一步证明
该目标线不能脱离 sampler protocol 使用。

为跨越该目标，后续训练不再沿用“Base 初始化 + 五轮全参数 hierarchical
SFT”。该配方已经表现出明显的窄域过训练和语义遗忘。新的训练路线为：

1. 从 Dream-Coder Instruct 初始化；
2. 移除已被 matched control 否定的 depth-banded schedule；
3. 使用以 leaf/content 为主的 rung mixture；
4. LoRA 只适配结构接口，并只训练新增结构 token 行；
5. 必要时加入 frozen-teacher KL/replay，锚定 ordinary-code 分布；
6. 最后用独立可执行题库进行 diffusion-native execution RL。

后续实验已经证明当前 shared-head full-generation 路线不应继续；本段不再
定义当前论文成功条件。

## 10.1 Runtime 主线继续标准

Adaptive 相比 Fixed-Tiny 至少满足：

- generation failure 降低 15 个百分点；或
- Plus pass@1 提升 5 个百分点。

Adaptive 相比 Fixed-Large 至少满足：

- Plus pass@1 不下降超过 1 个百分点；
- cumulative model tokens 或 wall-clock 下降 20%。

若两项都不满足，则 capacity-on-demand 不足以作为主贡献。

## 10.2 Revision 继续标准

在独立 64-task repair set 上：

- 新增 pass ≥ 5；
- regression ≤ 2；
- pass delta ≥ 5 个百分点；
- extra cumulative tokens ≤ 35%。

否则不将 revision 放入主论文。

## 10.3 泛化标准

至少满足一个：

- 两个同 schema checkpoint 上方向一致；
- HumanEval+ 和 MBPP+ 都显示 failure-tail 改善；
- 在 BigCodeBench/deep slice 上有一致结果。

---

## 11. 预期论文结构

1. Introduction
2. Related Work
3. Executable Scaffold Runtime
4. Static Capacity and Failure-Heavy Tails
5. Capacity-on-Demand Decoding
6. Termination and Optional Revision
7. Experiments
8. Analysis
9. Limitations
10. Conclusion

---

## 12. 最小可发表版本

最小论文版本可以建立在两个同 schema checkpoint 上：

1. 在 `global_step_4465` 上完成 Tiny/Small/Medium/Large sweep；
2. 证明静态容量产生质量—成本—failure trade-off；
3. 实现 Adaptive；
4. 证明 Adaptive 优于至少一个 fixed endpoint；
5. 在 HumanEval+ 和 MBPP+ 上复现；
6. 提供 failure/complexity 分析；
7. 在低学习率一轮 checkpoint 上验证容量结论不是单 checkpoint 偶然现象。
