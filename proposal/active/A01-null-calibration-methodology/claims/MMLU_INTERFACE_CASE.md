# MMLU Interface Case

## Surviving claim

受损模型上，answer-letter scoring 可能退化为 input-blind predictor；在使用其
模型间差异前，必须先证明每个 arm 显著高于 best-constant floor。

### OLMo

- best constant：always-D `0.2689`
- 三个 arm 显著低于该 floor，三个 arm 与其不可区分
- content interface 在十个 arm 上均高于自己的 `.2845` floor

### Qwen

极端受损 arm 的 failure mode 不是大量 tie，而是 letter prior collapse：
近乎总是输出同一个 letter。这支持“instrument failure 的实现机制随家族变化”。

## 已撤回

- “两个有效 interface 会翻转模型排序”。
- “fails below chance” 作为标题。正确对象是 best-constant floor，不是 `.25`。

## 独立成篇 gate

> **⚠️ 2026-08-11 重写。旧版本原文保留在下方 `SUPERSEDED` 里，不得静默删除。**
> 依据：用户 2026-08-11 指令「没必要晋升条件卡这么死，只要有发现就可以继续做」，
> 以及一个具体的自相矛盾 —— 旧条件 1 已被本 proposal 自己的数据判为**不可能通过**。

**当前状态：不设「必须三项全过」的准入门。有经核实的发现就继续做。**

已经站得住的（可以直接写）：

* **四个家族**都复现了核心 outcome —— 结构损伤下 letter 摸不到自己的 floor：
  OLMo-2-7B `0.2550`、Llama-2-7B `0.2295`–`0.2415`、Llama-3-8B `0.2329`–`0.2527`、
  Qwen3-8B-Base `0.2286`–`0.2301`，floor = always-D `0.2689`，n=14042/臂。
  → 旧条件 2「第三个模型家族」**早已超额满足**（有四个，不是三个）。
* **机制是家族特异、outcome 是一般的**：OLMo 走 bf16 exact ties（keep8 上 `30.64%`），
  Llama/Qwen 走直接 modal collapse（Llama-2 k12 = **100.0% modal / 0.00% ties**）。
* **fp32 已排除「数值 bug」这个平凡解释**（见下）。

还值得补、但**不是准入条件**：

* **第二个 MC benchmark 的完整复现。** 目前 BoolQ / OBQA 是点状证据
  （BoolQ 6 臂里 4 臂无法与常数预测器区分；OBQA residual fraction 在 chance 线下
  被夸大 `2.15×`），gate-2 的 winogrande 是结构性退化、只能当 negative control。
  补齐它会让 claim 更硬，但缺它**不阻止**继续推进。
* 更多家族 / 更多 benchmark：边际收益递减，按卡的空闲情况排。

### SUPERSEDED（旧版原文，保留以备复核）

> 只有同时完成下列三项，才从 A01 拆为独立 paper：
> 1. full-fp32 forward 消除 ties，并恢复 letter validity；
> 2. 第三个模型家族复现；
> 3. 第二个 MC benchmark 复现。

**为什么旧条件 1 必须废除 —— 它把最好的结果判成了不达标。**

`../GATE3_VERDICT.md` 的实测 verdict 是 **`MECHANISM_FALSIFIED`**：

* ties **是**纯 bf16 artifact —— fp32 在两个臂上移除了 **100%**；
* 移除后改变了受损臂 **18.03%** 的 letter argmax 决策；
* 但 letter accuracy **完全没动**：Δ = **−0.0015**，CI95 [−0.0064, +0.0033]，
  exact McNemar **p = 0.570**；
* 受损臂在 fp32 下**更显著**地低于自己的 floor（−1.54 pp, boot p = 0.0062，
  对比 bf16 的 −1.39 pp, p = 0.0192）。

旧条件 1 要求「消除 ties **并恢复 letter validity**」。validity 没恢复，
所以按字面读这条门**永远不可能通过** —— 它隐含假设了「这个缺陷是个可修的数值 bug」。
而实际结论正好相反，**且更强**：

> 一个能被 fp32 修掉的缺陷只是工程注记；修不掉的才是构念效度问题。
> Reshuffling 2,532 coin-flips does not recover information that is not there.

所以 fp32 实验的正确角色是 **claim 的支撑证据**（排除了平凡解释），
不是**晋升的前置条件**。

**通用教训（写进论文的方法论部分）**：预设 gate 是在知道答案之前写的。
当结果落在「既不是 H1 也不是 H2、而是第三种情况」时，要改的是 gate，不是丢掉发现。
这与本 proposal 已有的元教训同源 —— 撤回 #2 之所以被反撤回，
也是因为它**测了错误的条件**（intact base，而 clause 说的是 damaged）。

