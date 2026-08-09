# SKEPTIC3 — 反驳 AUDIT3 的「机制侧仍有可主张的新东西, 且能打败 non-architectural baseline」

日期 2026-08-06 ｜ 角色：adversarial skeptic ｜ 审计对象：`paperC_v2_research/AUDIT3_plasticity_mechanism.md`
被反驳的主张：「机制侧仍有可主张的新东西，且能打败 SAM / shrink-and-perturb / weight-decay 等 non-architectural baseline」
方法：refetch AUDIT3 引用的原文逐句核对 + 我自己的独立搜索轮（S2 Graph API / arXiv 全文检索 / **OpenReview API**，后者 AUDIT3 自己承认没查）

---

## 0. 结论

# **WEAKENED**

AUDIT3 的**否定部分（判决 6.1「不能主张」）我完全同意，且我把它加重了**：我找到了 AUDIT3 漏掉的
**两篇致命占位工作**，它们各自杀死 AUDIT3 保留的两扇门里的一扇半。

但我**没能**找到把 AUDIT3 的**第 2 扇门（语言 vs 知识的分层代价结构）**完全占掉的工作——我用 6 个新 query
搜过，未命中。所以不是 REFUTED。

**具体裁决（AUDIT3 的四扇门）**：

| AUDIT3 的门 | 我的裁决 | 杀手 |
|---|---|---|
| ★★★ 门1「regime 边界：forget-and-relearn 在大数据/大模型上是否消失」 | **REFUTED（死）** | **Alabdulmohsin et al. 2109.00267 §5 已经明确回答了这个问题**："For large datasets, however, reinitialization does not seem to offer a benefit." AUDIT3 引了这篇（§2.3 引它的 flatness），却**没读它的结论**，反而把它已回答的问题列为"文献明确没答"的头号卖点 |
| ★★★ 门2「PPL 能 heal 而知识不能，是 forget-and-relearn 文献里不存在的现象」 | **SURVIVES（唯一活着的）** | 我搜过 6 个 query 未命中。但它**只是 Paper B 的既有资产**，不是 cycling 的新发现（见 §4） |
| ★★ 门3「时机（when in training）是我们真正的自由变量，无人系统扫过」 | **REFUTED（死）** | **Springer et al., ICML 2025, arXiv:2503.19206（68 citations）**已把"破坏的代价 vs 破坏发生的 pretraining 时点"系统扫成一条曲线，还给了机制名字（progressive sensitivity）、OLMo 家族实测、以及一个理论刻画。**AUDIT3 完全没有这篇。** 另有 1905.13277 + 1711.08856（ICLR 2018）已建立"when 比 whether/how 重要"的整个 critical-period 框架 |
| ★ 门4「LLM 缺 replay buffer，故 RL reset 机制不可外推」 | **WEAKENED 到不足以支撑** | 这是一句 related-work 的辩护话术，不是一个可测命题。而且 **DASH (NeurIPS 2024, 2410.23495) Appendix C.1 已经实测**："Table 2 shows L2 INIT (Kumar et al., 2023) and Reset (Nikishin et al., 2022) cannot be a solution in our setting"——即 RL 的 reset 在 stationary 数据下**已被证伪过一次**，我们复述"不可外推"= 复述 NeurIPS 2024 的一张附录表 |

**AUDIT3 的"生死线实验"判断我同意，但它选错了对手。** AUDIT3 说生死线是 shrink-and-perturb 臂。
我认为真正的生死线是 **Springer et al. 的 progressive sensitivity 曲线**：如果"结构破坏的代价随 pretraining
时点单调上升"，那这正是 Springer 已经报告的现象（对 Gaussian 噪声和 fine-tune 都成立），我们只是**换了一种
perturbation 类型（离散层级 vs 连续数值）去复现一条已发表的单调曲线**。reviewer 一句"this is Springer et al.
with a different perturbation operator"就够拒。

---

## 1. 【致命发现 A】门3 已被 ICML 2025 占满：Springer et al., "Overtrained Language Models Are Harder to Fine-Tune"

**AUDIT3 没有这篇。全文 grep `2503.19206` / `Springer` / `overtraining` 在 AUDIT3_plasticity_mechanism.md 里
零命中**（同目录的 SKEPTIC2 有，AUDIT3 没有）。

**venue 核实（我独立做的，非二手）**：S2 Graph API `paper/arXiv:2503.19206` HTTP 200 →
`publicationVenue.name = "International Conference on Machine Learning"`, `type="conference"`, `year=2025`,
`citationCount=68`。**这是 peer-reviewed ICML 2025，不是 preprint。**
（AUDIT3 §1 把好几篇当"作者自述"处理却漏掉这篇真·peer-reviewed 的，是选择性覆盖。）

### 1.1 它把 AUDIT3 门3 的原话逐字做掉了

AUDIT3 §6.2 门3 原话：
> "**本次审计未在任何来源中找到有人系统扫过"结构破坏的代价 vs 破坏发生的训练时点"**……
> 若能把"代价 × 时点"画成一条曲线，这是机制侧最干净的新东西。"

Springer et al. §3.3（标题字面就是 "Warmup: Gaussian perturbations"），原文：

> "We take base models pre-trained to various token budgets and add Gaussian noise of the following form.
> Let [θ] denote the base model weights … where [Σ] is the covariance matrix of the initialization
> distribution of the parameters (prior to pre-training) and [λ] controls the magnitude of the perturbation."

> **"Progressive sensitivity to noise: For a fixed magnitude of perturbation, the change in perplexity between
> the base model and the perturbed model increases monotonically with the number of pre-training tokens."**（§3.3）

**这就是"代价 × 时点"的曲线本身。** 而且注意它的噪声协方差取的是**初始化分布的协方差**——也就是说
"把权重朝随机初始化方向推 λ 那么远"，**我们的"丢 K 层补随机 K 层"就是这个操作在 λ=1 且只作用于后 K 层
的离散极端情形**。它连 magnitude 依赖都扫了（§3.3）：

> "In Figure 3, larger perturbations are associated with a larger and more quickly increasing degradation of the
> pre-training loss. Thus the point at which the degradation from sensitivity surpasses the improvement in the
> base model is accelerated for larger perturbations, leading to an inflection point at a lower token budget."

→ **推论对我们极不利**：我们的 perturbation 是**最大幅度**那一类（完全重置，λ 等价 1）。按 Springer 的
magnitude 依赖，**我们的 inflection point 会出现在最低的 token budget**，即"早期做 cycling 代价小、晚期做代价
大"这条曲线**已经被预测了**，而且预测方向和 brief 里那句猜测（"若 cycling 在训练早期做，代价可能远小于对成品
模型做"）**完全一致**。我们跑出来只会是确认一个已发表的预测。

### 1.2 它不只有 Gaussian 玩具，还有 OLMo 家族的真实数字

Abstract：
> "the instruction-tuned OLMo-1B model pre-trained on 3T tokens leads to over **2% worse** performance on
> multiple standard LLM benchmarks than its **2.3T** token counterpart."

§3.1：
> "When instruction tuning on datasets such as Anthropic-HH and TULU, OLMo-1B models exhibit catastrophic
> overtraining at token budgets exceeding **2.5T** tokens."

§3.1 + §3.2：
> "Under the same fine-tuning and evaluation setups, catastrophic overtraining is **not observed on OLMo-7B**
> models for pre-training token budgets up to 3T tokens (Appendix E)."

⚠️ **最后这句对我们是双重打击**：(1) 它已经在 **OLMo-2-7B**（AUDIT3 §5 说我们的资产就是 OLMo-2-7B）上查过了；
(2) 它报告 7B/3T **看不到效应**——所以我们在 OLMo-2-7B 上做 cycling 想论证"时点效应"，很可能落在
**它已报告为 null 的 regime 里**。

§3 它还明确把 pruning 列为它 future work 的一部分：
> "In this paper, we focus primarily on modifying the pre-trained model by fine-tuning on different datasets.
> To understand catastrophic overtraining, we also study a simple generic modification of adding independent
> Gaussian noise to model weights. **We leave further modifications such as reinforcement learning and pruning
> to future work.**"

→ 这是我们**唯一**能站的位置："把 Springer 的 modification 集合从 {Gaussian, fine-tune} 扩到 {层级结构破坏}"。
但这是**明写在别人 future work 里的一格**，学术上是"填格子"，reviewer 会按 delta 打分而不是按 novelty 打分。

### 1.3 门3 还有第二层占位：critical-period 文献早于它 7 年

**Achille, Rovere, Soatto, "Critical Learning Periods in Deep Networks", ICLR 2018**
（S2 核实：`venue = "International Conference on Learning Representations"`, year 2018, 170 cites；
arXiv preprint 版是 1711.08856 "Critical Learning Periods in Deep Neural Networks", venue=arXiv.org, 124 cites
——**这两条是不同 S2 条目，引用时不要混，AUDIT3 若引要注意**）。

1711.08856 §2（Fig 1C 说明）逐字：
> "**Sensitivity during learning: (C) Final test accuracy of a DNN as a function of the onset of a short
> 40-epoch deficit.** The decrease in the final performance can be used to measure the sensitivity to deficits.
> The most sensitive epochs corresponds to the early rapid learning phase, before the test error (dashed line)
> begins to plateau. Afterwards, the network is largely unaffected by the temporary deficit."

**"损伤代价作为损伤 onset 的函数" = 门3 的定义。这条曲线 ICLR 2018 就画完了**，还带 Fisher-Information 的机制
解释（§3 "Information Plasticity"）。

同组 **Golatkar, Achille, Soatto, arXiv:1905.13277** Abstract 更是把门3 的口号先说了：
> "This suggests that what matters for training deep networks is not just whether or how, but **when** to
> regularize."
> "There is no shortage of literature on what regularizers to use … but, to the best of our knowledge, no work
> has addressed **when** to apply regularization."（§1）

→ 所以门3 的完整占位链是：**ICLR 2018（vision，损伤 onset 曲线 + 机制）→ ICML 2025（LLM 预训练，token-budget
曲线 + 机制 + OLMo 实测）**。AUDIT3 说"未命中任何来源"，是搜索覆盖失败，不是真空。

---

## 2. 【致命发现 B】门1 已被 AUDIT3 自己引用的那篇论文的结论句做掉

这一条是本报告最伤的：**AUDIT3 引了 2109.00267，但只引了它被 LLF 转述的 flatness 那半句，没读它的 §5 结论。**

`grep -n "Alabdulmohsin" AUDIT3_plasticity_mechanism.md` → **只有第 175 行一处**，且那一处是**转引自
2307.01163 的 Discussion**（"Alabdulmohsin et al., 2021" 出现在 active-forgetting 那段引文里）。
AUDIT3 从未把它当成一篇需要直接核对的论文。

**venue 核实（我做的）**：S2 `paper/arXiv:2109.00267` HTTP200 → `venue="arXiv.org"`, year 2021, 24 cites
→ **preprint**（这一点要诚实标注：它不是 peer-reviewed）。但它是 **LLF (ICLR 2022) 自己在 §4.1 与 §A2.4 里
点名比较的 concurrent work**，因此在这个 subfield 内部具有"被 ICLR 论文承认的 baseline"地位。

### 2.1 它是 layerwise reinitialization 的正典，机制解释比 AUDIT3 以为的更全

Abstract 逐字（三条机制全在这一句里）：
> "We also introduce a new **layerwise reinitialization** algorithm that outperforms previous methods and
> suggest explanations of the observed improved generalization. **First**, we show that layerwise
> reinitialization **increases the margin** on the training examples without increasing the norm of the weights,
> hence leading to an improvement in margin-based generalization bounds for neural networks. **Second**, we
> demonstrate that it **settles in flatter local minima of the loss surface**. **Third**, it encourages learning
> general rules and **discourages memorization by placing emphasis on the lower layers** of the neural network."

对照 AUDIT3 的四个候选机制：
- 机制 (b) 结构级正则化 → 它的 margin-bound 论证（§4.1）**比 AUDIT3 找到的任何来源都更形式化**
- 机制 (c) landscape 平坦度 → **它已经做了，而且用的正是 AUDIT3 §5 推荐给我们的那个"便宜探针"**。
  §4.2 逐字：
  > "One method for quantifying flatness is to compare the impact on the training loss when the model parameters
  > are perturbed by some standard Gaussian noise … Figure 5 shows that the solution reached by lw is **more
  > robust to model perturbation** than in standard training. More precisely, for every amount of noise added
  > into the model parameters w, the change in the training loss in lw is smaller than in standard training
  > suggesting that the **local minimum is flatter in lw**."
  → **AUDIT3 §5 把"用 Gaussian 扰动探针测平坦度"列为我们"可行但成本高"的机制 (c) 验证路径。这个实验
  在 layerwise reinit 上 2021 年就做完了，结论是正的。** 我们重做 = 在 LLM 上复现一张 2021 的图。
- 机制 (d) 信息重分布到早层 → 它的第三条机制就是这个，且给了 probe 证据（§2 转述 Baldock/Cohen 的 probe
  结论 + §5 讨论）。它和 LLF §5 的结论是同一件事的两种说法。

### 2.2 门1 的杀手句：它已经报告了大数据 regime 的负结果

AUDIT3 §6.2 门1 原话：
> "**forget-and-relearn 是一个已知的"小数据/易过拟合 regime"方法。LLM 单 epoch 大规模预训练是它的反面
> regime……"它在这里还成立吗" 是一个真问题，且文献明确没答。**"

2109.00267 §5 Discussion 最后一段（**结论段的最后一句**）：
> "Our takeaway message is that the accuracy of convolutional neural networks can be improved **for small
> datasets** using bottom-up layerwise reinitialization, where the number of reinitialized layers may vary
> depending on the available compute budget. At one extreme, one would benefit from reinitializing the
> classifier's head alone, but reinitializing all layers in sequence with rescaling and normalization yields
> better results. **For large datasets, however, reinitialization does not seem to offer a benefit.**"

§3.1（决策树分析）给了同一结论的定量版：
> "First, we observe that lw improves performance across the majority of combinations. **The only exception is
> using ResNet50 with large training sets, in which most methods perform as well as the baseline.**"
（决策树的分裂特征字面就是 `Training Set Size ≥ 35K?`，见 §3.1 Figure 3）

**所以"reinit 的收益在大数据上消失"不是一个开放问题，而是这个 subfield 2021 年就写在 takeaway message 里的
已知结论。** 门1 的正确表述只能是"把一个已知的 regime 边界从 CV(35K 图像) 外推到 LLM(万亿 token) 上再确认
一次"。这在 ARR 尺度上是 **replication，不是 finding**。

而且注意 **AUDIT3 §6.2 门1 已经自己承认"结果可能是负的"**。结合 2109.00267 的 takeaway，
**负结果的先验概率现在非常高，且负结果是"与已有结论一致"，连"意外的负结果"这个卖点都没有。**

### 2.3 顺手纠正 AUDIT3 的一处读漏（不致命但说明尽调不足）

AUDIT3 §2.1 说 LLF "证伪了机制 (d) 的朴素版本"。这我核对过，**LLF §5 原文确实支持**（我 refetch 了
ar5iv 2202.00155，第 193 行"we observe no significant performance improvement in this setting"、第 194 行
"much worse than the version with a different reinitialization each generation"逐字一致）。**这部分 AUDIT3 没读错。**

但 AUDIT3 漏了 LLF §4.1 里紧邻的一句，它直接削弱"LLF 是 layer-reset 的唯一正典"这一框架：
> "Concurrent work (Alabdulmohsin et al., 2021) studies various types of iterative reinitialization approaches
> and propose a layer-wise reinitialization scheme, which proceeds from bottom to top and reinitializing one
> fewer layer each generation. We refer to this method as LW and provide a comparison to their method in
> Table 1. **We find that LLF outperforms LW across all datasets we consider.**"

→ 也就是说 **"迭代式层级重初始化"在 2021-2022 已经有至少两个互相对照过的变体（LLF vs LW），带 12 数据集 ×
4 架构的横扫和显著性检验（2109.00267 Table 3：exact binomial test + Holm 校正）。** 这个设计空间被扫过的
密度远高于 AUDIT3 呈现的样子——AUDIT3 让人以为只有 LLF 一篇。

---

## 3. 【新增发现 C】AUDIT3 漏掉的两个 baseline，其中一个是 ICLR 2026 Oral

AUDIT3 §1 列了 B1–B7 七个 non-architectural baseline。我用 **OpenReview API**（AUDIT3 §7 自己承认
"我未检索 OpenReview / ACL Anthology / DBLP"）搜到两个它没有的：

### C1. FIRE — ICLR 2026 **Oral**（AUDIT3 零命中）

**来源**：OpenReview API `notes/search?term=FIRE Frobenius-Isometry Reinitialization`
→ 两条同名记录：`venue = "ICLR 2026 Oral"` 与 `venue = "CoRR 2026"`（同一工作的会议版 + preprint 版）。
⚠️ **我未能拿到它的 arXiv id，也未抓到全文，以下只据 OpenReview 上的 abstract**（诚实标注：**仅摘要**）。

Abstract 逐字：
> "Deep neural networks trained on nonstationary data must balance stability (i.e., retaining prior knowledge)
> and plasticity (i.e., adapting to new tasks). **Standard reinitialization methods, which reinitialize weights
> toward their original values, are widely used but difficult to tune: conservative reinitializations fail to
> restore plasticity, while aggressive ones erase useful knowledge.** We propose FIRE … FIRE is evaluated on
> continual visual learning (CIFAR-10 with ResNet-18), **language modeling (OpenWebText with GPT-0.1B)**, and
> reinforcement learning (HumanoidBench with SAC and Atari games with DQN). Across all domains, FIRE
> consistently outperforms both naive training without intervention and **standard reinitialization methods**."

**对我们的三重威胁**：
1. **"standard reinitialization methods are widely used"** —— ICLR 2026 Oral 的 abstract 第二句就把
   "reinitialization 恢复塑性"当作**众所周知的既有做法**。这把 AUDIT3 §6.1 第 2 条（"不能主张周期性结构重置
   恢复塑性"）从"2022 起的标准结论"升级为"2026 年 Oral 论文的 background 句"。
2. **它的问题陈述就是我们的 trade-off**："conservative fails to restore plasticity, while aggressive ones
   **erase useful knowledge**"。这句话与我们门2 想主张的"结构破坏在语言能力上可逆、在知识上不可逆"是
   **同一个 trade-off 的另一种措辞**——虽然它没有做 LLM 参数化知识的可分离测量（见 §4），但 reviewer 读到
   我们的动机段会立刻联想到这句。
3. **它评了 language modeling（OpenWebText + GPT-0.1B）**。所以"reinit-for-plasticity 从未在语言建模上做过"
   **不成立**。AUDIT3 §6.2 门1 依赖的"LLF/forget-and-relearn 全在 vision"这个前提，被这篇部分打破
   （规模仍只有 0.1B，这一点对我们仍是活的缝隙，见 §5）。

### C2. DASH — NeurIPS 2024（AUDIT3 零命中；同目录 AUDIT2 有，AUDIT3 没有）

**venue 核实（我做的）**：S2 `paper/arXiv:2410.23495` HTTP200 →
`publicationVenue.name="Neural Information Processing Systems"`, `type="conference"`, year 2024, 5 cites。
**peer-reviewed NeurIPS 2024。**

它对 AUDIT3 的伤害在两处：

**(a) 它把 AUDIT3 的 B4（shrink-and-perturb）升级成一个更强的对手，并且已经在同一张图里打赢了 S&P。**
Figure 1 说明逐字：
> "The Shrink & Perturb (S&P) method involves shrinking the model weights by a constant factor and adding noise
> (Ash and Adams, 2020). Notably, **DASH, our proposed method, achieves better generalization performance
> compared to both training from scratch and S&P, while requiring fewer steps to converge.**"

→ AUDIT3 §4 说"生死线是打赢 S&P"。**实际生死线更高：要打赢 DASH（NeurIPS 2024 已打赢 S&P 的方法）。**
AUDIT3 的对手清单低估了一档。

**(b) 它已经实测过"RL 式 layer reset 在 stationary 数据下无效"——这直接掐掉 AUDIT3 门4。**
§2（Related Work）逐字：
> "However, these explanations and methods diverge from the behavior observed in stationary data distributions.
> **Techniques aimed at mitigating loss of plasticity under non-stationarity are ineffective under stationary
> distributions, as shown in Appendix C.1**, in line with the observations in Lee et al. (2023)."

Appendix C.1 逐字：
> "In this subsection, we describe solutions that aim to mitigate plasticity loss under non-stationarity, which
> cannot remedy the loss of plasticity in an incremental setting with a stationary data distribution.
> **Table 2 shows L2 INIT (Kumar et al., 2023) and Reset (Nikishin et al., 2022) cannot be a solution in our
> setting.**"

§5（实验设置）再确认一次：
> "Appendix C.1 shows that L2 INIT, **Reset**, layer normalization, and reviving dead neurons, **are not
> effective in our setting**. Thus, we conducted the remaining experiments without these methods. Additionally,
> Table 1 shows that warm-starting with SAM does not outperform cold-starting with SAM, indicating that **SAM
> alone is not an effective method in our case.**"

AUDIT3 门4 想主张的是"RL 的 reset 机制解释依赖 replay buffer，LLM 预训练没有，所以不能外推"。
**DASH 已经把这个外推做了并报告失败**（"Reset cannot be a solution"），而且给出的机制解释不是 replay buffer
而是 **noise memorization**（§1："we identify **noise memorization** as the primary cause of plasticity loss
when warm-starting on stationary data"）。所以：
- 门4 的结论（不可外推）**已被 NeurIPS 2024 实证**，我们重复它没有增量；
- 更糟：**DASH 的实证结论是"层级 reset 在 stationary 数据上没用"，而 LLM 预训练正是 stationary/近-stationary
  的。这是一个对我们方向的直接负面先验。**

⚠️ 注意 DASH 的 stationary 设定是"数据集增量增长 + 每轮训到 99.9% train acc"，**与 LLM 单 epoch 流式预训练
不同 protocol，不能直接横比数字**（项目铁律）。但作为**先验和 reviewer 弹药**，它足够有力。
另外 §5 那句 "SAM alone is not an effective method in our case" 反而**对我们有利**——它削弱 AUDIT3 的 B1(SAM)
在 stationary/warm-start 场景下的威胁性。这是我唯一找到的对我们有利的新证据。

---

## 4. 逐条回答任务给我的五个问题

### Q1. AUDIT3 列的"必须打败的工作"里，有没有它读漏/读错的？

**读错（venue 层面）：1 处，且方向是"把 peer-reviewed 说成 preprint"（任务要求的第 4 问）**

| AUDIT3 的判定 | 实际（我 S2 独立核实，HTTP200） | 性质 |
|---|---|---|
| §2.5 + §8：RaPTr 2402.05913 → "S2 未查（本轮 429），arXiv COMMENT 为空 → **按 preprint 处理**" | `publicationVenue.name="International Conference on Learning Representations"`, `type="conference"`, **year=2024**, 10 cites → **ICLR 2024** | **把 peer-reviewed 误标为 preprint**。RaPTr 是 AUDIT3 §2.5 唯一"对我们有利"的先例（"层级扰动能改善 inductive bias"）；把它降格为 preprint **反而削弱了我们自己的弹药**。修正后这条弹药更硬：ICLR 2024 已认可"层级扰动改善 inductive bias 而非只加速"（§concl "improving QA tasks and SuperGLUE by 1-5%"——⚠️ **AUDIT3 写的是 "1.5%"，原文 ar5iv 第 17 行是 "1-5%"，第 294 行是 "1-2% better than baseline and stacking"；这是引数错误，不要照抄**） |
| §8：1910.08475 On Warm-Starting → "**未独立核实**（通常记为 NeurIPS 2020）… 保守按 preprint/未核实标注" | S2 search "On Warm-Starting Neural Network Training" HTTP200 → 首条 `venue="Neural Information Processing Systems"`, `type="conference"`, year 2019, **271 cites**（该条无 arXiv id 字段；1910.08475 在 S2 里是**另一条** "On the Difficulty of Warm-Starting Neural Network Training", venue=arXiv.org, 24 cites）。arXiv abs 页 `citation_title` 确认 1910.08475 的当前标题已是 "On Warm-Starting Neural Network Training" | **可核实但 AUDIT3 放弃了**。结论：**是 NeurIPS 的，B4 是 peer-reviewed**，威胁等级应上调。⚠️ 但 S2 把它拆成两条记录（旧标题 preprint + 新标题会议版），**引用时要注意别引成 24-cite 的那条** |

**读漏（更严重）：3 篇，两篇 peer-reviewed，全都直接落在 AUDIT3 保留的门上**
1. **2503.19206（ICML 2025, 68 cites）** → 占满门3（§1）
2. **2410.23495 DASH（NeurIPS 2024）** → 占掉门4 + 抬高 S&P 生死线（§3.C2）
3. **2109.00267（preprint，但被 ICLR 2022 点名对照）** → 占满门1，且它的 §4.2 已做完 AUDIT3 §5 推荐给我们的
   flatness 探针实验（§2）

**AUDIT3 引对了、我核实无误的部分（给它记功）**：LLF 2202.00155 的全部引文（§4.1 构造、§5 四组对照、
§4.1 CIFAR 负结果）我逐句 refetch 核对，**一字不差**；ICLR 2022 venue 也确认（S2 HTTP200，50 cites）。
LLF ≈ 我们构造这个核心判断，**我无法反驳，它是对的**。

### Q2. 有没有它没搜到的工作？（我的 query 清单，证明我真的试过）

**S2 Graph API `paper/search`（全部重试到 HTTP200 才算；429 我标注）**
- ✅ `The Impact of Reinitialization on Generalization in Convolutional Neural Networks` → total 1，命中 2109.00267 【发现 B】
- ✅ `layerwise reinitialization flatter minima margin generalization` → total 3，首条 2109.00267
- ✅ `Overtrained language models are harder to fine-tune` → total 1，**ICML 2025 confirmed** 【发现 A】
- ✅ `On Warm-Starting Neural Network Training` → total 6573，首条 NeurIPS 271 cites + 命中 **DASH 2410.23495** 【发现 C2】
- ✅ `Critical Learning Periods in Deep Networks Achille Rovere Soatto` → total 5，ICLR 2018 + 1711.08856 + **2308.12221 (ICLR 2023, 深线性网络也有 critical period)** + 2210.04643 (CVPR 2022)
- ✅ `Fortuitous Forgetting in Connectionist Networks` → ICLR 2022 confirmed
- ✅ `Forget forgetting continual learning in a world of abundant memory` → 2502.07274，`publicationVenue=null`（**ICLR 2026 据 2606.24752 参考文献页自述，S2 尚无 venue → 标 preprint/待核**）
- ✅ `Reset It and Forget It …` → 2310.07996 = **ECAI 2023**（AUDIT3 §8 标"未核实"，现已核实）
- ✅ `reinitialization does not help large datasets` → total 4020，前 8 条全噪声（说明这个概念只能靠论文名反查）
- ✅ `cyclic layer reinitialization pretraining language model same depth` → total 9，**无占位**（命中的是 layer-skip/loop 类，与我们无关）
- ✅ `regrowing pruned layers continue pretraining transformer` → total 46，前 8 条**无占位**
- ✅ `layer dropping stochastic depth pretraining BERT efficiency` → total 39，首条 RaPTr（**ICLR 2024 confirmed**）
- ✅ `iterative magnitude pruning regrowth dense LLM pretraining curriculum` → total 1，无关
- ✅ `when to prune during training timing pretraining` → total 11，**无占位**
- ✅ `knowledge recovery layer pruning healing large language model` → total 168，前 8 条全是**压缩向**（LoRAShear / LaCo EMNLP 2024 / E3-Pruner …），**没有一篇把"知识不可恢复 vs PPL 可恢复"当作机制命题** → 门2 未被占
- ❌ 429 未成功：`critical learning periods deep networks damage recovery`、`factual knowledge not recovered after pruning layers healing perplexity`、`perturb then retrain layers language model pretraining improve final quality`

**arXiv 全文检索（strict AND，易假空，我只据"有结果"下结论）**
- ✅ `"catastrophic overtraining"` → total **3**：2503.19206、**2604.13627 "(How) Learning Rates Regulate Catastrophic Overtraining"**、2605.23901（无关，Shannon/scaling）
- ✅ `"progressive sensitivity"` → total 8，全是无关领域同名词（说明这个术语在 ML 里几乎专属 Springer）
- ✅ `"layerwise reinitialization"` → total **1**：2109.00267（**这个 subfield 只有一篇用这个词，AUDIT3 却漏了它**）
- ✅ `"reinitialization" "large language model" pretraining` → total 2，均无关（ChocoLlama / Q-SFT）

**OpenReview API（AUDIT3 §7 明确说自己没查的三库之一）**
- ✅ `layer reinitialization pretraining plasticity` → 命中 **FIRE (ICLR 2026 Oral)** 【发现 C1】+
  "Reinitializing weights vs hidden units for maintaining plasticity in neural networks"（**ICLR 2025 Withdrawn
  Submission** —— 撤稿，不能当占位，但说明"权重级 vs 单元级重初始化"的对照已有人投过）
- ✅ `cyclic depth prune regrow` → **无占位**（只有音频 unlearning 的 prune-and-regrow + 无关的图模型）
- ✅ `periodic layer reset language model` → 命中 "Language Model Alignment with **Elastic Reset**"
  (NeurIPS 2023)——是 RLHF 对齐里的周期性 reset-to-pretrained，**不是层级**，可作 contrast 不算占位
- ✅ `later layer forgetting language model` → **无命中**（支持"LLF 未上 LLM"这一点仍成立）
- ✅ `depth cycling pretraining plasticity` → **无命中**

**⚠️ 我仍未检索**：ACL Anthology、DBLP。所以我不主张"穷尽"。

**据此我能负责地说的"未命中"**（门槛：列出 query + 明说不排除）：
> 我用上列 6 个针对"知识 vs 语言可分离代价"的 query（S2 3 个含 168-result 那条 + arXiv 2 个 + OpenReview 1 个）
> **未命中**任何工作把"结构破坏后 PPL 可恢复而参数化知识不可恢复"作为**机制命题**提出。
> **但不排除**——我有 3 个 query 被 429 打断，且未检索 ACL Anthology / DBLP。

### Q3. 它声称的"我们的差异点"，是否其实是无关紧要的差异？

**是。三个差异点里两个会被 reviewer 判为 minor variant。**

| AUDIT3/brief 的差异点 | reviewer 会怎么说 |
|---|---|
| "层级（不是权重级）" | "2109.00267 §1 已把 reinit 的 mask 选择列成 4 类（welsr/dsd/wels/fc）并新增 lw 层级方案，12 数据集 × 4 架构横扫 + Holm 校正显著性检验；LLF §4.1 是层级；Nikishin §5.1 是层级。**层级不是新轴，是这个 subfield 的默认轴。**" |
| "循环（不是单调）" | "Ash&Adams §4 标题字面是 'Shrink, Perturb, **Repeat**'；2109.00267 的每个方法都是多 round（"reinitialization round"是它 Eq.(1) 的定义单位）；LLF 是 N3/N8/N10 代。**循环是默认，单调（LW bottom-to-top）才是特例。**" |
| "终点回到原尺寸" | "plasticity injection (NeurIPS 2023) 已把 'without changing the number of trainable parameters' 写进 abstract；而且 LLF/LW/DSD **全都**终点原尺寸——因为它们从来没变过尺寸。**这不是差异，这是这一整类方法的共同属性。**" |

**唯一可能不 minor 的差异**：我们丢的是**已训练过的层的信息**并补**位置相同但内容随机**的层，
且做在**万亿-token 单 epoch 预训练**里。但这是 **regime 差异（Q5 讨论），不是构造差异**。
AUDIT3 §0 那句"除非我们把新层的位置/宽度/初始化分布改成与被丢层不同，否则与 LLF 是同一个方法"——
**我同意，而且我要加一句：即使改了，也只是 2109.00267 Eq.(1) 里换一个 mask s 的选法，
而它已经系统扫过 5 种 mask 选法了。**

### Q4. 它有没有把 preprint 当 peer-reviewed，或反之？

**有，两次，方向都是"反之"（把 peer-reviewed 当 preprint）**，见 Q1 表格：
- RaPTr 2402.05913：AUDIT3 "按 preprint 处理" → 实为 **ICLR 2024**
- 1910.08475 Ash & Adams：AUDIT3 "保守按 preprint/未核实" → **NeurIPS**（S2 271 cites 的会议条目）

**没有发现把 preprint 当 peer-reviewed 的情况**——AUDIT3 §8 的诚实标注（SAM 2605.02105 标"作者自述 ICML 2026、
我方未独立核实"；2602.11137 标 preprint）我核对后认为处理得当。
**我补一个独立核实**：S2 `paper/arXiv:2605.02105` HTTP200 → `venue="arXiv.org"`, year 2026, 6 cites
→ **S2 侧目前仍是 preprint 记录，与作者 COMMENT 自述的 "accepted to ICML2026" 不一致。
写作时必须写"arXiv preprint（作者自述 ICML 2026 接收，S2 未收录会议记录）"，不能写 ICML 2026。**
（AUDIT3 的处理方向正确，但它没拿到 HTTP200；我拿到了，结论是"S2=preprint"。）

**AUDIT3 §8 值得表扬的一条**（我核对过，是对的）：它标出 1903.01611 已被作者撤并到 1912.05671，
"**不要引 1903.01611**"。这个提醒有效。

### Q5. 它的"窄缝"是否窄到不足以支撑一篇论文？

**门1 / 门3 / 门4：不是窄，是没有（见 §1、§2、§3）。**

**门2（语言 vs 知识）：缝是真的，但它撑不起"机制侧新东西"这个定位，只能撑起 Paper B 的一个延伸章节。** 理由三条：

1. **它是 Paper B 的既有资产，不是 cycling 的产物。** "PPL 恢复到 1.428× 而 MMLU 只恢复 19.5%" 是对**成品模型
   剪层-heal** 测出来的。cycling 论文若把它当卖点，reviewer 会问"这个发现需要 cycling 吗？"——答案是不需要。
   ⚠️ 并且 AUDIT3 §5 自己已经警告：**`19.5%` 在 paperB/sections/*.tex 里 grep 不到，本审计不背书**。
   我未复核这个数字（不在我的任务范围，也不碰 .tex）。**一个连自己数字都待核的资产，不能当唯一支柱。**
2. **它的 trade-off 措辞已被 ICLR 2026 Oral 的 background 句抢先**（§3.C1）："conservative reinitializations
   fail to restore plasticity, while aggressive ones **erase useful knowledge**"。我们要主张的"可逆的语言能力
   vs 不可逆的知识"是这句话的**可测量化版本**。可测量化是真贡献，但它是**测量贡献**，不是**机制发现**。
3. **它需要的实验是 Paper B 已经做完的那一套**（held-out PPL + MMLU letter/content 拆分 + closed-book QA），
   而 cycling 需要的是**从头/早期开始的多轮训练**。两者的算力落点不同：要把门2 和 cycling 绑在一起，
   必须在 cycling 的每一轮之后都跑全套 5 项 eval，这是 Paper B 全部算力再乘轮数。

**收窄后的精确表述（我认为唯一还能写的东西）**：

> 我们不主张一个新机制，也不主张 cyclic prune-regrow 是一个新方法（它在参数空间等价于 LLF (ICLR 2022) 的
> later-layer reinit，且层级/循环/尺寸守恒三条属性都是 reinit 文献的默认属性）。
> 我们主张的是一个**测量**：在 ≥7B、万亿-token、单 epoch 的 LLM 预训练 regime 下，
> **对同一次层级结构破坏，"分布统计的恢复"与"参数化知识的恢复"是两条不同的曲线**，
> 且这个可分离性是 reinit / forget-and-relearn / plasticity 文献（全部在 vision 分类、RL、
> 或 ≤0.1B 语言建模上）**无法表达**的量。
> 我们把 Springer et al. (ICML 2025) 的 progressive-sensitivity 曲线从
> {Gaussian 噪声, fine-tune} 扩展到 {离散层级破坏}（这是他们 §3 明写的 future work），
> 并报告在这个 modification 类型下，代价的**两个分量随 pretraining 时点的分离速率不同**。

**这个表述的问题（诚实说）**：
- 它是 replication + 一格 future work + 一个新测量轴，**不含新机制、不含新方法**。
- 它必须接受 Springer §3.1 的负面先验：**OLMo-7B 到 3T token 都看不到 overtraining 效应**，
  所以我们在 7B 上很可能测到"两条曲线都还没开始分离"。
- 2109.00267 的 takeaway（"For large datasets, reinitialization does not seem to offer a benefit"）
  意味着**方法侧收益的先验是 0**。所以这只能写成 measurement/negative-result 论文，
  **不能写成"我们提出一个更好的预训练方法"**。

---

## 5. 我尽力反驳但**没能**反驳掉的（给方向记功，避免我这份报告变成单边打击）

1. **LLF ≈ 我们构造，这个判断是对的，我核实无误。** AUDIT3 §0 这条最重的指控站得住。
2. **"forget-and-relearn / later-layer reinit 从未在 ≥1B LLM 预训练规模做过"仍然成立。**
   我的 OpenReview `later layer forgetting language model` 无命中；arXiv `"later-layer forgetting"` （AUDIT3 做的）
   total 1 且是 SEAL；FIRE 的语言建模只到 **GPT-0.1B**。所以规模真空是真的。
   **但真空 ≠ 卖点**：2109.00267 已经说了大数据上没收益，所以填这个真空的预期结果是"确认没收益"。
3. **DASH §5 那句 "SAM alone is not an effective method in our case" 削弱 AUDIT3 的 B1。**
   AUDIT3 把 SAM 列为"最强对手"，但那是在 catastrophic-forgetting 口径下；
   在 warm-start/stationary 口径下 NeurIPS 2024 报告 SAM 无效。**B1 的威胁等级可以下调半档。**
4. **门2（知识 vs 语言可分离）我搜了 6 个 query 没找到占位。** 这是全报告里唯一我攻不下来的点。

---

## 6. 一句话给 MAIN

> AUDIT3 的"不能主张"清单我完全同意并加重；它保留的四扇门里**门1 死于 2109.00267 §5 的 takeaway 句
> （"For large datasets, however, reinitialization does not seem to offer a benefit" —— 而 AUDIT3 引了这篇
> 却没读它的结论），门3 死于 Springer et al. ICML 2025 §3.3 的 progressive-sensitivity 曲线（AUDIT3 完全没有
> 这篇），门4 死于 DASH NeurIPS 2024 Appendix C.1（"Reset cannot be a solution"）**；只有门2（语言 vs 知识
> 分层代价）我攻不下来，但它是 Paper B 的既有资产、数字待自核、且其 trade-off 措辞已被 ICLR 2026 Oral 的
> background 句抢先。
> **裁决 WEAKENED**：这个方向只能作为「Springer et al. future-work 一格 + 一个新测量轴」的
> measurement/negative-result 论文写，**不能作为"新机制"或"新预训练方法"写**；
> 且方法侧收益的文献先验是 **0**（2109.00267 大数据结论 + DASH stationary 结论 + Springer OLMo-7B null）。
> 若坚持做，请先接受"结果大概率是负的且与已有结论一致"。

---

## 7. 附：我独立核实的 venue 表（全部 S2 Graph API HTTP200，非二手）

| arXiv id / 检索键 | 标题（截断） | venue 判定 | 依据 | 与 AUDIT3 的差异 |
|---|---|---|---|---|
| 2503.19206 | Overtrained Language Models Are Harder to Fine-Tune | **ICML 2025** | `paper/arXiv:` HTTP200，`publicationVenue.name="International Conference on Machine Learning"`, type=conference, year 2025, **68 cites** | **AUDIT3 完全没有此篇** |
| 2410.23495 | DASH: Warm-Starting … without Loss of Plasticity | **NeurIPS 2024** | HTTP200，`name="Neural Information Processing Systems"`, type=conference, year 2024, 5 cites | **AUDIT3 完全没有此篇** |
| 2109.00267 | The Impact of Reinitialization on Generalization in CNNs | **arXiv preprint** | HTTP200，`venue="arXiv.org"`, year 2021, 24 cites | AUDIT3 仅在 §2.3 转引，未直接核对；**是 preprint，但被 ICLR 2022 LLF §4.1 点名对照** |
| （检索键，无 arXiv id）| FIRE: Frobenius-Isometry Reinitialization … | **ICLR 2026 Oral** | OpenReview API `notes/search`，两条记录 `venue="ICLR 2026 Oral"` / `"CoRR 2026"`。⚠️ **仅摘要，未抓全文，未取得 arXiv id** | **AUDIT3 完全没有此篇** |
| 2402.05913 | Efficient Stagewise Pretraining via Progressive Subnetworks (RaPTr) | **ICLR 2024** | HTTP200，`name="International Conference on Learning Representations"`, type=conference, year 2024, 10 cites | **AUDIT3 误标 preprint**；且它引的 "1.5%" 与原文 "1-5%"/"1-2%" 不符 |
| 2202.00155 | Fortuitous Forgetting in Connectionist Networks | **ICLR 2022** | HTTP200，type=conference, year 2022, 50 cites | 与 AUDIT3 一致 ✓ |
| （检索键）| On Warm-Starting Neural Network Training | **NeurIPS**（S2 首条 271 cites, type=conference, year 2019） | `paper/search` HTTP200。⚠️ S2 有两条：会议条目（无 arXiv id 字段）与 `1910.08475`（旧标题 "On the **Difficulty** of Warm-Starting…", venue=arXiv.org, 24 cites）。arXiv abs 页 citation_title 已是新标题 | AUDIT3 标"未独立核实/按 preprint" → **应上调为 peer-reviewed**，但引用时须避开 24-cite 那条 |
| 2605.02105 | Sharpness-Aware Pretraining Mitigates Catastrophic Forgetting | **arXiv preprint**（S2 侧） | HTTP200，`venue="arXiv.org"`, year 2026, 6 cites。与作者 COMMENT 自述 "accepted to ICML2026" **不一致** | AUDIT3 方向正确（未背书），但它未拿到 HTTP200；**我拿到了：S2=preprint。写作必须标 preprint** |
| （检索键）| Critical Learning Periods in Deep Networks | **ICLR 2018** | `paper/search` HTTP200，type=conference, year 2018, **170 cites** | AUDIT3 无此篇（门3 的最早占位） |
| 1711.08856 | Critical Learning Periods in Deep **Neural** Networks | **arXiv preprint** | 同上检索，`venue="arXiv.org"`, year 2017, 124 cites。**与 ICLR 2018 条目是不同 S2 记录，勿混** | AUDIT3 无此篇 |
| 2308.12221 | Critical Learning Periods Emerge Even in Deep Linear Networks | **ICLR 2023** | 同上检索，type=conference, year 2023, 14 cites | AUDIT3 无此篇 |
| 2310.07996 | Reset It and Forget It (zapping) | **ECAI 2023** | `paper/search` HTTP200，`name="European Conference on Artificial Intelligence"`, type=conference, 8 cites | AUDIT3 §8 标"未核实" → **现已核实** |
| 2502.07274 | Forget Forgetting: Continual Learning in a World of Abundant Memory | **preprint（待核）** | HTTP200，`publicationVenue=null`, `venue=""`, year 2025, 7 cites。2606.24752 参考文献页自述 "The Fourteenth ICLR"（=ICLR 2026） | AUDIT3 无此篇（它只从 2606.24752 正文看到 "Cho et al. 2026" 的转述） |
| 2602.11137 | Weight Decay Improves LM Plasticity | **未能核实（我本轮 12 次全 429）** | — | AUDIT3 报 S2=arXiv preprint（它拿到了 HTTP200）→ **采信 AUDIT3** |
| 2604.13627 | (How) Learning Rates Regulate Catastrophic Overtraining | **未核 venue**，arXiv 全文已抓 | arXiv `"catastrophic overtraining"` 检索 total 3 命中；Abstract 自述接着 Springer et al. 2025 做 | 两方均无；**Springer 的后续跟进，说明门3 已有 follow-up 生态** |

**未抓到全文、故未据以下结论**：FIRE（仅 OpenReview abstract）、2604.13627（抓到全文但只读了 abstract）、
2308.12221 / 2210.04643 / 2502.07274（仅检索结果，未抓）。
