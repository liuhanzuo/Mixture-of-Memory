# SKEPTIC2 — 反驳 AUDIT2「层级循环 prune-regrow 相对 DSD/RigL/LTH 有足够增量」

**角色**：adversarial skeptic。默认 AUDIT2 的主张是错的，尽力找证据反驳。
**审计日期**：2026-08-06
**审计对象报告**：`paperC_v2_research/AUDIT2_cyclic_and_weight_level.md`

---

## 0. 结论（三选一）

> # **WEAKENED**
>
> 但是**接近 REFUTED**，且**必须收窄到与 AUDIT2 建议的卖点几乎完全不同的一个东西**。

三条独立的致命打击，AUDIT2 全都漏了或读反了：

| # | 打击 | 证据 |
|---|---|---|
| **K1** | **AUDIT2 声称"层级 + 循环 + 训练期 + 终点原尺寸 从没人做过（只有 filter 级 RePr）"—— 这是错的。有两篇 peer-reviewed 论文（ICLR 2022 + CVPR 2023）做的就是逐字这个操作，且它们把"层级"作为核心卖点。** | **2202.00155 (ICLR 2022) 的 LLF**；**2304.04858 (CVPR 2023) 的 SEAL**；以及 **2109.00267** 的 `lw`（AUDIT2 把它当"非循环、非层级"打掉了，实际它就是层级循环算法） |
| **K2** | **AUDIT2 的核心论据「RePr Table 3 证明结构级(6.9) > 权重级 DSD(7.8)」是对原文的误读。RePr 原文明说结构级 vs 权重级 ≈ 打平（7.7 vs 7.8），全部增益来自 metric（ortho），不是来自粒度。** → AUDIT2 用来支撑"换粒度会换机制"的唯一定量证据反向了 | 1811.07275 §Table 3 讨论原文 |
| **K3** | **这一整支文献的收益已被原作者们实测限定在「小数据 / 会过拟合 / 多 epoch」制度内，并在大数据、强架构、迁移场景下明确报告"无收益"甚至"变差"。而 LLM 预训练恰好是这支文献自己划出的失效区（单遍、超大数据、不过拟合）。** | 2109.00267 §5 Discussion："For large datasets, however, reinitialization does not seem to offer a benefit."；2202.00155 附录 A2.4/Table A8："we do not see any improvements from LLF over our baselines"；2304.04858 摘要：LLF 特征 "degrade the transfer learning performance across all datasets we explored" |

外加两条 **AUDIT2 漏检的 LLM-规模先例**，直接拆掉它的两条"边界"：
- **K4（拆"规模边界"）**：**2303.10464 SPDF（UAI 2023，peer-reviewed）** 在 **1.3B GPT-3 XL** 上做 sparse-pretrain → dense-finetune（= DSD 的 LLM 版），AUDIT2 §4 明说"`dense-sparse-dense` 全库只有 6 条命中，没有一条是 transformer/LLM，这是本次审计最强的空白证据" —— 该"最强空白证据"是**检索措辞造成的假空白**。
- **K5（拆"plasticity 文献在 LLM 侧无干预"）**：**2503.19206 Springer et al.（ICML 2025，peer-reviewed）** 在 **OLMo-1B（3T tokens）+ OLMo-2-7B + Amber-7B** 上做了 "catastrophic overtraining"，并给出机制（progressive sensitivity）+ 理论；**2502.07274 Cho et al.（ICLR 2026）** 在 **>1B** 上给了维持塑性的算法。AUDIT2 §5-Q3 断言"没有任何一篇在 ≥1B LM 上做过任何 plasticity 干预并报告下游能力"，**这是错的**。

**同时必须承认（这是为什么不是 REFUTED）**：K1 的三篇全是 **CNN / 图像分类 / 小数据**，且它们 reset 的是 **"最后 k 层" 或 "block k 以上全部"**，**没有一篇是"从深度栈中间挖掉 K 层、总深度不变、在 ≥1B decoder-only LM 预训练上"**。所以技术上仍有一条缝。但这条缝在 K2+K3 之后**从"新机制"降级为"一个已知在小数据 CNN 上有效、在大数据上被原作者报告失效的技巧，在一个新制度里的 negative-or-null 复现"**。

---

## 1. K1 — AUDIT2 漏了/读反了三篇「层级循环」论文（这是最重的一条）

AUDIT2 §5-Q1 的原话：

> 「五个条件里，前四个的组合 **RePr 在 CNN filter 级已经做过（A1）**」
> 「**粒度边界**：RePr 是 filter 级，layer 级从未有人循环剪-补。」

AUDIT2 §5-Q3 洞 1 的原话：

> 「**"从中间挖掉 K 层再补 K 个新层、总深度不变"这个具体操作，plasticity 文献里没有。**」

### 1.1 LLF（Later-Layer Forgetting）— 2202.00155，**ICLR 2022（S2 http200 已核实）**

**venue 核实**：S2`publicationVenue.name = "International Conference on Learning Representations"`, `type=conference`，DBLP `journals/corr/abs-2202-00155`。→ **peer-reviewed ICLR 2022，不是 preprint。** 原始 JSON 落盘 `paperC_v2_research/_venue_raw_skeptic2/2202.00155.json`。

**它就是层级 + 循环 + 训练期 + 终点原尺寸。** §3（Targeted Forgetting）原文定义：

> "Based on the observations in Baldock et al. (2021), we hypothesize that by **reinitializing the later layers of the neural network**, we can remove information associated with difficult examples more precisely than the mask criteria used in KE. Thus, we propose a new forgetting procedure called **later-layer forgetting (LLF)**. Given a layer threshold $L$, we define the LLF mask criterion for each layer $l$ as: $M^l_{LLF} = 1$ if $l<L$, $0$ if $l \ge L$"

循环性（§2 引 KE 的 generation 机制，LLF 沿用）：

> "At the start of each generation, the weights in the fit-hypothesis are kept the same, and the weights in the reset-hypothesis are **reinitialized**. Then, the network is trained from this new initialization for $e$ epochs, where $e$ is the same for each generation."

轮数：Table 1 / A7 的 **N3 / N8 / N10** = 3 / 8 / 10 个额外 generation。→ **循环轮数比 RePr（N=3）和 DSD（2 轮）都多。**

具体 reset 的位置（不是只有输出头）：Table 1 caption —— "LLF uses $L \in \{10,14\}$, corresponding to **block 3 and 4 in ResNet18**"；Table A7 —— ResNet50/Tiny-ImageNet "reset layers starting from the **third block** ($L=23$)"；Table A10 DenseNet169 "$L \in \{40,68\}$"。→ **是整块 layer 级重初始化，不是 last-layer trick。**

**这一篇让 AUDIT2 的两句话同时失效**：
- "layer 级从未有人循环剪-补" → 假。
- "'挖掉 K 层再补 K 个新层、总深度不变'这个具体操作，plasticity 文献里没有" → LLF 就是（只不过它挖的是"top-k 层"而不是"中间 K 层"，且它是 reinit-in-place 而不是 remove-then-append；见 §1.4 的差异分析）。

**这一篇还是"forget-and-relearn"统一框架论文**，它 §2 明确把 IMP / DSD-family / KE / RIFLE / 迭代自蒸馏全部收编为同一个 paradigm 的实例：

> "We propose that many existing iterative algorithms are instances of a more general **forget-and-relearn** process, and that their success can be understood by studying the shared underlying mechanism."

⚠️ **对我们的直接后果**：reviewer 手上有一篇 ICLR 2022 论文，它的 thesis 就是"这一类算法（含权重级、层级、蒸馏级）背后是同一个机制"。我们如果主张"层级 vs 权重级机制不同"，**举证责任在我们，而且要正面反驳一篇 ICLR 论文的统一性论断**。AUDIT2 §5-Q2 第 2 条把"粒度换了会让机制换掉"列为 ★★ 增量 —— 在这篇存在的前提下，那不是增量，那是**需要被反驳的既有结论**。

### 1.2 SEAL — 2304.04858，**CVPR 2023（S2 http200 已核实：`venue='Computer Vision and Pattern Recognition'`, type=conference）**

摘要原文（我从 arXiv abs 页 `citation_abstract` 读到）：

> "**LLF (later-layer-forgetting) is a state-of-the-art method in this category. It strengthens learning in early layers by periodically re-initializing the last few layers of the network.** Our principal innovation in this work is to use Simulated annealing in EArly Layers (SEAL) of the network in place of re-initialization of later layers."

→ 到 2023 年，**"周期性重初始化整层"已经是这个子领域的 SOTA baseline，有专门论文来打它**。这不是"真空"，这是**一个有 SOTA、有后继工作、有专门 ablation 的成熟子领域**。

**SEAL 还给了我们方向一个负面结论**（同摘要）：

> "We further show that, compared to normal training, **LLF features, although improving on the target task, degrade the transfer learning performance across all datasets we explored.**"

⚠️ **这一句对新 Paper C 是要命的**。我们的方向是"用 cycling 得到一个更好的**预训练**模型"，而预训练模型的价值**几乎全部在于下游可迁移性**。CVPR 2023 已经报告：层级周期性重初始化产出的特征，**target task 变好但迁移变差**。这正好落在我们要卖的那个点上，且方向相反。

### 1.3 `lw`（layerwise reinitialization）— 2109.00267：**AUDIT2 明确读错了**

AUDIT2 §4 表格把 2109.00267 归类为：

> | `"reinitialize layers"` | 1 | 2109.00267（CNN reinit 对泛化影响，**非循环、非层级 regrow**） |

**实际它是一个层级循环重初始化算法，并且是那篇论文的主要贡献。** 摘要原文：

> "We also introduce a new **layerwise reinitialization algorithm** that outperforms previous methods... Our takeaway message is that the accuracy of convolutional neural networks can be improved for small datasets using **bottom-up layerwise reinitialization**, where the number of reinitialized layers may vary depending on the available compute budget."

Algorithm 1（`lw` 伪代码，我从 ar5iv 全文读到）：

> "... 12: **Reinitialize all layers above block $k$**; 13: Fine-tune the entire model until convergence; 14: end for 15: end for"

双层 `for` = $K$ 个 block 轮次 × $N$ 轮 → **就是循环**。而且 §3 明确说 "every reinitialization method uses the same number of rounds $K$"。

**注意 venue**：S2 `venue='arXiv.org'`，arXiv 无 JREF → 这一篇**必须标 arXiv preprint**（社区常引作 AAAI 2022，我未核实到，不写）。但它在 2202.00155 里被作为 concurrent work "LW" 正面对比过（Table 1），所以其存在性与内容有 peer-reviewed 论文背书。

### 1.4 那我们还剩什么差异？（诚实清点 —— 这就是"窄缝"）

| 维度 | LLF (ICLR22) / SEAL (CVPR23) / lw (preprint) | 我们 |
|---|---|---|
| 粒度 | **整层（block 级）** | 整层 —— **相同，不是差异** |
| 循环 | **是，N=3/8/10 generations** | 是 —— **相同，不是差异** |
| 训练期 curriculum | **是** | 是 —— **相同，不是差异** |
| 终点尺寸 | **原尺寸** | 原尺寸 —— **相同，不是差异** |
| reset 的位置 | **top-k 层（$l \ge L$）或 block k 以上全部** | 「中间/前段挖 K 层 + 尾部 append K 个新层」 | 
| 操作语义 | **reinit-in-place**（层还在原位置，权重换新） | **remove-then-append**（层被删除，新层加在别处）→ 若 keep_front + append，则等价于"reinit 后 L-K 层" ⚠️ **见下面的致命等价性** |
| 架构 | CNN（ResNet/DenseNet/WRN/VGG/MobileNet） | **decoder-only transformer** |
| 任务/数据 | 图像分类，**小数据 + 多 epoch**（Flower 1020 / CUB 5994 / Aircraft 3334 / MIT67 5360 / Dogs 12000 训练样本） | **LM 预训练，超大数据 + ~单遍** |
| 规模 | ResNet50 量级 | 1B–7B |

⚠️⚠️ **最危险的一条（AUDIT2 完全没意识到）**：我们的实现路径是 `--keep_front_layers K_f` + `--n_fresh_layers K`（保前段 + 补新层，令 `n_fresh = 原深度 − keep_front` 保持总深度）。**这在数学上就是"保留前 $K_f$ 层、把第 $K_f$ 层以上全部换成随机新层"= 逐字的 LLF mask $M^l = 1$ if $l < L$, $0$ if $l \ge L$（取 $L = K_f$）。** 差异只剩"新层是原层重初始化还是新建对象"——**这在参数空间里是同一个操作**（同架构、同 shape、同随机初始化分布）。

**结论**：如果我们用现成 flag 直接做循环，**产出的算法就是 LLF，逐字**。reviewer 一句 "this is LLF (Zhou et al., ICLR 2022) applied to LM pretraining" 就够了。要有差异，**必须真的做"从中间挖掉 K 层"（非 top-k、非连续到顶），而现成 flag 不支持**（`keep_front + n_fresh` 只能做 top-k）。这是一个**需要新写代码**的差异，不是"胶水代码"。AUDIT2 §7.3 说"缺的只是循环调度 + optimizer state 局部重置这两块胶水代码"—— **不对，若只加这两块胶水，得到的正是 LLF。**

---

## 2. K2 — AUDIT2 的核心定量论据读反了（RePr Table 3）

AUDIT2 §2 与 §7.1 两处把这条当作我们方向的支柱：

> §2：「**它已经跟 DSD 正面比过**：RePr Table 3（ResNet-20/CIFAR-10 test error）：Our Impl 8.4 / **DSD 7.8** / BAN 8.2 / RePr Weights 7.7 / **RePr Ortho 6.9**。即"结构级循环 > 权重级循环"这个结论 **RePr 已经在 CNN 上给出过了**。」
> §7.1：「**RePr Table 3 还证明结构级(6.9) > 权重级 DSD(7.8)**」

**数字抄对了，解读反了。** RePr 原文紧接 Table 3 的讨论（1811.07275，我从 ar5iv 全文读到）：

> "We compare our training scheme with other similar schemes like BAN and DSD in table 3. All three schemes were trained for three iterations i.e. N=3. All models were trained for 150 epochs with similar learning rate schedule and initialization. **DSD and RePr (Weights) perform roughly the same function - sparsifying the model guided by magnitude, with the difference that DSD acts on individual weights, while RePr (Weights) acts on entire filters. Thus, we observe similar performance between these techniques.** RePr (Ortho) outperforms the other techniques..."

**正确读法**：
- 控制住 metric（都用 magnitude）时，**权重级 7.8 vs 结构级 7.7 = 作者原话 "roughly the same function… similar performance"**。→ **粒度本身几乎不带来增益。**
- 6.9 那一档来自 **换 metric（inter-filter orthogonality）**，不是来自换粒度。

**后果（两层）**：
1. AUDIT2 §7.1 建议写进 intro 的第 1 段（"RePr Table 3 还证明结构级 > 权重级"）是**对被引论文的误述**。若照抄进 .tex，是 misrepresentation，reviewer 查原文即破。
2. AUDIT2 §5-Q2 第 2 条 ★★「粒度换了会让机制换掉，不是 straightforward extension（可实证）」的**唯一定量支撑就是这个误读**。去掉它之后，"粒度重要"这个假设在文献里的证据是 **negative**（RePr 自己说粒度不重要，metric 才重要）。
3. 引申：RePr 的真正 lesson 是 **"metric > 粒度"**。而**层级的 metric 空间已经被 Paper B / ShortGPT 一线吃掉了**（我们自己已有 keep-front / ShortGPT baseline / CKA / knowledge-onset）。所以"给层级设计一个 metric"这个 RePr-style 贡献点，对我们**恰好是已发表的 Paper B 领地**，不能再当新 Paper C 的卖点。

**附**：RePr 自己还给了一条对我们不利的 scaling 观察（§3）：

> "While our technique performs best with vanilla ConvNet architectures **it still marginally improves the performance of modern architectures.**"

即：越是现代化、越有 residual/规范化的架构，收益越小。decoder-only transformer 是 residual + 多重 norm 的极致。

---

## 3. K3 — 这支文献的收益被原作者限定在"小数据/过拟合/多 epoch"，而 LLM 预训练在其失效区

这是我认为**最可能让新 Paper C 直接死掉的一条**，而 AUDIT2 完全没有提。

### 3.1 三个 independent 的原文限定

**(a) 2109.00267 §5 Discussion（`lw` 作者自己的 takeaway）**：

> "Empirical results show that this method improves generalization across a wide range of architectures and hyper-parameters, **particularly for small datasets**. … Our takeaway message is that the accuracy of convolutional neural networks can be improved **for small datasets** using bottom-up layerwise reinitialization… **For large datasets, however, reinitialization does not seem to offer a benefit.**"

同篇 §3.1 的 decision tree（Figure 3）把 **"Training Set Size < 35K?"** 作为**根节点附近的分裂特征**——即"数据量"是决定 reinit 是否有用的一级变量。35K 样本。LLM 预训练是 $10^{12}$ tokens 量级。

**(b) 2202.00155 附录 A2.4 / Table A8（LLF 作者自己的负结果）**：

> "For WideResNet-28-10, we reset from the second block ($L=10$), and for DenseNet-BC, we reset only the fully-connected output layer ($L=99$) during LLF. We report our results in Table A8. **In these settings, we do not see any improvements from LLF over our baselines.**"

Table A8 数字（我从全文读到）：WRN CIFAR-10 `Smth long (N3) 96.32` vs `Smth + LLF (N3) 95.91`（**LLF 更差**）；CIFAR-100 `81.29` vs `80.95`（**LLF 更差**）。
→ 一旦换到**样本更多（CIFAR-10 每类 5000）+ 更强 baseline（WRN-28-10 / DenseNet-BC）**，**层级循环重初始化不再有收益，甚至变差**。

而 LLF 主表有效的那五个数据集是（Table A6）：Flower 训练 1020、CUB 5994、Aircraft 3334、MIT67 5360、Dogs 12000 —— **全是千级样本、明确"prone to overfitting"的 regime**：

> "Following Taha et al. (2021), we study tasks that have **a small number of training examples per class, where the network is prone to overfitting**."（§4.1）

**(c) 2304.04858 SEAL 摘要（迁移侧负结果）**：

> "**LLF features, although improving on the target task, degrade the transfer learning performance across all datasets we explored.**"

### 3.2 为什么这对 LLM 预训练是致命的

这支文献的机制解释，三篇口径一致，全部指向"抑制记忆化 / 抗过拟合"：
- 2109.00267：「it **encourages learning general rules and discourages memorization** by placing emphasis on the lower layers」；机制 = margin ↑ + flatter minima。
- 2202.00155：forget-and-relearn 的选择性遗忘目标是「information associated with **difficult examples**」，即长尾/记忆化样本。
- DSD（1607.04381 §5，AUDIT2 已引）：「Regularized and Sparse Training」+ escape saddle point。

**LLM 预训练的制度恰好没有这个病**：单遍（或近单遍）扫过 trillion-token 语料，训练 loss 远未到 100%，**不过拟合**。"抑制记忆化"在这里**不是收益而是纯损失**——因为我们同时还要模型记住事实知识（这正是 Paper B 的 MMLU/closed-book QA 轴）。

**并且我们自己的 Paper B 已经把这个损失量化了**：keep14 剪层后 continue-pretrain 到 200k step，**PPL 恢复到 baseline 的 1.428 倍，MMLU 只恢复 19.5%；keep8 到 200k MMLU 始终没超 chance**。

⚠️ **这是 AUDIT2 论证结构里的一个自相矛盾**，它没看出来：
AUDIT2 §5-Q2 第 1 条把 Paper B 的不对称（PPL 能 heal、知识不能）当作新 Paper C 的 **★★★ 最强卖点**。但同一个事实，**读作先验时是最强的失败预测**：
> 我们已经自己证明了：**层级结构损伤对知识的破坏是近乎不可逆的**（19.5%，keep8 never above chance）。
> cycling 就是**反复施加这种损伤**。每一轮都要付一次不可逆的知识税。
> ⇒ **先验预测：cyclic 层级 prune-regrow 在 LLM 预训练上的知识侧结果，随轮数 N 单调变差。**

AUDIT2 §5-Q2 用"在知识尚未沉淀的预训练早期做，代价应远小"来救。但这个救援有两个问题：
1. **它把主张变成了一个 timing 实验，而不是一个方法**。如果"早期做代价小"，最省的做法是 N=0（根本不做）—— 除非能证明 cycling 有一个**正向收益**来抵扣。而正向收益的唯一文献来源（§3.1 三篇）在大数据制度下被原作者报告为 **无 / 负**。
2. **"早期"这个窗口对 7B 预训练不可及**：我们盘上没有 OLMo-2-7B 的从零预训练能力（Paper B 做的是 continue-pretrain 已训练模型）。要在"知识未沉淀"的窗口做，只能退到 1B 从零，那就回到了 2606.24752 的 ≤314M / 1B 尺度，**规模优势也没了**。

---

## 4. K4 — AUDIT2 的"最强空白证据"是检索措辞造成的假空白（DSD 的 LLM 版存在）

AUDIT2 §3 第 4 问与 §4 的原话：

> §3.4：「**有没有在 LLM / transformer 上被复现过？** **搜遍 arXiv 没有。** `all:"dense-sparse-dense"` 全库**只有 6 条命中**… **没有一条是 transformer / LLM。** **这是本次审计最强的空白证据（见 §4）。**」

**我用不同措辞一次命中**：

`all:"sparse pre-training and dense fine-tuning"` → TOTAL=**1** → **2303.10464 SPDF: Sparse Pre-training and Dense Fine-tuning for Large Language Models**

**venue 核实（S2 http200）**：`venue = 'Conference on Uncertainty in Artificial Intelligence'`, `type=conference`, year=2023；arXiv COMMENT 亦自述 "Accepted to Uncertainty in Artificial Intelligence (UAI) 2023 Conference"。→ **peer-reviewed UAI 2023，不是 preprint。**

摘要原文（arXiv abs 页 `citation_abstract`）：

> "we propose to decouple the model capacity between the two phases and introduce **Sparse Pre-training and Dense Fine-tuning (SPDF)**. In this work, we show the benefits of using unstructured weight sparsity to train only a subset of weights during pre-training (Sparse Pre-training) and then **recover the representational capacity by allowing the zeroed weights to learn (Dense Fine-tuning)**. We demonstrate that we can induce up to 75% sparsity into a **1.3B parameter GPT-3 XL** model resulting in a 2.5x reduction in pre-training FLOPs, without a significant loss in accuracy on the downstream tasks relative to the dense baseline."

**这一篇同时打掉 AUDIT2 的两个论断**：
- "DSD 从没在 transformer/LLM 上被复现" → 假（SPDF 就是 sparse→dense 的 LLM 版，1.3B GPT-3 XL，且明确 cite DSD 谱系）。
- "所有循环 reset/regrow 工作的规模上限是 67M / 125M / 314M，**没有任何一条到 1B+**"（AUDIT2 §5-Q1 规模边界）→ 假（1.3B）。

**诚实限定（我必须说清楚，否则和 AUDIT2 犯同类错）**：SPDF 是 **sparse→dense 一次性（不是循环）**，且动机是**训练 FLOPs 效率**（不是最终质量/塑性），粒度是**非结构化权重**（不是层）。所以它**不是我们构造的直接占位**。但它足以摧毁 AUDIT2 用来支撑增量的"规模真空"论证，并且它证明**这条线的作者们早就想到要上 LLM，只是选择了权重级 + 单调方向**。reviewer 会问："既然 SPDF 在 1.3B 上用权重级做到了，你的层级循环相对它的收益是什么？" —— 这个问题 AUDIT2 没有准备答案。

---

## 5. K5 — "plasticity 文献在 ≥1B LM 上零干预、零下游能力报告" 是错的

AUDIT2 §5-Q3 洞 2 原话：

> 「这一支的 LLM 侧只有 2606.24752 一篇（**preprint、≤314M、纯诊断、无干预**）和 2307.01163（NeurIPS 2023，**只 reset embedding 层，125M RoBERTa**）。**没有任何一篇在 ≥1B LM 上做过任何 plasticity 干预并报告下游能力。**」

**反驳材料就在 AUDIT2 自己声称精读过的 2606.24752 的正文和参考文献里**（我从 arxiv.org/html/2606.24752 全文读到）：

**(a) 2606.24752 §II Background 原文**：

> "**Cho et al. (2026) presented an algorithm for maintaining plasticity and preventing forgetting when learning from text data using models with more than a billion parameters.** However, their evaluations of plasticity loss included only eight different tasks…"

该文献项 = **2502.07274 Forget Forgetting: Continual Learning in a World of Abundant Memory**，2606.24752 参考文献原文标注为 "In **The Fourteenth International Conference on Learning Representations**"（= ICLR 2026）。
⚠️ **venue 我自己的核实**：S2 对 2502.07274 返回 `venue=''`, `publicationVenue=None`（http200，不是 429）→ **S2 未收录，我只能标 arXiv preprint（ICLR 2026 系他人论文引用时的标注，非我核实到的 S2/DBLP 记录）**。

**(b) Springer et al. — 2503.19206 "Overtrained Language Models Are Harder to Fine-Tune"**
**venue 核实（S2 http200）**：`venue='International Conference on Machine Learning'`, `type=conference`, year=**2025** → **ICML 2025 peer-reviewed**。

摘要原文：

> "we challenge this assumption and show that extended pre-training can make models harder to fine-tune, leading to degraded final performance. We term this phenomenon **catastrophic overtraining**. For example, the instruction-tuned **OLMo-1B model pre-trained on 3T tokens** leads to over 2% worse performance on multiple standard LLM benchmarks than its 2.3T token counterpart."

§2.1 实验设置原文：

> "we experiment on three language models with open-sourced intermediate checkpoints: **OLMo-1B** (Groeneveld et al., 2024a), **OLMo-2-7B** (OLMo et al., 2024), and **LLM360-Amber-7B**"

机制原文（§1）：

> "another key factor influencing forgetting is what we term **progressive sensitivity**: for modifications of equal magnitude, models that have undergone longer pre-training exhibit greater forgetting"

**后果（三层，全部对我们不利）**：
1. AUDIT2 §5-Q3 的"洞 2（规模 + 领域）"**不存在**：ICML 2025 已经在 **OLMo-1B/3T + OLMo-2-7B**（**和我们同一个模型家族、同一个尺度**）上做了塑性退化的现象 + 机制 + 理论 + 下游 benchmark。
2. AUDIT2 §5-Q2 第 3 条 ★★「规模的定性转变有独立价值 / 该领域自己在喊缺 LLM 尺度的干预方法」的支撑（2606.24752 是 preprint、≤314M）**被同一篇 preprint 自己的 background 段推翻**：它 cite 的 Springer(ICML 2025) 和 Cho(ICLR 2026) 都在 ≥1B。AUDIT2 声称精读了 2606.24752 全文（§8 清单里有 2606.24752），却漏掉了其 §II 里两条直接反驳自己结论的句子。**这是 AUDIT2 的阅读失误，不是检索失误。**
3. **Springer 的机制与我们方向直接冲突**：progressive sensitivity = 训得越久，**对参数修改越敏感、遗忘越多**。cycling 是**故意反复施加大幅参数修改**。按 Springer 的机制，cycling 在预训练后期的每一轮都会被放大惩罚。→ 又一条 negative 先验，且出自 peer-reviewed ICML 2025。

---

## 6. 逐条回答用户的 5 个问题

### Q1：AUDIT2 列的"必须打败的工作"里，有没有读漏/读错的？
**有，三处，两处致命：**
1. **读错（致命）**：2109.00267 被标为"非循环、非层级 regrow" → 实际是**层级循环重初始化算法**（Algorithm 1: "Reinitialize all layers above block k"，双层 for = K×N 轮），且是该论文主贡献。见 §1.3。
2. **读反（致命）**：RePr Table 3 被读成"结构级 6.9 > 权重级 DSD 7.8 ⇒ 粒度重要"，原文明说 **"DSD and RePr (Weights) perform roughly the same function… we observe similar performance"（7.8 vs 7.7），增益来自 metric 不是粒度**。见 §2。
3. **漏读（致命）**：AUDIT2 §8 声称精读了 2606.24752 全文，但漏掉其 §II 中 "Cho et al. (2026)… **models with more than a billion parameters**" 与 Springer et al. (2025) 两条，导致 §5-Q3"洞 2"结论错误。见 §5。
4. 另有一处**过度归因**（非致命但需修）：AUDIT2 §2 表格写「RePr 明确说 "Other initialization methods are also worth trying"（那句其实在 DSD §5，RePr 用 orthogonal）」—— 它自己已经标注了，OK；但同表把 RePr 的补回初始化写成 "QR 分解求 null-space"，我在 RePr 全文里核到的是 inter-filter **orthogonality ranking metric** 与 orthogonal re-init 的表述（§1/§3 "we present a metric based on the inter-filter orthogonality within convolutional layers"），**QR/null-space 的具体表述我未在抓取到的全文中定位到** → 若要写进 .tex 需 MAIN 二次核实。

### Q2：有没有它没搜到的工作？（列出我的 query）
**有，5 篇关键的。** 我跑的 query 清单（全部走 **`https://export.arxiv.org`**）：

**⚠️ 先说一个方法论问题（可能是 AUDIT2 大量 0-hit 的根因）**：AUDIT2 §6 自述用的是 `http://export.arxiv.org/api/query?...`，并写「`https://` 变体返回空」。**我实测恰好相反**：
```
http://export.arxiv.org/...  → HTTP=301  SIZE=0     （空 body，任何 query 都"0 命中"）
https://export.arxiv.org/... → HTTP=200  SIZE=5727  （正常 Atom feed）
```
（用已知真实短语 `"attention is all you need"` 对照测得。）项目 `_FETCH_RECIPE.md` 也记「`export.arxiv.org/api/query` 在本机经代理返回空 → 不可用」，与 `http://` 的行为一致。
**→ AUDIT2 的 40+ 个 "TOTAL=0" 可能全部是 301 空响应，而非真实零命中。** 我用 `https://` 复跑了它的一批 0-hit query，**大部分确实仍是 0**（下表），所以它的结论不是全错；但**它的方法无法区分"真 0"和"301 空"**，任何未复核的 0-hit 都不可作为「无人做过」的证据。

| 我的 query（https，含 TOTAL） | 结果 |
|---|---|
| `"prune and regrow layers"` 0 / `"regrow layers"` 0 / `"layer regrowth"` 0 / `"resetting layers"` 0 / `"re-initialize layers"` 0 / `"layer reinitialization"` 0 / `"cyclic drop and restore"` 0 | 与 AUDIT2 一致（真 0） |
| `"reinitialize layers"` **1** | 2109.00267 → **AUDIT2 读错的那篇** |
| **`"later layer forgetting"` 1** | **2304.04858 SEAL (CVPR 2023)** ← AUDIT2 未提 |
| **`"forget and relearn"` 3** | **2202.00155 LLF (ICLR 2022)**、2310.07996 Reset-It-and-Forget-It (**S2: ECAI 2023**) ← 均未提 |
| **`"sparse pre-training and dense fine-tuning"` 1** | **2303.10464 SPDF (UAI 2023, 1.3B)** ← AUDIT2 §3.4 声称"搜遍 arXiv 没有" |
| **`"Overtrained Language Models Are Harder to Fine-Tune"` 1 / `"catastrophic overtraining"` 3** | **2503.19206 (ICML 2025, OLMo-1B/3T + OLMo-2-7B)** ← 未提 |
| **`"knowledge evolution"` 57** | **2103.05152 KE (S2: CVPR 2021)** ← AUDIT2 未提（LLF/SEAL/lw 三篇共同的祖先与 baseline） |
| **`"re-initializing the fully-connected layer"` 1** | **2007.03349 RIFLE (S2: ICML 2020)** ← 未提（周期性 reinit 末层） |
| **`"Forget forgetting continual learning in a world of abundant memory"` 1** | **2502.07274 Cho et al.（>1B plasticity 干预）** ← 未提 |
| `"periodic reinitialization"` 4 / `"periodically reinitialize"` 4 | 2406.02596 Hare & Tortoise (ICML 2024，AUDIT2 提过) 等，无新占位 |
| `"prune during pretraining"` 1 | 2205.12452 Sparse*BERT（压缩向，不撞） |
| `"progressive layer dropping"` 2 | 2010.13369（加速，权重不重置）/ 2412.11242 TrimLLM（压缩）—— 不撞 |
| `"model surgery"` 4 / `"knowledge evolution"`(筛后) / `"born-again networks"` 4 / `"shrink and perturb"` 5 / `"plasticity injection"` 9 / `"structural plasticity"` 67 | 无新占位 |
| `"sparse pretraining dense finetuning"` 0 / `"sparse-dense training large language"` 0 / `"reset and retrain"` 0 / `"layer swapping training"` 0 / `"structured dropout regrow"` 0 / `"cyclic training schedule pruning"` 0 / `"layer replacement training"` 0 / `"layer recycling"` 2(无关) / `"block recycling"` 3(无关) / `"depth annealing"` 2(无关) / `"expand and contract network"` 0 / `"iterative layer pruning and expansion"` 0 / `"prune and grow layers"` 0 / `"reinitialization language model pretraining"` 0 / `"resetting transformer blocks"` 0 / `"cyclic model growth"` 0 / `"prune-and-regrow"` 4(CNN/3D) / `"depth cycling"` 7(纯数学/CV 无关) / `"plasticity loss language model"` 0 / `"loss of plasticity in continual pre-training"` 0 / `"maintaining plasticity in large language models"` 0 / `"reset optimizer state pretraining"` 0 / `"weight re-initialization continued pretraining"` 0 / `"layer swap regularization transformer"` 0 / `"layer-wise reinitialization language model"` 0 / `"reinitialization scaling law"` 0 / `"reset layers large language model"` 0 / `"cyclic sparsification pretraining LLM"` 0 / `"sparse dense cycling pretraining"` 0 / `"prune layers and continue pretraining"` 0 / `"iterative depth reduction and restoration"` 0 / `"grafting layers"` 9(高分子物理) / `"grow and shrink"` 89(筛后无关) | 真 0 或无关 |

**免责（按硬规矩 3）**：我搜过上述 ~50 个 https-endpoint 短语 query，**未命中"层级循环 reinit 在 ≥1B decoder-only LM 预训练上"的完整组合**。但我**不排除**：(a) 我未做 S2 语义近邻检索（S2 search endpoint 全程 429，我只用它核 venue）；(b) 工业界 tech report 的 training-recipe 章节抓不到（此风险 AUDIT2 §7.4 已提，我认同且未能覆盖）；(c) 我未检索非 arXiv 场地（OpenReview rejected papers、期刊）。

### Q3：AUDIT2 声称的"我们的差异点"，是否其实是无关紧要的差异？
**是。逐条判决：**

| AUDIT2 的差异点 | 我的判决 |
|---|---|
| 「层级（不是权重级）」 | **无效差异**。LLF/SEAL/lw 已是层级；且 RePr 原文说粒度换了性能"roughly the same"。**reviewer 会说 this is LLF with a different granularity-of-nothing。** |
| 「循环（不是单调）」 | **无效差异**。LLF N=3/8/10、lw K×N、DSD 2 轮、RePr N=3 全是循环。 |
| 「终点回到原尺寸」 | **无效差异**。AUDIT2 §5-Q2 第 4 条自己已承认"不是我们独有，不要当亮点写"。LLF/SEAL/lw/KE/RePr/DSD 全部终点原尺寸。 |
| 「目的是最终模型质量/塑性（不是加速/压缩）」 | **无效差异**。LLF/lw/RePr/KE 的目的**就是**最终质量（泛化），不是压缩也不是加速。 |
| 「粒度换了机制换掉」（★★） | **证据反向**（§2）。且需正面反驳 ICLR 2022 的 forget-and-relearn 统一性论断。 |
| 「LLM 规模（≥1B）是真空」（★★） | **假**（§4 SPDF 1.3B UAI 2023；§5 Springer ICML 2025 OLMo-1B/3T + OLMo-2-7B）。 |
| 「与 Paper B 知识/PPL 不对称衔接」（★★★） | **这是唯一真差异**，但见 §3.2：它同时是**最强的失败预测**，且"在早期做代价小"这条救援会把主张退化为 timing 实验 + 丧失规模优势。 |

**最危险的 reviewer 一句话（我会这么写审稿意见）**：
> "The proposed method is Later-Layer Forgetting (Zhou et al., ICLR 2022) / layerwise reinitialization (Alabdulmohsin et al., 2021) transplanted from small-data image classification to LM pretraining. Both papers explicitly report that the benefit vanishes on larger datasets (Alabdulmohsin et al., §5: 'For large datasets, however, reinitialization does not seem to offer a benefit'; Zhou et al., Table A8: 'we do not see any improvements from LLF over our baselines'), and Ahn et al. (CVPR 2023) report that LLF features degrade transfer. The paper does not explain why the mechanism (suppressing memorization of difficult examples, effective under overfitting) should apply to single-pass trillion-token pretraining, where the model does not overfit. Reject."

### Q4：有没有把 preprint 当 peer-reviewed，或反之？
**AUDIT2 的 venue 工作总体是诚实且细致的**（它主动标了 DSD/Pangu/Can-Scale/Möbius 等为 preprint，并诚实报告了 2 个 429 未解）。我没发现它把 preprint 当 peer-reviewed。**但有两处"漏标身份"造成的实质影响**：
- **2109.00267 被当作可忽略的边角料**，实际是核心先例。**其身份必须标 arXiv preprint**（S2 http200 返回 `venue='arXiv.org'`，无 JREF）—— 但它被 ICLR 2022 论文（2202.00155 Table 1 / §A2.4）作为 concurrent work "LW" 正面对比，所以**内容有 peer-reviewed 背书**，不能因 preprint 身份而降低其占位效力。
- **AUDIT2 拿 2606.24752（preprint、S2 venue 为空）作为"该领域自己承认 LLM 侧没解法"的关键论据**（§5-Q2 第 3 条、§7.1 第 2 段）。身份标注正确，但**用一篇 preprint 的"我们没找到 smoking gun"去论证整个领域的空白，而该 preprint 自己的 §II 就 cite 了 ICML 2025 与 ICLR 2026 两篇 ≥1B 的工作** → 论证无效。

**我新增/核实的 venue（原始 JSON 落盘 `paperC_v2_research/_venue_raw_skeptic2/<id>.json`，全部 http200，非 429）**：

| arXiv | S2 venue | 身份 |
|---|---|---|
| 2202.00155 | International Conference on Learning Representations (conference) | **ICLR 2022，peer-reviewed** |
| 2304.04858 | Computer Vision and Pattern Recognition (conference) | **CVPR 2023，peer-reviewed** |
| 2103.05152 | Computer Vision and Pattern Recognition (conference) | **CVPR 2021，peer-reviewed**（COMMENT 自述 "CVPR Oral 2021"） |
| 2303.10464 | Conference on Uncertainty in Artificial Intelligence (conference) | **UAI 2023，peer-reviewed** |
| 2503.19206 | International Conference on Machine Learning (conference), 2025 | **ICML 2025，peer-reviewed** |
| 2007.03349 | International Conference on Machine Learning (conference) | **ICML 2020，peer-reviewed**（RIFLE） |
| 2310.07996 | European Conference on Artificial Intelligence (conference) | **ECAI 2023，peer-reviewed** |
| 2109.00267 | `'arXiv.org'` | ⚠️ **arXiv preprint**（社区常引 AAAI 2022，未核实，不得写） |
| 2606.24752 | `''`（空，http200） | ⚠️ **arXiv preprint** |
| 2502.07274 | `''`（空，http200） | ⚠️ **arXiv preprint**（他人引作 ICLR 2026，非我核实） |

### Q5：它的"窄缝"是否窄到不足以支撑一篇论文？
**是，按 AUDIT2 建议的形态不足以。**

AUDIT2 §7.1 建议的三段式卖点，第 1 段（RePr 结构级 > 权重级）是误读，第 2 段（两条边界从未越过）两条都假（LLF/SEAL/lw 越过粒度边界；SPDF/Springer 越过规模边界）。**三段式里两段半是错的。**

剩下的真窄缝，精确表述是：
> 「**从深度栈中间**移除 K 个 block 并在别处补 K 个新 block（区别于 LLF/lw 的"top-k / block-k 以上"），**循环多轮**，在 **≥1B decoder-only LM 的（继续）预训练**上，测量 **PPL 与知识/MMLU 的分离代价**。」

这条缝的问题不是"技术上有人做过"，而是**三个先验全部指向 null-or-negative**：
1. 该机制的原作者：大数据无收益（2109.00267 §5）、更强 baseline 无收益甚至更差（2202.00155 Table A8）、迁移变差（2304.04858 摘要）。
2. ICML 2025 progressive sensitivity：训得越久，参数修改的惩罚越大 → cycling 越后期越亏。
3. **我们自己的 Paper B**：单次层级损伤的知识税已经近乎不可逆（MMLU 19.5% / keep8 never above chance）。cycling = 反复交这个税。

**"一个 null result" 能不能成论文？** 可以，但要求变了：
- 必须是**足够有人相信的假设**的 null（要有人在 LLM 上主张过 cycling 有用 —— 目前**没有**，所以我们在打一个稻草人）；
- 必须**足够大规模、足够干净的控制**（LR-matched、compute-matched、N-sweep、常量地板）；
- 必须**有正面的机制解释**，而不只是"我们试了，不 work"。

**我的判断**：作为「新 Paper C 的主线」不够。作为**Paper B 的一节（"cycling 不能修复知识税：单次损伤的不可逆性在多轮下累积"）**是合适的，因为它把 Paper B 的 Finding 从"单次"推广到"任意轮数"，且我们已有全部设施。

---

## 7. 如果 MAIN 仍要做，收窄后的唯一可辩护表述 + 必加对照臂

### 7.1 唯一可辩护的定位（不要写成"我们提出一种新训练方法"）

> **"Cyclic layer-level reinitialization does not transfer to LLM pretraining, and we explain why."**
>
> 1. 层级周期性重初始化（LLF, ICLR 2022；layerwise reinit, Alabdulmohsin et al. 2021；SEAL, CVPR 2023）在小数据图像分类上是有效正则；其机制被原作者归因为**抑制困难样本的记忆化**，且原作者已报告**大数据下收益消失**、**迁移性能下降**。
> 2. LLM 预训练在制度上位于该机制的失效区（单遍、不过拟合、且**知识记忆本身是目标而非病灶**）。
> 3. 我们在 ≥1B decoder-only LM 上给出**首个**该家族方法的规模化检验，并用 **PPL vs 知识两轴分离**（CNN 分类上物理不可观测）定量给出失效的**代价结构**：每一轮 cycling 的 PPL 可 heal、知识税不可逆且累积。这与 Springer et al. (ICML 2025) 的 progressive sensitivity 一致，并把它从 fine-tuning 阶段推广到 pretraining 阶段的结构性扰动。

这个版本的**每一句都能被我们的数据支撑**，且**不需要 cycling 有效**才成立。

### 7.2 必加对照臂（在 AUDIT2 §7.2 之上新增，缺任一条会被本报告点出的某篇正面打）

| 臂 | 打谁 | 为什么必须 |
|---|---|---|
| **LLF 精确复现臂（top-k reinit in-place，不是 remove-then-append）** | 2202.00155 | 否则 reviewer 直接说我们就是 LLF。必须证明"中间挖 K"与"top-k reinit"**不是同一个东西**，且我们现成 `keep_front + n_fresh` 组合**等价于 LLF**（见 §1.4），不能当作我们的方法 |
| **`lw` 式 bottom-up 递减臂（每轮少 reinit 一层）** | 2109.00267 Alg.1 | LLF 论文自己做了这个对比（Table 1），我们不做就缺一格 |
| **N sweep 且必须包含 N=0（纯 continue-train，compute-matched）** | 全部 | LLF Table A7/A8 的关键 baseline 是 "Smth **long** (N3/N10)"，即同 epoch 数不 reset。**没有它，任何"cycling 有效"都可能只是多训了** |
| **知识轴 + 常量地板（MMLU chance、closed-book QA 恒定拒答基线）** | 项目铁律 | Paper B 已知 keep8 MMLU never above chance |
| **迁移/下游 SFT 轴（不只 PPL 与 zero-shot MC）** | 2304.04858 SEAL | SEAL 明确报告 LLF 特征"target 变好、迁移变差"。我们若只报 PPL+MC，会被这一篇打 |
| **数据/epoch 制度轴：单遍 vs 多遍（重复数据）** | 2109.00267 §3.1 decision tree（Training Set Size < 35K 为分裂特征）、§5 | **这是我认为最有科学价值的一个臂**：如果 cycling 只在"数据重复/过拟合"制度下有用，那我们就**定量给出了该家族的适用边界**，这本身是可发表的 negative 结论 |
| **权重级同 FLOPs 对照（SPDF 式 sparse→dense）** | 2303.10464 SPDF (UAI 2023, 1.3B) | AUDIT2 §7.2 已列（打 2508.00212/DSD），但**必须改成 cite SPDF 而不是只 cite DSD**，否则漏掉 LLM 规模的权重级先例 |
| **LR-matched（新层 LR == 旧层 LR）** | 项目铁律 + DSD §2（re-dense 用 1/10 LR） | AUDIT2 §7.2 已列，我完全同意，保留 |
| **optimizer state 局部重置** | RePr §4 | AUDIT2 §7.3 已列，保留。⚠️ 另注 2606.24752 §IV 说他们"optimizer was reset at the start of each task… implies that the plasticity degradations we observed were due to inherent plasticity loss in the weights rather than simply stale optimizer states" → **不重置 optimizer 会让结论无法归因到权重** |

### 7.3 实现层新增警告（AUDIT2 §7.3 漏的）

⚠️ **AUDIT2 §7.3 说"缺的只是循环调度 + optimizer state 局部重置这两块胶水代码"—— 这是错的判断。**
`--keep_front_layers K_f` + `--n_fresh_layers (L - K_f)` 保总深度 = **逐字的 LLF mask**（$M^l=1$ if $l<L$, 0 if $l\ge L$）。**只加胶水 = 复现 ICLR 2022。**
要做出 AUDIT2 声称的差异（"从深度栈中间挖掉一段"），**必须新写"任意 index 集合的 block 移除 + 在指定位置插入新 block"的能力**，现有 flag 不支持。这是一个真实的工程增量，MAIN 派 coder 前必须知道。

---

## 8. 我实际抓取并精读的全文清单（可复核）

**新抓取（本报告新增，全部实读并标了 section）**：
- `arxiv.org/html/` 或 `ar5iv`：**2202.00155**（LLF 全文 89k chars）、**2109.00267**（lw 全文 65k）、**2503.19206**（Springer 全文 179k）、**2310.07996**（Reset-It 全文 79k）、**2606.24752**（Can-Scale 全文 80k，重读其 §II）、**1811.07275**（RePr 全文 69k，重核 Table 3）
- 仅 abs 页 `citation_*` meta（**已显式标注**）：**2303.10464** SPDF（**全文未抓取，仅摘要 + COMMENT**）、**2304.04858** SEAL（**全文未抓取，仅摘要**）、**2103.05152** KE（**html/ar5iv 两通路均 <20KB，全文未抓取到，仅摘要 + COMMENT "CVPR Oral 2021"**）
- 未抓取全文、仅经他人引用得知：**2502.07274** Cho et al.（**我只读到 2606.24752 对它的引用与 bib 条目，未读原文**）、2007.03349 RIFLE（仅标题/S2 venue）

**venue 原始 JSON**：`/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/paperC_v2_research/_venue_raw_skeptic2/`（2202.00155 / 2109.00267 / 2103.05152 / 2303.10464 / 2007.03349 / 2310.07996 / 2304.04858 / 2503.19206 / 2606.24752 / 2502.07274，全部 http200）

---

## 9. 一句话给 MAIN

**AUDIT2 说"卖点要从'新方法'改成'scale/granularity transfer'"—— 我的结论是要再退一步：granularity 已被 LLF/SEAL/lw 占了（ICLR22/CVPR23），scale 已被 SPDF/Springer 占了（UAI23/ICML25），而这个机制的原作者们已经写明它在大数据下无收益、在迁移上有害。所以可发表的东西不是"cycling 有效"，而是"cycling 在 LLM 预训练上为什么必然无效，以及它的代价结构是知识不可逆而 PPL 可逆"——后者是 Paper B 的自然延伸，建议并入 Paper B 而不是开新 Paper C。若 MAIN 坚持开新 C，第一个实验必须是 §7.2 的 "N=0 compute-matched" 与 "LLF 精确复现" 两臂；如果 N=3 打不过 N=0，方向当场结束，不要再投入 40 卡。**
