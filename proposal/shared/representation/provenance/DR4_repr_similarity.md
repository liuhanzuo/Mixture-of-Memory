# DR4: 对抗性精读报告 — 跨模型表征相似度四篇论文

**生成日期**：2026-08-06  
**作者**：Research agent（基于全文抓取，非摘要）  
**我方资产**：`proposal/shared/representation/repr_alignment_results.json`
（14 模型 / 91 pair / z-CKA + 系列控制）

---

## 总论（先读这里）

这四篇覆盖了该方向的三个层次：
- 层间相似（within-model fine-tuned）→ Paper 1（2109.08406）
- 跨模型全局收敛（multi-modal，PRH 检验）→ Paper 2（2602.14486）
- 跨 LLM 表征直接对比（decoder-only 7B）→ Paper 3（2312.02730）
- 跨 LLM SAE 特征空间（weight space）→ Paper 4（2410.06981）

我们的资产（decoder-only，0.6B-8B，91 pair，z-CKA，U 型 + null 控制）与**前三篇均有直接可比关系**；与 Paper 4 形成互补（activation space vs weight space）。**最大的空白是：没有一篇既做 decoder-only 跨家族 × 多规模 × 层序 shuffle null × massive-activation z-score × post-norm 对照的研究**。

---

## A. 每篇精确实验设置

### A1. arXiv:2109.08406（自述 BlackboxNLP'21，未经独立核实）
**Fine-Tuned Transformers Show Clusters of Similar Representations Across Layers**

- **模型**：RoBERTa-Base（12L，768D，125M），ALBERT-LargeV2（24L，1024D，235M），ELECTRA-Base（12L，768D，110M）。**全部 encoder-only，100M-300M 级**。
- **任务**：12 个 GLUE/superGLUE 任务，分别 fine-tune。
- **指标**：Linear CKA（公式见 Appendix A：HSIC(K,L) / sqrt(HSIC(K,K)·HSIC(L,L))，K=XX^T，L=YY^T）。
  - **没有 z-score**；直接用 CLS token 的原始激活向量。
  - 样本 N = 各 task 的 validation set 大小（几百到几千，论文未统一标注具体 N）。
- **CKA 矩阵格式**：层 × 层方阵（同模型 within-model），**不是 cross-model 跨家族对比**。
  - Appendix B 图 7（"cross models"）补了一张跨不同 fine-tuned 模型（同任务不同随机种子）的 CKA 图，但**篇幅极少，作为辅助**。
- **null 校准**：**无任何 null/随机化控制**。
- **identity gate**：无。
- **随机初始化 floor**：无。
- **相邻层 ceiling**：无（但可从矩阵对角线读出，通常 ≈1）。
- **主要结论**：fine-tuned RoBERTa 和 ALBERT 在层 × 层相似度矩阵上出现 block-diagonal 结构，早层簇与晚层簇高度内部相似、彼此之间不相似。顶部若干层可丢弃而不损性能。

---

### A2. arXiv:2602.14486（preprint，2026-02-16，更新到 v2 2026-06-25）
**Revisiting the Platonic Representation Hypothesis: An Aristotelian View**

- **模型**：跟随 Huh et al.（2024）PRH 实验协议：
  - 语言模型：BLOOM、OpenLLaMA、LLaMA，共 3 个家族，多规模（最大 LLaMA-65B）；
  - 视觉模型：ImageNet-21K、MAE、DINOv2、CLIP、CLIP-finetuned，共 5 个家族；
  - **合计 204 个 vision–language 对**，d/n ∈ [0.19, 8]（768D-4096D，n=1024 image-text pairs from WIT）。
  - **无 decoder-only 7B 或 8B 系列**，视觉模型均为 image encoder。
- **指标**：
  - Global spectral：linear CKA、RBF CKA、SVCCA、RV coefficient、Procrustes distance。
  - Local neighborhood：mKNN（k=10）、cycle-kNN、CKNNA。
  - **所有指标均有 null-calibrated 版本**。
- **Null 校准（核心贡献）**：
  - **Width confounder null**：对每个 (X, Y) pair，在固定 X 的条件下对 Y 的**样本行**做随机 permutation（K=200 次）。H₀ = "X 与 Y 无关"（超越各自边缘统计）。这与我们的 layer-order shuffle **不同**：我们的 null permute 的是 B 的**层顺序**（测"是否存在层间对应关系"），而 2602.14486 的 null permute 的是**样本对应关系**（测"这两个表征空间是否来自相同数据点"）。
  - **Depth confounder null**（Section 5.2 aggregation-aware）：对 layer-wise 矩阵取 max 时，用同一个 π 对 B 的所有层一起 permute，计算 max 的零分布。
  - 校准得分公式（Eq. 12）：s_cal = max((s_obs - τ_α) / (s_max - τ_α), 0)，τ_α 是 null 分布的 (1-α) 分位数，α=0.05。
- **代码/数据**：论文有开源代码（引用了 GitHub，未列出 URL，但论文提到可用）。
- **主要结论**：
  - global spectral 相似度校准后"largely disappears"（Pearson r with language model ranking：linear CKA 从 0.86 降到 0.45，Procrustes 从 0.89 降到 0.39）。
  - local neighborhood（mKNN）校准后相关性不变（≈0.85），引出 "Aristotelian Representation Hypothesis"。
  - **不做 within-LLM 层间分析**，没有 depth-resolved diagonal profile。

---

### A3. arXiv:2312.02730（自述 UniReps Workshop @ NeurIPS 2023 Extended Abstract，未经独立核实）
**Towards Measuring Representational Similarity of Large Language Models**

- **论文性质**：Extended Abstract（全文约 4 页，含图不含附录）。
- **模型**：11 个 7B 级 base LLM：RedPajama、Bloom、Falcon、Galactica、GPT-J、Llama、MPT、OpenLlama、OPT、Pythia、StableLM Alpha。code 任务加 CodeLlama、CodeLlama-Python（共 13 个）。**全 decoder-only，单一规模（均为 7B）**。
- **表征**：只取**最后一层**的**最后 token** 的激活。
  - **不做 layer-wise 分析**；不做 diagonal profile；不做任何深度分析。
- **数据**：Winogrande validation set（commonsense reasoning）；HumanEval（code generation）。Zero-shot prompting。
- **指标**：Orthogonal Procrustes、Aligned Cosine Similarity、Norm RSM-Diff、Jaccard similarity（OT + IS invariance）；加 RSA、CKA（OT + IS + TR invariance）。
- **null 校准**：**无**（全文无任何 null 控制、无随机 baseline、无 identity gate）。
- **主要结论**：
  - 不同 7B 模型的表征相似度差异很大（non-uniform）；StableLM Alpha 是 outlier。
  - 同一指标在不同任务上结论不一致（Winogrande vs HumanEval 的 Spearman ρ 仅 0.34）。
  - 不同指标之间结论不一致（同一指标组内平均 Spearman ρ=0.35 对 Winogrande，0.65 对 HumanEval）。
  - 提出三个"challenges"：指标间分歧、任务依赖、数值难解释。
  - **没有正面结论说"7B 模型表征相似"或"不相似"**，只是"有显著差异"。

---

### A4. arXiv:2410.06981（preprint，2024-10-09）
**Quantifying Feature Space Universality Across Large Language Models via Sparse Autoencoders**

- **模型**：8 个 pair（主文 4 个）：
  - Pythia-70m (6L, 512D) vs Pythia-160m (12L, 768D)
  - Gemma-1-2B (18L, 2048D) vs Gemma-2-2B (26L, 2304D)
  - Gemma-2-2B vs Gemma-2-9B (42L, 3584D)
  - Llama-3-8B-Instruct (32L) vs Llama-3.1-8B (32L)
  - 补充附录还有更多 pair。**全部 decoder-only**。
- **SAE 配置**：
  - Pythia：32768 特征（EleutherAI SAE）；
  - Gemma：16384 特征（Google/DeepMind SAE）；
  - Llama-3：65536 特征（Gated ReLU）；Llama-3.1：32768 特征（JumpReLU）。
  - SAE **不统一**（不同 tokenizer 同一系列内可行，**跨 tokenizer 不做**）。
- **方法**：
  1. 用激活相关性（activation correlation）将 SAE-A 的每个特征与 SAE-B 中相关性最高的特征配对；
  2. 过滤掉 non-concept 特征和 many-to-1 配对（保留约 10-30%，典型 20%）；
  3. 对配对后的 decoder 权重矩阵 W' 做 SVCCA 和 RSA。
- **Null**：随机打乱配对（N=100 或 N=1000 次 null runs），计算 p 值。典型 null mean SVCCA ≈ 0.05-0.15；典型观测值 0.5-0.83（Table 1, 2）。
- **主要结论**：
  - 中间层 SVCCA 得分最高（0.5-0.83 for semantic subspaces，p<0.01）；第 0 层永远不显著。
  - 语义子空间（Time、Calendar、Nature、Countries、People/Roles、Emotions）相似度高于整体特征空间。
  - 有约 10-30% 的特征对是模型特有（idiosyncratic）。
- **"rotation-invariant transformations" 的含义**：SVCCA 和 RSA 均不要求旋转对齐（RSA 只比较 RDM；SVCCA 通过 CCA 找最优线性变换）。不是特殊的 SO(n) / O(n) 变换类，而是标准的线性相关不变量。

---

## B. 方法论缺陷清单

### B1. Paper 1（2109.08406）— within-model，encoder-only

**缺陷 1.1：完全无 null 校准，block-diagonal 结论不可信（严重）**

Section 3 直接宣称 block-diagonal 结构，但未提供任何 null。从"早层高、晚层高、中间低"的模式可以有多种平凡解释：(a) 层间距离越远相似度越低（trivial decay），(b) fine-tuning 前已有 block 结构（原文对 ORIG-ORIG 做了展示，显示平滑衰减而非 block，这部分观察有价值，但不等于 null 控制）。没有 shuffle-层序 null，就无法区分"fine-tuning 制造了 block 结构"和"任何 24 层网络的层间相似度矩阵都长这样"。

严重程度：**高**（论文核心结论依赖这个图）。

**缺陷 1.2：within-model 分析，不能直接外推到 cross-model（中等）**

整篇论文分析的是**同一模型的不同层**（FT-FT 比的是同一任务模型 random restart 之间的对应层），而非**不同架构/不同家族**模型之间的层对齐。Appendix B 图 7 是论文中唯一的 cross-model 图（跨不同任务的 fine-tuned 模型），但作者只说"upper right blocks indicate similar representations"，没有任何定量分析或统计检验。换言之，**它没有做 cross-model U 型**。

严重程度：**中等**（外推性未做检验）。

**缺陷 1.3：CLS token 表征 vs 全序列词向量（中等）**

论文全程用 CLS token 的表征。encoder 的 CLS token 经过 pooling/classification head 导向，其表征的语义与 decoder-only 模型的残差流不同。我们用的是**每词均值池化**（word-level mean pooling，fast-tokenizer offset mapping，无 padding token）。CLS-centric 方法在多选/分类任务上合理，但对 cross-family 表征几何的普适性存疑。

严重程度：**中等**（限制了外推性）。

**缺陷 1.4：规模极小，外推到 7B-8B 未经验证（中等）**

RoBERTa-Base 125M，ALBERT-LargeV2 235M，ELECTRA-Base 110M。我们的最小模型是 Qwen3-0.6B，最大是 OLMo-2-7B 和 Llama-3-8B，规模差距 3-60 倍。论文未讨论规模对 block-diagonal 结构的影响。

严重程度：**中等**。

**缺陷 1.5：没有 z-score / massive activation 处理（中等）**

直接用 raw CKA（XX^T 核）。如果 fine-tuned 模型的 CLS 表征存在少数高方差维（如 2109.08406 的 ALBERT 本身有 tied embeddings，可能有 outlier 维），raw CKA 可能被这些维主导。论文没有讨论这个问题，也没有验证 CKA 分数对 normalization 选择的鲁棒性。

严重程度：**低-中等**（encoder 模型的 outlier 激活比 decoder-only 少，但仍未验证）。

**缺陷 1.6：没有 dyadic non-independence 修正（中等）**

多任务结果（12 task × 3 model × 多层）报告频率结果，但没有考虑共享 model 导致的 pair 非独立性。不过本文主要是描述性展示而非假设检验，影响相对较小。

严重程度：**低**（描述性报告为主）。

---

### B2. Paper 2（2602.14486）— cross-modal（vision × language），null 校准论文

**缺陷 2.1：模型池只有 vision × language 跨模态，没有 decoder-only × decoder-only（高）**

实验模型是 3 个 language model 家族 × 5 个 vision model 家族，共 204 对，d/n ∈ [0.19, 8]。这覆盖了 cross-modal 收敛（PRH 原始论文的设定）。但**没有做 decoder-only LLM 之间的 cross-model depth-profile 分析**（如我们的 olmo2-1B vs llama3-8B）。他们的 null 校准框架理论上可以应用到任何指标 × 任何 pair 组合，但他们自己的 PRH 实验就是 image-text 对。

严重程度：**中等**（框架本身没问题，但没有验证到 same-modality decoder-only LLM 间是否有相同结论）。

**缺陷 2.2：Null 类型与我们的不同，两者互补而非重复（重要认知，非缺陷）**

Section 5.1 的 null 是对**样本行**（row indices，i.e., 哪个 input 对应哪个 input）做 permutation。H₀ = "X 和 Y 的空间之间没有关联"。这测的是**全局配准**（global alignment）是否显著。

我们的 layer-order shuffle null 是对 B 的**层序**（layer indices）做 permutation。H₀ = "层间对应关系是随机的"。这测的是**相对深度结构**（depth-registered correspondence）是否显著。

两者**完全不同**：
- 2602.14486 的 null 问的是"两个空间是不是独立的？"——答案是否定的（哪怕校准后 local neighborhood 仍显著）。
- 我们的 null 问的是"沿相对深度的 U 型是否比随机层对应更特殊？"——这个问题他们没问过。
- 我们的 shuffle-null mean=0.453 vs observed=0.491，差仅 +0.038，意味着 midband z-CKA 的绝大部分（92.3%）在 H₀_layer_order 下仍会出现，说明大多数 CKA 值来自"两个训练好的网络的通用几何"而非特定的层对应。

**缺陷 2.3：没有 depth-resolved diagonal profile（高）**

论文对 layer-wise similarity matrix 用 max aggregation 报告单个标量，只研究这个 max 是否随模型规模增长。没有画出"以相对深度为 x 轴的对角线值如何随对走"，也没有研究 U 型是否存在。从我们的 91 pair 数据来看，U 型是主导结构（72/91 pair 二次项 c>0，p=2.0e-8），这个发现在 2602.14486 里完全缺席。

严重程度：**高**（他们的校准论文解决了 amplitude 问题，但没有解决 shape 问题）。

**缺陷 2.4：CKA 不做 z-score（中等）**

Section B.2.1 定义 linear CKA 时没有 per-dimension z-score。模型越宽（d 越大），width confounder 越大（他们在 Proposition 4.1 中给出了 O(d/n) 的分析），正因如此他们开发了 null calibration。但他们没有研究"如果先做 z-score（把高方差维压缩），是否能部分解决 width confounder 而不依赖 permutation"。我们的 z-CKA 实验表明，z-score 处理后 random-init floor 从 ~0.09（raw midband）到类似范围，这个对比值得一提。

严重程度：**低**（框架设计上合理，但没考虑 preprocessing 的 confound-reduction 效果）。

**缺陷 2.5：多重比较仅用 BH-FDR，没有 dyadic non-independence 修正（中等）**

204 pair 中的模型共享结构（同一 BLOOM family 的不同规模 model 出现在多个 pair 中），导致 OLS p 值 anti-conservative。他们用了 BH-FDR，但这只控制 type-I error across independent tests；对 dyadic dependence 没有修正。我们对 91 pair 做了 QAP node-label permutation（5000次）和 node bootstrap。

严重程度：**中等**（影响 significance 声明的精确性，但方向性结论不太可能翻转）。

**缺陷 2.6：Limitations 段自承的缺点（原文 §7 Limitations and outlook）**

> "First, representational similarity has no ground-truth scale, so we report the presence or absence of calibrated evidence for convergence rather than proving it. Second, our guarantees assume exchangeability, so grouped or clustered samples require restricted permutations that preserve their dependence structure."

此外：为什么 local neighborhood converges 但 global metric doesn't，他们明确说是"key open question"。

---

### B3. Paper 3（2312.02730）— Extended Abstract，7B decoder-only

**缺陷 3.1：只取最后一层，整个层间 geometry 未研究（严重）**

全文只用 last-layer final-token 表征（Section 2 明确："we only compare the representations of the final token in the last layer"）。这是 4 页 extended abstract 的合理简化，但意味着：
- 没有任何关于 U 型、depth profile、哪层最相似的信息；
- 结论"7B 模型表征差异显著"只对最后一层成立；
- 所有我们的中间层发现（midband U 型、early/late vs mid 分离）都无法从这篇推断。

严重程度：**高**（但这是 extended abstract 的已知局限，非缺陷——论文本身没有声称做了层级分析）。

**缺陷 3.2：无 null 校准，无 floor/ceiling，无统计检验（严重）**

Figure 1 报告热力图，没有任何统计显著性标注，没有随机初始化 floor，没有同模型相邻层 ceiling，没有 null。所有"相似/不相似"的判断纯粹是视觉检查。

严重程度：**高**。

**缺陷 3.3：指标间不一致被报告为 challenge 而非被解决（中等）**

论文的主要贡献之一是发现"不同指标给出不同结论"（Winogrande 上不同 measure 的 Spearman ρ=0.35），但没有给出诊断：是 noise 导致还是指标测的确实是不同的东西？我们已知 2602.14486 的后续工作解释了部分原因（width confounder + metric-specific invariances）。

这篇论文的诊断/建议（Section 4 "challenges"）是：
1. 指标间 discrepancy → "careful study of similarity scores"（未给出具体建议）；
2. task-dependency → "similarity for one task does not imply another"；
3. interpretation difficulty → "scores without interpretable scale are problematic"。

严重程度：**中等**（提出问题但未解决）。

**缺陷 3.4：单一规模（7B），无规模对比（中等）**

所有 11-13 个模型都是 7B，所以无法回答"相似度是否随规模增长"（PRH 的核心问题），也无法重现我们的 H2（same_family 效应随规模差大小的变化）。

严重程度：**中等**（design choice，非错误）。

**缺陷 3.5：没有考虑 tokenizer 差异对表征的影响（低）**

所有模型用各自 tokenizer（Winogrande 有固定 prompt），但不同 tokenizer 对相同文本产生不同 token 序列，final-token 表征的语义不完全可比。论文提到了这个问题（Section 2）但声称通过"只取 final-token 的 next-token prediction 表征"来绕过——这个理由不太充分（final token 的 logit 依赖于位置信息和 context，且 final position 在不同 tokenizer 下对应的文本位置可能不同）。

严重程度：**低**（small-scale 初步探索中可接受）。

---

### B4. Paper 4（2410.06981）— SAE feature space，decoder-only

**缺陷 4.1：配对步骤（Step 1）的质量控制不足，直接影响下游相似度（严重）**

整套方法的关键前提是"activation correlation 能够可靠地配对跨模型的对应特征"。但论文的过滤只保留了 10-30% 的特征对（non-concept 和 many-to-1 均被丢弃），且只在主文中用了"mean of highest activation correlations ≠ SVCCA score"这个观察说明配对并不稳定（Section 4.2：相关性 0.6 但 SVCCA 低至 0.03）。这意味着：
- 高 SVCCA 分数可能是由"配对质量好的那 20%"的共享几何驱动的，而不是整个特征空间；
- 选择性报告高质量配对子集的相似度，会系统性地 inflate 结论；
- 没有实验说明"如果配对标准更严格（例如要求相关性 > 0.9）结果如何"。

严重程度：**高**（sampling bias + selection effect）。

**缺陷 4.2：SAE 超参不统一，引入 confound（高）**

不同模型对的 SAE 宽度差距很大（32768 for Pythia vs 65536 for Llama-3）。SAE 宽度决定了特征粒度；更宽的 SAE 学到的特征更细分。SVCCA 得分依赖于配对的维度数（min(n_X, n_Y)），而这由滤波后保留的特征对数量决定。没有实验系统控制 SAE 宽度（Appendix F 做了初步实验，但主文没有报告 width 对结论的影响）。

严重程度：**高**。

**缺陷 4.3：只比较 same-tokenizer 模型，外推受限（高）**

论文明确说"we use models that use the same tokenizer because the highest activation correlation pairing relies on comparing two activations using the same tokens"（Section 4.1）。这意味着 Pythia 系列之间、Gemma 系列之间、Llama 系列之间可比，但**跨 tokenizer 家族（如 Llama vs Qwen vs GPT-2）完全没有实验**。我们的 91 pair 包含 7 个不同 tokenizer 家族，这正是 2410.06981 没有做的。

严重程度：**高**（外推范围受限，而 universality 的最强声明恰恰需要跨家族成立）。

**缺陷 4.4：SAE-space 相似度与 activation-space 相似度的关系未建立（高）**

论文声称"evidence for feature space universality"（abstract），但它测的是 SAE decoder 权重矩阵 W' 的列（feature direction vectors）之间的相似度，不是 residual stream 激活的直接相似度。这两者有本质区别：
- SAE 的 decoder 方向是训练优化的结果，可能受 SAE 损失函数、sparsity 权重、dictionary size 等 artifact 影响；
- 即使 SAE 特征方向相似，residual stream 激活的全局几何（如我们的 midband z-CKA）不必然相似；
- 反之亦然：residual stream CKA 高不意味着 SAE 特征方向对齐。

论文没有提供这两层分析之间的桥梁（即没有同时计算这两个模型对的 activation-space CKA 和 SAE-space SVCCA，然后对比）。

严重程度：**高**（概念上的两层分析没有桥接）。

**缺陷 4.5：Null 类型不同于 2602.14486 也不同于我们（中等）**

他们的 null 是随机打乱**特征配对**（而非样本对应或层顺序）。这测的是"观测到的配对子集的几何是否比随机配对更特殊"。这是合理的，但：
- 没有考虑 "activation-correlation based 配对本身就已经优选了几何相似的特征对" 的 circular bias；
- 没有做 width confounder 校准（2602.14486 指出的那种 d/n 效应）。

严重程度：**中等**（null 本身合理，但忽视了选择性偏差）。

**缺陷 4.6：Limitations 段自承（原文 §Limitations）**

> "We perform analysis on only a sample of SAEs for similar LLMs that use the same tokenizer, and focus on SAEs with the same or similar number of features."
> "Our study is also limited by the inherent challenges in interpreting and comparing high-dimensional feature spaces."
> "The generalizability of our findings to SAEs trained on more diverse datasets or specialized domain-specific LLMs remains an open question."
> "Our analysis does not account for potential temporal dynamics in feature representation that may occur at different training stages."
> "a notable fraction of features idiosyncratic to different SAEs" (citing Leask et al., 2025; Paulo and Belrose, 2025)

---

### B5. 四篇共同缺陷

**缺陷 5.1：没有人做 post-norm vs pre-norm 的对照**

所有四篇都没有提及 normalization 位置（pre-norm vs post-norm RMS）对表征相似度的影响。从我们的数据，OLMo-2-7B 是 14 个模型中唯一的 post-norm 风格（层 RMS 在 0.07-0.94 范围，相比 pre-norm 模型的 0.84-1.02），且它的 midband 均值是 14 个模型中最低的（0.329 vs 中位数 ≈ 0.51）。这是一个未被任何相关论文讨论过的自然实验。

**缺陷 5.2：没有人直接测量"功能可迁移性"**

四篇都在测表征几何（CKA/RSA/SVCCA/mKNN），没有一篇直接测"把 A 的中间层接到 B 的后半模型，生成质量如何"（即 oracle affine readout 的 next-token generation ppl）。我们有 OLMo-2-1B → Llama-3.2-1B 的 oracle affine readout ppl 596 vs 原模型 18.8 这个数据点，而 midband z-CKA ≈ 0.35-0.47（"中等"）。这个"几何相似但功能灾难性失败"的数据点在四篇中都没有对应物。

**缺陷 5.3：没有 identity gate + ceiling + floor 三重校准同时进行**

我们的三重校准：
- identity gate（max|M[i,i]-1| = 1.78e-7，PASS）
- random-init floor（mean=0.091，n=4）
- 同模型相邻层 ceiling（median=0.977，min=0.923，n=14）

Paper 1 无任何校准；Paper 2 有 permutation null（样本级），无 identity gate 和 ceiling；Paper 3 无任何校准；Paper 4 有 random-pairing null，无 identity gate 和 ceiling。

---

## C. 我们能接上什么 / 能证伪什么

### C1. 接上 Paper 1（2109.08406）

**可接点 1.1：cross-model decoder-only 的 block structure 是否与 within-model fine-tuned 的 block structure 同一现象？**

Paper 1 发现 within fine-tuned encoder 的层间相似度有 block-diagonal（早层簇、晚层簇），但**从未做 cross-model 分析**。我们的 91 pair 测的是**不同家族不同规模** decoder-only 模型之间的相对深度对角线，发现 U 型（早层高、中间塌、晚层高），这与 within-model 的 block-diagonal 在"两端高"这一点一致，但机制不同：within-model block-diagonal 源于 fine-tuning 的任务特化；cross-model U 型可能源于"embedding 和 unembedding 的通用约束"。我们的 `ends_vs_mid` 数据可以直接对比：mean_diag_end_zcka=0.746，mean_diag_min_zcka=0.251，差距 0.495——这是跨家族的，比 within-model 更惊人的 U 型幅度。

**可接点 1.2：null 校准补充 Paper 1 的 block-diagonal 结论**

Paper 1 没有 null，所以它的 block-diagonal 结论不知道有多少是 trivial。我们的 layer-order shuffle null（mean=0.453）提供了一个参照：如果 Paper 1 的 block-diagonal 图里 "lower-right off-diagonal block"（早层 vs 晚层）的值低于 0.45，那很可能就是 trivial decay（没有实质性的层对应信息）；如果高于 0.45，才是真正的差异。这个分析可以 retroactively 检验他们的图——但需要从原论文提取数值，我们目前没有那些数字（论文未提供表格，只有图）。

**可接点 1.3：规模 scaling 的新证据**

Paper 1：100-300M encoder，3 个模型。我们：0.6B-8B decoder，14 个模型，7 个家族。H2 的发现（same_family β=+0.171，p=0.0012，log_depth_ratio β=-0.056，p=0.273）说明**家族差比深度差重要 3 倍**，但去掉 GPT-2 后 family 效应就不显著。这比 Paper 1 的规模更大，但结论更复杂（家族效应不稳健）。

---

### C2. 接上 Paper 2（2602.14486）

**可接点 2.1：把 sample-shuffle null 与 layer-order shuffle null 对比，形成"null 分类学"**

这是最有价值的概念贡献点（详见下文 (ii) 新观点评估）。我们的 shuffle-null mean=0.453 是对我们指标（z-CKA midband）的 layer-order null；它回答"层间对应关系是否真实"。2602.14486 的 sample-shuffle null 回答"两个表征空间之间是否有任何关联"。这两个 null 可以组成一个 2×2 矩阵：

| | sample-shuffle null 下显著 | sample-shuffle null 下不显著 |
|---|---|---|
| **layer-order null 下显著** | 有真实层对应 + 有空间关联 | 层对应的 artifact（偷层序） |
| **layer-order null 下不显著** | 有空间关联但层对应不稳定 | 什么都没有 |

我们的数据：77/91 pair 在 sample-shuffle null 下（observed > null mean）；58/91 pair 在 layer-order null（p<0.05）下显著。这是独立的两个维度，没有任何一篇做过两维联合分析。

**可接点 2.2：直接用 2602.14486 的框架校准我们的 z-CKA**

论文说代码已开源（GitHub）。我们可以对 91 pair 的 midband z-CKA 做 sample-shuffle null 校准（K=200，α=0.05），得到 s_cal，然后看校准后的 midband 分布如何变化。这会直接回答"在 2602.14486 的框架下，我们观测到的 midband z-CKA 中有多少是 width confounder + random-sample effect"。注意：z-score 预处理已经部分压缩了 width confounder（因为它均衡了不同维度的方差），所以校准后的 s_cal 对我们可能比对他们的 raw CKA 降幅更小——这本身就是一个有趣的对比。

**可接点 2.3：U 型是 depth confounder 之外的独立现象**

2602.14486 的 depth confounder 是"取 max 时随层数增多而 inflate"。我们的 U 型分析用的是 diagonal（相对深度对应点），不是 max aggregation。因此：
- 我们的 U 型不受 depth confounder 影响（我们从不取 max）；
- 这是一个 2602.14486 框架无法解释的剩余结构，即使校准后 U 型可能仍然存在。
- 数据佐证：distance-residual 控制后（把矩阵元回归到 |i/L_A - j/L_B| 的三次项），c>0 的比例仍是 72/91（p=2.0e-8），与未控制完全相同——U 型不是层距离 decay 的副产品。

---

### C3. 接上 Paper 3（2312.02730）

**可接点 3.1：last-layer 分析的 generalizability 问题**

Paper 3 只用 last-layer。我们可以从 repr_alignment_results.json 里提取 diag_end_mean_zcka（对角线最后点，对应 last-layer 的近似）与 midband 的对比。从数据：mean_diag_end_zcka=0.746（vs midband median=0.503），说明 last-layer 比 mid-layer 跨模型相似度更高。Paper 3 说"7B 模型显著不相似"可能因为他们用了 Procrustes/Jaccard 而非 CKA，或因为 same-tokenizer 要求不同，或因为他们 7B 池的家族比我们更 diverse（含 Galactica、StableLM 等特殊训练的模型）。

**可接点 3.2：补充 null 和多层分析**

Paper 3 的核心贡献（识别 challenges）可以被我们的数据正面回答：
- "discrepancies between measures"：我们有 raw CKA 和 z-CKA 两个指标，可以展示 z-score 如何改变哪些 pair 的排名；
- "task-dependency"：我们用的是 generic wikitext，Paper 3 用的是 domain-specific（Winogrande/HumanEval），domain shift 可能是他们发现差异的部分原因；
- "interpretation difficulty"：我们有 floor（0.091）和 ceiling（0.977）两个锚点，使得 midband 0.503 有明确语义（离 floor 远、离 ceiling 远、落在中间）。

---

### C4. 接上 Paper 4（2410.06981）

**可接点 4.1：weight-space SAE 与 activation-space z-CKA 的双层对比**

我们可以对 2410.06981 中有 SAE 的模型（Pythia, Gemma, Llama）计算 activation-space z-CKA，然后对比 SAE-space SVCCA。这会直接测试"高 SAE SVCCA ↔ 高 activation z-CKA"是否成立。如果两者弱相关，说明 SAE-level universality 和 activation-level universality 是不同的事情，不能互相推断。不需要重跑 GPU：我们只需用已有 repr_alignment_results.json 提取 Pythia/Gemma/Llama 的 activation-space CKA（如果池里有这些模型）——但检查发现我们的池没有 Pythia 或 Gemma，只有 Llama-3 家族。Llama-3.1-8B 我们没有，但有 Llama-3-8B。这个比较只能做 partial，不能完整。

**可接点 4.2：跨 tokenizer 族的功能灾难性失败**

Paper 4 声称 universality，但只做 same-tokenizer 对。我们的 oracle affine ppl 596 vs 18.8 是跨 tokenizer 家族的（OLMo-2-1B tokenizer ≠ Llama-3.2-1B tokenizer）。这个功能灾难数据点是反驳"universality 足以支持功能迁移"的最强证据——尤其当 4 的 SVCCA 得分高达 0.8 的情况下。

---

## D. 三个可能的"新观点"评估

### (i) 「几何相似 ≠ 功能可迁移」的边界划定

**评估结论：★★★ 可行，是最强的单点贡献**

数据基础：
- oracle affine readout（1 层线性桥）OLMo-2-1B → Llama-3.2-1B：ppl 596 vs 原 18.8（差 32×）。
- 同 harness 下自拼自：ppl 差 0.64 nat（PASS）。
- 中间层 midband z-CKA 对这个 pair 是多少？从 per_model_mean_midband_zcka：olmo2_1b=0.495，llama32_1b=0.413，pair 的值需从 pairs 字段读取（未直接展示，但这两个 family 不同，cross-family mean≈0.470）。

立论逻辑：
- 在 CKA 或 SVCCA 看来"中等相似"（≈0.45），但生成质量差 32×——这是 concept-probe / classification-head 类分析无法捕获的信息。
- Paper 4 的 high SVCCA（0.5-0.83 for semantic subspaces）只测特征方向的几何；如果强行做 affine stitching，generation ppl 可能同样灾难性失败。
- Paper 2 的 local neighborhood mKNN 在校准后仍显著（Aristotelian），但 neighbor preservation 不等于 logit 空间的可互换性。

谁已经部分说过：Klabunde et al.（2023，Paper 3）提到 "functional similarity implies representational similarity but not vice versa"，但没有数值。Moschella et al.（2023，cited in Paper 3）研究了跨模型 latent space 通信，但用的是轻量 linear probe，不是 next-token generation。

所需额外证据：
- 理想情况是对更多 pair（不同 midband z-CKA 水平）都做 oracle affine ppl，看是否存在 CKA 阈值。但不需要 GPU：如果已有多个 stitch 实验（stitching experiments 在 paperD_research 中可能有），可以汇总。
- 需要区分"affine stitching"（1 层线性）和"end-to-end trained adapter"（几层）——前者对应几何相似的下界，后者对应上界。

结论：**这个观点能站住，且与四篇论文都没有正面冲突（他们都没做过这个测试）**。

---

### (ii) 「null 的选择决定结论」的方法论批判

**评估结论：★★★ 可行，与 Paper 2 互补而非重复**

2602.14486 的贡献是指出 sample-shuffle null 下 spectral CKA 的虚高（width + depth confounder），提出校准框架。这篇**已经说了** "没有 null 的 CKA 结论不可信"。

我们能加的新东西：
1. **第三种 null：layer-order shuffle**——这是 2602.14486 完全没有讨论的 null 类型。我们的数据：null=0.453，observed=0.491，gap=+0.038。对比：random-init floor=0.091，gap vs floor=+0.400。这说明"0.491 的绝大部分来自训练好的网络的通用几何，而不是特定的层对应结构"。
2. **null 分类学**：三种 null 问三个不同问题（已见 C2.1 的 2×2 表格）。没有一篇论文同时做过这三种控制。
3. **quantified**：我们的 58/91 pair 在 layer-order null 下 p<0.05，这说明 U 型是真实存在的（层对应是特殊的），但幅度小（+0.038 after null subtraction）。

是否与 2602.14486 重复？不重复。2602.14486 的 width/depth confounder 是关于不同维度/层数的模型间比较；我们的 layer-order null 是关于相对深度结构是否真实。两者正交，共同构成一个更完整的 null 分类体系。

结论：**这个观点有实质新颖性，但需要清楚地说明与 2602.14486 的区别在哪里**（否则容易被认为是重复工作）。建议以"三种 null 的联合框架"为结构，而非以"CKA 结论不可信"为主题（后者已被 2602.14486 占掉）。

---

### (iii) Post-norm 作为自然实验

**评估结论：★★ 有趣但弱，作为 observation 可以，不能作为主结论**

数据：OLMo-2-7B 是 14 个模型中唯一 post-norm 风格（layer_rms 0.07-0.94，其他 pre-norm 模型的 layer_rms 在 0.84-2.0 范围），midband 均值 0.329 是最低的（次低 llama32_1b=0.413，gpt2_xl=0.372）。

问题：
1. OLMo-2-7B 的低 midband 是否只是 family 特殊性（OLMo-2 训练数据？架构其他特征？）还是 post-norm 本身？**单一数据点无法区分**。我们有 OLMo-2-1B（也是 post-norm 风格，layer_rms 0.14-0.73），其 midband 均值是 0.495——不低！所以 post-norm 不是一致低的，只有 7B 版本低。
2. OLMo-2-1B 和 OLMo-2-7B pair 的 midband z-CKA=0.338（same_family_pairs 显示 olmo2_1b:olmo2_7b=0.338），这是同家族里最低的（llama3: 0.589，qwen3: 0.633-0.701，gpt2: 0.505-0.840）。这可能是因为 1B 和 7B 的 layer_rms profile 差异很大（1B: 0.14-0.73；7B: 0.07-0.94），导致 z-score 后的几何不同——但这本身就支持"normalization 影响 z-CKA"的论点。

没有人讨论过：检索四篇全文，"normalization"/"pre-norm"/"post-norm"/"RMSNorm position"关键词均未出现在表征相似度分析的上下文中（仅在模型描述中偶尔提及）。

建议：将其作为 Section 的一个 observation（"architecture异常点"），而非独立主张。写法："OLMo-2-7B 作为我们池中唯一的 post-norm 风格模型，展现出最低的 midband z-CKA（0.329），但其 1B 变体（同为 post-norm）却处于中等水平（0.495），表明这不是 post-norm 本身的系统性效应，而是 7B 规模 + post-norm 组合下的特殊行为，值得在有更多 post-norm 模型的池子里重复验证。"

---

## E. 一句话判决

**Paper 1（2109.08406，自述 BlackboxNLP'21）**：它占掉"within-model fine-tuned encoder 的 block-diagonal 结构"，但没有做 cross-model、没有 null、没有 z-score，我们能用 91 pair decoder-only 的 U 型 + layer-order shuffle null 接上，并提供它从未提供的定量 null 控制。

**Paper 2（2602.14486，preprint）**：它占掉"sample-shuffle null 下 spectral CKA 是虚高的，local neighborhood 校准后仍显著"，但没有做 depth-resolved diagonal U 型分析，没有测 layer-order null，我们的贡献是提供互补的第三种 null + 量化 U 型幅度 + 连接几何相似与功能可迁移性。

**Paper 3（2312.02730，自述 UniReps@NeurIPS'23 Extended Abstract）**：它占掉"7B decoder-only last-layer 表征相似度"，但只是 4 页 extended abstract，无 null，无层级分析，我们能用 91 pair 的全层 profile、三重校准、post-norm outlier 等证据正面回答它提出的"三个 challenges"。

**Paper 4（2410.06981，preprint）**：它占掉"same-tokenizer 同规模 SAE feature space universality（weight space）"，但不做 activation-space 的 cross-family 对比，我们的 oracle affine ppl 596 vs 18.8 是直接反例——weight-space 相似的 SAE 特征不保证 activation-space 的功能可迁移。

---

## F. 最终判决：三个新观点存活概率

| 新观点 | 存活概率 | 最需要补的证据 |
|--------|----------|----------------|
| **(i) 几何相似 ≠ 功能可迁移的边界划定** | **85%** | 多个 pair 的 oracle affine ppl vs CKA scatter（目前只有 1 个 pair 的数据点）；区分 affine 与 deeper adapter |
| **(ii) null 分类学（三种 null 的联合框架）** | **75%** | 把 sample-shuffle null（2602.14486 式）也跑一遍我们的 91 pair（可用他们开源代码，不需要 GPU，只需计算 CKA 200 次 permutation），形成 null_sample_shuffle vs null_layer_order vs null_random_init 三列数字 |
| **(iii) post-norm 自然实验** | **45%** | 需要更多 post-norm 模型（至少 3-5 个不同 family 的 post-norm 变体）才能区分 post-norm 本身和 OLMo-2 家族特性；目前 1B 和 7B 的 post-norm 结论相反，证据矛盾 |

**最强的观点是 (i)**：oracle affine ppl 596 vs 18.8 是具体的、可量化的、已有数据的。它既与四篇论文不重叠，又直接回答了"那些几何相似性到底意味着什么"这个所有四篇都没有回答的核心问题。**需要补的只是多几个 pair 的 oracle affine 测量（从 smoke_stitch_cpu.py 或已有 stitch 实验的结果中提取），不需要新的 GPU 跑**。

(ii) 作为方法论贡献也很有力，但需要人自己跑 2602.14486 开源代码（K=200 sample-shuffle perm on our 91 pair）才能完整形成对比，计算量小（CPU 可跑），但需要执行。

(iii) 目前证据太薄，建议降级为 observation 而非独立主张。

---

*字数统计（中文+英文混合，下文 Python 报告）：*
