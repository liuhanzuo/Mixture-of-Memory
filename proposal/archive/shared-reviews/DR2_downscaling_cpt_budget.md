# DR2：三篇论文对抗性 Review — 压缩/续训后能力变化的静态 vs 时间维度

**作者**：主 agent（claude-sonnet-4-6[1m]）  
**日期**：2026-08-06  
**目标**：精读三篇论文全文，找方法论缺陷、可疑结论，以及我们能接上/证伪什么。

---

## 论文一：arXiv:2310.04680（ICLR 2024）

**全称**：The Cost of Down-Scaling Language Models: Fact Recall Deteriorates before In-Context Learning  
**作者**：Tian Jin, Nolan Clement, Xin Dong, Vaishnavh Nagarajan, Michael Carbin, Jonathan Ragan-Kelley, Gintare Karolina Dziugaite  
**Venue**：ICLR 2024（已核实，Appendix J 提及 TPU v3 + SparseGPT 实现，与 ICLR 2024 已知信息吻合）

---

### A. 究竟做了什么（精确到实验设置）

**模型与规模**：6 个模型，3 个家族
- OPT-13B、OPT-30B（训练于 180B tokens，14 和 5.5 tokens/param）
- LLaMA-13B、LLaMA-33B（1T 和 1.4T tokens）
- Pythia-12B（原版 + deduped，见 Appendix G）
- 规模范围：13B–33B（只测**最终 dense checkpoint**，无中间 checkpoint）

**Pruning 方法**：
- 主用 **SparseGPT**（Frantar & Alistarh, 2023）：one-shot 非结构化权重剪枝，在各 FC 层最小化 ℓ₂ 输出差，**剪后更新剩余权重**。
- 附录 F 验证 **Wanda**（Sun et al., 2023）：同为 one-shot 非结构化剪枝，但**不更新剩余权重**。
- **这是非结构化（unstructured）/ 权重级稀疏剪枝**，不是深度剪枝（layer removal）或宽度剪枝（head/FFN 维度削减）。
- dense scaling 实验用 OPT 全家族（同 300B tokens 训练），测不同参数量模型之间的能力变化。

**Benchmark（Table 1）**：
- 事实回忆（Fact Recall）：TriviaQA（closed-book）、WebQuestions（closed-book）
- In-Context Learning（ICL）：
  - Open-book QA：TriviaQA(Filtered)（881 questions，加上下文证据）、NaturalQuestions（factual context）
  - Overriding QA：DisentQA（合成 context，答案**故意与训练事实矛盾**）
  - 参数化函数学习：Linear Classifier、2-layer NN、Decision Tree（D=4, N=32 in-context exemplars，2048 task instances）
- 所有评估用 **exact match accuracy**，greedy decoding。

**有无训练轨迹**：**没有**。所有测量仅针对「特定稀疏度的最终 checkpoint」（sparsity 0% 到 90%，以 10% 为步长）。文章的横轴是 *sparsity level*，不是 training step。**没有任何时间轴**。

**核心结论出处（Figure 1, 2, 3, 4）**：
- Section 4 + Figure 2：accepting **5% relative accuracy drop** 阈值，closed-book 最高可接受稀疏度 = 30–40%；open-book = 50–60%；overriding = 70%。
- Section 5 + Figure 3：Linear/NN/Decision Tree ICL 任务可到 60%/60%/70% 而无明显退化。
- Appendix B + Table 2：PPL 数字（C4 验证集）：OPT-13B @ 60% = 13.7（vs 0% = 11.5）；LLaMA-13B @ 60% = 10.0（vs 0% = 6.6）。PPL 在 60% 稀疏度时上升但仍 reasonable；90% 时崩溃（822–40362）。
- Section 7 + Figure 5：FFW 层对 fact recall 更重要；attention 层对 ICL 和 fact recall 同等重要。

**样本量**：TriviaQA 7993 题，WebQA 2032 题，DisentQA 7700 题，ICL 每类 2048 instances。无 bootstrap CI 或显著性检验。

---

### B. 方法论缺陷清单

**[B1] 无 heal/continued pretraining → 这是最大的接入口**（严重程度：高）

Section 3（Pruning algorithms）和 Appendix A 均明确说：
> "Both are one-shot pruning algorithms that scale to LLMs and outperform magnitude pruning, **without computationally intensive re-training**." (Section 3)
> "More sophisticated one-shot/iterative pruning algorithms exist. They typically require re-training – redoing the training of these foundation models for every sparsity level we examine. **The cost of such an experiment is at least in the millions of dollars therefore beyond our means.**" (Appendix A)

**问题**：论文呈现的是 prune 之后**不做任何恢复训练**直接测性能——这测量的是「一次性破坏」的影响，不是「破坏后给多少 token 能恢复」的问题。它的稀疏度阈值（30%/60-70%）完全是单时间点快照，不反映「愈合能力」。这留下了一个完整的未占领空间：**同样的 structural damage，heal 之后能力回来的速度有多快，各能力的恢复 token budget 是否不同**。

**[B2] 非结构化 vs 深度剪枝：不可直接类比**（严重程度：中-高）

本文 100% 是非结构化权重稀疏（SparseGPT/Wanda），每层的权重矩阵被置零到指定比例，但**层数不变、深度不变、模型拓扑不变**（只是权重变稀疏）。  
我们的 Paper B（prune-heal）是**深度剪枝 + 嫁接**：keep 前 N 层，完全丢弃深层，添加 K=2 新层。两者在以下方面根本不同：
- 非结构化剪枝保留了所有层的信息流拓扑；深度剪枝完全切断了靠后 layer 的表示。
- 非结构化剪枝的"可修复性"未知；深度剪枝强制从深层信息流中的知识重新 heal。
- 所以本文的 30% = "PPL 开始明显上升的点"，与我们的 "keep 14/32 层 = 56% 参数保留" 在量级上可能对应，但机制完全不同，**数字不可直接比较**。

**[B3] 5% relative 阈值的任意性**（严重程度：中）

Section 4：
> "accepting a relative decrease of 5% from pruning in the mean accuracy over four models"

这个 5% 的"可接受下降"阈值是作者随意选定的，文中没有任何讨论为什么是 5% 而不是 10% 或 2%。如果用 10%，fact recall 的阈值会显著上升，30% 的结论会变成更接近 50%，整个"30% vs 60-70% 鸿沟"的叙事可能收窄甚至消失。作者也在 Figure 1 的图注中只是把这个阈值"定义"出来而没有论证。

**[B4] PPL 轴没有做系统对照**（严重程度：中）

PPL 测量被放在 Appendix B 里一带而过。论文的主线（fact recall vs ICL）都是用 **downstream accuracy** 表达的；PPL 只有 Table 2（C4 验证集上的 token-level PPL，Appendix B）。  
**关键缺口**：论文没有尝试把 PPL 纳入同一个坐标轴，即"当 PPL 退化 X 时，fact recall 退化 Y，ICL 退化 Z"。换句话说，它回答了"pruning depth 影响哪种能力更多"，但没有回答"PPL 是能力退化的好代理吗"——而这正是 2506.00288 的核心主张，也是我们 keep14 PPL 1.428× vs MMLU 19.5% 这组数字所在的空间。

**[B5] 任务设计对 ICL 组的 chance baseline 未充分报告**（严重程度：中）

Section 5 和 Appendix C-D 的参数化函数 ICL 任务（Linear CLS, 2-layer NN, Decision Tree）是 2-way 或 4-way 分类，chance baselines 分别是 50% 和 25%。作者在 Appendix D（"Evaluation label distribution"）说：
> "Each evaluation input has an equal probability of receiving one of the K possible classification labels."

但在主文图 Figure 3 中，dense model 的 ICL 准确率明显高于 chance（如 60–80%），且高度区分于 random。**问题是**：作者仅在 Appendix I 提到了"answer without context"的比较，但对于 2-way binary classification，dense 模型本身是否有强烈的默认偏好（label prior bias），剪枝后是否只是让模型更均匀地猜（因此 accuracy 不变甚至提升）——这一点作者自己在 Appendix I 也只是"假设，而非验证"：
> "We hypothesize, without testing, that pruning may improve task accuracy by enhancing the effect of contextual information on its prediction." (Appendix I)

**[B6] 多模型平均掩盖了 model-specific 异质性**（严重程度：低-中）

Figure 1 的核心曲线画的是 4 个模型的平均性能（红色），但 OPT 和 LLaMA 的 tokens/param 差距巨大（OPT-13B = 14 tokens/param vs LLaMA-13B = 77 tokens/param）。tokens/param 不同意味着 prune 的信息密度不同。此处平均可能掩盖了一个重要趋势：**高度 over-trained 的模型（如 LLaMA）是否比 under-trained 的模型（OPT）更能承受 pruning**。作者没有这个方向的分析。

**[B7] 自述 Limitations（Appendix A）**

1. 只用了 one-shot pruning，没有做 iterative pruning（承认这是计算代价问题）。
2. "our observations may not generalize to the full spectrum of tasks and large language models" — 承认 6 个模型的 empirical 范围有限。
3. 没有讨论 structured pruning、头剪枝等结构化方法。

---

### C. 我们能接上什么

**接口 1（最直接）**：B1 留下的真空  
本文证明了"静态 prune 后 fact recall 受损早于 ICL"。我们的资产是：**同样的 structural damage 之后，heal 200k steps，PPL/fact recall/ICL 的恢复轨迹各是什么形状**。具体可用：
- keep14 PPL 轨迹（step 0→200k）× 5 eval arry 证明"恢复 PPL ≠ 恢复 MMLU"
- matched-PPL 对照（keep14@step67500 vs baseline @ 同 PPL）证明"即使 PPL 匹配，MMLU 差 XX pp"
- 这直接把本文的"静态切片"推进成"时间轴分析"，且我们的因变量是同一批任务（closed-book QA, MC benchmarks）

**接口 2（对比论证）**：B2 的机制差异  
本文的"60-70% 保留 ICL"是非结构化 pruning 结论，我们的 keep14（56% 参数）是深度剪枝。如果我们的 keep14@200k ICL（如 hellaswag/piqa 类 MC）能恢复到 baseline 附近，而 MMLU（fact-intensive）不能，这**平行支持**了本文的核心主张，但用一个更激进的 structural damage 类型复现了它，增加了泛化性。如果两者不一致，说明 structural type 有影响，这是原文没有讨论的。

**接口 3（矛盾检测）**：本文"60-70% 削减基本保留 ICL"  
keep8（= 8/32 层 = 75% depth reduction + 新层，参数量比 keep14 少得多）的 MMLU 始终在 chance 水平。但我们需要区分：
- MMLU 究竟是"fact recall 任务"还是"ICL 任务"？在本文的分类（Table 1）中，MMLU 更接近**知识密集型 closed-book QA**（fact recall），而不是本文定义的 ICL（open-book extraction / parameterized function learning）。
- 因此 keep8 MMLU 不提升 ≠ 与本文矛盾——实际上可能是**支持**本文（fact recall 最难恢复）。
- 真正能考验"ICL 能力"的 eval 是我们的 core6 MC（hellaswag/piqa/winogrande）——这些更接近本文 open-book 或 pattern learning 类别，应该比 MMLU 恢复更快。**如果我们跑 core6 vs MMLU 的恢复速度对比，就是直接复现本文主张的「heal 版本」**。

**[B4 的接口]**：PPL-vs-ability 解耦  
本文没有做 PPL-vs-accuracy 的系统对照图。我们 keep14 PPL 1.428× vs MMLU 19.5% 的数据点，可以作为"PPL 和能力恢复不同步"的实证支撑——但这个观点不是本文的主张（本文没有讨论 heal 过程），而是 2506.00288 的核心主张。见下节。

---

---

## 论文二：arXiv:2506.00288（自述 ACL 2025 main，未经 S2/DBLP 独立核实）

**全称**：Emergent Abilities of Large Language Models under Continued Pretraining for Language Adaptation  
**作者**：Ahmed Elhady, Eneko Agirre, Mikel Artetxe（HiTZ Center, UPV/EHU + Reka AI）  
**Venue**：论文 PDF/HTML 无 explicit venue 字段；参考文献风格和摘要叙述与 ACL 2025 main 投稿格式一致，但这是**论文自述，未经独立数据库（S2/DBLP）核实**。

---

### A. 究竟做了什么（精确到实验设置）

**模型**：
- 主实验：Llama 2 7B（base，非 chat）
- 对比实验（全在 Basque）：Llama 2 13B、Llama 3.1 8B、Gemma 2 9B

**数据**：3 种目标语言（Basque、Arabic、Indonesian）。
- Basque：Latxa corpus（4.7B tokens），English：Pile 随机 500k docs（占总 CPT tokens 20%）。Arabic、Indonesian：CulturaX 随机采样（4.5–4.7B tokens，与 Basque 持平）。

**训练设置**：10k steps，48 × 8 A100 GPU，LR=1e-4（cosine + 10% warmup），max seq len=4096，effective batch size=256。

**有无 step-wise 轨迹**：**有，但粒度不明确**。Figure 1（PPL 曲线）、Figure 2（Copain ICL 曲线）、Figure 3（下游任务 choice perplexity 曲线）、Figure 4（L2 parameter distance 曲线）都是 step-wise 曲线，覆盖 0→10k steps。但**文中没有说明 checkpoint 频率**（是每 100 步、每 500 步还是每 1000 步），Figure 中曲线只能从视觉分辨出大致步点。

**Benchmark**：
- 验证集 PPL（Basque/Arabic/Indonesian 各自语言）
- 下游 multiple-choice：EusTrivia/EusProficiency/EusExams/EusReading（Basque）；ArabicMMLU；IndoMMLU
- **Copain**（本文自造的 language-agnostic ICL benchmark）：7 个任务，每个 150 例（共 1050），涵盖 min/max/median in list、odd/even 识别、字母序。Exact match。
- 附录 C：MGSM-eu（数学推理，人工翻译版 GSM8K）

**核心数字（Table 2，Llama 2 7B，Basque）**：

| 条件 | PPL(eu) | Downstream(avg) | Copain |
|------|---------|-----------------|--------|
| 初始模型 | 23.64 | 27.43 | **44.67** |
| +CPT (eu+en) | 3.35 | **34.14** | 43.43 |
| +CPT (eu) | 3.58 | 28.89 | 20.12 |

PPL 差距：3.35 vs 3.58（不到 7%）；Downstream 差距：34.14 vs 28.89（**+5.25 points**）；Copain 差距：43.43 vs 20.12（**-23.31 points catastrophic forgetting**）。

**"until later in training"具体指多少 steps**（Section 4.2 + Figure 1b）：
> "a sudden improvement of 8 points between steps 2k and 4k" （CPT eu+en 模型的下游准确率突升）  
> "Even if the difference in downstream accuracy becomes prominent later on training (around step 3k in Figure 1(b))" （CPT eu 模型始终未涌现）  
> Copain 退化：CPT eu 模型在"first few steps"降至接近零，Figure 2 显示约 step 1k 之前。  
> "at the 100th step, the cumulative L2 distance is 7x higher for the variant without English, reaching 15x by the 1000th step." (Section 4.4)

所以"until later"具体是：**Copain 崩溃发生在 step ~100–1000 内；PPL 差异要等到 step 3k 才在 accuracy 上体现**。两者之间有约 2–3k steps 的"沉默期"（damage 已发生，accuracy 还看不出来）。这个"延迟"**被定性描述了（~2k–3k steps），但没有被量化成 token budget 数字**（10k steps × 256 batch × 4096 seq = ~10.5B tokens 总量，2k steps ≈ 2.1B tokens）。

**L2 parameter distance 度量**（Section 4.4）：
> "at the 100th step, the cumulative L2 distance is 7x higher for the variant without English"  
> 用的是"average layer-wise L2 distance of model parameters from the initial Llama 2 7B model"（Figure 4）

---

### B. 方法论缺陷清单

**[B6] "only impacts validation perplexity" 结论的单语言偏差**（严重程度：高）

Abstract 和 Section 4.1 的核心主张：
> "including English does not impact validation perplexity, yet it is critical for the emergence of downstream capabilities"

这个结论**严格来说是相对于目标语言验证集 PPL**（eu/ar/id 的同语言验证集）。训练数据也来自同一语言语料的 split。所以"PPL 不受影响"的意思是：在**同分布 next-token prediction**上两者相同——但他们从未测过**英语 PPL** 是否保持。CPT (eu) 模型可能 Basque PPL 不变但英语 PPL 大幅上升，这跟 2310.04680 的 "C4 PPL" 轴根本是不同的东西。作者没有检验英语 PPL 是否下降，却用这个结果来主张"PPL 不反映能力"，这个因果链是有问题的——准确说应该是"**目标语言 next-token PPL 不反映 ICL 能力保留**"。

**[B7] "until later in training" 量化缺失**（严重程度：高）

Section 4.2 和 4.3 把"延迟"现象清晰地描述了（step 100–1000 vs step 3k），但整篇论文**没有一处把这个延迟量化为 token budget 数字**，也没有提出"给定损伤程度，recovery 需要 X tokens"这样的公式。"until later in training"停在定性层面，而不是"需要 2.1B tokens 的 critical 阶段"这样可预测的数字。这是本文最大的未完成部分。

**[B8] Copain benchmark 的 chance baseline 与难度分析不足**（严重程度：中）

Copain 任务的 chance baseline 约等于多少？  
- Min/Max of 3 integers：uniform 随机猜 chance = 33%。
- Median of 3 integers：chance = 33%。
- Even among odds（4 个数中找1个）：chance = 25%。
- 字母排序 first/last（3个）：chance = 33%。

CPT (eu) 模型 Copain = 20.12%，**已经低于 1/4 的 25% chance level**（对于部分任务甚至低于 33%）。这意味着模型不是在"随机猜"，而是有系统性地**错误应答**。这种 below-chance 现象是分类任务中 label bias 或 format confusion 的典型症状，文章对此没有诊断（例如：模型是否总是输出某个特定 token？format 是否崩溃？）。

**[B9] 同分布 PPL vs 泛化 PPL 混用**（严重程度：中）

Section 4.3 提出"perplexity of choice labels"（Eq. 1-2）作为衡量下游泛化的 PPL 代理，这个想法很好，但把它与 validation PPL（同训练分布）混在同一图内（Figure 3），没有区分两者不同的物理含义：
- Validation PPL（Figure 1a）= 目标语言 next-token，in-distribution
- Choice label PPL（Figure 3）= 多选题答案 token 的 conditional probability，**out-of-distribution**

两者都叫"PPL"但测的是不同东西。图 3 显示 CPT (eu) 模型的 choice PPL 飙升，但这与 Figure 1a 的 validation PPL 平稳并不矛盾——因为 choice PPL 是在一个完全不同的 prompt 格式下测的。文章的"两者都叫 PPL，但行为不同"的论述方式容易让读者误以为是同一个指标上的矛盾。

**[B10] CPT 步数太少（10k steps）限制了 long-term recovery 观察**（严重程度：中）

所有实验只跑了 10k steps（≈10.5B tokens）。Section 4.2 说 CPT (eu) 模型在 Copain 上"slow partial recovery"但"far from recovering"——但这是在 10k steps 内。没有人知道如果给 CPT (eu) 模型跑 100k steps 会怎样。**文章声称"catastrophic forgetting"是永久性的，但实际上没有足够的数据支持这个结论**，只是在有限训练量内观察到未恢复。

**[B11] 与 Llama 3.1 / Gemma 2 的异质性结果处理不充分**（严重程度：中）

Table 2 显示：
- Llama 3.1 8B：CPT (eu+en) Copain=42.04 vs CPT (eu) Copain=41.19（**差距只有 0.85 points**）
- Gemma 2 9B：CPT (eu+en) Copain=50.23 vs CPT (eu) Copain=43.59（差距 6.64）

对于 Llama 3.1，"English is critical"这个结论在 Copain 上几乎**不成立**（0.85 points 差距在统计上可能不显著）。Section 5.1 解释为"Llama 3.1 already decent at modeling the target language distribution"，但这只是事后解释，没有预测。整个论文的 main claim 是从 Llama 2 推广出来的，而 Llama 3.1 / Gemma 2 的结果实际上**弱化了"English is universally critical"的普适性主张**。

**[B12] 自述 Limitations**

1. "Our analysis of emergent abilities was limited to multiple-choice downstream tasks and language-agnostic ICL"——只测了 MCQ，没有 free-form generation 或 factual recall（单独的 closed-book QA）。
2. "experiments were limited to including English in combination with the target language. Experimenting with other high-resource languages could provide additional insights"——没有测非英语 pivot 语言。

---

### C. 我们能接上什么

**核心差异**：本文的"扰动"是 language distribution shift（换语言 CPT），我们的"扰动"是 structural damage（层剪枝）。

**接口 4（同向支持，增量是量化）**：  
本文发现"PPL 不反映能力退化"（B7），我们 keep14 PPL 1.428× vs MMLU 19.5% 是**同向支持**：PPL 损失 42.8%，MMLU 只恢复 19.5%。差距更大。  
但我们的增量是：**可以量化「恢复所需 token budget」**。本文只说"until later in training（~3k steps）"，我们有 200k steps 的轨迹，可以给出"MMLU 从 chance 到 50% 恢复需要多少 steps"这样具体的数字。  
关键：如果我们能在轨迹上找到"PPL 恢复点"（如 PPL 回到 1.05×）和"MMLU 恢复点"（如 MMLU 回到 baseline 50%），两者之间的 step 差距就是"延迟"的量化——这比本文的定性描述更强。

**接口 5（机制对应）**：  
本文用 "big shift in model parameters"（L2 distance，Figure 4）来解释 ICL 崩溃。我们 keep14 heal 的过程本质上也是一个 parameter shift（从剪枝后的 keep14 权重向健康 OLMo-2 分布靠近），可以测同样的 L2 distance from initial（但我们的"initial"是剪枝+heal 前的 OLMo-2 base），看 heal 轨迹上的 L2 distance vs ICL 能力恢复是否也有类似时序关系。

**接口 6（对比机制）**：  
本文用"critical period"解释（前 1k steps 是关键），我们的深度剪枝版本是否也有 critical period？从我们的轨迹来看，keep14 的 MMLU 提升几乎在**前 20k–40k steps 最快**，之后减速——是否也是一个"critical period"结构，可以用 CMR 的框架描述？

**接口 7（潜在矛盾）**：  
本文 B10 问题：10k steps 后 CPT (eu) 模型 Copain 仍"far from recovering"。如果我们 keep8（更大结构损伤）在 200k steps 后 MMLU 始终 chance，这是否证明**某些损伤超过了 "critical threshold"，在有限 token budget 内无法恢复**？本文没有做这个实验，我们的 keep8 vs keep14 vs keep16 ladder 可以直接验证这个假设。

---

---

## 论文三：arXiv:2407.17467（自述 EMNLP 2024 main，未经 S2/DBLP 独立核实）

**全称**：CMR Scaling Law: Predicting Critical Mixture Ratios for Continual Pre-training of Language Models  
**作者**：Jiawei Gu, Zacc Yang, Chuanghao Ding, Rui Zhao, Fei Tan（SenseTime Research + Sun Yat-sen University + Nanjing University）  
**Venue**：自述 EMNLP 2024 main，未经 S2/DBLP 独立核实。

---

### A. 究竟做了什么（精确到实验设置）

**模型**：
- 4 个 Llama 架构模型（同架构，不同规模）：460M、940M、1.6B、3.1B，从头训练 100k steps on 200B tokens。
- **注意：最大模型只有 3.1B**，远小于当代 7B+ 模型，且全部是 SenseTime 内部训练的模型（非公开 checkpoint）。

**数据**：
- 通用：中文+英文+代码（220B tokens，44%:36%:20%）
- 领域：Finance（≥20B tokens）和 Academic Papers（arxiv，≥20B tokens）
- CPT：每个模型跑 10k steps（20B tokens），mixture ratio R ∈ {1/8, 1/4, 1/3, 1/2} 以及一些额外点。

**CMR 公式（精确）**：

**核心公式（Eq. 6，按 T 的幂律）**：
```
L_Δdom(T) = α₁ · T^s₁ + β₁
L_Δgen(T) = α₂ · T^s₂ + α₃ · T^s₃ + β₂
```
其中 Δ 是相对训练前的 loss 变化量，T 是训练 token 数（以 200M 为单位，T=100 = 20B tokens）。domain loss 用单项幂律，general loss 用双项（因为先涨后降的 U 形）。

**CMR 公式（Eq. 8）**：
```
R_CMR = α₄ · T^s₄ + β₃
```
即 CMR 本身也是关于 T 的幂律（T 是最大可用 token 量）。预测结果：
- M_460M：29.8%，M_940M：34.9%，M_1.6B：41.4%，M_3.1B：47.8%（均对 Finance，T_max=100=20B tokens）
- Academic Papers（M_460M）：36.7%（domain 与 general 分布更近，故 CMR 更高）

**因变量是 loss 还是 accuracy**：**全部是 validation loss（next-token prediction）**，不是下游 accuracy。Section 3.2 明确说：
> "we use the pre-training loss on the validation set to measure the model's capability"

Limitations（Section 8）也承认：
> "this study did not directly evaluate performance on downstream tasks"

**有无 held-out 验证还是 post-hoc 拟合**：
- Section 5.1 和 5.3：Figure 3 中用"solid circles（实测值）"和"stars（预测值）"对比。预测方法是：拟合部分 R 点（non-endpoint），**外推**到新的 R 值，声称误差在 0.05% 以内（Table 2）。
- **注意**：这是对同一个训练集（同 20B token 的 CPT 训练曲线）做拟合后预测其他 R 点，而不是在**新的、不同规模的 T_max 上做前向预测**再与实测比较。Section 5.3（Generalization）把 CMR scaling law 外推到 T=250（=50B tokens），但这是**理论外推**，论文中没有报告在 T=250 时的实测值来验证外推准确性。

---

### B. 方法论缺陷清单

**[B13] 因变量全是 loss，没有 downstream accuracy**（严重程度：高）

如 Section 8（Limitations）自述：
> "this study did not directly evaluate performance on downstream tasks. Including downstream task performance could provide a more intuitive understanding of the observed trends."

这与 2506.00288 的核心发现（PPL ≠ downstream ability）形成直接矛盾：CMR 预测的是"loss 维持在 ε 公差内的最高 domain ratio"，但 2506.00288 证明即使 validation loss 相同，下游能力可以天差地别。换句话说，**CMR 优化的目标函数（loss）并不是我们真正关心的目标（downstream capability）**，而 CMR 论文自己没有验证这两者是否等价。

**[B14] 模型规模太小（最大 3.1B）且非公开**（严重程度：高）

Section 8 承认：
> "the largest model in our experiments is still relatively small among contemporary LLMs. It may lead to inaccuracy in estimation of model size scaling."

且所有模型都是 SenseTime 内部训练的，无法独立复现。更重要的是：3.1B 参数量的模型在现代 LLM 研究中属于"小模型"范畴，其 scaling behavior 是否外推到 7B/13B/70B 是未知的。CMR scaling law（Eq. 8）的 R vs T 曲线只有 4 个数据点（4 个模型规模），外推到大模型需要多少置信度没有讨论。

**[B15] "Post-hoc 拟合"的泛化性未严格验证**（严重程度：高）

**关键问题**：论文声称"CMR can be predicted"，但其验证方式是：
1. 用 {1/8, 1/4, 1/3} 的 R 值拟合公式参数
2. 预测 {1/2, 3/4} 的 R 值对应的 loss
3. 误差 <0.05%

这是**同一 T_max（20B tokens）下的内插/外插**，不是"用 small-T 数据预测 large-T 时的 CMR"。Section 5.3 的"generalization"实验换了 domain（Academic Papers）但用的仍然是同样的 T_max=100，没有跨 T_max 的前向预测验证。"CMR scaling law"（Eq. 8）的实测验证点只有 T=100 对应的 4 个模型，没有提供 T=50/T=200 时的实测 CMR 与预测 CMR 的对比。

**[B16] ε=0.05 的公差设定是超参数，结论对它敏感**（严重程度：中）

Definition 1 和 2 都依赖 ε=0.05 这个公差（general loss 允许上涨不超过 0.05 的绝对 loss 值）。这个值的选定没有任何论证。如果 ε=0.02，CMR 会显著下降；如果 ε=0.10，CMR 会上升。CMR 的数值（29.8%、34.9% 等）对这个超参数是强依赖的，但论文没有做 sensitivity analysis。

**[B17] 自变量是 data mixture ratio，不是 structural damage**（严重程度：中，但这是设计范围问题而非错误）

CMR 的整个框架建立在"general data vs domain data 的比例"上，完全没有涉及"模型结构损伤"。换句话说，它的理论框架对我们的 Paper B 场景（层剪枝后 heal）不能直接套用，因为：
- 在 CMR 框架中，模型始终是完整的（只是训练数据比例不同）
- 在我们的场景中，模型有 structural damage，heal 的 token budget 不只取决于 mixture ratio，还取决于 damage depth（keep N 的 N 值）

因此 CMR 的 prediction formula 不能直接应用，但其**分析框架**（loss-budget tradeoff，power-law 拟合）可以迁移。

**[B18] Chinese/English 混合通用数据的语言混淆**（严重程度：低）

通用数据是 44% 中文 + 36% 英文 + 20% 代码。但 Finance 和 Academic Papers 的语言分布不详（Finance 可能主要是中文，Academic Papers 可能是英文+中文混合），这影响 domain distribution shift 的程度估计，进而影响 CMR 的域间比较（Finance CMR < Academic Papers CMR 的解释）。

**[B19] 自述 Limitations（Section 8）**

1. 最大模型只有 3.1B，规模外推不可靠。
2. 只做了 Finance 和 Academic Papers 两个 domain，结论泛化性有限。
3. CMR scaling law 不能跨 model size 预测（不能用小模型的实验直接预测大模型的 CMR）。
4. 没有直接评估下游任务性能。

---

### C. 我们能接上什么

**接口 8（框架迁移，因变量替换）**：  
CMR 框架的精髓是"存在一个 critical ratio，使得在 token budget T 内，一个目标（domain loss）下降而另一个目标（general loss）不越界"。  
我们可以做类比定义：把 "domain loss" 换成 "keep N 层的结构损伤强度"（或直接用 keep N 作为离散变量），把 "general loss 不越界" 换成 "PPL 恢复到 baseline 1.05× 以内" 或 "MMLU 恢复到 baseline 80% 以上"——然后问：**在给定 token budget T 内，哪个 keep N 是"critical threshold"（既能 heal PPL，又能 heal downstream）**？  
这不是 CMR 原文做的事（它研究 data ratio，不研究 structural damage），但框架类比是合理的，我们的数据（5 个 keep level × 200k steps）可以直接用来拟合类似的 power-law。

**接口 9（B13 的补充：loss 到 accuracy 的校准）**：  
本文只有 loss，2506.00288 证明 loss ≠ accuracy，我们有**完整的 loss + 5 eval accuracy 轨迹**。可以直接做"general loss 退化量 vs MMLU accuracy 退化量"的校准曲线，看两者是否 power-law 关联，以及是否存在一个 loss threshold 以上 accuracy 急跌（类似 2310.04680 的 30% sparsity threshold）。这比 CMR 论文更接近实际应用需求。

**[B16 的反驳]**：  
我们可以用不同的"recovery target"（如 PPL 恢复到 1.05× vs 1.1× vs 1.2×）来测试"恢复 budget"对 target 的敏感性，类比 CMR 的 ε 敏感性分析——这是 CMR 论文自己没做但应该做的。

---

---

## 共同检查项

**[C1] chat_template / base-vs-instruct 混用**

- 2310.04680：测 base 模型（OPT、LLaMA-1 base，非 instruction-tuned），无 chat template，greedy decoding。没有混用问题。
- 2506.00288：明确说 "We use Llama 2 7B as the base model"，且说"we do not change the vocabulary"，用 5-shot MCQ prompting，非 chat。也是 base 口径，无 chat template 问题。
- 2407.17467：内部模型从头训练，无 instruction tuning，纯 language modeling。无混用问题。
- **三篇全部是 base 口径**，与我们的 chat_template=False 设定一致，可比。

**[C2] 多重比较未校正**

- 2310.04680：8 个任务 × 6 个模型 × 多个 sparsity 点，无任何 Bonferroni 或 FDR 校正。Appendix K 仅报告 bfloat16 vs float16 的 systematic error（~1% accuracy diff）。
- 2506.00288：Section 4.1，"wins in 3 instances and the one without wins in the remaining 3"——作者自己用"wins"计数作为证据，没有显著性检验。Table 2 数字差异如 Llama 3.1 8B 的 Copain（42.04 vs 41.19）= 0.85 points，显然不显著，但被算入"CPT eu+en wins in all cases"的 Copain 列内。
- 2407.17467：Table 2 和 3 只报告 MSE 和 R²（fitting quality），不是 confidence interval on CMR prediction。

**[C3] Limitations 段自述汇总（已在各论文 B 节逐条列出）**

---

## D. 一句话判决（每篇）

**2310.04680（ICLR'24）**：它占掉了「静态 pruning 后 fact recall 比 ICL 先崩溃」的主张，但**heal/continued pretraining 这一整条时间轴完全未做**，且 pruning 方法是非结构化权重稀疏（与我们的深度剪枝+嫁接机制不同），我们能用 keep8→keep14→keep16 的 200k step 恢复轨迹 + 5 eval 分组（fact-intensive MMLU vs core6 MC）来直接复现其结论的"heal 版本"，证明或证伪「heal 过程中同样是 fact recall 最后恢复」。

**2506.00288（ACL'25 自述）**：它占掉了「PPL 不反映下游能力，能力涌现有 critical period」的定性主张，但**没有量化 critical period 的 token budget，也没有把延迟量化为可预测的数字**，且其"扰动"是语言 distribution shift，不是结构损伤；我们能用 keep14（结构损伤）的 PPL-vs-MMLU 时序数据做"structural damage 版的 critical period 量化"，把他们的定性描述变成可预测的 token budget 律。

**2407.17467（EMNLP'24 自述）**：它占掉了「CPT 中 data mixture ratio 的 power-law 预测框架（CMR 律）」，但**全部只有 loss，没有下游 accuracy，且模型最大 3.1B，且 post-hoc 验证而非前向预测**；我们能把 CMR 框架的变量从"data ratio"替换为"structural damage level（keep N）"，且我们的因变量有真实 downstream accuracy，直接补足其最大的 stated limitation。

---

## 最终总判决：「Recovery Budget 律」假设被占了几成？

**我们的假设**：「结构损伤后，各能力恢复所需的 token 预算各不相同，且该预算随继承深度分数呈规律变化」

**三篇占掉了什么**：
1. 「不同能力对 down-scaling 的敏感度不同（fact recall > ICL）」—— **2310.04680 占掉了静态版本**（prune 后即测），但 heal 版本完全空白。
2. 「PPL 恢复不等于 capability 恢复，有 critical period 延迟」—— **2506.00288 占掉了定性描述**，但只在语言适应场景，且没有量化。
3. 「CPT 过程的 loss-budget power-law 可以预测关键阈值」—— **2407.17467 占掉了 loss 层面的 power-law 框架**，但用的是 data ratio 而非 damage level，且没有 downstream accuracy。

**剩下的窄缝（我们独有）**：

1. **时间轴 × 结构损伤**：三篇没有一篇做"structural damage + heal 轨迹 × per-step eval × 多能力分组"。我们的 keep8/10/12/14/16/full32 × 200k step × 5 eval 是三篇论文联合未覆盖的正交空间。

2. **PPL-to-accuracy 解耦的量化版本**：2506.00288 有"PPL 相同但 accuracy 不同"的定性结论；CMR 只有 loss 轴；我们可以做"PPL 恢复率 → MMLU/core6 恢复率"的校准曲线，加上 matched-PPL 对照（keep14@step67500 vs 对照在同 PPL 点），精确量化两个轴的解耦量。这是三篇都没有的。

3. **Recovery budget 随 damage depth 的规律**：keep8 vs keep10 vs keep12 vs keep14 vs keep16 的"到达 MMLU = baseline 50%/80% 所需 step 数"可以拟合成关于 damage depth（keep N/32）的函数——这就是我们的"recovery budget 律"。三篇没有任何人做过这个（CMR 做了 budget-ratio power law，但没有 structural damage 维度）。

**被占程度评估**：约 **30–35%**（定性 motivation 被占，但核心实验设计完全未占）。窄缝精确位置是：**（深度剪枝 + heal）× 步级轨迹 × 多能力分解 × recovery-budget 量化**。这四个条件联合取交集，文献中尚无覆盖。

---

*文件生成于 2026-08-06，字数约 7200 中文字，所有引用均来自全文抓取内容（ar5iv 2310.04680 = 200KB，arxiv HTML 2506.00288 = 156KB，arxiv HTML 2407.17467 = 990KB），全文逐节阅读，所有引用已标注 Section。*
