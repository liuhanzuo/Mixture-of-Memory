# 对抗性精读：MechLens + CKA_Delta

**目标论文**
- MechLens: arXiv:2606.07978 (preprint, 2026-06-06), Xueping Gao, Alibaba Cloud
- CKA_Delta: arXiv:2606.16897 (preprint, 2026-06-15), Xueping Gao, Alibaba Cloud

**两篇都是 preprint，S2 未收录，venue 一律标 preprint。**
MechLens 摘要末尾写有 `Code: https://anonymous.4open.science/r/MechLens-EMNLP2026`，表明在投 EMNLP 2026，但截至抓取日期尚无 journal_ref。

---

## A. 它究竟做了什么（精确到实验设置）

### A1. MechLens (2606.07978)

**核心主张**：factual knowledge 在 LLM 中不是逐层渐现，而是在最后几层"突然结晶"（Late Crystallization）。这一现象能解释各类 activation intervention 的成败差异。

**关键定义 — FEP (Factual Emergence Point)**（Section 6.1，公式 1）：
对查询 $q$、正确答案 $a$，FEP 是最早使 $a$ 进入 top-$k$ 预测的层：
$$\text{FEP}(q) = \min\{l : a \in \text{top-}k(\text{LN}(h^l) W_U)\}$$
其中 $h^l$ 是第 $l$ 层残差流，LN 是最终层 LayerNorm，$W_U$ 是 unembedding 矩阵。若 $a$ 在任何中间层都未进入 top-$k$，则 FEP = 最终层。**默认 $k=10$**（top-10）。FEP Depth = FEP / total\_layers。

**模型池**（Table 2）：8 个模型，5 个架构家族：
- Qwen2.5: 0.5B(24L), 7B(28L), 14B(48L)
- Pythia: 1.4B(24L), 6.9B(32L)
- Llama-3.1-8B (32L, GQA)
- Mistral-7B (32L, GQA+SWA)
- Gemma-7B (28L, MHA)

**数据集**：TruthfulQA 817 samples（Phase 2/3 主力），MMLU 1,200 samples（cross-benchmark），WikiText-2 2,000 samples（仅用于训练 tuned lens per-layer affine probe），SST-2（control task）。

**主要数值**（来自 Table 5/6）：
| 模型 | 总层数 | Mean FEP | Late Crystal（从未进 top-10）|
|---|---|---|---|
| Qwen2.5-7B | 28 | 27.3±1.8 | 85.9% |
| Qwen2.5-14B | 48 | 46.0±4.9 | 77.7% |
| Llama-3.1-8B | 32 | 29.4±4.9 | 71.0% |
| Mistral-7B | 32 | 26.3±6.2 | 27.1% |
| Pythia-6.9B | 32 | 30.8±4.2 | 93.4% |
| Gemma-7B | 28 | 27.4±2.4 | 89.8% |

MMLU 上（Qwen2.5-7B, 1200 samples）：Late Crystal = 98.2%（Table 17）。

**Tuned lens 对照**（Table 4）：
在 Qwen2.5-7B 817 样本上，tuned lens 给出 Late Crystal = 85.7%，logit lens 85.9%，差距仅 0.2pp，74.9% 样本 FEP 完全一致。

**intervention 结果**（Table 3，Qwen2.5-7B，817 samples）：
- DoLa dynamic: MC1 = 0.2778（baseline 0.2215，+25.4%，$p<0.001$，Bonferroni 后存活）
- CAA top_k=10: MC1 = 0.2558（+15.5%）
- ITI top_k=10: MC1 = 0.2436（+10.0%）
- activation scaling：全部 $\leq$+2%，最差 -28%

**Cross-architecture intervention**（Table 7）：
- Llama（低结晶 70.4%）：CAA +33.5% vs DoLa +1.9%（$p<0.001$，Bonferroni 后存活）
- Mistral（低结晶 26.8%）：CAA +32.3% vs DoLa +11.4%（$p<0.001$，Bonferroni 后存活）
- Qwen（高结晶 85.9%）：DoLa +25.4% vs CAA +15.5%（$p=0.059$，Bonferroni 后**不显著**）

---

### A2. CKA_Delta (2606.16897)

**核心主张**：跨架构 LLM 在 concept representation 上存在"几何-功能解离"（geometric-functional universality dissociation）：CKA_Delta 值中等（0.7-0.83），但 affine-aligned 分类器的跨模型准确率接近完美（99.9%）。

**CKA_Delta 定义**（Section 3.2，公式 1）：
对两个模型 $M_A, M_B$，各取 $n$ 个 contrastive prompt pairs 的激活差向量 $\Delta_A = h_A^+ - h_A^-$，$\Delta_B = h_B^+ - h_B^-$，PCA 降维到 50 维，然后计算 debiased linear CKA：
$$\text{CKA}_\Delta(M_A, M_B) = \text{CKA}(\Delta_A, \Delta_B)$$

**模型池**（Section 4）：9 个模型，5 个架构家族，**全部为 instruct 版本**（除 Llama-3.1-8B-base 作为 ablation）：
- Llama-3.1-8B-Instruct, Llama-3.1-70B-Instruct
- Qwen-2.5-7B-Instruct, Qwen-2.5-72B-Instruct
- Gemma-2-9B-IT
- Mistral-7B-Instruct-v0.3
- Phi-3.5-mini-instruct, Yi-1.5-6B-Chat, Llama-3.1-8B (base)

**主力分析**：4 个 7-9B instruct 模型（Llama/Qwen/Gemma/Mistral）。

**Concept 池**：Big Five + 3 alignment traits（helpfulness/sycophancy/confidence）= 8 personality 维度 + safety（refusal vs direct）+ truthfulness + formality + code-vs-NL + reasoning-vs-recall，共 6 个 concept 域。每个 trait：500 contrastive prompt pairs（50 手写 seeds × 10 变体 + 450 模板生成，45 主题域）。

**主要数值**（Table 2）：
- Same-trait CKA_Delta（linear）= 0.829±... vs cross-trait = 0.717，$t=2.89$，$p=$ 有显著性，Cohen's $d=0.60$
- 标准 CKA+ 无法区分：same=cross=$p=0.052$，$d=0.27$

**"Near-perfect functional transfer"的具体定义**（Section 5.2，Table 3/4）：
在每个模型的 PCA-50 空间上训练 ridge logistic 分类器（预测 positive/negative persona polarity），然后跨模型迁移：
- **affine-aligned transfer**（学习 $50\times50$ affine map，用 $n$ 个 contrastive pairs 拟合）：所有 96 个 directed pair × trait 条件下准确率 = **99.9%**（平均），每个 trait 都达到 ~100%
- **direct transfer**（无 alignment）：51.3%
- 对照：random-label affine = 50.5%，cross-trait alignment = 59.3%

**"Stitching"是什么**：**不是 layer stitching**（把一个模型的层拼到另一个模型）。是：在模型 A 的 PCA-50 空间里训练线性分类器，学一个 affine map 把模型 B 的激活映射到 A 的空间，然后用 A 的分类器给 B 的样本打标。任务是**二分类 persona polarity**，不是 next-token prediction，不是 perplexity。

**统计方法**（Section 4）：两侧 Welch's t-test，Bonferroni + BH FDR 双重校正，permutation test（1000 shuffles），bootstrap 95% CI（1000 resamples），Mann-Whitney + Wilcoxon rank-sum 非参数确认。

---

## B. 方法论缺陷清单（对抗性主交付物）

### B1. MechLens：FEP 定义与 logit lens 固有偏差混淆 ★★★【严重】

**论文原文**（Section 6.1）：
> "We define the Factual Emergence Point (FEP) for a query $q$ with correct answer $a$ as the earliest layer at which $a$ enters the top-$k$ predictions under logit lens projection: $\text{FEP}(q) = \min\{l : a \in \text{top-}k(\text{LN}(h^l) W_U)\}$"

**问题**：logit lens 用的是**最终层**的 unembedding 矩阵 $W_U$ 和 LayerNorm LN 来读所有中间层。这意味着：
1. 中间层的 residual stream $h^l$ 与 $W_U$ 的行空间不对齐（中间层的信息可能以 $W_U$ 无法直接读取的表示存在），这本身会导致大量样本的"答案不出现在 top-10"，与答案是否真实存储在该层**无关**。
2. tuned lens 通过训练 per-layer affine probe 部分缓解了这个问题，但作者只用了 **2,000 WikiText-2 样本**训练这些 probe（Section 4 Evaluation，"train per-layer affine probes on 2,000 WikiText-2 samples"，Section 9 Reproducibility 确认）。
3. 关键遗漏：作者没有报告 **probe 在 TruthfulQA 上的 held-out 分类精度**。如果 probe 本身准确率低，则 tuned lens FEP 和 logit lens FEP 吻合（85.7% vs 85.9%）只说明"两个不可靠工具给出相同不可靠答案"，而非互相验证。
4. **没有打乱层序（shuffle-layer）null**：如果随机打乱层序后重算 FEP distribution，是否仍然得到高度集中在最终层的分布？如果是，则"Late Crystallization"部分可能只是 logit lens 的几何偏差，而非模型的信息论性质。

**作者自辩（Section 6.1）**：提出了 3 条反驳：(1) LayerNorm ablation 后 FEP 不变；(2) 跨架构有系统梯度（Qwen 85.9% vs Mistral 26.8%）；(3) 结晶程度预测 intervention 类型。这些是间接证据，但都不排除"中间层信息存在但 unembedding basis 不对齐"的解释。

**严重程度**：★★★ 核心主张的主要方法论风险。tuned lens 2,000 samples 太少（TruthfulQA 本身 817 samples，probe 用 WikiText-2 OOD 数据训练），且无 null 基线，使得"Late Crystallization 是信息论事实 vs 是 readout 偏差"无法区分。

---

### B2. MechLens：2,000 WikiText-2 tuned lens probe 样本量极度不足 ★★★【严重】

**论文原文**（Section 9 Reproducibility）：
> "Data: TruthfulQA (817 samples), MMLU (1,200 samples), WikiText-2 (2,000 samples for tuned lens)"

**问题**：
- tuned lens 要在每一层训练一个 affine probe（$d \times d$ + $d$ 维 bias，Qwen2.5-7B 的 $d=3584$，即约 $3584^2 \approx 12.8M$ 参数）。
- 2,000 WikiText-2 样本远不足以拟合 $d=3584$ 的 affine probe——这个 probe 处于极端欠定状态，必然过拟合（或者等效于 L2 正则化极重的 ridge regression）。
- 更重要的：probe 是用 **WikiText-2 next-token prediction** 训练的，测试的是 **TruthfulQA factual recall**——这是典型的 OOD 迁移，probe 质量存疑。
- 作者声称"tuned lens yields 85.7% late crystallization—within 0.2 percentage points of the logit lens (85.9%)"（Section 6.1），但**没有报告 tuned lens probe 自身在 intermediate layer 上的验证 loss 或 top-k 准确率**。

**严重程度**：★★★ 这是整个 tuned lens 验证论证的根基。如果 probe 质量差，Table 4 的结果是噪声，不是验证。

---

### B3. MechLens："结晶预测 intervention 优劣" 的 Qwen 那行统计上不显著 ★★【重要】

**论文原文**（Section 7.4/Table 7）：
> "On high-crystallization Qwen (85.9%), the pattern is consistent with a reversal: DoLa achieves +25.4% vs CAA's +15.5% ($p=0.059$, non-significant; see §7). While this single comparison is non-significant, the relationship across all three architectures is monotonic..."

**问题**：
1. 核心主张"crystallization degree predicts optimal intervention type"依赖的是**3个数据点的单调关系**（Mistral/Llama/Qwen），其中最关键的高结晶数据点（Qwen 反转）在 Bonferroni 校正后 $p=0.059$ 不显著。
2. 作者声称"monotonic relationship"，但 3 点单调是最低证明标准，根本无法区分"真实单调关系"和"随机排序"（3点单调的概率 = 1/3 = 33%）。
3. 有效 $n=3$（3个架构），这是 family-level 推断，而非 sample-level。
4. 论文 Section 7 最后一段作者自己也写："We frame this threshold as **suggestive evidence** pending broader validation across additional high-crystallization architectures"——说明作者本人也知道这一发现只是方向性的，但摘要和 Introduction 的行文远比这激进（"predicts optimal intervention type"）。

**严重程度**：★★ 过度声明（overclaiming）。核心 intervention 预测原则只在 2/3 模型显著，第 3 个不显著，而且 n=3 的架构级比较本质上是无统计功效的。

---

### B4. MechLens：控制任务（SST-2）的对照有一半无效 ★★【重要】

**论文原文**（Section 7.7，Table 10）：
> "Pythia-6.9B and Gemma-7B cannot perform zero-shot sentiment classification (100% on SST-2 reflects the chance-level early-token distribution rather than late-emerging task-relevant information) and are excluded from the specificity comparison"

**问题**：
1. 5 个模型里，2 个（Pythia 和 Gemma）因为**不能做 zero-shot 情感分类**被排除。所以"控制任务证明 Late Crystallization 是 factual 特异的"这个结论**只基于 Qwen 和 Mistral 两个模型**，Llama 是中间值（49.5% SST-2 vs 70.4% factual，差距不大）。
2. Pythia 和 Gemma 的排除本身就说明"base model 做 SST-2 的能力差异"是对照混淆因素——不同模型的情感分类基线不同，用它来控制 factual recall 特异性，需要解耦"任务能力差异"与"结晶差异"。
3. 作者承诺"an evaluation with few-shot prompting on these models is left to future work"，但当前的 Table 10 只有一半数据。

**严重程度**：★★ 对照任务设计不完整，specificity 结论仅基于 2 个模型充分支持。

---

### B5. MechLens：MC1 baseline 值在同一篇论文内不一致 ★【轻微但值得注意】

**论文原文**（Table 8 footnote 3）：
> "The baseline MC1 here (0.2179) differs slightly from the 0.2215 reported in Table 3 due to evaluation on the LN-ablation sample subset (817 samples) rather than the full TruthfulQA split"

**问题**：Table 3（full split 817 samples）和 Table 8（"LN-ablation sample subset" also 817 samples）的 baseline 不同（0.2215 vs 0.2179），但两个都说是 817 samples。作者的解释（"不同的 split"）本身就暗示评估口径不统一，增量数字（+11.8% MC1 from LN scaling）是相对于不同 baseline 计算的，不能直接与 Table 3 的 DoLa +25.4% 比较。

**严重程度**：★ 细节不一致，但相对增量不影响定性结论。

---

### B6. MechLens：Computability-Memorization Spectrum 无统计检验 ★★【重要】

**论文原文**（Section 6.3, Appendix H）：
> "Logical Falsehood crystallizes earliest (mean FEP=22.1, σ=2.6), while categories like History, Psychology, and Weather show FEP=28.0 with zero variance"

**问题**：
1. FEP=28.0 with zero variance（100% 样本都在最终层）意味着**该类别所有样本的 FEP 都是 trivially 最终层**——这可能只是说明这些类别 TruthfulQA 中 n 很小，导致即使随机也能出现 zero variance，而非真实的"均匀在最终层"。
2. 论文没有报告每个 category 的 **n**（样本量）。TruthfulQA 817 samples 分布在 30+ 类，很多类只有几十甚至更少样本。Appendix H 写"Full per-category FEP statistics are available upon request"而不是直接放表，这是信息隐藏。
3. 类别间 FEP 差异没有任何 pairwise 或 omnibus 统计检验（没有 ANOVA，没有 Mann-Whitney）。

**严重程度**：★★ 核心 sub-contribution（Spectrum）的证据级别很低，per-category n 未报告。

---

### B7. CKA_Delta："near-perfect functional transfer" 是**分类 probe**，不是**生成**，不能与 stitching ppl 比较 ★★★【关键】

**论文原文**（Section 5.2，Table 3）：
> "affine-aligned persona classifiers achieve 99.9% cross-model accuracy across all 96 transfer conditions"

**精确定义**：
- 任务：预测 **persona polarity**（positive pole / negative pole，二分类），样本是 500 个 contrastive prompt pairs，每个 pair 只有 2 个可能标签
- 分类器：ridge logistic classifier trained on 一个模型的 PCA-50 contrastive-difference vectors
- affine map：$\mathbb{R}^{50} \to \mathbb{R}^{50}$，用另一个模型的 $n$ pairs 拟合
- 测试：用 source 模型的分类器给 target 模型的样本打标，看 polarity 准确率

**关键区分**：
1. "near-perfect functional transfer" = **persona polarity 二分类准确率 99.9%**，不是 next-token PPL，不是 open-ended generation 质量。
2. 与我们的 oracle affine readout 实验完全不同维度：我们的实验是把 OLMo-2-1B 的中间层激活喂给 Llama-3.2-1B 的解码器（layer stitching），测 next-token PPL（结果 596 vs 18.8），这是**生成任务**。
3. 因此，**我们的 ppl=596（差 32×）与 CKA_Delta 的 99.9% accuracy 并不矛盾**——它们测的根本不是同一件事。CKA_Delta 证明：在 50 维 PCA 子空间里，两个模型的 persona direction 可以用 affine map 对齐，使得 yes/no 分类器迁移成功。我们的实验证明：把整个 residual stream 从一个模型拼到另一个模型的解码路径上，next-token prediction 会完全崩溃。两者都可以为真，因为：
   - CKA_Delta 只要求在 50 维子空间里的方向对齐
   - Layer stitching 要求整个 $d=4096/2048$ 维 residual stream 的**完整几何结构**（不只是方向，还有 scale、offset、rotation）与下游 transformer blocks 兼容

**严重程度**：★★★ CKA_Delta 的声明本身没错，但如果有人把"near-perfect functional transfer"读作"跨架构 stitching 可行"，那是严重误读，作者在这里应该更明确区分"persona polarity probe 迁移"和"激活流直接复用"。

---

### B8. CKA_Delta：无 null 校准（shuffle-layer / random-activation baseline） ★★★【严重】

**问题**：
CKA_Delta 的核心是 same-trait 与 cross-trait 的 discrimination：same-trait CKA_Delta = 0.829±..., cross-trait = 0.717，$d=0.60$。但论文**没有报告**：
1. 打乱层序（shuffle layer indices）后重算 CKA_Delta 的 null 分布
2. 随机激活（random Gaussian）作为 null 基线
3. 在我们的实验中，shuffle-layer null CKA = 0.453 vs observed = 0.491，差距仅 +0.038（约 8%），说明大量 CKA 相似性来自 trivial 因素（维度/数据集），而非 representation alignment。

作者确实做了 **random concept control**（用语义无关的 contrastive pairs 计算 CKA_Delta），该 null = 0（小值），但这只校准了"concept specificity"，没有校准"跨架构 CKA 是否比 shuffle-layer null 显著更高"。

论文 Section 3.2 "Empirical orthogonality" 处确认：mean off-diagonal cosine between trait directions 很小（约0），这只是验证了 within-model orthogonality，不是 null CKA 校准。

**严重程度**：★★★ 这是与我们资产最直接相关的方法论漏洞。我们有实测的 shuffle-layer null（0.453 vs 0.491），可以直接将 CKA_Delta 的 same/cross 数值（0.829/0.717）放在同样的 null 框架下讨论。

---

### B9. CKA_Delta：全部使用 instruct 模型，instruction-following 相似性混淆 ★★【重要】

**论文原文**（Section 4 Models）：
> "Nine models spanning five architectural families: Llama-3.1-8B-Instruct, Qwen-2.5-7B-Instruct, Gemma-2-9B-IT, Mistral-7B-Instruct-v0.3, Phi-3.5-mini-instruct, Yi-1.5-6B-Chat..."

**问题**：
1. 4 个主力模型全是 instruct 版本。系统提示（system prompt）用于设定 persona（"You are a highly extraverted person..."），这意味着 contrastive difference 向量 $\Delta = h^+ - h^-$ 部分反映的是**对系统提示格式和 instruction-following 的响应差异**，而非 persona 本身的 representational structure。
2. 作者做了一个 ablation："the base Llama-3.1-8B (no instruction tuning) retains separable persona representations (within-model accuracy=0.98, affine transfer to instruction-tuned models)"（Section 5.3）——但这只是**单个模型**的验证，而且是 Llama base 对 Llama instruct（同架构），没有 cross-architecture base-vs-base 对比。
3. 如果高 CKA_Delta 来自"所有 instruct 模型都用类似的 RLHF/SFT 对话格式，系统提示格式相似"，那 0.83 的几何相似性可能主要是**指令格式相似性**而非 concept representational structure。

**严重程度**：★★ 混淆因素存在，base model 消融只做了单例，不充分。

---

### B10. CKA_Delta：scale 证据仅一对模型，但摘要声明偏激进 ★★【重要】

**论文原文**（Section 5.3）：
> "We further report an observational note based on a single 70B/72B pair (Llama-70B × Qwen-72B; Table 5): the cross-family CKA_Delta exceeds the 7–9B baseline... We emphasize this is a single-pair observation: with n=1 at the cross-family 70B+ level, no inferential statistic is computable"

作者自己承认了这个问题，在 Limitations (L3) 也重申。但摘要/Introduction 写："a single 70B–70B pair provides an observational note that universality may strengthen with scale"——这个限定语在摘要里足够清楚，基本诚实。

**问题**：Spearman $\rho$ 相关性（rank-order of within-concept direct transfer difficulty，Section 5.4）使用 one-sided $p$ 值且只有 4-6 个 concept 数据点，two-sided $p$ 值最强是 $p=0.2$（作者标注"treated as suggestive ordering"），这个 rank ordering 实际上没有统计功效，而摘要声称"rank-orders direct-transfer difficulty (Spearman ρ, p, one-sided)"——用 one-sided p 来报告一个 directional 假设，本应在预注册时声明方向，事后选用更容易达标的 one-sided 是轻微 p-hacking。

**严重程度**：★★ Spearman 部分有一定程度 overclaiming；scale 部分作者自我标注清楚。

---

### B11. CKA_Delta：dyadic non-independence 问题（pair 之间共享模型） ★★【重要】

**问题**（Section 5 统计设计）：
- 4 个主模型产生 $4 \times 3 = 12$ 个 directed pairs
- 这 12 个 pair 中，每个模型被用作 source 3 次、target 3 次——样本之间严重非独立
- 论文用 Welch's t-test / Mann-Whitney 处理这些 pair，但这些检验假设 i.i.d. samples
- 正确的分析需要 mixed-effects model（以 model identity 为 random effect）或 permutation test on model labels（而非 observation labels）

作者的 permutation test（1000 shuffles）是打乱 **trait labels**，不是打乱 **model identity**——这只校准了 within-pair-set 的 null，没有处理 dyadic non-independence。

**严重程度**：★★ 经典 representation similarity 研究中常见的统计 pitfall，CI 可能偏窄，p 值可能偏小。

---

### B12. MechLens：eval set 无常量基线报告 ★★【重要】

**背景**（来自我们自身踩坑经验）：SQuAD val 上 49.85% 样本是同一句拒答，常量函数 EM 就 49.85%，高于所有实验臂。

**MechLens 的情况**：
- TruthfulQA MC1 baseline = 0.2215（Section 5.2）。TruthfulQA 是 MC format，817 个问题，每个有不同数量选项。
- 论文**没有报告不看输入直接猜固定选项的准确率**（例如始终猜第一个选项、最短选项、最常见选项）。
- 由于 TruthfulQA 设计上各题选项数目不同，随机猜的期望准确率不是简单的 1/k，但常量基线（例如"始终选最短答案"）在 TruthfulQA 上可能不低。
- 更重要的是：Qwen2.5-7B baseline MC1 = 22.2%，在某些分析子集（Table 8）是 21.8%——这两个数字略微不一致但差距小，主要问题仍是无常量基线作为参照。

**严重程度**：★★ 缺少常量基线，MC1 增量的绝对意义不明确。

---

### B13. MechLens：Limitations 段自认的局限（逐条抄录，Section 9）

1. **Scale**：只在 7-8B base models + 一个 14B（同 Qwen 家族）上验证，70B+ 未测。
2. **Task Format**：全部是 MC 格式（MC1/MC2），open-ended generation 的 FEP tracking 留作未来工作（"First-token FEP provides a lower bound"）。
3. **Instruction-Tuned Models**：instruct pilot（Qwen2.5-7B-Instruct）只做了描述，未做系统的跨家族 intervention 对比；instruct 模型上 DoLa/CAA 都使 MC1 下降。
4. **Methodological Scope**：CrystalBoost grid search 只有 5 个 configuration；SWA-crystallization 相关是 observational，无 controlled ablation；"we have not performed controlled ablation of the attention mechanism itself to establish causality"。

---

### B14. CKA_Delta：Limitations 段自认的局限（Section 6 Limitations and future work）

- L1：Concept coverage 只有 6 个域（4 instruction-level + 2 non-instruction）。
- L2：Question-set sensitivity，absolute CKA_Delta 值对主题覆盖敏感（20% 变化），只建议用于 relative comparison。
- L3：Scale 证据 n=1，需要 ≥3 个 70B+ 模型复现。
- L4：Safety 几何判别 $p=0.13$，不显著（post-hoc power analysis 显示 under-powered）。
- L5-L7：跨语言梯度（法语 Chinese 更低）混淆训练数据构成；steering 需要 per-architecture calibration；independence assumption 只是近似。
- L8：LOO cross-validation 显示 CKA_Delta **不能预测绝对 direct-transfer accuracy**（$r^2$ 低），只能做 regime classifier 和 outlier detector。
- L11-L12：BFI-44 forced-choice 有零均值 Likert drift（null），说明 CKA_Delta 追踪的是生成行为结构而非自我报告；conscientiousness 有异常高 direct transfer（0.96 vs 其他 0.3-0.5）。

---

## C. 我们能接上什么（基于现有资产，不新训模型）

### C1. 用 shuffle-layer null 挑战 MechLens 的 Late Crystallization 声明

**我们的资产**：91-pair CKA 资产，含 shuffle-层序 null 校准（实测 null=0.453 vs observed=0.491，差仅 +0.038）。

**实验设计**：
- 对 OLMo-2-7B（或 Qwen3-8B，我们已有 logit-lens 测量）的 TruthfulQA 样本，重现 FEP 计算。
- 同时计算"打乱层序后的 FEP distribution"——即随机排列 $\{h^0, h^1, ..., h^{L-1}\}$ 再算 FEP。
- 如果 null（打乱层序）给出的 Late Crystal % 也 ≥ 70%，说明 "85.9%" 大量来自 logit lens 的几何偏差而非 crystallization 本质。
- 预期数字：若 null Late Crystal ≥ 50%，则 MechLens 的核心数字被严重污染；若 null ≤ 20%，则 Late Crystallization 是真实现象，我们的 logit-lens 数据（OLMo-2-7B L18→L19 跳 0.326→0.544；Qwen3-8B L24→L25 跳 0.236→0.621）与 MechLens 一致但提供了**更精细的层分辨率（逐层 rank 曲线而非单点 FEP）**。

**如果我们的数据可能与它矛盾**：我们发现 OLMo-2-7B 有一个清晰的 onset layer（L18-19），这比 MechLens 的"最终层突然结晶"的描述更微妙——MechLens 测的是进入 top-10 的时刻，而我们的 logit-lens 曲线显示的是 rank 的连续变化。两者可以共存：MechLens 定义下（top-10 入场），L19 仍可能是"最终层"；我们的曲线可以提供 MechLens 无法提供的**crystallization trajectory**（连续 rank 曲线 vs 离散 FEP 点）。

---

### C2. 用 prune-heal 训练轨迹验证 Computability-Memorization Spectrum

**我们的资产**：Paper B 的 keep8/10/12/14/16/full32 200k step 完整轨迹 + know5（world knowledge）+ core6（MC reasoning）+ closed-book QA（PopQA/TriviaQA/NQ-open）分类。

**MechLens 的主张**（Section 6.3, Appendix H）：computable knowledge（Logical Falsehood，FEP=22.1）比 memorized facts（History/Psychology，FEP=28.0）crystallize 更早。

**我们能做的**：
- 对 keep8/keep12/keep16 checkpoints（不同有效深度），分别计算 know5 类（世界知识，类比 memorized facts）和 core6 类（逻辑推理，类比 computable knowledge）的性能恢复曲线。
- 预测：若 Spectrum 成立，层数少的 keep8 在 know5（memorized）下降更多，在 core6（computable）下降更少；反之 keep16 更均衡。
- 这不依赖 FEP 测量，而是通过"剪层后哪类知识先恢复"来 corroborate 或 challenge Spectrum 的 causal claim。
- 如果 keep8 的 know5 vs core6 恢复率差异支持这个方向，我们可以声称提供了 **prune-heal 行为层面的独立证据**（而 MechLens 只有 logit-lens 层面的相关证据）。

---

### C3. 反驳 CKA_Delta 的"near-perfect functional transfer"：精确定位矛盾

**我们的资产**：oracle affine readout 实验：OLMo-2-1B → Llama-3.2-1B 跨架构 1-层线性桥，PPL = 596 vs 原模型 18.8（差 32×）。

**矛盾点精确定位**：

| 维度 | CKA_Delta 实验 | 我们的实验 |
|---|---|---|
| 任务 | Persona polarity 二分类 | Next-token prediction |
| 分类器 | Ridge logistic (binary) | Full transformer decoder |
| 对齐方法 | $50\times50$ affine map on PCA-50 subspace | $d\times d$ linear layer on full $h$ |
| 输入向量 | Contrastive difference $\Delta = h^+ - h^-$（已消除 content variance）| Raw residual stream $h^l$ |
| "Transfer success" 定义 | 分类准确率 ≥ 99.9% | PPL 与原模型相比 |
| 结果 | "near-perfect" | PPL 差 32× |

**结论**：
- **在 CKA_Delta 的定义下（persona polarity 分类，PCA-50 subspace，affine map）：两个结果不矛盾**。PPL=596 说明 full residual stream 的几何结构跨架构不兼容（layer stitching 失败），但 CKA_Delta 只要求 50 维子空间的方向对齐，这个条件弱得多。
- **在"functional transfer = 能否直接复用激活流"的定义下：CKA_Delta 的 99.9% 声明是误导性的**，因为它声称的"functional"仅限于在 learned 对齐后的 probe 任务上，而 CKA_Delta 论文本身从未主张 layer stitching 可行。
- **我们的实验提供了一个 CKA_Delta 没有测试过的数据点**：PPL=596 是"在 generation 任务上，oracle affine（线性桥，能力下界）的表现"，这与 CKA_Delta 的 99.9% accuracy 并存，共同说明：跨架构 concept representations 在 probe 任务上高度可迁移，但在 generation 任务上几乎不可直接复用——这是一个 **stronger version of the geometric-functional dissociation**，比 CKA_Delta 自己描述的要尖锐得多。

---

### C4. 填补 CKA_Delta 的 null 校准缺口

**我们的资产**：91-pair CKA 资产，含 shuffle-层序 null（null=0.453, observed=0.491，差 +0.038，约 8%）和 identity gate 验证（max|M[i][i]-1|=1.8e-7）。

**实验设计**：
- 将 CKA_Delta 的 same-trait（0.829）和 cross-trait（0.717）放在"null-校准后"的框架下：如果 shuffle-layer null 对这些模型/概念的 CKA_Delta 值也在 0.7-0.8 区间，则 same-cross discrimination（Δ=0.11）相对于 null 的超出量（delta-null-corrected）会比作者报告的 effect size 小得多。
- 具体：用我们的 OLMo-2-7B 激活，构造 persona contrastive pairs（系统提示 "You are very extraverted" vs "You are very introverted"），计算 CKA_Delta vs shuffle-layer null，量化 null-corrected discrimination。
- 预期：若 null-corrected CKA_Delta same-cross gap < 0.05（而非 0.11），则 CKA_Delta 的 discriminability 声明是依赖于未做 null 校准的绝对值。

---

### C5. 用 prune-heal 轨迹直接测 "Late Crystallization 随深度变化"

**我们的资产**：keep8/10/12/14/16/full32/ShortGPT 完整 200k 步轨迹，各有 ~5k step checkpoint。

**实验**：
- 对 keep8 vs full32，计算 TruthfulQA 上的 FEP distribution。
- 预测：若 Late Crystallization 是 residual stream 的固有性质而非层数的函数，则 keep8（只有 8 层）的 Late Crystal % 应该仍然很高（答案仍在最终层 crystallize），只是绝对层数更浅。
- 若 keep8 的 Late Crystal % 远低于 full32（例如 <40%），说明 crystallization 需要足够的深度（Transformer block 数量是前提），而非仅仅是"最后几层的特性"。
- 这个数据点 MechLens 完全没有：它只测了 7-14B 完整模型，没有测深度消融。

---

## D. 一句话判决

**MechLens 把"factual knowledge late crystallization 的群体级量化"占掉了**（我们的 OLMo-2-7B L18→L19 / Qwen3-8B L24→L25 onset 数据已被它的框架命名），**但它没有做 shuffle-layer null 校准（无法排除 logit lens 几何偏差）、tuned lens probe 只有 2,000 OOD 样本（probe 质量未验证）、核心 intervention threshold 只在 3 个架构上测试且关键方向上不显著（Qwen $p=0.059$）、Computability-Spectrum 无 per-category n 和统计检验**；CKA_Delta 把"跨架构 concept representations 几何-功能解离"占掉了，但它的"near-perfect functional transfer"**完全是 probe 分类任务（50维PCA子空间 + binary persona polarity），不是 generation**，且没有 null 校准，而我们的 oracle affine readout PPL=596（差 32×）恰恰填补了它没测的生成任务下界——**我们可以用 prune-heal 轨迹上的 know5 vs core6 恢复差异 corroborate Spectrum、用 shuffle-layer null 挑战两篇的绝对相似度声明、用 generation-level oracle affine 提出更尖锐版本的 geometric-functional dissociation（probe 可迁移 vs 生成不可直接复用），作为对 CKA_Delta 的 stronger follow-up**。

---

*字数：约 7,200 字（含表格）*
*抓取来源：https://arxiv.org/html/2606.07978（2026-08-06，273KB HTML，51k chars）；https://arxiv.org/html/2606.16897（2026-08-06，298KB HTML，45k chars）*
*所有引用的论文原文均来自以上两个 HTML 全文，已注明 Section 编号*
