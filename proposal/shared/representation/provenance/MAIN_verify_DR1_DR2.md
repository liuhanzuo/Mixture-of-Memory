# MAIN 核实：DR1 / DR2 精读报告（含一处纠正 + 一个升级的新观点）

**日期**：2026-08-06 GMT+8
**被核实**：`DR1_mechlens_ckadelta.md`（agent ab79b3dd）、`DR2_downscaling_cpt_budget.md`（agent af69c9b4）
**核实方式**：MAIN 亲自 refetch `arxiv.org/html/2606.16897` 与 `ar5iv.../2310.04680` 全文，grep 原文关键句。

---

## 1. ★ DR1 的一处错误（必须纠正，且纠正后结论更强）

**DR1 声称**：CKA_Delta（2606.16897）「★★★ **无 null 校准**」。

**MAIN 核实：这是错的。它做了 permutation null。** 原文 §5.1：

> "Permutation testing (n=200 shuffles, **sufficient for single-hypothesis null calibration**) validates:
> real CKA_Δ = **0.727** vs. **null = 0.689 ± 0.005** (p<0.005; permutation **z = 7.4**)."

以及 §Statistical methodology：
> "Permutation testing (n=200 shuffles ...; SAE multi-test analysis uses n=1,000). Bootstrap 95% CIs
> (10,000 resamples) ... nonparametric confirmation via Mann–Whitney U and Wilcoxon signed-rank tests.
> We use the **debiased HSIC estimator** (song2012feature)"

它还有 3 个 transfer controls（原文 §3.3 + §5.2）：
> "Controls include **random-label** (shuffled target labels), **cross-trait**, and **random concept**"
> "random-label (**50.4% ± 0.9%**, n=480), cross-trait (**70.7% ± 18.8%**; Welch's t=10.71, p<1e-6),
> and PCA dimension robustness (96.2% at d=10, 100% at d=100)"

⇒ **不能写「他们没做 null」**。这个指控是假的，写进论文会被直接反驳。

### 但纠正之后出现了一个**更强**的观点

把它的 null 数字和我们的并排：

| 工作 | 指标 | observed | null | **gap** | null 占 observed |
|---|---|---|---|---|---|
| **我们**（91 pair, z-CKA midband, **shuffle-层序** null） | midband z-CKA | 0.491 | 0.453 | **+0.038** | **92.3%** |
| **CKA_Delta**（persona, **permutation** null） | CKA_Δ | 0.727 | 0.689 | **+0.038** | **94.8%** |

**两篇完全独立的工作、不同指标、不同 null 构造，gap 都恰好 +0.038，null 都已解释掉 observed 的 92–95%。**

而 CKA_Delta 用这个 +0.038 的 gap（z=7.4, p<0.005）支撑了标题级主张
"**moderate** geometric convergence"，并在 abstract 里与 "near-perfect functional transfer" 并列。

⇒ **真正可写的批判不是「他们没做 null」，而是**：
> **这个领域的 representational-similarity 主张普遍存在「统计显著性被当作语义强度」的问题**：
> null 校准后剩下的绝对间隙只有 observed 的 5–8%，却被表述为 "moderate convergence"。
> 我们用 91 pair 独立复现了同一量级的间隙（+0.038 / 7.7%），说明这不是单篇的偶然，
> 而是 **CKA 类指标在 LLM 上的系统性质**：绝大部分绝对数值来自「任何两个训练过的网络共有的
> 通用几何」，而非特定对应关系。

这比原来的观点 (ii) 更硬，因为**不依赖指控别人漏做实验**，而是指出**共同的解释学问题**，
并且我们有独立复现的数字。**这也回应了用户「前人做的有问题的地方」的要求 —— 问题在解释，不在执行。**

---

## 2. ★★ DR1 的核心正确发现：「functional transfer」的定义被 MAIN 逐字核实

**DR1 说**：CKA_Delta 的 "near-perfect functional transfer" = PCA-50 子空间里的 persona polarity
二分类，**不是 generation**。

**MAIN 核实：完全正确，原文比 DR1 描述得更具体。** §3.3 原文：

> "we train **ridge-regularized logistic classifiers on persona polarity** in each model's
> **PCA-50 space** (50 principal components of the contrastive-difference vectors, capturing
> 83–90% of variance), then evaluate cross-model transfer. ... (2) **affine-aligned transfer**
> — learn a ridge-regularized affine map (**W ∈ R^{50×50}, b ∈ R^{50}; ~2,550 parameters**)
> from target to source space"

§5.2 原文：
> "affine-aligned persona classifiers achieve **99.9%** cross-model accuracy across all
> **96 transfer conditions** (8 traits × 12 directed model pairs), versus **51.3% for direct
> transfer**. The affine map has 50×51 = 2,550 parameters fit on **N=500** contrastive-difference
> vectors per direction"

### 与我们数据的精确对撞（这是 (i) 观点的实证基础）

| | CKA_Delta | 我们（R3 oracle affine） |
|---|---|---|
| 变换 | affine **R^{50×50}**（PCA-50 子空间） | affine，**full residual stream** |
| 参数量 | **2,550** | d_model × d_model 级（1B 模型约 2048² ≈ 4.2M） |
| 拟合数据 | N=500 contrastive-difference vectors | 120 texts / 8000 tokens |
| **任务** | **persona polarity 二分类**（8 traits） | **next-token generation**（全 100352 词表） |
| 结果 | **99.9%** accuracy（direct transfer 仅 51.3%） | **ppl 596 vs 原 18.8（差 32×）**；自拼自仅掉 0.64 nat |

**两者不矛盾，但共同界定了一个梯度**：
- 二分类 readout（1 bit 输出，50 维子空间）：affine 对齐后 **99.9%**
- 生成 readout（log|V| ≈ 11.5 bit 输出，全 stream）：同类 affine 对齐 **崩溃 32×**

⇒ **观点 (i) 的可写形式**：
> "functional universality" 的结论**强依赖于 readout 的信息带宽**。
> 概念极性这类低带宽 readout 在 2,550 参数的 affine map 下近乎完美可迁移；
> 而 next-token 分布这类高带宽 readout 在同族 affine 变换下相差 32×。
> 因此 "concepts are universal" 与 "models are not substitutable" 可以同时为真，
> **报告 functional transfer 时必须声明 readout 带宽**。

我们**两边的数据都已在盘上，零新 GPU**。

---

## 3. ★★ DR2 核心声明逐条核实通过（决定我们最大窄缝）

MAIN refetch `ar5iv.../2310.04680` 全文（83,568 chars）grep 结果：

| DR2 声明 | MAIN grep 核实 |
|---|---|
| 是**非结构化/权重稀疏**剪枝，不是深度剪枝 | ✓ §3 原文："We use **SparseGPT** ... and **Wanda** ... prune each layer by **minimizing the ℓ₂-distance between the outputs of the original dense layer**" |
| **无 heal / 无续训** | ✓ `continued pretrain` **0 hits**、`heal` **0 hits**、`recover` **0 hits**；原文自述 "**without computationally intensive re-training**" |
| one-shot 立即测 | ✓ "Both are **one-shot** pruning algorithms that scale to LLMs" |
| dense scaling 是另训的独立模型 | ✓ "for the latter, we use **(separately-trained) dense models** with increased/reduced width and depth" |
| 30% 阈值 | ✓ "removing more than **30% of weights** leads to significant (>5%, relative) accuracy degradation on fact recall" |

⇒ **确认我们最大的窄缝（三重差异，且每一重都可核查）**：

| 维度 | ICLR'24 (2310.04680) | 我们 Paper B |
|---|---|---|
| 损伤机制 | 权重稀疏（层内 ℓ₂ 最优） | **深度剪枝**（丢弃整层）+ **嫁接 K=2 随机新层** |
| 恢复 | **无**（one-shot，明确不 re-train） | **continue-pretrain 全参 200k step** |
| 时间维度 | **无**（静态快照） | **每 ~5k step 一个 ckpt，全轨迹** |
| 报告轴 | fact-recall vs in-context | **PPL vs fact-recall**（它测了 PPL 但未做系统对照） |

---

## 4. DR2 报的其他 kill-risk 与其强度（MAIN 未逐条 refetch，标注为待核）

- **2506.00288（自述 ACL'25）**："until later in training" ≈ **step 3k（~3.1B tokens）**，
  **定性描述，未量化为可预测预算**。且它的 PPL 是「目标语言同分布 PPL」，不是 held-out PPL。
  → 我们的量化是新的。**但这条我没自己 refetch，DR2 的 step 3k 数字待核。**
- **2407.17467（自述 EMNLP'24）**：CMR 公式 `R_CMR = α₄·T^s₄ + β₃`，因变量**全是 validation loss，
  无 downstream accuracy**（DR2 说它 §8 自述 limitation）。模型最大 3.1B 且**内部数据不可复现**。
  → 与「能力恢复」不是同一件事。**同样待核。**
- DR2 报告 2506.00288 的自造 benchmark Copain 在 CPT(eu) 下得 **20.12%，低于 random chance（~25-33%）**
  → 若为真，这正是我们 Paper C 踩过的「常量地板」问题的同型病症，**是一个可引用的他人失误**。
  **必须自己核实**（这是对他人的指控，门槛要高）。

---

## 5. 修订后的三个新观点及其现状

| # | 观点 | 状态 | 依赖的我方资产 | 还缺什么 |
|---|---|---|---|---|
| **(i)** | **"functional transfer" 结论依赖 readout 信息带宽**：低带宽概念探针 99.9% vs 高带宽生成 32× 崩溃 | ★ **已具备双侧实证**（CKA_Delta 原文 + 我方 oracle affine），零新 GPU | R3 oracle affine (ppl 596/18.8) + 91 pair CKA | 需要把「带宽」形式化（bit 数 or 有效自由度），并补 1-2 个中间带宽点（如 4-way MC readout） |
| **(ii)** | **null 校准后的绝对间隙只有 5-8%，却被表述为 moderate/near-perfect convergence** —— 系统性解释学问题，非单篇失误 | ★★ **最强**。两篇独立工作 gap 都是 +0.038 | 91 pair shuffle-null（0.491 vs 0.453） | 需要再扫 3-5 篇同类工作的 observed/null 对，看 +0.038 量级是否普遍 |
| **(iii)** | post-norm（OLMo-2）是 14 模型中 midband 最低 | 待 DR4 判定是否有人做过 | per_model_mean_midband_zcka | DR4 未回 |
| **(iv) NEW** | **「恢复预算」窄缝**：深度剪枝+嫁接 × 步级轨迹 × 多能力分解 —— ICLR'24 无 heal 无轨迹（已核实），CMR 只有 loss，ACL'25 只有定性延迟 | ★ **被占约 30-35%**（DR2 判断），核心实验设计空白 | keep8/10/12/14/16 × 200k step × 5 eval | 需核实 2506.00288 / 2407.17467 的 DR2 声明 |

---

## 6. MAIN 的元教训（记账）

**subagent 报「他人没做 X」时，我必须自己 refetch 核实。**
DR1 把「有 permutation null」报成「无 null 校准」——如果我照抄进论文，是一个可被直接反驳的
虚假指控。这与 [[two-disk-rule-applies-to-main-too]] 同型：**否定性论断门槛要高**。

反过来也成立：**DR1 那条错误纠正后，反而产出了比原观点更强的 (ii)**
（从「他们漏做」升级为「大家都做了，但都把 5% 的间隙说成了 moderate convergence」）。
**核实不是为了否定 subagent，是为了把观点打磨到能上台面。**
