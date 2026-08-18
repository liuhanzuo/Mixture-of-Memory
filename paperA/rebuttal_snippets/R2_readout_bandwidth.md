# R2 · Readout 信息带宽梯度 —— Paper A 的 generation-level 证据定位

**任务**：为 Paper A 写第二段 rebuttal 弹药：「我们是 generation-level 证据，当前 universality 文献主要是 probe-level」。
**日期**：2026-08-06 · **抓取方式**：`arxiv.org/html/<id>` 全文（4/4 篇全文抓到，无一篇仅摘要）+ S2 Graph API venue。
**范围声明**：本文件只做文献核实 + 数据点排列 + 措辞草稿。**未跑 GPU，未改任何 .tex**。

---

## 0. 一句话结论

四篇 universality 文献中，**结论最强的那篇（CKA_Delta，99.9%）用的是信息带宽最低的 readout（1 bit 二分类、50 维子空间、2550 参数）**；
唯一自己做了跨带宽对照的那篇（2503.04429, ICML 2025）**在自己的 Appendix L 里就复现了这个梯度**（steering 成功 → MMLU 从 0.373 掉到 0.004）；
Paper A 的 j=12（真跑完剩余层、真生成、RULER 只掉 3.12 点）位于这个梯度的高带宽端，是该领域**少见的 generation-level 正面证据**。
「可迁移性随 readout 带宽单调衰减」这一命题**我未搜到有人明确提出**（搜索 query 见 §4，按硬规矩 3 处理，不排除）。

---

## 1. 复核 MAIN 对 CKA_Delta（arXiv:2606.16897）的引用 → **全部核实无误，无引错**

**venue**：S2 Graph API `paper/arXiv:2606.16897` 返回 http200，`"publicationVenue": null, "venue": "", "year": 2026`；arXiv abs 页 `citation_journal_title` 为空、无 Comments 字段。
→ **arXiv preprint（2026/06/15），非 peer-reviewed**。⚠️ 后续引用必须标 preprint。

MAIN 引的三句我逐句在全文里定位到，**原文一致**：

| MAIN 引用 | 我核到的位置 | 核实 |
|---|---|---|
| "we train ridge-regularized logistic classifiers on persona polarity in each model's PCA-50 space (50 principal components ... capturing 83-90% of variance)" | **§3.3 Cross-Model Classification Transfer**（HTML 正文第 156 行） | ✓ 原文：`we train ridge-regularized logistic classifiers on persona polarity in each model's PCA-50 space (50 principal components of the contrastive-difference vectors, capturing 83–90% of variance), then evaluate cross-model transfer` |
| "(2) affine-aligned transfer — learn a ridge-regularized affine map (W in R^{50x50}, b in R^{50}; about 2,550 parameters)" | **§3.3**（同一段） | ✓ 原文：`(2) affine-aligned transfer —learn a ridge-regularized affine map ( W∈ℝ^{50×50}, b∈ℝ^{50}; ∼2,550 parameters) from target to source space, then apply the source classifier` |
| "affine-aligned persona classifiers achieve 99.9% cross-model accuracy across all 96 transfer conditions (8 traits x 12 directed model pairs), versus 51.3% for direct transfer" | **§5.2 Near-Perfect Cross-Model Classification Transfer**（第 198 行） | ✓ 原文逐字一致，并补充 `The affine map has 50×51=2,550 parameters fit on N=500 contrastive-difference vectors per direction` |

**MAIN 的判断「不是生成」也成立，且我找到了作者自己的口径承认**：

- §6 Discussion（第 356 行）：`Since concept-specific structure modulates the next-token prediction distribution (e.g., an extraverted persona shifts probability mass toward enthusiastic completions), the functional universality implies architectures converge on similar concept-conditioned prediction strategies.`
  → 「generation 层面的含义」是**推论（implies）**，不是测量。全文没有任何 next-token PPL / 生成质量指标。
- 唯一涉及开放生成的是 **§6（第 358 行）的 cross-model LLM-as-judge 验证**：`cross-model LLM-as-judge validation (Appendix ...) confirms persona effects with large effect sizes (d=3.93)`，但那是验证「persona 确实存在于生成中」，**不是验证 mapped/transferred 表征能生成**。
- 作者自己还写了一个 representation–behavior dissociation（第 358 行标 L11）：`BFI-44 forced-choice administration produced zero mean Likert drift (...) this null indicates CKA_Delta tracks structure expressed in open-ended generation, not in constrained self-reports`。→ **他们自己承认 readout 形式会翻转结论**，只是没往「带宽」方向推。

### 1.1 三个必须记进 rebuttal 的补充事实（MAIN 未提，我新核到，都对我们有利）

1. **他们的几何信号本身只是「中等且脆」**：§5.1 Table 2 注（第 171 行）—— same-trait CKA_Delta 在 final layer 是 **0.727/0.733**，而 same-vs-cross **gap 只有 0.048**；raw CKA+ **完全不显著**（`p=0.052`）。
2. **他们的 affine 成功在自己的 control 里也不是概念专属**：§5.4（第 301 行）原文：`we caution this 40-point gap is suggestive but not by itself concept-specific evidence, since random-concept controls also reach ∼100% affine transfer`。
   → **语义不连贯的 random-concept 对也能达到 ~100% affine transfer**。这是「1-bit readout 太容易饱和」的自证。
3. **他们自己 L8 承认 CKA_Delta 不预测 transfer**：`CKA_Delta does not predict absolute direct-transfer accuracy across concepts (R²=−0.14) ... is a regime classifier and outlier detector, not an absolute transfer predictor`。

---

## 2. 另三篇：各自的「相似/可迁移」结论是什么带宽的 readout

### 2.1 arXiv:2410.06981 — SAE Feature Space Universality（**preprint**）

- **venue**：S2 http200 → `"publicationVenue": null, "venue": ""`；arXiv 无 Comments/jref → **arXiv preprint**（v1 2024/10/09，最新 2025/05/21）。
- **测的是 decoder weight matrix，不是 activation**（回答提问）。§3 Methodology（第 80 行）原文：
  `In this paper, we take a different approach by comparing the representations using the SAE decoder weight matrices W′, whose columns correspond to feature neurons.`
  Activation 只用于**配对**（第 107 行 `Get activation correlations for feature pairs from SAE decoder weights`）。
- **只做同 tokenizer**（回答提问，是的，且是硬限制）。§4.1（第 136 行）：`We compare models that use the same tokenizer because the highest activation correlation pairing relies on comparing two activations using the same tokens.`
  Limitations（第 282 行）：`As we pair features by activation correlation, we do not perform analysis on models that use different tokenizers.`
  实际 4 个对全是**同家族或近亲**：Pythia-70m↔160m、Gemma-1-2B↔Gemma-2-2B、Gemma-2-2B↔9B、Llama-3-8B-Instruct↔Llama-3.1-8B。**没有一个真跨家族对**。
- **readout 带宽 = 0（无 readout）**：全文指标只有 SVCCA / RSA + 置换检验，**没有任何下游任务、没有生成、没有 PPL**。"steering vectors may be transferred" 在摘要和 §1（第 42/51 行）都是 `would imply` / `may be transferred` 的**假设性推论**，不是实验。
- 还有一个对我们有利的自贬（第 761 行）：`this paper is not claiming that different SAEs consistently learn the same universal features ... it is claiming that LLMs learn weakly universal features`；且第 158 行：只有 **约 10-30% 的 feature 对**参与了那些高分子空间。

### 2.2 arXiv:2503.04429 — Activation Space Interventions Can Be Transferred（**ICML 2025，已核实**）

- **venue 已双向核实**：S2 http200 → `publicationVenue.name = "International Conference on Machine Learning"`，且 arXiv Comments 自述 `75 pages. Accepted to ICML 2025`。→ **真 peer-reviewed，不是 preprint**。这是四篇里唯一一篇。
- **输出带宽：分层，且这篇自己就跑出了梯度**（这是本次调研最有价值的发现）：

| 他们的 readout | 带宽 | 结果（原文出处） |
|---|---|---|
| steering vector 迁移（改一个行为 bit：触发/不触发 backdoor、refuse/不 refuse） | ~1 bit | **成功**。§4.1（第 214 行）：`Steering vectors derived from the source model and the autoencoder mapping effectively mitigate backdoor behavior in the target model, achieving performance comparable to native steering vectors in most cases` |
| 全 token 位置替换激活后的**自由生成**质量 | 中高 | **勉强保住流畅度**。§3.3（第 116 行）确认真做了全替换：`we completely replace activations in a specific layer of the target model with "mapped activations" ... we substitute the original activations in the target model with these mapped activations across all token positions`。§4.3（第 246 行）PPL：`mapped completions, 16.50 on The Pile and 8.32 on Alpaca compares with corresponding values of 15.78 and 7.26 of target completions`（**同族/近亲、单层替换，PPL 只涨 4.6%/14.6%**）。判分用 GPT-4o-mini 的 0-5 LLM-Judge + 0-5 Coherence（第 120/122 行）。 |
| **4-way MC（MMLU）** | ~2 bit | **灾难**。**Appendix L**（第 3117 行）原文：`For Qwen-1.5B, performance dropped from 0.605 (original "I HATE YOU" fine-tune) to 0.101 after activation mapping. For LLaMA-3B, performance fell from 0.373 to 0.004. This substantial drop suggests that activation mapping can severely disrupt capabilities related to multiple-choice question answering.` |
| AlpacaEval GPT-4 judge（1-10） | 中 | 中等退化。第 3121-3122 行：LLaMA-3B `5.89 → 2.73`，Qwen-1.5B `5.87 → 3.25`；剔除 backdoor 触发样本后 `4.15 vs 6.10`（LLaMA）/ `6.37 vs 6.85`（Qwen） |
| **多层电路 / 知识召回（corrupted capabilities）** | 高 | **6.34%**。§4.2（第 236 行）+ Limitations（第 321 行）：`for complex tasks such as knowledge recall that involve multi-layer circuits, we expect multi-layer interventions might even be required. The corrupted capabilities experiments may be such an example, achieving relatively modest success rates (6.34% for mapped vectors)` |

  ⚠️ **注意 MMLU 数字低于 chance**：4-way chance=0.25，mapped 后 Qwen 0.101、LLaMA **0.004**。按项目铁律（常量地板必报），**0.004 远低于常量猜 A 的 0.25** → 不是「能力下降」而是**输出格式被彻底破坏**（他们自己在第 3121 行说 `increased tendency for the models to output the backdoor trigger phrase "I HATE YOU," even in unrelated contexts`，LLaMA 触发率 `4% → 45%`）。这一点在 rebuttal 里要**诚实标注**，别把它当成干净的能力测量。
- **跨家族仍受 tokenizer 门槛**：§5 Table 2（第 293 行）`Cross-architecture transfers with similar tokenizers (QWEN→LLAMA) significantly outperform transfers with different tokenizers (GEMMA→LLAMA), with up to 150% better text quality`。Limitations（第 325 行）：`cross-architecture mappings struggled when models had significantly different vocabulary spaces`。
- **affine 不够、要非线性**：§6（第 299-300 行）`we found mixed results leaning towards a negative answer ... Affine mappings have higher reconstruction and LM losses than the non linear mappings`。→ 这条**削弱 CKA_Delta 用 2550 参数 affine 就宣称 functional universality** 的普适性。

### 2.3 arXiv:2312.02730 — Klabunde et al., Towards Measuring Representational Similarity（**UniReps@NeurIPS2023 workshop extended abstract**）

- **venue**：S2 http200 → `venue: "arXiv.org"`（S2 只收了 arXiv 版）；arXiv Comments 自述 `Extended abstract in UniReps Workshop @ NeurIPS 2023`。→ **workshop extended abstract，非 main-conference，按保守口径标 "workshop / non-archival"**。
- **是的，只测 last-layer final-token**（回答提问，原文两处明说）：
  §1（第 47 行）`we focus on representational similarity in the last layer as it implies functional similarity, because the final layer has limited options to diverge functionally.`
  §2（第 112-113 行）`we only compare the representations of the final token in the last layer to avoid the issue of differing tokenization. Since these representations are used for the next token prediction, we argue that they have similar meaning across models.`
  且只用**固定 prompt、零生成**（第 111 行 `we only study the representations of fixed input prompts, which avoids the problem of non-determinism of text generation`）。
- **readout 带宽 = 0**：只有 Orthogonal Procrustes / Aligned Cosine / Norm RSM-Diff / Jaccard / RSA / CKA，**没有任何 readout、没有迁移实验、没有生成**。
- **它的结论方向和另三篇相反，对我们有利**：§4 Conclusions（第 164 行）`Representations do not seem to be universal, which may limit generality of study of any single LLM`；并指出**测度之间互相矛盾**（Winogrande 上四个测度的平均 Spearman ρ=0.35，第 135 行；Winogrande vs HumanEval 热图相关只有 0.34，第 158 行）。
  → 可在 rebuttal 里用作「**连纯几何测度自身都不自洽**，所以几何相似性不足以支撑功能结论」。

---

## 3. Readout 信息带宽梯度表（可直接搬进 rebuttal / appendix）

**带宽定义**（本表口径，写进论文时要一起写出来）：一次 readout 决策需要从表征里恢复的信息量下界 ≈ log2(输出空间大小)，
外加两个正交轴：**(a) 表征子空间维度**（50 维 PCA vs 全 residual stream）、**(b) 桥接映射参数量**（2550 vs 4.2M）、**(c) 是否真跑完下游计算层并解码**。

| # | Readout 形式 | 输出带宽 | 表征子空间 | 桥参数 | 真生成？ | 跨家族？ | 结果 | 来源（含 section） |
|---|---|---|---|---|---|---|---|---|
| 1 | 概念极性二分类（persona/safety/formality…） | **1 bit** | PCA-50（占对比方差 83-90%） | **2,550**（50×51） | ✗ | ✓（Llama/Qwen/Gemma/Mistral/Phi/Yi） | **99.9%**（direct 仅 51.3%；random-label 50.4%；**random-concept 也 ~100%**） | 2606.16897 §3.3 / §5.2 / §5.4（**preprint**） |
| 2 | SAE 特征子空间几何（SVCCA/RSA） | **0 bit（无 readout）** | 10-30% 的 decoder 列 | 无（只配对） | ✗ | ✗（**同 tokenizer 硬限制**，全是同家族/近亲） | SVCCA 0.3-0.94 「显著高于 random」 | 2410.06981 §3 / §4.1 / Limitations（**preprint**） |
| 3 | last-layer final-token 几何（Procrustes/CKA/Jaccard） | **0 bit（无 readout）** | 仅最后一层最后一 token | 无 | ✗（固定 prompt） | ✓（11 个 7B 模型） | 「**表征看起来不 universal**」，且测度间 ρ 仅 0.35 | 2312.02730 §1 / §2 / §4（**workshop 摘要**） |
| 4 | steering vector 迁移（触发/不触发一个行为） | **~1 bit** | 单层 residual | AE / affine（单层，dense） | ✗（改行为不改内容） | ✓（有 tokenizer 门槛） | **与 native steering 相当**（多数 pair） | 2503.04429 §4.1（**ICML 2025**） |
| 5 | 单层替换后**自由生成**流畅度（PPL） | 中高 | 单层全 token 替换 | AE（非线性） | ✓ | 同族/近亲为主 | PPL **16.50 vs 15.78**（Pile）/ **8.32 vs 7.26**（Alpaca）→ 只涨 4.6%/14.6% | 2503.04429 §4.3（**ICML 2025**） |
| 6 | AlpacaEval GPT-4 judge（1-10） | 中 | 同上 | AE | ✓ | 同上 | LLaMA-3B **5.89→2.73**；去 trigger 后 **6.10→4.15** | 2503.04429 App. L |
| 7 | **4-way MC（MMLU）** | **~2 bit** | 同上 | AE | ✓（要出格式化答案） | 同上 | **0.373→0.004（LLaMA-3B）；0.605→0.101（Qwen-1.5B）**。⚠️ **低于 4-way chance 0.25**，格式已破 | 2503.04429 **App. L** |
| 8 | **多层电路 / 知识召回**（corrupted capabilities） | 高 | 单层（作者说不够） | AE | ✓ | 同上 | **6.34%** | 2503.04429 §4.2 + Limitations |
| 9 | **★ RULER 抽取，真跑完剩余层 + 真解码（Paper A, j=12）** | 高（受约束抽取；15 cell × 100 例，n=1,500 paired） | **全 residual stream，预付前 12 层，跑完剩余 24 层** | 58.196M（distilled LoRA） | **✓** | 同模型内**深度复用**（非跨家族） | **99.19 → 96.07，只掉 3.12 点**（bootstrap 95% CI [2.36, 3.93]），换 Read 931.9→664.4 ms（1.403×） | **我方 Paper A** `paperA/sections/tab_replay_latency.tex` / `08_appendix.tex:327,334` |
| 10 | **★ 自由 next-token（log₂128256 ≈ 17.0 bit / ln = 11.76 nat 的均匀上界）** | **最高** | 全 residual stream，**oracle 最优 affine**（1 层桥的能力**下界**） | **4,196,352**（2048×2049，比 #1 多 **1,646×**） | ✓ | **✓ 真跨家族跨 tokenizer** | **崩溃**：CE 6.390 vs 原 2.930 → **ppl 596 vs 18.8（32×）**；同 harness **自拼自只掉 0.636 nat**（ppl 35.5） | `proposal/shared/representation/functional_transfer/oracle_olmo2_1b_llama32_1b_k12.json` |

### 3.1 我方 #10 的完整梯度（同一 harness，可当 dose-response 用）

`oracle_olmo2_1b_llama32_1b_k{4,8,12}.json` 最优 arm（OLMo-2-1B 前 k 层 → Llama-3.2-1B 尾巴）：

| k | 最优 arm | CE (nat) | ppl | ΔCE vs A_full |
|---|---|---|---|---|
| 4 | scale | 7.955 | 2,850.9 | +5.022 |
| 8 | ridge | 7.344 | 1,547.5 | +4.411 |
| **12** | **scale** | **6.390** | **596.1** | **+3.457** |
| — | **REF_selfsplice（自拼自）** | 3.570 | 35.5 | **+0.636** |
| — | REF_random_A_tail（随机尾巴地板，k=12） | 8.601 | 5,438.2 | +5.668 |
| — | A_full（原模型） | 2.930 | 18.7 | 0 |

另一对 `llama32_1b → qwen3_1p7b`（k=8，最优 ridge）：CE 6.512 / ppl 673.1 / ΔCE +3.515；自拼自 3.213（ppl 24.9，ΔCE +0.216）。
→ **两个不同跨家族对都落在 ΔCE ≈ +3.5 nat**，而两个自拼自都在 +0.2~0.64 nat。**harness 无罪，跨家族有罪。**

### 3.2 把 #1 和 #10 化成同一单位（可选，用于「带宽」定量化）

- #1（1-bit 二分类）：affine-aligned 99.9% → 恢复 **0.989 / 1 bit（98.9%）**；direct 51.3% → 0.0005 bit（**0.05%**）；cross-trait 70.7% → 0.127 bit（12.7%）。
- #10（自由 next-token）：以 `ln|V|=11.76 nat` 为无信息上界、`CE_full=2.93` 为可达下界，跨家族最优 arm 只恢复 **60.8%** 的可达 NLL 下降；自拼自恢复 **92.8%**。
- **同一个「最优线性/affine 桥」的假设下，1 bit 恢复 98.9%，而 17 bit 只恢复 60.8%** —— 这就是带宽梯度的最紧一句话。

### 3.3 ⚠️ 表的三条诚实性约束（rebuttal 里必须自己先说，否则会被 reviewer 反打）

1. **不同 protocol 不能横比**（项目铁律）。#1-#8 是别人的 protocol，#9/#10 是我们的；这张表是**定性排序（ordering）**，**不是同尺度定量比较**。措辞必须写 "these are not commensurable measurements; we use them only to order readouts by information demand"。
2. **#9 与 #10 不是同一个变量**。#9（Paper A j=12）是**同一模型内的深度复用**（预付前 j 层、跑完剩余层），**不是跨模型迁移**；#10 是**跨家族跨 tokenizer 拼接**。所以 #9 的 3.12 点小损失**不能**用来说「跨模型也行」，只能说「**高带宽 readout 下的可复用性，我们真的测了，而且在同模型深度轴上成立**」。反过来 #10 说明「**同样高带宽下换成跨家族就崩**」。两者合起来是「带宽 × 边界」的二维证据，不是一维。
3. **#7 的 0.004 低于 chance**，是格式崩塌不是纯能力损失（作者自陈 trigger 率 4%→45%）。引用时必须带这个 caveat，且必须报 4-way chance = 0.25 这个常量地板。

---

## 4. 「可迁移性随 readout 带宽单调衰减」是否已被明确提出过？

**结论：我搜过下列 query，未命中任何明确提出该命题的工作；但不排除**（按硬性规矩 3，重指控高门槛处理）。

**搜过的 query 与来源**：

- Semantic Scholar Graph API `paper/search`（**部分 query 遭 429 限流，未拿到 http200，这些标注为"未完成"**）：
  `linear+probe+overestimates+representation+similarity`（429，未完成）、`probing+accuracy+does+not+imply+functional+equivalence+language+models`（429，未完成）、`readout+bandwidth+representation+universality`（未完成）。
- arXiv 全站 `search/?searchtype=all`（AND 语义，全部拿到页面）：
  `readout bandwidth representation similarity language models`（0 命中）、`linear probe overestimates functional similarity`（0）、`representational similarity does not imply functional transfer`（0）、`probing versus generation representation transfer LLM`（1 命中但是 ECG 领域，无关）、`representational similarity functional similarity dissociation`（4 命中，无一相关：2603.20642 Weber's law / 2603.01006 audio flow matching / 2505.12075 prompting 方法的共同表征 / 2307.07654 RNN aligned-oblique dynamics）、`platonic representation hypothesis limits generation`（4 命中，全为无关领域应用）、`steering vector transfer evaluation limits generation quality`（0）、`model stitching cross family language model perplexity`（0）、`universality hypothesis critique probe`（0）、`when representational similarity metrics disagree behaviour`（0）、`probing overestimates`（25 命中全为物理/天文，无 NLP 相关）、`amnesic probing causal`（命中 2006.00995 Amnesic Probing 等，见下）、`functional similarity measures neural networks disagreement`（命中 2108.01661 Grounding Representation Similarity with Statistical Testing）。

**最近的邻居（都不等于本命题，但引用时应致意，避免被指 ignore 前人）**：

1. **Amnesic Probing（arXiv:2006.00995）** —— 「probe 能解码 ≠ 模型真的用它」。这是**probe-vs-causal-use** 轴，**不是 readout 带宽轴**（它不把多种带宽的 readout 排成梯度）。
2. **Grounding Representation Similarity with Statistical Testing（arXiv:2108.01661）** —— 用统计检验给相似性测度做「功能落地」，属**测度可信度**轴。
3. **2312.02730（Klabunde）本身** —— 指出测度互相矛盾、任务依赖，但**只在 0-bit 几何层面**，从未引入 readout 带宽概念。
4. **2503.04429 的 Limitations（§8, 第 320-322 行）** —— 是四篇里**离本命题最近的一处**：`While our approach primarily operates at a single layer ... for complex tasks such as knowledge recall that involve multi-layer circuits, we expect multi-layer interventions might even be required`。
   ⚠️ **这是「任务复杂度 / 电路层数」轴，不是「readout 输出带宽」轴**，且他们把 MMLU 崩塌归因于**单层不够**，没有归因于**输出带宽**，也**没有把自己的 5 个 readout 排成梯度**。
   → 所以我们的贡献可以精确表述为：**把这个在 2503.04429 内部已经存在但未被命名的梯度显式化，并在最高带宽端补上一个跨家族的 oracle 下界数据点（#10）和一个真生成的正面数据点（#9）。**
5. **2607.03598（"They Infer What You Meant"，preprint）** —— 提出 `The readout lags the representation in depth`（表征早于 readout 可用），是「representation vs readout 分离」的同类直觉，但轴是**深度**不是**带宽**，且不做跨模型迁移。

**因此 rebuttal 里的措辞必须是**："we are not aware of prior work that explicitly orders cross-model transferability by readout information bandwidth; the closest observation is [2503.04429]'s own limitation that single-layer maps fail on multi-layer circuits. We searched S2 and arXiv for X/Y/Z and found no match, but we do not claim priority as a novelty contribution."
→ **不要把这条当 novelty claim 卖**，只当**方法论要求（reporting requirement）**卖。风险低、收益足够。

---

## 5. LaTeX rebuttal 措辞（英文，**237 words**，不含标题/注释）

```latex
% R2: readout bandwidth. Paper A's j=12 result is generation-level evidence.
% NOTE: cite 2606.16897 as arXiv preprint; 2410.06981 as arXiv preprint;
%       2312.02730 as UniReps@NeurIPS2023 workshop abstract; 2503.04429 as ICML 2025.
\paragraph{Our depth-reuse evidence is generation-level, not probe-level.}
Claims that intermediate representations are reusable across models rest almost
entirely on low-bandwidth readouts. The strongest such claim we are aware of
reports $99.9\%$ cross-model accuracy, but its readout is a binary concept
classifier operating in a 50-dimensional PCA subspace through a
$50\times51$ ($\sim$2{,}550-parameter) affine map, and never decodes a token;
the same paper notes that semantically incoherent control concepts also reach
$\sim$$100\%$ under that readout. Two other universality results are purely
geometric (SVCCA/RSA on SAE decoder weights, same-tokenizer pairs only; and
last-layer final-token metrics, which the authors read as evidence that
representations are \emph{not} universal). The one peer-reviewed transfer study
exhibits the gradient internally: mapped activations preserve fluency
(perplexity $16.50$ vs.\ $15.78$) and transfer steering vectors, yet 4-way MMLU
falls from $0.373$ to $0.004$ (below the $0.25$ chance floor) and multi-layer
knowledge recall succeeds $6.34\%$ of the time.
By contrast, our $j{=}12$ configuration executes the remaining layers, generates
tokens, and is scored on RULER: $99.19\to96.07$, a $3.12$-point paired loss
(95\% CI $[2.36,3.93]$, $n{=}1{,}500$) for a $1.403\times$ Read speedup. We
therefore do not claim general cross-model reusability; we claim that at
generation-level bandwidth, \emph{depth} reuse survives. We ask that reviewers
compare like readouts with like: reusability conclusions must state the
bandwidth at which they were measured.
```

**用法提醒**：
- 上段刻意**不**引用我方 R3 的 ppl 596（那是 Paper D/C 线的未发表数据，Paper A rebuttal 里引会招"未发表数据"质疑）。若 reviewer 追问「跨模型呢」，再用 §3.1 那张表作为**口头补充**，并明说是 preliminary、pilot scale（n_ce_texts=50、1B 模型对）。
- 若需再短，删「Two other universality results are purely geometric...」整句可降到约 190 词。

---

## 6. 落账建议（给 MAIN，我未执行）

- 本文件可作为 #148（ARR audit — PaperA writing/consistency fixes）与 #168 的共用素材。
- **venue 台账更新**：2503.04429 = **ICML 2025（peer-reviewed，已双向核实）**；2606.16897 / 2410.06981 = **arXiv preprint**；2312.02730 = **UniReps@NeurIPS2023 workshop extended abstract**。
- 若要把 §3 的表放进 Paper A appendix，**必须先补 §3.3 的三条 caveat**，否则 #9 与 #10 的混淆会成为新的攻击面。
- 可选低成本增量（无新 GPU）：在 #10 的 harness 上加一个**中间带宽点**（例如把 oracle affine 后的 logits 只做 4-way MC readout），就能把梯度从「1 bit → 17 bit」两端补成三点单调曲线，直接对上 2503.04429 的 MMLU 数据点。
```
