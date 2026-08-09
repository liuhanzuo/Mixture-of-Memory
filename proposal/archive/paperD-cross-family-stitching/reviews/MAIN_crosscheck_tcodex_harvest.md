# tcodex 收获的 117 篇文献 × claude reviewer 结论的交叉复核

**日期**：2026-08-06 17:30 GMT+8
**来源**：tcodex（gpt-5.6-sol, effort=max, web_search=true）跑 1h45m 未出终稿，但 a1.log
已抓下 **117 篇唯一 arXiv 条目**，MAIN 从 log 提取到
`paperD_research/direction_20260806/tcodex_harvested_arxiv.json`。
**核实**：本机经 hy-proxy 直取 `arxiv.org/abs/<id>` 的 `citation_title`，下列 6 篇**逐条对上**。

⚠️ **tcodex 的 log 里有一处摘要串行**：`2310.04680` 的 SUMMARY 字段被 SOLAR (2312.15166)
的摘要覆盖了。MAIN 重新直取原文，本文用的是**真摘要**。这说明 log 提取的摘要不能盲信，
**载重结论必须自己 refetch**。

---

## ★★ 结论 1：claude 的 top-1 (D1「PPL vs 知识三向解离」) 被 ICLR'24 占掉

**arXiv:2310.04680 — "The Cost of Down-Scaling Language Models: Fact Recall Deteriorates
before In-Context Learning"**（Tian Jin, Nolan Clement, Xin Dong, Vaishnavh Nagarajan,
Michael Carbin, Jonathan Ragan-Kelley, Gintare Karolina Dziugaite）
venue-hint 自带：**"The Twelfth International Conference on Learning Representations (ICLR), 2024"**

摘要原文（MAIN 直取）：
> "We study two natural scaling techniques -- **weight pruning** and ... dense scaling -- and
> their effects on two core capabilities: (a) **recalling facts presented during pre-training**
> and (b) processing information presented **in-context** ... we find a **striking difference in
> how these two abilities evolve** due to scaling. **Reducing the model size by more than 30%
> significantly decreases the ability to recall facts seen in pre-training. Yet, a 60--70%
> reduction largely preserves** the various ways the model can process in-context information"

**这就是 claude D1 的命题**：「压缩后 fact recall 与 in-context/流畅度**解离**」。
claude reviewer 声称「as of 2026-08, no paper explicitly reports 'perplexity heals while
factual knowledge lags' as the main finding」——**这个判断错了**，2310.04680 把
「fact recall 先坏、in-context 后坏」做成了**标题**，而且是 ICLR'24。

**我们剩下的差异化空间**（窄了很多，但非零）：
- 它做的是**静态 scaling/pruning 后**的能力比较，**没有 heal 轨迹**（no continued pretraining
  trajectory）。我们有 keep8/10/12/14/16 × 200k step 的**完整 heal 曲线** + 每 rung 的
  recovery-lag，这是它没有的时间维度。
- 它的对比轴是 fact-recall vs in-context；我们的是 **PPL vs fact-recall**（PPL 恢复到
  1.428× baseline 而 MMLU 只恢复 19.5%）。PPL 这一轴它没测。
⇒ D1 **不能按 claude 的框定写**（「首次报告解离」是错的）。只能写成
  **「解离在 heal 轨迹上的时间结构」**，且必须把 2310.04680 作为**正面对话对象**。

---

## ★★ 结论 2：claude 的 top-2 (D3「logit-lens onset 作剪枝准则」) 被三篇夹击，其中两篇 2026

### (a) arXiv:2605.11416 — "Freeze Deep, Train Shallow: Interpretable Layer Allocation for
Continued Pre-Training"（2026-05-12，**preprint**，Yu-Hang Wu et al.）

摘要原文：
> "we propose **LayerTracer**, an architecture-agnostic **diagnostic framework** that reveals
> the evolution patterns of layer-wise representations and stability by **locating task
> execution positions and quantifying layer sensitivity**. ... **deep layers act as critical
> regions for task execution and maintain high stability** ... we conduct **three controlled
> continued pre-training trials** to compare diverse freeze-train strategies, demonstrating that
> **training shallow layers while freezing deep layers consistently outperforms full-parameter
> fine-tuning** and the opposite allocation on both C-Eval and CMMLU"

**这几乎是 D3 的完整实现**：诊断框架 → 定位「task execution position」→ 据此决定 freeze/train
的层分配 → 在 continued pretraining 上做受控验证。而且**结论与我们的方向相反**
（它说"训浅层冻深层"更好；Paper B/C 是"保留浅层、重训顶部"）——这个反向结论本身就是
一个必须处理的强 baseline。

### (b) arXiv:2606.07978 — "MechLens: Late Crystallization of Factual Knowledge Explains
Intervention Effectiveness"（2026-06-06，**preprint**，Xueping Gao —— **与 2606.16897
CKA_Delta 同一作者**）

摘要原文：
> "We systematically quantify **Late Crystallization: factual knowledge does not gradually
> emerge across layers but 'crystallizes' abruptly at the final layers**. Across **five model
> families (Pythia, Gemma, Qwen2.5, Llama-3.1, Mistral; 0.5--14B)**, 26.8%--93.4% of correct
> answers never enter top-10 predictions at any intermediate layer, with **late emergence (>80%
> depth)** consistent across architectures. Cross-benchmark (**MMLU: 98.2%**) ... **tuned lens
> rules out probe artifacts**. ... We further reveal a **Computability-Memorization Spectrum**:
> computable knowledge crystallizes earlier (layer 22.1/28) than memorized facts (28.0/28)."

**这直接占掉 D3 的测量部分**：我们的资产是「OLMo-2 logit-lens knowledge onset 在 L18→L19
有 0.326→0.544 陡跳；Qwen3-8B 在 L24→L25 有 0.236→0.621 陡跳」——**它把同一现象命名为
Late Crystallization，跨 5 个家族量化，还用 tuned lens 排除了 probe artifact，并给了
MMLU 98.2% 的数字**。我们那两个陡跳点就是它说的 late emergence 的两个实例。
更糟：它还做了**crystallization-guided intervention**，即"据此指导干预"——这正是 D3
"onset 作准则"的应用侧。

### (c) arXiv:2607.25663 — "Localized Adaptation Reveals Distinct Learning Signatures in
Transformers"（2026-07-28，**preprint**，Rebecca Ramnauth, Brian Scassellati）

摘要原文：
> "We introduce a **controlled benchmark spanning five objectives** (lexical binding, factual
> association, behavioral policy learning, causal mapping, procedural reasoning) and define each
> objective's **'adaptation geometry'** ... under full-stack and **early-, middle-, or late-layer
> LoRA**. ... **lexical binding favors early-layer adaptation**; **factual association favors
> later layers** among localized adapters; ... These patterns **largely persist under
> parameter-matched controls**, and most directional contrasts **replicate across five model
> families**. These findings establish **adaptation site as a key design variable**."

**这占掉「不同能力偏好不同 adaptation 深度」这个命题**，而且做了我们没做的
**parameter-matched control + 5 家族复现**——正是 Paper C 被 reviewer 批"缺 lr/参数匹配控制"
的那一块，它做了。

⇒ **D3 判定：从 claude 的「novelty risk 2/5、top-2」下调到「基本被占」。**
「onset depth 是原则性剪枝准则」这句话现在需要同时打败 LayerTracer（分配策略）、
MechLens（onset 测量 + 干预）、Localized Adaptation（能力 × 深度偏好）。

---

## ★ 结论 3：claude 的 Q3（「无人报告过 CKA U-shape」）证据更弱了

**arXiv:2109.08406 — "Fine-Tuned Transformers Show Clusters of Similar Representations Across
Layers"**（Jason Phang, Haokun Liu, Samuel R. Bowman；venue-hint 自带 **BlackboxNLP 2021**）

摘要原文：
> "we use **centered kernel alignment (CKA)** ... to measure the similarity of representations
> in task-tuned models **across layers**. In experiments across twelve NLU tasks, we discover a
> consistent **block diagonal structure** in the similarity of representations within fine-tuned
> RoBERTa and ALBERT models, with **strong similarity within clusters of earlier and later
> layers, but not between them**."

**"earlier 和 later 层各自内部相似、彼此之间不相似"就是 U 型的另一种说法**（两端块内高、
跨块低）。区别：
- 它是 **within-model**（同一个 fine-tuned 模型的层 × 层），我们是 **cross-model**（A 的层 × B 的层）
- 它是 encoder (RoBERTa/ALBERT)，我们是 decoder-only 7 家族
- 它没有 shuffle-null / distance-residual 控制

⇒ **U-shape 的"形状"本身在 CKA 文献里 2021 年就有近邻**（block-diagonal）。
claude 说的"没人报告过"**只在"cross-model depth-diagonal + null 校准"这个严格意义上成立**。
D5 若要写，**必须** cite 2109.08406 并明说边际贡献是 cross-model + null-calibrated，
不能声称"首次发现中层不可对齐"。

**另外两篇同域的**（claude 都没提）：
- `2312.02730` "Towards Measuring Representational Similarity of Large Language Models"
  （Klabunde et al., venue-hint: **UniReps Workshop @ NeurIPS 2023** extended abstract）——
  测 7B LLM 之间的表征相似度，并**明确警告 representational similarity measures 会导致
  false conclusions**。这是 D5 方法论上的直接前作。
- `2410.06981` "Quantifying Feature Space Universality Across LLMs via Sparse Autoencoders"
  （preprint）——用 SAE 做跨模型特征空间通用性，"**high similarities for SAE feature spaces
  across various LLMs**"。它是"跨模型表征是否通用"的另一条技术路线。

---

## ★ 结论 4：Paper D stitching 判死是对的，但漏了一篇给出**反向证据**的

**arXiv:2601.13580 — "Neural Organ Transplantation (NOT): Checkpoint-Based Modular
Adaptation for Transformer Models"**（2026-01-20，**preprint**，Ahmad Al-Zuraiqi）

摘要原文：
> "NOT extracts **contiguous layer subsets ('donor organs')** from pre-trained models, trains
> them independently on domain-specific data, and saves them as **standalone checkpoint files
> that can be transplanted into compatible recipient models**. Through experiments on **three
> decoder-only architectures spanning 124M to 20B (GPT-2, TinyLlama, GPT-OSS)**, we demonstrate
> that donor transplantation substantially outperforms existing adaptation methods, achieving an
> **order-of-magnitude improvement in perplexity over LoRA** while training significantly faster.
> The method exhibits **position dependence, with early insertion positions yielding optimal
> results**. ... **transformer middle layers can support efficient modular transfer for
> decoder-only architectures**"

**这与我们的 stitching 判死结论直接冲突**：我们实测跨家族 oracle affine ppl 596 vs 原 18.8
（差 32×），结论"1-2 层桥接不够"；它声称层子集移植能**比 LoRA 好一个数量级**，且
"middle layers can support efficient modular transfer"。

**但看清适用条件**（这是关键）：它写的是 "transplanted into **compatible** recipient models"，
且实验是 GPT-2 / TinyLlama / GPT-OSS —— **摘要没有声明跨家族**（"compatible" 很可能意味着
同架构/同 tokenizer）。而且它的 finding 是 "**early** insertion positions yielding optimal
results" —— **早层最优**，这与我们 R4 的 H1（中层最不可对齐）**方向一致**，不冲突。
⇒ 它反而**支持**我们的 U 型结论：能移植的位置在两端而非中间。
⇒ 但**必须引**，且必须说清"我们测的是跨 tokenizer 跨家族，它测的是 compatible recipient"。

---

## MAIN 修订后的方向评分（覆盖 claude 的矩阵）

| 方向 | claude 打分 | **MAIN 修订** | 变化原因 |
|---|---|---|---|
| D1 PPL-知识三向解离 | novelty 2, **top-1** | **novelty 4（差）**，降为「必须重框定」 | 2310.04680 ICLR'24 已把解离做成标题 |
| D2 post-norm 解释 CKA outlier | novelty 4（差），blocked | **仍 blocked** | 2606.16897 占 outlier detector；同作者 2606.07978 更强 |
| D3 onset 作剪枝准则 | novelty 2, **top-2** | **novelty 4-5（很差）** | 2605.11416 + 2606.07978 + 2607.25663 三面夹击 |
| D4 token-budget-to-recovery 律 | novelty 2 | **novelty 2-3（暂时最好）** | 无人做 post-pruning recovery 的 token-budget 律；2310.04680 无轨迹 |
| D5 CKA U-shape 短文 | novelty 3 | **novelty 3（不变），但边际贡献须重写** | 2109.08406 block-diagonal 是 2021 年近邻；须显式 cite |

**MAIN 的当前判断（待 tcodex 终稿或第二轮核实修正）**：

**唯一还没被占的窄格是 D4 的时间维度** —— 不是"解离存在"（已被 ICLR'24 占），
不是"onset 在哪"（已被 MechLens 占），不是"哪层该训"（已被 LayerTracer 占），
而是 **「结构损伤后，各能力恢复所需的 token 预算各不相同，且该预算随继承深度分数呈规律变化」**。
我们有唯一的资产：**keep8/10/12/14/16 × 200k step 全轨迹 + 5 项 eval 阵列**，
这是上述所有论文都没有的**时间轴**。

**但这个判断有一个尚未排除的 kill risk**：我还没搜过「post-pruning recovery scaling law」
这个精确表述。117 篇里有 `2402.04177`（Scaling Laws for Downstream Task Performance）、
`2401.05605`（Scaling Laws for Forgetting When Fine-Tuning）、`2407.17467`（CMR Scaling Law
for Continual Pre-training, EMNLP'24 main）——**`2407.17467` 需要立刻精读**，它是
continual-pretraining 的 mixture-ratio scaling law，可能已覆盖 recovery-budget 这一侧。

---

## 待办（MAIN 自己，不派 subagent）

1. **精读 `2407.17467` CMR Scaling Law 摘要** —— D4 的最大 kill risk。
2. tcodex 若出终稿，比对它的 top-1 是否也收敛到 D4-like 的时间维度。
3. **117 篇里还有 ~30 篇没看摘要**（early-exit 系、layer-skip 系、model-editing 系、
   continual-learning 系），其中 `2510.18871`（How Do LLMs Use Their Depth?）的
   "Guess-then-Refine" + "MC 任务前半选项后半定稿" 与我们 content-j 的发现相邻，需要看。
4. 所有 2026 preprint 进 .bib 前**必须**标 preprint（本文已标）。
