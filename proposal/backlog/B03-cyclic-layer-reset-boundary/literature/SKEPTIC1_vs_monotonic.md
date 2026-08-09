# SKEPTIC 1 — 反驳 AUDIT1「大→小→大的循环 depth 构造未被单调生长文献占掉」

审计日期 2026-08-06 · 角色：adversarial skeptic（默认 AUDIT1 的主张是错的，去找证据推翻它）
审计对象：`paperC_v2_research/AUDIT1_monotonic_growth.md`
方法：全文抓取（arxiv.org/html 优先，ar5iv 兜底，`/tmp/skep1/html/*.txt`）+ arXiv API（**https** 通路）+ S2 Graph API venue（重试到 http200）+ **S2 forward-citation 扫描**（AUDIT1 自己承认没做的那一步）

---

## 0. 结论（一句话）

# **WEAKENED**（且是重度收窄 —— 距离 REFUTED 只差「有人在 LLM 规模上做过」这一条）

AUDIT1 的**字面主张**（「大→小→大、终点回到原尺寸的循环，没有一篇做过」）在**LLM 预训练场景下勉强成立**，
但它赖以支撑该主张的**四根支柱有三根被推翻**，而且**它漏掉了一篇 ICLR 2022 的 peer-reviewed 论文
（LLF / Fortuitous Forgetting, arXiv 2202.00155），该论文的算法与我们的构造在数学上等价**。

**致命之处：AUDIT1 把「窄缝」定义在四个支柱的合体上（§4「四个必须同时成立」），
但 LLF 一篇就同时满足全部四个支柱 —— 层级、循环、终点等尺寸、破坏式随机 reset、目的是最终质量。
AUDIT1 的判决表格（§4「四格全占的组合不存在」）里三列（CGLS / DSD / SOLAR DUS）本该是四列，
缺的那一列 LLF 会把所有格子填满。**

我没有把它判为 REFUTED，唯一理由：LLF 是 **ResNet/CIFAR/Flower/Tiny-ImageNet 图像分类 + 语言涌现 Lewis game**，
**不是 LLM 预训练，不涉及 transformer decoder，不涉及知识/MMLU**。这个 domain gap 是真的，但它是
「**同一算法换个 domain**」类型的增量，不是 AUDIT1 声称的「**构造本身无人做过**」。
Reviewer 会说：this is LLF applied to LLM pretraining。这句话我无法反驳。

---

## 1. 主打击：AUDIT1 漏掉的那篇（ICLR 2022，peer-reviewed）

### 1.1 论文与 venue（双通道核实）

| 项 | 值 |
|---|---|
| arXiv ID | **2202.00155** |
| 标题 | Fortuitous Forgetting in Connectionist Networks |
| 方法名 | **LLF = Later-Layer Forgetting**；上层框架名 **forget-and-relearn** |
| arXiv COMMENT | `ICLR Camera Ready` |
| arXiv Journal-ref | **`ICLR 2022`** |
| S2 Graph API | `venue: 'International Conference on Learning Representations'`，`publicationVenue.type: conference`，`year: 2022`，`citationCount: 50` |
| 判定 | **ICLR 2022，peer-reviewed 主会。不是 preprint。** |

> 抓取产物：`/tmp/skep1/html/2202.00155.txt`（ar5iv，89324 chars 全文）、`/tmp/skep1/abs/2202.00155.html`、`/tmp/skep1/venue/v_2202.00155.json`

### 1.2 LLF 的算法（原文句子 + section）

**§3.2（提出 LLF，紧接 Baldock et al. prediction-depth 讨论之后）原文**：

> "Based on the observations in Baldock et al. (2021), we hypothesize that **by reinitializing the later layers of the neural network, we can remove information associated with difficult examples** more precisely than the mask criteria used in KE. Thus, we propose a new forgetting procedure called **later-layer forgetting (LLF)**. Given a layer threshold L, we define the LLF mask criterion for each layer l as:
> M^l_LLF = { 1 if l < L ; 0 if l ≥ L }"

**§2「Existing Algorithms as Instances of Forget-and-Relearn」的术语定义原文（脚注 3）**：

> "In this work, we use **'rewind' to mean resetting weights back to their original initialization. We use 'reset' and 'reinitialize' interchangeably to indicate a new initialization.**"

→ **LLF 的 reset 是 a NEW initialization（真随机新初始化），不是 rewind 到 step-0 权重。**
这一条直接击穿 AUDIT1 §3 对 lottery-ticket 系「rewind 不是随机」的划界逻辑 —— LLF 恰恰是随机的。

**§5.1 每一代（generation）反复做，且原文明确「每代用不同的新初始化」是必要条件**：

> "...except we keep the later layers **frozen at the same initialization in each generation**. This reduces the amount of variability seen during iterative retraining. As shown in Figure 3c under 'freeze fixed later layers', we find the performance of this to be **much worse than the version with a different reinitialization each generation**, demonstrating the importance of having variable conditions for relearning."

→ **不仅是循环（多 generation），而且论文实测证明「每轮必须是新的随机初始化」才有效**（把「每代都换新随机」和「固定同一个随机」拆开做了 ablation）。这正是我们构造里「补 K 个随机新层」的核心机制，而且**已经有人做过对照实验并给出结论**。

**§4 Table 1 caption（规模与轮数）**：

> "LLF uses L ∈ {10, 14}, corresponding to block 3 and 4 in ResNet18. **N3, N8, N10 indicate the additional number of training generations on top of the baseline model.** LLF consistently outperforms all other methods."

→ **N=3 / 8 / 10 轮循环**，不是单轮。

**§A（Extension to Larger Datasets）**：

> "We train ResNet50 on Tiny-ImageNet ... and **reset layers starting from the third block (L = 23)** during LLF. As illustrated in Table A7, we find LLF to outperform the baselines on Tiny-ImageNet."

### 1.3 为什么这与我们的构造是**数学等价**（不是"类似"）

我核对了我们自己的实现 `/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/scripts/train_olmo2_arch_probe2.py`：

- 文件头 docstring（line 9-11）：`keep the FRONT keep_front_layers decoder layers + ... DROP the top layers, append n_fresh_layers FRESH Olmo2-init`
- `transplant_front()`：只 copy 前 `keep_front` 层 + embed/norm/lm_head，`assert missing_layer_ids == expected_fresh_ids`（= `range(keep_front, keep_front+n_fresh)`），即**顶部 K 层保持随机初始化**
- 用户 prompt 自己写的关键能力：「令 n_fresh = 原深度 减 keep_front 即可保持总深度不变（丢 K 层 + 补 K 个随机新层）」

**在总深度不变的前提下，「丢弃顶部 K 层 + 在顶部补 K 个随机初始化层」与「把顶部 K 层原地随机重初始化」
是同一个操作**：两者都得到一个 L 层网络，其中前 L−K 层是继承权重、后 K 层是全新随机权重，且拓扑相同。
唯一差别是实现路径（新建 module vs 原地 re-init），**不构成任何函数层面/优化层面的差异**。

**这正是 LLF 的 `M^l_LLF = {1 if l<L, 0 if l≥L}`。**

→ 所以我们的构造在**算子层面**已经在 ICLR 2022 出现过，并且是该论文的**主方法**（不是附录 ablation）。

### 1.4 AUDIT1 的四支柱逐条被 LLF 满足

AUDIT1 §4「这条缝的四个『必须同时成立』的支柱，缺一个就掉进已有工作」：

| AUDIT1 的支柱 | LLF 是否满足 | 证据 |
|---|---|---|
| 1. **丢弃（destructive）而非 bypass**（否则 = RaPTr） | ✅ **满足** | §3.2 mask 把 l≥L 的层置为 new initialization；脚注 3 定义 reset = a new initialization。权重被销毁，不是跳过 |
| 2. **多轮循环而非单程**（否则 = CGLS/SOLAR） | ✅ **满足** | Table 1: N3/N8/N10 generations；§5.1 反复 "each generation" |
| 3. **终点等尺寸**（否则 = 生长/剪枝文献） | ✅ **满足** | LLF 全程同一个 ResNet18/50 架构，从不改变层数或参数量 |
| 4. **层级而非权重级**（否则 = DSD） | ✅ **满足** | 名字就叫 later-**LAYER** forgetting，mask 的索引 l 是**层**索引，不是权重索引 |
| （AUDIT1 §4 额外要求）目的是最终质量/塑性而非省 FLOPs/压缩 | ✅ **满足** | Abstract: "forgetting can in fact be favorable to learning"；§5.1 两个假设都关于 generalization；SEAL 转述 LLF 为 "typically rely on **training for longer** periods of time **in exchange for improved generalization**" —— 明确是**花更多算力换质量**，与省 FLOPs 相反 |

**五格全占。AUDIT1 §4 的「四格全占的组合不存在」是错的。**

### 1.5 LLF 还先手做掉了 AUDIT1 建议我们做的两个 baseline 臂

AUDIT1 §5 第 2 条建议我们预注册四个 baseline 臂来隔离变量。其中两个 LLF 已经做过：

- AUDIT1 建议的 **(c) delete-and-copy 臂（隔离"随机性"变量）** → LLF §5.1 的 "freeze fixed later layers"（每代同一个初始化）vs "different reinitialization each generation" 已经隔离了「随机性/变化性」这个变量，结论：**变化性是必要的**。
- AUDIT1 建议的**「reset 早层 vs 晚层」方向性对照** → LLF §5.1 "Analysis of LLF" 已做：`"instead of reinitializing the later layers, we can reinitialize the earlier layers ... We see in Figure 3b that the reverse experiments indeed perform worse than both LLF and the long baseline."`

→ 我们如果照 AUDIT1 的方案做，会**重跑 ICLR 2022 已发表的 ablation**。

---

## 2. 第二打击：LLF 的直接 follow-up 也是 peer-reviewed，且 AUDIT1 把它误读成了无关论文

AUDIT1 §6 搜索记录里 `abs:"re-initialize" AND abs:"layers" AND abs:"training"` 命中过
`2304.04858 Simulated Annealing in Early Layers`，但 AUDIT1 只在表格里一笔带过（列为「无冲突」），
**没有意识到这篇的整个 framing 就是「打败 LLF」**。

| 项 | 值 |
|---|---|
| arXiv ID | **2304.04858** |
| 标题 | Simulated Annealing in Early Layers Leads to Better Generalization（SEAL） |
| S2 venue | `Computer Vision and Pattern Recognition`，`type: conference`，`year: 2023`，DBLP `conf/cvpr/SarfiKCKRMB23`，DOI `10.1109/CVPR52729.2023.01935` |
| 判定 | **CVPR 2023，peer-reviewed** |

**Abstract 原文**：

> "Recently, a number of **iterative learning methods** have been introduced to improve generalization. These typically rely on training for longer periods of time in exchange for improved generalization. **LLF (later-layer-forgetting) is a state-of-the-art method in this category. It strengthens learning in early layers by periodically re-initializing the last few layers of the network.** Our principal innovation in this work is to use Simulated annealing in EArly Layers (SEAL) of the network in place of re-initialization of later layers."

**Figure 1 caption 原文**：

> "Our iterative training method (SEAL) compared to LLF. ... **LLF re-initializes the top layers right before a new generation begin.**"

→ 两点后果：
1. **「周期性重初始化最后几层」在 CVPR 2023 被称作 state-of-the-art method，是一个有名字、有 SOTA 地位的既有类别**（"iterative learning methods"）。AUDIT1 声称这个动作空间在层级+循环+等尺寸维度是空的，与此矛盾。
2. SEAL 还给出一个**对我们不利的负面结论**（Abstract）：`"we further show that, compared to normal training, LLF features, although improving on the target task, degrade the transfer learning performance across all datasets we explored."` → **LLF 式的 later-layer reset 会损害迁移能力**。我们的场景（LLM 预训练后要看 MMLU/知识/下游）本质上就是 transfer/general capability，这条负面证据必须回答。

---

## 3. 第三打击：AUDIT1 对它**读过**的 DSD 有实质误读（自证矛盾）

AUDIT1 §3 表格 + §4 判决反复强调 DSD「**不是多轮循环**」：

- §3：「**是，但只一轮 D→S→D**（不是 N 轮循环）」
- §4 表格：DSD 的「循环？」列打 **✗（单轮）**
- §4：「**但粒度完全不同（权重 vs 层）、且不是多轮循环**、且补回方式是 zero-init 而非随机新层」

**这是错的。DSD 原文自己写了迭代版本。**

我抓了 DSD 全文（`/tmp/skep1/html/1607.04381.txt`，ar5iv 215KB）：

**Algorithm 1「Workflow of DSD training」最后一行原文**：
> "Final Dense Phase ... while not converged do ... end while  **goto Sparse Phase for iterative DSD;**"

**§4.3 ResNet 原文**：
> "We prune to 30% sparsity uniformly, and **a single DSD pass** for these networks reduced top-1 error by 1.13% (ResNet-18) and 0.85% (ResNet-50), shown in Table 4. **A second DSD iteration can further improve the accuracy.**"

→ DSD 的 Algorithm 1 里**显式有回到 Sparse Phase 的循环指令**，并且实验里**跑了第二轮并报告有进一步提升**。
AUDIT1 用「DSD 只有单轮」把「多轮循环」当成我们的支柱 2，**这个支柱在权重级上也不成立**。

**连带影响**：AUDIT1 §5 R1 说「必须做的：给出层级 ≠ 权重级的实质论证」——
现在情况更糟：DSD 已经是「多轮 + 等尺寸 + 破坏后复原 + 目的是最终质量（regularization）」，
我们**只剩「层级」一个差异维度**（而层级那一格被 LLF 占了）。

---

## 4. 第四打击：AC/DC（NeurIPS 2021）—— AUDIT1 完全没搜到的「多轮循环 + 可交付等尺寸终点」

| 项 | 值 |
|---|---|
| arXiv ID | **2106.12379** |
| 标题 | AC/DC: Alternating Compressed/DeCompressed Training of Deep Neural Networks |
| arXiv COMMENT | **`Accepted at NeurIPS 2021`** |
| 判定 | **NeurIPS 2021，peer-reviewed**（S2 查询遇 429 未取到，按项目规矩以论文 COMMENT 自述为准并注明 S2 未核到） |

AUDIT1 §6 跑了 `abs:"alternating" AND abs:"pruning" AND abs:"growing" AND abs:"neural"`（8 命中）和
`abs:"dense-sparse-dense"`（6 命中），**都没命中 AC/DC**，因为 AC/DC 的措辞是 compressed/decompressed 而非 sparse/dense。
我用 `abs:"alternating" AND abs:"compressed" AND abs:"decompressed"` 一次命中。

**§3.2 原文（调度）**：
> "We partition the set of training epochs into compressed epochs C, and decompressed epochs D. We begin with a dense warm-up period of Δ_w consecutive epochs ... **We then start alternating compressed optimization phases of length Δ_c epochs each, with decompressed (regular) optimization phases of length Δ_d epochs each.**"

**§4 实验设置原文**：
> "the AC/DC training schedule starts with a 'warm-up' phase of dense training for 10 epochs, **after which we alternate between compression and de-compression every 5 epochs**, until the last dense and sparse phase."

**终点可以是等尺寸 dense 模型（§3.2 原文）**：
> "Alternatively, **if our goal is to return a dense model matching the baseline accuracy, we take the best dense checkpoint obtained during alternation, and fine-tune it over the entire support.** ... allowing a longer final decompressed phase of length Δ_D > Δ_d improves the performance of the dense model."

**并且它也 reset 优化器状态（§3.2 原文）**：
> "once all weights are re-introduced, **it is beneficial to reset to 0 the gradient momentum term of the optimizer**; this is particularly useful for the weights that were previously pruned, which would otherwise have stale versions of gradients."

→ **AC/DC = 多轮循环（每 5 epoch 交替，跑满整个训练）+ 破坏（top-k 置零）+ 复原到原尺寸 + 可交付 dense 终点 + reset 优化器状态。**
除了「粒度是权重不是层」，其余全中。AUDIT1 §4 表格里 DSD 那一行的「循环 ✗」和「终点等尺寸 ✓」，
在 AC/DC 这里是**「循环 ✓」+「终点等尺寸 ✓」**——AUDIT1 §4 的三列表格因此彻底失效。

**附带**：`2202.01290 Cyclical Pruning for Sparse Neural Networks`（arXiv COMMENT 无、Journal-ref 无 → **preprint**）
也是权重级循环 prune-regrow，AUDIT1 未命中。

---

## 5. 第五打击：AUDIT1 §5 R4 的「LLM 预训练侧塑性是空白」是错的（三篇反例，其中两篇 peer-reviewed）

AUDIT1 §5 R4 + §6 搜索表连续两处断言：
- 「那一支目前**几乎全在 RL 领域**（见 §6 搜索记录：q_lop_reset 命中的 7 篇全是 RL/bandit），**LLM 预训练侧是空白**」
- §6 表：`abs:"loss of plasticity" AND abs:"reset"` → 「**全部是 RL / bandit，无一是 LLM 预训练**」；`abs:"layer" AND abs:"plasticity" AND abs:"language model"` → 「空白确认」

**这三篇直接推翻它**：

### 5.1 Active Forgetting（NeurIPS 2023）—— 最致命，因为它就是「LLM 预训练里周期性 reset 一个模块来买塑性」

| 项 | 值 |
|---|---|
| arXiv ID | **2307.01163** |
| 标题 | Improving Language Plasticity via Pretraining with Active Forgetting |
| arXiv COMMENT | `NeurIPS 2023 Final Version` |
| S2 | `venue: 'Neural Information Processing Systems'`，`type: conference`，`year: 2023`，`cites: 48`，DBLP `conf/nips/ChenMRAS0A23` |
| 判定 | **NeurIPS 2023，peer-reviewed** |

**Abstract 原文**：
> "We propose to use an **active forgetting mechanism during pretraining**, as a simple way of creating PLMs that can quickly adapt to new languages. Concretely, by **resetting the embedding layer every K updates during pretraining**, we encourage the PLM to improve its ability of learning new embeddings within limited number of updates, similar to a meta-learning effect."

**§3 Our Method 原文**：
> "the forgetting mechanism operates by **intentionally clearing the weights of the embedding layer** ... and **reinitializing them to a new set of random values every K gradient updates**. Since pretraining involves advanced training strategies, like optimizers with states and learning rate schedulers, **we also reset them together with the token embedding layer**."

**§3 Figure 4 caption 原文（这段的叙事跟我们想写的一模一样）**：
> "The forgetting mechanism brings an **episodic pattern** into the loss curve: every embedding forgetting produces a **loss spike, from which the model learn to recover. Through such repeats of forget-relearn**, the model gets used to learn new embeddings from scratch."

**§3 原文（机制解释）**：
> "Each forgetting event kind of 'branches out' a novel environment for the model to explore, as if initiating a new episode of learning."

→ **「LLM 预训练过程中周期性把某个层随机重初始化，反复 forget-relearn，制造 loss spike 再恢复，目的是买塑性」
已经在 NeurIPS 2023 做过并命名。** 我们的构造与它的差别只有一个：**被 reset 的是 top-K decoder 层，而不是 embedding 层。**
这是一个「换 reset 目标模块」的差异，reviewer 说 minor variant 我很难反驳。

（诚实标注：它的塑性定义是 cross-lingual 适应的 sample/compute efficiency，不是"最终能力天花板"；模型是 RoBERTa-base 12 层，非 7B 规模。这是我们仅剩的差异空间，见 §8。）

### 5.2 Can Scale Save Us From Plasticity Loss in LLMs?（preprint）

| 项 | 值 |
|---|---|
| arXiv ID | **2606.24752** |
| 标题 | Can Scale Save Us From Plasticity Loss in Large Language Models? |
| 作者/机构 | Hernandez-Garcia, Figliolia, Millidge（Zyphra） |
| arXiv COMMENT | 无；Journal-ref 无 |
| 判定 | **arXiv preprint（显式标注）** |

**Abstract 原文**：
> "Although this phenomenon has been known for decades, it has mostly been studied in older, relatively small architectures and rarely in natural-language domains. **To determine whether loss of plasticity remains a problem in the modern transformer-based LLM paradigm, we study plasticity loss in GPT-style Transformer models** trained on a multilingual continual learning problem. Consistent with prior work, **we find evidence of plasticity loss across models ranging from 5M to 314M non-embedding parameters** ... **the onset of plasticity loss follows a predictable scaling law, growing sublinearly with model size.**"

→ **LLM 侧的 plasticity-loss 度量已经有人做了**，并且做了 dormant unit（全文 22 次）、lazy/collapsed attention head、参数 magnitude 等 correlate。
AUDIT1 §5 R4 建议我们「借 RL 侧标准量（dormant unit ratio / effective rank）填 LLM 空白」——这个空位已被占。

**更棘手（§V-B / 结论段原文）**：该文把 reset 类方法列为**已知的 mitigation 家族**：
> "Continual Backpropagation measures the activity of units and **periodically reinitializes units with low average activity** (Dohare et al., 2021, 2024). Other algorithms, such as ReDo (Sokar et al., 2023), Self-Normalized Resets, and GraMa, **also reinitialize neurons** but use different criteria for selecting which units to reset."

→ 「周期性重初始化组件以维持塑性」在 LLM 语境里已经是**综述里的一个标准家族**，不是空白。

### 5.3 Weight Decay Improves Language Model Plasticity（自述 ICML 2026）

| 项 | 值 |
|---|---|
| arXiv ID | **2602.11137** |
| 作者 | Tessa Han, Sebastian Bordt, Hanlin Zhang, Sham Kakade |
| venue 线索 | 被 2606.24752 引作 `In Proceedings of the 43rd International Conference on Machine Learning, ICML '26`；该文自身 arXiv COMMENT / Journal-ref **均为空** |
| 判定 | **仅第三方引用自述 ICML 2026；S2 未核到（429）→ 保守标「疑 ICML 2026，需 MAIN 复核」** |

**Abstract 原文**：
> "hyperparameter optimization and scaling laws are studied primarily from the perspective of the base model's validation loss, overlooking a crucial model property: downstream adaptability. In this work, **we study pretraining from the perspective of model plasticity**"

→ **「把 LLM 预训练超参当作『最终塑性』的自变量来研究」这个 research question 已被占**，而且是我们打算讲的同一个故事骨架（pretraining choice → downstream adaptability，且明确说 val loss 不够）。它选的自变量是 weight decay，我们选的是结构 reset。

（补充：`2603.20860 Restoring Neural Network Plasticity for Faster Transfer Learning`，Journal-ref = `SACAIR 2025, CCIS vol 2784, Springer` → **peer-reviewed workshop/conference**，做 reinitialization × utility/pruning function 的系统对照，含 ViT-B16。层级 reset 的方法空间在 vision 侧被扫过。）

---

## 6. 我核对了 AUDIT1 引的句子：哪些是准的（给它公道）

我 refetch 了 AUDIT1 的两篇关键引用做原文核对，**这两篇它引得准，没有脑补**：

| AUDIT1 的引用 | 核对结果 |
|---|---|
| RaPTr（2402.05913）§2 P1 `maintain a common base model of interest` | ✅ **原文确认**（全文 1 次命中，位置正是 (P1) 定义处）。上下文亦确认 AUDIT1 的解读正确 |
| RaPTr §3 `the depth variant of RaPTr, that drops fewer layers as training proceeds` | ✅ **原文确认**（全文 1 次命中）。且原文紧接 `Recall that PLD drops more layers as training proceeds` → AUDIT1「方向与 PLD 相反、是单调、不是循环」的判断**正确** |
| AUDIT1 对 RaPTr「bypass ≠ delete」的判决 | ✅ 成立（`bypass` 4 次命中，`random subnetwork` 15 次，权重确实留在 base model 里） |

→ **AUDIT1 对「生长文献」这一支的工作是扎实的**，它的失败不在读错读过的东西，而在**检索关键词全都围着 grow/prune/expand/shrink 打转，
从没搜 `reinitialize + later layers`、`forget-and-relearn`、`compressed/decompressed`、`plasticity + LLM`**
——也就是它自己在 §7「盲区」第 3 条承认没做的**forward-citation 扫描**，恰恰是命门。
我做了这一步（S2 `paper/arXiv:2202.00155/citations`，50 篇），立刻捞出 SEAL(CVPR23)、Active Forgetting(NeurIPS23)、
Reset It and Forget It(ECAI 2024)、Reinitializing weights vs units(CoLLAs 2025)。

---

## 7. 我搜过的 query 清单（证明我真的试过 —— 也证明「LLM 规模」那格确实还空着）

arXiv API（**https**://export.arxiv.org，`sortBy=relevance`；原始 XML 在 `/tmp/skep1/search/*.xml`）。
⚠️ 纠正 AUDIT1 §7 与项目旧笔记：**http 通路在本机经代理返回 0 字节，https 通路正常**（实测 http=0 / https=2812 bytes）。

**命中并成为打击点的 query（★）：**
| query | 命中 | 结果 |
|---|---|---|
| ★ `all:"fortuitous forgetting"` | 1 | **2202.00155 LLF, ICLR 2022** → §1 主打击 |
| ★ `abs:"later layer" AND abs:"re-initialize"` | 1 | 2304.04858 SEAL → §2 |
| ★ `all:"later layer forgetting"` | 1 | 2304.04858 SEAL |
| ★ `abs:"alternating" AND abs:"compressed" AND abs:"decompressed"` | 25(取10) | **2106.12379 AC/DC, NeurIPS 2021** → §4 |
| ★ `abs:"loss of plasticity" AND abs:"language model"` | 2 | **2606.24752** + 2605.12484 → §5.2 |
| ★ `ti:"Weight Decay" AND abs:"plasticity"` | 2 | **2602.11137** → §5.3 |
| ★ `ti:"plasticity" AND abs:"pretraining" AND abs:"transformer"` | 2 | 2603.20860（SACAIR 2025） |
| ★ `abs:"cyclical pruning"` | 3 | 2202.01290（preprint）、COLT 2212.12770 |
| ★ `abs:"reinitializing" AND abs:"layers" AND abs:"generalization"` | 6 | **2109.00267**（layerwise reinit，见下）|
| ★ `abs:"knowledge evolution" AND abs:"reset hypothesis"` | 1 | 2103.05152 KE（LLF 的前作/baseline） |

**S2 forward-citation 扫描（AUDIT1 未做的关键一步）：**
| 调用 | 结果 |
|---|---|
| `paper/arXiv:2202.00155/citations?limit=100` (http200，50 篇) | 捞出 SEAL(CVPR23)、**Active Forgetting(NeurIPS23)**、Reset It and Forget It(ECAI 2024)、Reinitializing weights vs units(CoLLAs 2025, arXiv 2508.00212)、Sample-efficient LLM Optimization with Reset Replay(2508.06412)、Layer-Wise Unlearning for Model Adaption |

**我搜过但 0 命中 / 无威胁的 query（这些支撑「LLM 规模那格还空着」）：**
| query | 命中 | 判定 |
|---|---|---|
| `abs:"reinitialize" AND abs:"large language model" AND abs:"pre-training"` | **0** | LLM 侧空 |
| `abs:"layer" AND abs:"reset" AND abs:"plasticity" AND abs:"pretraining"` | **0** | 空 |
| `abs:"layer" AND abs:"reinitialization" AND abs:"pretraining" AND abs:"transformer"` | 1（federated LoRA ViT） | 无威胁 |
| `abs:"iterative" AND abs:"prune" AND abs:"heal" AND abs:"large language model"` | **0** | 空 |
| `abs:"drop" AND abs:"layers" AND abs:"append" AND abs:"random" AND abs:"depth"` | **0** | 直击构造，空 |
| `abs:"layer" AND abs:"swap" AND abs:"fresh" AND abs:"pretraining"` | **0** | 空 |
| `abs:"generations" AND abs:"reinitialize" AND abs:"decoder" AND abs:"language"` | 1（diffusion decoding） | 无 |
| `abs:"forget" AND abs:"relearn" AND abs:"language model" AND abs:"pretraining"` | 2（unlearning / debiasing） | 无 |
| `abs:"forget" AND abs:"relearn" AND abs:"layers"` | 1（continual unlearning） | 无 |
| `abs:"repeatedly" AND abs:"replace" AND abs:"blocks" AND abs:"training"` | 1（looped LM iso-depth） | 无 |
| `abs:"prune-regrow" OR abs:"prune and regrow"` | 4（全 CNN/3D 压缩） | 无 |
| `abs:"periodic" AND abs:"reset" AND abs:"language model" AND abs:"training"` | 6（Elastic Reset(RLHF)、RL prolonged training 等） | 无（均非预训练结构 reset） |
| `abs:"iterative" AND abs:"reinitialization" AND abs:"forgetting"` | 1（federated） | 无 |
| `abs:"dormant" AND abs:"neurons" AND abs:"transformer"` | 1（理论） | 无 |
| `abs:"layer" AND abs:"regrowing" AND abs:"depth"` / `ti:"Layerwise" AND abs:"reinitialization"` / `abs:"born again" AND abs:"layer" AND abs:"reset"` / `abs:"cyclic" AND abs:"precision" AND abs:"reinitialize"` / `abs:"layer" AND abs:"recycling" AND abs:"pretraining"` | 0-3 全无关 | 无 |

**顺带一篇 AUDIT1 也漏的（层级 reinit 的系统性研究）**：
`2109.00267 The Impact of Reinitialization on Generalization in CNNs`（S2 venue `arXiv.org` → **preprint**；被 LLF §3.2 称为 concurrent work）。
它的 **`lw` (LayerWise) 算法 Algorithm 1 第 12 行原文**：`"Reinitialize all layers above block k"`，
第 13 行 `"Fine-tune the entire model until convergence"`，外层 `for k ∈ (1..K) / for n ∈ (1..N)` **双层循环**。
§1 原文亦把 dsd 归类为：`"This can be interpreted as a generalization to the sparse-dense-sparse (dsd) workflow of (Han et al. 2017) in which reinitialization occurs only once."`
→ **「层级 + 多轮 + 等尺寸 + 随机 reinit + 目的是 generalization」在 2021 年就有一篇系统对比 12 个数据集 × 多架构的论文。**

**我诚实声明的剩余盲区**：
1. arXiv `abs:` 只搜摘要；若某 LLM tech report 把「循环 reset 层」写在附录训练细节里，我搜不到。
2. 未覆盖 OpenReview 在审/撤稿、ICLR 2027 投稿、公司技术报告。
3. S2 forward-citation 我只扫了 LLF（50 篇上限），**未扫 AC/DC、Active Forgetting、2109.00267 的 citing papers** —— 这三个是下一轮最该扫的地方，很可能藏着 exact match。
4. 未做非英文渠道。

---

## 8. 收窄后的精确表述（WEAKENED 的具体含义）

AUDIT1 §4 的表述**必须废弃**：

> ~~「在层级（不是权重级）粒度上，反复执行「丢弃 K 层 + 补 K 个随机初始化层」…… 且评价标准是最终模型的能力/塑性」~~

**这个表述整体已被 LLF (ICLR 2022) + 2109.00267 占据**，只是在 CNN/图像分类上。

**能活下来的表述（我尽力压到最窄）**：

> 在 **decoder-only LLM 的大规模预训练/续训**（7B 量级、Dolmino/DCLM 级 token 预算）上，
> **周期性地把 top-K transformer 层随机重初始化（等价于丢 K 层 + 补 K 随机层，总深度不变）**，
> 并把评价落在 **知识型能力（MMLU / closed-book QA）与 PPL 的分离**上 ——
> 检验 LLF 类「破坏式 forget-and-relearn 课程」在**知识密集的 LM 预训练**中是否仍然有效，
> 还是会被我们 Paper B 已量化的「PPL 能 heal 但知识不能（1.428× vs 19.5%）」所吞掉。

这个缝的四个支柱要**全部换掉**（旧四支柱已失效）：

1. **规模与 domain**：7B decoder-only LLM 预训练 ≠ LLF/SEAL/lw 的 ResNet 图像分类；≠ Active Forgetting 的 RoBERTa-base 12 层
2. **被 reset 的模块**：top-K **decoder 层** ≠ Active Forgetting 的 **embedding 层**（后者不碰 transformer body，前者销毁 body 的上半部）
3. **评价维度**：**知识/事实型能力**（MMLU、PopQA/TriviaQA/NQ-open）≠ LLF 的 image accuracy ≠ Active Forgetting 的 cross-lingual 适应速度
4. **预期结论方向可能是负的**：SEAL 已给出「LLF 损害 transfer」的证据，我们 Paper B 已给出「知识不 heal」的证据 → **这更像一篇「破坏式课程在 LLM 上不 work / 有明确代价边界」的负结果论文**，而不是一个新方法论文

---

## 9. 给 MAIN 的硬话（不客气版）

**这个方向作为「我们提出一个新方法」是死的。死在三处：**

1. **算子层面死在 `arXiv 2202.00155`（LLF, ICLR 2022）§3.2**：`M^l_LLF = {1 if l<L, 0 if l≥L}` + 脚注 3「reset = a new initialization」+ Table 1「N3/N8/N10 generations」。我们的 `keep_front_layers=j / n_fresh_layers=K`（同深度）与之数学等价。**审稿人只需引这一篇，我们的 novelty claim 就归零。**
2. **「LLM 预训练里周期 reset 模块买塑性」这个 framing 死在 `arXiv 2307.01163`（NeurIPS 2023）Abstract**：`resetting the embedding layer every K updates during pretraining` + Figure 4 的 forget-relearn / loss-spike / episodic 叙事。**我们想写的 motivation 段落，人家 2023 年写过了。**
3. **「多轮循环 + 终点等尺寸」这条 AUDIT1 用来划界的护城河死在 `arXiv 2106.12379`（AC/DC, NeurIPS 2021）§3.2** 和 **DSD 自己的 Algorithm 1（`goto Sparse Phase for iterative DSD`）+ §4.3（`A second DSD iteration can further improve the accuracy`）**。AUDIT1 说 DSD 单轮，这是它读过却读错的地方。

**唯一还能活的形态**：不要当方法论文，当**负结果 / 边界刻画论文**。
具体：拿 LLF（ICLR 2022，图像上 SOTA 的 iterative 方法）当**明确的 prior method**，
提出的问题是「**LLF 类破坏式课程能否 scale 到 7B LLM 预训练；如果不能，代价卡在哪一类能力上**」。
我们手上有独家弹药可以让这个负结果有分量：
- Paper B 的 keep14@200k（PPL 恢复到 1.428×，MMLU 只恢复 19.5%）+ matched-PPL 对照（keep14@step67500）→ 一个**已量化的"知识不 heal"下界**，正好是 LLF 在图像域从未遇到过的失效模式
- keepN ladder（keep8/10/12/14/16/full32/ShortGPT，全 200k，每 5k 有 ckpt）→ 可以直接换算成「单轮 reset 的深度-代价曲线」，做 cycling 的**代价预算表**
- per-layer knowledge onset（OLMo-2 L18→L19 0.326→0.544；Qwen3 L24→L25 0.236→0.621）→ 可以论证「reset 到 L18 以下 vs 以上」是两个物理机制，这是 LLF 的 ResNet block 3/4 阈值选择没有对应物的地方
- SEAL 的「LLF 损害 transfer」+ SCALE 的「depth expansion 严重遗忘」→ 两个**独立的先验负面证据**，让我们的负结果不是孤例而是一条规律的第三个数据点

**如果 MAIN 坚持当方法论文，必须先回答**：LLF/SEAL/lw/AC/DC/Active Forgetting 五篇（其中四篇 peer-reviewed：ICLR 2022 / CVPR 2023 / NeurIPS 2021 / NeurIPS 2023）之外，我们的**机制假设**新在哪。目前我看不到，而且我认为答案是「不新，只是换了 domain 和被 reset 的模块」。

**必须立刻做的三件事**：
1. 派第二轮 audit 扫 **AC/DC + Active Forgetting + 2109.00267 的 forward citations**（我只扫了 LLF）。若这三条里出现 decoder-only LLM 的层级循环 reset，本方向应判 REFUTED 并彻底放弃。
2. 复核 `2602.11137` 的 venue（我只有第三方引用自述 ICML 2026，S2 429）。
3. 更新 AUDIT1：其 §4「四格全占的组合不存在」的表格、§4「四个必须同时成立的支柱」、§5 R1 对 DSD 单轮的描述、§5 R4「LLM 侧塑性空白」四处**均已被证伪**，不可作为写作依据。

---

## 10. venue 判定汇总（本报告新增的全部论文）

| arXiv ID | 标题 | venue 判定 | 依据 |
|---|---|---|---|
| **2202.00155** | Fortuitous Forgetting (LLF) | **ICLR 2022, peer-reviewed** ✅ | S2 `venue='International Conference on Learning Representations'`, `type=conference` + arXiv COMMENT `ICLR Camera Ready` + Journal-ref `ICLR 2022`（三重一致） |
| **2304.04858** | SEAL (Simulated Annealing in Early Layers) | **CVPR 2023, peer-reviewed** ✅ | S2 `venue='Computer Vision and Pattern Recognition'`, `type=conference`, DBLP `conf/cvpr/SarfiKCKRMB23`, DOI `10.1109/CVPR52729.2023.01935` |
| **2307.01163** | Active Forgetting (Language Plasticity) | **NeurIPS 2023, peer-reviewed** ✅ | S2 `venue='Neural Information Processing Systems'`, `type=conference`, DBLP `conf/nips/ChenMRAS0A23` + COMMENT `NeurIPS 2023 Final Version` |
| **2106.12379** | AC/DC | **NeurIPS 2021（论文自述）** | arXiv COMMENT `Accepted at NeurIPS 2021`；⚠️ S2 查询 429 未取到 → 按项目规矩以自述为准并注明 S2 未核 |
| **2109.00267** | Impact of Reinitialization on Generalization in CNNs | **arXiv preprint** | S2 `venue='arXiv.org'`, DBLP `journals/corr/abs-2109-00267`, cites 24 |
| **2606.24752** | Can Scale Save Us From Plasticity Loss in LLMs? | **arXiv preprint**（显式标注） | arXiv COMMENT 空、Journal-ref 空 |
| **2602.11137** | Weight Decay Improves LM Plasticity | **疑 ICML 2026，需复核** ⚠️ | 仅 2606.24752 引作 `ICML '26`；自身 COMMENT/Journal-ref 均空；S2 429 |
| **2603.20860** | Restoring NN Plasticity for Faster Transfer Learning | **SACAIR 2025 (Springer CCIS 2784), peer-reviewed** | arXiv Journal-ref `Coetzer, X., Schreuder, A., Bosman, A.S. (2026). SACAIR 2025. CCIS vol 2784, Springer` |
| **2202.01290** | Cyclical Pruning for Sparse NNs | **arXiv preprint** | COMMENT 空、Journal-ref 空 |
| **1607.04381** | DSD | **自述 ICLR 2017**（AUDIT1 已正确标注；本报告纠正的是其"单轮"描述，非 venue） | AUDIT1 §6 已核 |
| **2310.07996** | Reset It and Forget It | **ECAI 2024**（COMMENT 自述） | arXiv COMMENT `Published in ECAI 2024` |

---

*本文件只由本次 skeptic subagent 写入；未修改任何 .tex / status/ / versions/ / TODOList 文件，也未修改 AUDIT1。*
*原始抓取产物：`/tmp/skep1/html/*.txt`（全文）、`/tmp/skep1/abs/*.html`（COMMENT/Journal-ref）、`/tmp/skep1/venue/*.json`（S2）、`/tmp/skep1/search/*.xml`（arXiv query 原始返回）。*
*⚠️ /tmp 非持久盘；若需留档请 MAIN 转存到项目盘。*
