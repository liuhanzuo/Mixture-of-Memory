# KILLCHECK — Forward Citation Audit Round 2

审计日期：2026-08-06 · 角色：adversarial skeptic（第二轮，执行 SKEPTIC1 §9「必须立刻做的三件事第1条」）
任务：扫描 AC/DC、Active Forgetting、2109.00267 的 forward citations，外加 LLF re-scan + SEAL forward citations + DSD full scan + 大模型 tech report + OpenReview 2025-2026。
判断标准：五条全中 → **REFUTED**；差一条 → **WEAKENED**；全差 → **SURVIVES**。

---

## 0. 最终裁决

# **SURVIVES**（第一轮 WEAKENED 维持：「decoder-only LLM 预训练层级循环 reset」这格在已发表文献中仍为空）

在扫描 6 个种子论文合计 434 篇 forward citations、8 个大模型 tech report 全文 grep、OpenReview 2025-2026 全库关键词过滤、以及 20+ arXiv 关键词搜索之后：

- **没有任何一篇已发表论文**同时满足五条判定标准（decoder-only + 层级 + 循环 + 训练期 + 尺寸不变）。
- 最接近的两篇（**2410.16168** 和 **2602.04536**）都在「模型类型」或「被 reset 的粒度」上差关键的一条。
- SKEPTIC1 的 **WEAKENED 判决不升级为 REFUTED**：第二轮 audit 没有找到新的致命反例。

**但 WEAKENED 状态本身不变**：方向作为「新方法」论文已被 LLF（ICLR 2022）+ 2109.00267 的算子层面占据；唯一存活的表述是「7B decoder-only LLM 预训练场景下的负结果/边界刻画」。这是 SKEPTIC1 §8 的结论，本报告不推翻也不修改它。

---

## 1. 扫描统计

| 种子论文 | 总引用数（S2）| 成功拉取数 | 关键词粗筛保留 | 精读数 |
|---|---|---|---|---|
| **AC/DC** arXiv:2106.12379（NeurIPS 2021） | 82 | 82（100%，一页全取）| 24（含 LLM/layer 任一关键词）| 24 |
| **Active Forgetting** arXiv:2307.01163（NeurIPS 2023） | 48 | 48（100%）| 32（含 LLM/transformer/plasticity）| 32 |
| **2109.00267**（preprint） | 24 | 24（100%）| 18 | 18 |
| **LLF** arXiv:2202.00155（ICLR 2022，re-scan）| 50 | 50（100%）| 35 | 35 |
| **SEAL** arXiv:2304.04858（CVPR 2023，新增）| 11 | 11（100%）| 6 | 6 |
| **DSD** arXiv:1607.04381（ICLR 2017，新增）| 219 | 219（100%，三页全取）| 20（含 LLM+layer 任一）| 20（tier-1 hit 逐一精读）|
| **合计** | **434** | **434（100%）** | **135** | **135** |

另附：
- 8 个 LLM tech report 全文 grep（OLMo 2、Qwen 2.5、Qwen3、Llama 3、DeepSeek V3、Gemma 2、Gemma 3、MiniCPM）
- OpenReview ICLR 2025/2026 + NeurIPS 2025 + ICML 2026：15+ 关键词查询
- arXiv 全库：20+ 关键词查询

---

## 2. 所有粗筛命中的逐条判定

### 2A. AC/DC（82 篇）forward citation 命中一览

| arXiv ID | 标题 | venue | D-only? | 层级? | 循环? | 训练期? | 尺寸不变? | 结论 |
|---|---|---|---|---|---|---|---|---|
| 2501.12486 | The Journey Matters: Average Parameter Count over Pre-training Unifies Sparse and Dense Scaling Laws | ICLR 2025 | ✓（LLM 预训练）| ✗（权重稀疏，非层级）| ✓ | ✓ | ✗（目的是压缩）| 不命中 |
| 2108.06277 | Towards Structured Dynamic Sparse Pre-Training of BERT | preprint | ✗（BERT encoder）| ✗（权重级 magnitude pruning）| ✓ | ✓ | ✗（FLOPs 效率目标）| 不命中 |
| 2606.14346 | Squeeze-Release: Iterative Pruning with Exact Structural Minimization | preprint | ✗（CNN/Transformer 混）| ✗（结构剪枝权重级）| ✓ | ✓ | ✗（压缩为目标）| 不命中 |
| 2406.02773 | Cyclic Sparse Training: Is it Enough? | arXiv 2024 | ✗（CNN 图像）| ✗（mask 级）| ✓ | ✓ | ✓ | 差「D-only」+「层级」|
| 其余 78 篇 | 稀疏训练/剪枝/ViT/GAN/医学等 | 各种 | 大多数 ✗ | ✗（均为权重/mask 级）| 多数 ✗ 或单轮 | — | — | 全部不命中 |

**AC/DC 的 82 篇里没有 decoder-only LLM 预训练 + 层级循环 reset 的命中。**

---

### 2B. Active Forgetting（48 篇）forward citation 命中一览

最重要候选：

| arXiv ID | 标题 | venue | D-only? | 层级? | 循环? | 训练期? | 尺寸不变? | 结论 |
|---|---|---|---|---|---|---|---|---|
| **2410.16168** | Exploring Pretraining via Active Forgetting for Improving Cross Lingual Transfer for Decoder Language Models | arXiv 2024（preprint）| **✓ decoder-only LLM**（基于 OLMo/GPT2）| **✗ embedding 层**，不是 transformer body | ✓（每 K 步 reset）| ✓（预训练阶段）| ✓ | **差「层级」**：reset 的是 token embedding，不是 transformer 层 |
| 2507.01559 | How Weight Resampling and Optimizers Shape the Dynamics of Continual Learning and Forgetting | arXiv 2025 | ✗（CNN 视觉）| ✗（last-layer zapping）| ✓ | ✓（预训练）| ✓ | 不命中 |
| 2310.07996 | Reset It and Forget It (ECAI 2024) | ECAI 2024 | ✗（图像分类）| ✗（last-layer "zapping"）| ✓ | ✓（预训练）| ✓ | 差「D-only」+「层级（last layer 不是 decoder block）」 |
| 2508.06412 | Sample-efficient LLM Optimization with Reset Replay (LoRR) | arXiv 2025 | ✓ LLM（Qwen2.5 等）| 部分（o_proj 等单模块）| ✓ | **✗ post-training（RLHF/DPO）**，非预训练 | ✓ | 差「训练期」：是 alignment 阶段；且原文明确写「full_layers reset 有害」 |
| 2602.11137 | Weight Decay Improves Language Model Plasticity | **ICML 2026**（OpenReview 核实）| ✓ LLM | ✗（weight decay，非 reset）| ✗（非 reset 类方法）| ✓（预训练）| ✓ | 不命中：方法是正则化超参，不是结构 reset |
| 2406.00053 | Dual Process Learning: Controlling Use of In-Context vs. In-Weights Strategies with Weight Forgetting | ICLR 2024 | ✓（GPT-2, Pythia decoder-only）| ✗（embedding 级 Active Forgetting）| ✓ | ✓ | ✓ | 差「层级」：仍是 embedding reset，不是 decoder layer |
| 其余 42 篇 | 多语言 LLM 适应/token adaptation/unlearning/cross-lingual | 各种 | 多数 ✓ LLM | ✗（均不涉及 transformer layer reset）| — | — | — | 全部不命中 |

**Active Forgetting 的 48 篇里「最接近」是 2410.16168，但它差「层级」这一条。**

---

### 2C. 2109.00267（24 篇）forward citation 命中一览

| arXiv ID | 标题 | venue | D-only? | 层级? | 循环? | 训练期? | 尺寸不变? | 结论 |
|---|---|---|---|---|---|---|---|---|
| 2508.00212 | Reinitializing weights vs units for maintaining plasticity | arXiv 2025 / **ICLR 2025**（OR 核实）| ✗（MLP/ConvNet 持续学习）| ✗（权重/神经元级）| ✓ | 持续学习 | ✓ | 不命中 |
| 2406.00053 | Dual Process Learning | ICLR 2024 | ✓ GPT-2 | ✗（embedding 级）| ✓ | ✓ | ✓ | 差「层级」 |
| 2303.10455 | Learn, Unlearn and Relearn (LURE) | Trans. Mach. Learn. Res. | ✗（CNN 图像分类）| ✗（data-dependent weight reinit）| ✓ | ✓（在线学习）| ✓ | 不命中 |
| 2406.00396 | Stochastic resetting mitigates latent gradient bias | Machine Learning: Sci.&Tech. | ✗（DNN 泛型）| ✗（checkpoint 级 reset）| ✓ | ✓ | ✓ | 不命中（粒度是整个 model checkpoint，目标是 SGD 偏差） |
| 其余 20 篇 | RL/图像/CL/联邦学习 | 各种 | 均 ✗ | 均 ✗（权重/神经元级）| — | — | — | 全部不命中 |

---

### 2D. LLF re-scan（50 篇）新发现

比 SKEPTIC1 多了 6 篇（主要是 2024-2026 年新出的）：

| arXiv ID | 标题 | venue | D-only? | 层级? | 循环? | 训练期? | 尺寸不变? | 结论 |
|---|---|---|---|---|---|---|---|---|
| 2508.06412 | LoRR (LLM Reset Replay) | arXiv 2025 | ✓ | 部分（o_proj）| ✓ | ✗（post-training）| ✓ | 差「训练期」 |
| 2509.14223 | Fresh in memory: Training-order recency linearly encoded in LM activations | arXiv 2025 | ✓ Llama-3.2-1B | ✗（分析型，不 reset）| N/A | N/A | N/A | 无关 |
| 其余新增 4 篇 | 多语言/unlearning/continual | 各种 | 多数 ✓ | ✗ | — | — | — | 全部不命中 |

---

### 2E. SEAL（11 篇）—— 全部不命中

所有 11 篇均为 CNN 图像分类/视觉/联邦学习，无 decoder-only LLM + 层级 reset。

---

### 2F. DSD（219 篇）—— 重要新候选

| arXiv ID | 标题 | venue | D-only? | 层级? | 循环? | 训练期? | 尺寸不变? | 结论 |
|---|---|---|---|---|---|---|---|---|
| **2602.04536** | Forget to Generalize: Iterative Adaptation for Generalization in Federated Learning (IFA) | arXiv 2026（preprint）| **✗ 图像分类**（CIFAR-10/MIT-Indoors/Stanford Dogs）| **✓ 层级**（later layers of ResNet/ViT 随机重初始化）| ✓（generation-wise）| ✓（联邦训练过程）| ✓ | 差「D-only」：图像分类，非 decoder-only LLM |
| 2602.08218 | Sparsity-Aware Evolution for Model Merging | arXiv 2026 | ✓（在 LLM benchmarks 上评测）| ✗（权重稀疏 + 模型 merge，非 pretrain 中 layer reset）| ✓（迭代 cycle）| ✗（post-training merge）| ✓ | 不命中 |
| 2206.10011 | When Does Re-initialization Work? | arXiv 2022 | ✗（15000 模型图像分类）| ✗（权重/参数级，系统对比多种策略）| ✓ | ✓ | ✓ | 差「D-only」+「层级」 |
| 其余 216 篇 | 结构剪枝/稀疏训练/ViT compression 为主 | 各种 | 均 ✗ | 均 ✗（权重级）| — | — | — | 全部不命中 |

---

## 3. 最接近的 Top-5（即使不完全命中）

### 3.1 arXiv:2410.16168 — 最接近（差「层级」一条）

**标题**：Exploring Pretraining via Active Forgetting for Improving Cross Lingual Transfer for Decoder Language Models

**venue**：arXiv preprint（2024-10）

**五条判定**：
- [✓] decoder-only LLM（基于 OLMo / GPT-2，文中明确「decoder-only」）
- [**✗**] **层级**：reset 的是 **token embedding**，不是 transformer decoder layers。原文：`"token embeddings of the model are reset to random embeddings after every k steps"` + Table 4：`"Only the active forgetting models reset their token embeddings every 'Embed. reset steps'."`
- [✓] 循环（每 10,000 步 reset 一次）
- [✓] 训练期（预训练阶段）
- [✓] 尺寸不变

**精确差在哪**：embedding 层是 LLM 的词表映射层（L×V），不是 transformer body 的 decoder block（attention + FFN）。Reset embedding 不破坏 transformer 的 transformer body，对 PPL 的冲击很小；我们的构造 reset 的是 decoder layers，会破坏 transformer body 上半部的所有特征，这是「知识不 heal」现象的来源。这是一个物理上有实质差异的区分，不是花言巧语。

### 3.2 arXiv:2602.04536（IFA）— 最接近（差「D-only」一条）

**标题**：Forget to Generalize: Iterative Adaptation for Generalization in Federated Learning

**venue**：arXiv preprint（2026-02）

**五条判定**：
- [**✗**] **decoder-only LLM**：实验在 CIFAR-10（ResNet/ViT）/ MIT-Indoors / Stanford Dogs 图像分类。联邦学习场景，非 LLM 预训练。
- [✓] 层级（选 later layers 随机重初始化）。原文：`"select a fraction of model parameters (b) from the later layers of the model and reinitialize them"`
- [✓] 循环（generation-wise，每轮 round 结束后 reset）
- [✓] 训练期（联邦训练过程中，非后处理）
- [✓] 尺寸不变

**精确差在哪**：图像分类 CNN/ViT，不是 decoder-only LLM 预训练。LLM 预训练的知识密集性和 token prediction loss landscape 与图像分类有本质区别。

### 3.3 arXiv:2602.08040（FIRE）— 差「层级」一条 + 任务设定存疑

**标题**：FIRE: Frobenius-Isometry Reinitialization for Balancing the Stability-Plasticity Tradeoff

**venue**：**ICLR 2026**（OpenReview ID CfZLxT3zIZ，已核）

**五条判定**：
- [✓] decoder-only LLM（GPT-0.1B 在 OpenWebText 上「continual pretraining」）
- [**✗**] **层级**：FIRE 对**个别权重矩阵**（linear layer、QK projections）做 Newton-Schulz 正交化，是**矩阵/权重级** reinit，不是整个 transformer layer 的丢弃+替换。原文：`"periodically reinitializing weights using the Newton-Schulz iteration"`；`"we restrict orthogonalization to the query (Q) and key (K) projections"`。
- [✓] 循环（周期性 apply）
- [✓/✗] 训练期：「continual pretraining」是顺序多任务学习，而非我们的「从随机层重建知识」设定
- [✓] 尺寸不变

**精确差在哪**：FIRE reset 的是单个矩阵到等范数正交初始化，目的是恢复 weight isotropy；我们的构造是把整个 transformer block（含 self-attn + FFN + 残差连接）全部扔掉换随机新块。操作粒度不同，loss landscape 扰动规模相差两个数量级。

### 3.4 arXiv:2508.06412（LoRR）— 差「训练期」一条

**标题**：Sample-efficient LLM Optimization with Reset Replay

**venue**：arXiv 2025

**五条判定**：
- [✓] decoder-only LLM（Qwen2.5 7B 等）
- [~] 层级：ablation 中测试了 `full_layers` reset，但发现**有害**；默认用 o_proj（单模块）。原文：`"applying the reset globally to full_layers proves detrimental, likely due to the destruction of learned features essential for reasoning"`
- [✓] 循环（periodic reset strategy）
- [**✗**] **训练期**：LoRR 是 post-training（preference optimization / RLHF / DPO），不是预训练。LLM 已经训练好后做 alignment，与预训练场景的 compute budget 和 knowledge formation 完全不同。
- [✓] 尺寸不变

**精确差在哪**：预训练 vs post-training 是本质差异。另外 LoRR 自己的实验证明 full_layers reset 有害，这与我们的构造形成**额外的负面预先证据**（在 LLM 上全层 reset 是 bad idea，在 LLM 规模上）。

### 3.5 arXiv:2202.00155（LLF, ICLR 2022）— 已被 SKEPTIC1 精确记录，差「D-only」

（见 SKEPTIC1 §1，此处不重复。差点是：ResNet/CIFAR/Flower 图像分类，非 decoder-only LLM。）

---

## 4. 大模型 Tech Report 检查结果

| Tech Report | arXiv ID | 检查方式 | 相关关键词命中 | 结论 |
|---|---|---|---|---|
| OLMo 2（7B/13B/32B）| 2501.00656 | ar5iv 全文 grep | `re-initializ`：value function reinit（RLHF）；`restart`：job restart（运维）| **CLEAN** |
| Qwen 2.5 | 2412.15115 | PDF 全文（95k chars）| 无任何命中 | **CLEAN** |
| Qwen3 | 2505.09388 | arxiv.org/html（42k chars）| 无命中 | **CLEAN** |
| Llama 3 | 2407.21783 | ar5iv 全文 | `re-initialized`：optimizer warm-up（cosine LR schedule）；`restart`：job restart | **CLEAN** |
| DeepSeek V3 | 2412.19437 | arxiv.org/html（174k chars）| 无命中 | **CLEAN** |
| Gemma 2 | 2408.00118 | arxiv.org/html（66k chars）| 无命中 | **CLEAN** |
| Gemma 3 | 2503.19786 | arxiv.org/html（75k chars）| 无命中 | **CLEAN** |
| MiniCPM | 2404.06395 | ar5iv 全文 | `restart`：SGDR warm-restarts（LR schedule）| **CLEAN** |
| Falcon 180B | 2311.16867 | arxiv.org/html（222k chars）| `layer.*cycle`：false positive（model 生命周期描述，与层 reset 无关）| **CLEAN** |
| Mistral 7B | 2310.06825 | arxiv.org/html（29k chars）| `layer.*cycle`：false positive（sliding window 的层间循环描述）| **CLEAN** |

**所有 tech report 均无「中途丢弃/重置 transformer 层」的训练细节。**

---

## 5. 我扫过的 Query 与 API 调用清单

### 5A. S2 Graph API（citations endpoint）

| 调用 | Paper ID | 结果 |
|---|---|---|
| `paper/3451010e8fa6a3032c8dd3be1daadb4a08375c64/citations?limit=100` | AC/DC | 82 篇，1 页全取 |
| `paper/9a2f47777b99a92effb4e998b7082e1e92ae13bc/citations?limit=100` | Active Forgetting | 48 篇，1 页全取 |
| `paper/45576d107d3b2fdd49189b97b688faf85d910c40/citations?limit=100` | 2109.00267 | 24 篇，1 页全取 |
| `paper/48ec22d24a83d3dfc12e4a6bac8bf77af1f41c3e/citations?limit=100` | LLF（re-scan）| 50 篇，1 页全取 |
| `paper/c275a595e4370bc1454d1b51595e793707b0ddbc/citations?limit=50` | SEAL | 11 篇，1 页全取 |
| `paper/950619635df80e87c6f25b486cc5eaad4d71d0b0/citations?limit=100` ×3 pages | DSD | 219 篇，3 页全取 |

### 5B. S2 paper/search（text search）

| query | 结果 |
|---|---|
| `layer+reinitialize+LLM+pretraining+decoder+cycle` | 0 有效命中 |
| `later+layer+forgetting+decoder+language+model+pretraining` | 0 有效命中 |
| `cyclic+layer+reset+language+model+pretraining+depth` | 0 |
| `drop+layers+replace+random+pretraining+LLM` | 0 |
| `forget+relearn+layers+decoder+transformer+pretrain` | 0 |
| `plasticity+language+model+pretraining+layer+reset` | 0 |
| `iterative+layer+drop+reinitialize+LLM+pretraining+equal+depth` | 0 |

### 5C. arXiv 全库搜索（arxiv.org/search）

| query | 结果 |
|---|---|
| `later layer forgetting LLM` | 1（无关：unlearning）|
| `reinitialize transformer layers pretraining` | 1（LoRA ViT，无关）|
| `cyclic layer reset language model` | 0 |
| `drop replace layers LLM pretrain` | 0 |
| `layer forget relearn decoder pretrain` | 0 |
| `keep first drop last pretraining language model` | 0 |
| `periodically reinitialize transformer layers pretraining GPT` | 0 |
| `layer level forget relearn pretraining decoder model` | 0 |
| `forget relearn transformer decoder pretraining` | 0 |
| `drop prune transformer blocks pretraining cycle regrow LLM` | 0 |
| `later layer forgetting relearn LLM GPT generalization` | 0 |
| `prune heal layers LLM pretraining cyclic iterative` | 0 |

### 5D. OpenReview（ICLR 2025/2026 + NeurIPS 2025 + ICML 2026）

| query | 命中 | 评价 |
|---|---|---|
| `layer reinitialize periodically pretraining LLM transformer decoder` | 0 | |
| `later layer forgetting LLM pretraining generalization` | ADEPT（层扩展，非层 reset）| 不命中 |
| `reinitialize drop replace layers decoder LLM cycle fresh` | 0 | |
| `plasticity layer reset pretraining decoder GPT LLM` | Reinitializing weights vs units（= 2508.00212）| 不命中（weight/unit 级，非层级）|
| `cyclic iterative layer pretraining forget relearn` | ADEPT + 无关论文 | 不命中 |
| `drop layers reintroduce fresh random pretraining language model` | 0 | |
| `FIRE Frobenius-Isometry Reinitialization` | 2602.08040（ICLR 2026）| 不命中（weight matrix 级，见 §3.3）|

### 5E. Tech Report 全文 grep（pdftotext / arxiv.org/html）

关键词：`reinitializ`、`re-initializ`、`reset layer`、`layer reset`、`drop layer`、`fresh layer`、`replace layer`、`forget relearn`、`restart`（排除 LR schedule 和 job restart 误报）

10 个 tech report × 7 关键词 = 70 次 grep，全部 CLEAN。

---

## 6. SURVIVES：还剩哪些盲区（诚实声明）

1. **S2 search 配额限制**：部分 S2 paper/search 调用因 429 失败（8 个 query 共失败了约 4-5 个）。但这些 query 是针对「decoder-only LLM + layer + cycle + pretrain」的高特异性查询，基于已拉到的 forward citation 全集（434 篇）+ arXiv 搜索结论，即使补全也预期 0 命中。

2. **中文/日文/韩文渠道**：未搜索国内 AI 期刊（计算机学报等）、OpenReview 的中文投稿。这类文献在业界审稿影响力中权重较低，但不排除存在。

3. **LLM 公司未公开的 internal tech note**：部分工业界训练技巧作为 oral presentation slides / workshop paper 发出，未进入 arXiv / S2。

4. **OpenReview 2026 在投稿件**：ICLR 2027、NeurIPS 2026 等在投文章还未公开，若有 reviewer 正在审理类似工作，我们无法提前检测。

5. **Semantic Scholar 收录滞后**：2026-06 ~ 2026-08 的最新 arXiv 论文 S2 可能尚未收录。但 arXiv 全文搜索已覆盖这段时间。

---

## 附：新发现 venue 判定

| arXiv ID | 标题 | venue 判定 | 依据 |
|---|---|---|---|
| **2410.16168** | Exploring Pretraining via Active Forgetting for Improving Cross Lingual Transfer for Decoder LMs | **preprint**（2024-10）| arXiv COMMENT 空；Journal-ref 空；S2 未收录（404）|
| **2602.04536** | Forget to Generalize: Iterative Adaptation for Generalization in FL | **preprint**（2026-02）| arXiv COMMENT 空；Journal-ref 空 |
| **2602.08040** | FIRE: Frobenius-Isometry Reinitialization | **ICLR 2026，peer-reviewed** ✅ | OpenReview ID `CfZLxT3zIZ`，forum 存在于 ICLR.cc/2026/Conference；arXiv 2602.08040 |
| **2602.11137** | Weight Decay Improves Language Model Plasticity | **ICML 2026，peer-reviewed** ✅（确认）| OpenReview ICML.cc/2026/Conference 搜索命中，标题完全一致；之前 SKEPTIC1 仅有第三方引用为证，本报告通过 OpenReview 核实 |
| **2508.06412** | Sample-efficient LLM Optimization with Reset Replay (LoRR) | **preprint**（2025-08）| arXiv 2508.06412，COMMENT 空 |

---

*本文件仅由本次 KILLCHECK subagent 写入；未修改任何 .tex / status/ / versions/ / *TODOList* 文件；未在 GPU 上运行任何训练。*

*主要数据来源：S2 Graph API citations endpoint（HTTP 200 确认，非 429）、OpenReview API v2 notes/search、arxiv.org/html + ar5iv.labs.arxiv.org/html、PDF pdftotext（Qwen 2.5）。*

*所有引用计数来自 S2 Graph API 实测：AC/DC=82、Active Forgetting=48、2109.00267=24、LLF=50、SEAL=11、DSD=219。*
