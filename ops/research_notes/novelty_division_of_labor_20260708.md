# Novelty 检索：分工假设(命题A) + 语义 bottleneck from-scratch pretrain(命题B)

日期：2026-07-08
方法：WebSearch(arxiv API export) + curl arxiv abs 页逐篇查证标题/摘要（WebFetch 被企业网络策略拦，改用 hy-proxy + curl）。每个 arxiv id 均已核对标题，未凭记忆编号。

---

## 命题定义（我们要主张的 insight）

- **命题A【分工假设】**：Transformer 前/中层主要做语义理解；顶部少数层主要学 auto-regressive 生成策略。取前 j 层 hidden 即含"语义信息"，后面几层把语义转成 next-token 分布。已用 probing(POS/DEPREL/CoLA/WiC + logit-lens) + 截断下游 + 跨 backbone(Qwen3-8B & Llama-3-8B) 验证。
- **命题B【语义 bottleneck pretrain】**：在第 j 层插 low-rank funnel(down d→d_bottle→GELU→up，**无 residual**)，**从头 pretrain**，强制信息在缓存点压缩 → 模型天生"缓存友好"(缓存点 activation 可压缩、缓存深度可切分)。已在 1B from-scratch 验证：前 7 层生成 acc=0(分工被显式强制)、几乎无损、缓存点 dim99 从 1858→231，要 scale 到 7B。

---

## 方向 1：层功能分析 / probing（命题A 的直接前人）

| arxiv id | 标题 | 它做了啥 | 我们的 delta |
|---|---|---|---|
| **1905.05950** | BERT Rediscovers the Classical NLP Pipeline (Tenney et al.) | edge-probing + "cumulative/differential scoring"，证明 BERT **底层→顶层依次编码 POS→parsing→NER→SRL→coref** 经典 NLP 流水线；顶层做需要长程/消歧的高层任务。 | 我们把"低→高层功能递进"从 **encoder(BERT)** 搬到 **decoder LLM(Qwen/Llama)**，且新增一条 Tenney **没做的**主张：**顶部少数层是 AR next-token 生成专用**（Tenney 是 MLM encoder，无 AR 生成层这一说）。我们的 delta = "语义在中层已饱和 + 顶层≈生成头"，而非只是"任务分层"。 |
| **1905.06316** | What do you learn from context? Probing for sentence structure (Tenney et al., edge probing) | 提出 edge-probing 任务套件；发现 LM/MT 表示对**句法**强、对**语义**提升小。 | 我们用 POS/DEPREL/CoLA/WiC 作 probing，结论方向一致（句法早、语义中层），但目标不同：我们是为"截断到第 j 层做缓存"找**信息饱和深度**，不是纯解释性。 |
| **1906.01698** | Open Sesame: Getting Inside BERT's Linguistic Knowledge | 诊断分类器 + attention 分析，发现 **BERT 低层编码线性/位置信息，高层转向层次结构**。 | 同为"低层 vs 高层功能分化"证据，但仍是 encoder；我们主张 decoder LLM 顶层专做生成。 |

**背景综述**：2002.12327 (A Primer in BERTology) 系统汇总 150+ 研究，明确"BERT 底层句法、中层语义、顶层任务特化"是共识——**命题A 的"层功能递进"部分是已被反复证实的领域共识（非新）**。

**logit-lens / tuned-lens（命题A 的"顶层=生成"最相关证据）**：

| arxiv id | 标题 | 它做了啥 | 我们的 delta |
|---|---|---|---|
| **2303.08112** | Eliciting Latent Predictions with the Tuned Lens (Belrose et al.) | 每层训 affine probe 把 hidden 解码成词表分布，展示**预测是逐层迭代 refine 的**，后层越来越接近最终分布。 | tuned-lens 说"预测逐层 refine"，**隐含**顶层≈把语义 sharpen 成分布——但它没有主张"可以砍掉顶层只留语义/顶层是可分离的生成模块"。我们把这个观察**操作化**：既然顶层只做 sharpen，就把缓存点放在语义饱和处、顶层单独负责生成。nostalgebraist 的 logit-lens(博客，无 arxiv)是同族原始观察。 |

**★最强前人（命题A 几乎被完整描述）**：

| arxiv id | 标题 | 它做了啥 | 我们的 delta |
|---|---|---|---|
| **2406.19384** | The Remarkable Robustness of LLMs: Stages of Inference? | 删/换相邻层实验，提出 **4 阶段推理**：(1) detokenization[早层]、(2) feature engineering[中层，任务/实体特征迭代]、(3) **prediction ensembling[后层，hidden 聚合成 next-token 预测]**、(4) **residual sharpening[顶层，抑制无关特征定稿输出分布]**。发现**中层可删除鲁棒、早层和末层删除最伤**。 | ★**这是命题A 最接近的前人**：它的 stage 3+4(prediction ensembling + residual sharpening) 正是我们说的"顶部少数层做 AR 生成/转分布"。**我们的 delta**：(a) 他们是纯 interpretability(删层扰动)，我们做 **probing 定量 + 跨 Qwen/Llama backbone 复现 + 下游截断验证**；(b) 更关键——他们**没有据此改架构/pretrain**，我们用命题A 去指导"缓存点选深度 + bottleneck pretrain"(命题B)。他们的"中层可删"甚至和我们"缓存点放中层"互补(说明中层表示冗余、可压)。 |

---

## 方向 2：early-exit / layer-skipping（和命题A 的 delta）

| arxiv id | 标题 | 它做了啥 | 我们的 delta |
|---|---|---|---|
| **2207.07061** | Confident Adaptive Language Modeling (CALM) | 逐 token 动态早退：置信度够就在浅层出 token，省算最高 3×。 | CALM 是**推理时省算**(哪些 token 简单就早退)，前提也隐含"浅层已够预测简单 token"。**delta**：我们不是 per-token 动态退出，而是**结构性地把"语义层/生成层"分开**并在语义层做**可压缩缓存**；CALM 不改缓存、不 pretrain bottleneck。 |
| **2404.16710** | LayerSkip: Early Exit + Self-Speculative Decoding | 训练时 layer-dropout(浅层低、深层高)+ 共享早退 loss，使模型能在浅层早退；推理用早退层做 draft、剩余层 verify。**在 from-scratch pretrain / continual / finetune 都试了。** | LayerSkip **训练**模型使浅层可早退——和我们"训练使模型缓存友好"精神最像。**delta**：LayerSkip 目标是**加速解码(draft-verify)**，改的是 loss(早退 loss + layer dropout)，**没有在中间层插 bottleneck 压缩 activation、也不为"截断缓存深度/压缩缓存维度"服务**。我们的产物是"缓存点可压缩表示"，不是"浅层可出 token"。 |
| **2004.12993** | DeeBERT: Dynamic Early Exiting for BERT | encoder 分类早退，省 40% 推理。 | encoder 分类任务早退，与 AR 生成/缓存无关，仅作 early-exit 谱系背景。 |
| **2309.08168** | Draft & Verify: Self-Speculative Decoding | 推理时**跳过中间层**生成 draft，再用全模型一次前向 verify，无损加速、无需训练。 | 印证"中间层可跳过"(呼应 2406.19384 中层鲁棒)，但纯推理 trick、不训练、不压缩缓存。与命题B 无冲突。 |

**小结**：early-exit 系全部是"**浅层可出 token / 跳中层加速**"，共享"浅层信息足够"的直觉，但**没有一个**把这转化为"中层插 bottleneck、from-scratch pretrain、让缓存点 activation 低维可压 + 缓存深度可切"。命题A 的"层分工"直觉在此系是常识背景，命题B 的落地方式无人做。

---

## 方向 3：bottleneck / 信息瓶颈 pretrain（★命题B 最可能撞车，重点）

| arxiv id | 标题 | 它做了啥 | 我们的 delta |
|---|---|---|---|
| **2006.03236** | Funnel-Transformer | encoder **沿序列长度维**逐段 pool 下采样(缩短 token 数)、需要时再上采样恢复分辨率；省算并把节省投给更深/宽模型。**from-scratch pretrain**。 | ★名字最像但方向正交：Funnel 压的是**序列长度(token 数)**，我们压的是**hidden 维度(d→d_bottle)**、且只在**单个缓存层**、**无 residual**、目的是"缓存点可压缩+可切深度"。Funnel 不为缓存服务，压缩的是 seq 轴。 |
| **2110.13711** | Hourglass / Hierarchical Transformers | LM 版沙漏：中段沿**序列**下采样再上采样(U-Net 式)，enwik8/ImageNet32 更高效。from-scratch。 | 同 Funnel：压**序列轴**不是**特征维**，且是对称 U-Net 多层，不是单点 low-rank funnel-for-cache。 |
| **2107.14795** | Perceiver IO | 用固定数 latent array 把任意大输入 cross-attend 成小 latent(压 token 数)，再 query 出任意输出。 | 压的是**输入 token 数→固定 latent 数**(序列/set 压缩)，非"某层 hidden 维度压缩"。且 Perceiver 是 latent bottleneck 架构、非 decoder LLM 缓存点。 |
| **2311.05928** | The Shape of Learning: Anisotropy & Intrinsic Dimensions | 发现 **decoder 中层各向异性最高(钟形曲线)**；训练中 intrinsic dim 先升后**在训练后期压缩到更紧凑表示**。 | ★这是命题B 的**理论支撑而非撞车**：它说 decoder 中层表示本就趋向低内在维——**佐证我们"在中层放 bottleneck 压缩 activation 几乎无损"是可行的**（中层信息本就低维）。我们把它从"观察"变成"主动强制"(插 funnel pretrain)。可作为动机引用。 |

**信息瓶颈(IB)一族**（1909.07405 BottleSum、2110.01280、2305.12458 Infor-Coef）：都用 Tishby IB 做**文本摘要/token 剪枝**(压 X→压缩表示预测 Y)，**不是**"transformer 内部某层维度 bottleneck + from-scratch LM pretrain"。方向不同，仅共享"bottleneck"术语。

**★★ 命题B 的直接对撞检索结论**：以下 query 全部**返回空或无关**（arxiv API，2026-07-08）：
- `"cache-friendly" AND "language model"` → 无匹配架构工作
- `"compress" AND "expand" AND "hidden" AND "pretrain" AND transformer`(找 down-up funnel from-scratch) → **空**
- `"bottleneck layer" AND "neural network" AND representation` → 只出 CNN/音频，无 LLM 中层特征维 bottleneck pretrain
- `"low-rank" AND "hidden states" AND "language model" AND pretraining` → 只出 PEFT

→ **没有检索到"在 decoder LLM 中间层插 low-rank(无 residual)funnel、从头 pretrain、以获得缓存友好(可压缩+可切深)表示"的完全相同工作。**

---

## 方向 4：KV/hidden 缓存且"只存中间层"

| arxiv id | 标题 | 它做了啥 | 我们的 delta |
|---|---|---|---|
| **2410.05004** | HCache: Fast State Restoration in LLM Serving | 存**中间层 activation**(而非全 KV / 原始 token)，恢复时从中间 activation 重算上层，TTFT 降 1.93×、存储省 1.92–2.4×。 | ★这是"缓存中间层 hidden、上层重算"的**系统实现**——和命题B 的部署形态高度相关！**delta**：HCache 是**post-hoc serving 系统**，对**固定预训练模型**存中间 activation；它**没改模型、没 pretrain 让中间层可压缩、没 bottleneck**。我们主张 from-scratch 造一个"缓存点天生低维可压"的模型 → HCache 那套存中间层的收益在我们模型上会**成倍放大**(存的是 d_bottle 而非 d)。可把 HCache 当作"我们模型的下游受益系统"。 |
| **2405.05254** | YOCO: You Only Cache Once (decoder-decoder) | **self-decoder** 编码全局 KV 一次，上面叠 **cross-decoder** 复用同一份 KV → 全局只缓存一次，1M context near-perfect needle。**from-scratch 架构**。 | ★这是命题B 精神上最接近的**架构级**前人：都"改架构 + from-scratch 训 + 目标是缓存高效 + 上下部分分工(下=编码全局、上=复用)"。**delta**：(1) YOCO 省的是**层数×KV 份数**(只 cache 一层的 KV，跨上层共享)，我们省的是**缓存点的特征维度(d→d_bottle)+ 可切缓存深度**——正交的压缩轴；(2) YOCO 无 bottleneck 压 activation 维度；(3) 我们的"下=语义/上=生成"分工由 **probing(命题A)** 实证驱动并显式强制(前 j 层 gen acc=0)，YOCO 的分工是架构设计而非语义分工验证。**YOCO 是最需要在 related work 里正面区分的工作。** |
| **2512.03870** | Reconstructing KV Caches with Cross-layer Fusion | 跨层 KV 共享(YOCO/CLA 式)基础上用融合重建被省层 KV。 | 跨**层**共享 KV，不压特征维、不 pretrain bottleneck。谱系背景。 |
| **2310.07240** | CacheGen | 自定义 tensor encoder **量化+流式**传输 KV cache，降网络/加载延迟。 | 纯 post-hoc KV **压缩传输**(serving)，不改模型、不 pretrain。背景。 |

**★ 训练期让模型可压缩（命题B 最接近的"训练目标"对撞）**：

| arxiv id | 标题 | 它做了啥 | 我们的 delta |
|---|---|---|---|
| **2605.05971** | Training Transformers for KV Cache Compressibility (KV-CAT) | 形式化"KV 可压缩性是**学到的表示的性质**，不是上下文的性质"；证明同一函数有可压/不可压实现；提出 **KV-CAT continued pretraining**：train-time 随机 mask KV slots，逼模型学"可后压"表示。 | ★★**这是命题B 在"训练使模型可压缩"这一核心论点上最强的对撞**——它和我们共享同一 thesis("要在训练期引导可压缩表示，post-hoc 有上限")。**但机制完全不同**：(1) KV-CAT 用 **KV-slot masking**(沿 token/slot 稀疏化)，我们用 **hidden 维度 low-rank funnel bottleneck**(沿特征维、单缓存层、无 residual)；(2) KV-CAT 是 **continued pretraining**(改已有模型)，我们是 **from-scratch**；(3) KV-CAT 不涉及"层分工/缓存深度可切"，我们把命题A 的语义分工 + 缓存点深度选择绑进来。**必须在 related work 明确引用并区分**——它已占了"train for compressibility"的旗号，我们的增量必须落在"**维度 bottleneck + 深度可切 + 语义分工驱动的缓存点选择**"。 |
| **2603.13875** | GradMem: Learning to Write Context into Memory (test-time GD) | 冻结权重，test-time 对 prefix memory tokens 做梯度下降把 context 写进 compact memory。 | compressive memory 一族，但是**推理时优化 memory token**、不 pretrain bottleneck、不涉及层分工。远亲。 |

---

## 方向 5：representation collapse / anisotropy 随层变化

| arxiv id | 标题 | 它做了啥 | 与命题B 关系 |
|---|---|---|---|
| **2311.05928** | The Shape of Learning (见方向3) | decoder **中层 anisotropy 最高(钟形)**；训练后期 intrinsic dim 压缩。 | ★**支撑证据**：中层表示各向异性高/内在维低 → 在中层插 bottleneck 压 activation 代价小。我们的"dim99 从 1858→231"与"中层内在维本就低"一致。 |
| **2503.21718** | Outlier dimensions favor frequent tokens (last-layer) | **末层** outlier 维度服务高频 token 预测(生成启发式)。 | ★**命题A 支撑**：末层维度专为"输出分布/高频 token"服务——正是"顶层≈生成头"的机制证据。可引用。 |

（经典 anisotropy 起点 Ethayarajh 2019 "How Contextual are Contextualized Word Representations?"、Gao 2019 Representation Degeneration 未在本轮 API 命中，为已知背景文献，非直接对撞。）

---

## 结论表

| 命题 | novelty 等级 | 最近/最强前人 | confidence |
|---|---|---|---|
| **命题A（分工：中层=语义饱和，顶层=AR 生成）** | **新组合 / 部分已被做过** | 2406.19384 (Stages of Inference: prediction ensembling + residual sharpening ≈ 我们的"顶层生成层"); 2303.08112 tuned-lens; Tenney 1905.05950/06316; 2503.21718 (末层 outlier=高频预测) | **high**（"层功能递进"是 BERTology 共识；"顶层=AR 生成、中层语义已饱和"在 decoder LLM 上有 2406.19384 高度重叠，但我们的 probing+跨 backbone 定量 + 用它驱动架构是增量。命题A 单独发表 novelty 弱，作为命题B 的 motivation 才成立。） |
| **命题B（中层 low-rank 无 residual funnel、from-scratch pretrain → 缓存友好：activation 可压 + 缓存深度可切）** | **新组合（未见完全相同）** | 2605.05971 KV-CAT (train for KV compressibility，但用 slot-mask+continued，非维度 bottleneck+from-scratch); 2405.05254 YOCO (架构级 cache-once，但省 KV 份数非特征维); 2410.05004 HCache (存中间层 activation，但 post-hoc 不训练) | **medium-high**（无完全相同工作；但"训练使可压缩"旗号已被 KV-CAT 占，"改架构从头训做缓存高效"已被 YOCO 占，"存中间层 hidden"已被 HCache 占。我们的**独特交集 = 维度 bottleneck(非 slot/token/层) + from-scratch + 缓存深度可切 + 命题A 语义分工驱动缓存点选择**，这个交集未检索到。） |

---

## ★ 最关键回答：命题B 有没有人做过"完全一样"的？

**没有检索到完全相同的工作。confidence = medium-high。**

推理链：
1. **"训练期引导可压缩表示 > post-hoc"这个 thesis 已被 2605.05971 (KV-CAT) 明确提出并形式化证明**——所以"训练使模型缓存友好"这一句**不能再当作我们首创的口号**，必须引用 KV-CAT。
2. 但 KV-CAT 的机制是 **KV-slot masking + continued pretraining**，压的是 token/slot 维度、改的是已有模型；**没有** low-rank 特征维 funnel、**没有** from-scratch、**没有** 缓存深度可切、**没有** 语义分工驱动的缓存点选择。
3. YOCO(2405.05254) 是架构级 from-scratch cache-efficient，但压"缓存的层数/KV 份数"而非"缓存点的特征维度"；HCache(2410.05004) 存中间层 activation 但纯 serving 不训练；Funnel/Hourglass 压序列轴不压特征维。
4. 我们四个 query 直接找"中层 down-up funnel 无 residual from-scratch pretrain for cache" → **全空**。

**我们仍能主张的增量（写论文时的定位）**：
- **压缩轴独特**：沿 **hidden 特征维**在**单个缓存层**做 low-rank(无 residual)bottleneck，区别于 KV-slot masking(KV-CAT)、序列下采样(Funnel/Hourglass)、跨层 KV 共享(YOCO/CLA)。
- **from-scratch + 缓存点位置由 probing(命题A) 决定**：把"语义饱和深度"实证选出的 j 作为 bottleneck 位置，是 motivation 层面的新贡献（前人无一用层功能 probing 去选缓存点/bottleneck 深度）。
- **"缓存深度可切"(cache-depth truncation)**：前 j 层缓存即可支撑，顶层按需重算——这是 HCache 的 serving 收益在"天生可压模型"上的放大，前人没有 co-design 训练+这个部署形态。
- **必须在 related work 正面区分的三篇**：KV-CAT(2605.05971，同 thesis 不同机制)、YOCO(2405.05254，同 from-scratch cache 目标不同压缩轴)、HCache(2410.05004，同"存中间层"不同"训不训")。命题A 的定位靠 Stages of Inference(2406.19384) 区分。

**风险提示**：命题B 的"完全一样 confidence"只到 medium-high 而非 very_high，因为 arxiv 近月(2026)投稿量极大、API 关键词召回有限，KV/长上下文/压缩是最热赛道之一。建议正式投稿前再针对 **"latent bottleneck layer" / "activation compression pretraining" / "depth-truncatable cache"** 做一轮 Google Scholar + Semantic Scholar 引用图检索（本轮受限于 arxiv API），并重点追 KV-CAT(2605.05971) 的 citing papers。
