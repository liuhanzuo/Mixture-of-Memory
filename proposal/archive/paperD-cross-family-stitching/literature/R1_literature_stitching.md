# Paper D · R1 文献 + 概念定位调研报告

**主题**：跨模型 depth-wise layer stitching（取 A 的前 k_A 层 + B 的后段 + 1-2 层可训 stitching layer / adapter，冻结 A/B 骨干，只训 stitch）
**范围**：仅文献 + 概念定位（benchmark 设计与仓库可行性由另两个 agent 负责）
**调研日期**：2026-08-05
**证据规则**：本报告每条引用的标题都是我自己 `curl` 到 arXiv abs 页 `citation_title` 或 PDF/ar5iv 全文确认过的；venue 经 Crossref / ACL Anthology / OpenReview API2 / papers.nips.cc 交叉核实。凡我未 fetch 到原文的，明确标注。**没有任何一条引用来自我的记忆或他人报告。**

---

## 1. 一句话判决

**这个想法本身不是新的，"跨异构 foundation model 的 depth-wise stitching + 只训 stitch layer + 目标是博采众长（而非仅做表征探针）"这一整套已经在 vision 侧被 CVPR 2026 的 `Revisiting Model Stitching In the Foundation Model Era`(arXiv:2603.12433) 做完并做正了**（结论：异构 VFM 可缝、深层缝合点上缝合模型能**超过两个成分模型**）；**在 LLM 侧，跨家族层缝合已经被做过但只作为 interpretability 探针**（arXiv:2410.08255 用线性 adapter 缝 OPT/Pythia/Mistral/LLaMA，报的是 test loss 退化曲线，**不追求 benchmark 增益**），**而"用它来提升 benchmark"这条在 LLM 上最接近的尝试（Sakana DFS，Nature MachIntell 2025）给出的是相对负结果**：DFS 层路径搜索得到的 10B 模型 MGSM-JA 36.4，**显著低于同一批源模型做权重空间 merge 的 52.0**。

所以判决是：**(a) 概念不新（vision 已系统化，LLM 已有探针版）；(b) LLM 侧"提升 benchmark"的那一格确实还空着，但已有的最接近证据（Sakana DFS、SOLAR seam 分析、Bansal 的深层不可缝性）都指向"跨模型缝深层收益小、且弱于权重空间 merge"；(c) 若只训 1-2 层、backbone 全冻，novelty 和收益都很难撑起一篇论文——必须重定位（见 §5）。**

---

## 2. 引用表

格式：标题 / arXiv ID / venue（核实方式）/ 方法一句话 / 关键数字 / 与我们想法的关系。

### 2.1 Model stitching 本源（vision）

| # | 标题 | ID | venue | 方法 | 关键数字 | 关系 |
|---|------|----|-------|------|---------|------|
| L1 | Understanding image representations by measuring their equivariance and equivalence | arXiv:1411.5908 | **CVPR 2015**（Crossref: "2015 IEEE CVPR"；亦有 IJCV 2018 期刊版） | 提出 **stitching layer**：学一个 equivalence map `E_{φ1→φ1'}`（CNN 里实现为一组线性 filter，即 1×1 conv）把 A 的第 l 层特征映射到 B 的第 l 层输入，测两个表征是否"equivalent" | **把 E 设成 identity → top-1 error > 99%**（不同参数化的通道完全不兼容，必须学 stitch）。学了 stitch 后：Alexn↔Imnet **Conv1/Conv2 在所有情况下可互换**，**到 Conv4 仍能建立"很好"的 equivalence**；**Conv5 不完全可互换**，跨任务（Plcs, 场景分类）的深层"substantially less compatible"，但即使最差也远好于 chance | **先驱**（我们方法的祖先；且已给出"越深越难缝"的第一版证据） |
| L2 | Revisiting Model Stitching to Compare Neural Representations | arXiv:2106.07682 | **NeurIPS 2021 Poster**（OpenReview api：NeurIPS 2021 Poster；papers.nips.cc hash 页标题一致） | 把 stitching 系统化为表征比较工具：冻结 A、B，A 的 bottom + B 的 top，中间一个**低容量**可训层（CNN 里是 1×1 conv），定义 **stitching penalty** = 缝合后 loss − B 原 loss | stitch layer 明确设计为 **"very low capacity, only meant to align representations, rather than improving the model"**；stitching penalty 有**可解释单位**（"penalty 3% 意味着替换前几层后精度最多降 3%"）；**同架构不同训练方式（supervised vs SimCLR/SwAV/DINO 自监督，ResNet-50 backbone，各自 75%±1，SimCLR 68.x）可以无精度损失互缝**；提出 **stitching connectivity**（SGD 的极小点几乎都能互缝）；**"more is better"：更多数据/更宽/更长训练的表征可以"插进"弱模型并提升它**（5K vs 25K 样本那组 **stitching penalty 为负**，而 **CKA 可低到 0**） | **先驱 + 方法论基准**（"stitch 只对齐、不增强"这一条正是审稿人会用来打我们的） |
| L3 | Stitchable Neural Networks (SN-Net) | arXiv:2302.06586 | **CVPR 2023 Highlight**（arXiv comments 字段；Crossref: 2023 IEEE/CVF CVPR） | 在**同一个 model family**（DeiT-Ti/S/B, Swin）里把不同规模 anchor 切块，用简单 stitching layer 缝成大量子网，运行时切换缝合点做 accuracy-efficiency trade-off | "only a few epochs of training"即可在 anchor 之间**插值**性能；单个 SN-Net 挑战 Timm zoo 里数百个模型 | **先驱/正交**：SN-Net 是 **family 内**（同架构族）+ 目标是**弹性部署**，不是跨家族博采众长；关键差别是它**性能只在 anchor 之间插值，不超越** |
| L4 | Efficient Stitchable Task Adaptation (ESTA) | arXiv:2311.17352 | **CVPR 2024**（arXiv comments: "CVPR 2024 camera ready"） | SN-Net + PEFT：stitches 共享 low-rank 更新、独立 bias，一阶段部署 pipeline | 25 个下游视觉任务；**并且"stitching LLMs from LLaMA family, obtaining chatbot stitches of assorted sizes"** | **先驱（LLM 侧最早的 stitch 落地）** — 但仍是 **family 内**、目标是弹性尺寸 |
| L5 | **Revisiting Model Stitching In the Foundation Model Era**（VFM Stitch Tree, VST） | arXiv:2603.12433 | **CVPR 2026**（arXiv comments: "Accepted by CVPR 2026"；作者 Mai/Zhang/Wang/Chao 等） | **把 stitching 从探针升级为"整合互补 VFM 强项"的实用配方**：跨 CLIP / DINOv2 / SigLIP2 / DINOv3（目标、数据、模态混合都不同）缝合，系统扫 stitch point × stitch layer family × loss × 下游任务；提出 **Self-Stitch baseline**（把同一个 stitch 模块插进 source-only / target-only 模型）来排除"增益只来自新增容量" | (1) **stitch layer 的训练目标是关键**：直接 feature-match at stitch point 或 end-to-end task loss 在**浅缝点严重失败**；正确做法是 **在 target 倒数第二层做 feature-matching (FFM) 预训 stitch，再 task-loss 微调**。(2) 这样异构 VFM **可靠可缝**。(3) **深缝点上缝合模型可超过任一成分模型**，分类 **+0.7% ~ +5.5% 超过 self-stitch**，ADE20K 分割 **+0.5~0.7 mIoU**。(4) stitch layer family：**两层 MLP(ReLU) 最好**，linear 次之，**LoRA-on-source-layer（表达力最强）反而不如 MLP** —— 作者解释为"stitching 可能受益于 controlled mismatch"。(5) 明确指出前人 0-10% 精度下降的乐观结论是因为 source/target/stitch 都在同一数据集上训评 | **★最强竞品（vision 侧）**：**我们想做的东西，在 vision 上已被完整做过、且做出了正结果和方法论**。我们如果只是"换到 LLM"，novelty = 模态迁移，很薄。**同时它是我们最有用的方法论来源**：FFM 初始化、Self-Stitch baseline、MLP > LoRA、深缝点优于浅缝点，都是可直接搬的先验 |

### 2.2 LLM 上的层拼接 / 深度拼接

| # | 标题 | ID | venue | 方法 | 关键数字 | 关系 |
|---|------|----|-------|------|---------|------|
| M1 | **Investigating Representation Universality: Case Study on Genealogical Representations** | arXiv:2410.08255 | 仅 arXiv（未查到 venue；arXiv comments 为空） | **直接做跨家族 LLM stitching**：`B∘A = U_B (∏_{i=m-l+1}^{m-1} K_i) S(Λ) (∏_{i=0}^{k-1} H_i) E_A` —— A 的前 k 层 + **单个线性 stitch S(Λ)** + B 的后 l 层，只训 S 最小化 next-token CE。**跨 tokenizer 处理方式：`v_i` 取"字符串 v_i v_{i+1}… 用 B 的 tokenizer 切出的第一个 token"** | 模型：OPT(1.3/2.7/6.7B)、Pythia(410M/1.4B/2.8B)、Mistral-7B-Instruct、LLaMA-3.1-8B-Instruct（410M–8B）。训练：stitch 10,000 步，lr 1e-3 线性衰减，wd 1e-4，Pile，2048 tokens，2000 test 样本。**核心结论：早-中层跨模型对齐好，后期层对齐差**；随缝合点向后移，**ICL 精度下降**；把 A 的 embedding 层直接接到 B 的下游层 test loss 仍很低；**"mid-layer 表征常常兼容，但缝进后期层 loss 更高"**；**可以把 A 的 mid-layer 缝到 B 的 early layer**（如 Pythia-410M 的 0–15 层 → Pythia-1.4B 的 2–23 层）。Fig.6/7 的 test loss 数值：**cut position 从 0 到 1，loss 从 ~2.0-2.5 升到 4-6，原模型平均 loss 是水平虚线（明显低于大部分缝合点）**。**Limitations 自陈：假定线性映射够用，非线性特征（circular/helical）会被误判为不等价** | **★最强竞品（LLM 侧，机制方向）**：**跨家族 LLM 层缝 + 线性 adapter 已经做过了**。但它 (a) 目标是 interpretability，不是提升 benchmark；(b) **只用单个线性层**，不是 1-2 层 transformer；(c) **只报 test loss / ICL 退化，没有任何"缝出来比两个源模型都强"的证据**；(d) 正文图只展示 same-family 曲线（OPT↔OPT、Pythia↔Pythia），跨家族对（Mistral/LLaMA）**只在方法节声明用了，我没在正文/图注中 fetch 到明确的跨家族数值曲线** → 这既是它的弱点，也是我们可占的空隙 |
| M2 | SOLAR 10.7B: Scaling LLMs with Simple yet Effective Depth Up-Scaling (DUS) | arXiv:2312.15166 | **NAACL 2024 Industry Track**（aclanthology.org/2024.naacl-industry.3/；Crossref 同） | **同一模型自我复制拼接**：n=32 层的 base 复制一份，原件去掉最后 m=8 层、副本去掉前 m=8 层，两段拼成 s=2(n−m)=48 层，**然后 continued pretraining** | **明确说"depthwise scaled model 的性能一开始掉到 base LLM 之下"，靠 continued pretraining 快速恢复**；作者关于 seam 的分析很重要：简单 repeat（n→2n）会在 seam 处产生**最大 layer distance**，"discrepancy 可能太大以致 continued pretraining 难以快速解决"，所以 DUS **牺牲 2m 个中间层来减小 seam 处的 discrepancy**；作者把 DUS 成功归因于"减小 discrepancy + continued pretraining"。SOLAR 10.7B-Instruct 的 H6 超过 Mixtral-8x7B-Instruct-v0.1 和 Qwen 72B；ablation 里 SFT v1 H6=69.15、SFT v2=69.21、DPO v1=73.06（SFT base 70.03）、DPO v1+v2 merge=73.21（**比 DPO v2 更差**，且"加 Synth. Math-Alignment 带来的 GSM8K 增益消失了"） | **方法学相关，非竞品（同模型自缝）**，但**给出对我们最要命的先验**：即使是**同一个模型自己**拼接，seam 处的 discrepancy 也大到要牺牲 16 层 + 一整轮 continued pretraining 才能恢复。跨模型的 seam discrepancy 必然远大于此 |
| M3 | Arcee's MergeKit: A Toolkit for Merging Large Language Models | arXiv:2403.13257 | **EMNLP 2024 Industry Track**（aclanthology.org/2024.emnlp-industry.36/ 标题一致） | 开源 merging 工具库；**对 passthrough 的原文说明我逐字 fetch 到了**："building larger models without performing any parameter-space combination. Referred to online as **'FrankenMerging'**, the passthrough method in MergeKit allows the **piecewise combination of layers from multiple models** into a new model of unusual size. This technique is behind the popular model **Goliath-120b**, and is the **first step of the Depth Up-Scaling technique of (Kim et al., 2023)** used for SOLAR-10.7B and Yi-9B" | **⚠️ 关键发现：MergeKit 论文正文对 passthrough/FrankenMerging 只有这一段描述性文字，没有任何 passthrough 的定量评测。** 全文 492 行只有 4 处提到 passthrough/franken/depth-up。**"社区有无系统评测"→ 在这篇论文里：没有。** 论文只笼统说 "thousands of models have been merged by the community, leading to some of the world's most powerful open-source checkpoints, as assessed by the Open LLM Leaderboard"，未拆分归因到 passthrough | **既是我们想法的"社区先例"，也是可攻的空隙**：passthrough 缝层**没有 stitching layer，也不训练**，纯拼；且**学术界从未系统评测**。这是唯一一个"我们的 novelty 有落点"的方向 |
| M4 | Goliath-120B / MegaDolphin 等社区 franken-model | — | **未能核实，不得引用为学术工作** | — | 我在 arXiv API 上用 `all:"frankenmerge"`、`all:"frankenmerging"`、`all:"passthrough merge"`、`all:"Goliath"+merge`、`ti:"Franken*"` 五轮检索，**返回 0 篇论文**（唯一命中 `ti:Franken` 的是 arXiv:2502.08037 Franken-Adapter，做的是 embedding surgery，与层拼接无关）。**Goliath-120b 只在 MergeKit 论文中作为 footnote-3 的 HuggingFace 链接被提及（我 fetch 到了该句），本身没有论文，也没有我能核实的系统 eval。** | **结论：社区 franken-model 无正式论文、无可引用的系统评测。** 写作时只能引 M3 那句话，**不得把社区模型的表现当证据** |
| M5 | Transformer Layer Injection (TLI) | arXiv:2410.11654 | 仅 arXiv（未查到 venue） | 改进 DUS：**每 K 层注入一个新层**，而非在中间做一个大 seam | 声称在 LLaMA3 1B/3B/8B 上 initialization 更好、需要更少训练步、KoBEST/KMCQA 更优，**"models performing effectively even without additional training"**（⚠️ 我只 fetch 到摘要，**未核实其实验细节与表格**） | **正交（同模型自缝的变体）**；对我们有用的一条是"把 discrepancy 分摊到多个小 seam 好于一个大 seam"——如果我们要缝，**多点小 seam 可能优于单点大 seam** |
| M6 | Evolutionary Optimization of Model Merging Recipes | arXiv:2403.13187 | **Nature Machine Intelligence 2025**（arXiv jref 字段 "Nat Mach Intell (2025)"；Crossref 确认） | 进化搜索同时在 **PS（权重空间）** 和 **DFS（data flow space = 层的推理路径）** 上优化。**DFS 就是跨模型层拼接**："after the i-th layer in model A, a token may be directed to the j-th layer in model B"，**保持所有层权重不动**，只搜索 indicator array I（layer 包含/排除）+ **一个 M×M 的 scaling 矩阵 W 来缓解"层收到 OOD 输入"** | 源模型：shisa-gamma-7b-v1(JA)、WizardMath-7B-V1.1、Abel-7B-002，**全部 fine-tune 自 Mistral-7B-v0.1（即同一初始化、同 tokenizer）**。**MGSM-JA / JP-LMEH**：源模型 9.6/66.1、18.4/60.1、30.0/56.5；**PS merge = 52.0 / 70.5**；**DFS merge (10B) = 36.4 / 53.2**；PS+DFS = 55.2 / 66.2。**★负结果原文："our preliminary studies show that swapping a pair of neighboring layers in a language model makes its performance drop"**；且**必须**引入 W 缩放才能缓解分布偏移 | **★★ LLM 侧最致命的对照**：**层路径拼接（DFS）在同族、同 tokenizer、进化搜索加持、还有 W 缩放补偿的最有利条件下，MGSM-JA 只有 36.4，被权重平均类 merge（52.0）碾压 16.6 分，JP-LMEH 53.2 甚至低于所有源模型（56.5-66.1）**。这直接构成我们的 kill risk #1 |
| M7 | Layer Swapping for Zero-Shot Cross-Lingual Transfer in LLMs | arXiv:2410.01335 | **ICLR 2025 Spotlight**（OpenReview api2） | 从**同一个 pretrained 模型**（Llama 3.1）fine-tune 出 math expert（英文数学）和 language expert（目标语通用指令），**把 math expert 的 top / bottom transformer 层直接替换成 language expert 的层**（含 transition zone 做插值），**不训练任何 stitch** | MGSM 上**比各 expert 与其他 merging 方法（TIES、model soup）平均高 10%**，覆盖 Swahili/Telugu/Bengali/Japanese；all-langs avg 表格中层交换 45.4-46.0 vs souping/TIES 37.5-43.0；**"there is a wide range of possible configurations in which this methodology is still very effective"**（配置不敏感）；解释：top/bottom 层是"natural language interfaces to broader model intelligence"，中层是语言无关的推理核心 | **强竞品（同族版）**：**"取 A 的一段 + B 的一段来博采众长"在 LLM 上已经 work 并进了 ICLR Spotlight** —— 但**前提是同一个 base 微调出的 experts（权重可直接互换、无需 stitch）**。它反证了两件事：(a) 层级功能分工（外层=语言接口、中层=推理核心）真实存在且可利用；(b) **一旦同初始化，连 stitch layer 都不需要** → 我们加 stitch layer 的价值只在"跨初始化/跨家族"时才成立 |
| M8 | Rethinking the Multilingual Reasoning Gap with Layer Swap | arXiv:2605.26735 | 仅 arXiv | 同上思路的 2026 版：Qwen3-8B-Base 上 fine-tune native / English-pivoted reasoning specialists，**把英文 specialist 的中层推理层换进各语种 specialist** | native reasoning gap 从"大幅退化"缩到 **1.9–3.5%**；权重空间分析：**中层 fine-tuning 更新对齐、外层发散** → "language-agnostic reasoning core surrounded by language-specific layers" | **同族竞品的延伸**；对我们有用的是它给出的**层功能分区的量化证据** |
| M9 | BTS: Harmonizing Specialized Experts into a Generalist LLM (Branch-Train-Stitch) | arXiv:2502.00075 | **EMNLP 2025**（Crossref: "Proceedings of the 2025 Conference on EMNLP"） | **"stitch layer"这个词在 LLM 上已经被占用**：seed LLM 分支成 domain experts（continual pretrain），然后在 **冻结的 experts 与冻结的 seed（hub）之间插入 lightweight stitch layers**（交替 Experts-into-Hub / Hub-into-Experts，最后一层必为 Experts-into-Hub），**只训 stitch layers**，小量 expert 域数据 | 2.7B seed + 3 experts（code/multiling/math），**4 个 stitch layer**。MMLU/BBH/MBPP/HE/Flores(S)/Flores(T)/GSM8K/MATH/Avg：Seed 28.4/35.6/27.0/20.7/29.5/35.7/10.5/4.82/**24.0**；Code Expert Avg 25.4；Math Expert Avg 25.4；**BTS 35.8/36.9/32.2/22.0/30.9/36.2/20.2/10.6/Avg 28.1**（merged 模型中最佳；BAM 27.8、BTX Soft 27.4、Model Soup 25.7、Expert Routing 24.8）。**注意 BTS 是并行 expert + gating，不是 depth-wise 串行拼接**；论文自陈局限："connections between expert layers; this restricts the model's overall expressivity" | **★术语与定位上的直接竞品**：**"lightweight stitch layers + 冻结骨干 + 只训 stitch"这个卖点已被 BTS 用掉了**（EMNLP 2025）。差异：BTS 是**同 seed 派生 + 并行 hub-expert 结构**，我们是**跨家族 + 串行深度拼接**。写作时必须显式区分，否则审稿人直接判 incremental。**同时它的 Avg 28.1 vs seed 24.0（+4.1 绝对）给了"stitch-only 训练能拿多少"的量级参考** |
| M10 | Unconstrained Model Merging for Enhanced LLM Reasoning | arXiv:2410.13699 | 仅 arXiv（未查到 venue） | 同时支持同构（fine-grained layer-wise 权重 merge）和**异构**（走 instruction-response 数据上的**概率分布知识**，即蒸馏路线）架构 merge | 7 benchmarks × 9 reasoning LLM；声称"combinatorial reasoning emerges from merging which surpasses simple additive effects"（⚠️ **我只 fetch 到摘要，未核实具体表格数字**） | **竞品（异构 merge 空间的占位者）**：注意它对**异构**的解法也是**蒸馏/分布层面**，**不是层拼接** —— 这佐证了"异构模型融合，学界的默认解法是蒸馏而非缝层" |

### 2.3 表征空间跨模型对齐（能不能用线性/低秩映射对齐？）

| # | 标题 | ID | venue | 方法 | 关键数字 | 关系 |
|---|------|----|-------|------|---------|------|
| A1 | Relative representations enable zero-shot latent space communication | arXiv:2209.15430 | **ICLR 2023 notable top 5%**（arXiv comments 字段） | 用"样本对一组固定 anchor 的 latent 相似度"作为表征，**无需训练即获得对 latent isometry / rescaling 的不变性**，从而实现 **zero-shot model stitching** | 跨 images/text/graphs、跨 CNN/GCN/transformer、跨 classification/reconstruction 验证 | **先驱（对齐机制的另一条路）**：**它给出"不学映射也能缝"的选项**——如果我们要证明必须学 1-2 层 transformer，得先打掉 relative-rep 这个更便宜的 baseline |
| A2 | MoSECroT: Model Stitching with Static Word Embeddings for Crosslingual Zero-shot Transfer | arXiv:2401.04821 | 仅 arXiv | **就是用 relative representations 把 source-language PLM 与 target-language static embedding 缝到公共空间**，然后换 embedding 层做 zero-shot 跨语迁移 | **★这是一篇负结果论文（原文自陈）**："although our proposed framework is competitive with **weak** baselines when addressing MoSECroT, it **fails to achieve competitive results compared with some strong baselines**. In this paper, we attempt to explain this **negative result**" | **★关键负结果**：relative-representation 式跨模型缝合（在 embedding 层这个"最好缝"的位置）**打不过强 baseline**。写作必须引 |
| A3 | Transferring Linear Features Across Language Models With Model Stitching | arXiv:2506.06609 | 仅 arXiv | **两个独立 Linear 层（bias 初始化为 0）** 在两个 LM 的 **residual stream** 之间做 affine 映射；用它迁移 SAE / probe / steering vector | **范围限制原文自陈："we train stitches on general internet text data between models in the same family with the same tokenizer. Natural follow ups are verifying the findings in a cross-family setting"** → **跨家族没做**。数据：activation 取 OpenWebText 前 180k 样本、ctx 512（Gemma 128），mask 掉 special token。**Gemma-2B→9B 的 transferred SAE FUV 0.21–0.42**（"better than random but certainly worse than a fully trained SAE"）；小→大 transferred SAE 作初始化可省 **50%** SAE 训练 FLOPs；transferred probe **retrain 后几乎恢复 ground truth**，不 retrain 也显著优于随机；作者给的部分解释：迁移后的 SAE 权重矩阵 **rank ≤ d_A < d_B**，只能检测子空间 | **★对我们最重要的量化证据 + 我们的空隙**：(a) 同族同 tokenizer 下 **affine 映射就够用**（残差流线性结构强）；(b) **但迁移质量有明确损失（FUV 0.21-0.42），且明确指出跨家族未验证**；(c) 它提供了"仅两个 Linear + bias"的最小 stitch 配方 |
| A4 | Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping for Prefill Reuse | arXiv:2608.03893 | 仅 arXiv（2026-08） | 用 **per-head ridge 回归**把 source 模型的 KV cache 映射到 target 模型（先剥 RoPE 使映射 position-free），500 条 FineWeb-Edu×1024 token 校准 | **Qwen3 14B→32B：单个 source 层解释 target keys 方差的 56%、values 32%；用多个 source 层升到 79% / 65%**。跨三个家族六对：线性 mapper 在**四对上保留 73–98% 的 standalone-prefill 精度，但两对严重退化**；**非线性 MLP 在失败对上最多挽回 +37 pp HellaSwag** | **★最好的"线性够不够"量化答案**：**同家族跨尺度就已经有 21-44% 方差解释不了**，且**六对里两对线性映射直接崩，需要非线性才救回来**。这是"跨家族对齐必须非线性"的最强正面论据，也是"对齐本身很脆"的警告 |
| A5 | Harnessing the Universal Geometry of Embeddings (vec2vec) | arXiv:2505.12540 | 仅 arXiv | **无配对数据、无 encoder** 把文本 embedding 翻译到另一个空间 / 通用 latent | "high cosine similarity across model pairs with different architectures, parameter counts, and training datasets"（⚠️ 摘要级；我未 fetch 具体数值表） | **正交（embedding 层而非中间层）**，但**是"跨模型表征可对齐"的最强存在性证据**，值得引作动机 |
| A6 | mini-vec2vec: Scaling Universal Geometry Alignment with Linear Transformations | arXiv:2510.02348 | 仅 arXiv | vec2vec 的**纯线性**替代（伪并行匹配 → 变换拟合 → 迭代精化） | "exceeds the original instantiation of vec2vec by **orders of magnitude in efficiency**, while matching or exceeding their results"；学到的映射**是线性变换**（⚠️ 摘要级） | **正交 + 反向压力**：如果 embedding 空间的跨模型对齐能被**纯线性**搞定，审稿人会问"你的中间层为什么需要 1-2 层 transformer" |
| A7 | The Platonic Representation Hypothesis | arXiv:2405.07987 | 仅 arXiv（ICML 2024，⚠️ 我未在 arXiv 字段中核实到 venue，**引用时只写 arXiv**） | 论证不同模型的表征在收敛到共享的"platonic"统计模型 | 摘要级：模型越大，vision 与 language 模型度量数据点距离的方式越接近；**摘要明确提到"limitations and counterexamples"** | **正交（理论动机）** |

### 2.4 输出级 ensembling / 蒸馏融合（审稿人一定会问区别的）

| # | 标题 | ID | venue | 方法 | 关键数字 | 关系 |
|---|------|----|-------|------|---------|------|
| F1 | LLM-Blender: Ensembling LLMs with Pairwise Ranking and Generative Fusion | arXiv:2306.02561 | **ACL 2023 main**（arXiv comments: "Accepted to ACL 2023 (main conference)"） | PairRanker（cross-attention encoder 做候选两两比较）+ GenFuser（把 top-ranked 候选融成更好输出）；发布 MixInstruct benchmark | 声称显著超过单模型与 baseline（⚠️ 我只 fetch 到摘要，未核实表格） | **正交（output-level）**：**推理时要跑 N 个模型**；我们是单次 forward 的单模型。这是我们的效率论点 |
| F2 | **Knowledge Fusion of Large Language Models (FuseLLM)** | arXiv:2401.10491 | **ICLR 2024**（PDF 首页页眉逐字："Published as a conference paper at ICLR 2024"） | **用 source LLM 的生成分布做蒸馏**把异构 LLM 的知识融进单个 target LLM。**跨 tokenizer 解法：把 Fu et al. 的 exact-match (EM) token 对齐换成 minimum edit distance (MinED) 对齐** | 源模型 Llama-2 / MPT / OpenLLaMA（架构不同）→ target Llama-2 7B，42 个任务。**BBH（27 tasks）：FuseLLM 相对 Llama-2 平均相对增益 +5.16%**（对照：单纯 continual pretrain 的 Llama-2 CLM 只有 **+1.86%**，且"modest and inconsistent"）；**CommonSense 5 tasks：+1.25%**（Llama-2 CLM 仅 +0.16%；ARC-c +2.40%、OpenBookQA +2.71%）；**MultiPL-E：10 个任务赢 9 个**，R 的 pass@1 4.97→5.84；**vs 普通 KD 对照：BBH 上 FuseLLM 5.16% vs Llama-2 KD 2.97%**；Table 8：TrivialQA 52.46→54.49(+3.87%)、DROP 27.25→28.97(+6.31%)、LAMBADA 73.28→73.72(+0.60%) | **★★ 最可能的竞品（用户判断正确）**：**同样解决"异构 LLM 博采众长"，同样处理跨 tokenizer，已在 ICLR 2024。** 与我们的精确区别：**FuseLLM 用蒸馏（把知识搬进一个既有 target 的权重里，需要训练 target），我们用结构拼接（保留双方权重、只加 stitch）。** **它的增益量级是我们的标杆：+1.25% ~ +5.16% 相对（多数 <5% 相对，绝对值多在 1-3 分）** —— 这既说明"异构融合的天花板本来就不高"（对我们有利：不需要很大增益），也说明"要超过它不容易" |
| F3 | FuseChat: Knowledge Fusion of Chat Models | arXiv:2408.07990 | 仅 arXiv | 两阶段：先 pairwise 知识融合把异构 source chat LLM 蒸成同结构 target（**statistics-based token alignment** 作为跨结构基石），再在**参数空间** merge（融合系数按 fine-tune 前后参数更新幅度定） | 6 个 source（OpenChat-3.5-7B、Starling-LM-7B-α、NH2-SOLAR-10.7B、InternLM2-Chat-20B、Mixtral-8x7B-Instruct、Qwen-1.5-Chat-72B）；AlpacaEval 2.0 + MT-Bench 上 FuseChat-7B 超过各尺寸 baseline，**"comparable to Mixtral-8x7B-Instruct, approaches GPT-3.5-Turbo-1106 on MT-Bench"**（⚠️ 摘要级，未核实表格） | **竞品（FuseLLM 的 chat 版）**：注意它的路线是 **蒸馏对齐 → 再权重空间 merge**，**完全绕开了层拼接**。这是当前 SOTA 范式，我们必须解释为什么要偏离它 |
| F4 | LLM Augmented LLMs: Expanding Capabilities through Composition (CALM) | arXiv:2401.02412 | **ICLR 2024 poster**（OpenReview api2） | **在两个冻结模型之间引入 cross-attention** 组合它们的表征；只加少量参数与数据；**原模型权重完全不动** | PaLM2-S + 低资源语言小模型 → 低资源语言翻译/算术推理 **绝对提升最多 +13%**；PaLM2-S + code 模型 → 代码生成/解释 **相对提升 +40%**，"on-par with fully fine-tuned counterparts" | **★★ 结构上最接近我们的竞品（ICLR 2024）**：**冻结两个异构模型 + 只训中间的轻量连接层 + 目标是获得双方能力**——这就是我们的骨架，只差"cross-attention 并联 vs depth-wise 串联"。**审稿人几乎必然问"你和 CALM 的区别是什么"**。差异：CALM 保留两条完整前向（推理成本 ≈ 两个模型之和），我们串联只跑一遍（部分层）→ **效率是我们唯一清晰的差异化点** |
| F5 | Model Composition for Multimodal LLMs | arXiv:2402.12750 | 仅 arXiv（⚠️ 我只见到检索标题，**未 fetch 摘要，不得引用内容**） | — | — | 可能相关，**未核实** |
| F6 | Cool-Fusion: Fuse Large Language Models without Training / Token-level Ensembling of Models with Different Vocabularies | arXiv:2407.19807 / arXiv:2502.21265 | 仅 arXiv（⚠️ **只见检索标题，未 fetch 摘要**） | — | — | output/token-level 路线，**未核实，仅记录以备后查** |

### 2.5 跨 tokenizer / vocabulary transfer（跨家族拼接的前置问题）

这条线**极其拥挤**（我在 arXiv 上一次检索就命中 12+ 篇，2024–2026 持续高产），说明"跨 tokenizer 对齐"本身已不是可发论文的 novelty，而是一个有成熟工具的**已解决前置步骤**。

| # | 标题 | ID | venue | 方法 | 关键数字 | 关系 |
|---|------|----|-------|------|---------|------|
| T1 | Universal Cross-Tokenizer Distillation via Approximate Likelihood Matching | arXiv:2503.20083 | **NeurIPS 2025**（arXiv comments 字段） | 原理性跨 tokenizer 蒸馏（近似似然匹配）；"first to enable effective distillation across **fundamentally different** tokenizers" | 三个用例：tokenizer transfer as self-distillation（含 subword→byte-level）、把大数学 LLM 蒸进小的异 tokenizer 通用模型、训 embedding-prediction hypernetwork 做 training-free tokenizer transfer（⚠️ 摘要级） | **正交（可作为我们的前置工具）**：**如果要跨 tokenizer，标准解法是先做 tokenizer transfer，不是硬缝** |
| T2 | Towards Cross-Tokenizer Distillation: the Universal Logit Distillation Loss for LLMs | arXiv:2402.12030 | 仅 arXiv | 用 optimal transport 做 universal logit distillation | ⚠️ 仅标题+检索命中，**未 fetch 摘要** | 正交 |
| T3 | Multi-Level Optimal Transport for Universal Cross-Tokenizer KD | arXiv:2412.14528 | 仅 arXiv | 同上系列 | ⚠️ 仅标题，**未核实** | 正交 |
| T4 | Franken-Adapter: Cross-Lingual Adaptation of LLMs by Embedding Surgery | arXiv:2502.08037 | 仅 arXiv（33 pages） | **只做 embedding 手术**：给目标语造定制 vocab、只 tune embedding，再把这些 embedding 装到英文 instruction-tuned LLM 上做 zero-shot 迁移 | Gemma2 至 27B，**96 种语言最多 +20%**，**英文回退 <1%**；对数学优化 LLM 有 **+14%**（20 语言） | **正交但重要的对照**："只换 embedding、骨干不动"就已经能拿到 +20%。审稿人会问："既然换 embedding 就够，为什么要缝中间层？" |
| T5 | Training-Free Tokenizer Transplantation via Orthogonal Matching Pursuit | arXiv:2506.06607 | 仅 arXiv | ⚠️ 仅标题，**未核实** | — | 正交 |
| T6 | 其余同族（CTPD、Contextual Dynamical Mapping、Byte-Level Interface、DWA-KD、X-Token、SimCT、Breaking the Tokenizer Barrier 等 ~10 篇，2025-2026） | 见 §附录检索日志 | 多为 arXiv | 均为 cross-tokenizer distillation 变体 | ⚠️ **仅标题级，未 fetch，不得引用内容** | **仅用于说明"该子问题已高度饱和"** |

---

## 3. 概念定位：我们与五类工作的精确边界

设 A、B 为两个预训练模型，`L_A^{0:k}` 表示 A 的第 0..k 层，`S` 为可训 stitching 模块。
我们的方法：`y = Head_B ∘ L_B^{j:n_B} ∘ S ∘ L_A^{0:k} ∘ Emb_A(x)`，**A、B 全冻结，只训 S（1-2 层 transformer）**。

### (a) vs 权重平均类 merging（Model Soup / Task Arithmetic / TIES / DARE；工具见 M3）
- **他们**：在**权重空间**做 `θ_merged = f(θ_A, θ_B)`，**要求同架构、同初始化（linear mode connectivity）**，产出模型**层数不变**。
- **我们**：在**表征空间/深度方向**做函数复合，**不碰任何原始权重**，**允许跨架构、跨 d_model、跨层数、跨 tokenizer**，产出模型层数 = k + (n_B − j) + |S|。
- **边界一句话**："merging 是 θ 空间的插值，需要 A、B 在同一 basin；我们是函数空间的串联，只需要 A 的第 k 层输出流形能被映射进 B 的第 j 层输入流形。"
- **⚠️ 但审稿人的反击已经存在**：M6（Sakana）在同族条件下**同时**跑了 PS 与 DFS，**DFS(层拼接) 36.4 被 PS(权重空间) 52.0 完败**；M7（Layer Swapping, ICLR'25 Spotlight）证明**同初始化时连 S 都不需要**、直接换层就赢 TIES/soup 10%。所以我们必须把战场**严格限定在"跨初始化/跨家族"**——那是 merging 类方法**在定义上无法进入**的区域。这是我们唯一的合法领地。

### (b) vs output-level ensembling（F1 LLM-Blender / F6 token-level ensembling）
- **他们**：跑完 N 个模型，在 logits 或文本层面聚合。**推理成本 ≈ Σ cost(model_i)**，且需要 N 份权重常驻。
- **我们**：单条前向，成本 ≈ `k` 层 A + `n_B − j` 层 B + |S| ≈ **一个模型的量级**。
- **边界**："ensembling 在输出空间投票，成本乘 N；我们在中间层拼接，成本仍是 O(单模型)。" **这是我们最干净、最不可争议的差异化点。**

### (c) vs 蒸馏融合 FuseLLM / FuseChat（F2 / F3）
- **他们**：以 source LLM 的**生成分布**为监督，**训练一个 target 模型的全部权重**；跨 tokenizer 靠 token 对齐（MinED / statistics-based）。产物是**一个模型，源模型的结构与权重都不出现在产物里**。FuseChat 甚至在蒸完之后**再回到权重空间 merge**。
- **我们**：**不训练任何 backbone 权重**，源模型的层**物理地留在产物中**。训练成本 = |S|（百万级参数）vs FuseLLM 的 7B 全参 continual training。
- **边界**："FuseLLM 把 B 的知识**蒸**进 A 的权重（需全参训练、知识有损压缩）；我们把 B 的层**装**进 A 的前向（零权重训练、知识无损但需要接口对齐）。"
- **量级标杆（必须内化）**：FuseLLM 在 BBH 上只拿到 **+5.16% 相对**（CommonSense +1.25% 相对）。**说明"异构 LLM 融合"这个问题的天花板本身就不高**。好消息：我们不需要打出大数字就能 competitive；坏消息：**几个点的增益极易被"stitch layer 自身的新增容量"解释掉** → **必须做 M5(VST) 的 Self-Stitch baseline**（把同一个 S 插进 A-only 和 B-only 的同位置），否则结论不成立。

### (d) vs depth up-scaling（M2 SOLAR / M5 TLI / M3 passthrough-FrankenMerge）
- **他们**：**同一个模型**（或同一家族）自我复制/分段重排来**加深、加大**模型，目标是 **scale up**；**没有 stitching layer**（DUS 靠去掉 2m 中间层来减小 seam 落差，靠 **continued pretraining 全参恢复**；passthrough 甚至完全不训练）。
- **我们**：**两个不同来源**的模型，目标是**能力互补**而非 scale；用**可训 S** 替代"全参 continued pretraining"来消化 seam。
- **边界**："DUS 是 self-concatenation for capacity，我们是 cross-model concatenation for capability；DUS 用 continued pretraining 消化 seam，我们用 1-2 层可训 stitch 消化 seam。"
- **⚠️ SOLAR 给我们的最硬警告**（原文）：**同一个模型自缝，性能"initially drops below that of the base LLM"，要牺牲 16 层 + 一整轮 continued pretraining 才恢复**；且 seam 处的 layer distance 必须被刻意压小，因为"discrepancy 可能太大以致 continued pretraining 难以快速解决"。**跨模型的 seam discrepancy 一定远大于同模型自缝** —— 而我们**只训 1-2 层 stitch，不做 continued pretraining**。这是 kill risk #2。

### (e) vs vision model stitching（L1 / L2 / L3 / L4 / L5）
- **L1/L2**：stitching 是**诊断工具**，S 被刻意做成低容量（1×1 conv），明言"only meant to align representations, **rather than improving the model**"。
- **L3/L4（SN-Net/ESTA）**：**family 内**缝，目标是弹性部署，性能在 anchor 之间**插值**。
- **L5（VST, CVPR 2026）**：**这正是我们想做的事，只是在 vision 上**——跨异构 VFM（CLIP/DINOv2/SigLIP2/DINOv3）缝，明确目标是"integrating complementary VFM strengths"，并证明**深缝点能超过任一成分模型**（+0.7~5.5% over self-stitch）。
- **边界**："我们把 VST 的问题从 vision encoder（无自回归、无 tokenizer、有共享的 patch-token 语义）搬到 autoregressive LLM（有 KV cache、有 RoPE、有不同 tokenizer、有生成一致性要求）。"
- **⚠️ 这个边界是我们的 novelty 与最大风险同时所在**：模态迁移本身在顶会通常不够，除非我们能证明 **LLM 引入了 vision 上不存在的新困难**（tokenizer 不同、位置编码不同、生成过程会放大 stitch 误差），且给出针对这些困难的新机制。

---

## 4. Kill risks（按严重度排序，全部有文献依据）

### 🔴 R1（致命）：LLM 上"跨模型层拼接"已被证明**弱于权重空间 merge**
- **证据**：M6（Sakana, Nature MachIntell 2025）**同一批源模型、同一评测**下：DFS（层路径拼接，10B）**MGSM-JA 36.4 / JP-LMEH 53.2**；PS（权重空间 merge，7B）**52.0 / 70.5**。**层拼接落后 16.6 分，且 JP-LMEH 53.2 低于全部三个源模型（56.5 / 60.1 / 66.1）**。而这已经是最有利条件：同族、同 tokenizer、进化搜索、外加可学的 W 缩放矩阵。
- **对我们的含义**：审稿人只需引这一个表就能问"为什么你的串联缝合会比 PS merge 好？"我们**必须**要么把场景限定在 PS merge 结构上不可用的地方（跨家族/跨 d_model/跨 tokenizer），要么直接在同一 benchmark 上打赢 PS merge。
- **补充证据**：M6 原文另有 **"swapping a pair of neighboring layers in a language model makes its performance drop"** —— 连**同一模型内相邻两层交换**都会掉点。

### 🔴 R2（致命）：seam discrepancy 的代价远超 1-2 层 stitch 的容量
- **证据**：M2（SOLAR/DUS）**同一个模型自缝**，性能先掉到 base 之下，需要**去掉 2m=16 层**来减小 seam 落差 **+ 一整轮 continued pretraining** 才恢复；作者明说简单 repeat 的 seam"discrepancy 可能太大以致 continued pretraining 难以快速解决"。M6 需要**逐层可学的 W 缩放**才能缓解"层收到 OOD 输入"。L1 显示 **identity stitch → top-1 error > 99%**。
- **对我们的含义**：**"只训 1-2 层、backbone 全冻"很可能容量不足**。缓解方向：把 stitch 拆成多个小 seam（M5/TLI 的启发）、允许 stitch 附近少量层解冻、或按 M5(VST) 的 FFM 目标先做特征匹配预训。

### 🟠 R3（严重）：跨家族的深层表征对齐**已被量化为难**，而"深层"正是能力所在
- **证据链（三条独立）**：
  1. **L1（2015）**：Conv1/Conv2 处处可换，**Conv5 不完全可换**，跨任务深层"substantially less compatible"。
  2. **M1（2024，LLM）**：早-中层跨模型对齐好，**"缝进后期层 loss 更高"**；随缝合点后移 **ICL 精度下降**；Fig.6 的 test loss 从 ~2.0-2.5 一路升到 4-6，多数缝合点**高于原模型的平均 loss**。
  3. **A4（2026）**：**同家族跨尺度** Qwen3 14B→32B，多层 source 也只解释 target keys 方差 **79% / values 65%**；六对里**两对线性映射直接崩**，靠非线性 MLP 才挽回 +37pp HellaSwag。
- **张力**：我们的想法要"B 的 reasoning/knowledge"，那通常在 B 的**中后段**；但文献一致说**越深越难缝**。**"想要的能力所在的层"与"能对齐的层"存在系统性错位。**
- **反向利好**：**M5(VST) 恰恰相反**——它发现**深缝点更好**、浅层因编码 pretraining-specific 特征而"consistently underperform"。**这个 vision/LLM 的矛盾本身就是一个可发表的实证问题**（见 §5 候选 3）。

### 🟠 R4（严重）：增益极易被"stitch 自身容量 + 任务适配"解释掉
- **证据**：M5(VST) 明确指出前人"0-10% 精度下降"的乐观结论是因为 source/target/stitch 同数据集训评，"improvements could arise **simply from task adaptation in the stitch layer**"，因此专门设计 **Self-Stitch baseline**。L2 也明言 stitch 应低容量、"not meant to improve the model"。
- **对我们的含义**：**没有 Self-Stitch 对照的任何增益都不可信。** 我们必须跑 `A→S→A` 和 `B→S→B`（同 S、同位置、同数据），并且 **1-2 层 transformer 的容量本就不低（含 attention，可做 token 间交互）**，比 VST 用的 per-token MLP 更容易"自己学出能力"。反讽的是 **M5 发现 LoRA-on-source-layer（最像我们的 transformer stitch）反而不如两层 MLP**。

### 🟡 R5（中等）：Tokenizer 不同这一关**已有更便宜的标准解法**，硬缝没有优势
- **证据**：跨 tokenizer 蒸馏/vocab transfer 是一整个饱和赛道（T1 NeurIPS 2025 声称"first to enable distillation across fundamentally different tokenizers"，T2/T3 optimal transport，T4 Franken-Adapter 只换 embedding 就拿 96 语言 +20%、英文回退 <1%，T5 training-free transplantation，外加 ~10 篇 2025-2026 变体）。M1 处理跨 tokenizer 的方式是粗暴的"取 B tokenizer 切出的第一个 token"。A2（MoSECroT）在 embedding 层缝合**给出负结果，打不过强 baseline**。
- **对我们的含义**：审稿人会说"跨 tokenizer 请用 T1/T4 先对齐，别混进 stitching 的贡献"。**建议第一版直接选同 tokenizer 或已 transplant 到同 tokenizer 的模型对，把 tokenizer 问题隔离掉**，否则实验会变成"stitch 的功劳 vs tokenizer 对齐的功劳"分不清。

### 🟡 R6（中等）：核心卖点的**术语与结构位置都已被占**
- "**lightweight stitch layers + 冻结骨干 + 只训 stitch**" = **BTS（EMNLP 2025，M9）**，连名字都一样。
- "**冻结两个异构模型 + 只训中间轻量连接 + 获得双方能力**" = **CALM（ICLR 2024，F4）**，只是用 cross-attention 并联。
- "**跨异构 foundation model stitching 以整合互补强项 + 系统协议 + 能超越成分模型**" = **VST（CVPR 2026，M5）**，只是在 vision。
- **对我们的含义**：**必须在 abstract 第一段就把这三者划清**，且差异必须是**机制性的**（串联 vs 并联的推理成本；跨初始化 vs 同 seed；autoregressive vs encoder），不能只是"我们换了个模态/任务"。

### 🟡 R7（中等）：收益上限本来就低
- **证据**：F2（FuseLLM，ICLR'24）异构融合的增益是 **BBH +5.16% 相对 / CommonSense +1.25% 相对**（绝对多在 1-3 分）；M9（BTS）Avg 24.0→28.1；M5(VST) 分类 +0.7~5.5%。
- **对我们的含义**：**别指望"博采众长"能出大数字**。实验设计必须能在 1-3 分的量级上做出统计显著性（多 seed、多 benchmark、误差棒），否则结论站不住。**这一条应转达给负责 benchmark 设计的 agent。**

### ⚪ R8（低，但要防）：没有"社区 franken-model 有效"的可引用证据
- **证据**：五轮 arXiv 检索（frankenmerge / frankenmerging / passthrough merge / Goliath+merge / ti:Franken*）**返回 0 篇相关论文**；MergeKit 论文对 passthrough 只有一段描述性文字、**零定量评测**；Goliath-120b 仅以 HF 链接出现在 footnote。
- **对我们的含义**：**不得用"社区 franken-model 很成功"当动机**（无证据，会被打）。**反过来，"passthrough/FrankenMerging 被广泛使用却从未被系统评测"是一个真实、可核实、可写进 intro 的空白**。

---

## 5. 如果要做，novelty 应该定位在哪（3 个候选表述，按我推荐的顺序）

### 候选 1（推荐）：把 franken-merge 从民间玄学变成有 stitch layer 的可控方法，并系统证伪/证实它
> **"FrankenMerging 被社区广泛使用（Goliath-120b、SOLAR 的第一步）却从未被系统评测；我们给出第一个 LLM 跨模型 depth-wise stitching 的系统研究，并证明：一个 O(1) 层的可训 stitch layer 能否补上 seam discrepancy，取决于 [缝合深度 × stitch 目标函数 × 家族距离] 这三个可测因素。"**
- **为什么站得住**：R8 的空白是真实且可核实的（M3 原文 + 零检索命中）；VST 已经证明这条路在 vision 上能出正结果，说明问题不是无解的；我们提供 LLM 侧第一个带 Self-Stitch 对照的系统协议。
- **必备实验（从文献直接推出）**：Self-Stitch baseline（M5）、vs PS merge / TIES / soup 同预算对照（打 R1）、vs CALM 同预算对照（打 R6）、缝合深度扫描（验 R3 的 vision-LLM 矛盾）、stitch layer family 扫描（linear / MLP / 1 层 transformer / 2 层 transformer，对照 M5 发现的"LoRA 不如 MLP"）。
- **可接受的结论包括负结果**："在 LLM 上，1-2 层 stitch 不足以跨家族缝深层，seam 代价随家族距离与缝合深度单调上升"——**这仍然是一篇有价值的论文**（正是 M5 在 vision 上做的事，只是结论相反）。

### 候选 2：把战场限定在 merging 结构上不可达的区域（跨 d_model / 跨层数 / 跨 tokenizer）
> **"所有权重空间 merging（soup/TIES/DARE/layer-swap）在定义上要求同架构同初始化；我们研究当 A、B 连 d_model 和层数都不同时，能力组合还能不能做——此时唯一可行的接口是可学的 depth-wise stitch。"**
- **为什么站得住**：直接绕开 R1（M6 的 DFS-vs-PS 对照在跨 d_model 时不成立，PS 根本跑不了）与 M7（Layer Swapping 要求同 base）。
- **风险**：F2/F3（FuseLLM/FuseChat）**已经在这个区域**（Llama-2/MPT/OpenLLaMA 架构不同）**并且用蒸馏解决了**。所以必须打的是**成本/知识保真**这条轴：我们训百万级参数、源权重零改动，FuseLLM 训 7B 全参且知识经过有损蒸馏。**这会把论文变成"效率论文"，需要严格的 FLOPs / 训练成本对照表。**

### 候选 3（最学术、最保险）：做 LLM 侧的"可缝性图谱"，把 vision 的矛盾结论作为核心问题
> **"vision 侧（VST, CVPR'26）发现深缝点最好、浅层反而差；LLM 侧（arXiv:2410.08255）发现早-中层最好、后期层缝不动。我们给出 LLM 的可缝性图谱：哪些层对、哪些家族距离、哪种 stitch 目标下可缝，并解释 autoregressive 生成为什么反转了 vision 的结论。"**
- **为什么站得住**：**这个矛盾是我 fetch 到的两篇原文之间的真实冲突**（L5 "shallow layers consistently underperform / deep stitch points can surpass either constituent model" vs M1 "representations align more closely in early to mid layers than in later layers … stitching them into later layers yields higher loss"），不是我构造的。解释它需要的机制（LLM 后期层专做 next-token prediction 而非语义构建；vision encoder 没有 unembedding 压力）是可实验检验的。
- **优点**：**结论无论正负都可发**，且天然需要我们本来就要做的那套扫描实验；同时它自动包含候选 1 的实验，可作为候选 1 的 fallback。

### 三个候选共同的**必须做**清单（否则任一表述都会被拒）
1. **Self-Stitch baseline**（A→S→A、B→S→B，同 S 同位置同数据）—— 来自 M5，防 R4。
2. **同预算 PS merge / TIES / soup 对照**（在架构允许的模型对上）—— 防 R1。
3. **同预算 CALM 式并联对照 + 推理 FLOPs 表** —— 防 R6/F4。
4. **stitch 目标函数消融**：直接 task loss vs 在 B 的倒数第二层做 feature matching（FFM）再微调 —— M5 明确说前者在浅缝点会崩。
5. **tokenizer 隔离**：第一版用同 tokenizer 对，跨 tokenizer 单独作为扩展章节 —— 防 R5。
6. **增益量级预期设为 1-5 分（相对 1-5%）**，配多 seed + 误差棒 —— 防 R7。

---

## 6. 附录：检索日志与未核实清单（诚实性记录）

**已验证证据来源（我亲自 fetch）**
- arXiv abs 页 `citation_title` + `blockquote.abstract`：1411.5908、2106.07682、2209.15430、2302.06586、2311.17352、2312.15166、2306.02561、2401.02412、2401.04821、2401.10491、2402.15414、2403.13187、2403.13257、2405.07987、2408.07990、2410.01335、2410.08255、2410.11654、2410.13699、2502.00075、2502.08037、2503.20083、2505.12540、2506.06609、2510.02348、2603.12433、2603.17512、2605.26735、2608.03893
- **全文（ar5iv 或 pdftotext）**：1411.5908（lenc.txt）、2106.07682（bansal_full.txt）、2312.15166（solar.txt）、2401.10491（fusellm.pdf + fusellm_ar5iv.txt）、2403.13187（evo.pdf/evo.txt）、2403.13257（mergekit.pdf/mergekit_pdf.txt）、2410.01335（ls.txt）、2410.08255（gen.pdf/gen.txt）、2502.00075（bts.pdf/bts.txt）、2506.06609（t2506.txt）、2603.12433（vst.pdf/vst.txt）
- **venue 核实**：ACL Anthology 直取页面（SOLAR = 2024.naacl-industry.3；MergeKit = 2024.emnlp-industry.36）；Crossref API（Lenc&Vedaldi = CVPR 2015 + IJCV 2018；SN-Net = CVPR 2023；BTS = EMNLP 2025；Sakana = Nature Machine Intelligence 2025-01-27）；OpenReview api/api2（Bansal = NeurIPS 2021 Poster；Layer Swapping = ICLR 2025 Spotlight；CALM = ICLR 2024 poster）；papers.nips.cc（Bansal 标题匹配）；arXiv comments 字段（relative reps = ICLR 2023 notable top 5%；SN-Net = CVPR 2023 Highlight；ESTA = CVPR 2024；VST = CVPR 2026；ALM distillation = NeurIPS 2025；LLM-Blender = ACL 2023 main）；FuseLLM = PDF 首页页眉 "Published as a conference paper at ICLR 2024"。

**⚠️ 只见摘要、未核实实验细节（引用时须标注）**：2410.11654(TLI)、2410.13699(Unconstrained Merging)、2505.12540(vec2vec)、2510.02348(mini-vec2vec)、2405.07987(Platonic，且 ICML venue 未核实)、2306.02561(LLM-Blender 表格)、2408.07990(FuseChat 表格)、2503.20083(T1 细节)、2502.08037(Franken-Adapter 表格)。

**⚠️ 仅检索标题命中、连摘要都未 fetch —— 不得引用内容**：2402.12750(Model Composition for MLLMs)、2407.19807(Cool-Fusion)、2502.21265(Token-level Ensembling)、2402.12030(ULD loss)、2412.14528(Multi-Level OT)、2506.06607(OMP tokenizer transplantation)、2504.11426(Dual-Space KD)、以及 §2.5 T6 列出的 ~10 篇 cross-tokenizer 变体。

**未能核实，不得引用**：
- Goliath-120B、MegaDolphin 等社区 franken-model 的**任何**定量表现（无论文；MergeKit 仅 footnote 链接）。
- "mergekit passthrough 的社区系统评测"——**不存在**（五轮检索 0 命中 + MergeKit 论文零定量）。
- 用户提到的 "Model Composition (arXiv:2402.15414)"：**该 ID 实际是 "Does Combining Parameter-efficient Modules Improve Few-shot Transfer Accuracy?"（LoRA 模块组合，与层拼接无关）**，我已 fetch 摘要确认。**原设想的 "Model Composition" 论文 ID 有误。**

**工具坑（供后续 agent 参考）**：Semantic Scholar API 在本代理下**全程 429**（多次重试 + 25s 间隔仍 429），完全不可用；改用 **arXiv API (`export.arxiv.org/api/query`) + Crossref + ACL Anthology + OpenReview api2** 组合可完全替代。`ar5iv` 对 2024 前论文效果好，2026 新论文需走 `arxiv.org/pdf` + `pdftotext -layout`。Sakana 那篇 PDF 有 Adobe-Japan1 字体报错但正文可提取。

---

**报告完** · 本报告未跑任何 GPU、未修改仓库任何既有文件。
