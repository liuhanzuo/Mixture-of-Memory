# DR3 对抗性 Review：层分配 / Adaptation Site 三篇

**全文均已抓取（arxiv HTML），精读完毕。三篇均标 preprint，Semantic Scholar 均未收录，无 journal_ref。**
日期：2026-08-06

---

## 一、arXiv:2605.11416 — "Freeze Deep, Train Shallow" (LayerTracer)
**预印本，2026-05-21 v2，Nanhu Research Institute / Shanghai University of Engineering Science。**

### A. 实验设置精确描述

| 维度 | 实际值 |
|---|---|
| 模型 | Qwen3-0.6B-Base（**主要**）；Qwen3-8B-Base、Qwen3-14B-Base（仅用于 LayerTracer 诊断可视化，无 CPT 实验）；Nemotron-12B-Base（Mamba 架构，仅 LayerTracer 可视化）；混合架构 Nemotron-Qwen3-Base / Qwen3-Nemotron-Base（各 550M，CPT 实验） |
| **所有 CPT 训练数据** | CCI3.0-HQ 中文语料 262B tokens（Table 3）；中文续预训练 |
| Freeze 边界 | **midpoint split = 50%**（Qwen3-0.6B 共 28 层 → 前 14 层可训/后 14 层冻结 或 相反）；附录 C.2 额外做了 33%/66% 边界确认，均正向 |
| 训练步数 / tokens | 1 epoch × 262B tokens；seq_len 4096，global_bs 64，约 1.09M steps；lr=3e-5，warmup 0.1 |
| Eval benchmark | **C-Eval、CMMLU**（两个均为中文多选题 MC，4 选 1）；zero-shot；logit-based + generation-based 双协议 |
| 核心数字 | Train-Shallow/Freeze-Deep 在 C-Eval 得 29.40/29.55（Generate/Logit），Freeze-Shallow/Train-Deep 得 25.39/25.55，Full 得 26.62/26.32（Table 1/2） |
| 参数量匹配 | **是**：三个臂各可训参数约 0.3B、0.3B、0.6B（Table 7）——前两个 partial 臂可训量一致（均 ~0.3B），与 full 不匹配但两 partial 臂之间匹配 |
| LR | 三臂统一 3e-5（Appendix A.2） |
| 显著性检验 | **无**。Appendix C.3 仅报告了"五次重复 std=0.00"——因为用 greedy/logit 无随机性，std=0 只说明**结果可复现**，**不是统计显著性检验** |

LayerTracer 诊断集（§4）用的数据是 AntSynNET 反义/同义词数据集，500 个 structured prompts，按 Ratio(l) 与 ΔJS(l) 指标分析。

---

### B. 方法论缺陷清单

**缺陷 1（严重）：C-Eval / CMMLU 绝对分均在 chance 附近，但没有报 chance 基线**

4 选 1 的 chance = 25.00%。文中三个策略的 C-Eval 平均分分别是：
- Full：26.62 / 26.32
- Train-Shallow/Freeze-Deep：**29.40 / 29.55**（最优）
- Freeze-Shallow/Train-Deep：25.39 / 25.55

全部在 25–30% 之间。最优策略比 chance 只高约 4.4pp（Generate）。Freeze-Shallow/Train-Deep 的 25.39 在 Generate 协议下几乎等于 chance 随机猜测。CMMLU 情况更差——Full = 25.11/25.27，逼近 25.00。

作者**从未报告 chance 基线（25%）**，也从未讨论"模型可能主要在 chance 水平上徘徊"。声称"15.72% 相对提升"（Table 1/2 注释及 §1）是用 Freeze-Shallow 与 Full 之间差异除以 Freeze-Shallow 算出来的相对值，但这个分母本身就在 chance 附近，用相对提升数字会夸大效果。

从 Task Execution 理论出发：如果模型几乎在随机猜测，那深层"consolidate task evidence"的论据是自我循环的——你不能用"train-shallow 更好"来说明"deep layers are execution zones"，因为两条命题都依赖于模型在做真正的推理，但绝对数字表明模型能力极弱。

**缺陷 2（严重）："Deep layers as execution zones" 与 "Train Shallow" 建议之间存在逻辑悬置**

作者的因果链是：(a) 深层 Ratio(l) 高 → 深层执行任务 → (b) 因此冻结深层以保留执行能力。但这个 (b) 的逻辑是"保护"（protection）而非"解耦"（decoupling）。

关键缺失：没有验证"被更新的浅层是否真的在 CPT 后 Ratio(l) 分布发生了变化"。如果浅层被训练后，深层的 TP/LS 分布也可能改变，原来的"深层执行、浅层敏感"的结论是否仍然成立，文章没有报告 CPT 后的 LayerTracer 再分析。论据是事后解释（post-hoc rationalization）：先跑 CPT，发现 train-shallow 赢，再用 LayerTracer 解释，但这两个步骤之间没有反馈验证。

**缺陷 3（中等）：Task Particle 是 target-token 相对概率变化的绝对值，单调性非常容易人为产生**

Ratio(l) = |Pt(l) - Pt(l-1)| / (Pt(l) + ε)

注意分子是绝对值，分母是当前层概率。这个公式有一个数学特性：当 Pt(l) 在深层趋向于高且相对稳定时，分母大，Ratio 可能下降；而当 Pt(l) 在中/深层从低快速上升时，Ratio 会出峰。这与 logit-lens 的"onset"现象相同，不是独立发现。与 ROME 系列的"mid-late MLP 存知识"、我们自己的 onset 分析（OLMo-2-7B L18→L19，0.326→0.544）本质上是同类观察。

更关键：Task Particle 是用 t* = argmax(P_final) 定义的，即**追踪模型最终预测的那个 token**。这在推理时 t* 已知，但如果模型在某层 Ratio 高是因为 t* 的竞争对手被压制（而非 t* 自身被提升），公式就会漏掉这部分信息。没有与 null 条件比较（e.g., random masking vs. no masking 的 ΔJS 差）。

**缺陷 4（严重）：所有 CPT 实验只做了 0.6B 参数规模，Limitations 承认不了解大模型**

"Due to limited resources, we validate it on mainstream model sizes instead of extending the analysis to much larger parameter scales." (§7 Limitations)

LayerTracer 诊断（§4）涵盖 0.6B/8B/14B，但**CPT 训练实验**只有 0.6B。没有 7B/8B 规模的 train-shallow/freeze-deep CPT 实验。这与我们的 Paper C 完全不同——我们是对 7B 做真实的 freeze-front + CPT 训练。

**缺陷 5（中等）：split boundary 没有搜索，固定 midpoint = 50%**

作者在 §4.2.3 明确说"midpoint split provides a symmetric parameter partition and a clean controlled setting, instead of exhaustively searching for the optimal boundary"，附录 C.2 验证了 33%/66% 边界下 S(b) 仍正值，但并未训练 33%/66% 边界模型来确认 downstream 效果。这意味着"train-shallow 好"的结论可能对 split 点有敏感性，而文章没有验证。

**缺陷 6（轻微）：评估集是 validation set（不是 test set），且用 zero-shot**

C-Eval 和 CMMLU 在 26% 附近的 zero-shot 分数极低，与这些 benchmark 的 few-shot 条件下数字差异极大。如果用 5-shot 或 chain-of-thought 协议，数字会高很多，效果差异是否还显著无法推断。生成评估用 temperature=0.95，而 logit 评估直接用 argmax，两协议下分数基本一致（std=0.00，C.3），说明结果不依赖于随机性但也说明数字极低（生成基本是 greedy）。

**缺陷 7（中等）：Limitations 自我承认的项目**
- "Due to limited resources, we validate it on mainstream model sizes" — 仅 0.6B CPT
- 没有提及中文 benchmark only 的 benchmark 偏差
- 没有提及 chance baseline 问题
- 没有提及 CPT 后 LayerTracer re-analysis 缺失

---

## 二、arXiv:2607.25663 — "Localized Adaptation Reveals Distinct Learning Signatures" (Adaptation Geometry)
**预印本，2026-07-28 v1，Yale University (Rebecca Ramnauth, Brian Scassellati)。**

### A. 实验设置精确描述

| 维度 | 实际值 |
|---|---|
| 主模型 | Llama-3.1-8B（**instruct-tuned**，§ Methodology：base parameters frozen，LoRA 插入） |
| 跨模型复现 | Mistral-7B, Gemma-2-9B, OLMo-2-7B, Qwen2.5-14B（注意：**都是 instruction-tuned 模型**，§ Cross-Model Robustness） |
| Adaptation 方式 | **LoRA only**，不是真剪层，不是全参，不是 CPT |
| Localized 区间定义 | early/middle/late = **quarter-depth windows**（§ Cross-Model Robustness："quarter-depth windows"）；主模型 Llama-3.1-8B 共 32 层，则 early=L1–8, middle=L13–20（估计），late=L25–32 |
| 参数量匹配 | **是（两种方法）**：(1) expanded-localized rank：把局部 LoRA rank 从 8 扩到 32×8/8=32，使 rank×层数 匹配（32×8=256=8×32）；(2) reduced-full rank：把全栈 rank 从 8 降到 2，使 2×32=64=8×8（Appendix C, Table 11）|
| 任务 | 5 个**合成 benchmark**（自建）：lexical binding / factual association / behavioral policy / causal mapping / procedural reasoning；每个 25 latent specs，每 spec 12 train + 22 eval examples |
| 训练预算 B | lexical binding B=10, behavioral policy B=10, causal mapping B=10; factual association B=8, procedural reasoning B=8（每条 latent spec 的训练例子数，§Calibration and Budget Selection） |
| 核心三指标 | Acquisition（A，ID + paraphrase accuracy 均值）、Transfer（R，generalization accuracy）、Boundedness（B，negative-control accuracy） |
| 统计方法 | Paired seed-level contrasts + bootstrap 95% CI（3 seeds），omnibus permutation test（10,000 permutations） |

---

### B. 方法论缺陷清单

**缺陷 7（严重）：5 个任务全是合成 benchmark，lexical binding 的 "negative control" 结构天然有利于 early layer**

Lexical binding：训练 "daxel means bird"，positive = "What is a daxel?" → bird；negative control = "Is a daxel a chair?" → no（Table 1）。

Early layer LoRA 的 boundedness 比 full-stack **高 28.3 pp**（68.0% vs 39.7%，Table 8）。但这个"早层比全栈更 bounded"的结论有一个结构性原因：全栈更新会让模型泛化地把 "daxel" 连接到各种 bird-related 推理，从而在 "Is a daxel a chair?" 这类 negative-control 上也可能错误地调用该 binding（over-generalization）。早层 LoRA 的 transfer 只有 40.6%（vs full 84.9%），说明早层基本没学会泛化，因此当然也不会在 negative control 上犯错。换言之，boundedness 高可能只是 learning failure，不是 "selectivity"。作者在脚注 3/4/5 提到了这个问题（null-label audits），但没有在主文中量化这个混淆。

**缺陷 8（严重）：Calibration budget 在 Llama 上校准后直接迁移到其他模型，cross-model 实验可靠性存疑**

"we transfer the Llama-calibrated objective budgets rather than recalibrating every model–objective pair" (§ Cross-Model Robustness)

Gemma-2-9B 在 causal mapping 上的 full-stack performance 在转移 budget（B=10）下只有 acquisition=51.2%，transfer=16.0%（Appendix E），明显 underfitting。作者将此解释为 model-budget mismatch 而非架构不兼容，这个解释合理但也说明 cross-model replications 的结论都要带条件。

**我们直接相关的一点**：OLMo-2-7B 包括在 5 个复现模型里（Table 13），且被标为 instruction-tuned。但我们的 Paper B/C 用的是 OLMo-2-7B BASE，不是 instruct。这意味着 2607.25663 的 OLMo-2-7B 实验与我们的 OLMo-2-7B 基础模型不是同一条件。

**缺陷 9（中等）：LoRA 的低秩约束使结论对"真剪层 / 嫁接新层"的外推性质不确定**

LoRA 保留原始权重并加 low-rank delta，这意味着：(1) 模型深度不变；(2) 原始 pre-trained 知识结构完整；(3) LoRA 学的是 delta，不是重新初始化。

我们 Paper C 的构造是：把前 j 层冻结，**丢弃**后面的层，嫁接 K 个随机初始化的新层。这与"只加 LoRA"完全不同——我们的 fresh layers 没有任何 pre-trained 初始化。2607.25663 的"late layers favor factual association"结论，在我们的框架里应该解读为"late layers in the intact model support factual adaptation"，但当这些 late layers 被替换成随机初始化的新层（我们的 fresh layers），这个 favorability 是否还成立是未测试的。

**缺陷 10（中等）：behavioral policy "boundedness" 高于 late-layer 22 pp 的结论，transfer 差异不显著**

Table 8 中 Middle >> Late on boundedness：21.6 pp，CI [15.3, 26.7]，p<.001（显著）。
但 Middle vs Late on transfer：2.9 pp，CI [-5.3, 8.0]，p=.516（**不显著**）。

作者仍然声称"middle adaptation best supports policy gating"（§Behavioral Policy Learning is Distributed），但 transfer 差异实际上零。用 "policy acquisition and gating appear partially separable" 描述一个 transfer-uncertain 的差异存在 over-interpretation 风险。

**缺陷 11（轻微）：模型全是 7B–14B instruct，没有 post-norm 架构**

"We selected these models because they are open-weight, instruction-tuned transformers at a scale where multi-objective localized LoRA adaptation remains computationally feasible" (§ Cross-Model Robustness)

全部 5 个模型（Llama, Mistral, Gemma, OLMo-2, Qwen2.5）都是 pre-norm（RMSNorm before MLP/Attn）。OLMo-2 确实用了 post-norm（output_layer_norm + attention_layer_norm 在 block 外），但作者没讨论 norm 位置差异如何影响 localization。我们实测 OLMo-2 是 14 个模型里 CKA alignment 最难对齐的模型，这可能与 norm 位置有关，值得追问。

**缺陷 12（轻微）：Limitations 自我承认的项目**
- "benchmark is synthetic and isolates five operational learning objectives"
- "early, middle, and late windows provide only a coarse view of model depth"
- "study considers only LoRA"
- "cross-model experiments transfer budgets calibrated on Llama rather than independently optimizing them"

---

## 三、arXiv:2510.18871 — "How Do LLMs Use Their Depth?" (Guess-then-Refine)
**预印本，2025-10-21 v1；2026-03-01 v2，UC Berkeley / MIT（Anna Ivanova）。标注为 ICML（Machine Learning, ICML）。**

### A. 实验设置精确描述

| 维度 | 实际值 |
|---|---|
| 模型 | GPT2-XL（1.5B）、Pythia-6.9B、Llama2-7B、Llama3-8B（4 个）；全是 **base 模型**（未 instruction-tuned） |
| Probe 方式 | **TunedLens**（Belrose et al. 2023）= per-layer affine probe，minimize KL(final_dist || tuned_lens(h^l))；**不是** raw LogitLens |
| TunedLens 来源 | 直接使用 AlignmentResearch 预训练的 probe（Appendix D，2026-01-28 访问），自验证（D.2）通过改训 "the" 频率 1000x 下调后 probe 仍给出高频 token → 说明不是 probe 人为 artifact |
| Case Study I 数据集 | MMLU（4 选 1 ABCD）、SST（2 选 positive/negative）、NLI、MRPC；4-shot 格式 |
| Case Study II 数据集 | MQuAKE（多跳知识编辑数据集，Zhong et al. 2023），split 为 1/2/3 token 答案；**只分析模型答对的情况**（see Appendix H 讨论） |
| Case Study III 数据集 | English Wikipedia 100k prefixes；用 spaCy POS tag 预测 token |
| Activation Patching | SST forward pass 的 l-层激活 patch 进 MMLU forward pass 的同位置 → 看输出是 MMLU option 还是 SST option |
| Early-Exiting 实验 | 用 TunedLens 在中间层解码，计算 Top-1 match rate |
| 核心数字 | Pythia-6.9B 中，layer 1 的 Top-1 预测有 >75% 属于 Top10 最频 token；到最终层只剩 33%；约 80% 早层预测被最终层推翻；多 token fact 首 token 在约 layer 27 才稳定，后续 token 在 layer 20/12 稳定 |

---

### B. 方法论缺陷清单

**缺陷 13（中等）：TunedLens 使用的是 pre-trained probe，不是 per-task 校准，中间层激活解读有一个微妙假设**

TunedLens 的 (A_l, b_l) 是**一次性**针对整个模型训练的，目标是最小化 KL(P_final || TunedLens(h^l))。这个 probe 仅优化使中间层看起来像最终输出，但不区分"模型在 l 层真的已经形成了该预测"和"l 层的表示空间是 tuned lens 在拟合 P_final 时的巧合投影"。

作者在 §5 做了 frequency bias 检验（D.1/D.2）并验证 bias 不来自 probe，这是好的。但 tuned lens 本质上是**最优线性逼近**，不是因果证明。"early layers promote high-frequency tokens" 可能只是说"在 early layer 的线性子空间里，频率信息最容易被读出"，而不是"模型 mechanistically 在 early layer 做统计猜测"。作者承认这一点只在 Related Work（§6）中隐含提及（"saturation events" 相关工作），没有显式作为 limitation 列出。

**缺陷 14（中等）：MC 任务的"first half 收集 options，second half 推理"这个结论，option-collection 的 bias 未充分控制**

§4.1 中，作者追踪 MMLU 和 SST 的答案 token 的 rank，发现前半层 options 集中到 top-k。但这个观察有一个可能的 confound：MMLU 的 option letters（A/B/C/D）是高频 tokens，自然会在早层（统计猜测阶段）出现在 top ranks。换言之，"early layers collect valid options"可能只是 "early layers are biased toward high-frequency tokens 且 A/B/C/D 恰好是高频 tokens"，不需要对任务有特殊理解。

作者在激活 patching 实验（Figure 4c）中提供了因果支持：SST 激活 patch 进 MMLU forward pass，早层 patch 后模型仍输出 MMLU options，晚层 patch 后输出 SST options。这是一个相对干净的因果检验。但这个实验的对照条件是 SST（2 选）patch 进 MMLU（4 选），逻辑上不完全对称：SST 的 positive/negative 也是高频词。Llama3-8B 的 patching 实验在 Appendix G 中说"transition layer prediction by TunedLens does not match actual behavior for Llama3-8B"（Figure 19c），说明框架的预测能力并非普适。

**缺陷 15（中等）：MC "content-knowledge 与 letter-selection 没有拆开"**

我们在 Paper A/B 的分析中区分过两件事：(a) 模型在哪一层"知道"正确答案的内容，和 (b) 模型在哪一层把该内容映射到特定 letter（A/B/C/D）。2510.18871 只追踪 final prediction 的 rank 演化，无法区分这两件事——如果 final prediction 是 "A"（letter），那它追踪的是 letter 的 rank；但如果知识已经在 layer 15 形成而 letter-mapping 在 layer 25 才绑定，这个拆分文章没有做。我们自己的 knowledge onset 测量（OLMo-2-7B L18→L19 onset；知识 token 的 logit-lens 概率跳变）是针对内容 token 的，与这篇追踪的 letter token 是正交的。

**缺陷 16（中等）：模型池只有 4 个，全是英文、base、< 8B**

GPT2-XL、Pythia-6.9B、Llama2-7B、Llama3-8B。都是 casual decoder，英文，pre-norm（Llama 系列是 RMSNorm pre-norm，Pythia 是 LayerNorm pre-norm，GPT2 是 LayerNorm pre-norm）。没有 post-norm 变体，没有 Qwen/OLMo-2 类型，没有 >= 13B 的模型。"Guess-then-Refine" 是否在 post-norm 架构（如 OLMo-2 使用的 output_layer_norm + attention_layer_norm）或 Qwen3 这种有 QKV RoPE 变体的架构上成立，未验证。

**缺陷 17（轻微）：Case Study II 只计算模型答对的例子（Appendix H），可能有 selection bias**

"We only use the prompts where the model generated the correct answer." (§4.2)

作者解释这样做便于精确划定答案 span，在 Appendix H 中对 incorrect 情况也做了补充分析，发现多 token fact 首 token 需要更多层的规律在 incorrect 时稍弱但方向一致。这个 selective-correct 分析是合理的，但结论的强度依赖于在正确回答时 first token 需要更多层这个条件，对 overall accuracy 较低的任务（GPT2-XL 的 fact recall 数量就少得多）可能有显著的 selection bias。

**缺陷 18（轻微）：Limitations 自我承认的项目**
原文无显式 Limitations 段（Conclusion 仅说"Some avenues of future work includes a similar analysis for reasoning and chain-of-thought tasks and more recent reasoning models"）。隐含问题：仅 base model；仅英文；TunedLens 是 probe-based 而非 causal ground truth。

---

## C. ★ 我们能接上什么 / 能证伪什么

### C1. 2605.11416（LayerTracer）vs 我们 Paper C：正面对撞，四层剖析

**对撞现象**：2605.11416 报告"训浅冻深 > 全参 > 训深冻浅"（在 0.6B Qwen3 续预训练 + C-Eval/CMMLU）。我们 Paper C 的 freeze-front（冻浅层/只训顶部 fresh）在 SQuAD-v2 refusal-25 eval 上四个深度全部低于 constant baseline（25.00%）：keep14=22.50, keep20=20.45, keep24=22.65, keep28=22.35。

这两个结果**表面矛盾**，但仔细分析有四个不同维度：

**维度一：Benchmark 标签先验问题**

2605.11416 的 C-Eval/CMMLU 绝对分在 25–30% 之间（chance = 25%），其"成功"方向和"失败"方向的差异只有 4 pp。如果 C-Eval 某些题目有标签先验（比如某选项 A 在某类题中频率明显高于 25%），那"训浅冻深"的 29.40 可能只是更好地捕捉了答案分布的先验，与"深层 task execution 被保护"无关。

**我们可以用现有资产验证**：用我们的 `probe_linguistic_layerwise.py` + 知识 logit-lens，在 C-Eval 风格的中文 MC 题上做 OLMo-2-7B（或 Qwen3-8B）的 layer-wise 分析，看看 freeze-deep 后各层 onset 是否真的被保护。预测：如果 C-Eval 结果只比 chance 高 4 pp，那"被保护"的东西很可能是极浅的统计先验，不是 task execution。

**维度二：Benchmark 类型差异——MC vs 开放问答**

2605.11416 全程用 4 选 1 MC（C-Eval/CMMLU）。我们用 SQuAD 开放式 EM。MC 任务（如 2510.18871 也发现的）在前半层就能 collect options，letter prediction 对深层知识的依赖度远低于开放式问答。因此"冻结深层"在 MC 任务上代价更低。我们 Paper C 是真正测试深层知识保留能力的 open-domain QA，更能暴露 freeze-deep 的代价。

**这是一个可发表的对照点**：同一模型（OLMo-2 7B），同一 freeze-front 构造，跑 MC（用我们已有的 arc/hellaswag/mmlu 评测脚本 `eval_olmo2_probe2_downstream.py`）vs 开放 QA（已有 SQuAD / closedbook QA 脚本），直接比较 freeze-front 在两类任务上的代价差异。这比 2605.11416 只跑 MC 更完整，也更诚实。

**维度三：是否真实丢弃层——最关键的架构差异**

2605.11416 只是**冻结**深层（权重在，更新梯度为零，forward pass 仍然经过深层）。

我们 Paper C 是真实**丢弃**顶部 L-j 层，嫁接全随机初始化的 K 个新层。这是根本不同的操作：
- 2605.11416 的"freeze-deep"：模型深度 = 全部 28 层，所有 pre-trained 权重都在 forward path 上。
- 我们的"freeze-front + prune top"：模型深度 = j+K（显著缩短），深部 pre-trained 权重根本不参与 inference。

因此 2605.11416 不是真正测试"深层知识能否被取代"，而是测试"深层不被更新时是否保留得更好"。这两个问题完全不同。我们的失败（freeze-front 在 SQuAD 低于常量基线）反映的是"顶部 pre-trained 知识完全丢失后，K=2 个随机初始化层无法重建知识存取能力"，而不是在说"训浅层"是坏主意。

**这个对比可以直接写进 Paper B 的相关工作或 Paper C 的 limitation 讨论**。

**维度四：CPT（续预训练，大规模无标签语料）vs SFT（小规模监督 finetune）**

2605.11416 训练了 262B tokens（1 epoch over CCI3.0-HQ），是大规模 CPT，目标是语言模型的域适应。

我们 Paper C 训练了 SQuAD 1000 steps（约 166 epochs over 5000 SQuAD 训练例子），是严重 over-fit 的监督 finetune。

这个差异非常关键：大规模 CPT 的信号足以让浅层"学到"足够多的中文知识，从而在 C-Eval 上有数个 pp 提升；而 SQuAD 监督 finetune 的主要任务是"如何从给定段落里提取答案"，需要的是文章理解和 span-selection 能力，这种能力在 pre-trained 模型里高度依赖于深层的 factual association 和 contextual integration，fresh random 浅层无法快速学会。

**这个对撞本身的可发表性评估（我方资产维度）**：

我们已有的资产足以做以下实验来明确区分三个假说：
1. 用现有的 `train_olmo2_arch_probe2.py`，做"冻结前 j 层但不丢弃后面层"的变体（只训后面层），对比"丢弃后面层+嫁接"——这直接对齐 2605.11416 的构造。如果"冻深不丢"好而"丢深"差，说明问题在深层丢弃，不在 freeze 方向。
2. 把 C-Eval 风格的 MC eval 加进我们的 downstream eval（已有 arc/hellaswag/mmlu，可代替 C-Eval），对比 MC 任务 vs 开放问答任务下 freeze-front 的效果差异。
3. 用大规模 CPT 数据（PG19/Dolmino/SlimPajama）在我们的 freeze-front 架构上续训（而非只训 SQuAD），看是否在 CPT 条件下也能复现 2605.11416 的正向结论。

**结论性判断**：2605.11416 的"训浅冻深"建议与我们的 Paper C 失败结果**并不实质矛盾**，因为它们操作的是完全不同的构造（冻结 vs 丢弃）、不同的任务类型（MC vs 开放 QA）、不同的训练规模（262B CPT vs 2M SFT）。但这个三维对比恰恰是一个有价值的学术贡献，它能精确定位"freeze-front 最优解依赖的三个条件"。

### C2. 2607.25663（Adaptation Geometry）与我们的资产

**我们的 probe 设施与"lexical binding is early-localizable"的交叉**：2607.25663 的 lexical binding 任务（新词-概念绑定）与我们的 POS/DEPREL edge-probe 在内容上相邻，但不完全相同。他们的核心发现是 early LoRA 在 acquisition 上几乎匹配全栈，但 transfer 差 44 pp。

我们可以用 `probe_linguistic_layerwise.py` 的 POS probe（已实现），在 OLMo-2-7B 的 keep14/20/24/28 变体上直接测各层的 POS probe accuracy，检验"lexical binding stays in early layers"是否在我们的 prune-heal 架构（keep N 层后全参 CPT）下也成立。如果 POS probe 在 keep14-200k 上仍高于 keep28-200k，说明浅层 lexical 功能在 prune-heal 后被保留；如果相反则说明 CPT 改变了功能分布。这是一个廉价的正交验证，直接用现有资产。

**我们的"factual association favors late layers"与 OLMo-2-7B knowledge onset L18 的关系**：我们实测 OLMo-2-7B knowledge onset = L18（共 32 层，~0.563L），与 2607.25663 的"factual association favors late layers (localized)"在方向上一致（late ≈ L25–32 约 0.78–1.0L 比 middle/early 都强）。这不是对撞，而是支持。

我们可以用 Paper B 的 keep8/10/12/14/16/full32 全套 ckpt（均有 200k step），测各 truncation depth 下 factual association probe 的 accuracy（用现有的 knowledge_logit_lens）。预测：keep14 截断了 L15–31，丢弃了大量 knowledge-relevant 层，因此 factual probe 最低；keep28 保留到 L28，接近 onset zone，probe 较好。这种"prune depth vs factual probe performance"曲线，结合 2607.25663 的 late-layer favorability，能给出 prune boundary 的功能可解释性依据。

### C3. 2510.18871（Guess-then-Refine）与我们的资产

**"MC 任务前半收集 options，后半定稿"与我们 Paper A 中 QCMem 的 logit-lens 分析**：我们在 Paper A 分析过 memory-augmented model 的 layer-wise behavior。2510.18871 的 option-collection（前半）/ option-reasoning（后半）框架对我们的 BABILong eval 有参考价值——BABILong 是 QA 任务，在多层处理中 memory-augmented model 的 layer transition 可能与 baseline 不同。但这个联系是 loose 的，不是直接可接的方向。

**"fact recall first token requires more depth"与我们的 knowledge onset L18 一致**：我们的 OLMo-2-7B onset L18（0.326→0.544）正好对应 2510.18871 中 Llama3-8B fact recall single-token onset 约在 layer 20/27。这两个测量用的是不同 probe（他们用 TunedLens 追踪 rank-crossing；我们用 logit-lens 直接看 knowledge token 的 probability jump），但结论方向一致。

**可验证的交叉实验**：用我们已有的 logit-lens + knowledge_logit_lens 设施，在 OLMo-2-7B base 上做与 2510.18871 §4.2 类似的多 token fact recall depth 分析，看"首 token vs 后续 token"的 onset depth 是否有和他们相同的梯度分布。如果我们的测量方法（logit-lens）和他们的（TunedLens）给出一致结论，那两种 probe 可以互相校准，也强化了 onset depth 测量的可靠性。

### C4. 缺陷 1 的直接证伪实验

**对 2605.11416 缺陷 1（绝对分在 chance 附近，无常量基线）**：
- 我们现有的 `eval_olmo2_probe2_downstream.py` 包含 arc/hellaswag/mmlu，这些也是 4 选 1 或 2 选 1 MC。
- 在我们的 keep14/20/24/28/full32 各 ckpt 上，计算"常量猜测基线"（各答案选项的 prior 分布），验证模型分数是否真的高于非常量基线。
- 如果我们的结果中 freeze-front 模型的 MC 分数接近 chance（我们 Paper C §132 中 A4_MMLU=0.2596，仅比 chance=0.25 高 z=2.6），这个数字可以直接引用，说明"MC 评测非常脆弱，4 pp 的差异在 chance boundary 附近不可信"。

---

## D. 一句话判决

**arXiv:2605.11416 (LayerTracer)**：它占掉了「诊断浅层敏感 / 深层执行」的叙事框架，但其 CPT 实验只在 0.6B 模型做，benchmark 数字在 chance boundary 附近，且"冻结"不等于"丢弃"；我们能用 7B prune-heal 的 MC-vs-QA 对比 + onset probe 接上它的漏洞，把"什么条件下 freeze-front 有效"讲清楚。

**arXiv:2607.25663 (Localized Adaptation)**：它占掉了"LoRA 局部适应 / 5 类学习目标分区"的分析框架，包括 OLMo-2-7B 在内；但它全用 instruct-tuned 模型 + synthetic benchmark，不做真剪层，不做 CPT；我们用 OLMo-2 BASE + 真剪层 + Paper B 的 prune-heal ckpt 能给出 production-scale、architecture-genuine 的反例和补充。

**arXiv:2510.18871 (Guess-then-Refine)**：它建立了"早层统计猜测 / 晚层 contextual refinement"的量化框架，和我们的 knowledge onset 分析是友军而非对手；我们可以引用它的 TunedLens 框架来互相校验 onset depth，并把 multi-token fact 的"first token needs more depth"结论用到 prune boundary 选择的解释上。

---

### ★ 总判决：层分配这片地被占了几成？剩下的窄缝是什么？

**被占掉的部分**：
1. "诊断层功能差异 → 指导 freeze/train 分配"的 general argument（2605.11416 + 2607.25663 合力覆盖）
2. "MC 任务中早层 collect options，晚层 reason"（2510.18871）
3. LoRA 局部适应下各类目标的 depth preference（2607.25663）

**真正没人做的缝隙（我方资产覆盖）**：

1. **"丢弃深层"vs"冻结深层"对功能保留的差异**：所有三篇都不曾真正丢弃并替换层；我们是唯一做过 prune + random-graft 实验的（Paper B/C）。

2. **CPT → capability recovery 的 prune depth boundary**：2605.11416 是 CPT 但不丢层；我们有 keep8/10/12/14/16/full32 各 200k step ckpt，可以精确测"续预训练后功能恢复的 depth threshold"。

3. **Freeze-front 失败是否依赖任务类型（MC vs open QA vs SFT vs CPT）**：我们是唯一有 SFT（Paper C）+ CPT（Paper B）+ MC（downstream eval）+ 开放 QA（SQuAD / closedbook）全套实验的，可以做完整的 2×2 矩阵（冻结方向 × 任务类型）。

4. **OLMo-2 BASE（post-norm + 无 SFT）下的 adaptation geometry**：2607.25663 的 OLMo-2 实验用的是 instruct-tuned 版本，用的是 LoRA；我们的是 OLMo-2 BASE + full-param CPT，这是不同的（且是用于证明 base LM 能力的更干净）设置。

5. **我们的 Paper C 失败结果反过来可以成为资产**：Paper C 的 freeze-front 失败（SQuAD 低于常量基线）和 Paper B 的 prune-heal 成功（全参 CPT 后恢复能力），合在一起说明了**"只训 cap（freeze trunk）失败，全参 heal 成功"**，这正好回答了 2605.11416 没有验证的问题：在真实丢弃深层的条件下，"只训浅层"是不够的。这是一个方法论上更诚实、更完整的贡献。

---
*字数统计：本文档约 5900 字，全文读取了三篇论文 HTML 全文（2605.11416=264KB，2607.25663=460KB，2510.18871=234KB），每条引用均标注了 section。*
