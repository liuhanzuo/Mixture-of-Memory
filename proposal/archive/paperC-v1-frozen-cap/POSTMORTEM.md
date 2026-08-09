# Paper C 方向决策备忘录

日期：2026-08-05 · 依据：6 路 codex 复跑（`paperC_research/rerun_*.md`）+ 3 路打捞（`r{1,2,3}_salvaged.md`）+ **本备忘录作者当场新做的 per-example 分层复算**（下文 §0，是本文最重要的新证据）

---

## 0. 先说本轮新增的实测证据（推翻了两条既有认知）

这些数字是写这份备忘录时现场从 per-example 预测文件重算出来的，此前任何报告里都没有。

**(0a) per-example 预测文件其实存在。** r2 打捞报告断言"只有 summary，McNemar/bootstrap 无法复算"——**错**。文件在 .73（zwfy6 盘）：
`/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/paperC_squad_results/<arm>/per_example_shard0of1.jsonl`（7 臂全有，已拷到本机 `/tmp/pcpe/`）。
复算结果与已发布数字**逐位吻合**：A4 vs A3 McNemar b/c=212/147, χ²_cc=11.409, p=7.31e-4；paired bootstrap 95% CI=[+1.40pp, +5.15pp]。统计本身没有算错。

**(0b) 但把它按标签分层，headline 就废了。** val 里 49.85% 的 gold 是同一句中文拒答串。分层后：

| 分层 | n | A4_hero | A3_scratch | Δ | McNemar |
|---|---|---|---|---|---|
| **拒答层**（gold=拒答串） | 997 | 0.5827 | 0.5216 | **+6.12pp** | b/c=207/146, χ²=10.20, **p=1.4e-3** |
| **可答层**（gold=真答案） | 1003 | 0.0050 | 0.0010 | **+0.40pp** | b/c=**5/1**, χ²=1.50, **p=0.22 → NULL** |

**+3.25pp 的 93.8% 来自拒答层。** 可答层上只有 6 个 discordant pair（5 vs 1），完全是噪声。
→ **"A4 打赢同深度 from-scratch"这条 headline，在唯一有意义的那一半数据上不存在。**

**(0c) 可答层 EM 阶梯 = 唯一诚实的能力轴，而且比 raw EM 干净得多：**

| arm | 深度 | raw EM | **可答层 EM** | 占 BASE 的比例 | 捷径贡献占 raw EM |
|---|---|---|---|---|---|
| BASE_ref（无 SFT） | 32 | 0.3385 | **0.6750** | 100% | 0.0% |
| A2_lora_r160 | 32 | 0.6590 | **0.6770** | **100.3%** | 48.5% |
| A4_keep28 | 30 | 0.4190 | 0.2981 | 44.2% | 64.3% |
| A4_keep24 | 26 | 0.3560 | 0.1745 | 25.9% | 75.4% |
| A4_keep20 | 22 | 0.3440 | 0.0847 | 12.6% | 87.6% |
| A4_hero(keep14) | 16 | 0.2930 | 0.0050 | **0.7%** | 99.1% |
| A3_fromscratch | 16 | 0.2605 | 0.0010 | 0.1% | 99.8% |

两个副产品，都很重要：
- **A2(LoRA) 在可答层上与 BASE 统计上完全相同**（0.6770 vs 0.6750，McNemar p=0.94）。它相对 BASE 的 +32pp raw EM **100% 来自学会拒答**（拒答层 0→0.641，可答层 +0.20pp）。所以"LoRA 有能力税"在这个任务上**不成立**——税只出现在域外能力（#132 的 MMLU/TriviaQA），域内可答项零损失。这个对比本身反而是个干净的 dissociation。
- **捷径贡献随保留深度单调下降**（99.8→99.1→87.6→75.4→64.3→48.5→0.0%）。这是"压缩越狠越依赖捷径"的直接测量，已经在手。

**(0d) 这份 SQuAD SFT 数据的"不可答"标签是坏的（数据集层面的硬伤）。** 我逐项查了：
- 无 SFT 的 BASE 对拒答标注项，**74.2% 输出的是 context 里的逐字 span**，中位 1 个词——即模型认为这些题**可答**，而且答案就在给定 context 里。人工看前 8 条全部可答（例："What is France a region of?" → BASE 答 "Normandy"，context 里有）。
- 非拒答项里，`relevant_indices` 指向的那个 chunk **只有 31.6% 真含 gold**。
→ 所谓"不可答"是 LoCoMo 式 chunk 重打包时的**检索失配 artifact**，不是真的不可答。train/val 拒答率 17.56% vs 49.85%（2.8×）也是这个 artifact 的副作用。
**结论：`data/squad_{train,val}.jsonl` 不能作为任何论文 headline 的评测集，必须重建。**

**(0e) 一个真实存在的 A4-vs-A3 信号（唯一的）：** 拒答判别的 Youden J（=TPR_拒答 − FPR_拒答）：

| arm | J | TPR | FPR |
|---|---|---|---|
| A3_fromscratch | **−0.001** | 0.522 | 0.522 |
| A4_hero | **+0.202** | 0.583 | 0.381 |
| A4_keep20 / 24 / 28 | +0.370 / +0.361 / +0.319 | | |
| A2_lora_r160 | +0.585 | 0.641 | 0.056 |
| BASE_ref | 0.000 | 0.000 | 0.000 |

A3 的 J **恰好为 0**：它只学到了拒答的**先验频率**，零判别力。A4 的 J=+0.202 是真的、非退化的。
→ 继承+冻结 trunk 买到的东西可以被诚实地表述为：**一点点"可答性判别力"，而不是问答能力**。这是可以写的，但它很小，而且是一个二分类决策，不是 QA。

**(0f) A3 深度扫（#133）三个臂已训完/近训完，但还没 eval。** 日志：keep20/24 `TRAIN_DONE`，keep28 在 step 910/1000（本机，约 5 分钟后结束）。三个臂的 loss 全部收到 **0.0005-0.0006, ppl=1.00, gnorm=0.00** —— 166 epoch 纯记忆，capacity-bound 确认。**这三个 eval 是当前最便宜、最关键的待跑项**（见 §6 E0）。

---

## 1. 一句话判断

**需重构。当前框定（P-C1 构造 + P-C2 探针预测）不能投顶会，而且不是"再补几个实验"能救的——两个命题各自都被先例占住，headline 数字在分层后消失，评测集本身是坏的。** 6 路独立复跑 + 我的分层复算收敛到同一结论。可写的东西存在，但它是另一篇论文（见 §5）。

---

## 2. 被抢先了吗

### 2.1 可验证的先例（arXiv ID 已过 arXiv API 独立核验，见 `rerun_CITATION_VERIFICATION.md`）

| # | 标题 | ID | 与我们重叠在哪 |
|---|---|---|---|
| P1 | Streamlining Redundant Layers to Compress Large Language Models（LLM-Streamline） | **2403.19135** | **最近的一篇。** Llama-3.1-8B 剪掉连续 8 层 → 插入 1-2 个替换 transformer 层 → 净更浅（8B→5.4B）；代码 `train_llmloss.py` 里 `requires_grad=True` **只给替换层**，其余全 False = trunk 真冻结。7-8B decoder。venue = ICLR 2025 Spotlight（唯一被核实的 venue，来源为其 repo README） |
| P2 | Reassessing Layer Pruning in LLMs: New Insights and Methods | **2411.15558** | 摘要逐字："pruning the final 25% of layers followed by fine-tuning the `lm_head` and the remaining last three layer"。**丢顶部 + 只训顶部一小撮**，Llama-3.1-8B-Instruct，且显式 benchmark LoRA 系列做剪后恢复 |
| P3 | LLaMA Pro: Progressive LLaMA with Block Expansion | **2401.02415** | **冻结原 trunk + 新增可训 block** 这一架构范式的最近类比。但是净**更深**（7B→8.3B）且新 block 是 identity-init |
| P4 | Llama SLayer 8B: Shallow Layers Hold the Key to Knowledge Injection | **2410.02330** | 浅层扩展 + 深层剪枝 + 领域后训练 |
| P5 | Revisiting Few-sample BERT Fine-tuning（Zhang et al.） | **2006.05987** | FT 前 re-init 顶部 K 层。**但 grep 其 `run_glue.py` 只有 pooler 一处 requires_grad → 不冻结 trunk** |
| P6 | Surgical Fine-Tuning（Lee et al., ICLR 2023） | — | 只调**已有** block 子集，post-hoc gradient/Fisher 选 |
| P7 | ShortGPT: Layers in LLMs are More Redundant Than You Expect | **2403.03853** | Block Influence = 前向 hidden-state 相似度，**base 模型前向信号选剪哪层** |
| P8 | The Unreasonable Ineffectiveness of the Deeper Layers | **2403.17887** | 跨层表征相似度选**最优连续可剪块**，然后小量 QLoRA healing |
| P9 | RSRA: Training-Free Probing of Representation Sensitivity for Efficient LoRA Rank Allocation | **2607.09757** | **训练前、纯前向**表征敏感度探针 → 分配 LoRA rank |
| P10 | Dominant-Layer ZO: A Single Layer Dominates Zeroth-Order Fine-Tuning of LLMs | **2606.05516** | 摘要明说 dominant layer 可在**训练前**由 **inference-only** 激活离群分析确定 |
| P11 | Understanding and Guiding Layer Placement in PEFT（"Layer Card"） | **2602.04019** | Qwen3-8B 上做逐层 PEFT 放置选择（但其核心量涉及归一化梯度范数，非纯前向） |
| P12 | The False Promise of Imitating Proprietary LLMs | **2305.15717** | 逐字："imitation models are adept at mimicking ChatGPT's style but not its factuality" = **格式/能力解离的所有权** |
| P13 | LIMA: Less Is More for Alignment | **2305.11206** | superficial alignment hypothesis |
| P14 | Scaling Laws for Forgetting When Fine-Tuning LLMs | **2401.05605** | LoRA 也遭灾难性遗忘，目标任务性能 ↔ 遗忘成反比 |
| P15 | LoRA Learns Less and Forgets Less | **2405.09673** | 同上族 |
| P16 | Predicting Fine-Tuning Performance with Probing | **2210.07352** | 用探针预测 FT 最终性能（但不选层、不定深度） |

### 2.2 我们三个自称区分点，逐条裁决

**(1)「丢弃顶部 + 重长 K < L−j 使网络真变浅（不只是 re-init）」→ ❌ 不成立。**
2403.19135 已经做了：删层 + 插 1-2 个替换 transformer 层 + 净更浅 + 7-8B decoder，四项全中。2411.15558 也已做「丢顶部 25% + 只训顶部」。这条唯一残余是**"整段后缀截断 + 终端 fresh cap"vs LLM-Streamline 的"内部块替换"**——位置差异，不是原理差异。

**(2)「trunk 真冻结（Surgical FT 仍更新所选 block）」→ ❌ 对 LLM-Streamline 不成立，✅ 只对 Zhang et al. 成立。**
LLM-Streamline 的代码已被读过：`requires_grad=True` 只给替换层。所以"真冻结"不是我们的。**只有对 P5（Zhang'21）这一条能站住**——但没人会把"我们比一篇 2021 年的 BERT 论文多冻了一层"当作贡献。
**而且我们自己的实现也不干净**：`apply_freeze_front` 只冻 `model.layers.{<j}`；`embed_tokens` / `model.norm` / `lm_head` 是**继承来的且可训练**（LR 2e-5 / 2e-5 / 1e-4）。所以"train only the fresh cap"这个 slogan 在我们的代码里是**假的**。

**(3)「decoder LLM @7B，不是 BERT」→ ❌ 不成立。**
2403.19135（Llama-3.1-8B）、2411.15558（Llama-3.1-8B-Instruct）、2401.02415（LLaMA2-7B→8.3B）、2410.02330（8B）全在 7-8B decoder。

**残余楔子（全部列出，按 reviewer 在意程度排序）：**
1. 后缀截断 + 终端 fresh cap（vs 内部块替换）——位置差异
2. **随机初始化** cap（LLM-Streamline 显式偏好权重继承并称其更好；LLaMA Pro 用 identity-init）→ 我们选的是**已被文献判定为较差**的那一侧
3. 恢复目标是下游/instruction SFT，而非预训练式 recovery loss —— setting 差异
4. OLMo-2 家族的可移植性演示
**四条加起来不足以支撑一个 standalone contribution。构造不能当主贡献。**

### 2.3 P-C2 的 hook 也被占住

前向-only、base 模型、训练前 → 选深度切点：**2403.03853 + 2403.17887 已占**。
前向-only、训练前 → 分配 adaptation 容量：**2607.09757 + 2606.05516 已占**。
残余缺口只有「**任务条件化地联合预测 (j, K)**」，是**段落级**而非论文级缺口。

### 2.4 UNVERIFIED 清单（不得进 .bib）

- **除 LLM-Streamline(ICLR 2025 Spotlight) 外，所有 venue 字符串均未核实**：ShortGPT/Findings ACL 2025、2305.15717/ICLR 2024、LIMA/NeurIPS 2023、2210.07352/EMNLP 2022、LLaMA Pro/ACL 2024 全是模型记忆，入 bib 前必须重查。
- **2607.09757 的真名是 RSRA，不是 "RSLoRA"。** rerun 报告把它"纠正"成了错的名字，并连带编造了「N=64 校准集 / 结构化低秩噪声 / cs.CV 分类」等细节——这些**全部 UNVERIFIED**，不得引用。实质论断（training-free 前向探针分配 rank）由标题本身确认，成立。
- **2411.15558 是否显式冻结 trunk（embed/final-norm 可训性）仍未 100% 解决**（约 80%）。已取到正文句 "we fine-tune the model using partial-layer fine-tuning (lm head + last three layers) after each pruning step"，但 embed/norm 可训性未验。
- Llama SLayer 是否净更浅 = UNVERIFIED；LLaMA Pro identity-init 细节来自 abstract 级检索，未读代码。
- r3 提到的一篇 "Linear Probes Detect Task Format, Not Reasoning Mode" **从未被验证，可能不存在**，不得引用。

---

## 3. 现有证据的致命弱点

问题是"哪一个 confound 足以单独解释 +3.25pp"。答案：**有四个各自都够，而且第五个（分层）直接说明那 +3.25pp 根本不在能力轴上。**

| confound | 单独足以解释 +3.25pp？ | 依据 |
|---|---|---|
| **① 拒答先验主导评测**（本轮实测） | **是——而且更强：它解释 93.8%** | 分层后可答层只剩 +0.40pp，6 个 discordant pair，p=0.22。恒定拒答器 raw EM=0.4985，**高于所有剪枝臂** |
| **② A4 继承了可训练的 embed/norm/lm_head，A3 全随机** | **是** | 1.58M token 学不出词表 embedding + 输出映射。这一项就能远超 3.25pp，且意味着实验**根本没隔离出"trunk 继承+冻结"** |
| **③ A3 只有 1.58M token 训一个 4.06B 随机模型**（0.00039 token/param） | **是** | A3 不是可信的 from-scratch 基线，它的失败在检验任何假设之前就注定了 |
| **④ 166 epoch 过训 / capacity-bound** | **是** | 实测 loss=0.0006, ppl=1.00, gnorm=0.00。谁更好地记住高频整串就赢 |
| ⑤ LR 不同（A4 1e-4/2e-5 vs A3 3e-4，无 per-arm 搜索） | 不能排除 | 单 seed、小数据、高 epoch 下几个 EM 点很正常 |
| ⑥ optimizer 精度不同（A4 fp32 AdamW；A3 keep14 OOM 后改 bnb 8-bit） | 不能排除 | 但通常不足以造成 3.25pp |
| ⑦ 训练参数量 1.23B vs 4.06B | 不能排除 | 与 ②③ 高度耦合，无法分离 |
| ⑧ 单 seed（全臂 `--seed 42`） | 不能排除 | 显著性只覆盖 eval item 变异，不含训练变异 |
| ⑨ 按 passage 聚类未处理 | 否（只影响精度，不影响点估计） | 会高估显著性 |

**最大单一威胁不是 brief 强调的那个。** 它是 **②**：A4 继承了**可训练的**预训练 embed/norm/lm_head，而 A3 必须从随机初始化用 1.58M token 学会它们。任何 2×2 设计必须**四格共用同一套预训练 embed/norm/lm_head**，否则"trunk 继承"永远无法与"拥有一套能用的词表输入/输出接口"分离。

**"两个近-chance 模型比较是否有意义"——答案：在能力轴上没有意义；只在一个非常窄的问题上有意义。**
- #132：A4 MMLU 0.2596（比 0.25 下限高 z=+2.6），A3 0.2474（z=−0.7，就在 chance）；14 格 9 格 null；boolq A4 反输 1.56pp。汇总 +0.39pp 虽 p=1.2e-3 但把"统计显著"和"科学重要"混为一谈——+0.39pp 在可用 above-chance 信号区间里只占 ~2.7%。
- 本轮更硬：SQuAD 可答层 A4=0.0050、A3=0.0010，**两者都是 BASE 的 0.1-0.7%**。这不是"两个坏模型"，这是**两个不做 QA 的模型**。
- **唯一有意义的窄结论**（有数据支撑）：keep14 下，继承+冻结 trunk 让模型获得了一点**可答性判别力**（Youden J +0.202 vs −0.001），而随机初始化只学到了拒答的先验频率。这是真的，但它是关于一个二分类决策的，**不能包装成 general capability retention**。

---

## 4. P-C2 怎么救 —— 以及 CKA/knowledge-onset 不一致到底意味着什么

### 4.1 先正面处理这个新发现：**它削弱 P-C2，而且是决定性地削弱**

不是"两个探针略有分歧"，而是**四个前向探针族在同一个模型上给出横跨整个深度域的四个答案**（OLMo-2-7B, L=32）：

| 前向信号 | 它说的"adaptation/knowledge 深度" |
|---|---|
| 语言学 edge-probe sat95（POS/DEPREL/CoLA/WiC/SST2/RTE） | **0.000L – 0.156L**（layer 0-5） |
| knowledge logit-lens onset（P-C2 用的那个） | **0.562L**（layer 18） |
| next-token logit-lens sat95 | **1.000L**（layer 32） |
| **adaptation CKA drift 50% 质量点**（A2 全模型 FT 的实际漂移几何） | **0.938L**（layer 30）；25% 质量点 = L27 |

而且 CKA drift 的形状与 L18 毫无关系：它平滑单调下降，L14 以下只累积了总漂移质量的 **4.9%**，L20 以下 10.1%，L28 以下 30.7%——**膝点在 L28-32**（L30→L31 单步就吃掉 15.1 个百分点）。knowledge-onset 的 L18 在 CKA 曲线上什么都不是。

**更要命的是：哪个探针预测对了实测恢复曲线？** 用 #133 的 A4 深度扫（可答层 EM，占 BASE 比例：keep14 0.7% / keep20 12.5% / keep24 25.9% / keep28 44.2%）做回归：

- **CKA-drift 保留质量 vs 恢复比例：Pearson r = +0.9946**（n=4）
- **knowledge-logit-lens acc@cut vs 恢复比例：r = +0.7347**，且**系统性偏浅**：探针说 L18 之后知识就"在线"了（L18 acc 0.326 → L19 0.544），但实测 keep20（已过 onset）只恢复了 BASE 可答层 EM 的 **12.6%**，keep28 才 44.2%，**到 L28 都还在爬，没有任何饱和迹象**。

→ **P-C2 建立在那个具体探针（knowledge logit-lens onset）之上，而该探针被我们自己的深度扫falsify了：它给出的切点乐观了一大截。** 与它冲突的 adaptation-drift 几何反而几乎完美拟合——但 adaptation drift **必须先做一次全模型 FT 才能测**，因此**不是 forward-only**，恰好摧毁了 P-C2 的"廉价、免训练"卖点。

**这同时给出了 DoF 审计的实证版本**（比 rerun 的分析论证更硬）：名义上 α/β 只有 2 个拟合参数，但"选哪个探针"这一个自由度就横跨 0L→1L 的**全部**深度域。在 4 模型 × 6 任务 = 24 对上拟合 2 个参数听起来是 12:1，实际上分析自由度（探针族 × 阈值 × 平滑 × onset-vs-slope-vs-sat × 映射 ≈ 10 个选择）远超数据量。**不预注册 = 曲线拟合，不是预测。**

### 4.2 最难被打掉的可falsify版本（如果还要救）

一句话（不得再弱化）：

> 对 {OLMo-2-7B, Qwen3-8B-Base, OLMo-2-1B, Qwen3-4B} × {6 个 dev 任务} 这 24 个预先指定的 (模型, 任务) 对，**单一预注册的**前向探针量，在 leave-one-model-out **与** leave-one-task-out 两种折叠下，能预测出「保留全参 FT 增益 ≥95% 的**最小** (j, K) 配置」，且预测的 ĵ 与 oracle 相差 ≤2 层、K̂ 相差 ≤1，严格命中率 ≥80%，并**优于折内选出的最佳常数预测器 ≥5pp**。

必须钉死的五点：
1. **predicti­and 必须是"最小充分配置"，不能是 argmax。** 深度响应在 j 上单调（0.2930→0.3440→0.3560→0.4190；可答层更单调 0.7%→44.2%），argmax 平凡地等于最大 j → **argmax 版本 vacuous，不可用**。
2. **必须内置 abstain 输出**（当无任何 (j,K) 达到 95% 时预测"不可行"），且必须从一开始就在预测器里，不能事后加。
3. **零点必须是"折内最佳常数 (j,K)"**，不是 "always 0.6L"。若最佳常数已达 80%，claim 自动 vacuous。
4. **探针定义/阈值/平滑/映射必须在看到任何 freeze-graft 结果之前冻结。** 鉴于 §4.1 的四族分歧，这一条是生死线。
5. **必须处理探针本身的格式混淆**（这是本文另一半的核心指责，reviewer 一定会问）：容量匹配的 MLP 探针、标签置换、format-only 标签、跨格式平衡的内容标签、MDL/selectivity 控制、随机特征与未训练模型基线。

**成本（修正 rerun 的估算）：** 最小可信网格 = 6 切点 × 3 graft 宽 × 4 模型 × 6 任务 = **432 格**，+ 边界 3 seed 复制 ≈ **576 run**。rerun 按 24 GPU-h/run 算出 14,400 GPU-h ≈ 3 周——**这个单价错了**。实测：A3_keep20（22 层全参 fp32 AdamW）1000 步 = **51 分钟 / 8 卡 ≈ 7 GPU-h**；keep24=60min，keep28=70min；A4 freeze-graft 更便宜。所以 576 × 7 ≈ **4,000 GPU-h ≈ 100h / 40 卡 ≈ 4 天**，不是 3 周。

**但我仍然不建议做。** 理由：即使做成，§2.3 说明它填的是段落级缺口；而 §4.1 说明我们唯一手头的探针已经被falsify、唯一拟合得好的量不是 forward-only。**P-C2 不能当主贡献。** 它应降为"我们审计了 N 个前向探针，它们互相矛盾且预测不了前沿"的一个**否定性小节**（见 §5-R3）。

---

## 5. 建议的转向命题（排序）

资源前提（本轮实测）：**.252 8×B200-183GB 空闲；.104 8×H20 空闲；本机 8×L20A 约 5 分钟后空闲；.73/.82 被 PaperB 占用。** 单价：1000 步训练 ≈ 7 GPU-h（8 卡 ~50-70min）；freeze-graft 16L=50.8GB / 22L=56.4GB 可上 H20；全参 7B 16L=76.8GB 勉强，22L+ 必须 B200。SQuAD eval 2000 题 8 卡约 10-20 分钟。

---

### R1（★ 推荐写这一篇）压缩把监督变成捷径：结构化深度压缩下 Exact Match 会反转模型排名

**论断（可falsify）**：破坏性深度压缩会使 instruction/SFT 不成比例地学习高频输出模板而非问题条件化的内容，从而使常规 EM 高估能力，并在答案频率偏斜下**反转**模型排名；捷径依赖度随压缩程度单调上升。

**关键图**：双面板。x 轴 = 保留深度 j ∈ {14, 20, 24, 28, 32}；左 y = **可答层 EM**（占 BASE 比例），右 y = **捷径贡献占 raw EM 的比例**。每个点 = 一个 (模型族, 训练方法, 注入 skew 率) 三元组，方法线分 {full-depth LoRA, frozen-graft, same-depth scratch}，skew 率分 {0%, 25%, 50%}，OLMo-2-7B 与 Qwen3-4B/8B 分开画。**主 panel 的骨架数据已全部在手**（§0c 表：0.7/12.6/25.9/44.2/100.3% 与 99.1/87.6/75.4/64.3/48.5%）。

**已有证据**：§0b（93.8% 的 headline 来自拒答层；可答层 p=0.22 null）、§0c（可答层阶梯 + 捷径份额单调）、§0d（标签本身是坏的，BASE 74.2% 输出逐字 span）、#132（A4 在 52.4% PopQA / 53.2% NQ-open 上吐同一句拒答，BASE 只 2.2%/1.4%）、F1−EM≈0（A3 0.0007 / A4 0.0040 vs BASE 0.061）。

**需要的实验（2-4 个）**：
1. **重建干净数据集 + 注入式 skew（无 GPU，最关键）**：把 SQuAD SFT 重建为 train/val **同** dominant-target 率 ∈ {0%, 25%, 50%}，且"不可答"项通过**真正删掉含 gold 的 chunk** 来构造（不是现在的检索失配 artifact）。这是把观察变成因果的唯一途径。
2. **skew × 方法 × 深度因子实验**：{A4 keep14, A4 keep28, A2 LoRA} × {0%, 50%} × 2 模型族（OLMo-2-7B, Qwen3-4B）= 12 run。.104(H20，freeze-graft/LoRA 装得下) + .252(B200，全参臂)。≈ **84 GPU-h**。
3. **同 optimizer/同精度/3 seed 的去混淆复现**：A4 keep14 与 A3 keep14，统一 bf16 AdamW、同 LR grid、同数据序、全程 dump per-example。.104。≈ **42 GPU-h**。
4. **诊断电池（无新数据集，纯 eval）**：no-context / mismatched-context / counterfactual-context-swap（把 gold 实体换成另一个同型实体）/ answerability-分层拒答率 / 输出熵 / valid-span 率。这一组把"学到格式"与"学到能力"真正分开。≈ **16 GPU-h**。

**总计 ≈ 142 GPU-h ≈ 两节点并行不到 10 小时。**

**什么会falsify它**：如果「压缩 × skew」交互在**可答层准确率上 < 5pp**，或在两个模型族上方向不一致 → 死。（注意：主效应已经在手且远大于 5pp，风险主要在"交互"而非"主效应"。）

**最近竞争工作**：2305.15717（style vs factuality）——它没有因果检验破坏性结构压缩是否**放大**输出频率捷径，也没有展示 benchmark 排名反转。**次要风险（必须先查，见 §7）**：SQuAD 2.0 的 no-answer 校准文献可能已经报过"多数类先验主导 → 排名反转"。

**venue + 诚实录取概率**：EMNLP main / Findings，**~30%**。

---

### R2 继承分解：冻结的预训练 trunk 到底买到了什么

**论断**：在匹配了 optimizer/精度/token 预算/LR 搜索、且四格共用同一套预训练 embed/norm/lm_head 的条件下，预训练冻结前缀相对**随机**冻结前缀只带来很小但可复现的优势；而现有文献里"相对全随机浅模型的巨大增益"绝大部分来自继承的 embedding / norm / lm_head，而非 trunk 本身。

**关键图**：x = 保留前缀深度；y = 相对**随机冻结前缀**对照的干净下游增益（可答层 EM 与 Youden J 双指标）。每点 = (继承配置, 模型, 深度, seed)。配套一张分解表，把增益归因到 trunk / embed / norm+lm_head / cap-init 及其交互。

**已有证据**：§0e 的 Youden J（A4 +0.202 vs A3 −0.001，A3 恰好为 0 = 只学到先验频率）；§0b（可答层 Δ=+0.40pp, p=0.22 → 在 keep14 上这个效应**已经接近 null**）；`apply_freeze_front` 只冻 `model.layers.{<j}` 的代码事实。

**需要的实验**：
1. **代码先行（必须，1 个 coder 任务）**：加 `--random_trunk`——随机初始化 trunk 但**保留**预训练 embed/norm/lm_head。现在的 `--from_scratch` 会把三者一起随机化，这正是 confound ② 的来源，所以现有代码**无法**做这个对照。
2. 2×2 主因子：{继承, 随机} × {冻结, 不冻结}，keep14 + keep28，3 配对 seed，四格同 embed/norm/lm_head = 12 run（.252，全参格需 B200）≈ **84 GPU-h**。
3. cap-init 对照：{随机, 复制首个被丢层（LLM-Streamline 做法）, identity-like（LLaMA Pro 做法）} × keep14 = 3 run ≈ **21 GPU-h**。这条同时把"随机 init"这个我们唯一剩下的构造楔子**正面检验**掉——文献认为它更差，如果确实更差，就把它写成一个诚实的负结果。

**什么会falsify它**：匹配后的预训练冻结 trunk 在可答层平均增益 **< 1.0pp** 且没有任何深度 > 2pp → 继承效应太弱，不成篇。**当前证据倾向于它会 fail**（keep14 已 +0.40pp n.s.），所以它的生死取决于 §6-E0 那三个 A3 深度扫 eval。

**最近竞争工作**：2403.19135（目标是方法，不是因果分解）。**venue**：TMLR / COLM，**~35%**，但**前提是 E0 给出正信号**。

---

### R3 前向探针不能识别 adaptation 深度：一次审计（建议作为 R1 或 R2 的一个大节，而非独立篇）

**论断**：在同一个模型上，四族"廉价前向"信号对"adaptation/knowledge 起始深度"给出横跨 0L–1L 的互相矛盾的答案；其中唯一与实测 freeze-graft 恢复前沿高度相关（r=0.99）的那个量**需要先做一次全模型 FT**，因此前向-only 深度预测在当前证据下不可识别。

**关键图**：一张图叠四条归一化曲线（linguistic edge-probe sat95 / knowledge logit-lens / next-token logit-lens / CKA drift 累积质量），x = frac depth，竖线标出各自给出的"切点"（0.00-0.156L / 0.562L / 1.000L / 0.938L），再叠上实测可答层恢复曲线（keep14/20/24/28 的 0.7/12.5/25.9/44.2%）。配一张 r 值表（0.9946 vs 0.7347）。

**已有证据**：§4.1 全部数字**已在手**，来自
`results/probe_linguistic_{olmo2_7b,qwen3_8b,llama3_8b}.json`、
`results/knowledge_logit_lens_*.json` 和本目录
`evidence/adaptation_cka/onset_A2/cka_per_layer.json`。

**需要的实验**：几乎为零。补 Qwen3-8B 侧的 CKA drift 需要一次 Qwen LoRA FT（1 run，≈7 GPU-h）+ probe（前向，<1 GPU-h）。可选：#159 的 FRESH=4 网格补格，用来确认 K 也不可前向预测。

**什么会falsify它**：若某一族探针在 ≥3 个模型族上都与恢复前沿 r > 0.9 **且**是纯前向的 → 审计结论反转，P-C2 复活（这是好消息，欢迎发生）。

**最近竞争工作**：2403.03853 / 2403.17887 / 2607.09757 / 2606.05516 全部主张前向信号能选层——**正因如此这条审计有服务价值**（它直接质疑一条活跃的文献线）。**venue**：作为 R1/R2 的一节；若独立则 Findings / short，**~25%**。

---

### R4 蒸馏救援（最高上限，也最可能直接终结这个方向）

**论断**：K 层 cap 的失败原因是三者之一——容量不足 / 初始化-优化失败 / 下游监督诱导捷径——用"SFT 之前先蒸馏"可以把它们分开。

**做法**：冻结前缀，先在干净无标注文本（Dolmino）上训练 K 层 cap 去匹配原 32L 模型的 token logits（T=1 与 T=2），**在任何 SFT 之前**；然后做完全相同的 SFT。交叉 {无蒸馏, 有蒸馏} × cap-init {随机, 复制} × 切点 {14, 20, 28}。

**已有基础设施**：`scripts/train_olmo2_arch_probe2_distill.py` + `scripts/_run_olmo2_keep14_distill_heal.sh` 已存在且在跑（PaperB #99），λ=0.6、teacher_topk=64、NTP/KL 分开记录，可直接复用。

**需要的实验**：3-6 run（keep{14,20,28} × 有蒸馏），.252。≈ **21-42 GPU-h**（注意：蒸馏 step 更慢，实测 PaperB 侧 13.7-14.1 s/step，1000 步 ≈ 4h/8卡 ≈ 32 GPU-h/run，所以 3 run ≈ 100 GPU-h）。

**两种结局都有价值**：蒸馏后的 2 层 cap 若能在 SFT 之前恢复 base 能力、且 SFT 后不塌成拒答 → 我们有一篇**比上述任何提案都强的方法论文**。若连 keep28 都逼近不了被丢弃的那一叠 → **浅层重生本质上受容量限制，该方向应当终止**（这本身是一个可发表的、干净的负结论，也是最省时间的退出路径）。

**venue**：若成功 ICLR/NeurIPS ~25%；若失败，作为 R1/R2 的终结性小节。

---

### R5（不建议）最小可行 trunk 的可预测性 = 原 P-C2 最强可救版

见 §4.2。技术上可做（约 4,000 GPU-h ≈ 4 天全集群），但填的是段落级缺口，且唯一手头探针已被自己的数据falsify。**明确放弃作为主贡献。**

---

### 决策与放弃清单

**写 R1，把 R2 与 R3 作为它的两个机制小节，用 R4 作为 2 周内的高上限探索腿。** 决策规则：先跑 §6 的 E0（今天，几乎免费）。
- 若 A4 相对 A3 在 keep24/28 的**可答层** EM 差 **≥5pp** → 把 R2 升为正篇脊梁（"继承在中等压缩下才起作用"是一个更强的正面故事），R1 降为其评测-有效性小节。
- 若 < 5pp（当前证据倾向此结果）→ **按 R1 写**，R2 只作分解小节。

**明确放弃**：构造作为主贡献；P-C1 的"追平 full-FT"与"胜过参数匹配 LoRA"（A4 0.2930 vs LoRA 0.6590，可答层 0.0050 vs 0.6770，差两个数量级）；P-C2 作为主贡献；P-C3。

**必须撤掉的具体 claim（kill list）**：
1. "构造是新的" — 2403.19135 已做全套
2. "追平 full finetuning" — 差 36pp raw / 差 135× 可答层
3. "胜过参数匹配 LoRA" — 明确为假
4. "胜过 from-scratch"作为 headline — 93.8% 来自拒答层，可答层 p=0.22
5. **现有 SQuAD EM headline** — 恒定拒答器（0.4985）打赢每一个剪枝臂
6. **`data/squad_*.jsonl` 的"不可答"标签** — BASE 74.2% 用逐字 span 回答它们；数据集必须重建
7. BASE_ref 作对照 — 差两个轴（32L-vs-16L 且 no-SFT-vs-SFT）；只能作 intact 上限参照
8. argmax 深度预测 — 单调 ⇒ vacuous
9. 两模型的 adaptation-onset claim — 未交叉验证，且被 §4.1 falsify
10. keep14 "hero" 命名 — 它是最退化的那一格（可答层 = BASE 的 0.7%）
11. "随机 init 是贡献" — 是设计选择，且文献认为更差
12. 单 seed 结论（166 epoch 下尤甚）
13. 把 n=78,656 的 +0.39pp 显著性当作重要性
14. "LoRA 有能力税"作为新发现 — 2401.05605/2405.09673 已占；而且**在我们自己的域内可答层上 A2 与 BASE 统计相同（p=0.94），税只在域外**
15. 继续在被污染的 SQuAD split 上调参 — 只能作为一个被记录的病理案例

---

## 6. 最小可执行清单（假设只剩 10-20 个 training run）

**当前空闲：.252 8×B200（已确认 ALIVE + 0 进程）、.104 8×H20（0 进程）、本机 ~5 分钟后空闲。**
（注意：#155 PaperB keep14-distill 占 .73；#128 占 .82，均不要动。）

### E0 — 门禁项，今天，0 个新 training run

| 步骤 | 内容 | 节点 | 时长 |
|---|---|---|---|
| E0.1 | **A3_keep{20,24,28} 的 SQuAD eval**（训练已完成/近完成，ckpt 在 `outputs/paperC_pc1_squad_A3_keep*/`）。必须开 per-example dump | 本机（keep28 训完后立即） | ~1h |
| E0.2 | 全 10 臂统一按 **拒答层/可答层分层 + Youden J + 捷径份额 + F1−EM** 重打分（脚本已验证可行，见 §0） | CPU | 10 min |
| E0.3 | 把 §4.1 的四族探针对比图 + r 值表落盘（数据已在手） | CPU | 30 min |

**门禁判据**：E0.1 给出 keep24/28 上 A4−A3 的**可答层** EM 差。≥5pp → 走 R2 主线；<5pp → 走 R1 主线。

### E1 — R1 因果腿，13 run

| # | arm | skew | 模型 | 节点 | GPU-h |
|---|---|---|---|---|---|
| 1-6 | A4 keep14 / A4 keep28 / A2 LoRA | 0%, 50% | OLMo-2-7B | .104(H20) + .252(B200 跑 keep28) | 42 |
| 7-10 | A4 keep14 / A2 LoRA | 0%, 50% | Qwen3-4B | .104 | 28 |
| 11-13 | A3 keep14 | 0%, 50% + 1 复现 seed | OLMo-2-7B | .252 | 21 |

前置（无 GPU，必须先做）：重建同率 skew 数据集 + 真正的不可答项构造（删掉含 gold 的 chunk）。
**小计 ≈ 91 GPU-h ≈ 两节点并行 6h。**

### E2 — R2 分解腿，5 run（需先加 `--random_trunk`）

| # | 格 | 节点 | GPU-h |
|---|---|---|---|
| 14 | random-frozen trunk, keep14（共用预训练 embed/norm/lm_head） | .252 | 7 |
| 15 | random-frozen trunk, keep28 | .252 | 8 |
| 16 | inherited-unfrozen trunk, keep14 | .252 | 8 |
| 17-18 | cap-init = 复制首个被丢层 / identity-like, keep14 | .104 | 14 |

**小计 ≈ 37 GPU-h。** 阻塞项：coder 加 `--random_trunk`（约 15 行，`_classify_param` / transplant 路径各改一处）。

### E3 — R4 蒸馏救援，2 run（高上限探索）

| # | 内容 | 节点 | GPU-h |
|---|---|---|---|
| 19 | keep14 cap 先 logit-蒸馏（Dolmino，1000 步）再同样 SFT | .252 | ~32 |
| 20 | keep28 同上 | .252 | ~36 |

**小计 ≈ 68 GPU-h。** 复用 `train_olmo2_arch_probe2_distill.py`，零新代码。

### 总计

**20 个 training run ≈ 196 GPU-h ≈ .252 + .104 十六卡并行 13-15 小时**（含 eval 与失败重跑，给 2 天 buffer）。
关键路径不是算力，是**两个无 GPU 的前置**：(a) 重建干净/可控 skew 的评测集，(b) `--random_trunk` 代码开关。这两项今天就该派出去。

**执行顺序**：E0（今天，门禁）→ 前置 (a)(b) 并行 → E1（主线）+ E3（并行探索）→ E2（视 E0 结果决定是否升格）。

---

## 7. 诚实缺口

**因网关不稳而未查到、必须用别的方式补的：**

1. **R1 最大的未评估竞争风险**：SQuAD 2.0 的 no-answer / answerability-calibration 文献里，是否已经有人报过"多数类（拒答）先验主导 → 模型排名反转 / EM 高估能力"。这是 R1 的直接前置技术风险，**没查**。补法：手工检索 "SQuAD 2.0 no-answer bias", "majority-class baseline exceeds model", "answerability calibration", "degenerate refusal", "unanswerable question overprediction"；并读 SQuAD 2.0 原论文的 baseline 讨论。**在动 E1 之前必须查完。**
2. **2411.15558 是否显式冻结 trunk**（embed / final-norm 可训性）：三轮尝试全败（pdftotext "Couldn't find trailer dictionary"；repo 404）。现状 ~80% 解决。补法：换 ar5iv/HTML 版、或 OpenReview 页、或直接邮件作者。
3. **几乎所有 venue 字符串未核实**（仅 LLM-Streamline=ICLR 2025 Spotlight 已核）。补法：逐条走 DBLP / ACL Anthology / OpenReview，**入 .bib 前一条一条过**。
4. **2607.09757 的细节被污染**：rerun 编造了"RSLoRA"这个错标题以及 N=64 校准集 / 结构化低秩噪声 / cs.CV 分类等细节。正确标题 = **RSRA: Training-Free Probing of Representation Sensitivity for Efficient LoRA Rank Allocation**。其余细节 UNVERIFIED，需读原文。
5. **LLaMA Pro 的 identity-init 与 Llama SLayer 是否净更浅**：只到 abstract 级，未读代码/正文。
6. **r3 提到的 "Linear Probes Detect Task Format, Not Reasoning Mode" 未被验证，可能不存在**。这一条若真实，会直接冲击 R3；需确认。补法：arXiv 全文搜标题。
7. **三个 codex reviewer 首轮全挂**（根因已定位：`-c` 放在 `exec` **之后**会清掉 tcodex 注入的 `model_providers.tencent`，静默退回 `provider=openai` → 拨 `wss://api.openai.com` 死循环 40 分钟）。修复后 6 路复跑全 rc=0（58-133s）。**但 (d) 可发表性裁决与 (e) 转向命题是在"无网、纯判断"模式下补的**，其文献覆盖度未经检索验证——本备忘录 §5 的"最近竞争工作"一栏因此是**中等置信**，不是高置信。
8. **本备忘录 §0 的所有数字是我现场从 per-example 文件算的，未经第二人复核。** 复现命令：拷 `.73:/apdcephfs_zwfy6/.../paperC_squad_results/*/per_example_shard0of1.jsonl`，按 `data/squad_val.jsonl` 的 `target_text == '根据提供的信息无法回答这个问题'` 分层即可。建议 E0.2 独立复算一遍再入稿（铁律 2）。

---

## 附：关键文件路径

- 研究复跑：`/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/paperC_research/rerun_{scoop,confound,pc2,format,pc2prior,reframe}.md` + `rerun_CITATION_VERIFICATION.md` + `r{1,2,3}_salvaged.md`
- SQuAD 汇总：`paperC_squad_results/*_summary.json`（本机，仅 summary）
- **SQuAD per-example（关键，本机没有）**：`.73:/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/paperC_squad_results/<arm>/per_example_shard0of1.jsonl`；本轮副本 `/tmp/pcpe/*.jsonl`
- 探针：`results/probe_linguistic_{olmo2_7b,qwen3_8b,llama3_8b}.json`、
  `results/knowledge_logit_lens_{OLMo-2-1124-7B,Qwen3-8b-local}.json`、
  `evidence/adaptation_cka/onset_*/cka_per_layer.json`。
- 训练器（含 `apply_freeze_front`，L288-308）：`scripts/train_olmo2_arch_probe2.py`
- 蒸馏训练器（R4 复用）：`scripts/train_olmo2_arch_probe2_distill.py`
- SQuAD eval（已含 per-example dump）：`scripts/eval_paperC_squad_emf1.py`
- 数据（**需重建**）：`data/squad_train.jsonl`（10000，17.56% 拒答）、`data/squad_val.jsonl`（2000，49.85% 拒答）
- A3 深度扫 ckpt（待 eval）：`outputs/paperC_pc1_squad_A3_keep{20,24,28}fresh2/`；日志 `logs/paperC_133b_status.tsv`
- scoping：`scoping/SCOPING_AND_POSTMORTEM.md`；临时外部 brief 已删除。
