# 长程记忆 / 上下文压缩文献调研 —— 训练目标、防 shortcut、eval、selection

> 调研员：longmem-researcher，2026-06-20。纯文献调研，不写代码、不碰 GPU。
> 目的：为 mem_space 32k 长程精确读出撞墙后的决策提供文献依据，重点回答：
> (1) 有没有论文用「一致性/蒸馏目标」（teacher 看完整 context、student 只看压缩 memory）避免 LM-loss shortcut；
> (2) 推荐 eval benchmark；(3) selection 机制有没有不训独立 selector 的成功先例。
> 数字/机制均来自原文 abstract / HTML 正文；未核实处显式标注「待核实」。**不编造引用/数字。**

---

## 0. 一句话结论（给决策）

- **★最直接对标我们想验证方向的论文 = KV-Distill (arXiv:2503.10337, 2025)**：它**显式把「压缩 cache vs 未压缩 cache」当成 student-teacher，用 KL 散度匹配输出分布**——正是我们设想的「一致性/蒸馏目标」。结论：worst-case 抽取任务上显著优于其他压缩法，长文 QA/摘要逼近未压缩。**这条路有先例、且 work。**
- **★防 shortcut 最关键的两个工程手段（被多篇验证）**：(a) **chunk-wise 随机压缩比采样**（Activation Beacon，消融证明显著优于 instance-wise），(b) **对比式负样本构造**（Focused Transformer 的 crossbatch：同文档=正、他文档=负），逼 memory 学会判别相关而非抄邻接。
- **★selector-free 成功先例确凿**：Focused Transformer / Memorizing Transformers / Landmark **都没有训练独立 selector**——selection 靠「attention 表示空间被对比/bottleneck 塑造 + 推理时 kNN/top-k 涌现」。这直接支持我们 H2（显式 selector 训不出）的对策：**不要训 selector，要塑造表示空间 + 结构 bottleneck。**
- **★推荐 eval（除 BABILong qa1 外）**：**RULER**（多类 needle + multi-hop + aggregation，长程精确读出黄金标尺）、**LongBench**（真实长文档迁移性，我们已在用）、**∞Bench/InfiniteBench**（>100k 极端长程）。passkey/LongEval-lines 作为机制 sanity probe 保留。

---

## 1. 训练目标横向对照（最重要）

> 核心问题：连续文本上 next-token LM loss → 模型抄邻接 token 而非学长程检索。各方法用什么 loss？谁用了一致性/蒸馏？

| 方法 | 年份 | 训练目标 | backbone | 是否蒸馏/一致性 | 防 shortcut 手段 |
|---|---|---|---|---|---|
| **Landmark Attention** (2305.16300) | 2023 | **纯 LM loss** + in-window grouped-softmax | LLaMA-1-7B 全量 FT | 否 | landmark-token 是**读取必经 bottleneck**（结构，非 loss）|
| **AutoCompressor** (2305.14788) | 2023 | **无监督 LM loss**，长文分段、summary vectors 递归传递 | OPT/Llama, soft prompt | 否 | 分段递归 + summary 累积；无显式 anti-shortcut |
| **ICAE** (2307.06945, ICLR'24) | 2023 | **autoencoding（重建原文）+ LM 双目标** | Llama + LoRA(~1% param) | 部分（重建≈自蒸馏）| **重建 loss 逼 memory slot 精确保留内容**（非 gist）|
| **Gisting** (2304.08467) | 2023 | LM loss + gist-token attention mask | 指令压缩 | 否 | mask 强制信息过 gist token |
| **CCM / Compressed Context Memory** (2312.03414, ICLR'24) | 2023 | **LM loss**（target output PPL），conditional LoRA | LLaMA/Mistral | 否 | 并行化递归压缩训练 |
| **Activation Beacon** (2401.03462, ICLR'25) | 2024 | **纯 next-token LM loss**（beacon token 不计 loss）| Llama-2/Qwen-2，原参冻结、仅训 beacon proj | 否 | ★**chunk-wise 随机压缩比 {2,4,8,16,32}**（消融证明显著防 shortcut）|
| **CEPE** (2402.16617, ACL'24) | 2024 | LM loss，小 encoder + cross-attn，**decoder 冻结** | Llama-2 | 否 | 并行 chunk 编码 + cross-attn 注入 |
| **Focused Transformer / LongLLaMA** (2307.03170) | 2023 | **标准 LM loss + crossbatch 对比式数据构造** | OpenLLaMA-3B/7B | 否（对比 shaping）| ★**正=同文档 prev-ctx，负=(d-1) 他文档 KV**，逼判别相关 key |
| **Memorizing Transformers** (2203.08913, ICLR'22) | 2022 | 标准 LM loss + kNN 检索（非可微）| decoder-only | 否 | memory 是 raw KV，靠 kNN top-k |
| **RMT / Recurrent Memory Transformer** (2207.06881) | 2022 | LM loss + **BPTT over segments**，curriculum 加长 | memory token | 否 | 段间 BPTT 传梯度（待核实细节）|
| **★KV-Distill** (2503.10337) | 2025 | **★KL 散度：teacher=完整 KV cache，student=压缩 KV cache，匹配输出分布** | 多尺寸/架构，PEFT adaptor，原模型能力保留 | **★是（核心）** | KL 匹配天然抑制「靠局部续写」，强制全局输出一致 |

### 1.1 ★ 有没有论文用「一致性/蒸馏」避 shortcut —— 有，且是当前最相关方向

**KV-Distill（arXiv:2503.10337, 2025-03）是最直接的先例**，原文 abstract 原话：

> "We treat a compressed-uncompressed cache as a student-teacher pairing and apply a **KL-type divergence to match the generated outputs**. KV-Distill outperforms other compression techniques in **worst-case extractive tasks** and approaches uncompressed performance in long context QA and summarization... reduce lengths by up to **99%** while preserving downstream performance."

- **机制对我们的意义**：teacher 看完整 KV、student 看压缩 KV，KL 匹配「在 student 端复现 teacher 的输出分布」。这**正是我们 dolmino self-study 蒸馏（A=logits KL + B=hidden MSE）的思路**——区别在 KV-Distill 蒸的是 **KV cache 压缩**（保留 raw KV 子集），不是 lossy slot；且它**在 worst-case 抽取任务上专门验证**（这正是我们 32k 精确读出墙的形状）。
- **为什么 KL 可能比纯 LM loss 更抗 shortcut**：纯 LM loss 在连续文本上，target 的正确续写可从邻接 raw token 平凡获得（我们 keep_all leak 的根因 = `lm 0.0006`）。KL 蒸馏的监督信号是 **teacher 的完整分布**，teacher 也只能从全局 context 得出，student 若只抄邻接则无法匹配 teacher 在「依赖远端事实的 token」上的分布 → 提供了邻接抄不来的梯度压力。**待核实**：KV-Distill 是否直接对比过 KL vs LM-loss 的 anti-shortcut 效果（原文未在 abstract 给，需读正文 ablation）。
- **另一相关**：**Latent Context Compilation (arXiv:2602.21221, 2026)** 把 long context 蒸馏进极小 latent tokens，正文有「KL vs MSE 蒸馏目标对比」的 ablation（搜索结果可见，**全文待核实**）——可作为 logits-KL vs hidden-MSE 选择的二手依据。

### 1.2 ★ ICAE 的「重建 loss」是另一条防 gist-only 的路

ICAE 用 **autoencoding（从 memory slot 重建原文）+ LM** 双目标预训练。重建目标**逼 memory slot 精确、完整地表示原文**（abstract: "accurately and comprehensively represent the original context"），而非只保 gist。这对我们「保 gist、丢精确事实」的诊断是直接的反向药方——**但**注意 ICAE 只做到 4× 压缩（Llama），远低于我们 250:1，重建在高压缩比下是否成立**待核实**。

### 1.3 纯 LM-loss 阵营靠什么不抄邻接

不用蒸馏的方法（Landmark / Beacon / Beacon / Focused）靠**结构或数据**而非 loss：
- **Landmark**：landmark-token 是后续 token grouped-softmax 的**必经 bottleneck**，主梯度路径密集 → 被迫学概括（我们 team 已确认：landmark 绕开 H2 靠「读取必经」而非「可微 selector」）。
- **Activation Beacon**：beacon 不计 loss、原参冻结，靠 **chunk-wise 随机压缩比**让模型不能依赖固定 granularity 的抄写。
- **Focused Transformer**：靠**对比式负样本**（见 §4）。

---

## 2. 怎么避免 training-time leak / shortcut（具体手段清单）

按可迁移到我们的优先级排序：

1. **★对比/负样本构造（Focused Transformer crossbatch）**——最对症。
   - 「distraction issue」：文档越多，相关 key 占比越低，attention 均匀摊到相关+无关 key（r_d ≈ 1/d）。**这与我们「512 块挑 needle 选不准 + dilution」是同一现象。**
   - 解法：训练时给 memory 层喂 **正样本=同文档 prev-context 的 (K,V)，负样本=(d-1) 个他文档的 (K,V)**，在可微 attention 里逼模型判别。d 从 ≤8 起步（否则忽略 prev-ctx）后切到 ≥64。
   - **对我们的启发**：与其在连续 pg19 文本上做 next-token（target 可抄邻接），不如**构造「needle 所在块=正、其余 distractor 块=负」的对比任务**，让 readout/selection 学判别而非续写。这天然规避 leak（负块里没有 target 的答案）。
2. **★chunk-wise 随机压缩比（Activation Beacon）**：每个 chunk 的压缩比从 {2,4,8,16,32} 随机采。消融：chunk-wise（40.5）≫ instance-wise（37.7）。防止模型学到固定 granularity 的抄写捷径。
3. **合成 needle 任务做监督**：Beacon 用 GPT-3.5 合成 QA（每 segment 4 对）控制 context 长度；本质是**让 target 只能从远端 needle 得到**（我们的 T2 already 这么做：answer 物理只能跨块 readout，target 不含 code，t2_needle 1.94→0 证明是真检索）。**这条我们已对**——T2 无 leak，leak 只在 dolmino 连续文本 + keep_all。
4. **重建/自编码目标（ICAE）**：逼 memory 保精确内容（§1.2）。
5. **冻结原参 + 仅训压缩模块（Beacon / CEPE decoder 冻结）**：避免破坏 base LM，但**注意**这正是我们 frozen-reader 撞墙的范式——Beacon 之所以 work 是因为它注入的是 raw activation（beacon KV 进 self-attn）而非 lossy slot，且 reader 在训练时见过这条路径。

> **我们的 leak 诊断与文献完全吻合**：team 已确认 keep_all + pg19 连续文本 → target 从邻接 raw-KV 平凡复制（lm 0.0006）。文献给的两条出路：(a) 换成**对比式/合成 needle 数据**（target 不在邻接），(b) 换成**KL 蒸馏目标**（监督来自 teacher 全局分布）。两者都绕开「连续文本 next-token 可抄邻接」的根。

---

## 3. Eval Benchmark —— 测什么 + 我们该不该用

| benchmark | 测什么（一句话）| 长度 | 难度 | 我们该不该用 |
|---|---|---|---|---|
| **passkey retrieval**（Landmark 用）| 在垃圾填充里找一个随机数字 | 0–256k | 易（单 needle，无语义干扰）| **保留作机制 sanity**：破墙 demo 的最低门槛，不能当终点（RULER 证明 passkey 满分 ≠ 长程理解）|
| **LongEval lines-retrieval**（我们在用）| 从 N 行 key-value 里精确取一行 6 位数 | 1k–32k | 中（精确单条，强干扰）| **保留作 readout probe**：我们已发现 ≥8k 归零，是干净的开关性诊断 |
| **★RULER** (2404.06654) | **13 任务 4 类**：① NIAH 变体（多 needle/多类型）② **multi-hop tracing**（变量追踪）③ **aggregation**（频率统计/词频）④ QA | 自定义到 128k | **高**：扩展 vanilla NIAH，专测「超越检索」的行为。论文发现：几乎所有模型 passkey 满分但随长度大幅掉、声称 32k 只有半数真能撑 | **★强烈推荐**：长程精确读出的黄金标尺，且区分「检索 vs 聚合 vs 多跳」——正好对应我们想知道 memory 到底卡在哪一环 |
| **∞Bench / InfiniteBench** (OpenBMB) | >100k 超长 context 的理解+推理多任务 | **100k+** | 高（极端长度）| **推荐（后期）**：验证 memory 范式在极端长度的天花板，但当前我们 32k 都没破，优先级次于 RULER |
| **LongBench**（我们在用）| 真实长文档：单/多文 QA、摘要、few-shot、代码 | ≤32k | 中高（真实分布）| **★保留**：我们已用它证明「BABILong 突破不迁移真实长文档」。是迁移性的权威判据 |
| **BABILong** (2406.10149)（我们在用 qa1/2/5）| bAbI 20 类逻辑推理 + 长 noise 填充，测分布式事实推理/状态追踪 | 0k–1M | 中高（合成事实链）| **★保留 qa1（单事实=NIAH 语义）+ qa2/qa5（多事实/计数）**：我们的主控制轴。但记住它是**合成事实链**，与真实长文档测不同能力 |
| **LongMemEval**（我们在用）| 对话长期记忆：信息抽取、多会话推理、时序、知识更新 | 多会话 | 高 | **保留作对话记忆判据**：已证明我们「保 gist 丢精确」（base ≈3.8×）|

### 推荐组合（真实训练后该跑哪几个）
1. **RULER**（新增，最高优先）——细分检索/多跳/聚合，定位 memory 卡在哪一环，是比 passkey 严格得多的长程精确读出标尺。
2. **BABILong qa1 + qa2/qa5**（已在用）——单/多事实控制轴，与我们历史结果同口径可比。
3. **LongBench**（已在用）——真实长文档迁移性的硬判据（防止「合成突破不迁移」误判）。
4. **passkey + LongEval-lines** 降级为**机制 probe**（破墙 demo 的 sanity，不作终点指标）。
5. ∞Bench 留作极端长度的后期天花板验证。

---

## 4. Selection / 检索机制 —— 不训独立 selector 的成功先例

> 我们已证明显式 selector 训不出（H2，0% needle precision）。文献里靠涌现/结构 bottleneck 的成功先例：

| 方法 | selection 怎么做 | 是否训独立 selector | 关键点 |
|---|---|---|---|
| **★Focused Transformer / LongLLaMA** | 推理时 kNN top-k；**selection 从对比 shaping 的 (K,V) 空间涌现** | **否** | crossbatch 对比让相关 key 与 query 内积自然更高 → kNN 自然选对。**无 selector 网络，无 aux selection loss** |
| **★Memorizing Transformers** | 推理时 **kNN（非可微）**检索 raw KV top-k 进指定 memory 层 | **否** | memory = raw KV，靠 query·key 内积；selection 完全非参数化 |
| **★Landmark Attention** | grouped-softmax：landmark-token 分数门控块，推理 top-k 取真实块 raw KV | **否（无 aux loss）** | landmark-token 是**读取必经 bottleneck**，selection 信号 == 训出来的 attention 本身 |
| **我们的 mem_space gist selector** | TopKSelector：pooled hidden 打分 N slot，STE 硬 top-k + MoE aux | **是（失败）** | 0% precision。**旁路 bottleneck**：col_bias 反传无梯度压力 → 学成乱规则（H2）|

### ★ 对我们的核心启示（三方先例一致）

**所有 work 的方法都没有训独立 selector**——它们靠以下两个之一（或组合）：

1. **结构 bottleneck（Landmark）**：让 selection 信号成为**读取的必经路径**，使密集 LM 梯度被迫流经它 → selection 被「主任务」塑造，而非靠旁路 aux loss。我们的 gist selector 是旁路（col_bias），所以训不出——这与 team 已有判断「绕 H2 靠结构不靠可微」**完全吻合且有文献背书**。
2. **表示空间对比塑造（Focused / Memorizing）**：不显式训 selector，而是**用对比/负样本把 (K,V) 空间塑造成「相关 key 与 query 内积高」**，推理时 kNN/top-k 自然选对。Focused 的 crossbatch 正是解决「distraction（相关 key 占比低、attention 摊平）」——**这就是我们 dilution 墙的同构问题，且它有解。**

**可落地的方向（供 team-lead 决策）**：
- **方案 A（结构 bottleneck，对标 Landmark）**：放弃旁路 gist selector，把 selection 做成读取必经的 in-window grouped-softmax bottleneck（team 的 "B in-window summary-key bottleneck" 正是此路，文献强背书）。
- **方案 B（对比塑造 + kNN，对标 Focused Transformer）**：训练时构造「needle 块=正 / distractor 块=负」的 crossbatch 对比，塑造 raw-KV 空间，推理用 kNN top-k，**完全不训 selector**。这同时解决 (i) selector 训不出 (H2) 和 (ii) dilution（对比直接对抗 distraction）。
- **方案 C（KL 蒸馏，对标 KV-Distill）**：teacher 全 KV、student 压缩 KV，KL 匹配输出，专攻 worst-case 抽取——直接对标我们 32k 精确读出墙的形状，且避连续文本 LM-loss leak。

---

## 5. 证据缺口（待核实，未编造）

1. **KV-Distill 是否直接对比 KL vs LM-loss 的 anti-shortcut 效果**：abstract 未给，需读正文 ablation（arXiv:2503.10337 PDF 此次 fetch 失败，仅得 abstract）。
2. **Latent Context Compilation (2602.21221) 的 KL vs MSE ablation 结论**：搜索结果可见有此 ablation，全文待核实。
3. **ICAE 重建目标在高压缩比（>>4×）下是否仍逼出精确保留**：ICAE 只验证到 4×，我们 250:1 是否成立未知。
4. **RMT 的 BPTT/curriculum 具体配方**：本轮未深挖原文，仅据二手；如要复现需读 2207.06881 正文。
5. **Focused Transformer crossbatch 是否能迁移到「冻结 backbone + 注入」范式**：FoT 是 fine-tune memory 层（非冻结 reader），与我们 frozen-reader 范式不完全一致——对比塑造在冻结 reader 下是否成立待验证。
</content>
