# LoRA + Memory Token Injection 调研报告

**日期**: 2026-04-20
**目的**: 分析为什么 LoRA + 外部记忆注入在 7B-8B 模型上效果弱，以及文献中的成功/失败案例

---

## 1. RMT 原始论文用的什么规模？

**Bulatov et al., AAAI 2024** (arXiv: 2304.11062)

- **模型规模**: BERT-base, GPT-2 (124M), GPT-Neo (1.3B), OPT-125M ~ OPT-175B
- **微调方式**: **全量微调**（所有参数都参与训练，包括 memory token embeddings）
- **Memory tokens**: 2~10 个
- **Segment length**: 512 tokens
- **训练方式**: BPTT with curriculum learning（从 1 segment 逐步增加到 N segments）
- **关键结果**:
  - 在 BERT 上实现了 2M token 的记忆保持
  - PPL 随训练 segment 数增加而改善
  - **全量微调**才能让模型学到 memory 的读写操作

**关键发现**: 论文明确提到可以使用 parameter-efficient methods（附录），但所有展示的强大结果都是**全量微调**的。

---

## 2. LoRA + Memory Injection 有成功案例吗？

**没有找到在 7B+ 模型上用 LoRA + memory token injection 成功的论文。**

主要原因是：
- LoRA 只影响低秩子空间，而"学会使用外部 memory"需要改变 attention 的全局分配策略
- Memory token 本身的 embedding 不被 LoRA 覆盖（除非额外加入 trainable embedding）
- 7B+ 模型的 base 能力已经很强，在 4K context 内不需要记忆辅助

---

## 3. 成功案例的关键技巧

### RMT (Bulatov 2022, 2024)
- **全量微调**，让模型的所有参数都学会配合 memory
- **Curriculum learning**：从 1 segment 开始，逐步增加到目标 segment 数
- **Segment mixing**：训练时混合不同 segment 数量，提高泛化能力
- **BPTT**：梯度需要流过多个 segment 才能训练 memory 写入操作
- **专门的 memorization 任务**：不是用通用 NLP 数据，而是用需要跨 segment 记忆的合成任务

### Activation Beacon (ICLR 2025)
- 通过学习特殊的 "beacon token" 来压缩 KV cache
- 是 inference-time 的压缩，不需要改训练
- 但只适用于已训练好的长上下文模型

### LESS: Synthesizing Recurrence with KV Cache Compression (ICML 2024)
- 将 KV cache 压缩和 recurrence 结合
- 不是注入 memory token，而是直接压缩 KV cache
- 同样不需要改训练流程

---

## 4. 为什么 LoRA 在这个任务上特别弱

### 4.1 低秩限制 vs 全局行为改变
LoRA 的低秩更新只能影响 attention 矩阵的一个小子空间。但"学会使用 memory"需要：
- 让某些层的 attention 主动 attend to memory token positions
- 改变 attention 的分配策略（从均匀分布到集中到 memory）

这需要 attention 权重的**高秩变化**，LoRA 做不到。

### 4.2 Memory embedding 不在 LoRA 覆盖范围
Memory token 的 embedding 是模型参数的一部分，但 LoRA 只加在 linear layers 的 W_q, W_k, W_v, W_o 上。除非额外把 embedding 设为 trainable，否则 memory token 的表示空间很受限。

### 4.3 Base model 的捷径效应
7B-8B 模型在 4K context 内的 attention 已经很强。训练时如果下游 segment 的 loss 不严格依赖前面 segment 的信息，模型会走捷径——直接 attend 本 segment 内容，忽略 memory。这解释了：
- Alpha 饱和到 0.999（MAG）
- Memory token 变成噪声（RMT v10）
- Slot memory 只有 ctx=2048 时偶尔有效

### 4.4 BPTT 梯度瓶颈
在 bptt_depth=2 时，信息要流过 2 个 segment 才能更新 memory 写入模块。LoRA 的低秩梯度更容易在此路径上消失。

---

## 5. 近期（2024-2025）长上下文记忆方向的突破

### 5.1 KV Cache 压缩（主流方向）
- **Activation Beacon** (ICLR 2025): 学习特殊 token 压缩 KV cache，inference-time 适用
- **LESS** (ICML 2024): 合成 recurrence + KV cache 压缩
- **KVzip** (NeurIPS 2025): 利用模型自身重建原始 context 来评估 KV 重要性
- **SCOPE** (ACL 2025): 优化 KV cache 压缩

这些方法**不依赖训练时注入 memory**，而是直接压缩已有的 KV cache。

### 5.2 Memory-Efficient Reasoning
- **LightThinker++** (2026): 训练 LLM 动态压缩推理历史内容

### 5.3 Long-Term Memory Benchmarks
- **LongMemEval** (2025): 评估 LLM 聊天助手的 5 种长期记忆能力
- **MemEval** (2025): 评估 AI agent 的记忆系统

---

## 6. 对我们项目的建议

### 6.1 LoRA 方向基本可以放弃
在 7B+ 模型上，LoRA + memory injection 的组合在文献中没有成功先例。我们的实验（MAG alpha 饱和、RMT v10 memory 噪声、Slot memory 20% 准确率）与文献一致。

### 6.2 如果继续 RMT 方向
- **必须用全量微调**或至少 LoRA + trainable memory embedding + attention bias
- **必须用 curriculum learning**（从 1 segment 开始逐步增加）
- **训练数据必须有跨 segment 依赖**（我们的日常对话/小说数据可能不够）
- 考虑用 RMT 论文的合成 memorization 任务做预训练

### 6.3 更有前景的方向
1. **KV cache 压缩**（Activation Beacon 风格）：不需要训练时注入 memory，直接压缩推理时的 KV
2. **Full fine-tuning + RMT**：在 L20A 集群上用全量微调（7B 模型 32 GPU 可以做到）
3. **Mixtral/小模型全量微调**：用更小的模型（1.3B）做全量微调验证 RMT 概念

---

## 7. 关键论文列表

1. Bulatov et al. "Beyond Attention: Breaking the Limits of Transformer Context Length with Recurrent Memory" AAAI 2024
2. Bulatov et al. "Recurrent Memory Transformer" NeurIPS 2022
3. Activation Beacon: "Long context compression with activation beacon" ICLR 2025
4. LESS: "Get More with LESS: Synthesizing Recurrence with KV Cache Compression" ICML 2024
5. KVzip: "KVzip: Query-Agnostic KV Cache Compression" NeurIPS 2025
6. LongMemEval: "Benchmarking Chat Assistants on Long-Term Interactive Memory" 2025
7. SCOPE: "SCOPE: Optimizing Key-Value Cache Compression" ACL 2025
