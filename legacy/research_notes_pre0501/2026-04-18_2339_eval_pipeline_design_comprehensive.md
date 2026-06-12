# Task 1: 综合评测方案设计

**调研时间**: 2026-04-18  
**调研主题**: RMT 模型综合评测方案设计  
**调研目标**: 设计覆盖细节检索、语言建模质量、长程依赖、通用能力、SWA baseline、Full attention ceiling 的完整 eval pipeline

---

## 执行摘要

[**结论**] 设计了完整的 6 维度评测 pipeline，覆盖：
1. NIH（细节检索）
2. PPL（语言建模质量）
3. 长程依赖（multi-hop reasoning）
4. 通用能力（MMLU / C-Eval）
5. SWA baseline（局部注意力下限）
6. Full attention ceiling（全注意力上限）

每个维度包含：推荐 benchmark、具体评测方法、Qwen3-8B+RMT 适配方案、实现复杂度评估。

---

## 1. 细节检索（Needle-in-Haystack）

### [fact] 推荐数据集

#### A. 原生 NIH（Kamradt 2023）
- **来源**: `needle-in-haystack` GitHub repo（原始实现）
- **特点**:
  - 干草堆：Paul Graham 论文 / Wikitext-103 / Wikipedia
  - 针：随机事实句（如 "The special number is 983"）
  - 位置：从 0% 到 100% 均匀分布（10% 间隔）
- **适用原因**:
  - 标准评测 long-context retrieval 能力
  - RMT 论文使用此评测
  - 实现简单，结果直观

#### B. Multi-Needle（扩展变体）
- **特点**:
  - 单文档内插入多个针（2-10 个）
  - 测试记忆容量和去重能力
  - 询问多个针的组合信息
- **适用原因**:
  - 测试 RMT 的多记忆压缩能力
  - 原生 NIH 可能过于简单

### [inference] 具体评测方法

```python
# 伪代码
def eval_nih(model, tokenizer, num_trials=100):
    results = {"by_position": {i: [] for i in range(0, 101, 10)}}

    for trial in range(num_trials):
        # 1. 生成 hay + needle
        hay = sample_haystack(length=4000)  # 4 segments (1024 each)
        needle = random_needle()
        position = random_position(0, len(hay))

        # 2. 插入针
        doc = insert_needle(hay, needle, position)

        # 3. 构建问题
        question = f"What is the special number mentioned in the text?"

        # 4. 运行模型
        with_memory(doc, model) as memory:
            answer = generate(question, memory=memory)

        # 5. 验证答案
        is_correct = extract_number(answer) == extract_number(needle)
        results["by_position"][position_bucket].append(is_correct)

    # 计算准确率
    for pos_bucket in results["by_position"]:
        acc = sum(results["by_position"][pos_bucket]) / len(results["by_position"][pos_bucket])
        results["by_position"][pos_bucket] = acc

    results["overall"] = mean(results["by_position"].values())
    return results
```

### [inference] Qwen3-8B + RMT 适配方案

**适配要点**:
1. **Segment 处理**:
   - 输入长度：4000-12000 tokens（4-12 segments）
   - Segment 长度：1024（与训练一致）
   - Memory tokens：64-128（建议增加到 64，当前 16 太少）

2. **Prompt 模板**:
   ```
   {haystack_with_needle}

   Q: {question}
   A:
   ```

3. **生成配置**:
   ```yaml
   max_new_tokens: 20  # 短答案足够
   temperature: 0.0    # deterministic
   do_sample: False
   ```

4. **验证逻辑**:
   - 正则提取数字/日期
   - 模糊匹配（处理拼写错误）
   - 位置无关（答案不需要包含位置信息）

### [inference] 实现复杂度评估

| 维度 | 复杂度 | 说明 |
|------|-------|------|
| **代码实现** | ⭐⭐ (低) | 简单的字符串操作 + 推理 |
| **计算成本** | ⭐⭐ (低) | 100 trials × 4-12 segments ≈ 1 小时（单 GPU） |
| **数据准备** | ⭐ (极低) | 使用现成的 haystack 数据集 |
| **维护成本** | ⭐ (极低) | 无需额外维护 |

**总计**: ⭐⭐ (低复杂度，快速实现)

---

## 2. 整体语言建模质量（PPL）

### [fact] 推荐数据集

#### A. Wikitext-103（标准 LM benchmark）
- **特点**:
  - 验证集：3.7M tokens
  - 测试集：0.4M tokens
  - 长文档（适合 RMT）
- **适用原因**:
  - LM 标准 benchmark
  - RMT 原论文使用此数据集
  - 有公开 baseline（GPT-2, Transformer-XL）

#### B. PG19（书籍数据集）
- **特点**:
  - 验证集：11K 书籍
  - 平均长度：60K tokens/book
  - 超长文档（测试 RMT 长序列能力）
- **适用原因**:
  - 更长文档（测试多 segment 压缩）
  - RMT 论文主要评测数据集

#### C. 中文数据（适配 Qwen3）
- **C4-ZH**: Common Crawl 中文子集
- **Wikipedia-ZH**: 中文 Wikipedia（与训练数据一致）
- **CLUECorpus2020**: 中文大规模语料

**建议**: 优先使用 Wikitext-103，如果需要中文评测，使用 Wikipedia-ZH（与训练数据一致）

### [inference] 具体评测方法

```python
# 伪代码
def eval_ppl(model, tokenizer, dataset="wikitext-103"):
    doc_losses = []

    for doc in dataset.test:
        # 1. 分段（与训练一致）
        segments = split_into_segments(doc, segment_length=1024)

        # 2. 运行 RMT（memory forward）
        memory = None
        total_loss = 0
        total_tokens = 0

        for segment in segments:
            outputs = model(segment, memory=memory, labels=segment)
            loss = outputs.loss  # cross-entropy
            total_loss += loss * segment.num_tokens
            total_tokens += segment.num_tokens
            memory = outputs.memory  # 更新 memory

        # 3. 记录文档级 PPL
        doc_ppl = exp(total_loss / total_tokens)
        doc_losses.append(doc_ppl)

    # 4. 计算平均 PPL
    mean_ppl = mean(doc_losses)
    std_ppl = std(doc_losses)

    return {
        "mean_ppl": mean_ppl,
        "std_ppl": std_ppl,
        "num_docs": len(doc_losses)
    }
```

### [inference] Qwen3-8B + RMT 适配方案

**适配要点**:
1. **Segment 处理**:
   - 与训练一致：segment_length=1024
   - Memory forward：使用与训练相同的 memory 传递机制
   - Padding：文档末尾 padding 到 segment_length

2. **Loss 计算**:
   - 使用 model 的 built-in loss（CE on next tokens）
   - Mask padding tokens（label=-100）
   - 不使用 reconstruction loss（与训练一致）

3. **Baseline 对比**:
   - **RMT（我们）**: 64 memory tokens
   - **Full attention ceiling**: 不使用 memory，segment 内 full attention
   - **SWA baseline**: segment 内 SWA（window=512），无 memory
   - **LoRA-only**: 同 RMT，但 memory 不参与计算（只占位）

### [inference] 实现复杂度评估

| 维度 | 复杂度 | 说明 |
|------|-------|------|
| **代码实现** | ⭐⭐⭐ (中) | 需要正确处理 memory forward |
| **计算成本** | ⭐⭐⭐ (中) | Wikitext-103 test set ≈ 0.4M tokens，约 1-2 小时 |
| **数据准备** | ⭐⭐ (低) | 需要下载 Wikitext-103 |
| **维护成本** | ⭐ (极低) | 无需维护 |

**总计**: ⭐⭐⭐ (中等复杂度，需要确保 memory forward 正确)

---

## 3. 长程依赖（Multi-Hop Reasoning）

### [fact] 推荐数据集

#### A. HotpotQA（multi-hop QA）
- **特点**:
  - 需要跨段推理（2-hop）
  - 桥接实体推理
  - 维基百科上下文
- **适用原因**:
  - 测试跨 segment 信息整合能力
  - RMT 必须记住中间步骤
  - 标准长程依赖 benchmark

#### B. 2WikiMultiHopQA
- **特点**:
  - 需要跨多个文档推理
  - 复杂 multi-hop（3+ hops）
  - 更长的上下文

#### C. SQuAD v2 + Long Context 变体
- **特点**:
  - 答案在文档不同位置
  - 测试长程检索 + 推理
  - 可以构造长文档版本

**建议**: 优先使用 HotpotQA，因为它是标准 multi-hop benchmark

### [inference] 具体评测方法

```python
# 伪代码
def eval_multi_hop(model, tokenizer, dataset="hotpotqa"):
    results = {"correct": 0, "total": 0}

    for sample in dataset.test:
        # 1. 构造长文档（concatenate 相关段落）
        context = concatenate_paragraphs(sample.context_paragraphs)
        # 长度：4000-12000 tokens（测试跨 segment）

        # 2. 构造问题
        question = sample.question

        # 3. 运行模型（带 memory）
        with_memory(context, model) as memory:
            answer = generate(
                prompt=f"{context}\n\nQ: {question}\nA:",
                memory=memory,
                max_new_tokens=50
            )

        # 4. 验证答案
        is_correct = match_answer(answer, sample.answer)
        results["correct"] += is_correct
        results["total"] += 1

    # 5. 计算准确率
    accuracy = results["correct"] / results["total"]
    return {
        "accuracy": accuracy,
        "num_samples": results["total"]
    }
```

### [inference] Qwen3-8B + RMT 适配方案

**适配要点**:
1. **Prompt 模板**:
   ```
   {context}

   Q: {question}
   A:
   ```

2. **生成配置**:
   ```yaml
   max_new_tokens: 50
   temperature: 0.3  # 低温度，确定性
   top_p: 0.9
   ```

3. **验证逻辑**:
   - HotpotQA: 提取 span，与 reference span 进行 F1 匹配
   - 答案可能是实体名 / 数字 / 短语
   - 使用 F1 score（而非 binary exact match）

4. **Baseline 对比**:
   - **RMT**: 应该优于 SWA（因为需要跨 segment 信息）
   - **Full attention**: 上限
   - **SWA**: 下限（可能无法跨 hop）

### [inference] 实现复杂度评估

| 维度 | 复杂度 | 说明 |
|------|-------|------|
| **代码实现** | ⭐⭐⭐⭐ (高) | 需要解析 HotpotQA 格式，答案匹配逻辑 |
| **计算成本** | ⭐⭐ (低) | HotpotQA dev set ≈ 7K samples，快速推理 |
| **数据准备** | ⭐⭐ (低) | 需要下载 HotpotQA |
| **维护成本** | ⭐⭐ (低) | 需要维护答案匹配逻辑 |

**总计**: ⭐⭐⭐ (中等复杂度，主要在答案匹配)

---

## 4. 通用能力保持（MMLU / C-Eval）

### [fact] 推荐数据集

#### A. MMLU（Massive Multitask Language Understanding）
- **特点**:
  - 57 个任务（STEM, 人文社科, 其他）
  - 多选题（4 选 1）
  - 测试广泛知识
- **适用原因**:
  - LM 标准 benchmark
  - 测试 RMT 是否引入能力退化
  - 与 GPT-3, Claude 等可对比

#### B. C-Eval（中文评测）
- **特点**:
  - 52 个任务（覆盖 52 个学科）
  - 中文多选题
  - 测试中文能力
- **适用原因**:
  - Qwen3 是中文模型
  - 测试中文通用能力
  - MMLU 中文版

**建议**: 两者都评测，MMLU 主要，C-Eval 作为补充

### [inference] 具体评测方法

```python
# 伪代码
def eval_mmlu(model, tokenizer, dataset="mmlu"):
    results = {"by_subject": {}, "overall": {"correct": 0, "total": 0}}

    for subject, samples in dataset.by_subject.items():
        subject_correct = 0
        subject_total = len(samples)

        for sample in samples:
            # 1. 构造 prompt（few-shot）
            prompt = construct_few_shot_prompt(
                sample.question,
                sample.choices,
                num_examples=5
            )

            # 2. 运行模型（RMT，但不需要 memory）
            answer = generate(
                prompt=prompt,
                max_new_tokens=1  # 只需生成 A/B/C/D
            )

            # 3. 验证答案
            is_correct = match_choice(answer, sample.answer)
            subject_correct += is_correct
            results["overall"]["correct"] += is_correct
            results["overall"]["total"] += 1

        # 4. 记录科目准确率
        results["by_subject"][subject] = subject_correct / subject_total

    # 5. 计算平均准确率
    overall_acc = results["overall"]["correct"] / results["overall"]["total"]
    results["overall"]["accuracy"] = overall_acc

    return results
```

**注意**: MMLU 不需要长 context（prompt < 1K tokens），主要测试能力保持（没有退化）

### [inference] Qwen3-8B + RMT 适配方案

**适配要点**:
1. **Few-shot Prompting**:
   - 使用 5-shot examples（标准设置）
   - Prompt 格式：
     ```
     Question: {question}
     Choices: {A} {B} {C} {D}
     Answer: {answer}

     (repeat 5 times)

     Question: {test_question}
     Choices: {test_A} {test_B} {test_C} {test_D}
     Answer:
     ```

2. **生成配置**:
   ```yaml
   max_new_tokens: 1  # 只需生成单个字符 A/B/C/D
   temperature: 0.0
   do_sample: False
   ```

3. **验证逻辑**:
   - 匹配首个字符（A/B/C/D）
   - 忽略标点和空格
   - 大小写不敏感

4. **Baseline 对比**:
   - **LoRA-only（无 memory）**: 基线
   - **RMT**: 应该与 LoRA-only 持平（没有退化）
   - **退化检测**: 如果 RMT < LoRA-only > 2%，说明有问题

### [inference] 实现复杂度评估

| 维度 | 复杂度 | 说明 |
|------|-------|------|
| **代码实现** | ⭐⭐⭐ (中) | 需要 few-shot 构造，答案匹配 |
| **计算成本** | ⭐ (极低) | 57 tasks × 140 questions ≈ 8K 推理，快速 |
| **数据准备** | ⭐⭐ (低) | 需要下载 MMLU |
| **维护成本** | ⭐ (极低) | 无需维护 |

**总计**: ⭐⭐ (低复杂度，快速实现)

---

## 5. SWA Baseline（滑动窗口下限）

### [fact] 设计理念

SWA baseline 的目的是：**在相同 segment 长度下，纯丢信息的下限**。

#### 设置说明
- **Segment 内**: SWA（window=512 或 1024）
- **Segment 间**: 无 memory（信息丢失）
- **目的**: 证明 RMT 的 memory 确实有作用

### [inference] 具体评测方法

SWA baseline 在所有评测维度都运行，但设置不同：

```python
# 伪代码：SWA baseline 配置
swa_config = {
    "segment_length": 1024,      # 与 RMT 一致
    "window_size": 512,          # 或 1024（更严格）
    "memory_tokens": 0,          # 无 memory
    "memory_forward": False,     # 不传递 memory
}
```

### [inference] 对比策略

| 评测维度 | RMT | SWA Baseline | 预期结果 |
|---------|-----|--------------|---------|
| **NIH** | 高准确率 | 低准确率（尤其远位置） | RMT >> SWA |
| **PPL** | 低 PPL | 高 PPL | RMT << SWA |
| **Multi-hop** | 高准确率 | 低准确率 | RMT >> SWA |
| **MMLU** | 持平 | 持平 | RMT ≈ SWA（短 prompt） |

**关键洞察**: MMLU 应该持平（短 prompt，不需要 memory），说明 RMT 的 overhead 没有破坏基础能力。

### [inference] Qwen3-8B + SWA 适配方案

**适配要点**:
1. **模型配置**:
   - 使用项目中的 `SWABackbone`（已实现）
   - window_size=512（或 1024）
   - 无 memory injection

2. **评测运行**:
   - 与 RMT 使用相同的评测脚本
   - 仅修改 model 配置

3. **控制变量**:
   - 同 backbone（Qwen3-8B）
   - 同 LoRA rank（r=32）
   - 同训练数据
   - 同训练步数

### [inference] 实现复杂度评估

| 维度 | 复杂度 | 说明 |
|------|-------|------|
| **代码实现** | ⭐ (极低) | 已有 SWABackbone，直接使用 |
| **计算成本** | ⭐⭐ (低) | 比 RMT 更快（无 memory） |
| **数据准备** | ⭐ (极低) | 使用相同数据 |
| **维护成本** | ⭐ (极低) | 无需维护 |

**总计**: ⭐ (极低复杂度，直接复用)

---

## 6. Full Attention Ceiling（全注意力上限）

### [fact] 设计理念

Full attention ceiling 的目的是：**不压缩的 upper bound**。

#### 设置说明
- **Segment 内**: Full causal attention
- **Segment 间**: 无 memory（无必要，segment 内已全 attention）
- **目的**: 理论最佳性能（O(n²) cost）

### [inference] 具体评测方法

```python
# 伪代码：Full attention 配置
full_attention_config = {
    "segment_length": 1024,      # 与 RMT 一致
    "window_size": None,         # 全 attention
    "memory_tokens": 0,          # 无 memory
    "memory_forward": False,     # 不传递 memory
}
```

**注意**: Full attention 不需要 memory，因为 segment 内已经能看到所有 token。

### [inference] 对比策略

| 评测维度 | RMT | Full Attention | 预期结果 |
|---------|-----|----------------|---------|
| **NIH** | 中高准确率 | 最高准确率 | Full ≳ RMT |
| **PPL** | 中低 PPL | 最低 PPL | Full ≲ RMT |
| **Multi-hop** | 中高准确率 | 最高准确率 | Full ≳ RMT |
| **MMLU** | 持平 | 持平 | Full ≈ RMT |

**关键洞察**: Full attention 应该是上限，但 RMT 应该接近（gap < 10-20%）。

### [inference] Qwen3-8B + Full Attention 适配方案

**适配要点**:
1. **模型配置**:
   - 使用 `FullAttentionBackbone`（或禁用 SWA）
   - 标准 causal attention
   - 无 memory

2. **评测运行**:
   - 与 RMT / SWA 使用相同的评测脚本
   - 仅修改 model 配置

3. **控制变量**:
   - 同 backbone（Qwen3-8B）
   - 同 LoRA rank（r=32）
   - 同训练数据
   - 同训练步数

### [inference] 实现复杂度评估

| 维度 | 复杂度 | 说明 |
|------|-------|------|
| **代码实现** | ⭐ (极低) | 使用标准 attention，无需修改 |
| **计算成本** | ⭐⭐⭐⭐⭐ (极高) | O(n²) cost，长文档很慢 |
| **数据准备** | ⭐ (极低) | 使用相同数据 |
| **维护成本** | ⭐ (极低) | 无需维护 |

**总计**: ⭐⭐⭐ (中等复杂度，但计算成本高，建议只在小数据集上运行)

---

## 7. Fair Comparison 控制变量

### [inference] 控制变量设计

| 变量 | RMT | SWA | Full Attention | LoRA-only |
|------|-----|-----|----------------|-----------|
| **Backbone** | Qwen3-8B | Qwen3-8B | Qwen3-8B | Qwen3-8B |
| **LoRA rank** | 32 | 32 | 32 | 32 |
| **训练数据** | Wiki 10k docs | Wiki 10k docs | Wiki 10k docs | Wiki 10k docs |
| **训练步数** | 20 epochs | 20 epochs | 20 epochs | 20 epochs |
| **Segment length** | 1024 | 1024 | 1024 | 1024 |
| **Memory tokens** | 64 | 0 | 0 | 64（不使用）|
| **Window size** | None（full） | 512 | None | None |

### [inference] LoRA-only 说明

**LoRA-only** 的目的是：**同 RMT 架构，但 memory 不参与计算**。

#### 设置说明
- **Memory tokens**: 64（与 RMT 一致）
- **Memory forward**: 是，但不参与 attention（mask 掉）
- **目的**: 测试 RMT 的架构 overhead 是否影响性能

#### 对比意义
- **RMT vs LoRA-only**: 测试 memory 是否有用
- **LoRA-only vs SWA**: 测试 full attention vs SWA 的差异
- **LoRA-only vs Full**: 测试 LoRA-only 是否接近 Full

---

## 8. 评测优先级

### [inference] 快速验证（P0，必须）
1. **NIH**: 最直接测试 RMT 是否工作
2. **PPL**: 标准 LM benchmark
3. **MMLU**: 测试能力退化

### [inference] 深入分析（P1，重要）
4. **Multi-hop**: 测试长程依赖
5. **SWA baseline**: 下限对比

### [inference] 理论上限（P2，可选）
6. **Full attention ceiling**: 计算成本高，可选

---

## 9. 实现建议

### [inference] 模块化设计

```python
class EvalRunner:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def eval_nih(self, dataset):
        pass

    def eval_ppl(self, dataset):
        pass

    def eval_multi_hop(self, dataset):
        pass

    def eval_mmlu(self, dataset):
        pass

    def run_all(self, datasets):
        results = {}
        results["nih"] = self.eval_nih(datasets["nih"])
        results["ppl"] = self.eval_ppl(datasets["ppl"])
        results["multi_hop"] = self.eval_multi_hop(datasets["multi_hop"])
        results["mmlu"] = self.eval_mmlu(datasets["mmlu"])
        return results
```

### [inference] 输出格式

```json
{
  "model": "qwen3-8b-rmt-v1",
  "timestamp": "2026-04-18T23:39:00Z",
  "config": {
    "memory_tokens": 64,
    "segment_length": 1024,
    "max_segments": 6
  },
  "results": {
    "nih": {
      "overall_accuracy": 0.85,
      "by_position": {...}
    },
    "ppl": {
      "mean_ppl": 12.5,
      "std_ppl": 2.3
    },
    "multi_hop": {
      "accuracy": 0.62
    },
    "mmlu": {
      "overall_accuracy": 0.45
    }
  }
}
```

---

## 10. 关键发现与建议

### [fact] 当前配置问题
根据文献调研（参见 RESEARCH_LITERATURE.md 第 11 节）：
1. **Memory tokens 过少**: 当前 16 tokens，文献推荐 64-128（16:1-8:1 compression）
2. **Compression ratio 过高**: 当前 64:1，文献推荐 16:1-32:1
3. **Memory/segment ratio**: 当前 1.6%，文献推荐 5-10%

### [inference] 建议
1. **立即增加 memory tokens** 到 64（16:1 compression）
2. **增加训练数据** 到 10K+ docs（当前 3.6K）
3. **增加训练步数** 到 20+ epochs（当前 3 epochs）

### [guess] 预期结果
- **NIH**: 当前 16 tokens 可能导致准确率 < 50%（尤其远位置），增加到 64 tokens 后应该 > 80%
- **PPL**: 当前 16 tokens 可能导致 PPL > 20，增加到 64 tokens 后应该 < 15
- **Multi-hop**: 当前 16 tokens 可能导致准确率 < 40%，增加到 64 tokens 后应该 > 60%
- **MMLU**: 应该持平（与 LoRA-only），如果下降 > 2% 说明有问题

---

## 11. 参考文献

1. **Bulatov et al. (2023)** - "Recurrent Memory Transformer" (arXiv:2306.14095)
2. **Kamradt (2023)** - "Needle in a Haystack" (GitHub repo)
3. **Wu et al. (2022)** - "Memorizing Transformers" (ICML 2022)
4. **HotpotQA Team (2018)** - "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering" (EMNLP 2019)
5. **Hendrycks et al. (2020)** - "Measuring Massive Multitask Language Understanding" (ICLR 2021) - MMLU
6. **Huang et al. (2023)** - "C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for Foundation Models" (arXiv:2305.08322)
7. **Merity et al. (2017)** - "Wikitext-103" - PPL benchmark
8. **Beltagy et al. (2020)** - "Longformer" (ACL 2020) - SWA baseline
9. **Rae et al. (2019)** - "Compressive Transformer" (ICLR 2020) - Long-context compression
10. **Dai et al. (2019)** - "Transformer-XL" (NeurIPS 2019) - Recurrent memory baseline

---

## 12. 下一步行动

1. **立即执行**:
   - 修改配置：memory_tokens=64
   - 重新训练（20 epochs）
   - 运行 NIH 评测（验证是否有改进）

2. **短期目标**:
   - 实现 PPL 评测
   - 实现 MMLU 评测
   - 运行 SWA baseline 对比

3. **中期目标**:
   - 实现 Multi-hop 评测
   - 分析 memory utilization
   - 调整 hyperparameters

4. **长期目标**:
   - Full attention ceiling 评测（可选）
   - 多 memory tokens 数量对比（16 vs 32 vs 64 vs 128）
   - 不同 segment length 对比（512 vs 1024 vs 2048）

---

**调研完成时间**: 2026-04-18 23:39 GMT+8  
**调研人员**: researcher (subagent)  
**状态**: ✅ 完成
