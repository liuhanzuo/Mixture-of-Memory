# Code Review Request — Sparse Memory Architecture

## 项目概述

本项目探索 **固定大小 memory buffer 压缩长上下文** 的方案：在 Llama2-7B 的每层 attention 中插入一个外部 memory bank，通过 chunk-level retrieval 读取 + EMA 写入来辅助 sliding window attention 以外的信息。

核心代码路径：
- `src/memory/sparse_memory/` — 主要实现（model.py, attention.py, memory_bank.py）
- `scripts/train_sparse_memory.py` — 训练脚本
- `scripts/eval_sparse_memory_ppl.py` — PPL 评测脚本

## 当前架构

### Attention Flow (per layer)
```
Input → LlamaSelfAttention (sliding window W=256)
      → Memory Read: q_chunk [B,H,1,d] × k_mem.T → top-K slots → gather(k_mem, v_mem) → attention → o_mem [B,T,D]
      → Gated Fusion: gate_proj(h) → σ → g * o_local + (1-g) * o_mem
      → Memory Write: importance scoring → top-K selection → EMA update memory slots
```

### Memory Bank
- Per-layer: `[B, N_slots, D]`，N_slots ∈ {128, 256}
- Write: importance-based top-K selection (magnitude × attention surprise) + learnable scorer
- Read: chunk-level mean-pooled query → cosine similarity → top-K retrieval
- EMA update: `memory = α * new + (1-α) * old`

## 实验结果（SlimPajama-6B, 5000 steps, Llama2-7B）

| 实验 | Slots | 写入方式 | PPL (pg19) | vs Vanilla |
|------|-------|---------|------------|-----------|
| Vanilla Llama2-7B | — | — | 5102 | baseline |
| full_write_256 | 256 | 全量 | **584** | 8.7× |
| selective_256 | 256 | top-16 | 646 | 7.9× |
| full_write_128 | 128 | 全量 | 844 | 6.0× |
| selective_128 | 128 | top-8 | 1403 | 3.6× |

## 待解决问题 / 疑问（请重点关注）

### 1. 🔴 生成质量严重退化
所有 memory 模型在 greedy decode 下 output degenerate repetition（重复句号或无限 0），而 vanilla 能正常生成。
- **PPL 和 generation quality 存在巨大 gap**
- 可能原因？
  - sliding window + memory 的 attention pattern 在 autoregressive decode 时累积误差？
  - memory write/read 在 generate 时行为和 train 不一致？
  - gate 的输出在长序列上逐渐偏向某一侧？
- 尝试过 temperature/top_p sampling 吗？（还没试过）

### 2. 🔴 选择性写入没有带来预期收益
原始假设：全量写入导致旧内容被快速覆盖（旧内容保留率仅 3.4%），选择性写入应该更好。
- **实际结果：full_write 全面优于 selective_write**
- 可能原因？
  - 全量写入天然提供更多训练信号（所有 token 都参与 memory 更新）
  - 选择性写入虽然保留率高，但可能过滤掉了有用的信息
  - importance scoring 本身可能不够准确（基于 magnitude + attention entropy）

### 3. 🟡 Memory 读取的 attention weight 分析缺失
- 不知道 memory read path 的 attention weight 分布如何
- 不清楚是 memory 真的被有效利用了，还是 gate 学会了几乎忽略 o_mem（像之前的 alpha=0.999 问题）
- 需要分析：memory attention weight 的 entropy、gate 值的分布、o_mem vs o_local 的贡献比

### 4. 🟡 Gate 机制过于简单
当前 gate = σ(W_g · h)，一个简单的标量 sigmoid。
- 没有层间差异化：所有层的 gate 用相同结构，没有 layer id 或 position embedding
- 没有考虑 token 重要性：哪些 token 更需要 memory vs local context
- 之前在 MAG 项目中观察到 gate 饱和到 ~0.999（几乎只用 local），不知道当前是否也有类似问题

### 5. 🟡 Chunk-level retrieval 的信息瓶颈
用 `q.mean(dim=T)` 得到一个 chunk 的单一 query 向量来检索 memory slots。
- 这丢失了大量 token-level 的细粒度信息
- top-K retrieval 共享同一个 chunk query，意味着同一 chunk 内的所有 token 读取相同的 memory slots
- 是否应该做 token-level 的 memory retrieval？

### 6. 🟢 PPL 评测在 pg19 上（训练数据是 SlimPajama）
- 分布不匹配：训练用 SlimPajama（网页文本），评测用 pg19（书籍）
- 但 vanilla baseline 也在 pg19 上测，所以相对比较是公平的
- 应该在 SlimPajama held-out 上也做一次 eval

### 7. 🟢 训练步数可能不够
5000 steps × batch_size 2 × grad_accum 4 = ~40K chunks seen（SlimPajama 有 156 万 chunks）
- 只看了 ~2.5% 的数据
- 增加步数是否能让 PPL 进一步下降？

### 8. 🟢 没有隔离 fine-tuning 贡献 vs memory 贡献
当前所有模型都经过 fine-tuning，没有"fine-tuned but no memory"的对照
- PPL 改进中多少来自 LM fine-tuning，多少来自 memory？

### 9. 🟡 Memory 在推理时的初始化
推理时 memory bank 初始为全零。
- 第一个 chunk 完全没有 memory 可读
- 需要多少 chunks 才能让 memory 变得有用？
- 是否应该用某种 "memory warmup" 策略？

### 10. 🟢 跨层 memory 独立性
32 层各自有独立的 memory bank，没有跨层信息交互。
- 不同层的 memory 可能存储冗余信息
- 是否应该共享 memory 或有跨层 routing？

## 代码结构问题

### attention.py
- `forward()` 方法很长（~100 行），读/写/fusion 都在一个函数里
- `@torch.no_grad()` 的使用不一致：写操作部分有 no_grad，但 importance scoring 是有梯度的
- memory write 发生在 forward pass 中间（side effect），不利于调试

### memory_bank.py
- `update_slots()` 中用 Python for loop 遍历 unique slot indices，可能有性能问题
- `nn.Parameter` 重新赋值来更新 memory（`self.memory = nn.Parameter(new_mem, ...)`）— 这种方式在 DDP 下是否安全？
- importance_head 的梯度是否真的在训练中有效传播？

### model.py
- memory_banks 作为 ModuleList 注册，但每层的 attention 是通过 `_memory_bank` 引用注入的（不是标准 nn.Module 子模块关系）
- `gradient_checkpointing` 被完全禁用（与 memory write side effects 不兼容）

## 期望的 Review 方向

1. **架构层面**：当前 chunk-level retrieval + gated fusion + EMA write 的设计是否有根本性缺陷？
2. **生成质量**：为什么 PPL 低但 generation 差？如何修复？
3. **选择性 vs 全量写入**：为什么 full write 更好？如何改进 selective write？
4. **Memory 利用率**：如何验证 memory 真的被有效使用了？
5. **扩展性**：这个架构能否 scale 到更长的序列（16K, 32K, 128K）？
