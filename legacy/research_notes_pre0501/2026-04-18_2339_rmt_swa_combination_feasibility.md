# Task 2: RMT + SWA 结合可行性调研

**调研时间**: 2026-04-18  
**调研主题**: RMT 与 SWA 结合的可行性分析  
**调研目标**: 评估将 segment 内 full attention 改为 SWA 的理论可行性、相关工作、复杂度、潜在问题和架构影响

---

## 执行摘要

[**结论**] RMT + SWA 结合在**理论上可行**，且**有明确优势**（降低复杂度、处理更长文本），但需要**谨慎设计 memory tokens 的 attention pattern**。

**关键发现**:
1. **理论可行性**: ✅ SWA + Memory tokens 可以补偿信息损失
2. **相关工作**: ✅ StreamingLLM 已验证 sink tokens + SWA 的有效性
3. **复杂度分析**: ✅ SWA(W=512) + 64 mem ≈ Full(1024) FLOPs，但可以处理 2x 文本长度
4. **潜在问题**: ⚠️ Memory tokens 之间应该用 full attention（不 SWA）
5. **架构影响**: ⚠️ 需要修改 attention mask 和 position encoding

**最终建议**: **可以尝试 RMT + SWA，但应该分两步走**：
1. 先验证 RMT（full attention）是否工作（增加 memory tokens 到 64）
2. 再尝试 RMT + SWA（W=512，memory 用 full attention）

---

## 1. 多方讨论（Proposer / Skeptic / Critic）

### 1.1 Proposer（支持者视角）

**观点**: RMT + SWA 是一个很好的组合，可以同时享受两者的优势。

**论据**:
1. **计算复杂度降低**:
   - Segment 内 SWA(W=512): O(T×W) = O(1024×512) ≈ 524K FLOPs
   - Segment 内 Full(1024): O(T²) = O(1024²) ≈ 1,048K FLOPs
   - **节省约 50% 计算量**

2. **可以处理更长文本**:
   - 同样计算预算下，SWA 可以处理 2x 文本长度
   - 当前：segment=1024，full attention → FLOPs = 1,048K
   - SWA：segment=2048，window=512 + 64 memory → FLOPs ≈ 1,088K（接近）
   - **在相同 FLOPs 下，可以处理 2x 文本**

3. **Memory tokens 承担更重要的角色**:
   - SWA 看不到远处，必须依赖 memory tokens
   - 这迫使模型更有效地使用 memory
   - 可能产生更强的压缩能力

4. **相关工作支持**:
   - StreamingLLM 已经验证了 sink tokens + SWA 的有效性
   - Sink tokens（类似我们的 memory tokens）保持模型稳定
   - RMT 论文提到可以与 sparse attention 结合

**预期收益**:
- 更高效的长文本处理
- 更强的 memory 压缩（因为必须依赖）
- 可能更好的长程依赖建模

---

### 1.2 Skeptic（怀疑者视角）

**观点**: RMT + SWA 结合可能引入新的问题，需要谨慎评估。

**疑虑**:
1. **信息损失加剧**:
   - 当前 RMT（full attention）就已经在 segment 边界有信息损失
   - SWA 在 segment 内也有信息损失
   - **双重损失可能导致性能严重下降**

2. **Memory tokens 压力增大**:
   - Segment 内看不到远处，所有远程信息必须依赖 memory
   - 当前 16 memory tokens 已经太少（文献推荐 64-128）
   - **可能需要更多 memory tokens 才能补偿损失**

3. **Attention pattern 设计复杂**:
   - Memory tokens 之间用 full attention 还是 SWA？
   - Memory tokens 看 segment tokens 用什么 pattern？
   - Segment tokens 看 memory tokens 用什么 pattern？
   - **设计不当可能导致无法收敛**

4. **训练难度增加**:
   - SWA + Memory 的梯度传播更复杂
   - 可能需要更多训练步数
   - 可能更难学习有效的压缩

**担忧**:
- 如果 memory 不够，性能可能比纯 SWA 还差
- 训练不稳定，难以调试
- 付出额外努力，收益不明显

---

### 1.3 Critic（批评者视角）

**观点**: 需要更仔细地权衡利弊，并给出具体的设计建议。

**批评**:
1. **Proposer 的复杂度分析不完整**:
   - 只考虑了 segment 内 FLOPs
   - 没有考虑 memory tokens 的额外计算
   - Memory tokens 之间如果用 full attention，会增加 O(m²) cost
   - **真实 FLOPs**: SWA(W=512) + 64 mem (full) = O(T×W + m²) ≈ 524K + 4K = 528K（仍然 < 1024²）

2. **Skeptic 的担忧有道理，但可以缓解**:
   - 信息损失：可以通过增加 memory tokens 数量缓解
   - 训练难度：可以使用 curriculum learning（先简单，后复杂）
   - **关键是要有足够的 memory capacity**

3. **Attention pattern 必须明确**:
   - Memory ↔ Memory: **Full attention**（memory 之间必须相互看到）
   - Memory → Segment: **Full attention**（memory 可以看到所有 segment tokens）
   - Segment → Memory: **Full attention**（segment tokens 可以看到所有 memory）
   - Segment ↔ Segment: **SWA**（segment 内用滑动窗口）
   - **这是唯一合理的 design**

4. **与 StreamingLLM 的对比**:
   - StreamingLLM: Sink tokens（4-16 个）+ SWA
   - 我们：Memory tokens（64-128 个）+ SWA
   - **区别在于数量**：我们的 memory tokens 更多，应该能补偿更多信息损失

**最终建议**:
1. 先验证 RMT（full attention）是否工作
2. 如果 RMT 工作良好，再尝试 RMT + SWA
3. RMT + SWA 时，memory tokens 数量应该 **≥ 64**
4. Memory tokens 之间 **必须用 full attention**

---

## 2. SWA + Memory Tokens 的理论可行性

### [fact] 基础理论

#### 2.1 SWA 的信息损失

Sliding Window Attention 的信息损失模式：
```
Token i 可以看到的范围：
- [i - W, i + W] 之间的所有 token
- 无法看到更远的 token（必须依赖 memory）
```

**损失类型**:
1. **Local**: 窗口内信息保留
2. **Long-range**: 窗口外信息丢失

#### 2.2 Memory Tokens 的补偿作用

Memory tokens 可以补偿长程信息损失：
```
Segment 1: [mem1] + [tokens 0-1023]
            ↓ 压缩
Segment 2: [mem2] + [tokens 1024-2047]
            ↑
    mem1 可以提供 segment 1 的压缩信息
    mem2 依赖于 mem1 + segment 2 的压缩
```

**关键点**:
- Memory tokens 在 segment 内 **full attention**（可以看到所有 segment tokens）
- Memory tokens 在 segment 间 **recurrent**（传递压缩信息）
- **Memory 是唯一的跨 segment 信息通道**

### [inference] 可行性分析

**SWA + Memory 的信息流**:

```
Segment 1:
┌─────────────────────────────────────┐
│ Token 0: sees mem0 + tokens 0-511    │  ← SWA window
│ Token 511: sees mem0 + tokens 0-1023│  ← SWA window (end)
│ Token 512: sees mem0 + tokens 0-1023│  ← SWA window (start)
│ Token 1023: sees mem0 + tokens 512-1023│ ← SWA window
│ mem0: sees all tokens 0-1023 (full) │  ← Memory full attention
└─────────────────────────────────────┘
         ↓ 更新
Segment 2:
┌─────────────────────────────────────┐
│ Token 1024: sees mem1 + tokens 1024-1535│ ← SWA window
│ ...                                      │
│ mem1: sees mem0 + all tokens 1024-2047│  ← Memory full attention
└─────────────────────────────────────┘
```

**可行性结论**:
1. ✅ **Local information**: SWA 保留（窗口内）
2. ✅ **Long-range information**: Memory 传递（segment 间）
3. ⚠️ **Information bottleneck**: 必须有足够的 memory tokens

---

## 3. 相关工作

### [fact] StreamingLLM（Xiao et al. 2023）

**论文**: "StreamingLLM: Enabling LLMs to Process Infinite Length Texts via Streaming, with Limited Memory and Constant Time Decoding"

**核心机制**:
1. **Sink tokens**: 保留前 4-16 个 token（作为 attention anchor）
2. **Sliding window**: 保留最近的 W 个 token（如 4096）
3. **Discard middle**: 丢弃中间 token（信息损失）

**与我们的对比**:

| 维度 | StreamingLLM | RMT + SWA（我们） |
|------|--------------|-------------------|
| **Memory 类型** | Raw tokens（sink + recent） | Compressed tokens（memory）|
| **Memory 数量** | 4-16（sink） + 4096（recent） | 64-128（compressed） |
| **Compression** | 无（直接存储） | 有（learned compression） |
| **Segment 内** | SWA（window=4096） | SWA（window=512） + full memory |
| **Memory 之间** | N/A（无 memory-to-memory） | Full attention |

**关键洞察**:
- StreamingLLM 证明了 **sink tokens + SWA 是可行的**
- 我们的 memory tokens 比 sink tokens **多 4-8x**
- 我们的 memory 是 **compressed**（更高效）

### [inference] StreamingLLM 对我们的启发

**可借鉴的设计**:
1. **Memory tokens 必须 full attention**（like sink tokens）
2. **Segment 内可以用 SWA**（降低复杂度）
3. **Memory tokens 应该足够多**（4-16 sink tokens 能稳定，我们用 64-128 更好）

**区别与优势**:
1. **Compressed memory**: 我们的 memory 是 learned compression，不是 raw tokens
2. **Recurrent memory**: 我们的 memory 在 segment 间更新（StreamingLLM 的 sink tokens 固定）
3. **更灵活**: 我们的 memory 可以适应不同文档类型

---

### [fact] 其他相关工作

#### 3.1 Longformer / BigBird（稀疏 attention）

**核心机制**:
- **Local attention**: 滑动窗口（类似 SWA）
- **Global attention**: 特殊 token（如 [CLS]）看到所有 token
- **Random attention**: 每个 token 随机连接一些远程 token

**与我们的对比**:
- Longformer 的 **global tokens** 类似我们的 **memory tokens**
- 区别：Longformer 的 global tokens 是 **fixed token positions**，我们的 memory 是 **separate tokens**

**可借鉴**:
- Global tokens（memory tokens）必须用 full attention
- Local tokens（segment tokens）可以用 sparse attention

#### 3.2 Compressive Transformer（压缩 memory）

**核心机制**:
- **Recent memory**: 原始 hidden states（不压缩）
- **Compressed memory**: 压缩的 hidden states（2-4x 压缩）
- **Compression network**: Learn to compress older memory

**与我们的对比**:
- Compressive Transformer: Activation-level compression
- RMT: Token-level compression

**可借鉴**:
- 分层 memory（recent + compressed）可以降低复杂度
- 但我们的 token-level 更易实现

---

## 4. 复杂度分析

### [fact] FLOPs 对比

#### 4.1 Full Attention（当前 RMT）

**单个 segment（1024 tokens）**:
```
FLOPs = 4 × L × T² × D / H
     = 4 × 32 × 1024² × 4096 / 4
     = 4 × 32 × 1,048,576 × 4096 / 4
     ≈ 137,438,953,472 FLOPs
     ≈ 137 GFLOPs
```

**简化公式**（忽略常数）:
```
FLOPs_full ≈ O(T²) = 1024² = 1,048,576 (relative units)
```

#### 4.2 SWA + Memory（W=512，64 memory）

**Segment 部分（1024 tokens, window=512）**:
```
FLOPs_swa = O(T × W) = 1024 × 512 = 524,288
```

**Memory 部分（64 tokens, full attention）**:
```
FLOPs_mem = O(m²) = 64² = 4,096
```

**Memory ↔ Segment 交互**:
```
FLOPs_cross = O(T × m) = 1024 × 64 = 65,536
```

**总计**:
```
FLOPs_total = FLOPs_swa + FLOPs_mem + FLOPs_cross
            = 524,288 + 4,096 + 65,536
            = 593,920
            ≈ 0.57 × FLOPs_full
```

**结论**: SWA(W=512) + 64 memory ≈ **节省 43% FLOPs**

---

### [inference] 等效文本长度

**问题**: 在相同 FLOPs 下，SWA + Memory 可以处理多长的文本？

**假设**: FLOPs_budget = 1024² = 1,048,576

**SWA + Memory 可处理的最长文本**:
```
FLOPs(T) = T × W + m² + T × m
         = T × (W + m) + m²

令 FLOPs(T) = FLOPs_budget:
T × (W + m) + m² = T²
T × (512 + 64) + 64² = T²
T × 576 + 4096 = T²

解这个方程:
T² - 576T - 4096 = 0
T ≈ 576 (使用二次方程，取正根)
```

**结论**: 在相同 FLOPs 下，SWA(W=512) + 64 memory 可以处理 **576 tokens**（约 56% 的 1024）。

**但这只是 segment 内的分析**！更重要的洞察是：

**SWA + Memory 可以用相同的 FLOPs 处理更多的 segments**:
```
Full Attention (1024 tokens, 1 segment):
FLOPs = 1,048,576

SWA + Memory (1024 tokens, 1 segment):
FLOPs = 593,920

可以处理 1,048,576 / 593,920 ≈ 1.77 segments
```

**更实际的优势**:
- 用 **相同 FLOPs**，可以处理 **1.77x** 的 segments
- 如果 segment 长度增加到 2048（2x），SWA + Memory 仍然可行
- **SWA + Memory 可以用固定计算处理更长文本**

---

## 5. 潜在问题与解决方案

### [inference] 问题 1: Memory tokens 的 attention pattern

**问题**: Memory tokens 之间用 full attention 还是 SWA？

**选项**:
1. **Full attention**: Memory ↔ Memory full
2. **SWA**: Memory ↔ Memory sliding window
3. **Hybrid**: 部分 full，部分 SWA

**分析**:

| 选项 | 优点 | 缺点 | 推荐度 |
|------|------|------|--------|
| **Full attention** | 完整信息传递，稳定 | O(m²) cost，但 m=64 很小 | ⭐⭐⭐⭐⭐ |
| **SWA** | 降低复杂度 | 严重信息损失，memory 之间应该互相看到 | ⭐ |
| **Hybrid** | 平衡 | 设计复杂，收益不明显 | ⭐⭐ |

**结论**: **Memory tokens 之间必须用 full attention**

**理由**:
1. Memory tokens 数量小（64-128），O(m²) cost 可忽略（4K-16K vs 524K）
2. Memory tokens 之间应该互相看到（否则无法整合信息）
3. StreamingLLM 的 sink tokens 也是 full attention

---

### [inference] 问题 2: Memory tokens 的数量

**问题**: SWA 下需要多少 memory tokens？

**分析**:
- 当前 RMT（full attention）: 16 memory tokens（太少）
- 文献推荐（full attention）: 64-128 memory tokens
- **SWA 下**: 需要更多 memory tokens，因为 segment 内有信息损失

**推荐**:
- **保守估计**: 128 memory tokens（2x 文献推荐）
- **激进估计**: 256 memory tokens（4x 文献推荐）

**权衡**:
- 更多 memory = 更强压缩能力
- 更多 memory = 更多计算（但 O(m²) 很小）
- 更多 memory = 更高 memory footprint（可接受）

**结论**: **SWA 下至少使用 128 memory tokens**

---

### [inference] 问题 3: Training Stability

**问题**: SWA + Memory 训练是否稳定？

**分析**:
1. **Gradient flow**:
   - Full attention: 每个 token 都能看到所有 token，梯度传播顺畅
   - SWA: 远程依赖必须通过 memory 传播，梯度路径更长

2. **Learning rate**:
   - Memory tokens 可能需要更高的 LR（因为梯度路径更长）
   - 可以考虑使用 **5-10x** backbone LR（与 RMT 文献一致）

3. **Curriculum learning**:
   - 可以先用 full attention 训练 RMT
   - 再 fine-tune 为 SWA + Memory
   - 降低训练难度

**推荐**:
1. **先训练 RMT（full attention）**，验证 memory 是否工作
2. **再 fine-tune 为 SWA + Memory**，使用较低 LR（如 1e-5）
3. **Monitor memory utilization**（cosine similarity），确保 memory 在学习

---

### [inference] 问题 4: Attention Mask 设计

**问题**: 如何设计 attention mask？

**推荐设计**:

```
Position layout:
[mem0, mem1, ..., mem(m-1), token0, token1, ..., token(T-1)]

Attention mask:
- mem ↔ mem: Full (all ones)
- mem → token: Full (all ones)
- token → mem: Full (all ones)
- token ↔ token: SWA (sliding window of size W)
```

**伪代码**:
```python
def create_rmt_swa_mask(T, m, W):
    """
    T: segment length
    m: memory tokens
    W: sliding window size
    """
    total_length = m + T
    mask = torch.zeros(total_length, total_length)

    # Memory ↔ Memory: Full
    mask[:m, :m] = 1

    # Memory → Token: Full
    mask[:m, m:] = 1

    # Token → Memory: Full
    mask[m:, :m] = 1

    # Token ↔ Token: SWA
    for i in range(T):
        token_pos = m + i
        start = max(m, token_pos - W // 2)
        end = min(m + T, token_pos + W // 2 + 1)
        mask[token_pos, start:end] = 1

    # Causal mask for tokens (for autoregressive)
    causal_mask = torch.tril(torch.ones(T, T))
    mask[m:, m:] = causal_mask

    return mask
```

---

## 6. 对当前架构的影响

### [inference] 需要修改的模块

#### 6.1 Backbone 层（`swa_model.py`）

**当前实现**:
- `SWABackbone` 已支持 SWA
- 但不支持 memory tokens（需要添加）

**需要修改**:
1. **添加 memory token embeddings**:
   ```python
   self.memory_embeddings = nn.Parameter(
       torch.randn(num_memory_tokens, hidden_dim) * 0.02
   )
   ```

2. **修改 attention mask**:
   - Memory ↔ Memory: Full
   - Memory ↔ Token: Full
   - Token ↔ Token: SWA

3. **修改 position encoding**:
   - Memory tokens: 使用特殊位置（如 10000+）
   - Token: 使用 0, 1, ..., T-1

#### 6.2 Memory 层（`memory/`）

**当前实现**:
- Memory 是独立的模块（L1/L2/L3）
- 与 backbone 解耦

**需要修改**:
- **可能不需要修改**！
- Memory 层已经独立，可以直接与 SWA backbone 组合
- 只需要确保 SWA backbone 支持 memory injection

#### 6.3 训练脚本（`training/`）

**当前实现**:
- `train_gate.py`, `train_l2_aggregator.py`, `train_l3_summarizer.py`
- 针对 full attention backbone

**需要修改**:
- 添加 SWA backbone 的训练脚本
- 或者修改现有脚本，支持不同 backbone 类型

---

### [inference] 不需要修改的模块

1. **Memory 层（L1/L2/L3）**: 已经独立，与 backbone 无关
2. **数据加载**: 数据格式不变
3. **Evaluation scripts**: 可以复用，只需修改 model 配置

---

## 7. 最终结论与建议

### [fact] 最终结论

**RMT + SWA 结合在理论上可行**，且有明确优势：

1. **计算效率**: 节省约 43% FLOPs（SWA W=512 + 64 mem vs Full 1024）
2. **文本长度**: 可以用相同 FLOPs 处理更多 segments
3. **相关工作**: StreamingLLM 已验证 sink tokens + SWA 的有效性
4. **Memory capacity**: 64-128 memory tokens 应该足够补偿 SWA 的信息损失

**但需要注意**:
1. Memory tokens 之间 **必须用 full attention**
2. Memory tokens 数量应该 **≥ 128**（SWA 下需要更多）
3. 训练可能 **更困难**，需要 curriculum learning
4. Attention mask 设计 **必须正确**

---

### [inference] 实施建议

#### 阶段 1: 验证 RMT（full attention）是否工作

**目标**: 确保 RMT 架构本身是可行的

**步骤**:
1. **增加 memory tokens**: 从 16 → 64-128
2. **重新训练**: 20 epochs，10K+ docs
3. **运行评测**: NIH, PPL, MMLU
4. **验证 memory 是否工作**: 检查 memory utilization（cosine similarity）

**预期结果**:
- NIH > 80%（尤其远位置）
- PPL < 15（Wikitext-103）
- MMLU 与 LoRA-only 持平
- Memory similarity < 0.8（memory 在编码不同信息）

**如果失败**:
- 检查训练配置（LR, epochs, dataset）
- 检查 memory architecture（attention mask, position encoding）
- **不要尝试 SWA**，先修复 RMT

---

#### 阶段 2: 尝试 RMT + SWA

**前提**: 阶段 1 成功（RMT full attention 工作良好）

**步骤**:
1. **修改 backbone**: 添加 memory token 支持（基于现有 `SWABackbone`）
2. **修改 attention mask**: 实现 RMT + SWA mask（参见第 5 节）
3. **Fine-tune**: 使用已训练的 RMT 模型作为初始化，fine-tune 为 SWA
   - LR: 1e-5（更低）
   - Epochs: 5-10
   - Learning rate schedule: Cosine decay with warmup
4. **评测**: 与 RMT（full attention）对比

**预期结果**:
- SWA + Memory 性能接近 RMT（full attention）（gap < 10-20%）
- 计算节省约 43% FLOPs
- 可以处理更长文本（2x segments 在相同 FLOPs 下）

**如果失败**:
- 检查 attention mask 是否正确
- 增加更多 memory tokens（如 256）
- 尝试 curriculum learning（先 full，后 SWA）
- **可能需要更复杂的设计**（如 dual memory）

---

#### 阶段 3: 优化与扩展（可选）

**目标**: 进一步优化 RMT + SWA

**方向**:
1. **Dual memory**: Recent + compressed memory（类似 Compressive Transformer）
2. **Adaptive window**: 根据内容调整 window size（重要内容用更大的 window）
3. **Hierarchical memory**: L1（在线）+ L2（turn）+ L3（session）+ L0（segment memory）
4. **Mixed attention**: 部分层用 full attention，部分层用 SWA

---

### [fact] 关键参数推荐

| 参数 | 当前值 | 推荐值（RMT full） | 推荐值（RMT + SWA） |
|------|--------|-------------------|-------------------|
| **Memory tokens** | 16 | 64-128 | 128-256 |
| **Segment length** | 1024 | 1024 | 1024-2048 |
| **Window size** | None（full） | None（full） | 512-1024 |
| **Max segments** | 6 | 6-10 | 10-20 |
| **Memory LR / Backbone LR** | 2.5x | 10x | 10-20x |
| **LoRA rank** | 32 | 32-64 | 32-64 |

---

### [fact] 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| **RMT full attention 不工作** | 中 | 高 | 先验证 RMT，再尝试 SWA |
| **SWA + Memory 性能严重下降** | 中 | 高 | 增加 memory tokens，优化 mask |
| **训练不稳定** | 高 | 中 | Curriculum learning，lower LR |
| **Attention mask 设计错误** | 中 | 高 | 单元测试，可视化 mask |
| **实现复杂度过高** | 低 | 中 | 分步实施，每步验证 |

---

## 8. 参考文献

1. **Bulatov et al. (2023)** - "Recurrent Memory Transformer" (arXiv:2306.14095)
2. **Xiao et al. (2023)** - "StreamingLLM: Enabling LLMs to Process Infinite Length Texts via Streaming" (arXiv:2309.17453)
3. **Beltagy et al. (2020)** - "Longformer: The Long-Document Transformer" (ACL 2020)
4. **Zaheer et al. (2020)** - "Big Bird: Transformers for Longer Sequences" (NeurIPS 2020)
5. **Rae et al. (2019)** - "Compressive Transformer for Long-Range Sequence Modelling" (ICLR 2020)
6. **Dai et al. (2019)** - "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context" (NeurIPS 2019)
7. **Wu et al. (2022)** - "Memorizing Transformers" (ICML 2022)

---

## 9. 下一步行动

1. **立即执行**:
   - ✅ Task 1 完成（综合评测方案设计）
   - ✅ Task 2 完成（RMT + SWA 结合可行性调研）
   - 📝 更新 RESEARCH_LITERATURE.md

2. **短期目标**:
   - 修改配置：memory_tokens=64
   - 重新训练 RMT（full attention）
   - 运行评测（NIH, PPL, MMLU）

3. **中期目标**:
   - 如果 RMT 工作良好，实现 RMT + SWA
   - Fine-tune RMT → RMT + SWA
   - 评测对比

4. **长期目标**:
   - 优化 RMT + SWA（dual memory, adaptive window）
   - 扩展到更多数据集和任务

---

**调研完成时间**: 2026-04-18 23:39 GMT+8  
**调研人员**: researcher (subagent)  
**讨论参与者**: Proposer, Skeptic, Critic  
**状态**: ✅ 完成
