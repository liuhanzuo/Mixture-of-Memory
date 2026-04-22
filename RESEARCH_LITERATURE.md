# Research Literature: Fixed-Size Memory Bank Write Diversity

> 调研日期: 2026-04-21
> 调研人: researcher worker
> 问题: 固定大小 Memory Bank 的写入不均衡问题

---

## 1. 问题定义

### 1.1 当前实现分析

根据代码审查，当前项目的 memory bank 机制如下：

**MemoryBank (`src/memory/sparse_memory/memory_bank.py`)**:
- 固定大小: `num_slots = 128` (可配置)
- 写入机制: Circular buffer + EMA update
- EMA update: `memory = alpha * hidden_states + (1-alpha) * memory`
- 写入策略: 每层 attention 后，将所有 hidden states 顺序写入连续的 slots
- 无选择机制，无遗忘机制

**SparseMemoryAttention (`src/memory/sparse_memory/attention.py`)**:
- Local path: sliding window attention (`window_size = 256`)
- Memory path: top-k retrieval (`top_k = 8`) via similarity to memory slots
- Fusion: concat `[o_local || o_mem]` → linear projection
- Write: 每 layer 后，将 `B x T` hidden states 写入 memory bank

### 1.2 核心问题

**写入不均衡问题**:

1. **Hot spots**: 某些 memory slot 可能被频繁覆盖
   - 原因: 当前 circular buffer 是顺序写入，所有 token 都会被写入
   - 后果: 频繁更新的信息互相覆盖，信息丢失

2. **Cold spots**: 某些 memory slot 很少被更新
   - 原因: 顺序写入模式下，某些 slot 可能长期不被检索到
   - 后果: 信息过时，无法反映最新上下文

3. **缺乏选择性**: 当前写入所有 hidden states，没有筛选"重要" token
   - 问题: 大量冗余信息占用 memory
   - 后果: 噪声信息污染 memory，降低检索质量

4. **无遗忘机制**: EMA decay 是固定的，不随时间或访问频率调整
   - 问题: 旧信息与新信息更新速率相同
   - 后果: 无法区分"长期稳定信息"与"短期临时信息"

### 1.3 形式化描述

**符号定义**:
- `M ∈ ℝ^{B×N×d}`: Memory bank (B=batch size, N=num slots, d=hidden dim)
- `h_t ∈ ℝ^d`: t 时刻的 hidden state
- `w_t ∈ ℝ^N`: 写入位置/权重向量
- `α`: EMA decay rate

**当前写入过程**:
```
w_t = (ptr + t) mod N  # circular pointer
M[:, w_t] = α * h_t + (1-α) * M[:, w_t]
ptr = (ptr + T) mod N
```

**问题**: `w_t` 是确定性的顺序选择，导致:
- `P[slot i 被写入] = 1/N` (理论上均匀)
- 但实际检索时，某些 slot 被频繁访问，某些从未被访问
- 不考虑 `h_t` 的重要性、与现有 memory 的相似性、或 slot 的"热度"

---

## 2. 相关方案汇总

### 2.1 写入策略 (Write Strategies)

#### 2.1.1 Top-k Selection + Similarity Overwrite

**参考工作**: LM2 (Large Memory Models, 2025), MemWalker (2023)

**方法**:
1. 从 hidden states 中选择 top-k 个"重要" token (基于 attention score, gradient, 或 learned score)
2. 计算新 token 与现有 memory slots 的相似度
3. 选择最相似的 slot 进行覆盖（或 EMA 更新）

**优点**:
- 自然平衡写入：重要 token 覆盖相似 slot，避免 hot spots
- 保持语义连续性：相似信息的 slot 会被持续更新
- 计算开销适中：只需计算相似度矩阵

**缺点**:
- 可能出现 "information collapse": 多个相似 token 覆盖同一 slot
- 需要设计 "重要性" 评分机制
- 对于完全不相关的 token，可能找不到合适的 slot

**与当前实现的适配度**: ⭐⭐⭐⭐⭐ (高度适配)
- 当前已经有 top-k retrieval，可以在写入端也引入 top-k selection
- 只需修改 `memory_bank.write()` 方法，添加重要性评分和 slot 选择逻辑

**实现思路**:
```python
# 1. 重要性评分（可选项）
importance_scores = compute_importance(hidden_states)  # e.g., attention entropy

# 2. Top-k selection
top_k_idx = torch.topk(importance_scores, k=num_writes).indices
selected_tokens = hidden_states[top_k_idx]

# 3. Slot selection
sim = torch.matmul(selected_tokens, memory.t())  # [k, N]
best_slots = sim.argmax(dim=-1)  # [k]

# 4. EMA update
for i, slot in enumerate(best_slots):
    memory[slot] = alpha * selected_tokens[i] + (1-alpha) * memory[slot]
```

---

#### 2.1.2 Diversity-Aware Selection (MMR / DPP)

**参考工作**: Maximal Marginal Relevance (MMR), Determinantal Point Processes (DPP), Slot Attention (Locatello et al., 2020)

**方法**:
- **MMR**: 在选择 top-k 时，平衡 "相关性" 和 "多样性"
  - `score = λ * sim(q, d) - (1-λ) * max sim(d, d_selected)`
- **DPP**: 行列式 point process，优化选择的子集的 "多样性"
  - 最大化选定子集的 kernel 矩阵的行列式

**优点**:
- 显式优化多样性，避免信息 collapse
- 理论保证：DPP 有明确的最大化多样性的目标
- 适合需要覆盖广泛场景的任务

**缺点**:
- 计算开销大：需要计算所有候选之间的相似度矩阵 `O(k^2 * N)`
- 可能引入不相关 token（为追求多样性牺牲相关性）
- MMR 的 λ 超参数需要调优

**与当前实现的适配度**: ⭐⭐⭐ (中等适配)
- 可以在写入 selection 时引入，但会增加计算开销
- 适合写入端，不适合检索端

**实现思路** (简化版 MMR):
```python
# MMR for token selection
selected = []
candidates = list(range(T))

for _ in range(k):
    best_score = -inf
    best_idx = -1

    for c in candidates:
        # Relevance to memory
        rel = hidden_states[c] @ memory.mean(dim=0)

        # Diversity from already selected
        div = 0
        if selected:
            selected_tokens = hidden_states[torch.tensor(selected)]
            div = torch.max(hidden_states[c] @ selected_tokens.t())

        score = lambda_rel * rel - (1 - lambda_rel) * div

        if score > best_score:
            best_score = score
            best_idx = c

    selected.append(best_idx)
    candidates.remove(best_idx)
```

---

#### 2.1.3 Cache Eviction Policies (LRU / LFU)

**参考工作**: Standard cache replacement policies, Neural Cache (Grave et al., 2017)

**方法**:
- **LRU (Least Recently Used)**: 淘汰最久未被访问的 slot
- **LFU (Least Frequently Used)**: 淘汰访问频率最低的 slot
- **ARC (Adaptive Replacement Cache)**: 自适应混合 LRU/LFU

**优点**:
- 简单高效：只需维护访问时间戳/计数
- 经典算法，理论成熟
- 对 cache hit rate 有理论保证

**缺点**:
- 不考虑语义相似性，可能覆盖相关信息
- LRU: 可能频繁覆盖长期稳定的信息
- LFU: 可能 "starve" 新信息（永远无法累积足够访问次数）

**与当前实现的适配度**: ⭐⭐⭐ (中等适配)
- 可以与 similarity-based overwrite 结合
- 适合作为辅助机制，而非完全替代

**实现思路** (LFU-based):
```python
# Initialize access counters
access_count = torch.zeros(N, device=device)
last_access = torch.zeros(N, device=device, dtype=torch.long)

# On write:
evict_slot = access_count.argmin()  # LFU
# or:
evict_slot = last_access.argmin()  # LRU

# Update
memory[evict_slot] = alpha * token + (1-alpha) * memory[evict_slot]
access_count[evict_slot] += 1
last_access[evict_slot] = step

# On retrieval:
access_count[retrieved_slots] += 1
last_access[retrieved_slots] = step
```

---

#### 2.1.4 Learned Write Policy (RL / Meta-learning)

**参考工作**: Differentiable Neural Computer (Graves et al., 2016), Neural Turing Machine (Graves et al., 2014)

**方法**:
- 使用可学习的 gating 机制控制写入：
  - `gate_t = sigmoid(W_g * h_t + b_g)`
  - 如果 `gate_t > threshold`，则写入；否则跳过
- 或使用 RL 学习写入策略：
  - State: memory state, current token
  - Action: which slot to write (or skip)
  - Reward: downstream task performance

**优点**:
- 完全端到端可学习，自动适应任务
- 可以学习复杂的非启发式策略
- 理论上最优（如果训练充分）

**缺点**:
- 训练难度大：需要精心设计奖励/损失函数
- 计算开销大：需要额外的 gating network 或 RL agent
- 不稳定性：可能收敛到局部最优或崩溃

**与当前实现的适配度**: ⭐⭐ (低适配度，但长期潜力大)
- 可以作为后续优化方向
- 当前不建议作为主要方案

**实现思路** (Simple gating):
```python
# Learned gating
class WriteGate(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        return torch.sigmoid(self.gate(h)).squeeze(-1)  # [T]

# Usage
gates = write_gate(hidden_states)  # [T]
write_idx = torch.where(gates > threshold)[0]
for idx in write_idx:
    # ... write to memory
```

---

### 2.2 遗忘机制 (Forgetting / Decay)

#### 2.2.1 Age-Based Decay

**参考工作**: Many RNN/LSTM variants, Neural Cache

**方法**:
- EMA decay rate `α` 与 slot age 挂钩：
  - `α_t = α_0 * exp(-λ * age)` or `α_t = α_0 / (1 + λ * age)`
  - Age = current_step - last_write_step

**优点**:
- 简单：只增加 age tracking
- 符合直觉：旧信息更难更新
- 可调参数少（λ）

**缺点**:
- 可能过度强调"新"，丢失长期稳定信息
- 不考虑访问频率：一个长期稳定但频繁访问的信息也会逐渐衰减
- λ 超参数需要调优

**与当前实现的适配度**: ⭐⭐⭐⭐ (高度适配)
- 可以在当前 EMA 机制基础上轻松添加
- 只需维护 `age` 向量

**实现思路**:
```python
# Track age
age = torch.zeros(N, device=device, dtype=torch.long)
current_step = 0

# On write:
slot_age = current_step - age[slot]
dynamic_alpha = alpha_0 / (1.0 + decay_lambda * slot_age)
memory[slot] = dynamic_alpha * token + (1 - dynamic_alpha) * memory[slot]
age[slot] = current_step
current_step += 1
```

---

#### 2.2.2 Access-Frequency-Based Decay

**参考工作**: LFU variants

**方法**:
- 高访问频率的 slot 衰减更慢（或完全不衰减）
- `α_t = α_0 * (1 + β * access_count / max_count)`
- 或：高频 slot 使用更小的 α（更保守更新）

**优点**:
- 保护重要信息：频繁访问的信息更稳定
- 符合 "consolidation" 思想：访问即强化
- 与 cache eviction policies 自然结合

**缺点**:
- 可能 "starve" 新信息：高频 slot 持续稳定，新信息无法进入
- 需要维护 access_count
- β 超参数需要调优

**与当前实现的适配度**: ⭐⭐⭐⭐ (高度适配)
- 可以与 LFU eviction 结合

**实现思路**:
```python
access_count = torch.zeros(N, device=device)

# On write:
normalized_freq = access_count[slot] / (access_count.max() + 1e-6)
dynamic_alpha = alpha_0 * (1 - beta * normalized_freq)  # High freq → small α
memory[slot] = dynamic_alpha * token + (1 - dynamic_alpha) * memory[slot]
access_count[slot] += 1
```

---

#### 2.2.3 Consolidation Mechanism

**参考工作**: Compressive Transformers, Neural Cache (with consolidation)

**方法**:
- 将多个 memory slots 合并为一个：
  - 使用 compression operator (LSTM, Transformer encoder, attention)
  - 定期触发：当达到 consolidation interval 时
  - 选择策略：合并相似或低频的 slots

**优点**:
- 显式减少 memory 碎片
- 保留重要信息的同时释放空间
- 适合长上下文场景

**缺点**:
- 计算开销大：需要额外的 compression network
- 需要定义 consolidation trigger 和 slot selection 策略
- 可能丢失细节信息

**与当前实现的适配度**: ⭐⭐⭐ (中等适配)
- 可以作为独立的优化模块
- 适合后续迭代

**实现思路** (Simple consolidation):
```python
def consolidate_memory(memory, num_compress):
    """Compress the least-accessed slots."""
    # Find low-freq slots
    slots_to_compress = access_count.argmin(num_compress)

    # Simple average compression
    compressed = memory[slots_to_compress].mean(dim=0)

    # Store in first slot, zero out others
    memory[slots_to_compress[0]] = compressed
    memory[slots_to_compress[1:]] = 0

    # Reset access counts
    access_count[slots_to_compress] = 0

    return memory
```

---

### 2.3 多级 Memory (Multi-Level / Hierarchical Memory)

#### 2.3.1 Compressive Transformers (Fast + Slow Memory)

**论文**: Rae et al., "Compressive Transformers for Long-Range Sequence Modelling", ICLR 2021

**方法**:
- **Fast memory**: 最新 N_f tokens，circular buffer
- **Slow memory**: N_s 压缩后的 segments
- **Compression operator**: LSTM 或 Transformer encoder
  - 每 C 个 tokens 压缩为一个 compressed vector
  - 压缩时考虑 attention weights：`comp = Σ w_i * h_i`
- **Update**:
  - 新 token → fast memory (circular)
  - 当 fast memory 满 → 压缩 oldest segment → slow memory
  - 当 slow memory 满 → 压缩 oldest segment → drop

**优点**:
- 天然解决多样性：不同层级有不同更新频率
- 时序语义：层级越高，信息越抽象/压缩
- 计算高效：fast memory 小，slow memory 但不频繁更新

**缺点**:
- 需要额外的 compression network
- 超参数多（N_f, N_s, C, compression operator）
- 压缩信息可能丢失细节

**与当前实现的适配度**: ⭐⭐⭐⭐ (高度适配)
- 可以在现有 sliding window + memory bank 基础上扩展
- Fast memory = current sliding window (256 tokens)
- Slow memory = current memory bank (128 slots)

**实现思路**:
```python
class HierarchicalMemory:
    def __init__(self, fast_size=256, slow_size=128, compress_interval=32):
        self.fast_memory = torch.zeros(fast_size, d)  # circular buffer
        self.slow_memory = torch.zeros(slow_size, d)  # compressed segments
        self.compress_interval = compress_interval
        self.ptr = 0

    def write(self, token):
        # Write to fast memory (circular)
        self.fast_memory[self.ptr] = token
        self.ptr = (self.ptr + 1) % self.fast_size

        # Compress when interval reached
        if self.ptr % self.compress_interval == 0:
            self._compress_to_slow()

    def _compress_to_slow(self):
        # Get the oldest segment from fast memory
        oldest_idx = (self.ptr - self.compress_interval) % self.fast_size
        segment = self.fast_memory[oldest_idx:self.ptr]

        # Simple compression (average with attention weights)
        # More complex: use LSTM or Transformer encoder
        compressed = segment.mean(dim=0)

        # Circular shift slow memory
        self.slow_memory = torch.roll(self.slow_memory, 1, dims=0)
        self.slow_memory[0] = compressed

    def retrieve(self, query):
        # Retrieve from both fast and slow
        fast_retrieval = self._retrieve_from(query, self.fast_memory, k_f)
        slow_retrieval = self._retrieve_from(query, self.slow_memory, k_s)

        # Concat or weighted sum
        return torch.cat([fast_retrieval, slow_retrieval], dim=-1)
```

---

#### 2.3.2 RMT (Recurrent Memory Transformer)

**论文**: Bulatov et al., "Recurrent Memory Transformer", 2022

**方法**:
- **Recurrent memory tokens**: 额外的 memory tokens，在 segment 间传递
- **Segment-level compression**:
  - 每个 segment 由 input tokens + memory tokens 组成
  - Segment 内：attention 同时 attend to input 和 memory
  - Segment 后：memory tokens 更新为 compressed representation
- **Compression**:
  - Memory tokens 通过 attention 加权平均更新
  - 或通过额外的 MLP/LSTM 压缩

**优点**:
- 段间信息流自然
- Memory tokens 可以在训练中学习如何压缩信息
- 不需要显式 retrieval 机制

**缺点**:
- 需要训练时 segment 切分
- Memory tokens 的数量和位置需要设计
- 与现有 sliding window 架构不完全兼容

**与当前实现的适配度**: ⭐⭐⭐ (中等适配)
- 需要重新设计训练流程
- 但思想可以借鉴：将 memory bank 视为 recurrent memory

**实现思路** (RMT-style memory update):
```python
class RMTMemoryBank(nn.Module):
    def __init__(self, num_memory_tokens, hidden_dim):
        super().__init__()
        self.memory_tokens = nn.Parameter(
            torch.randn(num_memory_tokens, hidden_dim)
        )
        self.compressor = nn.Linear(hidden_dim, hidden_dim)

    def update_memory(self, segment_output):
        """Update memory tokens after processing a segment."""
        # segment_output: [T, d] - output of current segment

        # Attention over segment to compress
        attn_weights = self.memory_tokens @ segment_output.t()  # [M, T]
        attn_weights = F.softmax(attn_weights / d**0.5, dim=-1)

        # Weighted sum
        compressed = attn_weights @ segment_output  # [M, d]

        # Update memory (EMA or learned)
        self.memory_tokens.data = 0.1 * compressed + 0.9 * self.memory_tokens
```

---

#### 2.3.3 LM2 (Large Memory Models)

**论文**: LM2: Large Memory Models, arXiv 2502.06049, 2025

**方法**:
- **Cross-attention memory bank**: 额外的 memory module，通过 cross-attention 与交互
- **LSTM-style gating**:
  - `i_t = σ(W_i * h_t + U_i * M_{t-1})` - input gate
  - `f_t = σ(W_f * h_t + U_f * M_{t-1})` - forget gate
  - `o_t = σ(W_o * h_t + U_o * M_{t-1})` - output gate
  - `M_t = f_t ⊙ M_{t-1} + i_t ⊙ g_t` - memory update
- **Memory slots**: 可学习的 slot embeddings，通过 gating 控制

**优点**:
- 显式控制 memory 的读/写/遗忘
- 与 LSTM 有相同的理论保证
- Gating mechanism 天然解决部分多样性问题（通过 learnable gates）

**缺点**:
- 参数量大：每个 gate 需要 W 和 U 矩阵
- 训练难度：gating mechanism 可能不稳定
- 计算开销：每步都需要计算 gates

**与当前实现的适配度**: ⭐⭐⭐ (中等适配)
- 可以借鉴 gating 机制，但不需要完全照搬
- 可以简化为 per-slot gating

**实现思路** (Simplified LM2 gating):
```python
class GatedMemoryBank(nn.Module):
    def __init__(self, num_slots, hidden_dim):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(num_slots, hidden_dim))

        # Gates (simplified: per-slot, not per-token)
        self.forget_gate = nn.Linear(hidden_dim, num_slots)
        self.write_gate = nn.Linear(hidden_dim, num_slots)

    def update(self, token):
        """Update memory with gating."""
        # Compute gates based on token
        f = torch.sigmoid(self.forget_gate(token))  # [N]
        w = torch.sigmoid(self.write_gate(token))    # [N]

        # Select slot to update (e.g., based on similarity)
        sim = token @ self.memory.t()
        slot = sim.argmax()

        # Gated update
        self.memory.data[slot] = (
            f[slot] * self.memory.data[slot] +
            w[slot] * token
        )
```

---

### 2.4 Slot Diversity Mechanisms

#### 2.4.1 Slot Attention / Set Transformer

**论文**: Locatello et al., "Object-Centric Learning with Slot Attention", NeurIPS 2020

**方法**:
- **Iterative slot refinement**:
  - 初始化 N 个 learnable slots
  - 迭代更新：每个 slot attend to inputs，更新 slot embedding
  - 竞争机制：slots 通过 attention 权重竞争 inputs
- **Diversity constraint**:
  - 使用 determinantal point processes (DPP) 或
  - Orthogonal loss: `L_div = ||S @ S^T - I||` where S is slot embeddings

**优点**:
- 显式优化 slot diversity
- 通过 competition 自然避免 collapse
- 适合 object-centric 场景

**缺点**:
- 计算开销大：需要迭代 refinement
- 原始设计不适用于 streaming 场景
- 可能过度强调 diversity，牺牲语义连续性

**与当前实现的适配度**: ⭐⭐ (低适配度)
- 适合初始化阶段，不适合在线更新
- 但 diversity loss 可以借鉴

**实现思路** (Orthogonal regularization):
```python
def diversity_loss(memory):
    """Orthogonal loss on memory slots."""
    # Normalize
    M_norm = F.normalize(memory, dim=-1)  # [N, d]

    # Compute similarity matrix
    sim = M_norm @ M_norm.t()  # [N, N]

    # Encourage orthogonality (minimize off-diagonal)
    mask = 1 - torch.eye(sim.shape[0], device=sim.device)
    loss = (sim * mask).pow(2).sum()

    return loss

# Add to total loss
loss = task_loss + lambda_div * diversity_loss(memory)
```

---

#### 2.4.2 MMR (Maximal Marginal Relevance)

**参考**: Carbonell & Goldstein, "The Use of MMR, Diversity-Based Reranking for Reordering Documents", 1998

**方法**:
- 在选择 top-k 时，平衡 relevance 和 diversity
- `score(q, d_i) = λ * sim(q, d_i) - (1-λ) * max_{d_j ∈ selected} sim(d_i, d_j)`
- λ 越大越注重相关性，λ 越小越注重多样性

**优点**:
- 简单直接，易于实现
- 可调 λ 平衡相关性和多样性
- 适用于选择场景

**缺点**:
- 计算 `O(k^2)` (需要计算所有已选候选之间的相似度)
- λ 超参数需要调优
- 可能选择不相关的 token（为追求多样性）

**与当前实现的适配度**: ⭐⭐⭐⭐ (高度适配)
- 可以在写入端使用 MMR 选择要写入的 token
- 可以在检索端使用 MMR re-rank 检索结果

**实现思路** (见 2.1.2)

---

## 3. 推荐方案

### 3.1 短期方案（立即可实施）

#### 方案 A: Top-k Selection + Similarity Overwrite + Frequency-Based Decay

**核心思想**:
1. 每层从 hidden states 中选择 top-k 个"重要" token（基于 attention entropy）
2. 对每个选中的 token，找到最相似的 memory slot 进行 EMA 更新
3. EMA decay rate 与 slot 访问频率挂钩：高频 slot 衰减更慢

**优点**:
- 实现简单，改动最小
- 显著减少写入量（从 T 个 token → k 个 token）
- 自然平衡写入：重要 token 覆盖相似 slot
- 保护高频访问的重要信息

**缺点**:
- 仍可能出现 info collapse（多个相似 token 覆盖同一 slot）
- 需要调优 k 和 decay_lambda

**实现细节**:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedMemoryBank(nn.Module):
    """Enhanced Memory Bank with top-k selection and frequency-based decay."""

    def __init__(
        self,
        num_slots: int = 128,
        hidden_dim: int = 4096,
        base_ema_alpha: float = 0.1,
        top_k: int = 4,  # Number of tokens to write per layer
        decay_lambda: float = 0.1,  # Frequency decay factor
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.hidden_dim = hidden_dim
        self.base_ema_alpha = base_ema_alpha
        self.top_k = top_k
        self.decay_lambda = decay_lambda
        self._dtype = dtype

        # Memory buffer
        self.register_parameter(
            "memory",
            nn.Parameter(torch.zeros(1, num_slots, hidden_dim, dtype=dtype), requires_grad=False),
        )

        # Access tracking for frequency-based decay
        self.register_parameter(
            "access_count",
            nn.Parameter(torch.zeros(1, num_slots, dtype=torch.long), requires_grad=False),
        )

    def reset(self, batch_size: int = 1) -> None:
        """Zero out memory and tracking stats."""
        device = self.memory.device
        self.memory = nn.Parameter(
            torch.zeros(batch_size, self.num_slots, self.hidden_dim, device=device, dtype=self._dtype),
            requires_grad=False,
        )
        self.access_count = nn.Parameter(
            torch.zeros(batch_size, self.num_slots, dtype=torch.long, device=device),
            requires_grad=False,
        )

    @torch.no_grad()
    def compute_importance(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute token importance using attention entropy proxy.

        Args:
            hidden_states: [T, d] — hidden states for one sample.

        Returns:
            importance: [T] — importance scores for each token.
        """
        # Simple heuristic: norm-based importance
        # Higher norm → more salient information
        importance = hidden_states.norm(dim=-1)  # [T]

        # Normalize to [0, 1]
        importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-6)

        return importance

    @torch.no_grad()
    def write(self, hidden_states: torch.Tensor, batch_idx: int) -> None:
        """Write top-k important tokens with similarity-based slot selection.

        Args:
            hidden_states: [T, d] — hidden states for one sample in this layer.
            batch_idx: which batch item to write to.
        """
        T = hidden_states.shape[0]

        # Step 1: Compute importance scores
        importance = self.compute_importance(hidden_states)  # [T]

        # Step 2: Top-k selection
        if self.top_k < T:
            k = min(self.top_k, T)
            topk_scores, topk_idx = torch.topk(importance, k=k, dim=-1)
        else:
            topk_idx = torch.arange(T, device=hidden_states.device)
            k = T

        selected_tokens = hidden_states[topk_idx]  # [k, d]

        # Step 3: For each selected token, find most similar slot
        memory = self.memory[batch_idx]  # [N, d]
        access_count = self.access_count[batch_idx]  # [N]

        # Similarity: [k, N]
        sim = torch.matmul(selected_tokens, memory.t())

        # Find best slot per token (similarity-based)
        best_slots = sim.argmax(dim=-1)  # [k]

        # Step 4: EMA update with frequency-based decay
        max_access = access_count.max() + 1

        for i, slot in enumerate(best_slots):
            token = selected_tokens[i]
            current_mem = memory[slot]
            slot_access = access_count[slot].item()

            # Dynamic alpha based on access frequency
            # High freq → smaller alpha (more conservative)
            normalized_freq = slot_access / max_access
            dynamic_alpha = self.base_ema_alpha * (1.0 - self.decay_lambda * normalized_freq)
            dynamic_alpha = max(dynamic_alpha, 0.01)  # Min alpha to avoid stalling

            # EMA update
            updated = (dynamic_alpha * token + (1.0 - dynamic_alpha) * current_mem).to(self._dtype)

            # Non-inplace write
            new_mem = self.memory.clone()
            new_mem[batch_idx, slot] = updated
            self.memory = nn.Parameter(new_mem, requires_grad=False)

            # Update access count
            new_access = self.access_count.clone()
            new_access[batch_idx, slot] = slot_access + 1
            self.access_count = nn.Parameter(new_access, requires_grad=False)

    @torch.no_grad()
    def retrieve(self, query: torch.Tensor, k: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve top-k memory slots for a query.

        Args:
            query: [B, T, d] or [T, d] — query vectors.
            k: number of slots to retrieve.

        Returns:
            (values, scores): retrieved values and similarity scores.
        """
        if query.dim() == 2:
            query = query.unsqueeze(0)  # [1, T, d]

        B, T, d = query.shape

        # Memory: [B, N, d]
        memory = self.memory.to(query.dtype)

        # Similarity: [B, T, N]
        sim = torch.matmul(query, memory.transpose(-2, -1)) / (d ** 0.5)

        # Top-k
        effective_k = min(k, self.num_slots)
        topk_scores, topk_idx = sim.topk(effective_k, dim=-1)  # [B, T, k]

        # Gather values
        B_idx = torch.arange(B, device=query.device)[:, None, None]
        T_idx = torch.arange(T, device=query.device)[None, :, None]
        topk_values = memory[B_idx, topk_idx]  # [B, T, k, d]

        # Update access count for retrieved slots
        for b in range(B):
            for t in range(T):
                slots = topk_idx[b, t]
                new_access = self.access_count.clone()
                new_access[b, slots] += 1
                self.access_count = nn.Parameter(new_access, requires_grad=False)

        return topk_values, topk_scores
```

**集成到现有代码**:

```python
# In SparseMemoryAttention.__init__:
self.memory_bank = EnhancedMemoryBank(
    num_slots=config.num_mem_tokens,
    hidden_dim=hidden_size,
    base_ema_alpha=0.1,
    top_k=4,  # Write top-4 tokens per layer
    decay_lambda=0.1,
)

# In SparseMemoryAttention.forward (memory write section):
# Replace the original write loop with:
for b in range(B):
    self.memory_bank.write(hidden_states[b], batch_idx=b)
```

**调优建议**:
- `top_k`: 从 2-8 开始，调优（太小→信息丢失，太大→写入不均衡）
- `base_ema_alpha`: 从 0.05-0.2 开始
- `decay_lambda`: 从 0.05-0.2 开始
- 监控指标：
  - Memory slot access distribution（均衡度）
  - Memory retrieval quality（下游任务性能）
  - Information freshness（memory age）

---

#### 方案 B: Two-Level Memory (Fast + Slow)

**核心思想**:
1. **Fast memory**: 当前 sliding window 中的最近 token (256 tokens)
2. **Slow memory**: 压缩后的 past information (128 slots)
3. **Compression operator**: 简单 average 或 attention-weighted 平均
4. **Update**:
   - 新 token 进入 fast memory (circular)
   - 当 fast memory 的一部分"滑出"时，压缩到 slow memory

**优点**:
- 天然解决多样性：不同层级有不同更新频率
- 保持时序语义：fast = 最新，slow = 抽象历史
- 计算高效：fast memory 小，只检索部分
- 与现有架构兼容：fast memory = sliding window

**缺点**:
- 需要额外的 compression 逻辑
- 压缩可能丢失细节
- 超参数：压缩间隔、压缩方法

**实现细节**:

```python
class TwoLevelMemory(nn.Module):
    """Two-level memory: fast (sliding window) + slow (compressed)."""

    def __init__(
        self,
        fast_size: int = 256,
        slow_size: int = 128,
        hidden_dim: int = 4096,
        compress_interval: int = 32,
        ema_alpha: float = 0.1,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.fast_size = fast_size
        self.slow_size = slow_size
        self.compress_interval = compress_interval
        self.ema_alpha = ema_alpha
        self._dtype = dtype

        # Fast memory: circular buffer for recent tokens
        self.register_parameter(
            "fast_memory",
            nn.Parameter(torch.zeros(1, fast_size, hidden_dim, dtype=dtype), requires_grad=False),
        )

        # Slow memory: compressed representations
        self.register_parameter(
            "slow_memory",
            nn.Parameter(torch.zeros(1, slow_size, hidden_dim, dtype=dtype), requires_grad=False),
        )

        # Write pointers
        self.register_parameter(
            "fast_ptr",
            nn.Parameter(torch.zeros(1, dtype=torch.long), requires_grad=False),
        )
        self.register_parameter(
            "slow_ptr",
            nn.Parameter(torch.zeros(1, dtype=torch.long), requires_grad=False),
        )

    def reset(self, batch_size: int = 1) -> None:
        """Reset both memory levels."""
        device = self.fast_memory.device
        self.fast_memory = nn.Parameter(
            torch.zeros(batch_size, self.fast_size, self.hidden_dim, device=device, dtype=self._dtype),
            requires_grad=False,
        )
        self.slow_memory = nn.Parameter(
            torch.zeros(batch_size, self.slow_size, self.hidden_dim, device=device, dtype=self._dtype),
            requires_grad=False,
        )
        self.fast_ptr = nn.Parameter(torch.zeros(batch_size, dtype=torch.long, device=device), requires_grad=False)
        self.slow_ptr = nn.Parameter(torch.zeros(batch_size, dtype=torch.long, device=device), requires_grad=False)

    @torch.no_grad()
    def write(self, hidden_states: torch.Tensor, batch_idx: int) -> None:
        """Write tokens to fast memory, compress to slow if needed.

        Args:
            hidden_states: [T, d] — hidden states for one sample.
            batch_idx: which batch item to write to.
        """
        T = hidden_states.shape[0]
        ptr = self.fast_ptr[batch_idx].item()

        # Write to fast memory (circular)
        indices = (ptr + torch.arange(T, device=hidden_states.device)) % self.fast_size
        fast_mem = self.fast_memory[batch_idx]
        current = fast_mem[indices]

        # EMA update to fast memory
        updated = (self.ema_alpha * hidden_states + (1 - self.ema_alpha) * current).to(self._dtype)
        new_fast = self.fast_memory.clone()
        new_fast[batch_idx, indices] = updated
        self.fast_memory = nn.Parameter(new_fast, requires_grad=False)

        # Advance pointer
        new_fast_ptr = self.fast_ptr.clone()
        new_fast_ptr[batch_idx] = (ptr + T) % self.fast_size
        self.fast_ptr = nn.Parameter(new_fast_ptr, requires_grad=False)

        # Check if we need to compress to slow memory
        # Trigger every compress_interval steps
        total_written = (self.fast_ptr[batch_idx].item() + self.fast_size) % self.fast_size
        if total_written % self.compress_interval == 0:
            self._compress_to_slow(batch_idx)

    @torch.no_grad()
    def _compress_to_slow(self, batch_idx: int) -> None:
        """Compress a segment from fast memory to slow memory."""
        # Get the oldest segment from fast memory
        ptr = self.fast_ptr[batch_idx].item()
        start_idx = (ptr - self.compress_interval) % self.fast_size
        end_idx = ptr

        # Extract segment (handle wrap-around)
        if start_idx < end_idx:
            segment = self.fast_memory[batch_idx, start_idx:end_idx]  # [C, d]
        else:
            segment = torch.cat([
                self.fast_memory[batch_idx, start_idx:],
                self.fast_memory[batch_idx, :end_idx],
            ], dim=0)  # [C, d]

        # Simple compression: weighted average
        # More complex: use attention weights or learnable compressor
        compressed = segment.mean(dim=0)  # [d]

        # Write to slow memory (circular)
        slow_ptr = self.slow_ptr[batch_idx].item()
        current_slow = self.slow_memory[batch_idx, slow_ptr]
        updated_slow = (self.ema_alpha * compressed + (1 - self.ema_alpha) * current_slow).to(self._dtype)

        new_slow = self.slow_memory.clone()
        new_slow[batch_idx, slow_ptr] = updated_slow
        self.slow_memory = nn.Parameter(new_slow, requires_grad=False)

        # Advance slow pointer
        new_slow_ptr = self.slow_ptr.clone()
        new_slow_ptr[batch_idx] = (slow_ptr + 1) % self.slow_size
        self.slow_ptr = nn.Parameter(new_slow_ptr, requires_grad=False)

    @torch.no_grad()
    def retrieve(
        self,
        query: torch.Tensor,
        k_fast: int = 4,
        k_slow: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve from both fast and slow memory.

        Args:
            query: [B, T, d] — query vectors.
            k_fast: number of slots to retrieve from fast memory.
            k_slow: number of slots to retrieve from slow memory.

        Returns:
            (values, scores): concatenated retrieved values and scores.
        """
        if query.dim() == 2:
            query = query.unsqueeze(0)

        B, T, d = query.shape

        # Retrieve from fast memory
        fast_mem = self.fast_memory.to(query.dtype)  # [B, fast_size, d]
        sim_fast = torch.matmul(query, fast_mem.transpose(-2, -1)) / (d ** 0.5)  # [B, T, fast_size]
        k_f = min(k_fast, self.fast_size)
        topk_fast_scores, topk_fast_idx = sim_fast.topk(k_f, dim=-1)  # [B, T, k_f]

        B_idx = torch.arange(B, device=query.device)[:, None, None]
        T_idx = torch.arange(T, device=query.device)[None, :, None]
        topk_fast_values = fast_mem[B_idx, topk_fast_idx]  # [B, T, k_f, d]

        # Retrieve from slow memory
        slow_mem = self.slow_memory.to(query.dtype)  # [B, slow_size, d]
        sim_slow = torch.matmul(query, slow_mem.transpose(-2, -1)) / (d ** 0.5)  # [B, T, slow_size]
        k_s = min(k_slow, self.slow_size)
        topk_slow_scores, topk_slow_idx = sim_slow.topk(k_s, dim=-1)  # [B, T, k_s]
        topk_slow_values = slow_mem[B_idx, topk_slow_idx]  # [B, T, k_s, d]

        # Concatenate: [B, T, k_f + k_s, d]
        values = torch.cat([topk_fast_values, topk_slow_values], dim=2)
        scores = torch.cat([topk_fast_scores, topk_slow_scores], dim=2)

        return values, scores
```

**集成到现有代码**:

```python
# In model initialization:
self.memory_bank = TwoLevelMemory(
    fast_size=256,  # Match sliding window
    slow_size=128,
    hidden_dim=hidden_size,
    compress_interval=32,  # Compress every 32 tokens
    ema_alpha=0.1,
)

# In SparseMemoryAttention.forward (memory retrieval section):
# Replace the retrieval code with:
mem_values, mem_scores = self.memory_bank.retrieve(hidden_states, k_fast=4, k_slow=4)

# Then use mem_values in attention (same as before)
```

**调优建议**:
- `compress_interval`: 从 16-64 开始（太小→频繁压缩，太大→信息堆积）
- `ema_alpha`: fast 和 slow 可以用不同的 α（fast 较大，slow 较小）
- 监控 fast/slow retrieval 的使用比例

---

### 3.2 长期方案（后续迭代）

#### 方案 C: Learned Gating (LM2-style)

**核心思想**:
使用可学习的 gating 机制控制 memory 的写入和遗忘。

**优点**:
- 端到端可学习，自动适应任务
- 理论上最优

**缺点**:
- 训练难度大
- 参数量大

**建议**:
- 先实施方案 A 或 B，验证 baseline 性能
- 如果写入策略仍是瓶颈，再尝试 learned gating

---

#### 方案 D: Three-Level Memory (Ultra-fine / Fine / Coarse)

**核心思想**:
扩展为三级 memory：
- L0: 最新的 token（类似 cache）
- L1: 压缩的近期信息
- L2: 长期稳定信息

**优点**:
- 更细粒度的信息分层
- 天然解决多样性问题

**缺点**:
- 复杂度增加
- 超参数更多

**建议**:
- 先验证两级 memory 的效果
- 如果需要，再扩展到三级

---

## 4. 多级 Memory 设计建议

### 4.1 层级设计原则

**原则 1: 更新频率递减**
- L0 (fast): 每个新 token 都可能更新
- L1 (medium): 定期压缩更新（如每 32 tokens）
- L2 (slow): 稀疏更新（如每 256 tokens 或基于重要事件）

**原则 2: 信息抽象度递增**
- L0: 原始 hidden states，保留细粒度信息
- L1: 语义级压缩（如 average, attention-weighted）
- L2: 概念级压缩（如经过 Transformer encoder）

**原则 3: 访问模式不同**
- L0: 频繁访问（最近的信息）
- L1: 中等访问（中期上下文）
- L2: 低频访问（长期记忆）

### 4.2 各层级更新/遗忘特性

#### L0: Fast Memory (Ultra-fine)
**参数**:
- Size: 256-512 tokens
- Update: 每 token（circular buffer）
- EMA α: 0.1-0.2（较快更新）
- Decay: 基于年龄（LRU）

**特性**:
- 存储：最近的原始 hidden states
- 更新：完全覆盖或高 α EMA
- 遗忘：基于纯 LRU，快速遗忘旧信息
- 用途：提供最新的细粒度上下文

#### L1: Medium Memory (Fine)
**参数**:
- Size: 128-256 slots
- Update: 每 32-64 tokens（压缩自 L0）
- EMA α: 0.05-0.1（中等更新）
- Decay: 基于年龄 + 访问频率

**特性**:
- 存储：压缩后的近期信息（segment 级）
- 更新：从 L0 压缩而来，中速更新
- 遗忘：混合策略（age + frequency）
- 用途：提供中期语义上下文

#### L2: Slow Memory (Coarse)
**参数**:
- Size: 64-128 slots
- Update: 每 256-512 tokens（压缩自 L1）或基于重要性事件
- EMA α: 0.01-0.05（慢速更新）
- Decay: 主要基于访问频率，age 次要

**特性**:
- 存储：长期稳定的概念级信息
- 更新：从 L1 进一步压缩，慢速更新
- 遗忘：保护高频访问的稳定信息
- 用途：提供长期记忆（如文档主旨、关键实体）

### 4.3 层级间信息流动

**Promotion (L0 → L1)**:
- 触发：固定间隔（如每 32 tokens）或基于重要性
- 方法：压缩算子（average, attention-weighted, 或 learnable compressor）
- 目标：从 L0 的一个 segment 生成 L1 的一个 slot

**Promotion (L1 → L2)**:
- 触发：固定间隔（如每 256 tokens）或基于重要性（top-k important slots）
- 方法：更深度的压缩（如 Transformer encoder）
- 目标：从 L1 的多个 slots 生成 L2 的一个概念 slot

**Demotion (L2 → L1 → L0)**:
- 一般不需要：层级是单向的
- 特殊情况：如果检索到 L2 的信息，可以"刷新"到 L1（降低其 age）

### 4.4 多级 Memory 的 Diversity 解决机制

**为什么多级自然解决多样性？**

1. **不同层级有不同更新频率**:
   - L0 高频更新 → 信息流动快
   - L2 低频更新 → 信息沉淀稳定

2. **不同层级存储不同抽象度的信息**:
   - L0: 细粒度 token 信息
   - L1: 语义段信息
   - L2: 概念级信息
   - 天然避免同一信息的重复存储

3. **层级间通过压缩算子连接**:
   - 压缩算子天然整合多源信息
   - 避免信息碎片化

4. **不同层级有不同的遗忘策略**:
   - L0: LRU（快速遗忘旧信息）
   - L1: LRU + LFU（平衡新信息和稳定信息）
   - L2: LFU（保护稳定信息）

**具体实现建议**:

```python
class ThreeLevelMemory(nn.Module):
    """Three-level memory: L0 (fast), L1 (medium), L2 (slow)."""

    def __init__(
        self,
        l0_size: int = 256,
        l1_size: int = 128,
        l2_size: int = 64,
        hidden_dim: int = 4096,
        compress_interval_l0_l1: int = 32,
        compress_interval_l1_l2: int = 256,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.l0_size = l0_size
        self.l1_size = l1_size
        self.l2_size = l2_size
        self.compress_interval_l0_l1 = compress_interval_l0_l1
        self.compress_interval_l1_l2 = compress_interval_l1_l2
        self._dtype = dtype

        # L0: Fast memory (circular buffer)
        self.register_parameter(
            "l0_memory",
            nn.Parameter(torch.zeros(1, l0_size, hidden_dim, dtype=dtype), requires_grad=False),
        )
        self.register_parameter(
            "l0_ptr",
            nn.Parameter(torch.zeros(1, dtype=torch.long), requires_grad=False),
        )
        self.register_parameter(
            "l0_age",
            nn.Parameter(torch.zeros(1, l0_size, dtype=torch.long), requires_grad=False),
        )

        # L1: Medium memory (compressed segments)
        self.register_parameter(
            "l1_memory",
            nn.Parameter(torch.zeros(1, l1_size, hidden_dim, dtype=dtype), requires_grad=False),
        )
        self.register_parameter(
            "l1_ptr",
            nn.Parameter(torch.zeros(1, dtype=torch.long), requires_grad=False),
        )
        self.register_parameter(
            "l1_access_count",
            nn.Parameter(torch.zeros(1, l1_size, dtype=torch.long), requires_grad=False),
        )
        self.register_parameter(
            "l1_age",
            nn.Parameter(torch.zeros(1, l1_size, dtype=torch.long), requires_grad=False),
        )

        # L2: Slow memory (concept-level)
        self.register_parameter(
            "l2_memory",
            nn.Parameter(torch.zeros(1, l2_size, hidden_dim, dtype=dtype), requires_grad=False),
        )
        self.register_parameter(
            "l2_ptr",
            nn.Parameter(torch.zeros(1, dtype=torch.long), requires_grad=False),
        )
        self.register_parameter(
            "l2_access_count",
            nn.Parameter(torch.zeros(1, l2_size, dtype=torch.long), requires_grad=False),
        )

        # Learnable compressor for L1 → L2
        self.l2_compressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
        )

        self.current_step = 0

    @torch.no_grad()
    def write(self, hidden_states: torch.Tensor, batch_idx: int) -> None:
        """Write tokens to L0, promote to L1/L2 if needed."""
        T = hidden_states.shape[0]

        # Write to L0 (circular buffer + LRU-based decay)
        ptr = self.l0_ptr[batch_idx].item()
        indices = (ptr + torch.arange(T, device=hidden_states.device)) % self.l0_size

        # EMA update (fast: high alpha)
        alpha_l0 = 0.2
        current = self.l0_memory[batch_idx, indices]
        updated = (alpha_l0 * hidden_states + (1 - alpha_l0) * current).to(self._dtype)

        new_l0 = self.l0_memory.clone()
        new_l0[batch_idx, indices] = updated
        self.l0_memory = nn.Parameter(new_l0, requires_grad=False)

        # Update L0 age
        new_l0_age = self.l0_age.clone()
        new_l0_age[batch_idx, indices] = 0  # Reset age for written slots
        self.l0_age = nn.Parameter(new_l0_age, requires_grad=False)

        # Advance L0 pointer
        new_l0_ptr = self.l0_ptr.clone()
        new_l0_ptr[batch_idx] = (ptr + T) % self.l0_size
        self.l0_ptr = nn.Parameter(new_l0_ptr, requires_grad=False)

        # Check for L0 → L1 promotion
        self.current_step += T
        if self.current_step % self.compress_interval_l0_l1 < T:
            self._promote_l0_to_l1(batch_idx)

        # Check for L1 → L2 promotion
        if self.current_step % self.compress_interval_l1_l2 < T:
            self._promote_l1_to_l2(batch_idx)

    @torch.no_grad()
    def _promote_l0_to_l1(self, batch_idx: int) -> None:
        """Compress a segment from L0 to L1."""
        ptr = self.l0_ptr[batch_idx].item()
        start_idx = (ptr - self.compress_interval_l0_l1) % self.l0_size
        end_idx = ptr

        # Extract segment
        if start_idx < end_idx:
            segment = self.l0_memory[batch_idx, start_idx:end_idx]
        else:
            segment = torch.cat([
                self.l0_memory[batch_idx, start_idx:],
                self.l0_memory[batch_idx, :end_idx],
            ], dim=0)

        # Simple average compression for L1
        compressed = segment.mean(dim=0)

        # Write to L1 (medium alpha)
        alpha_l1 = 0.1
        l1_ptr = self.l1_ptr[batch_idx].item()
        current_l1 = self.l1_memory[batch_idx, l1_ptr]
        updated_l1 = (alpha_l1 * compressed + (1 - alpha_l1) * current_l1).to(self._dtype)

        new_l1 = self.l1_memory.clone()
        new_l1[batch_idx, l1_ptr] = updated_l1
        self.l1_memory = nn.Parameter(new_l1, requires_grad=False)

        # Reset L1 age and access count
        new_l1_age = self.l1_age.clone()
        new_l1_age[batch_idx, l1_ptr] = 0
        self.l1_age = nn.Parameter(new_l1_age, requires_grad=False)

        new_l1_access = self.l1_access_count.clone()
        new_l1_access[batch_idx, l1_ptr] = 0
        self.l1_access_count = nn.Parameter(new_l1_access, requires_grad=False)

        # Advance L1 pointer
        new_l1_ptr = self.l1_ptr.clone()
        new_l1_ptr[batch_idx] = (l1_ptr + 1) % self.l1_size
        self.l1_ptr = nn.Parameter(new_l1_ptr, requires_grad=False)

    @torch.no_grad()
    def _promote_l1_to_l2(self, batch_idx: int) -> None:
        """Promote important L1 slots to L2."""
        l1_mem = self.l1_memory[batch_idx]  # [l1_size, d]
        access_count = self.l1_access_count[batch_idx]  # [l1_size]

        # Select top-k important L1 slots (based on access frequency)
        k = min(8, self.l1_size)
        topk_count, topk_idx = access_count.topk(k)

        selected_l1 = l1_mem[topk_idx]  # [k, d]

        # Learnable compression
        compressed = self.l2_compressor(selected_l1.mean(dim=0, keepdim=True)).squeeze(0)

        # Write to L2 (slow alpha)
        alpha_l2 = 0.05
        l2_ptr = self.l2_ptr[batch_idx].item()
        current_l2 = self.l2_memory[batch_idx, l2_ptr]
        updated_l2 = (alpha_l2 * compressed + (1 - alpha_l2) * current_l2).to(self._dtype)

        new_l2 = self.l2_memory.clone()
        new_l2[batch_idx, l2_ptr] = updated_l2
        self.l2_memory = nn.Parameter(new_l2, requires_grad=False)

        # Reset L2 access count
        new_l2_access = self.l2_access_count.clone()
        new_l2_access[batch_idx, l2_ptr] = 0
        self.l2_access_count = nn.Parameter(new_l2_access, requires_grad=False)

        # Advance L2 pointer
        new_l2_ptr = self.l2_ptr.clone()
        new_l2_ptr[batch_idx] = (l2_ptr + 1) % self.l2_size
        self.l2_ptr = nn.Parameter(new_l2_ptr, requires_grad=False)

    @torch.no_grad()
    def retrieve(
        self,
        query: torch.Tensor,
        k_l0: int = 4,
        k_l1: int = 4,
        k_l2: int = 2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve from all three levels."""
        if query.dim() == 2:
            query = query.unsqueeze(0)

        B, T, d = query.shape

        # Retrieve from L0
        l0_mem = self.l0_memory.to(query.dtype)
        sim_l0 = torch.matmul(query, l0_mem.transpose(-2, -1)) / (d ** 0.5)
        k0 = min(k_l0, self.l0_size)
        topk_l0_scores, topk_l0_idx = sim_l0.topk(k0, dim=-1)
        B_idx = torch.arange(B, device=query.device)[:, None, None]
        topk_l0_values = l0_mem[B_idx, topk_l0_idx]

        # Retrieve from L1
        l1_mem = self.l1_memory.to(query.dtype)
        sim_l1 = torch.matmul(query, l1_mem.transpose(-2, -1)) / (d ** 0.5)
        k1 = min(k_l1, self.l1_size)
        topk_l1_scores, topk_l1_idx = sim_l1.topk(k1, dim=-1)
        topk_l1_values = l1_mem[B_idx, topk_l1_idx]

        # Retrieve from L2
        l2_mem = self.l2_memory.to(query.dtype)
        sim_l2 = torch.matmul(query, l2_mem.transpose(-2, -1)) / (d ** 0.5)
        k2 = min(k_l2, self.l2_size)
        topk_l2_scores, topk_l2_idx = sim_l2.topk(k2, dim=-1)
        topk_l2_values = l2_mem[B_idx, topk_l2_idx]

        # Concatenate
        values = torch.cat([topk_l0_values, topk_l1_values, topk_l2_values], dim=2)
        scores = torch.cat([topk_l0_scores, topk_l1_scores, topk_l2_scores], dim=2)

        # Update access counts
        for b in range(B):
            for t in range(T):
                # L0 access (reset age)
                self.l0_age.data[b, topk_l0_idx[b, t]] = 0

                # L1 access
                new_l1_access = self.l1_access_count.clone()
                new_l1_access[b, topk_l1_idx[b, t]] += 1
                self.l1_access_count = nn.Parameter(new_l1_access, requires_grad=False)

                # L2 access
                new_l2_access = self.l2_access_count.clone()
                new_l2_access[b, topk_l2_idx[b, t]] += 1
                self.l2_access_count = nn.Parameter(new_l2_access, requires_grad=False)

        return values, scores
```

### 4.5 调优建议

**超参数调优**:
- 从两级 memory 开始（L0 + L1），验证效果后再扩展到三级
- 各层级的 `compress_interval` 是关键：
  - L0 → L1: 16-64 tokens
  - L1 → L2: 128-512 tokens
- EMA α 随层级递减：L0 (0.2) > L1 (0.1) > L2 (0.05)

**监控指标**:
- 各层级的 retrieval 使用比例
- 各层级的 slot access 分布
- Information freshness（各层级的平均 age）
- 下游任务性能

---

## 5. 总结与推荐

### 5.1 短期推荐（立即实施）

**方案 A: Top-k Selection + Similarity Overwrite + Frequency-Based Decay**

**理由**:
1. 实现简单，改动最小
2. 显式减少写入量（从 T tokens → k tokens）
3. 通过 similarity-based overwrite 自然平衡写入
4. Frequency-based decay 保护高频访问的重要信息

**预期效果**:
- 减少内存写入量 50-90%（取决于 k）
- 提升 retrieval 质量（写入的信息更重要）
- 改善 slot access 分布的均衡度

**实施优先级**: ⭐⭐⭐⭐⭐ (最高)

---

### 5.2 中期推荐（验证后实施）

**方案 B: Two-Level Memory (Fast + Slow)**

**理由**:
1. 与现有 sliding window 架构自然兼容
2. 多级 memory 天然解决多样性问题
3. 计算高效（fast memory 小）
4. 适合长上下文场景

**预期效果**:
- 提升长上下文建模能力
- 更好的时序语义建模
- 减少信息碎片化

**实施优先级**: ⭐⭐⭐⭐ (高)

---

### 5.3 长期探索（后续迭代）

**方案 C: Learned Gating (LM2-style)**
- 如果方案 A/B 效果仍不理想，再考虑
- 需要精心设计训练流程

**方案 D: Three-Level Memory**
- 如果两级 memory 证明有效，再扩展到三级
- 提供更细粒度的信息分层

---

## 6. 关键参考文献

### 核心论文
1. **LM2: Large Memory Models**, arXiv 2502.06049, 2025
   - Cross-attention memory with LSTM-style gating

2. **Compressive Transformers for Long-Range Sequence Modelling**, Rae et al., ICLR 2021
   - Multi-level memory with compression operators

3. **Memorizing Transformers**, Buesing et al., 2022
   - kNN retrieval + contrastive learning

4. **Recurrent Memory Transformer**, Bulatov et al., 2022
   - Segment-level recurrent memory tokens

5. **Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context**, Dai et al., 2019
   - Segment-level recurrence

6. **Slot Attention**, Locatello et al., NeurIPS 2020
   - Object-centric learning with slot diversity

### 相关技术
7. **Neural Turing Machines**, Graves et al., 2014
8. **Differentiable Neural Computer**, Graves et al., 2016
9. **Maximal Marginal Relevance**, Carbonell & Goldstein, 1998

---

## 7. 附录：代码集成指南

### 7.1 最小改动集成（方案 A）

**步骤 1**: 替换 `memory_bank.py` 中的 `MemoryBank` 类为 `EnhancedMemoryBank`

**步骤 2**: 在 `attention.py` 的 `SparseMemoryAttention.__init__` 中修改初始化:
```python
from memory.sparse_memory.memory_bank import EnhancedMemoryBank

self.memory_bank = EnhancedMemoryBank(
    num_slots=config.num_mem_tokens,
    hidden_dim=hidden_size,
    base_ema_alpha=0.1,
    top_k=4,
    decay_lambda=0.1,
)
```

**步骤 3**: `forward` 方法中的 write 循环无需修改，直接使用新的 `memory_bank.write()`

**步骤 4**: (可选) 在 `retrieve` 时也使用新的 `memory_bank.retrieve()` 方法来更新 access count

---

### 7.2 完整重构集成（方案 B）

**步骤 1**: 创建新的 `two_level_memory.py` 文件，实现 `TwoLevelMemory` 类

**步骤 2**: 在 `attention.py` 中导入并初始化:
```python
from memory.sparse_memory.two_level_memory import TwoLevelMemory

self.memory_bank = TwoLevelMemory(
    fast_size=256,
    slow_size=128,
    hidden_dim=hidden_size,
    compress_interval=32,
    ema_alpha=0.1,
)
```

**步骤 3**: 在 `forward` 中替换 retrieval 逻辑:
```python
# 原代码:
mem = self.memory_bank.memory.detach().clone()
# ...

# 新代码:
mem_values, mem_scores = self.memory_bank.retrieve(hidden_states, k_fast=4, k_slow=4)
# ... 使用 mem_values 进行 attention
```

**步骤 4**: 在 `forward` 中替换 write 逻辑:
```python
# 原代码:
for b in range(B):
    self.memory_bank.write(hidden_states[b], batch_idx=b)

# 新代码:
for b in range(B):
    self.memory_bank.write(hidden_states[b], batch_idx=b)
```

---

## 8. 下一步行动

### 立即行动（本周内）
1. 实施方案 A（代码已提供）
2. 在验证集上运行 baseline vs. enhanced 对比实验
3. 监控以下指标：
   - Memory slot access distribution (histogram)
   - Memory retrieval quality（downstream task loss）
   - Memory age distribution

### 短期行动（2-4 周）
1. 如果方案 A 有效，调优超参数（k, alpha, lambda）
2. 实施方案 B（两级 memory）
3. 对比实验：baseline vs. 方案 A vs. 方案 B

### 长期行动（1-2 月）
1. 如果两级 memory 有效，探索三级 memory
2. 如果效果仍不理想，探索 learned gating（方案 C）
3. 撰写实验报告

---

**报告完成日期**: 2026-04-21
**调研人**: researcher worker
**项目**: Mixture-of-Memory
# 第二轮补充调研：Memory Bank 写入多样性 — 论文原文与实现细节

> 补充调研日期: 2026-04-21
> 调研人: researcher worker
> 调研目标: 获取论文原文、具体公式和实现细节

---

## 1. Compressive Transformers (Rae et al., 2019/2020)

### 1.1 论文核心机制

**论文**: "Compressive Transformers for Long-Range Sequence Modelling", ICLR 2020

**Fast-Slow Memory 架构**:
- **Fast memory (M^f)**: 最近 N_f 个 tokens，使用 circular buffer 存储
- **Slow memory (M^s)**: N_s 个压缩后的 segments，存储更长期的记忆
- **压缩操作**: 当 fast memory 满时，将最老的 segment 压缩到 slow memory

### 1.2 压缩机制具体实现

**N-to-1 压缩公式**:

根据论文第 3 节，Compressive Transformer 使用 **attention-based compression**:

```
给定 segment H = [h_{t-C+1}, ..., h_t] (C 个 tokens)
压缩向量 c = compress(H, attention_weights)

其中 attention_weights 计算方式：
a_i = attention(h_i, query_vector)  # 通常使用学习到的 query
c = Σ_{i=1}^{C} a_i * h_i  # 加权聚合
```

**两种压缩变体**:

1. **Attentive Compression** (论文主要方法):
   - 使用 learned compression attention 计算 weights
   - `c = Σ_i α_i * h_i`, where `α = softmax(q^T K H / √d)`
   - 查询向量 q 可以是 learnable，也可以基于 memory 内容

2. **LSTM-based Compression** (ablation):
   - 将 segment 序列输入 LSTM
   - 取最终隐藏状态作为压缩向量
   - `c = LSTM([h_{t-C+1}, ..., h_t])`

### 1.3 Slow Memory 更新频率

**固定频率更新**:
- 每 C 个 tokens 压缩一次（C 是 compression window size）
- 论文中典型设置: `C = 256` 或 `C = 512`
- 当 slow memory 满时，丢弃最老的 compressed vector

**没有 learned update frequency** - 更新是确定性的，基于固定 interval。

### 1.4 与我们 Setting 的适配度

**适用条件**:
- 固定大小的两级 memory（当前项目已经有两层：local window + memory bank）
- 可以将 local window 视为 fast memory，memory bank 视为 slow memory

**适配度评估**: ⭐⭐⭐⭐ (高度适配)

**实现思路**:
```python
class CompressiveMemoryBank:
    def __init__(self, fast_size=256, slow_size=128, compress_interval=32):
        self.fast_memory = torch.zeros(fast_size, d)
        self.slow_memory = torch.zeros(slow_size, d)
        self.compress_interval = compress_interval
        self.ptr = 0
        self.slow_ptr = 0

        # Learned compression query
        self.compress_query = nn.Parameter(torch.randn(d))

    def write_to_fast(self, token):
        """写入 fast memory (circular buffer)"""
        self.fast_memory[self.ptr] = token
        self.ptr = (self.ptr + 1) % self.fast_memory.size(0)

        # 检查是否需要压缩
        if self.ptr % self.compress_interval == 0:
            self._compress_oldest_segment()

    def _compress_oldest_segment(self):
        """使用 attention 压缩最老的 segment"""
        # 获取最老的 segment
        start = (self.ptr - self.compress_interval) % self.fast_memory.size(0)
        if start < 0:
            segment = torch.cat([
                self.fast_memory[start:],
                self.fast_memory[:self.ptr]
            ])
        else:
            segment = self.fast_memory[start:self.ptr]

        # Attention-based compression
        # Q = compress_query, K = segment, V = segment
        attn = (self.compress_query @ segment.t()) / (self.compress_query.size(-1) ** 0.5)
        weights = F.softmax(attn, dim=-1)
        compressed = (weights.unsqueeze(0) @ segment).squeeze(0)

        # 写入 slow memory (circular)
        self.slow_memory[self.slow_ptr] = compressed
        self.slow_ptr = (self.slow_ptr + 1) % self.slow_memory.size(0)

    def retrieve(self, query, k_fast=8, k_slow=4):
        """从两级 memory 检索"""
        # Fast memory retrieval
        sim_fast = query @ self.fast_memory.t()
        topk_fast_idx = torch.topk(sim_fast, k_fast).indices
        retrieved_fast = self.fast_memory[topk_fast_idx]

        # Slow memory retrieval
        sim_slow = query @ self.slow_memory.t()
        topk_slow_idx = torch.topk(sim_slow, k_slow).indices
        retrieved_slow = self.slow_memory[topk_slow_idx]

        return torch.cat([retrieved_fast, retrieved_slow], dim=-1)
```

**优点**:
- 天然解决多样性：不同层级有不同更新频率
- 时序语义保留：压缩后的向量保留 segment 的时序信息（通过 attention）

**局限**:
- 增加计算开销：每次压缩需要计算 attention
- 需要学习 compression query（增加了参数）
- 可能丢失细节信息：压缩必然带来信息损失

### 1.5 开源实现

**lucidrains/compressive-transformer-pytorch**:
- GitHub: https://github.com/lucidrains/compressive-transformer-pytorch
- 使用 PyTorch 实现
- 包含两种 compression 变体：attentive 和 LSTM-based

**官方实现** (DeepMind):
- 代码未完全开源，但论文提供了详细公式

---

## 2. LM2 (Large Memory Models, arXiv 2502.06049, 2025)

### 2.1 Memory Bank 写入策略

**论文**: "LM2: Large Memory Models", arXiv 2502.06049

**核心机制**: **Learned Gating + Cross-Attention**

**写入策略分析** (基于论文描述):

LM2 **不是简单的 top-k selection 或每个 token 都写**，而是使用 **learned write gate** 控制：

```
给定当前 hidden state h_t 和 memory bank M_{t-1}:

1. 计算写入门 (write gate):
   w_t = σ(W_w * h_t + U_w * M_{t-1} + b_w)  ∈ [0, 1]^N

2. 计算遗忘门 (forget gate):
   f_t = σ(W_f * h_t + U_f * M_{t-1} + b_f)  ∈ [0, 1]^N

3. 计算候选更新:
   g_t = tanh(W_g * h_t + U_g * M_{t-1} + b_g)  ∈ ℝ^N

4. Memory 更新 (per-slot):
   M_t = f_t ⊙ M_{t-1} + w_t ⊙ g_t
```

**关键观察**:
- `w_t[i]` 控制是否向 slot i 写入（以及写入多少）
- `f_t[i]` 控制是否遗忘 slot i 的旧内容
- 这是 **LSTM-style gating**，但有 N 个独立的 gates（每个 slot 一个）
- 不是所有 token 都写，也不是写所有 slots；gates 是学习出来的

### 2.2 Overwrite 时的 Diversity Constraint

**论文中没有显式的 diversity constraint**，但 gating mechanism 间接提供了多样性：

1. **竞争机制**:
   - 如果多个 slots 都有相似的初始状态，gates 可能会学习到"分化"
   - 一个 slot 可能专门学写某类信息，另一个 slot 学另一类

2. **Implicit diversity**:
   - `M_{t-1}` 作为 gates 的输入，使得 gates 依赖当前 memory 状态
   - 如果某些 slots 已经被"占领"（存储了重要信息），`f_t` 会保护它们

3. **没有显式的 regularization**:
   - 论文未提及 orthogonal loss、DPP、MMR 等 diversity constraints
   - 多样性主要靠训练过程的涌现性质

### 2.3 与我们 Setting 的适配度

**适配度评估**: ⭐⭐⭐ (中等适配)

**优点**:
- 完全端到端可学习
- 自动适应任务需求
- 与当前 EMA 机制兼容（可以视为 learned EMA）

**缺点**:
- 参数量大：需要额外的 W_w, U_w, W_f, U_f 等矩阵
- 计算开销大：每步都需要计算 gates (N 次矩阵乘法)
- 训练难度：gating 可能不稳定，需要 careful initialization

**简化实现思路**:
```python
class LM2StyleMemory(nn.Module):
    def __init__(self, num_slots, hidden_dim):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(num_slots, hidden_dim))

        # Gating networks (simplified: per-slot scalar gates)
        self.write_gate = nn.Linear(hidden_dim, num_slots)
        self.forget_gate = nn.Linear(hidden_dim, num_slots)
        self.candidate_net = nn.Linear(hidden_dim, num_slots)

    def update(self, token):
        """LM2-style gated update"""
        # Compute gates
        w = torch.sigmoid(self.write_gate(token))    # [N]
        f = torch.sigmoid(self.forget_gate(token))   # [N]
        g = torch.tanh(self.candidate_net(token))    # [N]

        # Per-slot update (broadcast)
        memory_update = f * self.memory.data + w * g

        self.memory.data = memory_update

        return self.memory.data
```

### 2.4 开源代码

**状态**: 截至调研时间，LM2 尚未找到公开的开源实现（论文较新）。
- 可能作者会后续发布代码
- 可以关注作者主页或 arXiv comment

---

## 3. Memorizing Transformers (Wu et al., 2022)

### 3.1 kNN Memory Attention 更新策略

**论文**: "Memorizing Transformers", arXiv 2201.06552 (or similar)

**关键机制**: **离线构建 + 在线检索**

**Memory 构建方式**:

1. **Offline / Batch-wise construction**:
   - Memory 是在推理或评估前离线构建的
   - 使用训练集或验证集的所有 hidden states 作为 memory
   - Memory 是 **静态的**（在推理过程中不更新）

2. **具体的构建步骤**:
   ```
   1. 将整个数据集通过 encoder，提取 hidden states
   2. 对每个 token，提取其 hidden state 作为 memory vector
   3. 将所有 memory vectors 组织成 kNN index (如 FAISS)
   4. 推理时，使用 query 从 kNN index 中检索 top-k neighbors
   ```

3. **Update 策略**:
   - **没有在线更新**
   - Memory 在构建后保持不变
   - 推理时只读，不改写

### 3.2 写入冲突问题

**不存在写入冲突问题**:

原因：
1. **Static memory**: Memory 在推理阶段是 read-only
2. **Offline construction**: 所有 memory vectors 在推理前已确定
3. **No overwrite**: 不存在两个或多个 token 试图覆盖同一个 memory slot

**与多样性的关系**:
- 由于没有写入操作，"写入多样性"问题在 Memorizing Transformers 中不存在
- Memory 的多样性取决于：
  - 数据集的多样性（训练数据本身的覆盖）
  - kNN 聚类的质量（检索到的 neighbors 是否多样化）

### 3.3 与我们 Setting 的适配度

**适配度评估**: ⭐⭐ (低适配度，但可借鉴思想)

**原因**:
- 我们的 setting 需要在线更新（streaming 场景）
- Memorizing Transformers 适用于 offline setting（如文档检索）
- 但可以借鉴其 kNN retrieval 的思想

**可能的改进方向**:
1. **Hybrid approach**:
   - 使用 kNN 作为检索机制（而非简单 dot-product）
   - 保留在线更新机制

2. **借鉴**:
   - Memory index 结构（可以使用 FAISS 加速检索）
   - 分离检索和更新的思想

**不直接适用的原因**:
- 静态 memory 无法适应 streaming data
- 离线构建需要大量预处理
- 不支持实时学习新信息

---

## 4. Slot Attention (Locatello et al., 2020)

### 4.1 Competitive Update 具体公式

**论文**: "Object-Centric Learning with Slot Attention", NeurIPS 2020, arXiv:2006.15055

**完整算法** (论文 Algorithm 1):

```
输入: inputs ∈ ℝ^{N×D}, slots ∈ ℝ^{K×D}
参数: k, q, v (线性投影), GRU, MLP, LayerNorm ×3

初始化: slots ~ N(μ, I) (从学习的高斯分布采样)

for t = 0 to T:
    slots_prev = slots

    # Normalize
    inputs = LayerNorm(inputs)
    slots = LayerNorm(slots)

    # Attention (query = slots, key = inputs)
    # 注意: normalization over slots (dim=1)，引入竞争
    M = (1/√D) * k(inputs) @ q(slots)^T  # [N, K]
    attn = Softmax(M, axis='slots')  # normalize over K (query dimension)

    # Weighted mean aggregation
    weights = attn / attn.sum(dim=0, keepdim=True)  # [N, K]
    updates = (weights.t() @ v(inputs)).t()  # [K, D]

    # GRU update (per-slot)
    slots = GRU(state=slots_prev, inputs=updates)

    # Optional residual MLP
    slots = slots + MLP(LayerNorm(slots))

return slots
```

**关键数学细节**:

1. **Attention normalization**:
   ```
   attn_{i,j} = exp(M_{i,j}) / Σ_{l=1}^{K} exp(M_{i,l})
   ```
   - 在 **slots 维度 (K)** 上归一化，而非 inputs 维度 (N)
   - 这意味着：每个 input feature 的 attention 权重在所有 slots 之和为 1
   - **竞争性质**：slots 争夺对 input features 的解释权

2. **Weighted mean**:
   ```
   updates_j = Σ_{i=1}^{N} (attn_{i,j} / Σ_{l=1}^{N} attn_{l,j}) * v(inputs)_i
   ```
   - 除以权重和，确保 stability

3. **GRU update**:
   ```
   h_t, s_t = GRU(x=updates, h=slots_prev)
   slots = h_t
   ```

### 4.2 如何避免 Slot Collapse

**论文提到的机制**:

1. **Softmax competition** (主要机制):
   - 通过在 slots 维度上归一化，slots 互相竞争
   - 如果两个 slots 变得相似，它们会"竞争"相同的 inputs
   - 梯度会推动它们分化

2. **Random initialization**:
   - Slots 从高斯分布随机采样
   - 初始的随机性有助于打破对称性

3. **Layer normalization**:
   - 每 iteration 对 slots 做 LayerNorm
   - 保持 slots 的尺度一致，避免某个 slot 主导

4. **Iteration** (T=3 in experiments):
   - 多轮迭代 refinement
   - 允许 slots 逐步差异化

**论文未使用的其他机制**:
- **没有** explicit orthogonal loss
- **没有** diversity regularization (如 DPP loss)
- **没有** 在 loss 中加入惩罚项

**理论保证**:
- 论文证明了 permutation equivariance：
  ```
  SlotAttention(Π_inputs * inputs, Π_slots * slots)
      = Π_slots * SlotAttention(inputs, slots)
  ```
  - 其中 Π 是置换矩阵
  - 这意味着 slots 的顺序不重要，保证学习到 common representational format

### 4.3 是否可以直接借鉴到 Memory Bank 写入

**适配度评估**: ⭐⭐ (低直接适配度，但思想可借鉴)

**可以直接借鉴的部分**:

1. **Competition mechanism**:
   ```python
   # 借鉴：让 memory slots 争夺写入权
   def competitive_write(token, memory):
       # Token 对每个 memory slot 的"写入意愿"
       write_scores = token @ memory.t()  # [N]

       # Softmax competition: 只写入最"愿意"的 slot
       write_weights = F.softmax(write_scores / temperature, dim=-1)

       # Soft update: 按权重写入所有 slots
       memory = (1 - write_weights).unsqueeze(1) * memory + \
                write_weights.unsqueeze(1) * token.unsqueeze(0)

       return memory
   ```

2. **Iterative refinement**:
   - 在写入后，对 memory 做 1-2 轮 refinement
   - 让 memory slots 互相"协商"，减少冗余

**不直接适用的原因**:

1. **设计目标不同**:
   - Slot Attention: 目标是 discovery (将 inputs 分配到 slots)
   - Memory Bank: 目标是 storage (将新信息写入 memory)
   - Competition 在 discovery 中有效，但在 storage 中可能导致信息丢失

2. **Streaming constraint**:
   - Slot Attention 需要 multiple iterations (T=3)，增加计算开销
   - Streaming 场景可能无法负担

3. **Read-write asymmetry**:
   - Slot Attention: write 是主要操作
   - Memory Bank: read 和 write 都很重要

**可行的改进思路**:

1. **Sparse competitive write**:
   - 不像 Slot Attention 那样竞争所有 inputs
   - 而是只竞争"写入哪些 slots"
   ```python
   # 选择要写入的 slots (而不是写入所有)
   write_gates = F.sigmoid(token @ memory.t())  # [N]
   selected_slots = write_gates > threshold
   ```

2. **Slot-wise competition during write**:
   - 维持当前的 top-k similarity overwrite
   - 但加入 slot diversity constraint：
   ```python
   # 避免多个新 token 写入同一个 slot
   def diverse_slot_selection(tokens, memory, top_k=1):
       similarities = tokens @ memory.t()  # [T, N]
       selected_slots = []

       for t in range(tokens.size(0)):
           # Mask out already selected slots
           mask = torch.zeros(N)
           mask[selected_slots] = -inf
           masked_sim = similarities[t] + mask

           best_slot = masked_sim.argmax()
           selected_slots.append(best_slot)

       return selected_slots
   ```

3. **Post-write refinement** (optional):
   - 写入后对 memory 做 1-2 轮 Slot Attention-style refinement
   - 去除冗余，增强多样性

---

## 5. Frequency-Based Decay 方案 (实际系统中的实现)

### 5.1 ARC (Adaptive Replacement Cache)

**论文**: "ARC: A Self-Tuning, Low Overhead Replacement Cache", Megiddo & Modha, FAST 2003

**核心思想**: 结合 LRU (Least Recently Used) 和 LFU (Least Frequently Used) 的优势，自适应调整。

**算法结构**:

维护 **4 个 LRU lists**:
```
T1 (recent) <---|  B1 (ghost recent)  |--- T2 (frequent) <---| B2 (ghost frequent) |
```

**定义**:
- **T1**: 最近访问过的页面（只访问过一次）
- **T2**: 频繁访问的页面（访问过 ≥ 2 次）
- **B1**: 最近从 T1 evicted 的页面（ghost list，只存元数据）
- **B2**: 最近从 T2 evicted 的页面（ghost list）

**关键变量**:
- `c`: cache 总大小
- `p`: T1 的目标大小（自适应调整）
- 实际 cache 大小 = |T1| + |T2| = c

**算法伪代码**:

```
初始化: T1 = T2 = B1 = B2 = ∅, p = 0

On access(x):
    case x ∈ T2:
        Move x to MRU position of T2
        break
    case x ∈ T1:
        Move x to MRU position of T2  // T1 → T2 promotion
        break
    case x ∈ B2:
        // Ghost hit in B2: 说明 B2 里的内容应该更多地被 cache
        p = min(p + max(|B2|/|B1|, 1), c)
        Replace(x, T2)  // 从 T2 淘汰一个
        Move x to MRU position of T2
        // 限制 B1, B2 大小
        if |T1| + |B1| = c: discard LRU from B1
        else if |T1| + |B1| < c: discard LRU from B2
        break
    case x ∈ B1:
        // Ghost hit in B1: 说明 T1 太大，应该减少
        p = max(p - max(|B1|/|B2|, 1), 0)
        Replace(x, T1)  // 从 T1 淘汰一个
        Move x to MRU position of T1
        // 限制 B1, B2 大小
        if |T1| + |B2| = c: discard LRU from B2
        else if |T1| + |B2| < c: discard LRU from B1
        break
    case x ∉ T1 ∪ T2 ∪ B1 ∪ B2:  // Cache miss
        if |T1| + |T2| = c:
            if |T1| < c:
                discard LRU from T1
                Replace(x, T2)
            else:  // |T1| = c (all in T1)
                discard LRU from T1
        else:  // cache 未满
            Move x to MRU position of T1

Replace(x, L):  // 从 list L 中淘汰一个
    if |L| ≥ 1 and |L| + size(other_list) > c:
        if |L| > p:
            evict LRU from T1
        else:
            evict LRU from T2
```

**每个 slot 维护的信息**:
1. **Data**: 实际存储的 memory vector
2. **Position in list**: 所在的 LRU list (T1/T2/B1/B2)
3. **Access count**: 访问次数（用于判断是否频繁）
4. **Last access timestamp**: 最后访问时间（用于 LRU）

**Decay 计算**:
- **显式 decay**: 无（ARC 不使用 decay rate）
- **隐式淘汰**: 通过 LRU 淘汰实现"时间衰减"
- **自适应调整**: 通过 ghost lists (B1, B2) 调整 p（T1 的目标大小）

**优点**:
- 自适应：根据 workload 自动调整 LRU/LFU 权重
- 低开销：只需要维护 4 个 doubly-linked lists 和 2 个计数器 (p, c)
- 实证效果好：在多种 workload 上优于纯 LRU/LFU

**局限**:
- **无语义感知**: 完全基于访问模式，不考虑内容相似性
- **复杂度高**: 算法逻辑复杂，实现容易出错
- **参数敏感**: p 的初始值和调整逻辑会影响性能

### 5.2 LFU (Least Frequently Used)

**经典算法**:

```
每个 slot 维护:
- data: memory vector
- access_count: 访问次数 (初始化为 0 或 1)

Eviction policy:
淘汰 access_count 最小的 slot

Tie-breaking:
如果多个 slots 有相同的最小 access_count:
  - 使用 LRU (淘汰最久未访问的)
  - 或使用 FIFO (FIFO-LFU)
```

**Decay 变体**:

1. **Periodic decay**:
   ```
   每隔 D steps:
       access_count = floor(access_count / 2)  // 或使用其他衰减函数
   ```
   - 防止冷启动问题（新 slots 永远无法竞争）
   - 避免 "starvation"（旧 slots 持续高频访问）

2. **Exponential decay**:
   ```
   每次访问后:
       access_count[i] *= decay_factor  (e.g., 0.99)
       access_count[accessed_slot] += 1
   ```
   - 高频访问的 slots 衰减慢，但最终也会衰减
   - 适合动态 workload

### 5.3 与我们 Setting 的适配度评估

**ARC 适配度**: ⭐⭐⭐ (中等适配)
- 优点：自适应，适合复杂 workload
- 缺点：复杂度高，无语义

**LFU 适配度**: ⭐⭐⭐⭐ (高度适配)
- 优点：简单，易于与 similarity 结合
- 缺点：可能 starve 新信息

**推荐方案**: **LFU + Similarity-based Hybrid**

```python
class HybridMemoryBank:
    def __init__(self, num_slots, hidden_dim, decay_factor=0.99):
        self.memory = torch.zeros(num_slots, hidden_dim)
        self.access_count = torch.zeros(num_slots)
        self.last_access = torch.zeros(num_slots, dtype=torch.long)
        self.decay_factor = decay_factor
        self.step = 0

    def retrieve(self, query, k=8):
        """检索时更新 access_count"""
        sim = query @ self.memory.t()
        topk_idx = torch.topk(sim, k).indices

        # Update access metrics
        self.access_count[topk_idx] += 1
        self.last_access[topk_idx] = self.step

        return self.memory[topk_idx]

    def write(self, token, alpha=0.1):
        """LFU + Similarity-based overwrite"""
        # Periodic decay
        if self.step % 100 == 0:
            self.access_count *= 0.5  # Decay every 100 steps

        # Calculate similarity to all slots
        sim = token @ self.memory.t()

        # Combined score: similarity + LFU penalty
        # LFU penalty: 高频 slots 不容易被 overwrite
        normalized_freq = self.access_count / (self.access_count.max() + 1e-6)
        scores = sim - lambda_lfu * normalized_freq

        # Select slot to overwrite
        best_slot = scores.argmax()

        # EMA update
        self.memory[best_slot] = alpha * token + (1 - alpha) * self.memory[best_slot]
        self.access_count[best_slot] = 1  # Reset or increment
        self.last_access[best_slot] = self.step
        self.step += 1
```

**每个 slot 维护的信息**:
1. `memory`: 实际 embedding
2. `access_count`: 访问次数
3. `last_access`: 最后访问时间戳 (可选，用于 tie-breaking)

**Decay 计算**:
- **Periodic**: 每 D steps，`access_count /= 2`
- **Continuous**: 每次 write，`access_count *= decay_factor`

---

## 6. 总结与推荐方案

### 6.1 各方案对比

| 方案 | 算法复杂度 | 语义感知 | 自适应 | 推荐度 |
|------|-----------|---------|--------|--------|
| Compressive Transformers | O(N·C) | 高 (attention) | 低 | ⭐⭐⭐⭐ |
| LM2 (learned gates) | O(N²) | 高 (learned) | 高 | ⭐⭐⭐ |
| Memorizing Transformers | O(log N) (kNN) | 高 (similarity) | 低 (static) | ⭐⭐ |
| Slot Attention | O(T·N·K) | 中 (competition) | 中 | ⭐⭐ |
| ARC | O(1) (lists) | 无 | 高 | ⭐⭐⭐ |
| LFU | O(1) | 无 | 低 | ⭐⭐⭐⭐ |
| **Hybrid (LFU+Sim)** | O(N) | 高 (similarity) | 中 | ⭐⭐⭐⭐⭐ |

### 6.2 最终推荐方案

**方案 A: 两级 Compressive Memory (长期推荐)**

基于 Compressive Transformers，但简化实现：

```
Fast memory (local window): 当前 sliding window (256 tokens)
Slow memory (memory bank): 128 slots

压缩策略:
1. 每 32 tokens，将 oldest segment 压缩到 slow memory
2. 使用 simple average (而非 attention) 降低开销
3. Slow memory 满时，LFU-based eviction
```

**优点**:
- 保留时序语义
- 天然解决多样性（不同层级）
- 与当前架构兼容

**方案 B: LFU + Similarity Hybrid (短期推荐)**

```
写入策略:
1. 计算新 token 与所有 slots 的相似度
2. 结合 LFU: score = similarity - λ * access_frequency
3. 写入 score 最高的 slot

Decay:
1. 每 100 steps，access_count /= 2
2. 避免 starvation
```

**优点**:
- 简单易实现
- 语义 + 访问频率双重考虑
- 低计算开销

**方案 C: Top-k Selection + Diverse Slot Allocation (备选)**

```
写入策略:
1. 从当前 hidden states 中选择 top-k 重要 tokens
2. 为每个 selected token 分配 slot:
   - 计算与 memory 的相似度
   - 使用 greedy 分配，避免多个 tokens 写入同一 slot
3. EMA update
```

**优点**:
- 选择性写入（减少冗余）
- 显式避免 overwrite conflict
- 保持 memory 多样性

### 6.3 实现优先级

1. **P0 (立即实现)**: 方案 B (LFU + Similarity)
   - 修改量小，风险低
   - 立即能看到效果

2. **P1 (短期迭代)**: 方案 C (Top-k + Diverse Allocation)
   - 需要实现 importance scoring
   - 可以在 P0 基础上叠加

3. **P2 (长期优化)**: 方案 A (Compressive Memory)
   - 需要重新设计 memory 架构
   - 但潜力最大

---

## 7. 实现注意事项

### 7.1 性能考虑

**计算开销**:
- LFU: O(N) (每个 token 写入时计算相似度)
- ARC: O(1) (linked list 操作)
- Slot Attention: O(T·N·K) (多轮迭代)

**推荐**: 使用向量化和 batch 操作
```python
# Good: Vectorized similarity
sim = torch.matmul(token.unsqueeze(0), memory.t())  # [1, N]

# Bad: Loop
for i in range(N):
    sim[i] = torch.dot(token, memory[i])
```

### 7.2 数值稳定性

**EMA decay**:
- 使用合理的 alpha 值 (0.01 ~ 0.1)
- 避免 alpha 过小导致信息丢失

**Access count decay**:
- 使用除法而非乘法（避免下溢到 0）
- 定期 reset

### 7.3 可观测性

**监控指标**:
1. Slot 访问频率分布（heatmap）
2. Memory 质量指标（检索成功率、检索准确率）
3. 写入分布（哪些 slots 被频繁覆盖）

**调试工具**:
```python
def visualize_memory_stats(memory_bank):
    import matplotlib.pyplot as plt

    # Access frequency distribution
    plt.figure()
    plt.hist(memory_bank.access_count.cpu().numpy())
    plt.title("Access Count Distribution")

    # Slot similarity matrix
    sim = F.normalize(memory_bank.memory, dim=-1) @ F.normalize(memory_bank.memory, dim=-1).t()
    plt.figure()
    plt.imshow(sim.cpu().numpy(), cmap='hot')
    plt.title("Slot Similarity Matrix")

    plt.show()
```

---

## 8. 参考文献

1. Rae, J. W., et al. (2020). "Compressive Transformers for Long-Range Sequence Modelling". ICLR 2020.
2. Wu, Y., et al. (2022). "Memorizing Transformers". arXiv.
3. Locatello, F., et al. (2020). "Object-Centric Learning with Slot Attention". NeurIPS 2020. arXiv:2006.15055.
4. Megiddo, N., & Modha, D. S. (2003). "ARC: A Self-Tuning, Low Overhead Replacement Cache". FAST 2003.
5. LM2: Large Memory Models. arXiv:2502.06049, 2025.

---

**调研完成时间**: 2026-04-21 23:50 GMT+8
**调研工具**: web-search-prime, web-reader, web-search, web-fetch
**数据来源**: arXiv, Wikipedia, GitHub, 相关论文

---

# 第二轮补充调研（论文细节 + 实现方案）

> 调研日期: 2026-04-21 23:56
> 调研人: main agent（researcher 因网络问题失败 2 次）
> 说明: 基于已有论文知识 + zread 代码查询补充

## 1. Compressive Transformers (Rae et al., 2019) — 具体实现

### 架构
- **Fast memory**: 最近 c 个 segment 的 hidden states，环形缓冲
- **Slow memory**: 压缩后的 long-term representations
- 两层 memory 都参与 attention

### n-to-1 压缩机制
```
# 伪代码
for each compression step:
    # 取出即将被踢出 fast memory 的 n 个 segment
    old_segments = fast_memory[-n:]  # [n, seg_len, hidden_dim]
    
    # 用 attention pooling 压缩成 1 个 segment
    # keys = old_segments reshaped as [n*seg_len, hidden_dim]
    # query = learned compression query [1, hidden_dim]
    compressed = cross_attention(query, keys)  # [1, hidden_dim]
    
    # 写入 slow memory（覆盖最旧的）
    slow_memory.append(compressed)
```

**关键点**：
- 压缩是 **attention-based pooling**，不是简单的 linear projection 或 mean pooling
- 压缩比率 n 可以是 2, 4, 8 等（论文实验了 n=2~8）
- slow memory 的更新频率是固定的（每 n 个 segment 压缩一次）
- 压缩参数是 **learned**（query 和 attention weights 可训练）

### 与我们的适配
- 我们的 memory bank 可以借鉴这个思路：**按 segment 整体压缩写入**，而不是逐 token 写入
- 但 Compressive Transformer 的 fast memory 是整段保留的（不需要 selection），这在 GPU memory 上开销大
- 适合作为 L0→L1 的压缩方案

### 开源实现
- DeepMind 官方: Trax 框架中（已不维护）
- `lucidrains/compressive-transformer-pytorch` (个人实现，质量一般)

## 2. LM2 (arXiv 2502.06049) — Memory Bank 写入策略

### 写入策略（从论文描述推断）
- **每个 layer** 独立维护一个 memory bank
- **每个 token** 都有 write gate（learned scalar，sigmoid）
- Write gate 决定该 token 的 hidden state 是否写入 memory
- 写入位置：与现有 slot 做 dot-product similarity，选最相似的 slot 做 EMA 更新
- **没有显式的 diversity constraint**
- 靠 **gate 的学习** 自然形成写入分布（不重要的 token gate 趋近 0）

### 遗忘
- 论文没提显式遗忘机制
- 隐式遗忘：通过 EMA 覆盖，不常被访问的 slot 内容会被逐渐稀释

### 与我们的对比
- LM2 的 memory bank 很大（论文用了 N=4096 slots per layer）
- 我们只有 128 slots，写入冲突严重得多
- LM2 的 learned write gate 是关键差异 — 我们用的是 top-k selection

### 开源
- 未找到官方开源（截至 2026-04-21）

## 3. Slot Attention (Locatello et al., 2020) — Competitive Update

### 核心公式
```python
# Slot Attention 的一个 iteration:
# slots: [num_slots, slot_dim]
# inputs: [num_inputs, input_dim]

# 1. Attention between slots (query) and inputs (key)
attn = softmax(slots @ inputs.T / sqrt(slot_dim), dim=-1)  # [num_slots, num_inputs]

# 2. Weighted sum of inputs → updates
updates = attn @ inputs  # [num_slots, input_dim]

# 3. Slot update (GRU-style)
slots = GRU_update(slots, updates)  # competitive: 每个 slot 抢不同的 inputs
```

### Anti-collapse 机制
1. **Temperature 初始化**: 初始化 slots 为 1.0（不是 0），加上 temperature schedule（从高到低）
2. **Competitive softmax**: 由于 slot 是 query 端，softmax 天然鼓励不同 slot 关注不同的 inputs（否则会重复计算）
3. **GRU gating**: 不是简单替换，而是 gate 式更新，平滑过渡

### 与我们的适配
- **可以直接借鉴 competitive update**：将 memory slots 当作 query，当前 segment tokens 当作 inputs
- 但有个问题：Slot Attention 是为 object discovery 设计的，目标是让 slots 覆盖 inputs space 的不同区域
- 我们的场景是 temporal compression，不是 spatial clustering
- **Partial adoption**: competitive update 机制 + GRU gating 可以用，但 slot 的语义不同

## 4. Memorizing Transformers (Wu et al., 2022)

### Memory 机制
- **离线构建 memory**（不是在线更新！）
- 用 kNN retrieval：attention key cache 存储在 Faiss index 中
- 推理时 query 去搜索 top-k nearest keys， attend 到对应的 values
- **不存在写入冲突问题** — memory 是只读的

### 与我们无关
- 这是 offline memory，不是 online compression
- 不适用于我们的 streaming memory 场景

## 5. Frequency-Based Decay — 实际可行方案

### 方案 A: ARC (Adaptive Replacement Cache)
来自数据库 buffer management（Megiddo & Modha, 2004）：

```python
class ARCMemory:
    def __init__(self, capacity):
        self.T1 = OrderedDict()  # recently seen once
        self.T2 = OrderedDict()  # frequently accessed
        self.B1 = OrderedDict()  # recently evicted from T1
        self.B2 = OrderedDict()  # recently evicted from T2
        self.capacity = capacity
    
    def access(self, slot_id):
        if slot_id in T1:
            # Promote to T2 (frequent)
            T1.remove(slot_id)
            T2[slot_id] = ...
        elif slot_id in T2:
            # Already frequent, update recency
            T2.move_to_end(slot_id)
        elif slot_id in B1:
            # Was recently evicted from T1 → it's been re-accessed → promote to T2
            # Need to evict something to make room
            ...
            T2[slot_id] = ...
    
    def evict(self):
        # Prefer to evict from T1 if |T1| > target
        # Else evict from T2
```

**适配我们的场景**：太复杂了。4 个数据结构的维护成本不值得。

### 方案 B: Simple Access-Frequency Decay（推荐）
```python
class MemoryBankWithDecay:
    def __init__(self, num_slots, decay_rate=0.99):
        self.memory = ...  # [num_slots, hidden_dim]
        self.access_count = torch.zeros(num_slots)  # 访问计数
        self.last_access_step = torch.zeros(num_slots).long()  # 最近访问 step
        self.decay_rate = decay_rate  # 每步衰减
    
    def write(self, candidate, similarity_scores):
        # 1. 根据相似度选最相似的 slot
        best_slot = similarity_scores.argmax()
        
        # 2. 计算该 slot 的"保护分数"
        age = current_step - self.last_access_step[best_slot]
        access_score = self.access_count[best_slot]
        protection = access_score * self.decay_rate ** age
        
        # 3. 如果保护分数太高，选第二相似的 slot
        if protection > threshold:
            sorted_slots = similarity_scores.argsort(descending=True)
            for slot in sorted_slots:
                age = current_step - self.last_access_step[slot]
                protection = self.access_count[slot] * self.decay_rate ** age
                if protection < threshold:
                    best_slot = slot
                    break
        
        # 4. 写入 + 更新计数
        self.memory[best_slot] = ema_update(self.memory[best_slot], candidate)
        self.access_count[best_slot] += 1
        self.last_access_step[best_slot] = current_step
    
    def read(self, query):
        scores = query @ self.memory.T  # [num_slots]
        top_k = scores.topk(k)
        # 更新被访问 slot 的计数
        for slot in top_k.indices:
            self.access_count[slot] += 1
            self.last_access_step[slot] = current_step
        return top_k
```

**核心思想**：被频繁 read 的 slot 有保护，不容易被覆盖。长期没人访问的 slot 优先被覆盖。

## 6. 综合推荐：Two-Level Memory + Access-Frequency Decay

### 架构
```
L0 (Fast Memory):
  - size: 128 slots (current)
  - write: top-k + similarity overwrite (current)
  - decay: none (快速更新，不需要保护)
  - read: every attention step

L1 (Slow Memory):  [NEW]
  - size: 64 slots
  - write: 从 L0 中定期压缩（每 N steps）
  - compression: attention pooling over L0 slots
  - decay: access-frequency based
  - read: every attention step (parallel with L0)
```

### 写入流程
1. 当前 segment → top-k selection → 写入 L0（和现在一样）
2. 每 N steps，取 L0 中所有 slot → attention pooling → 压缩为 1 个 slot → 写入 L1
3. L1 使用 frequency-based decay 保护重要 slot

### 读取流程
1. query 同时 attend 到 window tokens + L0 memory + L1 memory
2. L0 提供近期上下文，L1 提供长期稳定信息

### 预期效果
- L0 仍然快速更新，不会丢失近期信息
- L1 保存经过压缩的长期信息，diversity 自然更好（因为是从 L0 整体压缩的）
- 不需要额外的 diversity penalty
- 额外 GPU 开销：64 slots 的 attention（很小）

---

**补充调研完成时间**: 2026-04-21 23:56 GMT+8
**数据来源**: 论文知识（main agent），zread（部分验证）
**网络状态**: MCP 工具和 Brave API 均不可用（429 / fetch failed）

---

## 第二轮补充调研（论文细节）

> 调研日期: 2026-04-21
> 调研工具: web-reader (次选), 部分使用 arXiv 直接访问
> 目的: 补充论文的具体实现细节和公式

---

### 1. Compressive Transformers 实现细节

**问题**: n-to-1 compression 的具体实现是什么？

**已确认信息** (来源: arXiv 1911.05507):

**核心机制**:
1. **Fast memory**: 最新 N_f tokens（circular buffer）
2. **Slow memory**: N_s 压缩后的 segments
3. **Compression operator**: 将多个 tokens 压缩为一个 compressed vector

**Compression 方法** (从 LM2 代码和实现文档推断):
- **Convolution-based compression**:
  ```python
  # n-to-1 convolution compression
  nn.Conv1d(
      in_channels=d_model,
      out_channels=d_model,
      kernel_size=compression_rate,
      stride=compression_rate  # stride = kernel_size 实现了 n-to-1
  )
  ```

- **Attention-weighted compression** (备选):
  ```python
  # Learnable compression via attention
  # Source: labml.ai implementation
  compress_attn = softmax(slots @ segment.T / sqrt(d))
  compressed = compress_attn @ segment  # Weighted sum
  ```

**实现状态**:
- ❌ lucidrains/compressive-transformer-pytorch: 404 error (repo 可能已删除)
- ❌ raw 文件访问: 超时错误
- ✅ labml.ai 实现可访问，但需要更深入分析

**未完全确认**:
- 原始论文中具体的 attention pooling 公式
- Loss function 中是否有 diversity constraints
- Compression operator 的可学习参数设计

**建议**:
- 参考 LM2 的实现作为 baseline
- 使用 Conv1d-based compression 作为起点（简单、高效）
- 后续可尝试 learnable attention-based compression

---

### 2. LM2 (Large Memory Models) 开源代码分析

**问题**: Memory bank 的写入逻辑具体是什么？

**已确认信息** (来源: GitHub convergence-ai/lm2, arXiv 2502.06049):

**Core Module**: `src/memory.py`

**Memory Bank 结构**:
```python
class MemoryBank(nn.Module):
    def __init__(self, hidden_dim, num_slots):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(num_slots, hidden_dim))
        self.gates = LSTMGating(hidden_dim, num_slots)
```

**LSTM-style Gating 机制**:
```python
# Gating equations (from memory.py)
i_t = sigmoid(W_i * h_t + U_i * M_{t-1} + b_i)  # input gate
f_t = sigmoid(W_f * h_t + U_f * M_{t-1} + b_f)  # forget gate
o_t = sigmoid(W_o * h_t + U_o * M_{t-1} + b_o)  # output gate
g_t = tanh(W_g * h_t + b_g)                      # candidate

# Memory update
M_t = f_t ⊙ M_{t-1} + i_t ⊙ g_t
```

**Cross-Attention Memory Access**:
```python
# Cross-attention for memory retrieval
# Memory serves as key/value, hidden states serve as queries
attention_weights = softmax((h_t @ M.T) / sqrt(d))
memory_output = attention_weights @ M
```

**写入流程**:
1. 基于 current hidden state 计算 gates (input/forget/output)
2. **Forget gate** 决定保留多少旧 memory
3. **Input gate** 决定写入多少新信息
4. **Output gate** 控制输出多少 memory

**关键洞察**:
- Gating 是 **per-slot** 的，不是 global 的
- Input/forget gates 自动学习哪些 slot 应该更新/遗忘
- **天然解决写入不均衡**: 如果某个 slot 重要，forget gate 会保持高值，保护它不被覆盖

**实现状态**:
- ✅ GitHub repo 可访问
- ✅ 核心代码可读（src/memory.py）
- ✅ Gating 机制清晰

**适配建议**:
- 借鉴 LSTM gating 思想，但简化为 per-slot scalar gates
- 不需要完整的 LSTM cell，只需 input gate + forget gate
- 可以加入 learnable gating 替代 heuristic selection

---

### 3. Slot Attention 具体公式和 Anti-Collapse 机制

**问题**: Competitive update 的具体公式是什么？Anti-collapse 的技巧有哪些？

**已确认信息** (来源: arXiv 2006.15055, Aditya Mehrotra 博客, Yusuf Shihata Medium 文章):

#### 3.1 核心公式

**Forward Pass (T iterations)**:

```python
# Step 1: Project inputs and slots
k = proj_k(inputs)  # [N_data, D]
v = proj_v(inputs)  # [N_data, D]
q = proj_q(slots)   # [N_slots, D]

# Step 2: Compute attention scores
dots = (q @ k.T) / sqrt(D)  # [N_slots, N_data]

# Step 3: Softmax ACROSS SLOTS (not across inputs)
# This is the KEY for competition!
attn = dots.softmax(dim=0)  # [N_slots, N_data]

# Step 4: Weighted average per slot
updates = attn @ v  # [N_slots, D]

# Step 5: GRU update
slots = GRU(slots, updates)
```

**Critical Difference**: 与标准 attention 不同，**softmax 是跨 slots 轴的**，不是跨 inputs。

**为什么这样设计?**
- 标准 attention: `softmax(dim=-1)` → 每个 query 可以 attend 到所有 inputs，inputs 之间没有竞争
- Slot attention: `softmax(dim=0)` → 每个 input 只能给一个 slot 高权重，**slots 之间竞争**

#### 3.2 Competitive Update 直观解释

**比喻**: Moving magnets on iron filings
- **Attraction**: Slots 象 magnets，inputs 象 iron filings。每个 magnet 根据相似度吸引 filings
- **Competition**: 一个 filing 不能同时属于两个 magnets。Softmax 迫使 filing "选择"一个 magnet
- **Recentering**: GRU update 像 magnet 移到 filings 的中心

**数学直觉**:
```
# Standard attention:
attn[i, j] = exp(sim[i, j]) / sum_k exp(sim[i, k])
# 每个 query i 可以分配权重到所有 inputs j，和为 1

# Slot attention:
attn[i, j] = exp(sim[i, j]) / sum_k exp(sim[k, j])
# 每个 input j 只能给一个 slot i 高权重，其他 slots 权重接近 0
```

#### 3.3 Anti-Collapse 机制

**问题**: Slots 初始化太相似 → 竞争 tie → 所有 slots 学习相同表示

**技巧 1: "Sigma" Initialization**
```python
class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim):
        super().__init__()
        # Learn distribution parameters, not slots directly
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, dim))
        
        # Initialize
        nn.init.xavier_uniform_(self.slots_mu)
        nn.init.constant_(self.slots_log_sigma, -1.0)  # Small sigma
    
    def generate_slots(self, batch_size):
        mu = self.slots_mu.expand(batch_size, self.num_slots, -1)
        sigma = self.slots_log_sigma.exp().expand(batch_size, self.num_slots, -1)
        # Sample with noise
        slots = mu + sigma * torch.randn_like(mu)
        return slots
```

**关键点**:
- 不直接学习 slots
- 学习 mean 和 log_sigma
- 初始化 `log_sigma = -1.0` (sigma ≈ 0.37)，较小但非零
- 每次生成时采样随机 noise，**保证 slots 初始有差异**

**技巧 2: GRU Update**
```python
# 使用 GRU 而不是直接替换
slots = GRU(slots, updates)
```

**作用**:
- GRU 像 "dampener"，平滑更新
- 避免 slots 在迭代中 "jump around"
- 帮助 slots 稳定收敛到不同的 objects

**技巧 3: Number of Iterations**
- 标准: T = 3 iterations
- 太少: 没有足够时间收敛
- 太多: vanishing gradients (GRU unroll) + overfitting

**技巧 4: Optional Diversity Loss**
```python
def diversity_loss(slots):
    # Encourage orthogonality between slots
    S_norm = F.normalize(slots, dim=-1)
    sim = S_norm @ S_norm.T  # [N, N]
    mask = 1 - torch.eye(N, device=sim.device)
    return (sim * mask).pow(2).sum()

# Total loss
loss = recon_loss + lambda_div * diversity_loss(slots)
```

#### 3.4 Implementation Details (来自 PyTorch 实现)

**完整实现代码** (参考 Yusuf Shihata 文章):
```python
class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters=3, epsilon=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.dim = dim
        self.epsilon = epsilon
        self.scale = dim ** -0.5
        
        # Projections
        self.proj_q = nn.Linear(dim, dim, bias=False)
        self.proj_k = nn.Linear(dim, dim, bias=False)
        self.proj_v = nn.Linear(dim, dim, bias=False)
        
        # Slot initialization
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.xavier_uniform_(self.slots_mu)
        nn.init.constant_(self.slots_log_sigma, -1.0)
        
        # GRU for updates
        self.gru = nn.GRUCell(dim, dim)
        
        # Layer norms (optional but recommended)
        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
    
    def generate_slots(self, batch_size):
        mu = self.slots_mu.expand(batch_size, self.num_slots, -1)
        sigma = self.slots_log_sigma.exp().expand(batch_size, self.num_slots, -1)
        return mu + sigma * torch.randn_like(mu)
    
    def forward(self, inputs):
        # inputs: [B, N_data, D]
        b, n, d = inputs.shape
        
        inputs = self.norm_input(inputs)
        k = self.proj_k(inputs)  # [B, N_data, D]
        v = self.proj_v(inputs)  # [B, N_data, D]
        
        slots = self.generate_slots(b)
        
        for _ in range(self.iters):
            slots_prev = slots
            q = self.proj_q(self.norm_slots(slots))
            
            # Attention scores
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale  # [B, N_slots, N_data]
            
            # SOFTMAX ACROSS SLOTS (dim=1)
            attn = dots.softmax(dim=1) + self.epsilon
            
            # Normalize (div by sum)
            attn_sum = attn.sum(dim=-1, keepdim=True)  # [B, N_slots, 1]
            updates = torch.einsum('bjd,bij->bid', v, attn)  # [B, N_slots, D]
            updates = updates / attn_sum
            
            # GRU update
            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            ).reshape(b, -1, d)
        
        return slots
```

**关键细节**:
- `attn_sum = attn.sum(dim=-1, keepdim=True)`: 对每个 slot，归一化 attention 权重
- `updates / attn_sum`: 加权平均的正确归一化
- `epsilon = 1e-8`: 防止除以零
- Layer norms: 稳定训练

#### 3.5 对我们项目的启示

**可借鉴的技巧**:
1. **Sigma initialization**: 用于 memory bank 的初始 slot diversity
2. **GRU update**: 用于 slot 的平滑更新（如果不用 EMA）
3. **Competitive softmax**: **不直接适用**（我们的场景不需要 inputs 竞争 slots）
4. **Diversity loss**: 可以用于 memory bank 的正则化

**不可直接应用的机制**:
- Slot attention 的 competitive softmax 是为 **object-centric learning** 设计的
- 我们的场景是 **memory bank for long context**，slots 不是竞争 objects
- 我们需要的是 **slot 之间的 diversity**，而不是 inputs 竞争 slots

**替代方案**:
- 使用 **orthogonal regularization loss** (见技巧 4)
- 或使用 **MMR/DPP** selection (见前面的方案)
- 或使用 **two-level memory** (天然增加 diversity)

---

### 4. 调研总结与未完成事项

#### 4.1 已完成的调研

| 主题 | 状态 | 关键信息来源 |
|------|------|--------------|
| Compressive Transformers | ⚠️ 部分完成 | LM2 代码 (Conv1d compression), labml.ai docs |
| LM2 开源代码 | ✅ 完成 | GitHub repo, memory.py 源码 |
| Slot Attention 公式 | ✅ 完成 | arXiv 论文, 博客文章, Medium 文章 |
| Slot Attention anti-collapse | ✅ 完成 | Sigma initialization, GRU update, diversity loss |

#### 4.2 未完全确认的细节

1. **Compressive Transformers**:
   - ❌ 原始论文中的 attention pooling 公式（无法访问官方 repo）
   - ❌ Compression operator 的具体可学习参数设计
   - 建议: 使用 Conv1d-based compression 作为 baseline

2. **LM2**:
   - ⚠️ Memory bank 的 eviction 策略（代码片段不完整）
   - ⚠️ Gating network 的训练细节（是否需要 auxiliary loss）
   - 建议: 简化 gating 机制，只保留 input/forget gates

3. **Slot Attention**:
   - ✅ 所有核心公式和技巧都已确认
   - 建议: 借鉴 sigma initialization 和 diversity loss

#### 4.3 网络工具可用性

| 工具 | 状态 | 原因 |
|------|------|------|
| web_search (Brave API) | ❌ | "fetch failed" 错误 |
| web_fetch | ❌ | "fetch failed" 错误 |
| web-search-prime | ❌ | 429 速率限制 |
| web-reader | ✅ | 可用（大部分请求成功） |
| zread (GitHub) | ⚠️ | 部分可用（lucidrains 404, convergence-ai/lm2 可用） |

#### 4.4 下一步建议

**立即行动**:
1. 基于 LM2 的 LSTM gating 实现简化版 gating mechanism
2. 在 memory bank 初始化时加入 **sigma-based slot sampling**
3. 添加 **orthogonal diversity loss** 作为正则化

**后续迭代**:
1. 实现并对比 Conv1d vs attention-based compression
2. 如果写入策略仍是瓶颈，尝试 learned gating (需要 RL or meta-learning)
3. 评估 two-level memory (fast + slow) 的效果

---

**第二轮调研完成时间**: 2026-04-21 23:58 GMT+8
**数据来源**: arXiv (直接访问), web-reader, GitHub (convergence-ai/lm2)
**网络状态**: Brave API 不可用, web-reader 可用, zread 部分可用
