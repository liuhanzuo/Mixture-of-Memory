# Version 4: Per-Layer Sparse Memory Bank (2026-05-02)

> 设计者：用户 + Claude
> 核心思想：每层维护一个固定大小的 memory bank，用 chunk 末尾 hidden state 填充；
> 填满后用 top-k 相关性选择性更新；模型通过训练学会读取这些 slot。

---

## 1. 核心设计

### 1.1 整体结构

```
每一层 Transformer Layer k 有一个独立的 MemoryBank：
  bank_k: [B, N, d_k]   (N = num_slots, d_k = hidden dim at layer k)

处理序列时，切分为若干 chunk（每个 4096 token）：

Chunk 1 处理：
  Layer k input:   [emb(t_1..t_4096)]          # 无 memory prefix（bank 为空）
  Layer k output:  hidden_k                      # [B, 4096, d_k]
  Bank update:     bank_k[0] = hidden_k[:, -1, :]  # 末尾 token hidden 写入 slot 0

Chunk 2 处理：
  Layer k input:   [bank_k[0] | hidden_{k-1}(chunk_2)]   # 1 个 slot prefix
  Layer k output:  hidden_k                                # [B, 1+4096, d_k]
  Bank update:     bank_k[1] = hidden_k[:, -1, :]         # 写入 slot 1

...

Chunk N 处理（bank 已填满）：
  top-k 选择：从 bank_k 选出与当前 hidden 最相关的 k 个 slot
  Layer k input:   [top-k slots | hidden_{k-1}(chunk_N)]
  Layer k output:  hidden_k
  Bank update:     被选中的 k 个 slot → 用 hidden_k[:, -1, :] 更新（EMA）
                   未被选中的 (N-k) 个 slot → 保持不变（仍携带旧 chunk 信息）
```

### 1.2 关键设计决策

**决策 A：Memory 不参与反传，模型参与反传**

```
forward(chunk_i):
    slots = bank_k.read_top_k(hidden, k)     # detach()，不参与 grad
    extended = cat([slots, hidden])
    out = decoder_layer(extended)             # 模型学会如何使用 slots
    bank_k.write(selected_idx, out[:, -1, :])# forward-only state update

backward():
    grad 只流向 decoder_layer 的权重
    grad 不流向 bank_k（bank 是纯运行时状态）
```

**原因**：
- Slot 是 running state（类似 Mamba 的 state），不是参数
- 模型需要学的是"如何处理 prefix slot token"，而不是"如何生成好的 slot"
- 这天然解决了 BPTT 截断问题：不需要跨 chunk 反传

**决策 B：每层独立的 memory bank**

- Layer k 的 slot 来自 layer k 的 hidden，语义完全对齐
- Layer_1 的 slot 是 embedding-space 附近的向量，Layer_32 的 slot 是 logit-space 附近的向量
- 各层各自使用自己语义空间里的 slot，无 OOD 问题

**决策 C：两阶段策略**

- Phase 1（未填满）：直接 append，slot[i] = last_hidden(chunk_i, layer_k)
- Phase 2（已填满）：top-k 选择 + sparse update

Phase 1 无需 top-k 路由，完全确定性，不会有路由退化问题。
Phase 2 的 top-k 只做 selection，不参与 grad，路由退化无影响（不会导致 loss 退化）。

**决策 D：Memory slots 不经过 RoPE**

Memory slot 的位置是"历史时间"的抽象，没有一个合理的 absolute position 可以分配。
最干净的方案：slot 对应的 attention 不施加 RoPE（position-free prefix）。

**决策 E：信息保留的 sparse update**

未被 top-k 选中的 slot 不更新。chunk_1 的信息可以通过"一直未被覆盖的 slot"保留到 chunk_100。
这不是 FIFO，而是 usage-based retention。

---

## 2. 完整前向传播

### 2.1 Per-Layer Forward Pass

```python
class MemoryAugmentedDecoderLayer(nn.Module):
    """
    包裹预训练的 LlamaDecoderLayer，添加 per-layer memory bank。
    backbone 权重 frozen，只训练 layer（通过 LoRA 或 full fine-tune）。
    """

    def __init__(self, base_layer, num_slots=64, k=8):
        self.base_layer = base_layer   # 预训练 LlamaDecoderLayer（frozen）
        self.num_slots = num_slots     # memory bank 容量
        self.k = k                     # top-k 选取数

    def forward(
        self,
        hidden_states: torch.Tensor,    # [B, T, d]（当前 chunk 的 hidden input）
        bank: MemoryBank,               # 该层的 memory bank（运行时状态）
        position_ids: torch.Tensor,     # [1, T]（只给 token 部分）
        attention_mask: torch.Tensor,   # 会被扩展为 prefix causal mask
    ) -> torch.Tensor:

        B, T, d = hidden_states.shape
        n_filled = bank.num_filled      # 当前已填充的 slot 数

        if n_filled == 0:
            # Phase 1 & Phase 2 开始：无 memory prefix，正常 forward
            out = self.base_layer(
                hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )
            last_hidden = out[:, -1, :].detach()  # [B, d]，detach：不参与反传
            bank.append(last_hidden)
            return out

        # Phase 1（未填满）或 Phase 2（已填满）：有 memory prefix
        if n_filled < self.num_slots:
            # 未填满：用所有已有 slot
            slots = bank.get_all()         # [B, n_filled, d]，detach
        else:
            # 已填满：top-k 选择最相关的 k 个 slot
            query = hidden_states.mean(dim=1)  # [B, d]，用 chunk 均值做 query
            slots, selected_idx = bank.top_k(query, self.k)
            # slots: [B, k, d]，detach

        n_slots = slots.shape[1]

        # 构造扩展序列：[slots | tokens]
        extended = torch.cat([slots, hidden_states], dim=1)  # [B, n_slots+T, d]

        # 构造 position IDs：slots 不参与 RoPE（position = 0）
        slot_pos = torch.zeros(1, n_slots, dtype=torch.long, device=hidden_states.device)
        token_pos = position_ids                                # [1, T]
        ext_pos = torch.cat([slot_pos, token_pos], dim=1)      # [1, n_slots+T]

        # 构造 attention mask（prefix causal）
        ext_mask = make_prefix_causal_mask(n_slots, T, hidden_states.device)

        # Forward（模型接收 slot prefix + token）
        ext_out = self.base_layer(
            extended,
            position_ids=ext_pos,
            attention_mask=ext_mask,
        )                              # [B, n_slots+T, d]

        # 只取 token 部分的输出
        out = ext_out[:, n_slots:, :]  # [B, T, d]

        # Memory update（forward-only，不参与 grad）
        with torch.no_grad():
            last_hidden = out[:, -1, :]  # [B, d]
            if n_filled < self.num_slots:
                bank.append(last_hidden)
            else:
                bank.update_selected(selected_idx, last_hidden)

        return out
```

### 2.2 MemoryBank 实现

```python
class MemoryBank:
    """
    Per-layer, per-batch 的 memory bank。
    纯运行时状态，不是 nn.Module（无参数）。
    """

    def __init__(self, num_slots: int, d_model: int, ema_decay: float = 0.9):
        self.num_slots = num_slots
        self.d_model = d_model
        self.ema_decay = ema_decay
        self.slots = None       # [B, N, d]，lazy 初始化
        self.num_filled = 0

    def append(self, hidden: torch.Tensor):
        """Phase 1：追加新 slot（未填满时）"""
        # hidden: [B, d]
        if self.slots is None:
            B, d = hidden.shape
            self.slots = torch.zeros(B, self.num_slots, d,
                                     dtype=hidden.dtype, device=hidden.device)
        self.slots[:, self.num_filled, :] = hidden.detach()
        self.num_filled += 1

    def get_all(self) -> torch.Tensor:
        """返回所有已填充的 slot，detached"""
        return self.slots[:, :self.num_filled, :].detach()

    def top_k(self, query: torch.Tensor, k: int):
        """
        query: [B, d]
        返回: (selected_slots [B, k, d], selected_idx [B, k])
        全部 detach，不参与 grad
        """
        slots = self.slots.detach()  # [B, N, d]
        # cosine similarity
        q_norm = F.normalize(query, dim=-1).unsqueeze(1)    # [B, 1, d]
        s_norm = F.normalize(slots, dim=-1)                  # [B, N, d]
        scores = torch.bmm(q_norm, s_norm.transpose(1, 2)).squeeze(1)  # [B, N]
        _, idx = scores.topk(k, dim=-1)                      # [B, k]
        selected = slots.gather(1, idx.unsqueeze(-1).expand(-1, -1, self.d_model))
        return selected, idx  # both detached

    def update_selected(self, idx: torch.Tensor, new_hidden: torch.Tensor):
        """
        EMA 更新被选中的 k 个 slot。
        idx: [B, k]
        new_hidden: [B, d]
        """
        with torch.no_grad():
            # 每个被选中的 slot 都用 new_hidden 做 EMA 更新
            # 注意：同一个 chunk 末尾只有 1 个 hidden，
            # 但 k 个 slot 被选中——都用同一个 last_hidden 更新
            # （若需要更精细的 per-slot update，可改为 per-position hidden）
            B, k = idx.shape
            new_h = new_hidden.unsqueeze(1).expand(-1, k, -1)  # [B, k, d]
            current = self.slots.gather(1, idx.unsqueeze(-1).expand(-1, -1, self.d_model))
            updated = self.ema_decay * current + (1 - self.ema_decay) * new_h
            self.slots.scatter_(1, idx.unsqueeze(-1).expand(-1, -1, self.d_model), updated)

    def reset(self):
        """新文档开始时重置 bank"""
        self.slots = None
        self.num_filled = 0
```

### 2.3 Prefix Causal Mask 构造

```python
def make_prefix_causal_mask(
    n_slots: int,
    n_tokens: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    构造 [1, 1, n_slots+n_tokens, n_slots+n_tokens] 的 additive attention mask。

    规则：
    - slot → slot：可见（值 = 0，不屏蔽）
    - slot → token：不可见（值 = -inf，slots 不 attend 到未来 token）
    - token → slot：可见（值 = 0，所有 token 都能看到所有 slot）
    - token → token：causal（下三角 = 0，上三角 = -inf）

    注：RoPE 对 slot 位置（pos=0）不施加有效 bias，
    因为 slot 的 position_ids 全部为 0，cos/sin(0) = 1/0，
    等价于不旋转（Q @ K 就是普通点积，无相对位置编码）。
    """
    N = n_slots + n_tokens
    mask = torch.zeros(N, N, device=device, dtype=dtype)

    # slot → token：屏蔽（右上角的 slot 行 × token 列）
    mask[:n_slots, n_slots:] = float('-inf')

    # token → token：causal（右上角的 token × token 块）
    token_block = torch.triu(
        torch.full((n_tokens, n_tokens), float('-inf'), device=device), diagonal=1
    )
    mask[n_slots:, n_slots:] = token_block

    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]
```

---

## 3. 关键问题分析

### 3.1 信息保留：chunk_1 的信息如何存活到 chunk_100？

```
初始状态（N=8 slots，k=4）：

Chunk 1 → slots [0..7] 全部填满，slot_i = last_hidden(chunk_1, layer_k)

Chunk 2 → top-4 选中 slots {0, 2, 5, 7}
           slots {0, 2, 5, 7} 被 EMA 更新（混入 chunk_2 信息）
           slots {1, 3, 4, 6} 保持 chunk_1 的原始值 ← 仍携带 chunk_1 信息

Chunk 3 → top-4 选中 slots {1, 3, 6, 7}
           slots {1, 3, 6, 7} 被更新
           slots {0, 2, 4, 5} 保持（其中 0,2,5 = chunk_2 混合值，4 = chunk_1 原始值）

...经过多个 chunk 后：
   每个 slot 都是"从某时刻到最近一次被选中"的 EMA 摘要
   从未被选中的 slot → 保持最初写入时的值（chunk_1 原始值永远不消失，除非被全部选中）
```

这是一个自然的 **基于相关性的记忆保留机制**：
- 与当前内容相关的历史信息会被持续更新（保持"鲜活"）
- 与当前内容不相关的历史信息会静止保留（"休眠"但不消失）

### 3.2 Top-k 路由的退化风险

**与 v1 的根本区别**：v1 的路由退化会导致 loss 退化（因为路由连接着梯度）。
本方案的路由**不参与 grad**，路由退化只影响"哪些 slot 被更新"，不影响模型的 LM loss。

即使 top-k 退化成随机选择，模型仍然能够正常训练（只是 slot 的内容质量下降）。
路由质量是渐进优化的问题，不是训练稳定性问题。

**潜在风险**：如果 top-k 总是选同一组 slot，其余 slot 永远不更新，
变成了 "静止 slot"（carry chunk_1 信息但永远不更新）。
在 Phase 1 填满时这是好事；但如果模型后来想看这些静止 slot，它们的信息已经过时。

**缓解**：加入轻微的随机探索（ε-greedy）或者定期强制更新所有 slot：
```python
# ε-greedy top-k
if random.random() < epsilon:   # epsilon = 0.05
    idx = random_k_slots()      # 随机选 k 个
else:
    idx = cosine_top_k()        # 正常 top-k
```

### 3.3 RoPE 位置编码

**当 slot position_ids = 0 时发生了什么？**

```
RoPE 旋转：q_rotated = q * cos(θ * pos) + q_perp * sin(θ * pos)
当 pos = 0：cos(0) = 1, sin(0) = 0
            q_rotated = q * 1 + q_perp * 0 = q（不旋转）

因此：slot 的 Q/K 不会被位置编码旋转
     slot-token 的注意力 = Q_token @ K_slot^T（纯语义相似度，无 position bias）
     token-slot 的注意力 = Q_slot @ K_token^T（但 slot attend to token 被 mask 掉了）
```

**等效效果**：slot 对所有 token 的 relative position distance = 0，
即所有 token 都认为 slot "在位置 0"。
这在 预训练 LLM 里，position=0 通常是 BOS token 的位置，有固定的 attention 模式。

**更好的替代方案（可选）**：使用 ALiBi 风格的 position-free attention for slots：
在 FlashAttention 的 `custom_mask` 参数里，slot 对应的 position distance 直接填 0，
不经过任何 positional bias（不依赖 pos=0 的 pretrained behavior）。

### 3.4 Layer 语义对齐

**当前设计下每层的语义层次：**

```
Bank_1: embedding-space hidden（接近 token embedding，语法+形态信息）
Bank_8: 浅层语义（句法 + 部分实体信息）
Bank_16: 中层语义（句义 + 上下文依赖）
Bank_24: 深层语义（推理 + 长程依赖）
Bank_32: logit-space（预测下一个词的表示）
```

各层 slot 都在自己的语义空间里，memory-augmented attention 不会产生 OOD 输入。
这是本方案相比 RMT v8 的核心优势。

### 3.5 Last Token vs Pooling

**Last token 的问题**：末尾 token（如标点、停用词）的 hidden 偏向 "预测下一词"，
而非 "总结本 chunk"。对于 `...这个问题很复杂。[END_CHUNK]`，
last hidden ≈ representation for predicting `下面` or `因此` etc.

**Pooling 的问题**：mean pooling 会均匀混合所有 token，包括大量功能词。

**折中方案（Phase 1 可用）**：用 last token 是可以接受的临时方案，
因为 last token 通过 attention 看过了整个 chunk，信息量并不比 pooling 差很多。
更精细的方案（attention pooling）可以作为 Phase 2 的改进。

---

## 4. 与之前失败方案的对比

### 4.1 解决了 v1 的哪些问题

| v1 问题 | 解决方式 |
|---|---|
| top1_sim 路由退化到 1/N | 路由不参与 grad，退化不影响 loss |
| bf16 slot key 不可分辨 | slot 内容来自真实 hidden，量级正常 |
| slot_output_gate=0 代数消零 | 无 gate（slot 直接 prepend，模型 attend） |
| VQ-EMA 编码簿崩溃 | 无 VQ，用 EMA 直接更新 hidden |
| 梯度链条上的多个 detach | bank 完全 detach，模型梯度无干扰 |
| M_sel_hidden 范数爆炸 | slot 来自 last_hidden，量级 = hidden 量级 |

### 4.2 解决了 RMT v8 的哪些问题

| RMT v8 问题 | 解决方式 |
|---|---|
| Layer_32 hidden 注入 layer_1（OOD） | Per-layer bank，各层在自己语义空间 |
| Repetitive generation | 无 OOD token → 无 attention sink → 无重复模式 |
| memory_gate init = 2.0（影响太大） | 模型通过 LoRA fine-tune 学习如何 attend to slots |
| 无法保留早期 chunk 信息 | Sparse update：未选中 slot 永久保留 |

---

## 5. 训练设计

### 5.1 训练目标

```
模型训练目标：学会"从 slot prefix 中提取有用信息"

具体来说：
- 模型接收 [slots | tokens] 序列
- 计算正常 NTP loss（只在 token 部分）
- 梯度只流向 backbone 的 LoRA weights（slot bank 不参与）
- 训练数据：长文档切块（pg19、书籍语料），2-8 个 chunk 拼接
```

### 5.2 训练阶段

**Phase 0（验证冷启动，100 steps）**：
- 只用 Phase 1（未填满，slot append 模式）
- slot = last_hidden.detach()（正确实现）
- 期望：PPL ≤ vanilla PPL × 1.05（slot 是 detach 的 hidden，不污染 LM）

**Phase 1（单文档训练，1000 steps）**：
- 用完整 8 chunk 序列（4096 × 8 = 32K tokens per sample）
- Phase 1（前 8 chunk 填满 bank）→ Phase 2（后续 chunk 用 top-k）
- 期望：8 chunk 的 PPL < 1 chunk 的 PPL（说明 slot 提供了有效信息）

**Phase 2（长文档评估）**：
- pg19：16-32 chunk 串联评估
- NIAH：needle 在前半段，question 在后半段
- 期望：slot 数量越多，长文档 PPL 下降越明显

### 5.3 超参数建议

| 参数 | 推荐值 | 理由 |
|---|---|---|
| num_slots N | 64 | 64 chunks × 4096 = 262K token 覆盖范围 |
| top-k K | 8 | N/8 的选取比例，足够信息密度 |
| chunk_size | 4096 | 与 LLM 预训练 context length 对齐 |
| ema_decay | 0.9 | slot 更新较慢，保留历史信息 |
| epsilon (greedy) | 0.05 | 5% 随机探索，防止 slot 完全静止 |
| LoRA rank | 16 | backbone 微调，使模型适应 slot prefix |
| lr (LoRA) | 1e-4 | 标准 LoRA 学习率 |
| batch_size | 4 | 4 × 8 chunks = 4 × 32K = 128K tokens/batch |

---

## 6. 实现优先级

### Step 1（最小可行原型，~200 行）
- [ ] `MemoryBank`：append / get_all / top_k / update_selected / reset
- [ ] `make_prefix_causal_mask`：prefix causal mask 构造
- [ ] Patch `LlamaDecoderLayer.forward`：注入 bank prefix
- [ ] 训练脚本：2-chunk pair training（Phase 1 only，验证 PPL 不退化）

**Go/No-go**：PPL ≤ vanilla × 1.05 at step 50

### Step 2（Phase 2 验证，~100 行）
- [ ] top_k 选择路径
- [ ] update_selected EMA 更新
- [ ] ε-greedy 探索
- [ ] 8-chunk 串联训练
- [ ] PPL per-chunk 曲线（应随 chunk 序号下降）

**Go/No-go**：chunk_8 的 PPL < chunk_1 的 PPL（slot 提供了有用信息）

### Step 3（长程 + NIAH 评估）
- [ ] 32-chunk 串联评估
- [ ] NIAH accuracy vs. needle distance
- [ ] 与 Infini-attention v3 对比

---

## 7. Known Issues（当前版本）

1. **Last token 质量**：最后一个 token 的 hidden 不是最优的 chunk 摘要。Phase 1 先用，后续考虑 attention pooling compressor。

2. **Per-layer bank 通信开销**：DDP 训练时，bank 是 per-sample 状态，需要确保相同 sample 在不同 chunk 上分配到同一个 GPU（不能跨 chunk shuffle batch）。

3. **Generation 时 bank 冻结**：推理时 memory bank 应该只读，禁止 write，防止 slot 被 generation 的 output 污染（这是 RMT v3-v10 inference 退化的根本原因之一）。

4. **Attention mask 兼容性**：`make_prefix_causal_mask` 需要和 LLM 的 Flash Attention 实现兼容（SDPA 的 `attn_mask` 参数 dtype 必须是 float，不能是 bool）。

5. **Slot position=0 与 BOS token 冲突**：如果序列本身有 BOS token（position=0），slot 和 BOS 会共用同一个 RoPE 旋转值。可以把 slot position 设为一个固定的大值（如 position=8192），或者改用 NoPE（直接不做 RoPE）。

---

## 8. 与相关工作的关系

| 工作 | 相似点 | 关键区别 |
|---|---|---|
| TransformerXL (Dai et al., 2019) | Per-layer cache，语义对齐 | XL 保留全部 KV，本方案固定 N slots + sparse update |
| RMT (Bulatov et al., 2022) | Memory token prepending | RMT 用 learnable memory tokens，本方案用 real hidden states |
| MemoryLLM (Wang et al., 2024) | Hidden state injection | MemoryLLM 是 dense injection，本方案是 sparse top-k selection |
| Infini-attention (Munkhdalai, 2024) | Compressive memory | Infini 是 continuous delta rule，本方案是 discrete chunk-level slots |
| H3 / Mamba | State-based sequential memory | Mamba 的 state 是 token-level，本方案是 chunk-level 摘要 |
