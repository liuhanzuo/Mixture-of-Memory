# Slot-Forward Memory 实现计划

## 背景

CrossAttentionMemoryV2 的 delta-rule write 产生统计平均而非精确存储。V3 4-arm 实验（400步）确认：0/203 NIAH 正确，ratio 在所有 arm 上退化至 1.0。

**用户指令**：把 memory slot 当成普通 hidden state 来 forward，不再使用单独的 cross-attention + delta-rule。

## 架构变更

**旧（delta-rule）**：
```
hidden → self_attn → hidden'
                      ↘ cross_attn_read(slots) → memory_output
hidden' + scale × memory_output → output
                      ↘ delta_rule_write → slot += lr * (avg_content - slot)
```

**新（slot-forward）**：
```
[slots, hidden] → decoder_layer(self_attn + FFN) → [new_slots, new_hidden]
                                                        ↓            ↓
                                                   store slots    output
```

**关键优势**：
- 写入 = self-attention（可选择性地关注特定 token 如 needle），不是加权平均
- 经过 FFN = 非线性特征提取
- 梯度直接通过 self-attention 流动，无 EMA 衰减
- **零新增参数** — 复用模型已有的 Q/K/V/O 和 FFN 权重

---

## 修改文件

只修改 `scripts/train_cross_attn_memory.py`。不修改 `selector.py`。

---

## 实现步骤

### Step 1: 添加 `--slot_forward` CLI 参数

在 `parse_args()` 中添加：
```python
parser.add_argument("--slot_forward", action="store_true", default=False,
                    help="Forward memory slots through decoder layers like regular tokens")
```

在 `main()` 中构造模型时传入：`slot_forward=args.slot_forward`。

### Step 2: 修改 `CrossAttentionMemoryModel.__init__()`

添加参数 `slot_forward: bool = False`，存为 `self.slot_forward`。

当 `slot_forward=True`：
- **不创建** `cross_attn_modules`（不需要 CrossAttentionMemoryV2）
- 仍然创建 `self.slot_values: list[torch.Tensor | None] = [None] * self.num_layers`
- 也创建 `self.slot_keys: list[torch.Tensor | None] = [None] * self.num_layers`（保持 reset_slots 兼容，但不会被使用）
- 添加 `self._ext_attn_mask_cache = None`

### Step 3: 添加 `_build_extended_attn_mask()` 方法

```python
def _build_extended_attn_mask(self, S, T, dtype, device, batch_size):
    """构建 [B, 1, S+T, S+T] 加性注意力掩码。
    - Slots（行 0..S-1）：attend to 一切（全零）
    - Tokens（行 S..S+T-1）：attend to 所有 slots + tokens 间 causal mask
    """
    # 使用缓存
    if (self._ext_attn_mask_cache is not None
        and self._ext_attn_mask_cache.shape[-1] == S + T):
        return self._ext_attn_mask_cache

    L = S + T
    mask = torch.zeros(L, L, dtype=dtype, device=device)
    neg_inf = torch.finfo(dtype).min
    causal = torch.triu(
        torch.full((T, T), neg_inf, dtype=dtype, device=device),
        diagonal=1,
    )
    mask[S:, S:] = causal
    # Slot 行和 Slot 列已经是 0（allowed）
    result = mask.view(1, 1, L, L).expand(batch_size, 1, L, L).contiguous()
    self._ext_attn_mask_cache = result
    return result
```

参考：`src/memory/mem_space/layer.py` 第 99-160 行的 `_build_extended_attn_mask()`。

### Step 4: 添加 `_extend_position_embeddings()` 方法

```python
def _extend_position_embeddings(self, position_embeddings, S):
    """在 RoPE cos/sin 表前添加 S 个 position-0 条目。
    Position 0: cos=1, sin=0 → 不旋转，slots 位置无关。
    """
    cos, sin = position_embeddings  # 各 [B, T, head_dim]
    cos0 = cos[:, :1, :]
    sin0 = sin[:, :1, :]
    cos_ext = torch.cat(
        [cos0.expand(cos.shape[0], S, cos.shape[-1]), cos], dim=1
    )
    sin_ext = torch.cat(
        [sin0.expand(sin.shape[0], S, sin.shape[-1]), sin], dim=1
    )
    return cos_ext, sin_ext
```

参考：`src/memory/mem_space/layer.py` 第 163-184 行的 `_extend_position_embeddings()`。

### Step 5: 添加 `_forward_slot_forward()` 方法 — 核心变更

```python
def _forward_slot_forward(self, input_ids, labels=None, enable_write_grad=False):
    """Slot-forward 模式：slots 与 hidden_states 一起通过 decoder layer。"""
    B, T = input_ids.shape
    device = input_ids.device
    dtype = next(self.parameters()).dtype
    S = self.num_slots

    embed_tokens = self._get_embed_tokens()
    hidden_states = embed_tokens(input_ids).to(dtype)

    position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
    rotary_emb = self._get_rotary_emb()
    position_embeddings = rotary_emb(hidden_states, position_ids)

    # 扩展位置编码（slots 用 position 0）
    ext_pos_emb = self._extend_position_embeddings(position_embeddings, S)

    # 构建扩展注意力掩码（带缓存）
    ext_attn_mask = self._build_extended_attn_mask(S, T, dtype, device, B)

    for layer_idx, layer in enumerate(self._decoder_layers):
        self._init_slots(layer_idx, hidden_states)
        slots = self.slot_values[layer_idx]  # [B, S, d_model]

        # Prepend slots 到 hidden_states
        extended = torch.cat([slots, hidden_states], dim=1)  # [B, S+T, d_model]

        # 通过 decoder layer（self_attn + FFN）
        layer_out = layer(
            extended,
            attention_mask=ext_attn_mask,
            position_ids=None,
            past_key_value=None,
            use_cache=False,
            position_embeddings=ext_pos_emb,
        )
        output = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        # 拆分输出
        new_slots = output[:, :S, :]
        hidden_states = output[:, S:, :]

        # 更新 slot 状态
        if enable_write_grad:
            self.slot_values[layer_idx] = new_slots
        else:
            self.slot_values[layer_idx] = new_slots.detach()

    # 最终 norm + LM head（只对 token 部分，不包括 slots）
    norm = self._get_norm()
    lm_head = self._get_lm_head()
    hidden_states = norm(hidden_states)
    logits = lm_head(hidden_states)

    result = {"logits": logits}
    if labels is not None:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fn = nn.CrossEntropyLoss(reduction="mean")
        loss = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        result["loss"] = loss
    return result
```

### Step 6: 修改 `forward_chunk()` 路由

在 `forward_chunk()` 中，`if not self.use_memory` 检查之后、现有 cross-attention 代码之前，添加：

```python
if getattr(self, 'slot_forward', False):
    return self._forward_slot_forward(input_ids, labels, enable_write_grad)
```

位置：大约在第 482 行 `if not self.use_memory or not self.use_cross_attn_memory:` 之后。

### Step 7: 修改 `_init_slots()` 支持 slot_forward

```python
def _init_slots(self, layer_idx, hidden_states):
    B, T, D = hidden_states.shape
    device = hidden_states.device
    dtype = hidden_states.dtype

    if self.slot_values[layer_idx] is None or self.slot_values[layer_idx].shape[0] != B:
        stride = max(1, T // self.num_slots)
        indices = torch.arange(0, T, stride)[:self.num_slots]
        if len(indices) < self.num_slots:
            pad_indices = indices[-1:].expand(self.num_slots - len(indices))
            indices = torch.cat([indices, pad_indices])

        sampled = hidden_states[:, indices, :].detach()
        noise = torch.randn_like(sampled) * 0.02

        if getattr(self, 'slot_forward', False):
            # slot_forward: 只需要 slot_values（用 sampled+noise 使 slots 多样化）
            self.slot_values[layer_idx] = (sampled + noise).clone()
        else:
            # 原始模式: keys 和 values 分开
            self.slot_keys[layer_idx] = (sampled + noise).clone()
            self.slot_values[layer_idx] = sampled.clone()
```

### Step 8: 修改 optimizer 设置

找到创建 optimizer 参数组的地方（搜索 `cross_attn_lr_factor` 或 `cross_attn_params`）。当 `slot_forward=True` 时，没有 cross_attn 参数：

```python
if not getattr(model, 'slot_forward', False):
    # 现有 cross_attn 参数组代码
    ...
else:
    # slot_forward: 只有 base model 参数
    decay = [p for p in model.base_model.parameters()
             if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.base_model.parameters()
                if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": args.weight_decay, "lr": args.lr},
        {"params": no_decay, "weight_decay": 0.0, "lr": args.lr},
    ])
```

### Step 9: 简化 NIAH 训练路径

在 `forward_niah_sample()` 中，当 `slot_forward=True`：
- 移除 contrastive loss（self-attention 自然处理检索）
- 保留 NIAH 准确率追踪
- 保留 streaming chunks + CE loss

```python
is_slot_forward = getattr(root_model, 'slot_forward', False)
```

在 contrastive loss 部分加条件：
```python
if not is_slot_forward:
    # 现有 contrastive loss 代码
    ...
```

---

## 关键参考文件

| 文件 | 用途 |
|------|------|
| `scripts/train_cross_attn_memory.py` | **唯一要修改的文件** |
| `src/memory/mem_space/layer.py` 99-184行 | 掩码和位置编码扩展的参考实现 |
| `src/memory/mem_space/selector.py` 714-887行 | 现有 delta-rule 实现（只读参考） |

## 约束

1. **不修改** `selector.py` — CrossAttentionMemoryV2 保持不变
2. **不删除** 现有代码 — 只添加新路径和条件分支
3. **不改变** 任何现有 CLI 参数的默认值
4. 现有 cross-attention 路径在 `--slot_forward` 未指定时必须正常工作

## 验证

1. **Step-0 一致性**：训练前 `evaluate_memory_ppl` — ratio < 1.02（64 额外 token = 1.5% 稀释）
2. **梯度流**：1 步 backward 后 `decoder_layer.self_attn.q_proj.weight.grad` 非零
3. **NIAH smoke test**：100 步 `niah_mix_fraction=0.3` — accuracy > 0%（对比 delta-rule 的 0%）
4. **LM 回归**：500 步 Dolmino — vanilla PPL 不超过基线 5%
5. **无 NaN/OOM**：先单 GPU 验证，再 8-GPU DDP

## 风险：Step-0 注意力稀释

64 slots + 4096 tokens = 4160。Self-attention softmax 分母多 64 项 → 1.5% 稀释/层。32 层累积 worst-case `(1-64/4160)^32 ≈ 0.61`。

**缓解**：如果 step-0 PPL ratio > 1.02，添加 bypass gate（类似 MemorySpaceLayer 的 tanh(alpha)，init=0）：
```python
bypass = layer(hidden_states, ...)                    # 原始 forward
ext = layer([slots, hidden_states], ...)              # 扩展 forward
hidden = bypass + alpha * (ext[:, S:] - bypass)       # init: alpha=0 → 纯 bypass
```
这会让计算量翻倍但保证 step-0 一致性。

## 预计工作量

- ~120 行新代码 + ~10 行修改
- 只改 `scripts/train_cross_attn_memory.py` 一个文件
- 实现 1-2 天 + 测试 1 天
