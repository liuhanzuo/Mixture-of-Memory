# Mixture-of-Memory — 当前算法说明 (Memory-Space v0)

> 本文件用于回答："我们现在用的是 joint attention 吗？算法到底长什么样？"
>
> 答：**是的**。我们当前的核心机制就是 **per-layer memory bank + Flamingo 风格的
> joint self-attention（KV-prepend）+ EMA 写回**，并由 chunked streaming 在
> 推理时把超长上下文切成 ≤4k 的小片喂进去。
>
> 主要源码：
> - `src/memory/mem_space/layer.py:411` — `MemorySpaceLayer.forward`（核心）
> - `src/memory/mem_space/memory_bank.py` — 记忆槽存储 / 写回
> - `src/memory/mem_space/selector.py` — Top-k 路由
> - `src/memory/mem_space/patch.py` — 替换 `LlamaDecoderLayer`
> - `scripts/run_babilong_mem_space.py` — chunked 推理入口

---

## 1. 一图概览

```
                                输入 hidden_states  H_l ∈ [B, T, d]
                                          │
                ┌─────────────────────────┼──────────────────────────┐
                │                         │                          │
                ▼                         ▼                          │
         (1) lazy-init              (2) Top-k 选 slot                │
         memory_bank (N=512)        idx ∈ [B, k=64]                  │
                │                         │                          │
                ▼                         ▼                          │
         slots ∈ [B, N, slot_dim]   M_sel = slots[idx]               │
                                          │                          │
                                          ▼                          │
                                   slot_to_hidden                    │
                                   M_sel_hidden ∈ [B, k, d]          │
                                          │                          │
                                          ▼                          │
                          (3) 拼接 [M_sel_hidden ; H_l]              │
                                  RoPE: slot 全部用 pos=0            │
                                  Mask: slot 行/列全可见, H 块因果   │
                                          │                          │
                                          ▼                          │
                          (4) 同一个 LlamaDecoderLayer 跑两遍:       │
                              ext_h   = layer( [M ; H], ext_mask )   │
                              bypass  = layer( H, causal )  ◀────────┘
                                          │
                                          ▼
                       (5) Flamingo 门控融合:
                           α = tanh(slot_output_gate)   (init=0.5)
                           slot_delta = ext_h[k:] − bypass
                           next_h     = bypass + α · slot_delta
                                          │
                                          ▼
                       (6) EMA 写回 (gradient-bearing):
                           O_mem_slot = hidden_to_slot(ext_h[:k])
                           slots[idx] ← (1−β)·slots[idx] + β·O_mem_slot
                                          │
                                          ▼
                                  next layer / next chunk
```

---

## 2. 核心机制：Joint Self-Attention via KV-Prepend

我们 **没有** 引入新的 cross-attention 模块；我们复用 Llama 自带的
`LlamaDecoderLayer`（self-attention），只是把 sequence 在前面拼上 k 个
"memory tokens"，让它们在同一个 softmax 里被一起 attend 到。

具体实现 (`layer.py:567-641`):

1. 选出来的 slot 通过 `slot_to_hidden`（一个 `nn.Linear(slot_dim, d_model)`）
   投影到 hidden 空间，得到 `M_sel_hidden ∈ [B, k, d]`。
2. 拼接成 extended sequence：
   ```
   extended_hidden = concat([M_sel_hidden, hidden_states], dim=1)   # [B, k+T, d]
   ```
3. 构造一个 4-D additive attention mask（`_build_extended_attn_mask`）：
   - **slot 行（前 k 行）**：可以看到所有位置（全 0），slot 是 *global tokens*。
   - **H 行（后 T 行）**：可以看到全部 slot 列；H 之间是因果。
4. 构造 extended `position_embeddings`：所有 slot 的 RoPE 位置 = 0
   （`_extend_position_embeddings` line 163），让 slot 成为 "position-less
   memory tokens"，避免跟当前 chunk 的位置冲突。
5. 对 **同一层的同一组权重** 调两次：
   - `bypass_h = wrapped_layer(H, ..., is_causal)` —— 纯 vanilla 路径
   - `ext_h   = wrapped_layer([M;H], ext_mask, ext_pos_emb, ...)` —— 带记忆
6. **Flamingo 门控融合**（这是关键的 "可逆退化" 设计）：
   ```python
   α = tanh(slot_output_gate)            # init=0.5 → α≈0.462
   slot_delta = ext_h[:, k:, :] - bypass_h
   next_hidden = bypass_h + α · slot_delta
   ```
   - 如果 `α=0`，输出严格等价 vanilla，记忆通路被完全旁路（debug / ablation 可控）。
   - 训练把 `α` 学起来，让记忆增量 `slot_delta` 注入到 hidden stream。

> **为什么要跑两次而不是一次拼接？**
> 单次拼接会有 *phantom-logit denominator pollution*：k 个 slot key 在 softmax
> 分母里贡献 k 项，但它们在 init 时几乎是 0，会把 H 自己的 attention
> 衰减到 `S_H/(k+S_H)`。32 层叠加掉 60–90% 信号。Flamingo 输出门
> (`α=0` 时 bypass 完全恢复) 在结构上保证了可逆退化，详见
> `ops/research_notes/20260426_mem_space_v0_tier3_fix3_fail.md §5`。

---

## 3. Memory Bank 与 Top-k 路由

### 3.1 Bank 结构

| 维度 | 默认值 | 说明 |
|------|--------|------|
| 槽位数 N | 512（config 里 default 是 128，跑出来好结果的是 N=512） | `MemoryBank.slots ∈ [B, N, slot_dim]` |
| 每步选 k | 64 | 每个 token step 都把 k 个 slot 拼到 sequence 前 |
| `slot_dim` | = d_model = 4096 | 也可以独立 |
| 初始化 | `"hidden_pool"` | 用第一次 forward 的 H 池化 + N(0,σ) 噪声 |
| **跨 32 层共享** | `shared_memory_bank=True` | 全部 32 层 wrapper 指向同一个 bank，layer-i 的写入立刻被 layer-(i+1) 读到，BPTT 通过深度 |

`shared_bank` 的实现技巧：用 `object.__setattr__` 绕过 `nn.Module` 的子模块
注册，避免 32 个副本污染 state_dict（`layer.py:244-256`）。

### 3.2 Top-k Selector (`selector.py`)

- 每个 token 独立打分：`Q_sel(H) ∈ [B, T, S]`、`K_sel(slots) ∈ [B, N, S]`，
  cosine + softmax 得到 `scores ∈ [B, N]`（per-batch 聚合后 top-k）。
- Hard top-k 选索引 `idx ∈ [B, k]`；同时保留 soft scores 用于 STE 梯度。
- **辅助损失**（aux losses）：
  - `load_balance` (Switch-Transformer 风格)
  - `entropy_aux` (鼓励路由多样性)
  - `key_repulsion` (slot key 互斥)
  - `peak_routing` (鼓励路由 peak 化)

### 3.3 STE / Soft-proxy 梯度

`layer.py:519-556`：forward 用 hard 选择拿到正确内容，backward 用 soft
加权和 (`einsum("bn,bnd->bd", scores, slots)`) 提供 O(1) 梯度，让
`Q_sel` / `K_sel` / `slot_keys` 都能被 LM loss 拉动。
（Fix H, 2026-04-29 之后的设计；此前 STE 信号近 0，路由会塌陷。）

---

## 4. EMA 写回（Gradient-Bearing）

`layer.py:669-712` — 拿到 `O_mem_hidden = ext_h[:, :k, :]` 之后：

```python
O_mem_slot = hidden_to_slot(O_mem_hidden)         # [B, k, slot_dim]
β = sigmoid(gate_param) · warmup_frac · gate_max  # 默认 max=0.3, warmup=2000步
memory_bank.write(idx, O_mem_slot, β)             # slots[idx] = (1-β)*slots[idx] + β*new
```

要点：
- 写回 **不 detach**（Branch-3, 2026-04-26 起）。配合 `shared_bank` 可让
  "把 slot 写好对下一层有帮助" 成为端到端可导信号。
- 跨 chunk 的图断开：每个 BABILong sample / 每段新 chunk，
  `_reset_banks(model)` 调 `MemoryBank.reset()` 把 slots 清空（init 时还有
  `.detach()`）。所以 BPTT 是 **chunk 内**通过深度，**chunk 间**断开。
- 可选 dual-gate（LM2 风格）：input gate + forget gate，由 `(new_repr,
  current_slot)` 共同决定。当前 BABILong baseline 跑的是 single-gate。

---

## 5. Chunked Streaming 推理（`run_babilong_mem_space.py`）

这是让算法能在 16k / 32k 上"赢" vanilla 的关键：

```
长 prompt (e.g. 32k tokens)
        │
        ├─── _reset_banks(model)          ← 每个 sample 开始
        │
        ├─── chunks = tokens.split(4096)  ← chunk_size=4096 (默认)
        │
        ├─── for chunk in chunks[:-1]:
        │       model(chunk)              ← 不要输出，只是为了写记忆 bank
        │       (use_cache=False, no_grad)
        │
        ├─── _freeze_banks(model)         ← 切换到生成模式，不再 EMA 写
        │
        └─── generate(chunks[-1], greedy) ← 在最后一段上 greedy decode
```

为什么这条路径会 win：

| 路径 | 16k+ 行为 |
|------|----------|
| vanilla Llama-3-8B-Instruct (`max_position_embeddings=8192`, 无 rope_scaling) | RoPE 外推失败，输出乱码 (`"the most—- the most—\n"`) |
| 我们的 chunked path | 每 chunk 都在训练区间 `[0, 4095]` 内，记忆 bank 把过去信息浓缩到 N=512 个 slot 里 |

**当前最好结果（2026-05-15）**：
- 长上下文 16k+32k 平均：**mem_space 19.0% vs vanilla 0.3% = +18.7pp**
- 5 个 clean win cell: qa5/16k +43, qa5/32k +31, qa1/16k +19, qa1/32k +14, qa2/16k +5
- 短上下文 8k 上有 cost: qa1/8k -37, qa2/8k -23（chunked 风格切换的代价）

---

## 6. 我们和其他论文的关系

| 论文 | 核心机制 | 我们的对应位置 |
|------|---------|--------------|
| **Flamingo** (2022) | tanh-gated cross-attention 接到 LM 残差流 | 我们的 `α=tanh(slot_output_gate)` 输出门，`α=0` 时严格 bypass |
| **RMT / HMT** | 把上下文摘成几个 memory token，循环传给下一段 | 我们也是 chunk 间断开 BPTT、靠 bank 跨 chunk 传 |
| **Memorizing Transformers** | kNN 从外部 KV 库检索，单层 cross-attn | 我们用 **per-layer** memory bank、joint self-attn 而不是 cross-attn |
| **MoE / Switch** | 路由 + load-balance | 我们的 selector 借 Switch 风格 aux loss 防止路由塌陷 |
| **LM2** | 双门控（input + forget）的可写记忆矩阵 | 我们的 `use_dual_gate` 路径就是这个，默认关闭，BABILong 用 single-gate |

**最大区别**：我们是 **per-layer + shared bank + joint self-attn (KV-prepend) +
chunked streaming**，没有显式 cross-attention 模块，所有"读记忆"动作都
fold 进 LlamaDecoderLayer 自带的 self-attention 里。这是和 Memorizing
Transformer / Flamingo 路线最不一样的地方。

---

## 7. 训练阶段（adapter checkpoint 来源）

当前 BABILong 跑出来的 champion 是 `adapter` checkpoint，由 NIAH
(needle-in-a-haystack) 训练得到（之前的 stage1/2 流程）。**算法本身没改**，
"突然 work" 的两次修复是：

1. **Backbone 切换**：Llama-3-8B (base) → Llama-3-8B-**Instruct**。base
   model 不会跟 BABILong 的 prompt 模板做指令跟随，adapter 再好也输出
   不出答案。
2. **bf16/fp32 dtype 修复**（commit `59a39c2`，`memory_bank.py:307`）：
   `updated = (updated * scale).to(self.slots.dtype)` —— 之前 `slot_norms.norm()`
   返回 fp32，乘法把 `updated` 提升到 fp32，scatter 回 bf16 slots 时报
   `RuntimeError`，只在 16k+ 的 chunked 流式推理触发。

正在跑的 **Fix 2 SFT Phase 1** (PID 209291, ~5.8h) 是"在 BABILong-style prompt
上多任务 SFT"的下一步优化，目标是把短上下文 8k 上的 cost 拉回来，跟"突然
work"无关。

---

## 8. 速查：超参表

```python
# 来自 src/memory/mem_space/config.py + 当前 champion 启动命令
MemorySpaceConfig(
    num_slots               = 512,        # N
    top_k                   = 64,         # k
    slot_dim                = None,       # = d_model = 4096
    selector_dim            = 128,
    writeback_gate_init     = 0.0,        # σ(0)=0.5
    writeback_gate_warmup_steps = 2000,
    writeback_gate_max      = 0.3,        # β_max
    slot_init               = "hidden_pool",
    slot_init_noise         = 1.0,
    load_balance_weight     = 0.01,
    entropy_aux_weight      = 0.0,
    key_repulsion_weight    = 0.01,
    peak_routing_weight     = 0.1,
    slot_dropout            = 0.0,
    use_rope_for_slots      = False,      # slot 永远 pos=0
    enable_writeback        = True,
    shared_memory_bank      = True,       # 32 层共享
    use_dual_gate           = False,      # BABILong baseline 用 single-gate
)
# slot_output_gate (α)：init=0.5 → tanh(0.5)≈0.462
# chunk_size: 4096 (eval), pg19=200 chunks, bab=7 ratio (训练)
```

---

*Last updated: 2026-05-15. 主要源码 commit: 5ad010d, 51b1043, 59a39c2.*
