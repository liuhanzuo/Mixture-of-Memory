# Sparse Memory — Bug Fix & Refactor Plan

> **目标读者**：后续接手动手修改代码的 agent。
> **范围**：`src/memory/sparse_memory/` 下的 `model.py`、`attention.py`、`memory_bank.py`。
> **前置依据**：`CODE_REVIEW_REQUEST.md` 中列出的 #1（生成退化）与 #2（selective 劣于 full）。
> **非目标**：本文档不修 gate instrumentation、不改 chunk-level retrieval 架构、不动 L1 stub。这些留给后续单独的 PR。

---

## 0. TL;DR

本次修 4 个 bug，按优先级从高到低：

| ID | 位置 | 性质 | 对应文档问题 | 改动规模 |
|----|------|------|------------|--------|
| **A** | `model.py::forward` / `generate` | 正确性（致命） | #1 生成退化 | 单行逻辑改动 + 新增 helper |
| **B** | `memory_bank.py::update_slots` + `attention.py` | 梯度正确性 | #2 selective 不如 full | ~20 行 |
| **D** | `attention.py::forward`（memory write 处） | 语义错误 | #2 | 3 行 |
| **E** | `memory_bank.py::reset/update_slots` | DDP 安全 + 性能 | 代码结构问题 | ~30 行 |

> Bug **C**（memory K 无 RoPE）经过二次 review 后判定**不是 bug**，是合理的设计选择，不改。

修完 A+B+D 就能同时闭合 `CODE_REVIEW_REQUEST.md` 的 #1 和 #2。E 属于基础设施层面的清理，独立于正确性，但强烈建议一起做。

---

## 1. Bug A — `generate()` 路径每步 reset_memory 导致推理时 memory 恒为零

### 1.1 现象
- 所有 memory 模型在 greedy decode 下输出 degenerate repetition（重复句号、无限 0）。
- Vanilla Llama2-7B 正常。
- PPL（teacher-forced）和 generation quality 之间存在巨大 gap。

### 1.2 根因
当前实现：

```python
# src/memory/sparse_memory/model.py
def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
    B = input_ids.shape[0]
    self.reset_memory(batch_size=B)      # ← 每次 forward 都 reset
    return self.model(input_ids=input_ids, ...)

def generate(self, *args, **kwargs):
    return self.model.generate(*args, **kwargs)
```

HF `generate()` 内部是一个 `while not_finished: outputs = self(input_ids=next_token, past_key_values=..., ...)` 的循环，**每个 decode step 都会重新进入这个 `forward()`**。后果：

1. Prompt 阶段 `forward(input_ids=prompt)` → reset 一次 → memory 被 prompt 写入（~有效）。
2. Decode 第 1 步 `forward(input_ids=next_token[T=1])` → **reset 再次清零 memory** → attention 里 `mem ≡ 0`，`o_mem` 贡献几乎为零（gate 初始 σ(2)≈0.88 的 local 权重把 memory 分量进一步压低）。
3. 同时 decode 步 T=1，`q.mean(dim=2)` 退化成"单 token 自己的 query"，chunk-level retrieval 的假设也崩了。

训练时因为每条样本是一次 forward、且有 teacher forcing，这个 bug 被掩盖 → PPL 看起来 OK，但 autoregressive 分布完全不同。

### 1.3 修复方向

把"reset 的时机"从 `forward()` 里挪出去，让它由**样本边界**显式触发，而不是**每次 forward**。

**推荐方案（最小侵入）**：用 `past_key_values is None` 作为"新样本"的信号——HF 在每个独立 sample / `generate()` 初次调用时 `past_key_values` 一定是 `None`，而在 decode step 中一定非空。

```python
def forward(
    self,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    past_key_values=None,                         # ← 新增参数捕获
    **kwargs,
) -> dict:
    """Forward pass with memory reset at sample boundaries.

    Memory is reset only when past_key_values is None, i.e. at the start
    of a new sample (training step) or at the first step of generate().
    Subsequent generate() decode steps (past_key_values is not None) will
    *preserve* memory built up from the prompt.
    """
    B = input_ids.shape[0]
    if past_key_values is None:
        self.reset_memory(batch_size=B)
    return self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        past_key_values=past_key_values,
        **kwargs,
    )
```

**同时修改 `generate()`**：保险起见，在进入 HF 的 `generate` 之前显式 reset 一次（避免"上一个样本 memory 残留"的问题），然后 decode step 中 `forward` 因 `past_key_values is not None` 不再 reset。

```python
def generate(self, input_ids, *args, **kwargs):
    B = input_ids.shape[0] if torch.is_tensor(input_ids) else 1
    self.reset_memory(batch_size=B)
    return self.model.generate(input_ids, *args, **kwargs)
```

### 1.4 验证标准
- 跑 `scripts/eval_sparse_memory_ppl.py` 的 PPL 不应有明显变化（训练期 `past_kv is None`，语义不变）。
- 跑 generate（例如 `scripts/debug_rmt_inference.py` 或任意一个 `run_nih_*.sh`）时：
  - 期望输出**不再是** `..................` 或 `000000`；
  - 至少能产生语义可读的段落；
  - 可选：打印每个 decode step 下 `memory_banks[0].memory.abs().mean()`，应该在 prompt 阶段增大，decode 阶段保持非零、不归零。

### 1.5 风险
- 若 `generate` 之外的调用者在一次"逻辑样本"内多次调用 `forward` 但没传 `past_key_values`（例如手写的逐 chunk 训练循环），行为会和之前等价（每次 reset）。本项目内搜索一下，确认无此类调用即可。

---

## 2. Bug B — `importance_head` 永远收不到梯度

### 2.1 现象
- Selective (top-K) writing 在两组配置下均劣于 full write，和原始假设（"全量写入导致旧信息被快速覆盖"）相反。
- 伴生现象：即使换 `importance_mode`，结果差异很小——因为起决定作用的 `importance_head` 根本不是训练出来的。

### 2.2 根因
看 `memory_bank.py::update_slots`：

```python
if self.write_top_k and self.write_top_k > 0 and self.write_top_k < T:
    importance = self.compute_importance(hidden_states, attention_weights)  # 有梯度
    topk_vals, topk_idx = importance.topk(self.write_top_k)
    # ↑ topk 对 values 可导、对 idx 不可导；但返回的是 idx 用作下标

    hidden_states_selected = hidden_states[topk_idx].detach()   # ← 硬截断
    slot_indices_selected = slot_indices[topk_idx]
```

两个独立的截断都把梯度断了：

1. **`topk_idx` 本身不可导**（argmax-like）——把它作为下标去 index `hidden_states` 后，梯度无法通过"选中了哪些 token"这件事回传给 `importance_head`。
2. **`topk_vals` 被完全丢弃**——`compute_importance` 里的 `learned_score` 计算出来后根本没出现在任何 loss 的计算图里。
3. 即使 1 和 2 都不存在，`update_slots` 的返回值是 `None`、`memory` 本身 `requires_grad=False`，整个 importance 计算图最终也是个孤岛。

**结论**：`importance_head.weight` 和 `importance_head.bias` 自训练开始以来**从未收到过非零梯度**。它们永远停留在 `zeros(weight) + zeros(bias)` 的初始值上，意味着实际的 importance 评分退化为：

```
importance = magnitude * (1 + surprise) + sigmoid(0) = magnitude * (1 + surprise) + 0.5
```

—— 完全由固定公式决定，没有任何学习。

### 2.3 和 CODE_REVIEW #2 的对应
"selective 不如 full" 的直接原因之一就是这个：selective 需要 importance_head 学会"什么 token 值得写"，但它完全没学，等于用一个从未被训练的启发式规则做了"信息筛选"——很容易过滤掉关键信号。全量写入反而没有这个筛选噪声。

### 2.4 修复方向

我们不动 "hard top-K 选择哪些 token 参与 EMA 写入" 这个语义（因为这是论文核心），但要给 `importance_head` **一条真实的、连通到 loss 的梯度路径**。最小侵入方案：

**让 importance score 额外作为 read path `o_mem` 的 per-token 软权重**，通过 `o_mem → output → loss` 反传梯度。

具体做法：

#### Step 1. 在 `MemoryBank` 里暴露"只算 learned_score"的接口
在 `memory_bank.py` 中加入一个轻量方法（不破坏现有 `compute_importance`）：

```python
def learned_importance(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Compute the *learnable* part of importance only (with gradient).

    Args:
        hidden_states: [B, T, D] or [T, D].
    Returns:
        [B, T] or [T]: sigmoid-squashed learnable importance in [0, 1].
    """
    score = self.importance_head(hidden_states.float()).squeeze(-1)  # [..., T]
    return torch.sigmoid(score)
```

> 注意：这里**保留梯度**，不做 `.detach()`。

#### Step 2. 在 `SparseMemoryAttention.forward` 里把它用进 `o_mem`
在计算 `o_mem` 之后、fusion 之前，插一段：

```python
# o_mem: [B, T, D]
# Use learnable importance as a soft per-token gate on memory path.
# This is the ONLY way importance_head receives gradient from the LM loss.
imp = self._memory_bank.learned_importance(hidden_states)  # [B, T], fp32
imp = imp.to(o_mem.dtype).unsqueeze(-1)                    # [B, T, 1]
o_mem = o_mem * imp                                        # broadcast to [B, T, D]
```

这样 `o_mem → g * o_local + (1-g) * o_mem → output → loss` 的反向传播会把梯度带回 `importance_head`。

#### Step 3. 写入路径（`update_slots`）的 hard top-K 逻辑不动
`update_slots` 里继续用 `importance.topk(write_top_k)` 选 token 写 memory。此时 `importance_head` 已经被训练出来了（通过 read path 学到"哪些 token 与 LM loss 相关"），`topk_idx` 自然会选到真正有用的 token。

#### Step 4. 初始化调整
当前 `importance_head.bias` 初始化为 0，导致 `sigmoid(0) = 0.5`，memory 贡献被初始砍半。改为初始化 bias 为 ~2.0 让 `sigmoid ≈ 0.88`，避免训练初期 memory 分量被过度抑制：

```python
with torch.no_grad():
    nn.init.zeros_(self.importance_head[0].weight)
    self.importance_head[0].bias.fill_(2.0)    # σ(2) ≈ 0.88
```

### 2.5 为什么这样改不会破坏原设计
- `compute_importance` 函数签名/返回值不变，`update_slots` 语义不变。
- 新增的 `learned_importance` 只是同一个 `importance_head` 的另一个调用点。
- Read path 多乘了一个 `[B,T,1]` 的 sigmoid 门控，等价于"让模型学习哪些 token 的 memory 更值得读"——这本来就是 importance 该做的事情。
- 不增加可学参数量。

### 2.6 验证标准
训练 100 steps 后检查：
- `model.memory_banks[0].importance_head[0].weight.grad.abs().mean()` 应**非零**（当前为 0）。
- `model.memory_banks[0].importance_head[0].weight.abs().mean()` 应偏离初始值 0。
- 重跑 `selective_128` 和 `selective_256`，期望 PPL 至少不劣于 `full_write_*`（这次 importance_head 真正在学了）。

---

## 3. Bug D — `avg_shared_idx` 用算术平均聚合 head 索引是语义错误

### 3.1 现象
在 `attention.py::forward` 尾部：

```python
# shared_idx_for_write: [B, H, K]
avg_shared_idx = shared_idx_for_write.float().mean(dim=1).long()  # [B, K]
avg_shared_idx = avg_shared_idx.clamp(0, self._memory_bank.num_slots - 1)
```

把不同 head 的 top-K slot 索引做**算术平均再取整**。假设 head 0 选了 slot `[5, 20, 100]`，head 1 选了 slot `[7, 22, 102]`，平均后写入 `[6, 21, 101]`——**这 3 个 slot 没有任何 head 真正检索到**。

### 3.2 根因
索引不是可以做算术平均的量，它是类别型。正确的聚合要么取某个 head 的结果，要么做 union/多数投票。

### 3.3 修复方向

**推荐：union（去重合集）+ padding 到定长**。这样所有被任一 head 检索到的 slot 都会被 EMA 更新，最贴近 retrieval 的原意。

```python
# shared_idx_for_write: [B, H, K]
B_ = shared_idx_for_write.shape[0]
write_slots_list = []
for b in range(B_):
    # Union of all heads' top-K for this batch entry
    uniq = torch.unique(shared_idx_for_write[b].flatten())  # variable length
    write_slots_list.append(uniq)

# update_slots loop 也要相应改为接受 variable-length slot set
for b in range(B_):
    self._memory_bank.update_slots(
        hidden_states[b],                                # [T, D]
        batch_idx=b,
        slot_indices=write_slots_list[b].unsqueeze(0),   # [1, K_b]
        attention_weights=per_token_attn[b],             # [T, K]（注意 K≠K_b，见下）
    )
```

> **注意**：`per_token_attn` 的最后一维 K 是 "per-token read 时看到的 K 个 shared slot"，和 write 用的 K_b（union 后）维度不同。`update_slots` 里 `attention_weights` 只用于 `compute_importance` 算 per-token 重要性，不要求和 `slot_indices` 的最后一维匹配；保持现状即可。

**备选方案（更保守、改动更小）**：直接取 head 0 的索引，不做跨 head 聚合：

```python
avg_shared_idx = shared_idx_for_write[:, 0, :]  # [B, K]
```

这个方案改动 1 行，语义干净但浪费了其他 head 的信息。如果 agent 觉得 union 实现复杂度高，先用这个版本；后续再升级到 union。

### 3.4 验证标准
- 训练不报 shape 错误。
- 打印每次 write 命中的 `unique_indices.numel()`：union 版应该 ≥ K（跨 head 差异越大越多）；head-0 版应该 ≤ K。
- `memory_bank.get_write_stats()['retention_rate']` 应接近预期的 `write_top_k / T`。

---

## 4. Bug E — `nn.Parameter` 重新赋值 + Python for loop

### 4.1 现象
`memory_bank.py` 中两处：

```python
# reset()
self.memory = nn.Parameter(
    torch.zeros(batch_size, ..., dtype=self._dtype),
    requires_grad=False,
)

# update_slots(), 末尾
with torch.no_grad():
    new_mem = self.memory.clone()
    unique_indices = torch.unique(slot_indices_selected.flatten())
    for idx in unique_indices:                       # ← Python for loop
        mask = (slot_indices_selected == idx).any(dim=-1)
        ...
        new_mem[batch_idx, idx] = updated
    self.memory = nn.Parameter(new_mem, requires_grad=False)   # ← 重新赋值 Parameter
```

### 4.2 问题

1. **DDP 安全性**：`nn.Parameter` 的重新赋值会在 DDP 下导致"每个 rank 的 parameter 身份不同"，autograd 引擎和 DDP reducer 的 parameter handle 可能失效。虽然 `requires_grad=False` 在单卡下不会报错，但在多机多卡下可能出现隐蔽的同步问题。正确做法是用 `register_buffer` + `self.memory.copy_(...)` in-place。
2. **Python for loop 性能**：对 `unique_indices` 逐个做 mask + mean，每层每样本一次，完全可以向量化。
3. **`new_mem = self.memory.clone()` 再整体赋回** 是 N×D 的整份拷贝，也是浪费。

### 4.3 修复方向

#### Step 1. `memory` 改成 buffer 而非 parameter

在 `MemoryBank.__init__`：

```python
# 旧
self.register_parameter(
    "memory",
    nn.Parameter(torch.zeros(1, num_slots, hidden_dim, dtype=dtype), requires_grad=False),
)
# 新
self.register_buffer(
    "memory",
    torch.zeros(1, num_slots, hidden_dim, dtype=dtype),
    persistent=False,   # 不进 state_dict
)
```

> `persistent=False` 是因为 memory 是运行时状态（每个 sample reset），不该存进 checkpoint。

#### Step 2. `reset()` 改为 in-place

```python
def reset(self, batch_size: int = 1) -> None:
    device = self.memory.device
    # 如果 batch size 变了，重新分配；否则原地清零
    if self.memory.shape[0] != batch_size:
        self.memory = torch.zeros(
            batch_size, self.num_slots, self.hidden_dim,
            device=device, dtype=self._dtype,
        )
    else:
        self.memory.zero_()
    self._write_count = 0
    self._total_tokens_seen = 0
```

> 注意：buffer 可以直接用 `self.memory = tensor` 赋值——`nn.Module.__setattr__` 识别到同名 buffer 时会替换而不报错。这个赋值不会触发 DDP 问题因为 buffer 不参与 allreduce。

#### Step 3. `update_slots` 向量化 + in-place

目标：把 `for idx in unique_indices` 换成 `scatter_reduce_` 或等价的向量化实现。

```python
# hidden_states_selected: [K', D]
# slot_indices_selected:  [K', K_slots]  （K' = write_top_k 或 T；K_slots = write 时的 slot 数）
# 每个选中的 token 要写到 slot_indices_selected[i] 的每一个 slot
# 步骤：
# 1) 展平成 (token_idx, slot_idx) 对
# 2) 用 index_add_ / scatter_reduce_ 聚合同一 slot 的 hiddens（做 mean）
# 3) EMA: memory[b, slot] = alpha * agg + (1-alpha) * memory[b, slot]

Kp, Ks = slot_indices_selected.shape
D = hidden_states_selected.shape[-1]

# [K', K_s] -> [K' * K_s]
flat_slots = slot_indices_selected.reshape(-1)                  # [K'*Ks]
# hidden repeated for each slot it is assigned to: [K'*Ks, D]
flat_hids = hidden_states_selected.unsqueeze(1).expand(Kp, Ks, D).reshape(-1, D).to(self._dtype)

# Per-slot sum and count via scatter_add
N = self.num_slots
slot_sum = torch.zeros(N, D, device=flat_hids.device, dtype=self._dtype)
slot_cnt = torch.zeros(N, device=flat_hids.device, dtype=self._dtype)
slot_sum.index_add_(0, flat_slots, flat_hids)
slot_cnt.index_add_(0, flat_slots, torch.ones_like(flat_slots, dtype=self._dtype))

# Slots that got any write
active = slot_cnt > 0                                           # [N]
if active.any():
    agg = slot_sum[active] / slot_cnt[active].unsqueeze(-1)     # [A, D]
    # EMA in-place on this batch entry
    cur = self.memory[batch_idx, active]                        # [A, D]
    self.memory[batch_idx, active] = (
        alpha * agg + (1.0 - alpha) * cur
    ).to(self._dtype)
```

改动效果：
- 无 Python loop，一次 `index_add_` 搞定；
- `self.memory` 全程 in-place 更新，无 Parameter 重建；
- GPU 利用率↑，对大 `num_slots` / 大 `T` 的场景（例如 N=256, T=1024）加速明显。

### 4.4 验证标准
- 单卡训练结果数值上与旧版等价（在 FP32 下）：可以加一个临时对比脚本把新旧两种实现的 `memory` 在同样输入下做 `allclose`（bf16 容差放宽）。
- 多卡 DDP 启动不再有 "parameter ... was marked as ready twice" 或类似警告。
- `nvprof` / `torch.profiler` 看 `update_slots` 耗时应显著下降。

### 4.5 副作用：`object.__setattr__(self, '_memory_bank', memory_bank)` 不再必要
原代码用 `object.__setattr__` 绕过 `nn.Module` 的子模块注册，是因为 `memory` 作为 `nn.Parameter` 会出现在 state_dict 里、造成重复。改成 buffer（且 `persistent=False`）以后这个担忧不存在。但**不要动这一行**——`_memory_bank` 本身作为 nn.Module 如果走正常注册会导致"同一个 MemoryBank 在顶层 ModuleList 和底层 attention 里重复"，这才是 `object.__setattr__` 的主因。保持现状。

---

## 5. Bug C — memory slots 没有 RoPE / 位置信息（不修）

### 5.1 现象
`attention.py` 里 RoPE 只作用于 local path 的 `q, k`，memory path 的 `k_mem / v_mem` 是对 memory slots 直接投影得来，没有位置编码。

### 5.2 判定：不是 bug
- Memory slots 是聚合后的"压缩表示"，没有对应的绝对位置，硬套 RoPE 反而会引入噪声；
- 当前架构中 memory slot 的"身份"靠 EMA 内容本身区分，是 set-of-slots 的语义；
- 如果后续想引入"slot id"作为可学位置 bias，属于架构改进而非 bug 修复，单开 PR。

**本次 Plan 不做任何改动。**

---

## 6. 实施顺序与验证流程

建议一次 commit 做一个 bug，便于回滚和归因：

1. **Commit 1 — Fix A**（generate path）
   - 跑 PPL：应无显著变化。
   - 跑 generate：degenerate repetition 消失。

2. **Commit 2 — Fix D**（head 索引聚合）
   - 跑短训练（500 steps）：loss 曲线和之前同量级，无 NaN。

3. **Commit 3 — Fix B**（importance head 梯度）
   - 加一个 `scripts/diag_importance_grad.py`（可选），训练 100 steps 后 assert
     `memory_banks[0].importance_head[0].weight.grad is not None and .norm() > 0`。
   - 重跑 `selective_128` / `selective_256`，PPL 期望不劣于对应 `full_write_*`。

4. **Commit 4 — Fix E**（DDP/性能清理）
   - 单卡前后 `memory` 数值 allclose（bf16 容差 1e-2）。
   - DDP 2 卡 smoke test 通过。
   - `torch.profiler` 对比 update_slots 耗时。

每个 commit 都要通过以下最小 smoke test：

```bash
# 1. 模型能加载
python -c "from src.memory.sparse_memory.model import SparseMemoryLlamaForCausalLM; \
           m = SparseMemoryLlamaForCausalLM('<base_model_path>', memory_slots=128, top_k=8, \
               write_top_k=16); print(m)"

# 2. 能 forward
python scripts/debug_forward.py

# 3. 能 generate（文档 #1 的核心验收）
python scripts/debug_rmt_inference.py    # 或任一 run_nih_smoke.sh

# 4. PPL 评测可跑
python scripts/eval_sparse_memory_ppl.py --ckpt <path> --n_chunks 10
```

---

## 7. 不在本次修改范围内的事项（仅记录，不做）

- #3 memory 利用率 instrumentation（gate 分布 / o_mem 贡献）——独立 PR，加 logging 即可。
- #4 Gate 结构升级（per-layer bias / learnable layer-id）——架构实验。
- #5 chunk-level retrieval 的信息瓶颈（token-level retrieval）——架构实验。
- #9 memory warmup at inference——A 修完后可重新评估必要性。
- #10 跨层 memory routing——架构实验。
- L1 memory compression stub（`model.py` 里的 `use_l1` 分支）——另起炉灶。

---

## 8. 文件变更 checklist（给 agent 打勾用）

- [ ] `src/memory/sparse_memory/model.py`
  - [ ] `forward`: 根据 `past_key_values is None` 判断是否 reset_memory。（Fix A）
  - [ ] `generate`: 入口先 `reset_memory(B)`，再调用 `self.model.generate`。（Fix A）

- [ ] `src/memory/sparse_memory/attention.py`
  - [ ] `forward`: 在计算 `o_mem` 后乘上 `self._memory_bank.learned_importance(hidden_states)`。（Fix B）
  - [ ] `forward`: 把 `avg_shared_idx = ... .float().mean(dim=1).long()` 替换为 per-batch union 或 head-0 选取。（Fix D）

- [ ] `src/memory/sparse_memory/memory_bank.py`
  - [ ] 新增 `learned_importance(hidden_states) -> Tensor` 方法。（Fix B）
  - [ ] `importance_head` bias 初始化改为 2.0。（Fix B）
  - [ ] `self.memory` 从 `register_parameter` 改为 `register_buffer(persistent=False)`。（Fix E）
  - [ ] `reset`: in-place `zero_` 或必要时重建 buffer，不再 `nn.Parameter(...)`。（Fix E）
  - [ ] `update_slots`: 用 `index_add_` 向量化替换 Python for loop，`self.memory` in-place 更新。（Fix E）

全部勾完后跑第 6 节的 4 条 smoke test + 重训 `selective_256`，如 PPL/generation 两项验收通过则本次修复完成。
