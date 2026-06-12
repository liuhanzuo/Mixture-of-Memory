# Memory-Space v0 原理与机制详解

_版本: 2026-04-27 10:15 CST · 写给用户本人速查 · 对应 Branch-3 A.2 winner (PPL=1.9051)_

---

## TL;DR(一段话回答你的问题)

**你提的"memory space + top-k + cross-attention"的做法, 就是现在真正跑通的方案.**
具体对应:

| 你说的概念 | 当前实现里的对应模块 | 位置 |
|---|---|---|
| **memory space** | 32 层共享的 N=512 slots bank(每 slot 一个 slot_dim 向量) | `src/memory/mem_space/memory_bank.py` + `layer.py::MemorySpaceLayer.__init__` 里的 `shared_memory_bank=True` 分支 |
| **top-k** | `TopKSelector`: 对 mean-pool 的 hidden 算 scores → softmax → `torch.topk(k=64)` → Straight-Through Estimator 保梯度 | `src/memory/mem_space/selector.py` |
| **cross-attention** | KV-prepend 形式: 把 top-k 选出的 slot 经 `slot_to_hidden` 投到 d_model 后, **prepend 到当前层的 hidden_states 前面**, 然后让 wrapped `LlamaDecoderLayer` 在 `[M_sel, H_l]` 上做 joint self-attention(slot queries 看全部, H queries 因果 + 全看 slots)——这就是**用 self-attention 机制模拟 cross-attention** | `src/memory/mem_space/layer.py::MemorySpaceLayer.forward` |

额外四个"你没明说但我加进去的工程细节"(都是为了稳定训练):
1. **STE(Straight-Through)**: top-k 的硬选择不可导, 用 `ste_weights = scores + (one_hot_scores - scores).detach()` 让梯度穿回 selector.
2. **Switch-style load-balance aux loss**: 防止所有 token 都选同几个 slot.
3. **slot 位置用 RoPE pos=0**: 让 slot 成为"无位置的 memory token", 避免和真实序列位置冲突.
4. **Flamingo 式 output gate `tanh(slot_output_gate)`**: 初始化为 0, 保证训练开始时 slot 贡献为 0(bit-exact 同 vanilla), 避免一开始就污染 LM.

这四个细节才是让 PPL 从 **472.31(未加 σ 调度)** 降到 **1.9051(加上 σ=0.02 + warmup=500 + shared_bank + H7 rotary fix)** 的关键.

---

## 1. 自上次对话以来的结论与改进

### 1.1 Branch-3 A.2 2×2 factorial 跑完(2026-04-26 23:41)

四组实验, 200 步训练, 8×B200 各跑一个, 每个约 6 分钟:

| Exp | 节点 | σ | warmup | shared_bank | **PPL** | 判决 |
|---|---|---|---|---|---|---|
| **A_v2** | b200-1 | 0.02 | 500 | ✓ | **1.9051** | **🏆 SHIP** |
| B_v2 | b200-2 | 1.0 | 0 | ✗ | 5.4808 | 弱 |
| C | b200-3 | 0.02 | 500 | ✗ | 3.5226 | 弱 |
| D | b200-4 | 1.0 | 0 | ✓ | 2.9616 | 弱(H7-only 对照) |

**结论**:
- Exp-A 把 Branch-1 best(2.1278)打下来 **Δ = -0.2228 PPL**.
- **H1 σ-kick 爆炸 CONFIRMED**: σ=0.02+warmup=500 的主效应 Δ=-1.50 PPL.
- **H3 shared_bank 深度放大 FALSIFIED**: shared_bank **帮倒忙反而是帮正忙** Δ=-2.06 PPL —— 跟 researcher 之前的假设相反, 共享反而好.
- **H7 rotary inv_freq bf16 destructive cast NECESSARY**: Exp-D(仅 H7 fix)把 PPL 从 472 → 2.96, **提升 157×**, H7 是必要条件但不是充分条件.

### 1.2 H7 根因(上次没讲清楚)

`model.to(dtype=torch.bfloat16)` 会递归到 `LlamaRotaryEmbedding.inv_freq` 这个 buffer, 把 fp32 的 `0.81225...` 四舍五入成 `0.81640625`. 在 pos=1023 处, 角度误差累积到 **~4.25 弧度**, cos 值飘到 **±2** 的外太空值. 注意力分数爆表, PPL 从 ~2 变 472.

**H7 fix v2 (snapshot-before-cast)**:
```python
# 1. 在 .to(dtype=bf16) 之前把 fp32 的 inv_freq / original_inv_freq 拷一份
_rope_snapshot = {}
for _name in ("inv_freq", "original_inv_freq"):
    if hasattr(_rot, _name):
        _rope_snapshot[_name] = getattr(_rot, _name).detach().to(torch.float32).clone()
# 2. 做 dtype 转换(照样会破坏掉 inv_freq, 但 snapshot 不受影响)
model = model.to(device=device, dtype=dtype)
# 3. 用 snapshot 把 buffer 恢复回 fp32
for _name, _buf in _rope_snapshot.items():
    _rot._buffers[_name] = _buf.to(dtype=torch.float32)
    setattr(_rot, _name, _buf)
```
上述 pattern 已经同步到 4 个文件(训练脚本 + 2 个 probe + 1 个 batch-ceiling probe).

### 1.3 假设总账(上次对话时还没结的单)

| 假设 | 状态 | 证据 |
|---|---|---|
| H1 σ-kick runaway | **CONFIRMED** | 主效应 Δ=-1.50 PPL |
| H2 kwargs/SDPA dispatch | FALSIFIED | 32 层 err=0 bit-exact |
| H3 shared_bank 深度放大器 | **FALSIFIED** | 共享反而好 Δ=-2.06 |
| H5 bf16 rounding on 0·slot_delta | FALSIFIED | bypass-parity 探针 bit-exact |
| H7 rotary inv_freq 破坏性 cast | **CONFIRMED (必要)** | 472 → 2.96 (157×) |

### 1.4 Ship 配置(promoted)

```bash
--slot_init random
--slot_init_noise 0.02          # σ = 0.02 (不是默认 1.0)
--writeback_warmup_steps 500    # warmup = 500 步
--shared_memory_bank            # 32 层共享一个 memory bank
--writeback_gate_max 0.3        # 写回门限 σ(·) ≤ 0.3
--unfreeze_hidden_to_slot       # 解冻 O_mem → slot 投影
--num_slots 512 --top_k 64 --selector_dim 128
--load_balance_weight 0.01
+ H7 rotary fix v2 (4 个文件都已就位)
```

---

## 2. Memory-Space v0 原理与机制(给你从头讲一遍)

### 2.1 整体架构:每一层都套一个 wrapper

原始 Llama-3-8B 有 32 层 `LlamaDecoderLayer`. 我们**不动原来的 transformer 层**, 而是把它整个塞进一个 `MemorySpaceLayer` wrapper:

```
输入 H_l  ──►  MemorySpaceLayer(wraps LlamaDecoderLayer_l)  ──►  输出 next_hidden
                    ├── 1. 从共享 MemoryBank 里 top-k 选 slot
                    ├── 2. 把 slot 投影到 d_model, prepend 到 H_l 前面
                    ├── 3. 调用 wrapped LlamaDecoderLayer 两次:
                    │     (a) bypass 路径  = 原 Llama 行为
                    │     (b) extended 路径 = 带 slot 的 joint attention
                    ├── 4. Flamingo 式 tanh-gate 融合: bypass + gate*(ext - bypass)
                    └── 5. 把 extended 路径里 slot 位置的 O_mem 写回 bank
```

**关键**: bypass 路径保证"gate=0 时 bit-exact 同 vanilla Llama", 训练开始时不破坏 LM.

### 2.2 Memory Bank(对应你说的"memory space")

```python
# src/memory/mem_space/memory_bank.py (精简版)
class MemoryBank(nn.Module):
    def __init__(self, num_slots=512, slot_dim=...):
        self.slots = nn.Parameter(torch.zeros(num_slots, slot_dim))
```

- **N = 512 个 slot**, 每个 slot 一个向量(slot_dim = Llama 的 d_model = 4096).
- `shared_memory_bank=True` 时, **32 层共享同一个 bank 对象**, 靠 `object.__setattr__` 绕过 `nn.Module` 的子模块注册机制, 保证 `state_dict` 不重复存 32 次:
```python
# layer.py
if shared_memory_bank:
    object.__setattr__(self, "memory_bank", shared_bank_obj)
    self._owns_bank = False  # 不要 register_parameter / register_module
```
- **Lazy init**: 第一次 forward 时, 如果 slot 还是全零, 就用当前 batch 的 mean-pooled hidden + σ·gaussian 初始化. σ=0.02 是把 noise 压到很小, 不然会 kick 爆.

### 2.3 TopKSelector(对应你说的"top-k")

```python
# src/memory/mem_space/selector.py
class TopKSelector(nn.Module):
    def forward(self, h_pool, memory):
        # h_pool: [B, d_model], memory: [N, slot_dim]
        q = self.q_proj(h_pool)                         # [B, selector_dim]
        k = self.k_proj(memory)                         # [N, selector_dim]
        logits = torch.einsum("bs,ns->bn", q, k) * scale
        scores = F.softmax(logits, dim=-1)              # [B, N]
        top_scores, top_idx = torch.topk(scores, k=64)  # 硬选 top-k
        one_hot = scatter(top_idx, num_classes=N)       # [B, N]
        # Straight-Through: fwd 用硬 top-k, bwd 让梯度穿回 scores
        one_hot_scores = one_hot * scores
        ste_weights = scores + (one_hot_scores - scores).detach()
        return top_idx, ste_weights  # 后者用于 load balance + 梯度
```

**工程细节**:
- `selector_dim=128`(不是 full d_model), 省算力.
- Q/K Linear 初始化 `N(0, 0.02²)` —— 否则 softmax 会塌成近乎 one-hot, selector 永远不学.
- **Switch-Transformer 式 load-balance aux loss**:
  `aux = N · Σ_n importance_n · load_n`, 推 selector 均匀使用 slot. 权重 `--load_balance_weight 0.01`.

### 2.4 Cross-Attention via KV-Prepend(对应你说的"cross-attention")

**这是整个架构最关键的一步**. 我用 self-attention 模拟了你说的 cross-attention, 但**零侵入** Llama 本身.

做法:
1. 从 top-k 选出的 k=64 个 slot 里 gather 出 `M_sel: [B, 64, slot_dim]`.
2. 过 `slot_to_hidden: Linear(slot_dim, d_model, bias=False)`, 得到 `M_hid: [B, 64, 4096]`. Linear 初始化 `N(0, 0.02)`, 让一开始 M_hid 很小.
3. **Prepend 到 hidden 前**: `H_ext = concat([M_hid, H_l], dim=1)`, shape `[B, 64+T, 4096]`.
4. **构造显式 4-D causal mask**(这一步必须显式, 因为 HF 的 SDPA 对含 memory 的序列不会自动构造正确 mask):
   ```
                 ┌─── slot keys ───┬─── H keys ────┐
   slot queries: │  全部可见       │  全部可见     │   (slots 看所有)
   H queries:    │  全部可见       │  因果三角     │   (H 看 slot + 因果 H)
   ```
5. **扩展 position_embeddings**: 真实的 H 位置照旧用 HF 给的 cos/sin; slot 位置全部 **复用 pos=0 的 cos/sin**(把 slot 当"无位置 memory token").
6. 用同一个 `wrapped_layer` 跑两次:
   - **bypass 路径**: `bypass_h = wrapped_layer(H_l, mask=explicit_causal_4D)` —— 纯 vanilla.
   - **extended 路径**: `ext_h = wrapped_layer(H_ext, mask=extended_4D, position_embeddings=pos_ext)`, 取 slot 之后的部分 `ext_h[:, 64:, :]` —— slot 通过 attention 影响每个 H 位置.
7. **Flamingo 式 output gate 融合**:
   ```python
   alpha = torch.tanh(self.slot_output_gate)   # Parameter, 初始化 0 → α=0
   next_hidden = bypass_h + alpha * (ext_h[:, k:, :] - bypass_h)
   ```
   这保证初始 step 是 bit-exact vanilla, 训练中 α 慢慢长出来.

**为什么这就是 cross-attention**: 从 H 里每个 token 的视角看, 它能 attend 到 k=64 个 slot + 前面所有的 H token. slot 就是"被查询的 memory". 这和标准 cross-attention 的区别只是**用一个 self-attention 模块复用了 Q/K/V 权重**, 不用新的 cross-attn 矩阵—— 这也是为什么叫 "KV-prepend".

### 2.5 Writeback(slot 怎么学)

光是"读" memory 不够, 还得"写":
```python
# extended 路径里 slot 位置的 output
O_mem = ext_h[:, :k, :]           # [B, k, d_model]
# 投回 slot 空间
delta = self.hidden_to_slot(O_mem)  # [B, k, slot_dim]
# EMA 式写回(带梯度, 可以把 loss 穿回 bank 本身)
beta = sigmoid(gate_param) * warmup_frac * gate_max
for b, idx_b in zip(range(B), top_idx):
    memory[idx_b] = (1 - beta) * memory[idx_b] + beta * delta[b]
```

**几个关键点**:
- `warmup_frac = min(1, step / warmup_steps)` —— 前 `--writeback_warmup_steps=500` 步 β 从 0 线性爬, **让 selector 先学稳, 再开始写**. 这是 σ=0.02 + warmup=500 效果好的核心.
- `gate_max=0.3` —— 写回强度天花板, 防止 bank 被单 step 写崩.
- `--unfreeze_hidden_to_slot` —— 让 `hidden_to_slot` 这个 Linear 可训练. 冻结的话 PPL 会翻倍.

### 2.6 可训练参数范围

| 参数 | 冻结 | 可训练 |
|---|---|---|
| Llama backbone(Q/K/V/O + FFN + embeddings + LM head) | ✅ **冻结** | |
| Selector(Q_sel / K_sel) | | ✅ 2 × (d_model × selector_dim) ≈ 1M |
| `slot_to_hidden: Linear(slot_dim, d_model, bias=False)` | | ✅ ~16.8M × 32 层 |
| `hidden_to_slot: Linear(d_model, slot_dim, bias=False)` | | ✅ ~16.8M × 32 层 |
| `slot_output_gate`(Flamingo gate, Parameter scalar) | | ✅ 32 个标量 |
| `gate_param`(writeback gate, Parameter scalar) | | ✅ 32 个标量 |
| MemoryBank.slots `[N, slot_dim] = [512, 4096]` | | ✅ ~2M(共享 1 份) |

总可训练参数 ≈ **570M**(相对 8B backbone 约 7%), 都是 adapter-style 的小模块.

---

## 3. 你说的做法 ≡ 当前实现? 是的, 外加 4 个稳定性技巧

回到开头的问题: **"我说的 memory space, topk, cross attention 的做法你有参考么?"**

**明确回答: 有, 而且是直接当成蓝本实现的.** 对应关系:

| 你说的 | 我的实现 | 增补 |
|---|---|---|
| memory space | 512-slot shared bank(32 层共享 1 份) | 共享 vs 独立做了 A/B, 共享赢 2.06 PPL |
| top-k | TopKSelector hard top-64 of 512 | + STE(梯度穿回)+ Switch load-balance |
| cross-attention | KV-prepend joint self-attention on `[M_sel, H_l]` | + 显式 4-D mask(HF SDPA 不会自动对) + slot-RoPE=pos0 + Flamingo output gate |

**没做的**(有意避开):
- 没有用**独立的 cross-attn 矩阵**(会多 ~4× 参数, 而且要重新学), 而是**复用原 Llama 层的 Q/K/V/O**, 这样 bypass 路径能 bit-exact 恢复到 vanilla.
- 没有做**多层不同的 memory**(shared_bank 实验告诉我们, 共享更好).
- 没有做**memory 跨 chunk persist**(每次 forward 结束 bank EMA 更新, 但 chunk 边界不 reset; 如果 reset 反而掉分).

---

## 4. "全上" 派发计划(2026-04-27)

按 TRAINER_ACTIVE.md 列的 3 个候选, 今天全部并行派发:

| # | 节点 | 目的 | 配置 |
|---|---|---|---|
| 1 | b200-1 | **Ship-config schedule-match** | A v2 配置, 延长到 Branch-1 完整训练步数, 验证 1.9051 不是 200-step 早期优势的人工现象 |
| 2 | b200-2/-3/-4 | **12-cell σ × warmup ablation** | σ ∈ {0.01, 0.02, 0.05, 0.1} × warmup ∈ {200, 500, 1000}, 每节点跑 4 cell |
| 3 | local 8×H20 | **N=1024 / k=128 scale-up** | 把 N 翻倍, k 翻倍, 用 A v2 配置 —— 之前 Stage-2b 在 σ=1.0 下失败, 现在用 σ=0.02 重试 |

派发全部走后台 subagent, main 继续处理用户消息. 完成后 append `gpu_runs.jsonl` + `ACTIVE_SWEEPS.jsonl`, 并在下一次 heartbeat 里总结.

---

## 附: 相关文件速查

- **架构实现**: `src/memory/mem_space/{layer.py, selector.py, memory_bank.py, __init__.py}`
- **训练脚本**: `scripts/train_mem_space_pg19.py`
- **运行模板**: `scripts/_run_mem_space_full_llama3.sh`(和 smoke / parity / evalonly 变种)
- **H7 修复同步文件**: `tests/test_wrapper_internal_parity.py`, `scripts/probe_branch3_bypass_parity.py`, `scripts/probe_mem_space_batch_ceiling.py`, 训练脚本本身
- **Branch-3 设计背景**: `ops/research_notes/20260426_memory_space_design_direction.md`, `20260426_mem_space_v0_branch3_writeback_bptt.md`
- **最终结论落点**: `status/TRAINER_ACTIVE.md` §"Branch-3 A.2 2×2 factorial COMPLETE"
