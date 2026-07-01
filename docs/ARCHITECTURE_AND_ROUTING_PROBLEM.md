# Mixture-of-Memory：架构设计 + 当前核心问题（讨论稿）

> 用途：给合作者讨论用。一页讲清楚「我们在做什么架构」+「卡在哪」+「几个候选解法」。
> 日期：2026-06-01。对应代码 commit `773b29f`。
> 详细 forward 见 `docs/CURRENT_ALGORITHM.md`；本文聚焦**路由失效**这个当前最大卡点。

---

## 0. 一句话

我们想让一个 8B LLM 用**固定大小的 memory buffer（N 个 slot）**压缩任意长的上下文。
机制已经搭好且能跑，但**「按内容寻址 slot」的路由（routing）训不起来——它塌缩成均匀分布，等于随机选 slot**。这是现在唯一真正卡死的地方。

---

## 1. 架构（我们在做什么）

**Per-layer memory bank + Flamingo 风格 joint self-attention（KV-prepend）+ 门控写回**，配合 chunked streaming 把超长上下文切片喂入。

### 数据流（单层，单 chunk）

```
H ∈ [B, T, d]   (T=1024 当前 chunk 的 hidden states)
   │
   ├─(1) Selector: 从 N=128 个 slot 里选 top-k=16 个   ← 【问题就在这一步】
   │        idx ∈ [B, k]
   │
   ├─(2) M_sel = slots[idx] → 投影到 hidden → 拼到序列前面
   │        ext = [M_sel ; H]   (slot 用 pos=0，全局可见)
   │
   ├─(3) 同一个 LlamaDecoderLayer 跑两遍：
   │        ext_h  = layer([M;H])      带记忆
   │        bypass = layer(H)          纯 vanilla
   │
   ├─(4) Flamingo 门控融合（α=0 时严格退化回 vanilla，可逆）：
   │        next_h = bypass + α·(ext_h[k:] − bypass)
   │
   └─(5) 门控写回 slot（gradient-bearing，EMA 或 LM2 双门）：
            slots[idx] ← gate·new + (1−gate)·slots[idx]
```

- **32 层共享同一个 bank**（`shared_memory_bank=True`），layer-i 写入立刻被 layer-(i+1) 读到 → chunk 内 BPTT 穿透深度。
- **chunk 间断开**：每个 sample 重置 bank，所以记忆靠 slot 内容跨 chunk 传，不靠 BPTT。
- 另有 **L3 summary 模块**（已实现并在跑）：一个 Q-Former 式 cross-attn pool，把每个 chunk 的 T 个 token 压成 **64 个 dense summary token**，prepend 到序列里。**这 64 个 token 是下面解法 1 的关键弹药。**

### 关键设计点（为什么这么搭）
- **不引入新 cross-attention 模块**，复用 Llama 自带 self-attn，靠 KV-prepend 把「读记忆」fold 进去。
- **跑两遍 + Flamingo 门**：避免单次拼接的 phantom-logit 分母污染（k 个近零 slot key 会把 H 自己的 attention 衰减掉，32 层叠加掉 60–90% 信号）。α=0 严格可逆退化是结构保证。

---

## 2. 核心问题：路由塌缩成均匀分布

### 现象（Dolmino CPT 训 44k 步后）
- `top1_sim ≈ 0.012`（= 1/128，均匀分布地板）→ selector 选 slot 跟掷骰子没区别。
- 门控参数（β/α）几乎不动 —— 但这是**结果**不是病因：路由随机 → 写回也随机 → slot 之间没区分度 → 门控拿不到有效梯度。
- 下游：BABILong ~25%、LongBench F1 ~7（vanilla baseline 34）。**memory 不仅没帮上忙，还在污染生成。**

### 病因（这是讨论的重点，已经定位得比较清楚）

Selector 打分逻辑：`logits[slot_i] = pool(query) · key_i`，其中 `key_i = K_sel(slot_i) + slot_key_bias_i`。
问题出在**怎么把整个 chunk 的 T=1024 个 token 压成 query 去和 N 个 slot key 打分**：

| 方案 | 机制 | 结果 |
|------|------|------|
| **max-pool over T**（原始）| 每个 slot 取它在 1024 个 token 上的最高分 | 每个 slot 都能找到一个「冠军 token」→ 所有 slot 的 max-logit 都差不多 → softmax 退化均匀 |
| **mean-pool → chunk_query**（这轮试的修复）| 先把 1024 token 平均成 1 个 query 再打分 | 平均出一个「万金油」query，对任何 slot 打分都差不多 → 同样塌缩 |

**实测 chunk_query 的塌缩轨迹**（`per_tok_logit_std` 从 0.57 单调衰减到 0.066，logits 几乎全相等）：

| step | top1_sim | logit_std | 说明 |
|------|----------|-----------|------|
| 4    | 0.028    | 0.57      | slot 初值 norm≈0，content routing 此时**有**区分度 |
| 75   | 0.0099   | 0.11      | slot 经 EMA 长到 norm≈5 后，`K_sel(slots)` 投影后所有 key 从 query 视角变得几乎一样 |
| 299  | 0.0098   | 0.066     | 完全均匀 |

> 注意：`key_max_cos` 一直健康（0.38~0.56）→ **key 在原始空间没塌缩**，塌缩发生在「query 太泛化、无法区分 key」这一侧。

### 结论（已被两组实验证实，根因表述已更新）

> **认知演化（给合作者看，避免抓矛盾）：**
> 1. **2026-05-31**：初步认为 `max-pool over T` 是主因——每个 slot 都能找到 champion token，logits 被极值抹平 → 当时推荐改成 mean-pool chunk query。
> 2. **2026-06-01**：实验证明 mean-pool（chunk_query）**也塌缩**。所以根因不是 max-pool 这个单点，而是更一般的 **single-query routing bottleneck**。

**更新后的根因（canonical 表述）：**
> Routing collapse is caused by a single global chunk query being unable to preserve intra-chunk semantic heterogeneity. **Max-pooling collapses through extreme-value equalization; mean-pooling collapses through over-generalization.** Both lead to slot logits becoming nearly indistinguishable.

一句话：**问题不是 memory bank 机制整体失败，而是 selector 的 routing unit 选错了——它不该是一个 chunk-level global query，而该是一组 semantic sub-queries。**

---

## 3. 候选解法（要讨论的）

### 方向 1（已选定，优先做）：Multi-query routing + logsumexp aggregation + global top-k
**不要把 chunk 压成 1 个 query。** 用 L3 已经产出的 **64 个 summary token 当 64 个独立 sub-query**，每个 sub-query 对所有 slot 打分，**logsumexp 在 query 维度聚合成每个 slot 的全局 relevance**，再做 global top-k。保留 chunk 内部多样性，从根上避免万金油。
- ✅ 弹药现成（L3 模块已实现并在跑，64 个 summary token 已 cache 在 layer 里）。
- ✅ 改动集中在 selector.forward + layer 把 l3_summaries 传进去，**不动读/写/门控/训练目标**。
- ✅ 保留 content-based routing 初衷，不退化成 deterministic bucket。

**union 策略（不用 naive union，避免 slot 数失控 + 缺全局排序）：**
```python
# q: [B, M=64, S]   (L3 summary tokens 投影 + normalize)
# k: [B, N=128, S]  (K_sel(slots) + slot_key_bias, normalize)
score = torch.einsum("bms,bns->bmn", q, k) * temp        # [B, M, N]
slot_score = torch.logsumexp(score / tau_q, dim=1) * tau_q  # [B, N] 在 query 维聚合
idx = slot_score.topk(k=top_k, dim=-1).indices            # global top-k
```
为什么 logsumexp 而非 max/sum：max 只要一个 query 强匹配就选中（噪声敏感）；sum/mean 要求很多 query 都匹配（偏向 generic slot）；**logsumexp 介于两者之间**，保留多 query evidence 又不被平均抹平。τ_q 控制软硬程度。保留 chunk 内部多样性，从根上避免万金油。
- ✅ 弹药现成（L3 模块已实现并在跑，64 个 summary token 已经 cache 在 layer 里）。
- ✅ 改动集中在 selector.forward + layer 把 l3_summaries 传进去。
- ❓ 待定：64 个 query 各自 top-k 后怎么 union（取并集？按分数加权？union 后 slot 数会超过 k，要不要二次裁剪）。

### 方向 2：Cross-attention routing（slot-as-query）
反过来：让**每个 slot 作为 query** 去 attend 整个 chunk 的 hidden states，自己算「我关心的 token 在不在这个 chunk 里」的 relevance 分数。
- ✅ 概念上最干净（slot 主动寻址，而非被动被查）。
- ❓ 计算量：N×T 的 attention，N=128、T=1024 还好。

### 方向 3：放弃 content-based routing
用 position/hash 把 token 段**确定性**分配到 slot（类似 Infini-attention 分块）。
- ✅ 简单、绝不塌缩。
- ❌ 放弃了「按内容寻址」的初衷，退化成固定分桶。

### 防塌缩通用补丁（无论选哪个方向都建议加）
- 对 `K_sel(slots)` 输出加 LayerNorm，或 query/key normalize 后再点积（当前 query 在 mean-pool 后模长信息已丢）。
- 早停监控：`per_tok_logit_std` 连续 N 步 < 0.15 → 自动判失败、kill、换方向，别浪费 GPU 跑满 2000 步。

---

## 4. 诊断指标速查（怎么判断路由健康）

grep log 里的 `QUERY_DIAG` 行：
- `top1_sim_mean`：> 0.05 才算有意义的非均匀路由（1/128 = 0.0078 是地板）。
- `per_tok_logit_std`：< 0.15 基本就是塌缩警报。
- `key_max_cos`：看 key 之间是否塌缩（目前一直健康 0.38~0.56，说明病不在 key 侧）。

`WRITEBACK_DIAG` 行：
- `gate_val(beta)` / `inject_gate_mean`：钉在初值不动 = 路由没给有效梯度（**结果**，非病因）。

### Multi-query routing 新增监控（方向 1 必加）
multi-query 的失败模式**不是 uniform**，而是「64 个 query 表面不同，但最终都挤到同一批 high-norm / generic slot」。所以光看 top1_sim 不够，要加：

**A. query diversity**（看 64 个 L3 summary query 自己有没有塌缩；若它们已高度相似，multi-query 退化回 single-query）：
- `summary_query_max_cos` / `summary_query_mean_cos`

**B. selected slot coverage**（看不同 query 是否真覆盖不同 slot）：
- `unique_selected_slots_per_chunk`：≈16 → 所有 query 挤同一批 slot（坏）；≈60~100 → query 有分工（好）
- `query_slot_entropy`

---

## 5. 留给讨论的开放问题

1. 方向 1 的 union 策略：64 个 query × 各自 top-k 怎么合并成最终选中的 slot 集合？
2. 是不是 **slot 内容随 EMA 长大** 本身就是个独立的病（norm≈5 后 K_sel 投影塌缩）？要不要在写回侧也动手（更激进的 norm cap / 写回前 LayerNorm）？
3. content-based routing 到底值不值得救？还是方向 3 的确定性分桶 + 强 L3 summary 已经够用？
4. 有没有可能问题不在路由结构，而在「**slot 根本没存下有区分度的内容**」——即写回侧才是真病因，路由只是背锅？（这个假设目前证据较弱，但值得反驳一下。）

---

*配套阅读：`docs/CURRENT_ALGORITHM.md`（完整 forward）、`ops/research_notes/20260430_1106_fix_z_analysis.md`（query collapse 早期分析）、`ops/research_notes/20260531_routing_not_learning_rootcause.md`、`status/HANDOFF_20260601.md`。*
