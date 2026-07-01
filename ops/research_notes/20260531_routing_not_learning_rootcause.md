# Routing 不学习根因分析 (2026-05-31)

## TL;DR

**根因**：`selector.py:185` 在 **T≈1024 个 token 维度上做 max-pool** 聚合 routing logits，
这在结构上强制 softmax 趋于 uniform，与训练无关。

- 每个 token 投票选 slot → 对每个 slot 取 1024 个 token 中的**最大** logit
- N=128 个 slot，每个都能找到「某个最匹配它的 token」→ 所有 slot 的 max-logit 都很高且接近
- → softmax over 128 slot 近乎均匀 → top1_sim 塌到 ~1/128（随机基线）
- → **训练越久越糟**：Q_sel/K_sel 学习让更多 slot 找到 champion token → 更平坦
  （观测：top1_sim 从 step4 的 0.054 跌到 0.015）

**次要根因（互相强化的死循环）**：注入门 g≈0.12（sigmoid(-2)），memory 只贡献 12%，
routing 对 loss 的梯度 ∝ g 极小 → routing 学不动 → memory 一直是噪声 → 门学会保持关闭。

**最推荐修复**：把 per-token max-pool 改成「先池化 query 再投影」——每个 chunk 做**一次**
routing 决策（mean-pool 或 attention-pool hidden_states 得到单一 query），而不是 1024 个
token 各投一票再 max。预期 top1_sim 立刻脱离随机基线。

---

## 梯度路径追踪（已确认未被 detach 截断）

```
LM loss
  → next_hidden            (layer.py:975: bypass_h + g·slot_delta + fast_mem)
  → slot_delta             (= 扩展序列注意力输出 - bypass_h)
  → M_sel_hidden 注入扩展序列 (layer.py:813 cat 进 attention)
  → STE backward = M_sel_hidden_soft   (layer.py:760)
  → M_sel_slot_soft = einsum(scores, slots)  (layer.py:750, slots 未 detach ✓)
  → scores = softmax(logits)            (selector.py:186)
  → logits = per_token_logits.max(dim=1) (selector.py:185)  ★瓶颈★
  → per_token_logits = einsum(q,k)·temp  (selector.py:179)
  → q=Q_sel(hidden), k=K_sel(slots)+bias (selector.py:167,174)
```

梯度路径**完整连通**，no_detach_slots_in_selector=True 时 `slots` 不被 detach
（selector.py:173），slot_to_hidden 也已解冻。所以 **H1（梯度截断）排除**。

## 假设逐条核验

### H1 梯度截断 — 排除
- selector.py:173 `_slots_for_key = slots`（no_detach 模式）
- layer.py:739-749 注释明确移除了 slots.detach()，einsum 路径连通
- idx 被 detach（selector.py:190）是正常的（硬选择不可导），soft proxy 补偿梯度

### H2 温度过高 — 部分成立但非主因
- selector.py:179 `logits = einsum(q,k) * temperature`，temperature=20
- q,k 单位归一化 → 点积 ∈[-1,1]，随机初始 ~N(0, 1/√128)=0.088
- ×20 → per_token_logit std ≈ 1.77（**实测 per_tok_logit_std=1.75，吻合**）
- 单 token 的 softmax 其实够尖锐；问题不在温度，而在后面的 max-pool

### H3 aux loss 主导 — 部分成立
- layer.py:1110-1120 收集 4 个 aux：load_balance(0.01)、entropy(0.001)、
  key_repulsion(1.0)、weight_ortho(1.0)
- **key_repulsion 和 weight_ortho 权重都是 1.0（用同一个 key_repulsion_weight）**，很大
- 但这俩作用在 K_sel **keys 之间**的多样性（实测 key_max_cos=0.35-0.50 健康），
  不直接把 routing 推向 uniform
- entropy_aux 显式最大化路由熵（推向 uniform），但权重仅 0.001，量级小
- **结论**：aux 不是主因，但 key_repulsion=1.0 偏大值得后续调

### H4 读写路径解耦 / 门太小 — 成立（次要根因）
- layer.py:975 `next_hidden = bypass_h + g·slot_delta`，g=sigmoid(inject_gate)≈0.12
- routing 对 loss 的梯度 ∝ g ≈ 0.12，先天微弱
- 形成死循环：随机 memory → 门保持 ~0.12 → routing 梯度小 → routing 学不动
- 实测：inject_gate_mean 600 步纹丝不动（=sigmoid(bias_init)），3-arm 消融已证实

### H5 query/key 空间错位 — 表象，非根因
- Q_sel、K_sel 独立初始化独立更新，但梯度路径连通，理论上能学对齐
- 真正阻止对齐的是 H4（梯度太小）+ 下面的 max-pool（信号被抹平）

### ★核心根因：max-pool 抹平 softmax（H2 的真正版本）
- selector.py:180-185：注释声称 max-pool「保留多样性」，**在大 T 下事实相反**
- 数学：N(0,1.77) 取 1024 样本的最大值 ≈ 1.77·√(2ln1024) ≈ 6.6，
  且 128 个 slot 的 max 彼此接近（极值分布 spread 很小）
- → softmax(128 个≈6.6 的值) ≈ uniform → top1_sim ≈ 1/128
- → 训练让更多 slot 找到 champion token，**反而更平**（解释 top1_sim 单调下降）

## 修复方向（按推荐度）

### 修复 1（最推荐）：chunk 级单 query routing
把 per-token max-pool 改成先池化再投影，每 chunk 一次 routing 决策：
- 位置：selector.py:167-186
- 改法：`q_pool = normalize(Q_sel(hidden_states.mean(dim=1)))` → [B,1,S]，
  直接得 logits=[B,N]，去掉 max(dim=1)
- 注意：旧的「mean 导致 uniform」教训是指 **softmax-then-mean（平均概率分布）**，
  而这里是 mean-pool **输入表示再投影**，单一 query 仍可尖锐，不同问题
- 预期：top1_sim 立刻脱离 0.015 随机基线
- 风险：低（只改 routing 聚合，不动注入/写回）

### 修复 2：降低温度 + 配合修复 1
- temperature 20→ 可学习或固定 ~5，避免修复 1 后单 query 过尖
- 位置：config 的 selector_temperature

### 修复 3（治 H4 死循环）：训练初期强制开门
- 让 inject_gate 在前 ~500 步用更大固定值（如 g=0.5），给 routing 足够梯度信号
  逃离 uniform，之后再交还给可学习门
- 或：用 relevance gate，g ∝ top1_sim 置信度（routing 越确信，注入越多）
- 位置：layer.py:939-941

### 修复 4（次要）：解耦 key_repulsion 和 weight_ortho 权重
- layer.py:1120 用了 key_repulsion_weight(=1.0) 给 weight_ortho，偏大
- 建议给 weight_ortho 单独的小权重（如 0.01）
