# v11 — Slot-as-Query Cross-Attention Routing (`routing_pool_mode="slot_query"`)

> 日期：2026-06-01。对应 selector.py 新增 `slot_query` 分支。
> 背景文档：`docs/ARCHITECTURE_AND_ROUTING_PROBLEM.md`（方向 2）。

## 1. 动机：为什么绕开 L3-as-router 死结

routing 塌缩调查走过三步：

1. **single-query routing**（max-pool / mean-pool 把 chunk 压成 1 个 query）→ 塌缩成均匀。
   - max-pool：每个 slot 都能找到 champion token → 所有 logit 顶到相似上限 → 均匀。
   - mean-pool（chunk_query）：平均出一个"万金油" query，对任何 slot 打分都差不多 → 同样塌缩。
2. **multi_query routing**（v8–v10，用 L3 的 M 个 summary token 当 M 个 sub-query，logsumexp 聚合）→ 想法对，
   但 **L3 的 Q-Former 输出架构性塌缩**：M 个 learnable query softmax-attend 同一个 chunk H，
   所有输出被驱向同一个 chunk 加权平均。实测 `S_max_cos → 1.0000`，加 diversity loss（weight 0.1→0.3）压不住。

**结论：query 多样性是个死结。** slot_query 反转 query/key 角色，**让 query 变成 N 个 slot**——
多样性已经由 slot 内容 + `key_repulsion_loss` 保证，不再依赖"把 chunk 压成有区分度的 query"。

## 2. Architecture（forward 伪代码）

输入：`pool_of_H` = chunk 完整 hidden states `H [B, T, d_model]`，`slots [B, N, slot_dim]`。

```
# query = slots（N 个，多样性有保证），key/value = chunk token H（T 个）
q_slot = K_sel(slots_for_key) + slot_key_bias        # [B, N, S]  复用 K_sel 投影 slot
q_slot = normalize( q_sel_ln(q_slot) )               # [B, N, S]  复用 v10 LayerNorm 破投影塌缩
k_tok  = normalize( Q_sel(H) )                       # [B, T, S]  复用 Q_sel 投影 hidden（= 公共 q）
attn   = einsum("bns,bts->bnt", q_slot, k_tok) * temperature   # [B, N, T] 每个 slot 对每个 token 的相似度

# 聚合 [B,N,T] → slot relevance [B,N]：用 softmax-weighted 相似度期望（soft-max pooling）
attn_w = softmax(attn, dim=T)                        # [B, N, T] 每个 slot 在 T 上的注意力分布
logits = (attn_w * attn).sum(dim=T)                  # [B, N]    软最大相似度，介于 mean 与 max 之间

# 之后统一走现有路径：
scores = softmax(logits); idx = topk(scores, k); ste = scores + (one_hot*scores - scores).detach()
```

- `slots_for_key = slots if _no_detach_slots else slots.detach()`（沿用现有逻辑）。
- 复用现有 `Q_sel`/`K_sel`/`q_sel_ln`/`slot_key_bias`，**不新增大投影层**（保持参数量 + 复用已有正则）。
  - `Q_sel: d_model→S` 投影 H；`K_sel: slot_dim→S` 投影 slot；维度一致。
- slot_query 分支**完全不碰 `l3_summaries`**，`_last_q_multi_diversity_loss` 保持 None（layer 的 `_q_div is not None` 守卫不会加 stale loss）。

## 3. 为什么是 soft-max pooling 而不是硬 max

把 `[B,N,T]` 聚合成 `[B,N]` 时：
- **硬 max over T** 是已知塌缩源：每个 slot 都能在 1024 个 token 里找到一个最匹配 token →
  所有 slot 的 max-logit 都顶到相似上限 → softmax 退化均匀（这正是 max_pool 模式的病）。
- **mean over T** 又太平：把"匹配若干 token 的 slot"和"啥都不匹配的 slot"抹平。
- **softmax 加权的相似度期望** `Σ_t softmax(attn)_t · attn_t` 介于两者之间：
  匹配多个 token 的 slot 得分高，啥都不匹配的 slot 得分低，但没有"人人有 champion"的均衡化。

## 4. Initialization

无新增参数，全部复用现有初始化：
- `Q_sel.weight ~ N(0, 0.02)`、`K_sel.weight ~ N(0, 0.02)`。
- `slot_key_bias = normalize(randn(N, S)) * 2.0`（topic prior，content 信号可覆盖）。
- `q_sel_ln`（v10 LayerNorm）默认 weight=1 / bias=0。
- `temperature = selector_temperature`（启动脚本默认 20.0）。

## 5. Relationship to prior work

- **vs single/multi query（本项目 v3–v10）**：那些都"把 chunk 压成 query 去查 slot"，受限于 query 多样性。
  slot_query 反转角色，query = slot，根本不需要 chunk 端有多样性。
- **vs Memorizing Transformer / kNN-augmented attention**：那里是 token 作为 query 去 kNN 检索 memory；
  这里是 **slot 作为 query** 主动 attend chunk，算"我关心的内容在不在这个 chunk 里"，是 slot 端主动寻址。
- **vs CrossAttentionMemory（本仓库已有类）**：那是 read/write cross-attn 直接产出 hidden/slot 内容；
  slot_query 只用 cross-attn 算 **routing relevance 分数**（top-k 选 slot），不改读/写/门控/训练目标。

## 6. Sanity 结果（d_model=64, slot_dim=64, S=8, N=16, k=4, T=1024, temp=20）

喂"前 8 个 slot ≈ chunk 真实 token、后 8 个 = 随机噪声"的构造：
- 输出形状 idx `[2,4]`、scores `[2,16]`、ste `[2,16]`，idx ∈ [0,16)。
- **匹配 slot(0:8) 平均 score 0.0965 vs 噪声 slot(8:16) 0.0285（3.4× 区分度）**。
- top1_sim ≈ 0.20，远高于 uniform 地板 1/16=0.0625。
- `slot_attn_entropy ≈ 2.75`（max=log1024≈6.93，说明 attention 有聚焦）。
- `per_tok_logit_std ≈ 1.25`（有区分度，非均匀）。
- backward：`Q_sel`/`K_sel`/`q_sel_ln`/`slot_key_bias` grad 均非 None。
- max_pool / multi_query 分支不受影响。

## 7. Known issues

- **soft pooling 是否也会塌缩**：soft-max pooling 缓解但不保证根除均衡化。如果训练中所有 slot 都把
  attention 摊平到全 chunk（`slot_attn_entropy` 高、`per_tok_logit_std` 低），说明 slot 没学会寻址 →
  退化回均匀。靠新增的 **`slot_attn_entropy`** 诊断（QUERY_DIAG 行）监控：
  熵接近 log(T) 且持续上升 = 警报。
- **slot 内容随 EMA 长大**可能仍让 `K_sel(slots)` 投影塌缩（与 multi_query 同源风险）；
  `q_sel_ln` 缓解，但写回侧的 norm 增长是独立隐患，需观察。
- N×T attention 计算量：N=128、T=1024 量级可接受。
