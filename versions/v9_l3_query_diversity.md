# v9: L3 Summary-Token Diversity Regularizer

> 日期：2026-06-01  ·  commit 待填  ·  延续 v8 (multi_query routing, commit 8c57ae7)

## 背景

v8 用 L3 的 64 个 summary token 当 64 个 sub-query 做 multi-query routing
(logsumexp 聚合 → global top-k)。300-step health check 实测：
- 早期 `top1_sim` 从 0.012(均匀) 回升到 0.0356 → routing 方向对。
- 但 `summary_q_max_cos=1.000, summary_q_mean_cos=1.000, uniq_sel_slots=1`
  → **L3 的 64 个 summary token 塌缩成几乎同一个向量**。

数学后果：M 个相同 query 的 logsumexp = 同一行 + 常数，topk 严格不变
→ multi_query 精确退化回 single_query，top1_sim 衰减回 0.016。

根因：`L3SummaryPool` 是 Q-Former 式 cross-attn pool，输出端没有任何
diversity 压力 → 经典 Q-Former 输出塌缩：所有 query 都 attend 成 chunk 均值，
输出 64 个几乎相同的向量。`_collect_aux_loss` 里也没有任何 L3 相关 aux loss。

## Architecture

### (1) Diversity loss（作用在输出 S，而非 learnable queries）

```
S ∈ [B, M, d]          # L3 输出（实际喂给 routing 的 summary token）
Sn  = normalize(S, dim=-1)
sim = Sn @ Sn^T         # [B, M, M] pairwise cosine
penalty = relu(sim - threshold)      # threshold=0.5
loss = mean_{b, i<j} penalty[b,i,j]  # 只取上三角 i<j
```

- threshold=0.5：比 selector 的 key_repulsion (0.3) 宽松，因为 summary token
  本就共享 chunk-level 共性，不应强迫完全正交。
- mean over batch（比 batch[0] 更稳）。
- M<2 时返回 0。

**为什么作用在输出 S 而非 learnable queries**：
1. 塌缩观测在**输出端**（summary_q_*_cos≈1.0 是 S 之间的相似度）。
2. L3 递归时 `prev_summary` 会覆盖 `self.queries`，所以约束 queries 在递归
   chunk 上根本不起作用，必须约束实际输出 S。

### (2) Orthogonal init（根治"初始就相似"）

```python
# 旧：nn.init.normal_(self.queries, std=0.02)   # 64 个 query 初始高度相似
self.queries = nn.Parameter(torch.empty(num_summary, d_model))
nn.init.orthogonal_(self.queries)
```

std=0.02 让 64 个 query 初始都是 0 附近的微小随机偏移 → 高度相似，是塌缩起点
之一。orthogonal init 保证初始 64 个 query 互相正交/最大多样，给 cross-attn
机会把每个 query 特化到 chunk 不同部分。

### (3) Aux loss 接线

- `layer.py`：l3_pool 存在 + `_layer_idx==0` + `l3_summaries is not None` 时算
  `l3_div = l3_pool.query_diversity_loss(l3_summaries, threshold=cfg.l3_diversity_threshold)`，
  `aux["l3_diversity"] = l3_div * cfg.l3_diversity_weight`。
  - **只在 layer_idx==0 收集一次**：l3_pool 是 32 层共享单例
    (`object.__setattr__` 挂载)，否则会 32× 重复计入。
- `config.py`：`l3_diversity_weight=0.1`, `l3_diversity_threshold=0.5`。
- `train_mem_space_dolmino_cpt.py`：`_collect_aux_loss` 的 keys tuple 加
  `"l3_diversity"`；CLI `--l3_diversity_weight`/`--l3_diversity_threshold`；
  config 接线 + JSON inherit 映射。

## Initialization

| 参数 | 值 | 理由 |
|------|-----|------|
| `queries` init | `orthogonal_` | 直接根治初始相似，保证 64 query 正交 |
| `l3_diversity_weight` | 0.1 | 与 load_balance(0.01)/key_repulsion(0.05) 同量级，略强 |
| `l3_diversity_threshold` | 0.5 | 比 key_repulsion(0.3) 宽松，summary token 允许共性 |

## Sanity 验证（.venv/bin/python）

L3SummaryPool(d_model=64, num_summary=8)：
- 输出 shape [2,8,64] ✓
- orthogonal init 后 query max_cos ≈ 5e-8（之前 std=0.02 ≈ 1.0）✓
- 初始 forward 输出 pairwise max_cos = **0.60**（之前 ≈ 1.0），mean_cos = 0.28 ✓
- `query_diversity_loss` 返回正标量 (0.0029) 且可 backward ✓

## Relationship to prior work

- **Q-Former (BLIP-2) collapse**：Q-Former learnable query 在无 diversity 压力
  下会塌缩成均值池化，是已知失败模式。本 fix 在输出端加 repulsion 正则。
- **selector.key_repulsion_loss**：完全类比——penalise pairwise cos > threshold，
  只是作用对象从 slot key 换成 L3 summary token，threshold 从 0.3 放宽到 0.5。
- **DPP / orthogonal regularization**：orthogonal init 是更轻量的 one-shot
  多样性保证，不引入持续的 determinant 计算开销。

## Known issues

1. orthogonal init 只保证**初始**多样，训练后仍可能被 attention 拉回塌缩——
   靠 diversity loss 持续约束，需 health check 监控 `summary_q_*_cos` 是否
   维持 < 1.0。
2. threshold=0.5 / weight=0.1 是初始猜测，若塌缩仍发生可调强（threshold↓ 或
   weight↑）；若伤害 LM loss 则调弱。
3. 只在 layer_idx==0 收集，依赖 l3_pool 确为共享单例的假设——若未来改成
   per-layer L3 pool，需移除该守卫。
4. recursive L3 下 `prev_summary` 覆盖 queries，orthogonal init 的好处主要体现
   在 cold-start 第一个 chunk；后续 chunk 的多样性完全靠 diversity loss 维持。
