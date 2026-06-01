# v8 — Multi-Query Routing (logsumexp aggregation + global top-k)

> 日期：2026-06-01　对应 commit：见 gpu_runs.jsonl
> 方向 1（`docs/ARCHITECTURE_AND_ROUTING_PROBLEM.md` §3）的实现。
> 只动 selector + layer 把 l3_summaries 传入 + config + train CLI + 监控。
> **不动** 读路径 / 写回 / 门控 / 训练目标。

## 动机（一句话）

`max_pool` 和 `chunk_query`（mean-pool）都把整个 chunk 压成**单一 query** 去查 slot，
必然塌缩成均匀路由：
- max-pool 通过极值抹平塌缩（每个 slot 都有 champion token）；
- mean-pool 通过过度泛化塌缩（一个万金油 query 对所有 slot 打分都差不多）。

根因是 **single-query routing bottleneck**。解法：不要压成 1 个 query，
用 L3 模块已产出的 64 个 summary token 当 64 个独立 sub-query，保留 chunk 内部语义多样性。

## Architecture（multi_query forward 伪代码）

```python
# 输入：
#   query_tokens = l3_summaries  : [B, M=64, d_model]   (L3 summary tokens)
#   slots                        : [B, N=128, slot_dim]
# 复用现有 key 计算（与 max_pool/chunk_query 完全一致）：
_slots_for_key = slots if no_detach_slots else slots.detach()
k = normalize(K_sel(_slots_for_key) + slot_key_bias)        # [B, N, S]

# multi_query 分支：
q_multi   = normalize(Q_sel(query_tokens))                  # [B, M, S]
score     = einsum("bms,bns->bmn", q_multi, k) * temperature  # [B, M, N]
tau_q     = multi_query_tau                                  # 默认 1.0
logits    = logsumexp(score / tau_q, dim=1) * tau_q          # [B, N]  在 query 维聚合

# 之后走与其它分支完全相同的后续：
scores      = softmax(logits)                               # [B, N]
idx         = topk(scores, top_k)                           # global top-k
one_hot     = scatter(idx)
ste_weights = scores + (one_hot*scores - scores).detach()
return idx, scores, ste_weights
```

**为什么 logsumexp 而非 max/sum**：max 只要 1 个 query 强匹配就选中（噪声敏感）；
sum/mean 要求很多 query 都匹配（偏向 generic slot）；logsumexp 介于两者之间，
保留多 query evidence 又不被平均抹平。`tau_q` 控制软硬：tau→0 ≈ max，tau→∞ ≈ mean。

**Fallback**：当 `routing_pool_mode=="multi_query"` 但 `query_tokens is None`
（第一个 chunk 冷启动 / 未开 L3）时，自动退回 `max_pool` 分支逻辑，不崩。

## Initialization

- `multi_query_tau = 1.0`：平衡默认值。logsumexp(x/τ)·τ 在 τ=1 时即标准 logsumexp，
  介于 max（τ→0）与 mean（τ→∞）之间，无需先验偏向任一端，留给实验扫描。
- `Q_sel` / `K_sel`：沿用现有 std=0.02 小初始化，query/key 均做 F.normalize（cosine 打分）。
- `slot_key_bias`：沿用现有 [N, S] per-slot prior，不改。

## 新增监控（QUERY_DIAG 行）

multi-query 的失败模式不是 uniform，而是「64 个 query 表面不同，但最终都挤到同一批
high-norm / generic slot」。所以新增：

- `summary_q_max_cos` / `summary_q_mean_cos`：对 q_multi[0]（[M,S] 已 normalize）算 pairwise
  cosine，去对角后的 max / mean。衡量 64 个 L3 query 自身是否塌缩（若已高度相似，
  multi-query 退化回 single-query）。
- `uniq_sel_slots`：对每个 sub-query 各自 argmax slot（[M]），数 unique 个数（batch[0]）。
  范围 1~min(M,N)。≈16 → 所有 query 挤同一批 slot（坏）；≈60+ → query 有分工（好）。

全部在 `torch.no_grad()` 下，M≈64 计算开销可忽略，无条件计算。

## Relationship to prior work

- **vs single-query chunk pooling（v0 max_pool / P1-v3 chunk_query）**：本质区别是
  routing unit —— 不是 1 个 chunk-level global query，而是一组 semantic sub-queries，
  保留 chunk 内部多样性。这是对 §2 根因（single-query bottleneck）的直接修复。
- **vs MoE routing（Switch Transformer）**：MoE 是 per-token route to expert；这里是
  per-chunk，用 M 个 summary token 聚合后做一次 chunk-level top-k（global，非 per-token）。
  load_balance / entropy aux loss 仍可叠加（未改）。
- **vs Perceiver / cross-attention（方向 2）**：方向 2 让 slot 主动 attend chunk；本方向
  仍是 query→slot 打分，但 query 由单一变多个，改动更小、复用现有 read/write 路径。

## Known issues

- **query 自身塌缩**：若 L3 的 64 个 summary token 训出来高度相似（summary_q_mean_cos 高），
  multi-query 会退化回 single-query。靠新增的 `summary_q_*_cos` 监控；若发生需在 L3 侧加多样性约束。
- **tau_q 未调**：1.0 为默认，可能需要扫描 {0.5, 1.0, 2.0}。
- **冷启动 fallback**：第一个 chunk 用 max_pool，路由可能与后续 multi_query 不一致，但只影响首 chunk。
- 仍未触及写回侧：若「slot 没存下有区分度的内容」是真病因（§5 开放问题 4），本方向无法单独解决。
