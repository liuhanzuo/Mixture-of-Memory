# v10 — Q_sel Projection-Collapse Fix (multi_query routing)

> 日期：2026-06-01
> 关联 commit：见本次 fix commit
> 前置版本：v8 (multi_query routing)、v9 (L3 query diversity loss)
> 关联文件：`src/memory/mem_space/selector.py`、`src/memory/mem_space/layer.py`、
> `scripts/train_mem_space_dolmino_cpt.py`

---

## 1. 背景与根因（已实验确证）

v8 引入 multi_query routing：用 L3 的 M=64 个 summary token 作为 64 个独立
sub-query，每个 sub-query 对所有 N=128 个 slot 打分，logsumexp 在 query 维聚合
→ global top-k。v9 在此基础上加了 L3 summary 的 diversity loss，作用在
**Q_sel 投影之前的原始 L3 输出 S（4096 维）** 上。

**两组 300-step health check（l3_diversity_weight = 0.1 本机 / 0.3 远程，3 倍强）
实测**：

- `summary_q_max_cos = 1.000`（64 个 routing query 完全相同，钉死）
- `top1_sim_mean ≈ 0.017–0.02`（均匀地板，routing 失效）
- **加强 3 倍 diversity weight 完全没区别** → 排除"weight 太小"，坐实结构性 gap。

**根因**：诊断量 `summary_q_max_cos` 测的是
`q_multi = F.normalize(self.Q_sel(query_tokens))`，即 L3 summary token **经过
Q_sel 线性投影 [d_model=4096 → selector_dim=128] 之后** 的 query。而 v9 的
`l3_diversity_loss` 作用在 **投影之前的 S（4096 维）**。

即使 S 在 4096 维是多样的（v9 orthogonal init 初始 S max_cos≈0.60），无约束的线性层
Q_sel 会把这些多样的 S **投影坍缩到几乎同一个 128 维方向 + 同一模长** → 经过
`F.normalize` 后 64 个 routing query 几乎完全相同 → multi_query 退化回 single_query
→ 均匀路由。**L3 diversity loss 永远看不到投影后的空间，所以 3 倍 weight 也无效。**

### 本次新增诊断已 100% 坐实（sanity 数值）

构造 **near-rank-1 的 Q_sel**（模拟训练塌缩后的权重）+ **diverse 的 S**：

| 配置 | `S_max_cos` (投影前) | `summary_q_max_cos` (投影后) |
|------|---------------------|------------------------------|
| 无 LayerNorm（v9 行为） | 0.0000 | **1.0000** ← 复现 bug |
| 加 LayerNorm（v10 修复） | 0.0000 | **0.3334** ← 解除钉死 |

> S 完全多样（max_cos=0），但投影后没有 LN 时 query 完全相同（1.0）→ 投影塌缩是
> 真凶。加 LayerNorm 后投影后 max_cos 从 1.0 降到 0.33。

---

## 2. 修复（两个互补改动，同一 commit）

### 修复 1：Q_sel 投影后加 LayerNorm（仅 multi_query 分支）

`selector.__init__`：新增 `self.q_sel_ln = nn.LayerNorm(selector_dim)`。
multi_query 分支里：

```python
q_multi = F.normalize(self.q_sel_ln(self.Q_sel(query_tokens)), dim=-1)  # [B, M, S]
```

LayerNorm 在 Q_sel 与 normalize 之间，**对每个 query 独立 re-center + re-scale**，
打破"所有输出投影到同一方向 + 同一模长"的塌缩：被 Q_sel 映射到相近位置的两个输入，
会被各自的 per-feature 偏差重新拉开。

**作用域**：只在 multi_query 分支用，`max_pool` / `chunk_query` 分支的 q
**完全不动**（避免影响其它模式行为）。

### 修复 2：diversity loss 改为作用在投影后空间 q_multi（关键）

v9 的 diversity loss 作用在 S（投影前 4096 维），看不到 routing 实际使用的空间。
v10 在 selector multi_query 分支里新增一个 **作用在 q_multi（投影后、normalize 后
128 维）** 的 diversity loss：

```python
# selector.forward, multi_query 分支, 在 no_grad 诊断块之外（保证可导）：
qsim_loss = torch.bmm(q_multi, q_multi.transpose(1, 2))         # [B, M, M]
_pair = F.relu(qsim_loss - self._q_multi_diversity_threshold)[:, iu(i<j)]
self._last_q_multi_diversity_loss = _pair.mean()
```

- 梯度可流回 **Q_sel 和 q_sel_ln**（q_multi 不 detach；loss 在 no_grad 块外计算）。
- layer.py 在 aux 收集处（`layer_idx==0` 守卫内）读取
  `selector._last_q_multi_diversity_loss`，以 `aux["q_multi_diversity"] =
  _q_div * cfg.l3_diversity_weight` 加入总 loss。
- threshold 复用 `cfg.l3_diversity_threshold`（默认 0.5），语义一致。

### 设计决定：S-loss 留还是不留？→ **两个都留**

- **S-loss（v9，作用在 L3 输出 S）**：保留。无害，且让 L3 Q-Former 的输出保持多样，
  给投影提供"多样的输入"——LayerNorm + q_multi-loss 修的是投影端，S-loss 守的是
  输入端，两端互补。
- **q_multi-loss（v10，作用在投影后）**：load-bearing 的主项，因为它约束的是
  routing **实际使用的空间**。
- 两者共享 `l3_diversity_weight`。S 上的 orthogonal init（v9）也保留（无害）。

> 若只留一个，必须是 q_multi-loss（它直接管住病灶）。但因 S-loss 零成本且补足输入端
> 多样性，决定两个都留。

---

## 3. Architecture（multi_query forward，v10）

```
query_tokens = l3_summaries   ∈ [B, M=64, d_model=4096]   (L3 Q-Former 输出 S)
slots                          ∈ [B, N=128, slot_dim]

# --- key 侧（未改）---
k = normalize(K_sel(slots.detach()) + slot_key_bias)        # [B, N, S]

# --- query 侧（v10 加 LayerNorm）---
q_multi = normalize( q_sel_ln( Q_sel(query_tokens) ) )      # [B, M, S]  ← LN 新增

# --- 打分 + 聚合（未改）---
score  = einsum("bms,bns->bmn", q_multi, k) * temperature   # [B, M, N]
logits = logsumexp(score / tau_q, dim=1) * tau_q            # [B, N]
scores = softmax(logits)
idx    = topk(scores, k=top_k)                              # global top-k

# --- v10 可导 diversity loss（投影后空间）---
q_div  = mean_{b, i<j} relu( cos(q_multi_i, q_multi_j) - threshold )  # → aux

# --- 诊断（no_grad）---
S_max_cos          = pairwise_max_cos( normalize(query_tokens[0]) )   # 投影前
summary_q_max_cos  = pairwise_max_cos( q_multi[0] )                   # 投影后
```

`max_pool` / `chunk_query` 分支保持 v3 行为不变（不经过 q_sel_ln）。

---

## 4. Initialization

| 参数 | 初始化 | 理由 |
|------|--------|------|
| `q_sel_ln` (LayerNorm) | weight=1, bias=0（PyTorch 默认） | 标准 LN 初值；不引入偏置方向 |
| `Q_sel.weight` | normal_(std=0.02)（未改） | 沿用 v0，小初值避免早期 softmax 偏置 |
| `_q_multi_diversity_threshold` | 0.5（从 cfg.l3_diversity_threshold 注入） | 与 v9 S-loss threshold 一致 |
| `l3_diversity_weight` | 0.1（cfg 默认，未改） | 同时缩放 S-loss 和 q_multi-loss |

---

## 5. Relationship to prior work

- **v8**：引入 multi_query + logsumexp，但 routing query 经无约束 Q_sel 投影后塌缩，
  退化回 single_query。
- **v9**：加 S-space diversity loss，但作用在投影前，看不到塌缩——无效（3× weight
  实测无差别）。
- **v10（本版）**：定位到 Q_sel 投影塌缩，(a) 投影后加 LayerNorm 打破方向/模长塌缩，
  (b) diversity loss 迁到投影后空间 q_multi，直接约束 routing 实际用的空间。S-loss 保留
  作为输入端互补。
- 与 MoE router（Switch Transformer）的区别：MoE router 直接对 token 打分，无"先压成
  global query"的瓶颈；我们的 multi_query 用 M 个 L3 summary 作为 sub-query 保留 chunk
  内多样性，LayerNorm 防止投影端坍缩是这条路线特有的修补。

---

## 6. Known issues / 待验证

1. **训练侧尚未验证**：本次只做了单元 sanity（投影塌缩坐实 + 梯度流通 + 其它分支不受影响）。
   是否真能在 multi_query 训练里把 `summary_q_max_cos` 从 1.0 拉下来、把
   `top1_sim_mean` 抬过 0.05，需要一次实际 health check（**本任务不启动训练**）。
2. **per-layer selector**：selector 是 per-layer 的，但 q_multi-loss 与 v9 一致只在
   `layer_idx==0` 收集（避免 32× 放大、保持与 v9 量级可比）。其余层的 q_sel_ln 仍会训练
   （它们在 forward 中被使用），只是不额外计 diversity loss。
3. **LayerNorm 与 normalize 叠加**：q_sel_ln 之后紧跟 F.normalize，模长信息被 normalize
   抹掉，LN 的 scale 部分主要通过改变方向分布间接起作用——这正是我们想要的（拉开方向），
   但 LN 的 bias/weight 是否会收敛到退化解仍需训练观察。
4. **threshold 选择**：q_multi-loss 复用 0.5 阈值。投影后 128 维空间的"健康 max_cos"
   量级未知，可能需要单独调（留作后续 hyperparam）。
