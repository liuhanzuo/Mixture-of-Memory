# Mixture-of-Memory：核心设计、进展与已解决的问题

> 日期：2026-06-02 · 对应代码 commit `d2ea7fc`（含 KEYSTONE 修复 `f73fd97`）
> 用途：一份自包含的现状总览——我们在做什么、卡过哪、解决了什么、现在在哪。
> 配套：`docs/ARCHITECTURE_AND_ROUTING_PROBLEM.md`（路由问题讨论稿，部分结论已被本文更新）、`status/MEMORY_PROTOCOL_PLAN.md`（后续实施计划）。

---

## 0. 一句话

让一个 **8B LLM** 用**固定大小的 per-layer memory bank（N=128 个 slot）**压缩任意长的上下文，使有限 KV budget 下也能处理超长序列。

核心机制（读 / 写 / 门控 / 多层共享）已经搭好并能跑。**过去几个月真正卡死的不是设计，而是一个梯度被切断的 bug**——它伪装成「路由塌缩」，让我们绕了 5 轮路由实验。**2026-06-02 定位并修复（commit `f73fd97`），用 toy task 证明 memory 闭环本身是通的。** 现在的工作从「救路由」转向「显式设计 write/read 学习协议」。

---

## 1. 架构（我们在做什么）

**Per-layer memory bank + Flamingo 风格 joint self-attention（KV-prepend）+ 门控写回**，配合 chunked streaming 把超长上下文切片喂入。

### 数据流（单层，单 chunk）

```
H ∈ [B, T, d]   (T=1024，当前 chunk 的 hidden states)
   │
   ├─(1) Selector：从 N=128 个 slot 里选 top-k=16 个
   │        idx ∈ [B, k]      ← 路由（routing）
   │
   ├─(2) M_sel = slots[idx] → 投影到 hidden → 拼到序列最前
   │        ext = [M_sel ; H]   (slot 放 pos=0，对全序列可见)
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

### 关键设计点（为什么这么搭）
- **32 层共享同一个 bank**（`shared_memory_bank=True`）：layer-i 写入立刻被 layer-(i+1) 读到，chunk 内 BPTT 穿透全深度。
- **chunk 间靠 slot 内容传递记忆**（每个 sample 重置 bank，不靠 BPTT 跨 chunk）——这条「跨 chunk 写入梯度」正是被 bug 切断的命脉（见 §3）。
- **不引入新 cross-attention 模块**：复用 Llama 自带 self-attn，靠 KV-prepend 把「读记忆」fold 进去。
- **跑两遍 + Flamingo 门**：避免单次拼接的 phantom-logit 分母污染（k 个近零 slot key 会衰减 H 自己的 attention，32 层叠加掉 60–90% 信号）；α=0 严格可逆退化是结构保证。
- **L3 summary 模块**：Q-Former 式 cross-attn pool，把 chunk 的 T 个 token 压成 64 个 dense summary token，既做 prepend 又当 multi-query routing 的 sub-query 弹药。

---

## 2. 走过的弯路（路由塌缩调查，5 轮）

很长一段时间，所有指标都指向「按内容寻址 slot 的路由训不起来」：

- `top1_sim ≈ 0.012`（= 1/128，均匀分布地板）→ selector 选 slot 跟掷骰子一样。
- 门控参数（β/α）几乎不动。
- 下游惨：BABILong ~25%、LongBench F1 ~7（vanilla baseline 34）——memory 不仅没帮上忙，还在污染生成。

我们系统性地试了 4 种「怎么把 chunk 压成 query 去打分」的方案，**全部塌缩**：

| 路由方案 | 机制 | 塌缩原因 |
|------|------|------|
| `max_pool` | 每个 slot 取它在 T 个 token 上的最高分 | 极值均衡：每个 slot 都能找到「冠军 token」，max-logit 都差不多 |
| `chunk_query`（mean-pool）| 1024 token 平均成 1 个 query | 过度泛化：万金油 query 对任何 slot 打分都差不多 |
| `multi_query` | L3 64 summary token 当 sub-query + logsumexp 聚合 | 仍塌缩 |
| `slot_query` | slot 作 query 去 attend chunk（temp40 最锐）| top1_sim 能到 0.32（寻址成功），**但下游仍无改善** |

**决定性的转折**：`slot_query+temp40` 把路由 top1_sim 救到 0.32、寻址 overlap 0.29，**但 toy task 的 retrieval_exact_acc 仍然全是 0**。这说明——**路由锐度和「记忆能不能读回」是解耦的**。路由是红鲱鱼，真正的病在别处。

---

## 3. ★ KEYSTONE：真正的根因（2026-06-02，commit `f73fd97`）

**根因是一个 bug：`src/memory/mem_space/memory_bank.py` 的 `slot_value_norm_cap` 在 `torch.no_grad()` 里 rebind `self.slots`，切断了跨 chunk 的写入梯度。**

```python
# 病灶（修复前）：在 no_grad 里把 slots 重新绑定到 detached tensor
with torch.no_grad():
    self.slots = self.slots / scale_all          # ← 跨 chunk 写入梯度死在这里

# 修复后（grad-preserving norm-cap，不在 no_grad 内）：
scale_all = scale_all.detach()                    # 只 detach 缩放因子
self.slots = self.slots / scale_all               # slots 本身保留计算图
```

配套修复：把 `inject_gate` 加入可训练参数（`_mem_space_params`）——之前它被冻在初值。

### 为什么这是命脉
架构设计里「chunk 间记忆靠 slot 内容传递」。要让 writer 学会「写入可被未来读回的内容」，loss 必须能从「未来 chunk 的读取」反传到「过去 chunk 的写入」。norm-cap 在 no_grad 里 rebind `self.slots`，等于在每个 chunk 边界把这条反传链剪断 → writer 永远收不到「你写的东西有没有用」的信号 → 写进去的是随机内容 → 路由再准也读不回 → 门控拿不到有效梯度。

**之前观察到的一切（路由塌缩、门控不动、下游差）都是这一个 severed-gradient bug 的下游表象。**

### 验证：toy task 被解出（grokking）
2-chunk passcode toy（chunk1 写 "The passcode is 7392."，chunk2 读 "The passcode is"→预测）：

- `toy_r2_w0_s42_long`（1500 步，**weight=0，无 recon 辅助 loss**）：`retrieval_exact_acc 0 → 0.188`，`lm 3.0 → 0.78`。
- **grokking 模式**：lm 在 ~3.0 平台期持续到 step~1045，然后相变骤降（1045→1195→1270→0.78）。
- **结论**：**cross-chunk 梯度修复本身（不需要任何辅助 loss）就解决了 toy。** recon / key-value 分离等是次要加速器，不是解锁的必需条件。
- ⚠️ 需要 ~1000+ 步才 grok（所有 800 步的旧 arm 都停在相变前的 lm~3.0 平台，这也解释了为什么早期实验全看着「没学到」）。

---

## 4. 已解决的问题清单

| # | 问题 | 根因 | 修复 | 状态 |
|---|------|------|------|------|
| 1 | **路由塌缩（数月）** | `slot_value_norm_cap` 在 no_grad 里 rebind slots，切断跨 chunk 写入梯度 | grad-preserving norm-cap | ✅ `f73fd97`，toy 验证（exact_acc 0→0.188，grokking） |
| 2 | **inject_gate 冻在 0.12** | 没加进 `_mem_space_params` | 加入可训练参数 | ✅ `f73fd97`（注：修后仍移动很小，留作开放问题） |
| 3 | **训练 step~490 必崩（确定性）** | `init_process_group(timeout=30min)` vs rank0 慢速 7.5GB CEPH checkpoint save → 其他 rank 在 barrier 超时 → watchdog SIGABRT | timeout 30min → 2h | ✅ `d2ea7fc`，当前 run 已平稳过 step 452 |
| 4 | **内联 BABILong eval 致 NCCL 崩** | DDP 循环里变长 greedy generation 让各 rank desync → ALLREDUCE 等满 watchdog | 训练 `--eval_interval 0`，eval 离线单独跑 | ✅ launch 脚本默认 |
| 5 | **heartbeat 成本 ~$10** | 会话内 cron 每 30min 把整段增长的对话当 input 重发 | 改为系统 crontab 跑无状态 `codebuddy -p` 全新进程，状态全靠 status/ 文件 | ✅ 本次（`scripts/heartbeat_cron.sh`，2h 一次） |

> 误诊记录（供反思）：问题 3 早期被当成「CEPH flakiness / eval bug / NCCL 网络」，做了 NCCL_IB hardening 没用——真因是那条显式写死的 30min timeout 撞上慢速 save。

---

## 5. 当前状态（2026-06-02 12:50）

- **本机 H20**：`dolmino_bugfix_slotq_t2h` 训练中（slot_query temp40, eval off, seed42, 2000 步, commit `d2ea7fc`）。step 452/2000，lm≈2.6，已平稳越过老崩溃点 → **timeout 修复确认生效**。
- **远程节点**：盘A 远程偶发 NCCL hang；盘B 两节点 `torch-base` env 缺 `transformers`，**暂不能跑本项目训练**。重要 run 优先本机。
- **集群**：4 个 H20 节点 = 32 卡（盘A 本机+远程 / 盘B 两节点），盘间需 rsync。

---

## 6. 下一步（详见 `status/MEMORY_PROTOCOL_PLAN.md`）

方向已从「指望 LM loss 自发涌现 slot 语义分工」转向「**显式设计 write/read 学习协议**」。按优先级、一次只动一个变量、先 toy 验证再上 Dolmino：

- **P1** summary reconstruction aux loss：写入的 slot value 用小 decoder 重建 chunk 的 L3 summary（`L_recon = MSE(S_hat, stopgrad(S_L3))`）。判据：toy exact_acc 能否从 0 起来 / grok 相变是否提前。
- **P2** read selector / write allocator 接口分离（读=找过去需要什么；写=新信息放哪，含 allocation 行为）。
- **P3** slot 拆 key（routing 用）/ value（readout 用），分别 norm，防 value norm 长大破坏 routing 空间。
- **P4** 双熵正则：单 chunk routing 要 sharp（low entropy）、全局 slot usage 要 balanced（防 dead slot）。
- **P5** read-back 一致性 weak supervision（synthetic KV：写入 slot set 未来要找得回）。

**核心原则**：先在 toy task 证明 slot 能学会稳定的 write→保持→read 协议（exact_acc 起来），再追 LongBench / BABILong。利用 32 卡把每个 Round 的等待从「串行天」压成「并行小时」。

---

## 7. 诊断指标速查（含精确定义）

> 记号：B=batch，T=chunk token 数（~1024），N=slot 数（128），k=top-k（16），M=L3 summary query 数（64），S=selector_dim。
> 训练 log 里 grep `QUERY_DIAG` / `WRITEBACK_DIAG`。每行打的是 **layer 0** 的诊断（避免 32 层刷屏）。

### `QUERY_DIAG` 行（routing 健康）

| 指标 | 代码定义（`layer.py` / `selector.py`）| 健康判据 |
|------|------|------|
| **top1_sim_mean** | `scores.max(dim=-1).values.mean()`——每个 batch item 选中 slot 的**最高 routing 分数**，在 B 上取均值。scores 是 softmax 后的 slot 概率分布 `[B,N]`。 | > 0.05 才算非均匀。地板 = 1/128 = 0.0078（=完全均匀，掷骰子）。|
| **per_tok_logit_std** | `logits.std(dim=-1).mean()`——单个 query 对 N 个 slot 打出的 raw logits（softmax 前）在 **slot 维度的标准差**，再在 B 上取均值。衡量「这些 slot 分数有没有拉开差距」。 | < 0.15 = 塌缩警报（所有 slot 分数几乎相等 → softmax 退化均匀）。 |
| **key_max_cos** | `K = normalize(K_sel(slots) + slot_key_bias)`，取 batch[0] 的 `K Kᵀ` 去掉对角线后的 **最大绝对 cos**。衡量 **slot key 之间** 是否互相塌缩成同一个向量。 | 健康 0.38~0.56（key 彼此可区分）。→1.0 = key 侧塌缩。**目前一直健康，证明病不在 key 侧。** |
| **retrieved_norm_mean** | 选中的 k 个 slot 向量（投影前）的 **L2 norm 均值**。 | 监控 slot 内容是否随 EMA 异常长大（曾观察到 →5，怀疑会破坏 routing 空间，见 plan P3）。 |

### `WRITEBACK_DIAG` 行（写回 / 门控）

| 指标 | 含义 | 判据 |
|------|------|------|
| **gate_val(beta)** | 写回门 β（slot ← gate·new + (1−gate)·old 里的 gate）。 | 钉在初值不动 = 路由没给有效梯度。**是结果，非病因**（KEYSTONE 修复前因梯度被切断）。 |
| **inject_gate_mean / std** | 读取注入门 α=sigmoid(inject_gate(h)) 的均值/标准差（Flamingo 融合 `next=bypass+α·(ext−bypass)`）。 | α≈0 = 模型在抑制记忆注入；修复后仍移动很小 = 开放问题。 |
| **slot_delta_abs_mean / max** | 本 step slot 内容的变化幅度。 | max 偏大（曾见 6.0）= 个别 slot 剧烈漂移。 |

### Multi-query 专用监控（`routing_pool_mode=multi_query` 时才有意义）

multi-query 的失败模式**不是 uniform**，而是「64 个 query 表面不同，却都挤到同一批 generic slot」。光看 top1_sim 不够：

| 指标 | 代码定义 | 解读 |
|------|------|------|
| **summary_q_max_cos / mean_cos** | 64 个 routing query（L3 summary 经 Q_sel 投影 + LayerNorm 后，**routing 实际使用的空间**）两两 cos，`max`/`off-diagonal mean`。 | max→1.0 = 64 query 已塌缩成同一个 → multi-query 退化回 single-query（坏）。 |
| **S_max_cos** | 同上，但在 **Q_sel 投影前** 的原始 L3 summary token（`query_tokens[0]`）上算。 | 与上一项对比诊断：`S_max_cos 低（0.6）但 summary_q_max_cos≈1.0` ⇒ **塌缩发生在 Q_sel 投影**，不是 L3 本身。 |
| **uniq_sel_slots** | 让 64 个 query 各自取 argmax（最爱的 1 个 slot），数有几个 **不同 slot**（batch[0]）。 | ≈16 → 所有 query 挤同一批 slot（坏）；≈60~100 → query 有分工（好）。 |

### slot_query 专用监控（`routing_pool_mode=slot_query`，当前用的就是这个）

| 指标 | 代码定义 | 解读 |
|------|------|------|
| **slot_attn_entropy** | 每个 slot 作 query 对 chunk 的 T 个 token 做 softmax attention `attn_w [B,N,T]`，算其在 T 维的**熵** `−Σ attn_w·log attn_w`，在 N、B 上取均值。logits 用 `(attn_w·attn).sum` 软最大池化（slot 匹配多个 token 分数更高，避免 max-pool 的「人人有冠军」均衡）。 | 高熵 = slot 把注意力摊平在所有 token 上（没学会定位，坏）；低熵 = slot 锐利定位到少数 token（好）。 |

### toy task 金标准（`toy_memory_bootstrap.py`）

| 指标 | 含义 |
|------|------|
| **retrieval_exact_acc** | chunk2 预测出的 passcode **完全匹配** chunk1 写入值的比例。比 top1_sim 更本质——直接量「写进去的能不能读回成答案」。**这是判断 memory 闭环是否真通的金标准。** |
| **chunk1to2_overlap** | chunk1 写入的 slot 集合 W 与 chunk2 读取的 slot 集合 R 的 \|W∩R\|/\|W\|（可寻址性）。 |
| **top1_sim / alpha_mean / slot_norm_delta** | 同训练侧，但在受控 2-chunk 任务上，归因更干净。 |
