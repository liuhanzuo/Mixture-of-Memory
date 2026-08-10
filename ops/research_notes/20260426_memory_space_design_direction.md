# Memory-Space Architecture — Prior-Art + Design Direction (v0)

**Date**: 2026-04-26
**Context**: user 2026-04-26 turn 1 提出的 memory-space 方案 — 每层维护一个 slot bank,
cross-attention 抽 top-k most-important slot,top-k + current-layer tokens **joint attention**,
再把更新后的 slot 表示 **写回** memory space。调研背景研究 agent `a70327cf516fece2b` 14:27 回报。

---

## 1. Prior-Art 结论(引用 agent 调研,原始表见 agent 输出)

最接近的工作是 **MemoryLLM (arXiv:2402.04624)** + **TTM (arXiv:2211.09119)** + **RMT (arXiv:2207.06881)** 的组合,但三者各缺一块:

| 方案 | per-layer | joint attn | top-k retrieval | writeback updated repr | per-slot identity |
|---|---|---|---|---|---|
| MemoryLLM | ✅ (7680/层) | ✅ | ❌ (全量参与) | ✅ (shift-append + 随机丢弃) | ❌ (slot 无稳定身份) |
| TTM | ❌ (单全局) | ✅ | partial | ✅ | ❌ |
| RMT | ❌ (输入级) | ✅ | ❌ | ✅ | ❌ |
| Memorizing Transformers | ❌ (仅顶层) | ❌ (gated 分叉) | ✅ | ❌ | — |
| Landmark Attention | ✅ | partial (仅路由) | ✅ | ❌ | — |
| Infini-Attention | ✅ | ❌ (线性压缩旁路) | ❌ | ✅ (delta 矩阵) | ❌ |
| **Ours (proposed)** | ✅ | ✅ | ✅ | ✅ (in-place 更新被选中的 slot) | ✅ |

**Novel gap**: per-layer × top-k retrieval × joint softmax × in-place writeback of selected slots 四者同时闭环,
目前公开文献中未见完整对应。最大 framing 风险是被 reviewer 视为 "MemoryLLM + top-k 稀疏化" 的增量,
需要在评测上强调 **slot 身份稳定性** 和 **long-range recall 增益**。

---

## 2. 架构草案 v0

### 2.1 每层结构(不改 backbone)

```
[Layer l]
  current_tokens:  H_l ∈ R^{B × T × d}         (T = 4k, d = 4096)
  memory_bank:     M_l ∈ R^{B × N × d}         (N = 128, per sample, per layer)

  1. Retrieval (top-k selection)
     scores = softmax(Q_sel(H_l_pooled) @ K_sel(M_l)^T / √d)   # [B, N]
     idx    = top_k(scores, k=16)                               # [B, k], straight-through
     M_sel  = gather(M_l, idx)                                  # [B, k, d]

  2. Joint attention (single softmax)
     KV_all = concat([M_sel_kv, H_l_kv])          # [B, k + T, d]
             (相当于 prepend k 个 "memory tokens" 到该层 KV 序列)
     O_l    = attention(H_l_q, KV_all)             # [B, T, d]  (只取 H_l 位置的输出)
     O_mem  = attention(M_sel_q, KV_all)           # [B, k, d]  (被选 slot 的更新表示)

  3. Writeback (in-place update of selected slots only)
     M_l_new = M_l.clone()
     M_l_new[idx] = (1 - β) · M_l[idx] + β · O_mem     # gated update, β 可学且 warmup
```

### 2.2 关键设计点

| 决策 | 选项 | 推荐 | 理由 |
|---|---|---|---|
| Joint attention 形式 | (a) prepend 到序列做 full attn (b) cross-attn 分叉 + gate | **(a) prepend** | agent 报告 Memorizing Transformers 等 gated 方案都退化成 ignore;必须共享 softmax |
| Top-k selector | (a) hard top-k + STE (b) Gumbel-softmax (c) load-balance | **(a) + load-balance aux loss** | hard top-k 简单;必须加 load-balance (MoE 的 importance/load loss) 防 slot collapse |
| Writeback 目标 | (a) raw H_l (MAG 做法) (b) updated repr O_mem | **(b) updated repr** | agent 明确指出写 raw hidden 会导致 slot 与该层 hidden 分布错配(MAG 失败原因之一) |
| Writeback gate β | 固定 0.1 / 可学 / warmup | **可学 + warmup (初始 0, tanh-gated)** | Flamingo-style 初始化;不破坏 base model 分布 |
| Slot 初始化 | zero / random / 首段 hidden copy | **首段 hidden 的 pooled copy + 少量噪声** | 保证初始态就在该层流形上,避免冷启动 |
| 跨段传递 | M_l^t → M_l^{t+1} | ✅ 直接传,不重置 | 这是 memory 的全部意义 |

### 2.3 Slot 身份稳定性(与 MemoryLLM 最大差异)

MemoryLLM 用 shift-append(新 slot 推到头部,老 slot 被挤出)导致**slot 位置与内容解耦**,
slot i 在 t=5 和 t=50 时意义完全不同。

我们的方案:**被选中的 slot 原地更新**,未被选中的 slot 完全保留。slot i 的内容沿时间单调演化
(可能长时间不动),**位置 i 即身份**。这给两个好处:
- 可以做 **slot probing**:固定位置 i 追踪信息演化
- **load-balance loss** 有意义:所有 slot 长期被均衡使用是训练目标

---

## 3. 训练 recipe(**必须 SFT**,这是用户本轮重点提问)

### 3.1 为什么必须训练

用户原话:"模型训练的时候是直接看到前面的所有 token,我们的记忆架构也需要 SFT 来训练?"

**完全正确**。理由:
- Llama-2-7B / Llama-3-8B 都是 full-attention 训练,从未见过"只看 window + slot"的分布
- Slot 向量是 EMA / projection 的产物,与 pretrain token embedding 的几何分布**完全不同**
- attention head 的 W_q/W_k **没有训练去读懂 slot** — 它们会把 slot 当噪声过滤(MAG 的 20% PPL 退化、RMT v1-v10 NIH=0% 都是这个原因)
- 我们当前 Q-Filters 所有数字都带这个 caveat:这是 **training-free 下限**,训练过后应该还能压更多

### 3.2 两阶段 recipe(参考 MemoryLLM / LongLLaMA)

**Stage 1 — Memory module warmup (frozen backbone)**
- 冻结 Llama 所有参数,只训练 {Q_sel, K_sel, V_sel, writeback gate β, slot init}
- 数据:PG19 + RedPajama-long 切成 4k 段,段间传递 memory
- Loss:LM loss + **load-balance aux loss**(slot utilization 均匀化)
- 规模:~1-3B tokens,8×H20 估计 2-3 天
- 验收:PPL 不退化 vs dense baseline(最低目标);slot utilization 分布熵 ≥ 0.8·log(N)

**Stage 2 — Joint SFT(unfreeze top layers)**
- 解冻 top 4-8 层 + LoRA(其他层冻结)
- 数据:PG19 长序列 + NIH / passkey 合成数据(10% 混合,**强制模型用到 slot**)
- Loss:LM loss + **fact-recall loss**(MemoryLLM 的 QA probe,监督"写回 N 次后 slot 仍能答对注入 fact")
- 规模:~500M-1B tokens
- 验收:**NIH@32k ≥ 80%**,**PPL 比 Q-Filters 同压缩比 baseline 好 ≥ 20%**

### 3.3 Curriculum(agent 强调的关键)

从短段序到长段序:
- Week 1: 2 段 × 4k = 8k 上下文,memory 传递 1 次
- Week 2: 4 段 × 4k = 16k,传递 3 次
- Week 3+: 8-16 段,传递 7-15 次
- 直接从长 rollout 开始会**完全不收敛**(RMT 1M 版的教训)

---

## 4. 前三大技术风险(agent 总结 + 我们的对策)

### R1: Slot collapse — top-k 退化到固定几个 slot
- **机制**: selector 贪心,某些 slot 被频繁选中→被更新→继续被选中,形成正反馈
- **对策**:
  - Load-balance loss(MoE 的 importance + load loss)
  - Slot dropout(训练时随机 mask 掉 5-10% 的 slot,强迫 selector 使用更广的集合)
  - 监控指标:slot 使用分布熵、长期未被选中的 slot 比例

### R2: Writeback 不稳定导致 PPL drift
- **机制**: updated repr 写回后被下段读取,反馈回路放大错误(RL bootstrapping 不稳)
- **对策**:
  - β 初始化为 0,tanh warmup 到 0.1-0.3(Flamingo 的 gated xattn 初始化)
  - 对 writeback 路径部分步 stop-grad(只在某些 segment 允许梯度回传)
  - 分层 β:底层 β 小(保持 "低级特征"稳定),顶层 β 大

### R3: Train/inference distribution mismatch
- **机制**: 训练时 segment 数少(≤16),推理时写回几十上百次,分布漂移(MemoryLLM 自己就观察到 >20k token PPL 衰减)
- **对策**:
  - 训练中加**长 horizon rollout**(teacher-forcing 混 autoregressive)
  - 显式监督"写回 N 次后 slot 仍能答对注入 fact"(MemoryLLM 的 knowledge-retention probe)
  - Inference 时可选:周期性重置 slot(类似 KV 重新计算),权衡速度 vs 稳定

---

## 5. 与当前 Q-Filters 工作的衔接

- Q-Filters 给我们 **training-free baseline**(Llama-3 rank=1 kv=1024 → PPL 2.365,dense floor 1.5468)
- 新 memory 架构的目标是:**训练后**在 kv=1024(或更小的有效 budget,例如 N=128 slots + 256 recent)做到 PPL ≤ 1.8(接近 dense,且只用 ~1/4 KV)
- 关键验收:
  - **NIH@32k ≥ 80%**(Q-Filters training-free 这点做不到)
  - **PG19 PPL ≤ 1.8**(超过 Q-Filters rank=1 kv=1024 的 2.365)

如果上面两点都达到,就是一个可发表的 story:**training-free Q-Filters 给 PPL 下限,training-based memory 补 retrieval 能力**。

---

## 6. 下一步建议(等 user 决定优先级)

候选方向(选一个开始):
1. **实现 v0 原型**: `src/memory/mem_space/` 下新增 compressor+retriever+writeback + 单层 unit test。估计 2-3 天 /coder。
2. **先补完 §11.4 retraction**: WikiText rank sweep + Streaming eval ≥32k(剩 2 项),再切架构。
3. **先做 Stage 1 可行性 demo**: 最小 1-layer prototype 在 pg19 上训 1h,看 PPL 是否不崩。

推荐 **#3 先做小规模 demo**,验证 joint-attn + writeback 的稳定性再投入大规模实现。

---

## 7. 扩展调研修订 (2026-04-26 14:44, agent aa178dbe8e74860a2)

第二轮 14-paper 调研增补的关键信息:

### 7.1 更接近的 prior art (需纳入 framing)

| Paper | 与我们的重叠 | 我们的 delta |
|---|---|---|
| **CAMELoT** (2402.13449) | top-k 检索 + joint self-attn | 我们: trainable + 写回 **updated repr**(CAMELoT: training-free + EMA 写 **raw KV**) |
| **Memformer** (2010.06891) | updated repr writeback | 我们: slots 作为 **first-class token 进 unified softmax**(Memformer: reader/writer 独立 cross-attn 模块,不共享 softmax) |
| **Focused / LongLLaMA** (2307.03170) | per-layer top-k + joint self-attn | 我们: **写回**(LongLLaMA 纯只读) + 不用对比学习预训练 retriever(但可以借他们的 **contrastive retrieval loss** 作为 Stage 2 辅助信号) |
| **Extended Mind** (2406.02332) | — | 专门研究 **RoPE OOD for slot tokens** — 支撑我们 v0 的 `use_rope_for_slots=False` 默认选择 |

### 7.2 最清晰的 story framing

**"MemoryLLM + 语义 top-k 检索替代 FIFO shift-append"** — 这是最锋利的 claim。MemoryLLM 的 retrieval 是时间近邻(FIFO),我们换成 query-aware 语义 top-k,配上 slot 身份稳定(in-place writeback)。

### 7.3 新增 risk R4 — RoPE OOD for slot tokens

- **机制**: slots 作为 token 进序列,位置编码没有天然定义;若复用 RoPE 会产生 OOD 距离
- **对策**:
  - v0 默认 `use_rope_for_slots=False`(给 slot 位置 0 或专属 sentinel)
  - Stage 2 可实验 learned slot position embedding 或 ALiBi-style slot bias
  - 参考: Extended Mind Transformers 2406.02332

### 7.4 Stage 1/2 recipe 增补

**Stage 1 aux loss 升级**(原方案只有 load-balance):
- **+ Contrastive retrieval loss** (LongLLaMA 风格): 正例=同 document 的历史 segment,负例=其他 document,提升 retriever 的语义可分性
- **+ Attention-reconstruction aux loss** (Compressive Transformer 1911.05507): 监督 selector 选出的 slot attention 分布与 full-attn baseline 一致

**R2(writeback drift) 修订 — 不要过度 stop-grad**:
- Memformer 的 MRBP(Memory Replay BPTT)说明 **writeback 路径的梯度必须回流**,纯 stop-grad 学不会写
- 原方案写的"某些 segment 允许梯度回传"需收紧:**至少 50% 的 segment 需要 BPTT 到 writeback**,否则 gate β 永远学不到有意义值

### 7.5 评测升级 — PPL 不够

MemoryLLM 本身就是 PPL 过关但 NIH 稀烂的教训。我们的验收不能只看 PPL:

| 评测 | 最低要求 |
|---|---|
| PG19 PPL | ≤ 1.8(vs Q-Filters rank=1 kv=1024 = 2.365) |
| **NIH @ 32k** | **≥ 80%**(Q-Filters training-free 做不到) |
| **LongBench** | ≥ 85% dense baseline(multi-doc QA + summarization) |
| **BABILong** | ≥ 70% @ 32k(reasoning chains) |
| **PPL-vs-distance** 曲线 | 20k+ token 后不崩(MemoryLLM 这里塌) |

---

## Sources(第一轮 + 第二轮调研合并)
- MemoryLLM: https://arxiv.org/abs/2402.04624
- M+ Extending MemoryLLM: https://arxiv.org/abs/2502.00592
- CAMELoT: https://arxiv.org/abs/2402.13449
- Memformer: https://arxiv.org/abs/2010.06891
- RMT: https://arxiv.org/abs/2207.06881
- TTM: https://arxiv.org/abs/2211.09119
- Memorizing Transformers: https://arxiv.org/abs/2203.08913
- Focused Transformer / LongLLaMA: https://arxiv.org/abs/2307.03170
- Landmark Attention: https://arxiv.org/abs/2305.16300
- Infini-Attention: https://arxiv.org/abs/2404.07143
- Compressive Transformers: https://arxiv.org/abs/1911.05507
- Extended Mind Transformers: https://arxiv.org/abs/2406.02332
- NTM: https://arxiv.org/abs/1410.5401
- Fast Weight Programmers: https://arxiv.org/abs/2102.11174
- UniMem survey: https://arxiv.org/abs/2402.03009
- Transformer-XL: https://arxiv.org/abs/1901.02860
