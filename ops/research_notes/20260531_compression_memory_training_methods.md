# 压缩记忆系统的训练方法调研 — SFT vs Pretrain 稠密目标

**日期**: 2026-05-31
**背景**: v4g_topk16_v2 (content-based routing + TBPTT 稀疏读取 SFT) 出现 gate 冻结
(beta=0.500, alpha=0.463 训练 2000 步不动)、LongBench 泛化差 (avg F1=12.66 vs baseline 34)。
用户怀疑稀疏读取 SFT 范式本身不 work，考虑改为 pretrain 式稠密 next-token 目标。

---

## 0. 我们系统的现状 (代码分析)

### 0.1 训练范式 (train_mem_space_dolmino_cpt.py)

代码里实际有 **3 个 train step**，默认 Dolmino 走的是 TBPTT：

| step 函数 | context chunks | target chunk | 用在 |
|---|---|---|---|
| `dolmino_train_step` | `torch.no_grad()` 流过写 memory | 只有 target 算 LM loss | 未启用 (旧路径) |
| `dolmino_train_step_tbptt` (默认) | **每个 chunk 都 `labels=ctx`，各自 backward** | target 也 backward | line 1053 默认 |
| `babilong_train_step` | `no_grad` 流过 | 只有最后 chunk 算 loss | BABILong mix (15%) |

关键点：默认 Dolmino 路径 **每个 chunk 都有 LM loss**（不是纯稀疏末尾读），看似稠密。
但 **每个 chunk 之间 `_detach_banks(model)`**（line 643），把跨 chunk 的梯度图切断。

### 0.2 读/写顺序 (layer.py forward)

单个 chunk 内部：
1. 从 bank 取 slots（含**上一个 chunk 写入的内容**，但已 detach）
2. selector top-k → gather M_sel → prepend 到 H → 跑 wrapped layer（**read**）
3. `next_hidden = bypass_h + alpha * slot_delta + fast_mem_out`，alpha=tanh(slot_output_gate)
4. **writeback**：把本 chunk 的 O_mem 写回 bank（dual-gate g_in/g_forget）

因为 `shared_memory_bank=True`，32 层共享一个 bank：layer i 的 write → layer i+1 的 read
→ loss，所以 **within-chunk 跨层** 有梯度训练 write path。
但 **跨 chunk** 被 `_detach_banks` 切断 → "现在写好信息，让未来 chunk 预测更准" 这条
credit assignment **完全没有梯度**。

### 0.3 核心诊断 (代码层面)

1. **跨 chunk 梯度被 detach 切断**：memory 真正的用途（长程：chunk_i 写 → chunk_{i+k} 读）
   永远学不到梯度。target chunk 读的是 detached memory，而产生该 memory 的 write 拿不到
   target loss 的梯度。模型只学到 within-chunk 跨层 recurrence，不是"跨 chunk 记忆"。
   → 这才是"稀疏目标"的真正含义：不是 read 稀疏，而是 **write 的训练信号在时间维上稀疏/断裂**。

2. **alpha 是 per-layer 单标量，content-independent**：`slot_output_gate` 是 scalar，
   `alpha=tanh(0.5)=0.462`。它**不依赖检索是否相关**。LongBench 上 top1_sim≈0.015（随机）
   时 alpha 仍 = 0.462，memory 仍全强度注入 → 拖累生成。没有 input-conditioned gate 能说
   "这次检索是垃圾，别注入"。这就是 gate 冻结 + 泛化差的直接机制。

3. **beta/dual-gate 训练信号弱**：`slot_output_gate` 的梯度 = `slot_delta · dL/d(next_hidden)`。
   slot_delta 被 clip 到 bypass norm 且方向在未见数据上接近随机，batch 平均后梯度极小；
   叠加 lr=5e-6 极小 → 2000 步纹丝不动。

4. **可训练参数**：selector(Q_sel,K_sel,slot_key_bias)、gate_param、slot_output_gate、
   slot_to_hidden、hidden_to_slot、gate_proj_new/mem、gate_bias、l3_pool、l2_compressor。
   backbone 全冻结，lr=5e-6。**没有任何 autoencoding/重构预训练阶段**——memory 从未被
   单独训练过"能存住信息"。

---

## Part A：压缩记忆论文训练方法对比

### A.1 Activation Beacon (arXiv 2401.03462)

**TL;DR**: 稠密 next-token CPT（compression-based auto-regression），交错压缩，无独立预训练阶段，
只训练 beacon 专属投影，base 冻结。是与我们最对位的方法。

1. **训练目标**：**稠密 next-token**（"compression-based auto-regression"）。把长文本切成
   interval，每个 interval 的 activations(K/V) 被压缩成少量 beacon token；后续 token 预测时
   attend 到压缩后的 beacon KV 而非原始 KV。**每个 token 的 loss 都穿过压缩记忆**。同时用
   plain text + instruction data。每步**随机采样压缩比** (e.g. 2/4/8/.../128)，让模型支持任意比率。

2. **read 梯度怎么稠密**：核心是**交错压缩 (interleaved)**。序列被分成固定 interval（如 1024），
   每个 interval 末尾插入 k 个 beacon token，beacon 用专属 attention 读该 interval；之后整段
   只保留 beacon 的 KV，下一段的 token attend 这些压缩 KV。因为是流式 auto-regression，
   **每个位置的 next-token loss 都依赖之前所有 interval 的压缩 KV** → read 路径梯度天然稠密，
   不需要 detach。这与我们 `_detach_banks` 切断跨 chunk 梯度形成鲜明对比。

3. **独立预训练/重构阶段**：**没有**。直接 LM loss 端到端，不需要 autoencoding warmup。
   作者认为压缩 KV（而非 soft prompt）信息量足够大，LM loss 直接可学。

4. **数据**：RedPajama 采样的 plain text + 少量 instruction（如 LongAlpaca）。原版约 **1B token 量级**，
   训练长度上限 ~20K，但能外推到 128K。

5. **可训练参数**：**base 全冻结**。只为 beacon token 新增一套 attention 投影
   (Q/K/V/O 的 beacon 副本，类似给 beacon 加 LoRA-like 旁路) + beacon embedding。
   新参数初始化为 base 对应层的副本（warm copy），不是零初始化。

6. **gate/软选择**：**无显式 gate**。压缩与注入完全由 attention softmax 完成——beacon KV 直接进
   KV cache，token 通过 attention 自己决定看多少。没有我们这种 content-independent 标量 alpha。

### A.2 ICAE (arXiv 2307.06945)

**TL;DR**: 两阶段——先 autoencoding+LM **预训练**让 memory slots 学会"存信息"（重构原文），
再 instruction fine-tune。是"先让 memory 学会存，再学会用"的范式标杆。

1. **训练目标**：**两阶段**。
   - 阶段 1 (pretrain)：**autoencoding（重构）+ language modeling** 两个目标，在大规模纯文本上。
     AE 目标：给定 context → 编码成 k 个 memory slots → decoder 仅条件于 slots **重建原始 context**
     （加一个特殊 [AE] token 触发重构）。LM 目标：给定 slots，继续做 next-token。
   - 阶段 2 (instruction FT)：在指令数据上，条件于 memory slots 产生回答。
2. **read 梯度**：encoder（加 LoRA 的 LLM）把 context 压成固定 k 个 memory slot embedding；
   decoder（同一冻结 LLM）把 slots 当 soft prompt 前置，做重构/续写。梯度通过 decoder→slots→encoder
   全程可导。重构目标天然稠密：**每个被重建 token 的 loss 都穿过全部 slots**。
3. **"先学会存"的预训练阶段**：**这正是 ICAE 的核心卖点**。autoencoding 重构损失
   `L_AE = -Σ log p(x_t | slots)`（decoder 仅看 slots 重建原文 x）强迫 memory slots 成为
   原文的充分统计量。**没有这步，slots 学不到"存什么"**。这与我们系统最大的缺口——我们从未
   单独训练过 memory 能复原信息。
4. **数据**：阶段1 用 Pile/类似大规模纯文本（数百 M ~ 1B token 级），context 长度 512；
   阶段2 用 PwC（Prompt-with-Context）指令数据约 24万条。
5. **可训练参数**：base LLM **冻结**，加 **LoRA**（~1% 额外参数）到 LLM 作 encoder；
   memory slot embedding 可训练；decoder 复用同一冻结 LLM（无额外参数，或共享 LoRA）。
   memory slot token embedding 随机初始化。
6. **gate/软选择**：**无 gate**。固定 k 个 slot，全部作为 soft prompt 注入，无选择无门控。
   注意：ICAE 是固定压缩比、固定 slot 数，不做检索/top-k——没有"该不该注入"的问题。

### A.3 AutoCompressor (arXiv 2305.14788)

**TL;DR**: 稠密 LM 目标 + **segment 递归**，summary vector 累积，**全程梯度不 detach**
（BPTT over segments，含 stop-gradient 的随机化 segment 训练技巧）。这是 segment-recurrent 路线
里训练最稳的，且与我们 chunk 范式最像——但它**不 detach 跨 segment 梯度**。

1. **训练目标**：**稠密 unsupervised LM**。长文档切成 segment，每个 segment 产生 summary vectors，
   **后续所有 segment 的 LM loss 都条件于之前所有 summary vectors**（soft prompt 累积）。
   每个 token 的 next-token loss 都穿过累积 memory → 稠密。
2. **read 梯度**：summary vectors 作为 soft prompt 前置到下一 segment。关键技巧：
   - **summary accumulation**：所有历史 segment 的 summary 拼接，越往后 memory 越长。
   - **randomized segmenting + BPTT**：训练时随机切 segment 边界，梯度 **跨多个 segment 反传**
     （不像我们每 chunk detach）。为省显存用了 "stop-gradient on older summaries" 的部分截断，
     但相邻 segment 间梯度保留 → write 路径能学到"为后文压缩有用信息"。
3. **独立预训练/重构阶段**：**没有 AE 重构**。但它从已预训练 LM 继续训（继承 OPT/Llama-2），
   summary token 的能力靠 LM 目标长时间训练涌现。
4. **数据**：在 OPT/Llama-2 上 fine-tune，序列长达 **30,720 token**；语料为 RedPajama/Pile 类纯文本，
   规模数 B token（fine-tune 而非 from scratch）。
5. **可训练参数**：**full fine-tune 整个 LM**（不是冻结！这是与 ICAE/Beacon 的关键差异）
   + 新增 summary token embedding。后续工作可只训 soft prompt，但原版动了 backbone。
6. **gate/软选择**：**无 gate / 无 top-k 检索**。summary vectors 全部前置，attention 自行加权。

> 对我们的启示：AutoCompressor 证明 **"summary 累积 + 跨 segment 不 detach 的 LM 目标"** 足以让
> memory 学会跨段记忆，且不需要 gate/检索。我们的 `_detach_banks` 恰好砍掉了这条最关键的信号。

### A.4 Gist tokens (arXiv 2304.08467)

**TL;DR**: 纯 **instruction SFT**，靠**改 attention mask** 强制 prompt 信息流经 gist token，
**零额外训练成本**。是"SFT 也能 work"的最强证据，但任务设定关键不同（压 prompt 不是压长上下文）。

1. **训练目标**：**标准 instruction fine-tuning（SFT）**，无任何额外目标。loss 还是对 response
   的 next-token。唯一改动是 attention mask。
2. **read 梯度怎么稠密**：核心机制是 **gist mask**——把 prompt 后插入 k 个 gist token，并修改
   causal mask 使得 **response 的 token 无法直接 attend 到 prompt token，只能 attend gist token**。
   这强制 prompt 的全部信息必须"挤过" gist token 这个瓶颈。因为 response 每个 token 都只经 gist
   读 prompt，所以 read 路径对每个 response token 都稠密且**强制性**（不是可选注入）。
3. **独立预训练/重构阶段**：**没有**。无 AE，无 warmup。
4. **数据**：Alpaca+ 指令数据（约 13 万条指令）。规模很小——SFT 级别，不是 pretrain 级别。
5. **可训练参数**：原版 **full fine-tune** LLaMA-7B / FLAN-T5；gist token 是新增 embedding。
   "no additional cost over standard instruction finetuning"。
6. **gate/软选择**：**无 gate / 无检索**。gist token 数固定，靠 mask 强制。

> 关键差异：gisting 压缩的是 **prompt/指令**（短、且 response 必须用到它），不是**长文档上下文**
> （长、且大部分内容与最终问题无关）。它能用纯 SFT work，是因为
> (a) mask 强制信息必须经 gist 流动（瓶颈强制，不是可选），
> (b) prompt 信息与 response 强相关（监督信号密集且对齐）。
> 我们的设定两条都不满足——这正是 Part B 要回答的"SFT work 的必要条件"。

### A.5 2025 近期相关工作

选两篇互补的 2025 工作。

#### A.5.1 "Simple, Strong Baselines for Context Compression" / BenchPress (arXiv 2510.20797, 2025)
**TL;DR**: 系统对比 soft context compression 基线，发现 **causal compression-token（即我们这种
因果地让 token 写进 compression slot）显著弱于 (a) mean pooling 和 (b) 双向 compression token**。
1. **训练目标**：soft compression + 下游 reading comprehension/RAG，标准 LM/SFT。
2. **关键发现（对我们最重要）**：**计算压缩表示时用 bidirectional attention 远好于 causal**。
   我们的 write 是因果流式（chunk 内 causal + 跨 chunk detach），属于最弱的 causal 类。
   且 "simple mean pooling 就是很强的压缩算子" → 复杂 top-k 路由不一定比 pooling 强。
3. **数据/参数**：英文阅读理解，<8K 上下文；对比了不同压缩比与模型规模。
4. **gate**：无；强调 **bidirectional encoder 计算 compressed repr** 是涨点关键。

#### A.5.2 "Memory-Efficient Reasoning with Compression Beacons" (arXiv 2510.13797, 2025)
**TL;DR**: 用 **learned special token 周期性压缩 KV cache + 蒸馏/RL** 训练。
1. **训练目标**：**joint distillation + RL**（不是纯 SFT 也不是纯 LM）。用未压缩模型作 teacher
   蒸馏压缩模型，复用 RL rollout 降低开销。
2. **机制**：周期性把生成的 KV 压进 beacon token 并 evict，达到 memory-accuracy Pareto 改进。
3. **启示**：**distillation（teacher=无压缩 full-context 模型）是给 memory 提供稠密监督的另一条路**
   ——比稀疏 LM loss 信号强得多，因为 teacher 在每个位置都给出 full-context 的 logits 目标。

### A.6 Part A 对比表

| 方法 | 训练目标 | read 梯度稠密机制 | 独立"学会存"阶段 | 数据/规模 | 可训练参数 | gate/选择 |
|---|---|---|---|---|---|---|
| **Activation Beacon** | 稠密 next-token (compression-based AR)，随机压缩比 | 交错压缩 KV，流式 AR，每 token loss 经压缩 KV | 无 | plain+instr，~1B tok，≤20K | base 冻结，仅 beacon 投影副本 (warm init) | 无 gate，纯 attention |
| **ICAE** | 阶段1: **AE 重构 + LM**；阶段2: instr SFT | decoder 仅看 slots 重建原文，每 token 经全部 slots | **有（AE 重构）** ← 核心 | Pile 类 + 24万 PwC，ctx 512 | base 冻结 + LoRA(~1%) + slot emb | 无 gate，固定 k slot |
| **AutoCompressor** | 稠密 unsupervised LM，summary 累积 | segment 递归 + **跨 segment BPTT（不 detach）** | 无（继承预训练 LM） | RedPajama 类，≤30K tok，数 B | **full FT 整个 LM** + summary emb | 无 gate，无检索 |
| **Gist tokens** | 标准 instruction SFT | **gist mask 强制** response 只能经 gist 读 prompt | 无 | Alpaca 13万指令（SFT 级） | full FT + gist emb | 无 gate，mask 强制 |
| **2025 BenchPress** | soft compression + RC/RAG | **bidirectional encoder** 计算压缩表示 >> causal | 无 | <8K 英文 RC | 视基线而定 | 无；强调双向 + pooling |

**横向结论**：
- 5 篇里 **4 篇用稠密目标**（next-token 或 AE 重构每 token 都过 memory），唯一纯 SFT 的 Gist
  靠 **mask 强制 + prompt-response 强相关** 这个特殊设定才 work。
- **没有任何一篇用"流式写 + detach 跨段梯度 + 只在末尾读一次"** 这种我们的范式。
- ICAE 是唯一显式加了"先学会存"的 AE 预训练；其余靠稠密 LM 目标隐式逼出存储能力。
- **没有一篇用 content-independent 标量 gate**。要么无 gate（attention 自decide），要么 mask 强制。
- AutoCompressor 与我们最像（segment 递归），但它**不 detach 跨 segment 梯度**——这是关键分水岭。

---

## Part B：SFT 式记忆方法成功案例分析

### B.1 哪些"SFT 式"记忆压缩 work 了

| 方法 | 是否纯 SFT | work 的关键 | 与我们失败 SFT 的差别 |
|---|---|---|---|
| **Gist tokens** | 是（标准 instr SFT） | mask 强制 response 只经 gist 读 prompt；压缩对象=prompt（与 response 强相关） | 我们压长文档（大部分与问题无关），且无 mask 强制 |
| **xRAG (2405.13792)** | 是（冻结 LLM，只训一个 projector） | 压缩对象=**检索器已选出的相关文档**（外部 retriever 保证相关）；用 modality-bridging + **self-distillation** | 我们靠内部 top-k 路由选 slot，泛化数据上路由≈随机；无 teacher 蒸馏 |
| **ICAE 阶段2** | 阶段2 是 instr SFT，但**前面有 AE 预训练** | SFT 前 memory 已通过 AE 学会"存信息" | 我们直接 SFT，memory 从未学会存 |
| **LLMLingua 系列** | **training-free**（不训练，用小模型算 perplexity 删 token） | 不涉及可训练 memory，纯启发式剪枝 | 不可比，但说明"压缩"≠"可训练 memory" |

### B.2 SFT 能 work 的**必要条件**（综合 Part A + B.1）

1. **压缩对象与监督信号强相关**。Gist 压 prompt（response 必用），xRAG 压 retriever 选出的相关
   doc。我们压**整段长文档**，target chunk 的 next-token 大多只依赖**局部上下文**，与 memory 里
   存的远端信息**弱相关** → LM 梯度对 memory 几乎不产生有用信号。
2. **信息流必须被强制经过 memory（瓶颈），不能是"可选旁路"**。Gist 用 mask 物理切断 response→prompt
   的直连。我们的 memory 是 **加性旁路**（`bypass_h + alpha*slot_delta`）：模型完全可以靠 bypass_h
   自己预测，memory 是可选项 → 模型学会**忽略** memory（alpha 停在 0.46 不动，因为推不推都行）。
3. **要么有"先学会存"的阶段，要么用稠密目标隐式逼出存储**。纯 SFT 若两者都没有，memory 会塌缩成
   "对 loss 无贡献" → gate 冻结。
4. **(可选但强力) teacher 蒸馏**给每个位置稠密监督（xRAG、Compression Beacons 都用）。

> **判定我们的 SFT 失败根因**：上面 1、2、3 我们**全不满足**。
> - 压缩对象（长文档）与 target 的 LM 监督弱相关（条件1 ✗）
> - memory 是加性可选旁路，非强制瓶颈（条件2 ✗）
> - 无 AE 预训练、跨 chunk 梯度被 detach 导致目标在时间上稀疏（条件3 ✗）
> 三条同时缺失 → memory 对 loss 边际贡献≈0 → gate 无梯度 → 冻结。**这不是"SFT 一定不 work"，
> 而是"我们这版 SFT 缺了让 SFT work 的全部必要条件"。**

---

## Part C：诊断 — 我们的 gate 冻结 + 泛化差根因

### C.1 gate 冻结的机制链（代码级，confidence: high）

`alpha = tanh(slot_output_gate)`，是 **per-layer 单标量、content-independent**。
beta(写) 同理由 `gate_param` 单标量驱动。两者梯度：

```
dL/d(slot_output_gate) = (1 - alpha^2) · Σ_t < slot_delta_t , dL/d(next_hidden_t) >
```

- 训练分布内（BABILong），memory 偶尔有用 → 该内积偶尔为正 → alpha 想增大；但同时很多步无用 →
  正负抵消 → 净梯度≈0 → **alpha 停在初始 0.46**。这正是观测到的"纹丝不动"。
- 根因不是 lr 太小，而是 **gate 的梯度期望本身≈0**：因为 memory 是**加性可选旁路**，
  模型用 bypass_h 已能预测，slot_delta 与 loss 梯度方向**不稳定相关**。

### C.2 泛化差的机制（confidence: high）

- alpha content-independent → LongBench 上 top1_sim≈0.015（路由≈随机、取到无关 slot）时，
  `next_hidden = bypass_h + 0.46 * slot_delta`，仍**全强度注入一坨随机检索内容** → 污染生成 → F1 掉。
- 没有"检索质量低就少注入"的机制。Activation Beacon/ICAE 用 attention softmax 自然实现"没相关
  就不取"，我们用固定标量 alpha 把这个能力废掉了。

### C.3 "训练目标太稀疏"是否根因？（confidence: high，但需细分）

**是，但要精确定义"稀疏"**。不是 read 稀疏（chunk 内每 token 都读 memory），而是两层稀疏：
1. **时间维稀疏（最致命）**：`_detach_banks` 每 chunk 切断梯度 → "chunk_i 写好信息帮 chunk_{i+k}"
   这条**长程 credit assignment 完全无梯度**。memory 的唯一卖点（跨 chunk）从未被训练。
2. **监督相关性稀疏**：target 的 next-token 大多依赖局部，与远端 memory 弱相关 → 即便有梯度也很弱。

对比 AutoCompressor（跨 segment BPTT 不 detach）和 Beacon（流式 AR 每 token 经压缩 KV）——
它们的训练信号在时间维上**稠密且连续**。**这是我们与所有成功方法最本质的差别。**

---

## Part D：3 个可落地方案 (按优先级)

> 设计原则（从 Part A/B 提炼）：让训练信号在**时间维稠密**、让 memory 成为**强制瓶颈**、
> 让注入**content-conditioned**、(可选) 先用**重构**让 memory 学会存。

### 方案 1（最高优先级）：跨 chunk 不 detach 的稠密 LM 目标 + content-conditioned 注入门
**confidence: high**。这是 AutoCompressor + Beacon 的核心，改动最小、命中根因最准。

**解决的问题**：C.3 时间维稀疏（detach 切断长程梯度）+ C.1/C.2 gate content-independent。

**改哪些文件**：
- `scripts/train_mem_space_dolmino_cpt.py`：`dolmino_train_step_tbptt`。
- `src/memory/mem_space/layer.py`：`slot_output_gate` 改为 input-conditioned gate。

**改什么（伪代码级）**：
1. **去掉跨 chunk detach，做真 BPTT（窗口 2~4 chunk）**。把 `_detach_banks` 从"每 chunk"
   改为"每 K 个 chunk 才 detach 一次"，让相邻 chunk 间梯度连通：
   ```python
   # dolmino_train_step_tbptt 内
   for i, ctx in enumerate(all_chunks):
       out = model(ctx, labels=ctx)
       (out.loss/scale + aux/scale).backward(retain_graph=False)  # 仍每 chunk backward 省显存
       if (i+1) % bptt_window != 0:   # 关键：窗口内不 detach
           pass                        # 保留 bank 的 grad 连接
       else:
           _detach_banks(model)
   ```
   注意：真 BPTT 要 memory bank 的 read 在 backward 时还连着上个 chunk 的 write graph，
   因此**不能每 chunk 都 backward 后立即释放图**。折中：用 `bptt_window=2`，两个 chunk
   合成一个 graph 一起 backward，窗口末尾 detach。显存×2，H20 可接受（已有 gradient_checkpointing）。
2. **content-conditioned 注入门**替换标量 alpha：
   ```python
   # layer.py: 新增 self.inject_gate = nn.Linear(d_model, 1)  # 零初始化 bias 使初始≈当前
   g = torch.sigmoid(self.inject_gate(hidden_states))         # [B,T,1] per-token
   next_hidden = bypass_h + g * slot_delta + fast_mem_out
   ```
   让模型在检索无关（LongBench 随机 slot）时**自己学会把 g 压到 0**。梯度对 inject_gate 的权重
   是 `slot_delta · dL/dnext_hidden · hidden_states`，content-dependent → 不会塌成单点。

**为什么会 work**：把"chunk_i 写好帮 chunk_{i+1}"变成有梯度的目标（条件3），
且注入变 content-conditioned（条件2 部分满足、解决泛化）。这是性价比最高的一步。

---

### 方案 2（次优先）：加 ICAE 式 autoencoding 预训练阶段（"先学会存"）
**confidence: medium-high**。直接补 C 节诊断的"memory 从未学会存信息"。

**解决的问题**：条件3——memory 在 SFT 前先成为原文的充分统计量。

**改哪些文件**：
- 新增 `scripts/pretrain_mem_space_autoencoding.py`（fork dolmino 脚本）。
- `src/memory/mem_space/layer.py`：加一个 `forward_reconstruct` 路径（或复用 forward + 特殊 [AE] query）。

**改什么（伪代码级）**：
```python
# 阶段0（在 CPT 之前，~5k步）：
#   1. 用 write 路径把一个 chunk 压进 bank（slots）
#   2. 仅以 slots 为条件，重建该 chunk 的 token：
ctx = chunk                                   # [B, T]
_reset_banks(model); model(ctx, use_cache=False)   # 写满 bank（带梯度）
# decoder 只看 memory：把 H 置空/换成 [AE]+BOS，让它仅经 slots 预测 ctx
recon_logits = model.reconstruct_from_memory(prompt=[AE], target_len=T)
loss_ae = CE(recon_logits, ctx)               # 重构损失，每 token 都过 slots → 稠密
loss_ae.backward()
```
关键：重构时 decoder **不能看到原 chunk 的 token**（否则走捷径），只能看 slots。
这强迫 write 路径 + slots 学会"无损存下整段"。训完阶段0 再进方案1 的 CPT。

**为什么会 work**：ICAE 证明没有 AE 阶段 slots 学不到存什么。我们的 write path
(hidden_to_slot, dual-gate) 目前只有微弱 LM 梯度；AE 给它**稠密、强制、与存储直接对齐**的信号。
**风险/不确定**：我们 slot 是检索式 top-k（非 ICAE 固定 k），重构时要 attend 全部 slot 还是 top-k？
建议重构阶段 attend 全部 slot（关掉 top-k），让存储能力先建立。

---

### 方案 3（架构改造，最大改动）：改成 Activation Beacon 式交错压缩，read 梯度天然稠密
**confidence: medium**（命中根因最彻底，但改动最大、与现有 top-k 检索范式冲突）。

**解决的问题**：从架构上消灭"加性可选旁路"——让信息**强制**经压缩 KV 流动（条件2 满足）。

**改哪些文件**：
- `src/memory/mem_space/layer.py` + `patch.py`：把"prepend M_sel + 加性 slot_delta"改为
  "在序列里插入 beacon token，后续 token 的 KV 只保留 beacon"。
- `scripts/train_mem_space_dolmino_cpt.py`：去掉 chunk-detach，改流式 AR + 随机压缩比。

**改什么（伪代码级）**：
```python
# 每 interval（如 1024 token）末尾插入 k 个 beacon token
# beacon 用专属 K/V 投影读该 interval 的所有 token（双向）
# interval 结束后，丢弃原 token 的 KV，只把 beacon 的 KV 留进 cache
# 下一 interval 的 token attend [所有历史 beacon KV] + [本 interval 内 token]
for interval in stream:
    h = concat(interval_tokens, beacon_tokens)
    out = layer(h, kv_cache=past_beacon_kv)      # 流式，causal
    past_beacon_kv = append(past_beacon_kv, beacon_kv(out))   # 只留 beacon
loss = next_token_CE(all_positions)              # 每 token 都经压缩 KV → 稠密、强制
```
随机采样压缩比（每步 k∈{...}）。base 冻结，只训 beacon 专属投影（warm copy 初始化）。

**为什么会 work**：Beacon 在 128K 上做到接近无压缩 baseline，正是因为 read 强制经压缩 KV +
每 token 稠密 loss。这是论文里**最对位我们目标（固定 budget 压长上下文）**的方法。
**风险**：放弃 top-k 检索/slot 路由（项目已投入大量精力），相当于换架构。建议作为方案1/2
验证"稠密目标确实是关键"后的中期重构方向。

---

## Part E：特别评估

### E.1 是否该加 autoencoding 预训练阶段（ICAE 式）？
**建议：该加，作为方案2，优先级在方案1 之后。confidence: medium-high。**
- 理由：我们 memory **从未被单独训练过"能存信息"**。ICAE 明确证明没有 AE 阶段，slots 学不到存什么。
  我们的 write path（hidden_to_slot + dual-gate）目前只有微弱、间接的 LM 梯度。
- 但**不是单独靠它就够**：ICAE 的 AE 之后仍需 SFT 学"怎么用"。所以 AE 解决"存"，方案1 的稠密目标
  解决"用 + 跨段"。两者**互补**，建议组合：先 AE warmup（~5k 步）→ 再方案1 CPT。
- 实施注意：重构阶段先关掉 top-k（attend 全部 slot），让存储能力建立后再恢复检索。

### E.2 是否该改成 Activation Beacon 式交错压缩让 read 梯度稠密？
**建议：是正确的中期方向（方案3），但不作为第一步。confidence: medium。**
- Beacon 是 5 篇里**最对位我们目标**（固定 budget 压超长上下文，128K 接近无损）的方法，
  其"read 强制经压缩 KV + 每 token 稠密 loss"正是我们缺的两条。
- 但它要放弃 top-k 检索/slot 路由架构（项目已大量投入），改动最大。
- **务实路径**：先用方案1（去 detach + content gate，改动小）验证"稠密时间维目标 + content gate"
  是否就能让 LongBench 起色。若方案1 已显著改善，说明根因确认；再投入方案3 重构收益更稳。
  若方案1 改善有限，更要上方案3（说明加性旁路架构本身是天花板）。

### E.3 若坚持 SFT 怎么改才能 work（基于 Part B 必要条件）？
纯 SFT **可以 work**，但必须补齐 B.2 的必要条件，按重要性：
1. **强制瓶颈**（条件2，最关键）：把 memory 从"加性可选旁路"改成"强制路径"。最低成本做法：
   训练时**部分 mask 掉 bypass 的远端 token**（类似 gist mask），强迫 target 只能经 memory 拿远端信息。
   或直接上方案3 的 beacon mask。
2. **压缩对象与监督对齐**（条件1）：增大 BABILong/指令数据中 **target 答案确实依赖远端 context**
   的样本比例（QA 式，答案在远端 chunk）。当前 Dolmino 纯 LM 的 target 大多局部可预测，
   对 memory 几乎不施压。可提高 babilong_mix_fraction 或换更"长程依赖"的 SFT 数据。
3. **content-conditioned gate**（方案1 第2点）：避免无关检索强行注入。
4. **(强力可选) teacher 蒸馏**：用 full-context（无压缩）模型作 teacher，对每个位置的 logits 做 KD。
   这给 memory **每个 token 的稠密监督**，是 xRAG/Compression Beacons 成功的关键之一，
   且不依赖跨 chunk BPTT（绕开显存问题）。**若 BPTT 显存吃紧，KD 是性价比很高的替代。**

### E.4 一句话总结
我们的失败不是"SFT 不行"，而是**同时缺了"稠密时间维目标 / 强制瓶颈 / content gate / 先学会存"
这四件让压缩记忆 work 的事**。最小可行修复 = 方案1（去 detach + content gate）；
要更稳 = 叠加方案2（AE warmup）或 E.3.4（teacher 蒸馏）；彻底对位 SOTA = 方案3（Beacon 重构）。
