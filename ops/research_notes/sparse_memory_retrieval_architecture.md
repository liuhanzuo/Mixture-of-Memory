# Sliding Window + Sparse Memory Retrieval 架构技术文档

**日期**: 2026-04-20
**目标模型**: Llama2-7B 全量微调
**训练资源**: 8×B200 (8*181GB HBM)
**目标**: 将 Llama2-7B 的有效上下文从 4K 扩展到 32K+ tokens，同时保持线性推理复杂度

> 更广泛的 LoRA+memory 失败案例分析见 `memory_injection_lora_survey.md`，本文不重复。

---

## 1. 相关工作深度调研

### 1.1 MemoryLLM (ICML 2024)

**论文**: "MEMORYLLM: Towards Self-Updatable Large Language Models" (arXiv: 2402.04624, ICML 2024)
**机构**: ByteDance

- **核心机制**: 在 Transformer 层间插入固定数量的 memory tokens，模型通过 attention 自然读写 memory。Memory 在 inference 间持久化。
- **Memory 更新策略**: Memory tokens 作为额外前缀 token，所有层 attention 都可 access。无显式 write/read gate，完全依赖 learned attention pattern。
- **训练方式**: Full fine-tuning，混合 memorization task + general NLP task
- **模型规模**: Llama2-7B, 13B, 70B
- **与我们设计的异同**: MemoryLLM 用 full attention over memory (O(T·N))，我们用 sparse top-k retrieval。我们无跨 session 持久化需求。MemoryLLM 证明了 7B 全量微调 + memory 可以 work，是我们的关键先例。

### 1.2 M+ (NeurIPS 2023)

**论文**: "Memorization in Large Language Models" (arXiv: 2310.15914)

- **核心机制**: 分析预训练 LLM 的 memorization 能力，通过 instruction tuning 增强记忆
- **Memory 更新策略**: 无显式 memory module，依赖内隐记忆
- **训练方式**: Instruction tuning
- **模型规模**: Llama-7B, 13B
- **异同**: M+ 不改架构，证明了"记住特定内容"可以通过 data engineering 实现，但容量和可控性有限。我们提供外部 memory 可控扩展。

### 1.3 RMT — Recurrent Memory Transformer (NeurIPS 2022 / AAAI 2024)

**论文**: "Beyond Attention: Breaking the Limits of Transformer Context Length with Recurrent Memory" (arXiv: 2304.11062)

- **核心机制**: 固定数量 memory tokens 作为 segment boundary 的 recurrent state，通过 BPTT 训练跨 segment 信息传递
- **Memory 更新策略**: Memory tokens 作为 segment prefix，无显式 write/read gate
- **训练方式**: Full fine-tuning + curriculum learning（从 1 segment 逐步增加到 N segments）
- **模型规模**: BERT-base, GPT-2 (124M), GPT-Neo (1.3B), OPT-125M~175B
- **异同**: RMT 需要 curriculum（我们不需），全 attention（我们 sparse），需要 BPTT（我们 online single-pass）。⚠️ 我们的实验中 RMT 在 Llama2-7B+LoRA 上失败，详见 `memory_injection_lora_survey.md`。

### 1.4 MemLong (arXiv: 2312.04465, 2024)

**论文**: "MemLong: Memory-Augmented Long-Context Understanding for Large Language Models"

- **核心机制**: Learnable memory bank + cross-attention 处理超长上下文，输入分段，每段通过 cross-attention 从 memory bank 读写
- **Memory 更新策略**: 显式 write encoder + cross-attention read
- **训练方式**: Two-stage (memory pre-training → task-specific fine-tuning)
- **模型规模**: Llama2-7B, 13B
- **异同**: MemLong 需要额外 write encoder 参数，我们共享 W_K/W_V。MemLong 用 full cross-attention，我们用 top-k sparse。

### 1.5 Memorizing Transformer (ICML 2022)

**论文**: "Memorizing Transformers" (arXiv: 2203.08913)
**机构**: FAIR

- **核心机制**: 可微 kNN lookup 替代 full attention，K/V pairs 存入外部 memory
- **Memory 更新策略**: Append-only (FIFO/eviction)，无压缩
- **训练方式**: 从头训练
- **模型规模**: 小模型
- **异同**: kNN retrieval 思路成熟，但 append-only 无压缩。我们用 EMA 实现压缩。

### 1.6 Longformer / BigBird (2020)

**论文**: Longformer (arXiv: 2004.05150), BigBird (arXiv: 2007.14062)

- **核心机制**: Sparse attention patterns (sliding window + global/random)，**无外部 memory**
- **训练方式**: From scratch / fine-tuning
- **模型规模**: Encoder-only, base/large
- **异同**: 我们在 sliding window 基础上增加了 memory retrieval path，获得全局信息同时保持 O(T)。

### 1.7 Elastic Memory (2025)

**论文**: "Elastic Memory: Scaling Long-Context LLM Inference with Adaptive Memory Hierarchy" (arXiv: 2501.13185)

- **核心机制**: 多级 memory hierarchy (HBM→SSD→disk)，基于热度动态 tier migration
- **Memory 更新策略**: 基于访问频率/重要性的动态迁移
- **训练方式**: 不需要额外训练（inference-time 优化）
- **异同**: Inference-time KV cache offloading，不改模型。与我们的 architecture-level memory 互补。

### 1.8 Hymba (2025)

**论文**: "Hymba: A Hybrid-head Architecture for Small Language Models" (arXiv: 2411.13614)

- **核心机制**: 混合 attention head（部分 softmax attention + 部分 SSM），推理时 SSM head 不需 KV cache
- **Memory 更新策略**: SSM hidden state 作为压缩记忆
- **训练方式**: 从头训练
- **模型规模**: 1.5B-2B
- **异同**: Hymba 是 head-level 硬分配（某些 head 是 SSM），我们是 token-level soft gating（gate 动态调节），更灵活。两者都体现了"混合 global+local 信号"的思路。

### 1.9 对比总结

| 方法 | Memory 类型 | Read 机制 | Write 机制 | 训练 | 最大规模 | 我们的关键差异 |
|------|------------|-----------|-----------|------|---------|--------------|
| MemoryLLM | Token emb | Full attn | Attention-based | Full FT | 70B | Sparse read |
| RMT | Token emb | Full attn | Attention-based | Full FT+curriculum | 175B | No curriculum, sparse |
| MemLong | Memory bank | Cross-attn | Write encoder | Two-stage FT | 13B | Shared W_K |
| Memorizing TF | K/V pairs | kNN | Append-only | From scratch | Small | EMA 压缩 |
| Longformer | 无 | Sparse attn | N/A | FT | Encoder | 我们有 memory |
| Hymba | SSM state | Softmax+SSM | SSM recurrence | From scratch | 2B | Fine-tuning |

---

## 2. 架构形式化

### 2.1 单层 Forward Pass 完整公式

输入 hidden states **H** ∈ ℝ^(T×d)（T=序列长度, d=4096）。

**投影**（共享 W_Q, W_K, W_V）:
```
q = H @ W_Q          # (T, d)
k_local = H_local @ W_K   # (w, d)  — sliding window
v_local = H_local @ W_V   # (w, d)
```

**Local Path** (sliding window attention):
```
o_tok = softmax(q @ k_local^T / sqrt(d_h)) @ v_local   # (T, d)
```
每个 token 的 softmax 只在 window 内 w 个 key 上计算。

**Memory Path** (sparse retrieval):

Memory bank **M** ∈ ℝ^(N×d):
```
k_mem = M @ W_K       # (N, d_h)
v_mem = M @ W_V       # (N, d_h)
s_t = q_t @ k_mem^T / sqrt(d_h)    # (N,)
idx_t = topk(s_t, k)                # (k,)
o_mem_t = softmax(s_t[idx_t]) @ v_mem[idx_t]   # (d_h,)
```

**Gated Fusion**:
```
g = sigmoid(W_g @ H + b_g)    # (T, 1), W_g ∈ ℝ^(d×1)
o = g ⊙ o_tok + (1-g) ⊙ o_mem
```

**Online Memory Update** (EMA):
```
M[write_idx] = α * h_t + (1-α) * M[write_idx]
```
write_idx 按 circular buffer 递增。

### 2.2 多层扩展

**方案 A: Per-layer Memory（推荐）**
- 每层 l 有独立 M^(l) ∈ ℝ^(N×d)
- 优点: 底层/高层捕获不同粒度信息，与 MemoryLLM 一致
- 缺点: 32×N×d 参数（N=128 时 ~64MB，可忽略）

**方案 B: Shared Memory**
- 所有层共享 M，作为全局信息枢纽
- 缺点: 底层和高层 memory 需求不同，共享可能产生 interference

**方案 C: Grouped Memory（折中）**
- 32 层分 G=4 组，每组共享

**推荐方案 A**。参数量极小（<100MB），独立性更安全。

### 2.3 Memory 容量

| 配置 | N | k | 等效 context | 适用 |
|------|---|---|-------------|------|
| Small | 64 | 4 | ~256 extra tok | 快速验证 |
| **Medium** | **128** | **8** | **~512 extra tok** | **MVP 推荐** |
| Large | 256 | 16 | ~1K extra tok | 最终目标 |
| XL | 512 | 32 | ~2K extra tok | 需验证收益 |

等效 context 解释: N=128 slots，每 slot 压缩 w=256 tok 信息 → 128×256=32K tokens 信息容量。

### 2.4 Write 策略

**EMA Write（推荐 MVP）**:
```
M[write_idx] = α * h_t + (1-α) * M[write_idx]
```
- 简单、无额外参数、稳定、天然压缩
- α 初始=0.1（慢更新），可学习

**FIFO Write**: 直接覆盖，无压缩。不推荐——退化为延迟 sliding window。

**Learned Write Gate**:
```
g_write = sigmoid(W_write @ h_t + b_write)
M[write_idx] = g_write * h_t + (1-g_write) * M[write_idx]
```
- 可学习何时写/写多少，但训练初期不稳定
- 建议: MVP 用 EMA，后续可替换

### 2.5 Read 策略

**Top-k Retrieval（推荐）**: dot-product similarity 选择 top-k。

替代方案:
1. **Temperature-scaled top-k**: s_t/τ 控制检索锐度
2. **Threshold-based**: 只检索 similarity>θ 的 slots，|idx|≤k
3. **Multi-head retrieval**: 不同 head 选择不同 top-k subset，增加 diversity

MVP 用标准 top-k (τ=1.0)。

### 2.6 Gated Fusion 设计空间

**Per-token shared gate（当前设计）**:
```
g_t = sigmoid(W_g @ h_t + b_g)  ∈ [0,1]
```

**Per-head gate**: 每个 attention head 独立 gate，g_t ∈ ℝ^(n_h)。更灵活但 32× 参数（仍很小）。可作为 ablation。

**Bias 初始化（关键决策）**:

| b_g init | σ(init) | 初始偏好 | 评价 |
|----------|---------|---------|------|
| +2.0 | ~0.88 | 偏 local | ✅ 推荐：初始 memory 为零，偏 local 合理 |
| 0.0 | 0.50 | 均匀 | ⚠️ 浪费 50% 计算在 zero memory |
| -2.0 | ~0.12 | 偏 memory | ❌ 初始 memory 为零，无意义 |

**最终推荐 b_g=+2.0**: 初始偏 local 但保留 12% memory 梯度信号。不要 +5.0（memory 无梯度→失败，与 RMT 同根因）。

### 2.7 Position Encoding

**Memory 不用 RoPE（推荐）**: Memory 存 content representation，不携带位置信息。Query 用 RoPE，memory key 不用 RoPE → 检索完全基于 content similarity，与绝对位置无关。

**理由**: Memory 内容可能来自序列中任意位置，给它固定位置编码无意义。

---

## 3. 训练方案

### 3.1 训练数据

- **主训练集**: RedPajama Books 子集 + Wikipedia，过滤后约 50-100B tokens（长文档为主）
- **验证集**: PG-19 test set + LongBench dev
- **不用 C4**: 网页文本太短，不适合长 context 训练
- **不需要合成 memorization task**: 我们的设计有显式 memory write/read

### 3.2 损失函数

**MVP: NTP-only**。标准 next token prediction，让模型自己学会何时写/读 memory。

可选辅助 loss（仅当 gate 坍缩时启用）:
- Gate utilization loss: KL(uniform ‖ mean(g))
- Memory coverage loss: -H(write_index_distribution)

### 3.3 Curriculum Learning

**本设计不需要 curriculum**。原因:
1. Online single-pass update，每个 token 直接更新 memory，不需 BPTT
2. Sliding window 保证 local loss 不为零，模型 fallback 到纯 local 也能工作

如果 gate 坍缩，可尝试 soft curriculum: Phase 1 (0-10% steps) 短文档≤2K, Phase 2 混合所有长度。

### 3.4 Memory 初始化

**Zero init + b_g=+2.0**: Memory 初始为零，gate 偏 local，不浪费计算。

### 3.5 Optimizer 设置

```python
{
    "optimizer": "AdamW",
    "lr": 2e-5,
    "betas": (0.9, 0.95),
    "weight_decay": 0.1,
    "scheduler": "cosine",
    "warmup_steps": 2000,
    "min_lr": 2e-6,
    "max_grad_norm": 1.0,
    "gate_lr_scale": 5.0,     # gate 是新参数，需更快学习
    "alpha_lr_scale": 0.1,    # EMA alpha 慢学习
}
```

### 3.6 预估训练时间和资源

| 配置 | Seq len | Tokens | 时间 (8×B200) |
|------|---------|--------|--------------|
| Fast | 4K | 10B | ~7h |
| Medium | 8K | 30B | ~25h |
| Full | 16K | 50B | ~120h |

**显存估算** (fp16 + gradient checkpointing):
- 模型: 14GB, Optimizer: 56GB, Gradients: 14GB, Activations: ~8GB
- **总计**: ~92GB/GPU (B200 有 192GB，充裕)
- Memory bank: ~2MB (忽略)

**推荐**: 先 Fast (7h) 验证 → Medium (25h) 可用模型 → Full 视情况

---

## 4. 复杂度分析

### 4.1 O(T(w+k)d) 精确推导

**Local attention**: O(T·w·d_h) (scores) + O(T·w·d_h) (output) = O(T·w·d_h)
**Memory retrieval**: O(T·N·d_h) (similarity) + O(T·N) (topk) + O(T·k·d_h) (output) = O(T·(N+k)·d_h)
**Projection**: O(T·d²) (所有方法共享)

Total = O(T·d² + T·(w+N+k)·d_h)

由于 d=4096, d_h=128, w+N+k≈400 → d²≫(w+N+k)·d_h，投影占主导。

### 4.2 与 Full Attention 对比

T=32K, d=4096, w=256, k=8, N=128:

| 方法 | Attention FLOPs | vs Full |
|------|----------------|---------|
| Full attention | T²·d_h ≈ 1.37×10¹¹ | 1.0× |
| SW only | T·w·d_h ≈ 1.07×10⁹ | 0.8% |
| **SW + Memory** | T(w+k+N)·d_h ≈ 1.60×10⁹ | **1.2%** |

**结论: attention 计算量减少 ~830×**，memory path 保持了全局信息。

### 4.3 Retrieval 优化

- **Chunked retrieval**: N>1024 时有用，MVP 不需要
- **ANN (HNSW/IVF)**: GPU 上 brute-force top-k 对 N≤1024 更快，不推荐

### 4.4 Memory 参数量占比

| 组件 | 参数量 |
|------|--------|
| Llama2-7B | 6.7B (99.97%) |
| Per-layer memory (N=128, L=32) | ~33M (K,V) |
| Gate (L=32) | ~131K |
| **Total new** | **~33M (0.5%)** |

新增参数 <1%，对训练和推理开销几乎无影响。

### 4.5 推理 I/O 开销

- KV cache (sliding window): 32×256×128×2×32×2 = **128MB** (vs full attention 32K: **16GB**)
- Memory bank: 32×128×4096×2 = **32MB**
- **总推理内存: ~160MB**，对比 full attention ~16GB，减少 ~100×

---

## 5. 评估方案

### 5.1 PG-19 PPL

- 序列长度: 16K, 32K, 64K
- Baseline: Llama2-7B 原始 + 纯 sliding window (无 memory)
- 预期: 32K+ 上显著优于纯 sliding window

### 5.2 LongBench

子任务: NarrativeQA, Qasper, MultiFieldQA, MuSiQue, HotpotQA, GovReport
主要验证不退化（LongBench ≤16K，我们的优势在更长序列）。

### 5.3 Passkey Retrieval

- 距离: 1K, 4K, 8K, 16K, 32K
- Passkey 位置: 开头/中间/结尾
- 预期: SW baseline 距离>w 时急剧下降，我们通过 memory 保持平缓

### 5.4 Needle-in-a-Haystack

Context 1K-128K, needle 位置 0-100%

### 5.5 Gate 分布分析（关键）

1. P(g) 分布 — 不应 degenerate 在 0 或 1
2. Gate vs token 位置 — 开头偏 local，远处偏 memory
3. Gate vs 层 — 底层/顶层可能不同
4. 训练过程中 gate 演变

**失败诊断**: g→1.0 (忽略 memory, 同 RMT v10) / g→0.0 (忽略 local) / g≈0.5 不变 (没学会分配)

### 5.6 Ablation Studies

| # | Ablation | 优先级 |
|---|----------|--------|
| a | w/o memory | 最高 |
| b | w/o gate (直接相加) | 高 |
| c | Different k (1,2,4,8,16,32) | 高 |
| d | Gate bias ablation (-2,-1,0,1,2) | 高 |
| e | FIFO vs EMA | 中 |
| f | Full memory (k=N) | 中 |
| g | w/o sliding window | 低 |
| h | Different N (32,64,128,256) | 低 |
| i | Shared vs per-layer memory | 低 |

---

## 6. 可行性分析与风险

### 6.1 为什么比 RMT 更可能成功

| 维度 | RMT (失败) | 我们 |
|------|-----------|------|
| Memory 读写 | 依赖 attention（间接） | 显式 write + top-k read |
| 梯度路径 | BPTT through segment | Online single-pass |
| 训练要求 | 必须用 curriculum | 不需要 |
| Attention 复杂度 | O(S²) per segment | O(T(w+k)d) |
| Memory 信息损失 | 完全依赖 softmax 传递 | EMA 显式压缩 |
| 可解释性 | 黑盒 | Gate 值可直接分析 |

核心论点: RMT 失败根因是 **gate 坍缩 + BPTT 梯度瓶颈 + attention 零和博弈** (见 `memory_injection_lora_survey.md` §4)。我们通过:
1. 显式 write gate — 不依赖 attention pattern
2. Online update — 不需 BPTT
3. Gated two-path — 分离 local 和 memory 信号

### 6.2 Gated Two-Path 解决 Softmax 零和博弈

**问题**: 标准 attention 中 memory token 和 local token 竞争 softmax 概率质量。增加对 memory 的 attend 意味着减少对 local 的 attend。模型倾向于走捷径——只用 local（因为 local 总是足够有用的）。

**我们的解法**: 两条路径完全分离，各自独立做 softmax，然后通过 gate 加权组合。不存在零和博弈——attend 更多 memory 不会减少 local 的 attention weight。

这是最关键的架构创新点。

### 6.3 潜在失败模式

| 失败模式 | 症状 | 原因 | 缓解 |
|---------|------|------|------|
| Gate 坍缩到 1.0 | PPL=SW baseline | 模型走捷径，不用 memory | 降低 b_g init; 加 gate utilization loss; 短文档 curriculum |
| Gate 坍缩到 0.0 | PPL 差于 baseline | Memory 过拟合 | 增大 b_g init; 加大 weight decay |
| Memory 退化 | Retrieval 效果不随训练改善 | EMA α 太大/太小 | Ablation α ∈ {0.01, 0.05, 0.1, 0.5} |
| Write 覆盖 | 重要信息被覆盖 | Circular buffer 太快 | 增大 N; 降低 write 频率 |
| Top-k 不相关 | Retrieval 到的 memory 无用 | K/Q 空间不对齐 | 检查 W_K 是否正确共享 |
| 预训练能力退化 | Short-context 任务变差 | 全量微调破坏原有能力 | 降低 lr; 加 replay data |

### 6.4 资源估算

- **训练**: 8×B200, Fast 配置 7h → 56 GPU-hours, 成本可控
- **存储**: Llama2-7B checkpoint ~14GB × 3 (model+optimizer+grad) = ~42GB
- **数据**: RedPajama 子集 ~100GB 下载
- **总计**: 1 个 B200 节点 1 天内可完成 MVP 验证

---

## 7. 实现路线图

### 7.1 MVP (第 1 周)

**目标**: 在 PG-19 上验证 sliding window + sparse memory 可以降低 PPL。

**Step 1: 修改 Llama2 Attention 模块**

需要修改的文件（基于 HuggingFace transformers）:
```
src/transformers/models/llama/modeling_llama.py
```

修改 `LlamaAttention` 类:
1. 添加 memory bank `M` 作为 `nn.Parameter` (或 `nn.Buffer`)
2. 添加 gate 参数 `W_g`, `b_g`
3. 添加 EMA alpha（可学习标量或固定值）
4. 修改 `forward()`:
   - Local path: 用 `torch.nn.functional.scaled_dot_product_attention` + `is_causal` + sliding window mask
   - Memory path: 计算 `q @ k_mem^T` → topk → gather → softmax → weighted sum
   - Gated fusion
   - Memory update (EMA write)

**关键实现细节**:
```python
class SparseMemoryAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        # 原始 Llama2 注意力参数不变
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        # 新增参数
        self.N = 128   # memory slots
        self.k = 8     # top-k
        self.w = 256   # sliding window
        self.alpha = 0.1  # EMA decay
        
        # Memory bank (per-layer)
        self.register_buffer('memory', torch.zeros(self.N, config.hidden_size))
        self.write_idx = 0  # circular buffer pointer
        
        # Gate
        self.gate_proj = nn.Linear(config.hidden_size, 1, bias=True)
        # 初始化 bias=+2.0
        nn.init.constant_(self.gate_proj.bias, 2.0)
    
    def forward(self, hidden_states, attention_mask=None, position_ids=None, ...):
        bsz, seq_len, _ = hidden_states.shape
        
        # 投影
        q = self.q_proj(hidden_states)  # (B, T, d)
        
        # === Local Path (Sliding Window) ===
        # 使用 Flash Attention with local mask
        # 注意: Llama2 已有 RoPE, 直接用
        k_local = self.k_proj(hidden_states)
        v_local = self.v_proj(hidden_states)
        
        # Sliding window attention (flash_attn 支持 local attention)
        o_tok = self._sliding_window_attention(q, k_local, v_local, self.w)
        
        # === Memory Path ===
        k_mem = self.k_proj(self.memory)  # (N, d)
        v_mem = self.v_proj(self.memory)  # (N, d)
        
        # Reshape for multi-head
        q_heads = q.view(bsz, seq_len, self.num_heads, self.head_dim)
        k_mem_heads = k_mem.view(1, self.N, self.num_heads, self.head_dim)
        v_mem_heads = v_mem.view(1, self.N, self.num_heads, self.head_dim)
        
        # Similarity + top-k (per query per head)
        sim = torch.einsum('bthd,nhd->bthn', q_heads, k_mem_heads) / math.sqrt(self.head_dim)
        topk_val, topk_idx = sim.topk(self.k, dim=-1)  # (B, T, H, k)
        
        # Gather top-k V
        v_mem_expanded = v_mem_heads.unsqueeze(0).expand(bsz, -1, -1, -1)  # (B, N, H, d_h)
        # Gather and weighted sum
        topk_weights = F.softmax(topk_val, dim=-1)  # (B, T, H, k)
        # ... gather v_mem[topk_idx] and compute weighted sum → o_mem
        
        o_mem = self._gather_and_weight(v_mem_expanded, topk_idx, topk_weights)  # (B, T, d)
        
        # === Gated Fusion ===
        gate = torch.sigmoid(self.gate_proj(hidden_states))  # (B, T, 1)
        output = gate * o_tok + (1 - gate) * o_mem
        
        # === Memory Update (EMA) ===
        with torch.no_grad():  # 或 enable grad for learnable α
            for t in range(seq_len):
                idx = self.write_idx % self.N
                self.memory[idx] = self.alpha * hidden_states[0, t] + (1 - self.alpha) * self.memory[idx]
                self.write_idx += 1
        
        return self.o_proj(output)
```

**⚠️ 注意**: Memory update 在训练时需要通过梯度流（如果 α 可学习），但 EMA 的 `memory` buffer 需要 `detach()` 来防止梯度穿过旧的 memory 值（否则显存爆炸）。

**推荐**: Memory 内容本身不需要梯度反传——梯度只需要通过 read path (k_mem, v_mem 的 W_K, W_V 投影) 和 write path (α 和 h_t) 流动。Memory buffer 用 `detach()`。

### 7.2 Step 2: 训练脚本架构

```
Mixture-of-Memory/
├── src/
│   ├── models/
│   │   ├── sparse_memory_llama.py    # 修改后的 LlamaAttention
│   │   └── memory_bank.py            # Memory bank 模块
│   ├── data/
│   │   ├── redpajama_loader.py       # 数据加载
│   │   └── pg19_loader.py            # PG-19 数据
│   ├── train.py                      # 主训练脚本
│   └── eval.py                       # 评估脚本
├── configs/
│   ├── mvp.yaml                      # MVP 配置
│   └── full.yaml                     # 完整训练配置
└── ops/
    └── research_notes/               # 本文档所在目录
```

**训练框架**: 使用 HuggingFace Trainer + DeepSpeed ZeRO-3 (或 FSDP)
- ZeRO-3: 将 optimizer states 和 gradients 分片到 8 个 GPU
- Gradient checkpointing: 减少激活显存
- Flash Attention 2: 加速 local attention

### 7.3 Step 3: 验证流程

1. **Sanity check** (1h): 在小数据 (1000 steps) 上训练，检查:
   - Gate 值是否从 ~0.88 开始变化
   - Memory 内容是否不再是零
   - Loss 是否在下降
   
2. **PG-19 PPL** (7h): Fast 配置训练，评估 PPL
   - 对比 baseline (纯 SW, 无 memory)
   - 如果 PPL 没改善 → 检查 gate 分布
   
3. **Passkey retrieval** (2h): 合成测试，检查 memory 是否能传递信息

### 7.4 完整路线图

| 周 | 任务 | 产出 |
|----|------|------|
| 1 | 实现 MVP + sanity check | 可运行的代码 |
| 2 | Fast 配置训练 + PG-19 评估 | PPL 数字 |
| 3 | Ablation (a)(b)(c)(d) | 架构设计验证 |
| 4 | Medium 配置训练 | 可用模型 |
| 5 | LongBench + Needle-in-Haystack | 完整评估报告 |
| 6 | 论文撰写 | 技术报告 |

---

## 附录: 关键超参数速查

```yaml
# MVP 推荐配置
model: llama2-7b
memory_slots: 128          # N
top_k: 8                  # k
sliding_window: 256       # w
ema_alpha: 0.1             # 写入衰减
gate_bias_init: 2.0        # 偏向 local
memory_rope: false         # memory 不用 RoPE
memory_init: zero          # 零初始化
per_layer_memory: true     # 每层独立 memory

# 训练
optimizer: adamw
lr: 2e-5
weight_decay: 0.1
warmup_steps: 2000
max_grad_norm: 1.0
gate_lr_scale: 5.0
seq_len: 4096
batch_size_per_gpu: 2
num_gpus: 8
data: redpajama_books_subset
tokens: 10B

# 评估
eval_datasets:
  - pg19_ppl
  - passkey_retrieval
  - longbench
  - needle_in_haystack
```

---

**文档结束**