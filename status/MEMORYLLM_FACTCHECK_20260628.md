# MemoryLLM / M+ 实现核实报告 (Fact-Check)

- 日期: 2026-06-28
- 任务: 为 Mixture-of-Memory 论文 related work 差异定位, 核实 MemoryLLM 及其后续 M+ 的确切实现。
- 联网方式: 系统代理 (hy-proxy.woa.com:3128) + curl 抓取 arxiv abstract 页 + ar5iv 全文 HTML。
  - WebSearch: 命中一次 502 (上游临时故障), 改用 curl, 成功。
  - WebFetch: 被环境拦截 (走 claude.ai, 提示 "Unable to verify domain")。改用 curl, 成功。
- 数据来源 (均为联网实抓):
  - MemoryLLM 摘要: https://arxiv.org/abs/2402.04624 (og:description)
  - MemoryLLM 全文: https://ar5iv.org/abs/2402.04624 (398 KB HTML → 解析)
  - M+ 摘要: https://arxiv.org/abs/2502.00592
  - M+ 全文: https://ar5iv.org/abs/2502.00592 (149 KB HTML → 解析)

标注规则: **【确证-联网】** = 本次从论文原文核实; **【未核实】** = 仅训练知识/推断。

---

## 1. MemoryLLM ("MEMORYLLM: Towards Self-Updatable Large Language Models", arXiv 2402.04624)

### (a) memory 是什么? 维度? — 【确证-联网】
- memory = 一个**固定大小的 memory pool**, 由 **每一层** 的 "memory tokens" 组成 (不是只在输入层)。
- 记 θ = {θ_l}_{l=1}^L, L = transformer 层数。每层 θ_l 形状为 **N × d**:
  - N = 每层 memory token 数量;
  - d = **语言模型的 hidden size (word embedding 维度)**, 即 full d_model。
- 原文逐字: *"Each θ_l is of dimension N × d, corresponding to N hidden states and the word embedding dimension d in φ."*
- 主模型实例化: backbone = **Llama2-7B** (32 层, d=4096), **N = 7680** memory tokens/层。
  - 故 θ ∈ ℝ^{32 × 7680 × 4096}, 合计 **1.066B 参数** ("a 7B model + 1B memory pool")。

### (b) ★关键: vanilla MemoryLLM 存/读时【降维】吗? — 【确证-联网】: **不降维**
- **vanilla MemoryLLM 不做特征维度压缩 (no down-projection)。** 每个 memory token 都是 full d (=4096) 维向量。
- 它说的 "compress / 压缩" 指的是 **token 数量上的压缩**: 把一个 chunk x_c (n_x 个真实 token) 通过 φ_l forward, 取最后 **K** 个输出 hidden states 作为 K 个新 memory token (K=256)。即 "n_x 个 token → K 个 latent token" 的**数量压缩**, 而非 "d 维 → 更低维" 的特征降维。
- 结论: **用户印象中 "把 hidden 映射成更低维再存" 不适用于 vanilla MemoryLLM。** 那个低维投影出现在 M+ 的 retriever (见 §2b)。

### (c) 自更新机制 (注入新内容 / 丢弃旧 memory) — 【确证-联网】
- 注入新内容 (Self-Update, Fig.1b / §3.1.2):
  1. 取 θ_l 的**最后 K 个** memory token (e_θ^l), 与新 chunk 的 hidden states h_l 拼接, 作为 φ_l 的输入做 forward;
  2. 取输出的**最后 K 个 hidden states** 作为新 memory token e_θ^l'。(只用最后 K 个 token, 而非把整池都喂进去, 以省显存。)
- 丢弃旧 memory (forgetting):
  - **随机丢弃 (random drop)**: 从 θ_l 中**随机丢 K 个** token, 把剩余 token 左移压实 (θ^l(d)), 再把新 K 个 token 拼到右侧, 得到 θ_l'。
  - 这是 **graceful / exponential forgetting**: 每次更新统计上丢掉 K/N 比例的旧知识 → 知识以 K/N 速率指数遗忘 (作者类比 Ebbinghaus 遗忘曲线 / MemoryBank)。
  - 设计取向: 减小 K (压缩率)、增大 N (容量) → 遗忘更少。论文用 N=7680, K=256。

### (d) ★关键: 读出方式 = 【KV/inject 注入】还是【reforward 原始 token】? — 【确证-联网】: **inject (latent 前缀注入), 不是 reforward**
- 生成时 (Fig.1a, §3.1.1): **所有 memory token 作为前缀, 被当前 query 的 hidden states 通过 attention "attend"**。
  - 原文: *"all memory tokens in the l-th layer ... are attended by the hidden states h_l"*; *"e_θ^l is concatenated with h_l ... where h_l can attend to the preceding context e_θ^l"*; attention map 形状 n_x × (n_x + N)。
  - M+ 论文回顾 MemoryLLM 时也说: *"the memory pool θ_l is perceived using cross-attention"*。
- 即: 读出的是**已压缩的 latent memory token (hidden state 向量)**, 作为 KV/前缀注入到每层的 attention 中。
- **不存在 "把原始文本 token 重新 forward 整个模型" 的操作。** 与 reforward 范式根本不同。

### (e) 训练目标: CE? reconstruction loss? — 【确证-联网】
- 仅 **language modeling 交叉熵 (cross-entropy) loss**: 把前 n-1 段注入 memory 后, 在最后一段 x_n 上算 CE (next-token prediction)。多处 (§3.2.2, Alg.1 line 13/17/21) 均为 "Calculate the cross-entropy loss on x_n"。
- **没有显式 reconstruction loss。** 压缩质量靠 CE loss 经由 self-update 的梯度流隐式学习 (作者强调要保持 self-update 的 gradient flow)。
- 三个训练子任务都围绕 CE: (1) 单段注入后预测下一段; (2) 连续多段注入后预测末段; (3) 跨多文档注入后预测主文档末段 (缓解遗忘)。

---

## 2. M+ ("M+: Extending MemoryLLM with Scalable Long-Term Memory", arXiv 2502.00592)

### (a) 相比 MemoryLLM 加了什么? 是否引入 retriever? — 【确证-联网】: **是, 引入了 co-trained retriever**
- M+ = MemoryLLM + **long-term memory (LTM) 机制** + **联合训练 (co-trained) 的 retriever**。
- 关键改动:
  1. **短期/长期分离**: 原 MemoryLLM 的 memory pool θ 改称 **short-term memory**; 新增 **long-term memory Θ** (每层 Θ_l, 最大 M=150k tokens, **存在 CPU 上**, 不占 GPU)。
  2. **不再丢弃, 而是下沉**: MemoryLLM 中被随机丢的 K 个 token 会被永久丢弃; M+ 把这些被丢的 token **存入 LTM Θ**, 并给每个 token 一个 "age" 属性。Θ 满了就丢 age 最大 (最旧) 的。
  3. **retriever**: 生成时每层从 Θ_l 用 retriever 取 K_0 个 token, 按 age 排序, 与 short-term memory θ_l 拼接, 供 query 通过 cross-attention 读取。
  4. **Multi-LoRA**: 两套 LoRA, 一套用于 update (写), 一套用于 generation (读)。
  5. backbone 升级到 **Llama-3.1-8B**; 三阶段 data curriculum (continual train → 长文档 → 引入 LTM)。
  6. 配置: K=256, short-term N=10240, 从 LTM 取 K_0=2560, 合计 12,800 memory token/层。
- 效果: 知识保持从 <20k token 扩到 >160k token, GPU 开销相近。

### (b) ★降维是不是在 M+ 的 retriever 侧? — 【确证-联网】: **是, 而且确切是 d/20**
- retriever 有两个 projector: query 投影 f_q 和 key 投影 f_k, 均为 **两层 MLP**。
- 输出维度 **d_proj = d / 20** (远小于 d) —— 原文逐字: *"we set d_proj to be d/20 where d is the hidden size"*。
- 工作方式:
  - 把 token 下沉到 Θ 时, 对被丢的 memory token 施加 f_k, 得到 d/20 维的 **key 向量**, 单独存一个 key 池;
  - 生成时对 query hidden states 施加 f_q 得 d/20 维 query 向量, 与 key 做点积检索 top-K_0。
- **重要区分**: 降维 (d/20) 只发生在 **retrieval 的 key/query 投影空间**, 用于"检索打分"; **存储的 memory token 本身仍是 full d 维 hidden state** (key 只是额外的小指纹)。
- retriever 训练目标 = 对比式: min over f_q,f_k of [ -log(p+) - log(1-p-) ], 其中 p+ = <f_q(h_n), f_k(θ+)>, p- = <f_q(h_n), f_k(θ-)>; θ+ 为与 x_1..x_{n-1} 相关的 memory token, θ- 为无关的。
- 结论: **用户记忆中的 "降维" 准确对应 M+ 的 retriever key/query 投影 (d→d/20), 不是 vanilla MemoryLLM, 也不是 M+ 存储的 memory token 维度。**

---

## 3. 我们 vs MemoryLLM / M+ 的本质区别

我们的做法 = **训练一个 selector 选 top-K chunk → 把选中 chunk 的【原始 token】重新 forward 整个模型 (reforward, query 在场)**。

### 核心结论 — 【确证-联网, 据原文判定】
**是的, 本质区别就是 inject (注入压缩 latent memory 的 KV/前缀) vs reforward (重读原始 token)。**

- MemoryLLM / M+ 走的是 **compress-then-inject** 路线:
  - 把过去内容 (有损) 压成 **per-layer latent memory token (hidden state 向量)**;
  - 生成时把这些 latent token 作为 KV/前缀**注入**各层 attention, query 去 cross-attend;
  - **从不重新 forward 原始文本 token**; memory 的内容在写入时即被压缩固化, 与未来 query 无关 (query-independent encoding)。M+ 的 retriever 让"选哪些 latent token"变成 query-aware, 但 token 内容本身仍是 query-independent 压缩的。
- 我们走的是 **select-then-reforward** 路线:
  - 不压缩内容, 保留**原始 token**; "压缩"体现在 selector 的 top-K 选择上;
  - 把选中 chunk 的原始 token 与 query 一起**重新 forward 整个模型**, 得到 **query-conditioned 的完整表示** (full fidelity, 无有损 latent 压缩)。

### 差异对照表

| 维度 | MemoryLLM (2402.04624) | M+ (2502.00592) | 我们 (Mixture-of-Memory) |
|---|---|---|---|
| 存什么 | per-layer latent memory token (压缩 hidden state) | 同左 + CPU 上的长期 latent token 池 (含 age) | **原始 token (raw text), 不做 latent 压缩** |
| 压缩形式 | token 数量压缩 (chunk→K 个 latent), **不降特征维** | 同左; **retriever key/query 降维到 d/20** | 内容不压缩; "压缩"=selector 的 top-K 选择 |
| 读出方式 | **inject**: latent memory 作 KV/前缀, query cross-attend | 同左 (短期+检索到的长期 latent 拼接后注入) | **reforward**: 选中 chunk 原始 token 连同 query 重过整模型 |
| 检索/选择 | 无 retriever, 全部 memory 恒被 attend; 随机 drop 遗忘 | **co-trained retriever** (d/20 投影) 选 K_0 个 latent token | **训练的 selector** 选 top-K 个 chunk |
| query 感知 | 写入时与未来 query 无关 (query-independent) | 检索是 query-aware, 但 token 内容仍 query-independent 压缩 | **完全 query-conditioned** (reforward 时 query 在场) |
| 粒度 | 每层 latent token 池 | 每层短期池 + 每层长期池 | chunk 级原始 token |
| 保真度 | 有损 (压缩) | 有损 (压缩) | **无损 (原始 token)** |
| 遗忘 | 随机 drop, 指数遗忘 | 下沉 LTM, 满则丢最旧 (age) | 由 selector 决定, 非破坏式 |
| backbone | Llama2-7B (+1B memory) | Llama-3.1-8B | (本项目自有) |
| 训练目标 | LM 交叉熵 (无 reconstruction loss) | LM CE + retriever 对比 loss + multi-LoRA | (selector + LM, 见项目设计文档) |

---

## 4. 对写 related work 的建议

1. **定位轴线选 "inject vs reforward"**: 这是与 MemoryLLM/M+ 最干净、可被原文坐实的区分轴。一句话可写:
   *"Unlike latent-space memory methods such as MemoryLLM and M+, which compress past context into per-layer latent memory tokens and inject them as cross-attention prefixes at generation time, our method selects top-K chunks and re-forwards their original tokens through the full model jointly with the query, yielding query-conditioned, lossless representations rather than query-independent compressed memory."*

2. **务必纠正 "降维" 的归属, 避免 reviewer 抓错**:
   - 不要说 "MemoryLLM 把 hidden 降维存储" —— **vanilla MemoryLLM 不降维** (memory token 是 full d=4096)。
   - 若要提降维, 明确指向 **M+ 的 retriever**: key/query 投影到 d/20 仅用于检索打分, 存储的 memory token 仍是 full-d。

3. **强调三点我们独有的属性** (都能与原文形成对比):
   - (i) **lossless / 原始 token** vs 它们的有损 latent 压缩;
   - (ii) **query-conditioned encoding (reforward 时 query 在场)** vs 它们写入时 query-independent;
   - (iii) **selector 选 chunk** vs MemoryLLM 无检索 / M+ retriever 检索 latent token。

4. **承认相似点以示公允**: 三者都做 "选择性使用过去 + 固定/受控的工作集大小 + 端到端训练选择/压缩模块", 都旨在用有限 GPU 预算扩展有效上下文。我们的差异不在动机, 而在**记忆表示 (raw token vs latent) 和读出机制 (reforward vs inject)**。

5. **训练目标对比**: MemoryLLM 纯 LM CE、无 reconstruction loss; M+ 额外加 retriever 对比 loss + 双 LoRA。写作时可据此对比我们 selector 的训练信号来源。

---

## 5. 核实状态汇总

| 问题 | 结论 | 状态 |
|---|---|---|
| 1a memory 结构/维度 | per-layer, N×d, full d_model (N=7680, d=4096, 1.066B) | 【确证-联网】 |
| 1b vanilla 是否降维 | **不降维** (token 数量压缩, 非特征降维) | 【确证-联网】 |
| 1c 自更新/遗忘 | 取末 K + forward → 新 K token; 随机 drop K → 指数遗忘 | 【确证-联网】 |
| 1d 读出方式 | **inject** (latent memory 作 KV/前缀被 cross-attend), 非 reforward | 【确证-联网】 |
| 1e 训练目标 | LM 交叉熵, **无 reconstruction loss** | 【确证-联网】 |
| 2a M+ 新增 | 长期 memory (CPU, age) + **co-trained retriever** + multi-LoRA, Llama-3.1-8B | 【确证-联网】 |
| 2b 降维在 retriever 侧? | **是, retriever key/query 投影 d→d/20**; 存储 token 仍 full-d | 【确证-联网】 |
| 3 inject vs reforward | **是, 本质区别即此** (compress-inject vs select-reforward) | 【确证-联网】 |

全部 8 项均由联网抓取的论文原文 (arxiv abstract + ar5iv 全文) 坐实, **无未核实项**。
"我们的做法"细节以用户描述为准, 未逐字比对项目内部设计文档。
