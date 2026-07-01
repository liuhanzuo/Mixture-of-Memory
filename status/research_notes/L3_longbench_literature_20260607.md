# L3 summary 模块 vs LongBench 领先压缩方法：文献调研 + 落地建议

> 调研者：general-purpose-9（2026-06-07）。纯调研，不改代码/不跑训练。
> 目标：对标 LongBench/聚合类任务上领先的"学习式摘要/记忆压缩"方法，找出能让本项目 **L3 summary 模块** 变好的具体改进。
> 已确认本项目现状：adapter LongBench F1≈2.9 vs base 13.9（聚合类大幅退化），BABILong 检索强 → slot/L3 学到 token 级 hidden 而非语义摘要。

---

## 0. 本项目 L3 现状（代码确认）

- L3 = Q-Former 式 attention pool：K 个 learnable query（`l3_summary.py:104`，v9 改 orthogonal init 防塌缩）→ 2 个 cross-attn block 从 chunk hidden pool 出 K=64 个 summary token（`l3_summary.py:107/150`）。
- summary 进 LM forward：extended 序列 `[L3(k_l3)|L1(k)|H(T)]`，H token 可 attend L3 keys（`layer.py:113-123`）；L3 还作为 multi-query 子查询喂 selector（`layer.py:848`）。
- **L3 唯一的直接信号是 `l3_diversity`（防 query 塌缩，`l3_summary.py:157`），内容全靠间接 LM loss 学** → 没有任何东西逼 L3 输出"语义对、可聚合"。
- recon aux 已存在但 **P1 在真 Dolmino 上 REJECTED**（`recon_decoder.py` + MEMORY_PROTOCOL_PLAN [P1]，norecon 8.74% > recon 6.89%）。

---

## 1. 文献对比表：LongBench/长上下文领先的学习式压缩方法

| 方法 (年份/出处) | 压缩机制 | **训练 objective**（关键） | 与本项目 L3 的差异 | 实证支持 |
|---|---|---|---|---|
| **ICAE** (In-context Autoencoder, Ge et al. ICLR 2024, arXiv:2307.06945) | LoRA encoder 把 context 压成 ~128 memory slot，**固定 LLM 当 decoder** | **(a) autoencoding：从 slot 重建原始输入 token（离散文本，过 decoder）；(b) text continuation。有 bottleneck：decoder 只能透过 slot 看上文** | 我们 recon 重建的是 **L3 连续 hidden（stopgrad）**、且 LM **没有 bottleneck**（H 仍能 attend 全部 H key）→ slot 不被迫承载内容 | ✅ 强（4× 压缩几乎不掉点） |
| **AutoCompressor** (Chevalier et al. EMNLP 2023, arXiv:2305.14788) | summary vector 当 soft prompt，**跨 segment 递归累积** | 标准 **LM loss**，但 summary 在多 segment 间累积 + 随机 segmenting；关键是**递归 carry-over** 让 summary 编码全局 | 我们 L3 已有 `_prev_summary` 递归钩子（`l3_summary.py:127/145`）但**默认未跨 chunk 真正循环训练**；且只有 LM loss 同样绕 | ✅ 中-强（长文 ICL/PPL 受益） |
| **Gisting** (Mu et al. NeurIPS 2023, arXiv:2304.08467) | 把 prompt 压成少量 **gist token** | LM loss + **attention masking：gist 之后的 token 物理上看不见 gist 之前的内容，一切必须经 gist token** → 等价蒸馏 | 这正是我们缺的 **bottleneck**：我们 H→L3/L1 是"额外可选"，没有"必须经过"的强制 | ✅ 强（26× 压缩、prompt 场景） |
| **RMT** (Recurrent Memory Transformer, Bulatov et al. 2022/2023) | 少量 memory token 在 segment 间**递归读写** | 端到端 task loss，无专门重建 | 强在 BABILong 检索（与我们 NIAH 强同源）；**对全局聚合无专门监督** | ✅ 中（BABILong 检索强，聚合弱） |
| LLMLingua (Jiang et al. 2023) | 训练小模型**丢 token**（硬压缩，离散） | 困惑度/对齐筛 token | 非 learnable summary token，与 L3 路线不同，借鉴价值低 | ✅ 中 |

**核心规律**：在 LongBench 聚合类任务上有效的 learnable 压缩，**几乎都不靠纯 LM loss**——要么 **autoencoding 重建离散输入文本（ICAE）**，要么 **attention bottleneck 强制信息流经压缩 token（Gisting）**。RMT 这类只靠 task loss + 递归的，结果正好是"检索强、聚合弱"——和我们现状一模一样。

---

## 2. 为什么文献的 recon 有效、我们的 P1 recon 没用

**不是 recon 本身没用，是我们的 recon 实现缺两个关键要素：**

1. **目标错**：ICAE 重建**离散输入 token**（语义完整、可读回文本）；我们 `recon_loss` 重建的是 **L3 连续 hidden 且 stopgrad**（`recon_decoder.py:143-151`）——拟合一个本身就没学好的连续向量，等于"互相迁就"成平凡解，不提供语义压力。
2. **缺 bottleneck**：ICAE/Gisting 的 decoder/LM **物理上只能透过 memory 看上文**；我们 H token 在 extended seq 里仍能 causal-attend 全部 H key（`layer.py:124`），slot/L3 是"可选旁路"，所以模型没有动力把内容写进 L3。P12 已猜到这点但没落地。

---

## 3. L3 改进建议（标 confidence + 改动量）

| 建议 | 机理（对照文献） | 改动量 | confidence |
|---|---|---|---|
| **A. L3 加 token 级 autoencoding 重建（ICAE 式）** | 新增 1 个 frozen-LM-tied 或小 decoder，从 L3 summary **重建 chunk 原始 token（CE loss，离散）**，而非重建连续 hidden | 中（写新 loss+decoder，复用 recon_decoder 骨架但改成预测 vocab logits） | **medium-high**（文献最强证据） |
| **B. H→chunk attention bottleneck（Gisting 式）** | 训练时**周期性切断 H 对早期 H token 的 attention**，强制信息经 L3/L1，逼 summary 承载内容 | 中（改 `_build_extended_attn_mask`，已有 P2 decoupled-read 钩子 `layer.py:214`，加 mask 即可） | medium |
| **C. 真正开启 L3 跨 chunk 递归（AutoCompressor/RMT 式）** | `_prev_summary` 钩子已存在，确保跨 chunk carry-over 且梯度流过，逼 L3 累积全局语义 | 小-中（确认递归在训练 loop 真生效 + 不 detach） | medium |
| D. scale L3 容量（n_summary 64→128、n_layers 2→4） | 纯加容量 | **调 flag**（但要从头训，改 ckpt 形状） | low（无新 objective 时多半只是更多没学好的 token） |
| E. 调 `l3_diversity_weight` | 只改多样性不改语义本质 | 调 flag | low |

**便宜 vs 贵**：D/E 是调 flag 但预期增益低；真正改"语义 vs token"本质的 A/B/C 都要写新代码（需 coder）。**A 与 C 可叠加**（ICAE 重建 + AutoCompressor 递归）。

---

## 4. 推荐的第一个 L3 改进实验（最小可落地）

**实验名：L3-autoencode（建议 A，文献证据最强）**
- baseline：现有最佳底座（P11 delta-rule + normalize_readout, chunk512），L3 默认 64/2L。
- arm：加 `--l3_recon_token_weight 0.1`（新 flag），新增一个小 decoder（复用 `recon_decoder.py` 骨架，输出改为 vocab logits），目标 = **重建该 chunk 的输入 token（CE）**，梯度回流 L3 pool。**关键：重建离散文本而非连续 hidden、不 stopgrad target（target 是真 token id，天然不塌缩）。**
- 因改 ckpt（新增 decoder）需从 base Llama-3-8B 重训或 strict=False warm-start。
- eval：现成 `scripts/eval_longbench_mem_space.py`（6 QA 聚合任务，对比 base F1=13.9 锚点）+ BABILong qa1/qa2/qa5 确认 NIAH 不退化。
- 判据：LongBench F1 是否从 ~3 往 base 13.9 方向抬，同时 BABILong 不崩。
- confidence：**medium-high**——直接照搬 LongBench 上被验证有效的 ICAE objective，且精准修正了 P1 recon 失败的两个根因（连续 hidden→离散 token、加重建压力）。

**次选**：若 A 改动大，先做 C（开启 L3 真递归，改动最小，AutoCompressor 证明递归累积有助全局），作为低成本探针。
