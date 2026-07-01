# PAPER OUTLINE — Mixture-of-Memory (Select-then-Reforward Hidden Memory)

> 起草日期: 2026-06-28 · 作者: paper-writing agent (只读+写outline, 未跑实验/未改代码)
> 全部数字均为 **干净实测 (`babilong_mix=0`)**; 泄漏 b25「破墙」分数一律不引用。
> 数据状态标注: **[已确证]** = 干净实测、可直接进表; **[待补]** = 需补样本/补长度/补step才能锁定。
> 参考: CLEAN_SOTA_SURVEY_20260625 / FIFO_FINDINGS_SUMMARY_20260627 / MEMORYLLM_FACTCHECK_20260628 /
> RESEARCH_REFORWARD_COST_20260628 / HEARTBEAT_LATEST_20260628 / TOKEN_REFORWARD_DESIGN / LEARN_TO_SELECT_DESIGN.

---

## 工作标题 (候选)

- "Select-then-Reforward: Breaking the Read-Out Wall of Hidden Memory for Long-Context LLMs"
- "Re-reading Beats Re-injecting: Query-Conditioned Token Reforward over a FIFO Hidden Memory"

**一句话定位 (thesis):** 长程隐藏记忆的瓶颈不是「容量/遗忘」, 而是**两堵可分解的墙**——(1) **读出墙**: 冻结的 hidden 快照是 query-blind 的, 即使完美隔离 needle 也只能读到 ~20-24; (2) **选择墙**: 在大候选集上无监督打分信噪比不够。我们用 **token-reforward** (重读选中 chunk 的原始 token, query 在场重新 contextualize) 破读出墙, 用**训练的 reader-attn 选择器**部分破选择墙, 在 BABILong 长档上取得 2.4-3.6× SOTA, 并诚实给出长档选择的信息论上界。

---

## Abstract — 要点 + 数字

- 问题: 隐藏记忆 (per-layer latent) 读出长程信息差; 现有 latent-memory 方法 (MemoryLLM/M+) 走 compress-then-inject, 注入的是 query-independent 压缩 latent。
- 方法一句话: FIFO per-layer hidden memory + 训练的 reader-attn chunk 选择器 + **token-reforward** 读出 (select-then-reforward, 无损原 token, query-conditioned)。
- 核心发现: 把读出/选择拆成两堵独立的墙。读出墙被 token-reforward 破 (hidden-oracle ~20-24 → oracle-token qa5 8k=**66**); 选择可训练 (零训练 28 → 训练 step1000 **46**, 朝 oracle 66 走)。
- 结果: 干净部署 (K4) qa5 16k=**38**/32k=**32**, vs 真 SOTA 锚点 pg19 nctx7 16k=16/32k=9 → **2.4-3.6×**, 达 MemoryLLM teacher (32k=34) 水平, 无数据泄漏。
- 诚实边界: 长档选择是信息论墙 (大候选集打分信号被 distractor 稀释), eviction 假说被证伪。
- 数字盘点: 主结果 [已确证]; 完整 qa2 全档 / step3000 满 100 样本 / E0-E2 满 100 样本 = **[待补]**(见末节"数据缺口")。

---

## 1. Introduction — 要点 + 用哪些数字

1. **动机**: 长上下文 LLM 的有限工作集 → 记忆机制。把过去内容压成固定大小记忆是主流 (MemoryLLM/M+)。
2. **gap / 我们的反范式**: 主流是 **compress-then-inject** (有损 latent + KV 注入 + query-independent 编码)。我们提 **select-then-reforward** (无损原 token + 重新 forward + query-conditioned)。一句话对照 (见 §2 表)。
3. **关键洞察 (本文核心叙事, 按重要性)**:
   - (i) ★**读出墙**: 纯 hidden 读出受限。即使 oracle 完美隔离 needle 的 hidden 快照也只 ~20-24 (qa5 8k 训练前 12)。把选中 chunk 的**原始 token 重新 forward** (query 在场) → oracle 达 **66**。机制: 冻结快照 query-blind, 重读原 token 让 needle 在 query 在场下重新 contextualize。
   - (ii) ★**选择可训练**: 零训练 reader-attn 选择 qa5 8k=**28** → 监督训练 (T2 合成 needle, mix=0) → step1000=**46**, 朝 oracle 66 走 (曲线 28→39→46→平台)。qa1 不退化。
   - (iii) ★**长档选择墙 (诚实结果)**: 长档候选集大 (buffer cap=25, 16k/32k), reader-attn 打分信噪比不够, needle 中位 rank 7-10, recall@8 仅 0.34-0.54 ≈ chance 0.32 → 选不准。
4. **贡献清单 (4 点)**:
   - C1. 把长程隐藏记忆瓶颈**分解为读出墙 + 选择墙**两个可独立度量的问题 (诊断框架)。
   - C2. **token-reforward 读出机制**: 破读出墙 (hidden-oracle 20 → token-reforward 66), 且廉价部署路径 W0 经训练 12→28→34 也破 20 墙。
   - C3. **干净 SOTA**: 部署 qa5 16k=38/32k=32, 2.4-3.6× over 真 SOTA 锚点, 达 teacher 水平, 全程 mix=0 无泄漏。
   - C4. **诚实负结果 + 证伪**: 长档选择是信息论墙; eviction 假说被证伪 (keep_all 反而更差)。
5. **图建议**: Fig.1 = select-then-reforward 总览 (FIFO 流式写入 + reader-attn 选 chunk + 原 token 重 forward) 对照 compress-then-inject。

---

## 2. Related Work — 重点 MemoryLLM 对比

> 全部基于 MEMORYLLM_FACTCHECK_20260628 (联网核实原文, 8/8 项确证)。

1. **Latent / token-compression memory (核心对比)**:
   - **MemoryLLM** (2402.04624): per-layer memory pool N×d (N=7680, d=4096 full d_model, Llama2-7B + 1B memory)。**token 数量压缩** (chunk→K=256 latent token), **不降特征维**。读出 = **inject** (latent 作 KV/前缀被 query cross-attend)。遗忘 = **随机 drop** (指数遗忘)。训练 = 纯 LM CE, 无 reconstruction loss。
   - **M+** (2502.00592): + CPU 上长期 memory (age) + **co-trained retriever** (key/query 投影 **d→d/20**, 仅用于检索打分, 存储 token 仍 full-d) + multi-LoRA, Llama-3.1-8B。
   - **★务必纠正的常见误解** (避免 reviewer 抓): vanilla MemoryLLM **不降特征维**; 降维只在 M+ retriever 的检索投影侧。
2. **我们 vs 它们 (差异对照表, 直接搬 FACTCHECK §3)**:

   | 维度 | MemoryLLM | M+ | 我们 (Mixture-of-Memory) |
   |---|---|---|---|
   | 存什么 | per-layer latent token (压缩 hidden) | +CPU 长期 latent 池 | **原始 token, 不做 latent 压缩** |
   | 压缩形式 | token 数量压缩, 不降维 | +retriever key/query d→d/20 | "压缩"=selector 的 top-K 选择 |
   | 读出 | **inject** (latent 作 KV 前缀 cross-attend) | 同左 | **reforward** (原 token 连同 query 重过整模型) |
   | 选择 | 无 retriever, 全 attend; 随机遗忘 | co-trained retriever 选 latent | **训练的 selector 选 top-K chunk** |
   | query 感知 | 写入时 query-independent | 检索 query-aware, 内容仍 query-indep | **完全 query-conditioned** |
   | 保真度 | 有损压缩 | 有损压缩 | **无损原 token** |

3. **定位轴线一句话** (FACTCHECK 建议原句): *"Unlike latent-space memory methods such as MemoryLLM and M+, which compress past context into per-layer latent tokens and inject them as cross-attention prefixes, our method selects top-K chunks and re-forwards their original tokens through the full model jointly with the query, yielding query-conditioned, lossless representations."*
4. **实测支持反范式的证据 (放本节末或 Analysis)**: 我们实测 inject 式 (Method A raw-KV) 给对证据也只 +1-2.5 (frozen reader 用不上注入的 KV) → 才转 reforward。这是「为什么不走 inject」的实测理由, 不只是设计偏好。
5. **公允承认相似点**: 三者都做"选择性用过去 + 受控工作集 + 端到端训练选择/压缩"; 差异在**记忆表示 (raw token vs latent) 和读出机制 (reforward vs inject)**。
6. 其他相关: 长上下文/KV 压缩 (SnapKV 等 reader-native 选择思想)、recurrent memory (RMT)、retrieval-augmented。简述, 不展开。
   - **[待补]**: 这些次要 baseline 的精确引用与一句话差异化需补全 (当前文档只确证了 MemoryLLM/M+)。

---

## 3. Method — FIFO memory + 选择器 + token-reforward

### 3.1 FIFO per-layer hidden memory (背景设定)
- backbone = Llama-3-8B (32 层, d=4096, bf16)。文档切成 chunk (chunk_size=512)。
- 流式: chunk 依次 forward, 每层把 chunk 的 detached hidden 快照写入 per-layer FIFO buffer (cap=25 chunk); 读出时 reader 把 buffer 作前缀 attend。
- **关键性质 (后续诊断的根)**: 写入的是「该 chunk 当初作为 current 时算的 hidden 快照」, 从未 attend 过 query → query-blind。
- 图建议 Fig.2: FIFO 写入 + 快照 query-blind 示意。

### 3.2 读出墙与 token-reforward (核心方法)
- **诊断**: 纯 memory hidden 读出 (W0) 受限; 即使 hidden-oracle (完美隔离 needle 的 hidden 快照) 也只 ~20-24 → 证明问题在**表示**不在选择。
- **token-reforward**: 保留每 chunk 的**原始 token-id**; 读出时取选中 chunk 的原 token ∥ last chunk, 拼接成 window, **走全 32 层重新 forward** (query 在场)。
- 机制论证: 每层重新 attend query → needle↔query 多跳耦合重建; positions packed (concat 即连续 RoPE, in-distribution)。
- **存储洞察 (反直觉, 进 method 也进 cost)**: reforward 的读出 payload = token-id (32k 仅 ~0.26 MB), 比 slot bank (1 MB) 还小; 用「存 tiny token-id + 重算」换「存 4 GB 稠密 KV」。
- 图建议 Fig.3: hidden 快照读出 vs token-reforward 读出 的 data-flow 对照。

### 3.3 reader-attn chunk 选择器 (训练)
- **选择信号**: reader 自身的 native q·k salience (非 bolt-on head) —— 这是历史上唯一不崩到随机的选择器 (零训练 reader-attn precision 55% = 8.8× random)。
- **训练方案 (scheme c)**: T2 合成 needle 的已知 chunk 位置作监督, CE 把 per-chunk salience 推到 needle chunk (grad-bearing, 与 no_grad eval 同一 q·k); LM loss 经 token-reforward window 反传。mix=0 可训练。
- **训练/eval 一致性**: 选择层 (L16) 与 topk 训练 == eval; 选择层必须 unfreeze (L_select 才有梯度)。
- **关键防过拟合** (写进 method 的设计决策): needle 随机 chunk (非恒 chunk0) + ≥3 keys + 只在 held-out BABILong 判分 (不信 T2 loss)。
- 图建议 Fig.4: 选择器训练数据流 (T2 needle 监督 + token-reforward LM loss)。

### 3.4 部署变体
- **K (选几个 chunk)**: K=2 甜点 (in-distribution, ~6×W0, 不 OOM); K=4 上限 (18×, 偶发 OOM); K≥6 禁用 (OOM + 稀释反伤)。
- **廉价路径 W0**: 不做 reforward, 纯训练后的 hidden 读出 (见 §5 cost trade-off)。

---

## 4. Experiments

### 4.0 Setup
- backbone Llama-3-8B; BABILong qa1/qa2/qa5 × {0k,1k,2k,4k,8k,16k,32k}; chunk512; FIFO cap25; n=100 (部分 negative-result probe n=40/13, **需在表注明 n**)。
- **红线声明 (写进 setup, 这是本文公信力关键)**: 所有训练/eval `babilong_mix=0`; 不引用任何 `mix>0` 的泄漏分数 (历史 b25「破墙」65/76/68 是 ~85% 泄漏伪迹, 显式排除)。
- 真 SOTA 锚点 (干净): pg19 nctx7 qa5 = 75/73/51/29/**19/16/9** (0k-32k)。外部锚点: MemoryLLM teacher qa5 = 47/50/45/39/39/38/**34**。

### 4.1 主表 — SOTA (Table 1)
- 我们部署 (K4, token-reforward) qa5: 16k=**38** / 32k=**32**。
- 对比: pg19 nctx7 SOTA 16k=16/32k=9 → **2.4× / 3.6×**; 达 MemoryLLM teacher (32k=34) 水平。
- 表结构: 行 = {真 SOTA 锚点 pg19 nctx7, MemoryLLM teacher, 我们 W0, 我们 K4-reforward}; 列 = qa5 各长度。
- **[已确证]** qa5 8k/16k/32k 部署点; **[待补]** qa5 0k-4k 部署点补全、qa1 全档部署、**qa2 全档** (当前 qa2 长档部署点缺)。

### 4.2 机制消融 — 读出墙的分解 (Table 2, 核心)
- 阶梯 (qa5 8k/16k/32k, 干净 base): W0(纯 memory) **12/8/2** → hidden-oracle(隔离 needle 的**快照**) **~20-24** → oracle-token(隔离 needle 的**原 token**重 forward) **66/70/60**。
- 结论: 快照读出墙 ~20; token-reforward 破墙到 60-70。
- qa1 同阶梯佐证: 12 → 20 → 50 (8k)。
- **[待补]**: oracle-token qa5 32k 不同复核给出 60 与 50 两值 (HEARTBEAT 60 vs workflow 复核 50) → 需统一锁定; hidden-oracle 精确点 (20 vs 24) 补满 n=100。

### 4.3 选择训练曲线 (Fig.5 + Table 3)
- qa5 8k 部署 (reader-attn 选 + token-reforward): 零训练 C-probe **28** → step500 **39** → step1000 **46** → 平台 (step1000 后饱和), 朝 oracle 66 走。
- qa1 8k: 不退化 (14 → 平台)。
- **大 K 召回旁证**: step1500 K6 qa5 8k≈65≈oracle66, K2=46/K4=48 → **召回不足是 46→66 的主瓶颈, 非排序错** (8k 上 K 越大越逼近 oracle)。
- **[待补]**: step3000 满 100 样本锁定 (当前部分点 n=40); 训练曲线完整 step 序列补满。

### 4.4 W0 廉价路径也被训练改善 (Table 3 或并入 4.3)
- W0 (raw hidden, 无需 reforward): qa5 8k 12 → 28 → **34** (破 20 快照墙)。
- 意义: 无需 reforward 的廉价部署路径 (reforward 贵 3-6× 算力)。

### 4.5 长档诊断 (Table 4 + Fig.6, 诚实 negative result)
- 部署 vs oracle 缺口: K4 部署 16k=38/32k=32 vs oracle 70/60 → **头寸在但 reader-attn 够不着**。
- 选择 recall 诊断 (E2, 满档): reader-attn 长档 recall@4/@8/@16:
  - qa1 16k 0.17/0.35/0.68; qa5 16k 0.15/0.45/0.68; qa1 32k 0.00/0.31/0.77; qa5 32k 0.15/0.31/0.54。
  - chance @4/@8/@16 ≈ 0.16/0.32/0.64 → **全部 ≈ chance**, 中位 rank 7-10.5 (25 候选正中)。
- 结论: **信号弱被 distractor 稀释** (非完全随机, 非召回不足——扩 K 到 16 recall 也才 ~0.68≈chance 0.64); 短档候选小 (8k≈16 chunk) reader-attn 选得准, 长档候选大 (cap25) 选不准 = **信息论墙**。
- **[待补]**: E0/E2 recall 满 100 样本锁定 (当前 n=40/13)。

---

## 5. Analysis

### 5.1 两堵墙的分解 (本文中心论点)
- 总等式: **部署准确率 = 选择 precision × token-reforward 读出质量**。
- 读出墙: token-reforward 已破 (20→66)。选择墙: 短档可训练 (28→46), 长档信息论上界。
- 用一张图把 {W0, hidden-oracle, oracle-token, reader-attn 零训, reader-attn 训练后} 五点画在同一坐标 (Fig.7), 直观展示"墙在哪、破了哪堵"。

### 5.2 eviction 假说被证伪
- keep_all (装全 64 chunk, 不 evict) qa5 32k=**15** < 普通 evict (cap25) 32k=**32**。
- 结论: 长档瓶颈**不是 needle 被 evict**, 是大候选集选择精度下降 (1-of-64 难于 1-of-25); 加大 buffer 反而有害 (稀释更重)。

### 5.3 cost 分析 (Table 5)
- **存储**: reforward token-id 32k=0.26 MB (≈免费) vs 稠密全 KV 4 GB vs slot bank 1 MB。存储**不是**瓶颈。
- **算力**: 读出窗口 ∝(K+1)~(K+1)² (`use_cache=False` 每步重算 → 20× 冗余); K=2 慢 ~6×(8k)/~4×(16k, streaming 稀释), K=4 慢 18×, K≥6 OOM。
- **生产化优化** (纯工程, 不改机制): window 加 KV-cache (~20× 提速); 选择索引从 32 层 FIFO 砍到单层/池化 (6.25 GB → 8-256 MB)。
- 一句话权衡: 用「免费 token-id + 重算」换「4 GB KV」; 速度是 reforward 唯一真代价, 可工程优化。
- **[已确证]**: 存储为码算精确值; 速度为实测墙钟 (注意 A/B 两组机型不可跨组比, 表内分组)。

### 5.4 为什么 inject 不行 (呼应 §2)
- 实测 Method A raw-KV (inject 式) 给对证据只 +1-2.5: frozen reader 用不上注入的 KV → 印证 reforward 的必要性 (不只是表示无损, 还因 frozen reader 只会用自己 attend 出来的东西)。

---

## 6. Limitations (诚实)

1. **长档选择信噪比 = 信息论墙**: 大候选集 (cap25, 16k/32k) reader-attn 打分 ≈ chance, 扩 K 救不动。短档 (8k) 才是 SOTA 战场; 长档部署 38/32 是"选不准"而非"读不出"(oracle 头寸在)。
2. **reforward 算力**: K=2 慢 3-6×, K≥6 OOM; 当前 eval 未加 window KV-cache (probe 实现, 生产需补)。
3. **训练数据合成性**: 选择器用 T2 合成 needle 监督, T2→BABILong 迁移历史上是 death-list 风险 (已用随机 needle/≥3 keys/held-out 判分缓解, 但泛化到真实长档仍有限)。
4. **选择器为单层 q·k**: 多层投票/层级选择 (HNST) 未做, 是长档可能的后续方向。
5. **规模/任务范围**: 仅 Llama-3-8B + BABILong qa1/qa2/qa5; RULER/LongEval 下游迁移在相关探索里 ≈0 (SWA readout gain 不迁移到那些格式) —— 需诚实提及范围限制。

---

## 7. Conclusion

- 把长程隐藏记忆瓶颈分解为**读出墙 + 选择墙**。
- token-reforward 破读出墙 (select-then-reforward 反 compress-then-inject 范式); 选择短档可训练、长档是信息论墙。
- 干净 (mix=0) 部署 2.4-3.6× SOTA, 达 teacher 水平。
- 未来: 长档选择器 (多层投票/层级导航/非 reader-attn 信号); reforward 的 KV-cache 生产化。

---

## 图表清单 (汇总)

| 编号 | 内容 | 数据状态 |
|---|---|---|
| Fig.1 | select-then-reforward 总览 vs compress-then-inject | 概念图, 可画 |
| Fig.2 | FIFO 写入 + query-blind 快照 | 概念图 |
| Fig.3 | hidden 快照读出 vs token-reforward data-flow | 概念图 |
| Fig.4 | 选择器训练数据流 (T2 监督 + reforward LM loss) | 概念图 |
| Fig.5 | 选择训练曲线 28→39→46→平台 + oracle 66 线 | [部分待补满 n/step] |
| Fig.6 | 长档 recall@k vs chance (E2) | [待补满 100] |
| Fig.7 | 五点墙分解坐标图 (W0/hidden-oracle/oracle-token/reader-attn 零训/训练后) | [待补] |
| Table 1 | 主表 SOTA (我们 vs pg19 nctx7 vs teacher) | [qa5 核心已确证; qa1/qa2 全档待补] |
| Table 2 | 读出墙阶梯 (W0/hidden-oracle/oracle-token) | [oracle-token 32k 需统一; hidden-oracle 补满] |
| Table 3 | 选择训练曲线 + W0 改善 | [step3000 满 100 待补] |
| Table 4 | 长档诊断 (部署 vs oracle + recall) | [E0/E2 满 100 待补] |
| Table 5 | cost (存储 + 速度) | [已确证, 速度分机型组] |

---

## ★ 数据缺口总表 (确证 vs 待补) — 写作前必须对齐

### 已确证 (可直接进表/进文)
- 读出墙阶梯主干: W0 qa5 8k=12, hidden-oracle ~20-24, oracle-token qa5 8k=66/16k=70。
- 选择训练曲线主干: C-probe 28 → step500 39 → step1000 46 (平台); qa1 不退化; W0 12→28→34。
- 部署 SOTA 核心点: K4 qa5 16k=38/32k=32; 锚点 pg19 nctx7 16k=16/32k=9; teacher 32k=34。
- eviction 证伪: keep_all 32k=15 < evict 32k=32。
- 长档 recall ≈ chance (定性确证, 趋势稳): E2 recall@4/8/16 各档。
- cost: 存储码算值 + 速度实测墙钟 (分 A/B 机型组)。
- MemoryLLM/M+ 差异: 8/8 项联网确证。

### 待补 (开写可先占位, 但终稿需补)
1. **qa2 全档** — 当前主表/消融以 qa1/qa5 为主, qa2 长档部署点缺, 需补 qa2 8k/16k/32k (部署 + oracle)。
2. **step3000 满 100 样本** — 训练曲线终点目前部分 n=40, 需 step3000 全档 n=100 锁定平台值 (确认 step1000 后确实饱和而非继续涨/崩)。
3. **E0 / E2 recall 满 100** — 长档诊断 recall 当前 n=40 (16k) / n=13 (32k), 统计噪声大, 需补满 100 才能在论文里下"≈chance"的硬结论。
4. **oracle-token qa5 32k 统一** — HEARTBEAT 记 60, workflow 复核记 50; 需一次干净复跑统一 (影响 Table 2 与 Fig.7 的 32k 上界)。
5. **hidden-oracle 精确点** — "~20-24" 区间需补满 n=100 给出单值 (qa5 各档)。
6. **qa1 全档部署点** — 主表 qa1 行需补全 0k-32k。
7. **次要 baseline 引用** — Related Work 中 SnapKV/RMT/retrieval 等的精确引用与差异化句子 (当前仅 MemoryLLM/M+ 确证)。
8. **(可选) 下游迁移** — 若声称通用性, 需 RULER/LongEval 干净点 (现有探索 ≈0, 需诚实定位为"BABILong-specific readout gain")。

### 红线 (写作期间持续遵守)
- 只用 `babilong_mix=0` 数字; 泄漏 b25 (65/76/68 等) 完全不碰。
- 表内任何点必须标 n (样本数); n<100 的点标注"preliminary"。
- A 组 (taskpool 95GB) 与 B 组 (a1000) 速度禁止跨组比, 表内分组。
