# QCMem 论文骨架 (draft)

> 2026-07-08 初稿。汇总本项目 definitive 证据(全 n=50/100 官方判分)。定位: 效率/超长上下文 + 分工机制解释。
> 配套详据: `status/QCMEM_ASSESSMENT.md`(完整数据) / `status/RUN_REGISTRY.md`(逐 run)。

---

## 标题(候选)
**Depth-Partitioned Retrieval Readout: Caching Mid-Layer Hidden States for Unbounded-Context Memory**

## 一句话贡献
把长上下文处理从"重算部分 token 的全深度 KV"(token-partial, 现有 RAG-KV-reuse)转成"缓存单个 mid-depth hidden + 重算上层"(layer-partial)。read 计算与上下文长度无关(固定 ~6657 tok)、显存恒定(~18GB), 在超出 backbone 外推能力的超长档(128k+)做到 full-context 完全做不到的事。

---

## 1. 方法 (QCMem)
- **split at depth j**(Qwen3-8B L=36, j=12)。
- **WRITE**: 每 512-tok chunk 过 layers[0:j] 缓存 depth-j 残差流 hidden h_j(chunk-local RoPE)。存储 ~1/(2L) full KV。
- **READ**: bm25 检索 topk chunk → pack[sink; 选中 h_j; query h_j] 全新 RoPE 重算 layers[j:] → logits。
- **j 旋钮**: j=0=full RAG 重算(self_test 精确等于 full forward, max diff 0), j=L=closed-book。
- **训练**: 自蒸馏 LoRA(teacher j=0 / student j=12, 纯 PG19 零合成数据)。

## 2. 核心结果

### 2.1 超长上下文精度 (RULER niah_single, Qwen3-8B, n=50, string_match_all)
| 长度 | 8k | 16k | 32k | 64k | 128k | 256k |
|--|--|--|--|--|--|--|
| QCMem | 100 | 100 | 100 | 100 | **100** | **98** |
| full-context | 100 | 100 | 100 | 100 | **0** | **OOM** |
| StreamingLLM(同budget) | 90 | 42 | — | 16 | **4** | — |
- 分水岭 = backbone 外推极限(Qwen 64k→128k)。>该点只有 QCMem 可用。
- vs StreamingLLM(同 6657-tok/16.9GB 固定 budget): 128k 100 vs 4(25×)。检索保留"相关"上下文, StreamingLLM 只保留"最近"→丢中间→niah 针 miss。

### 2.2 效率 (bench_qcmem_vs_fullctx.json, median-of-3)
| 长度 | prefill 加速(full/QCMem) | full 显存 | QCMem 显存 |
|--|--|--|--|
| 32k | 2.5× | 34GB | 18GB |
| 64k | 4.4× | 52GB | 18GB |
| 128k | **7.8×** | 89GB | **18GB(恒定)** |
- QCMem read 长度与上下文无关 → prefill O(L)(write) + 固定 read; full O(L²)。

### 2.3 泛化 + 跨 backbone
- 纯 PG19 自蒸馏(零 babilong/RULER) → RULER niah zero-shot 强(证通用记忆非特化)。
- Llama-3-8B 自蒸馏也 work(niah_single 100), Qwen 难任务更强(backbone 质量差异)。
- babilong: qa1/16k=57 vs MemoryLLM 20(2.85×), qa5/16k=63-67 vs 38。

## 3. 机制: 为什么缓存中层有效 —— 理解-生成分工

### 3.1 分工命题 (跨 backbone robust)
- **probing**(相关): 语义任务中层达峰、顶层回落; next-token 仅顶层成形。Qwen+Llama 两曲线分离 True/True。
- **3a 截断下游**(因果): 只用前 ~4-8 层(深度 0.12-0.22)达全模型 95% 下游语义; 中层全面超顶层(RTE 中层>顶层 +0.06~0.10); Part C verbalizer 顶层反超中层 probe(RTE native 0.79 > probe 0.62) = 顶层用"表征线性可分性"换"生成有用性"。Qwen+Llama 一致。
- **精确表述**: 相对分工(理解层 < 生成层), 语义绝对深度 backbone-dependent(不说"固定第12层")。

### 3.2 与 QCMem 挂钩
- QCMem 零训练 j-sweep(RULER niah 16k): j≤9(深0.25)=100, j12 崖跌14, j18=0。**可缓存深度上限 ≈ 3a 理解饱和点**。
- 缓存"理解已完成"的层(≤j)无损, 重算上层(生成策略)基于 query 重新执行 → **方向B(缓存顶层)判负**(顶层是生成层, 缓存丢 query-conditioning)。
- 自蒸馏把可缓存上限从 j9 推到 j12+(训练学会从更深缓存 readout)。

### 3.3 semantic-bottleneck pretrain (可行性 + QCMem-friendly 已验)
- 1B from-scratch, layer-6 加 rank-512 funnel: **前 7 层 next-token acc=0.000**(baseline 从 L1 渐进), 生成推到 bottleneck 后; top-acc 几乎无损(0.331 vs 0.334)。→ 分工可被**几乎无损地显式强化**。
- **★缓存点更 QCMem-friendly (已验, commit 0a68ad5)**: bottleneck 模型 layer-6 hidden vs baseline:
  | | baseline | bottleneck |
  |--|--|--|
  | PCA dim@99% | 1859 | **427** |
  | effective rank | 407 | **149** |
  | 压缩 rank-256 后 readout 掉 | +0.028 | **+0.001(几乎无损)** |
  | 压缩 rank-64 后 readout 掉 | +0.133 | **+0.036** |
  - bottleneck 缓存点内在维度骤降(被 funnel 强制入 ≤512 子空间)+ 压缩后 readout 远更 robust。可压缩性 collapse 仅在 j=6 独有(其他层 dim99>1800)。
- **★救赎 §5 的"浅层不可压"证伪**: vanilla 浅层不可压(baseline dim99=1859 复现证伪)是 vanilla 性质; semantic-bottleneck pretrain **能显式制造缓存点可压性** → 缓存更省 + readout 抗压。这把"可压缩"从 QCMem 的软肋变成 pretrain 的设计产物。
- caveat: 1B/2000步未收敛, 读**相对差异**(不含糊)非绝对 acc(~0.31)。

### 3.4 bottleneck (layer j, dim) sweep ablation (commit f357070, 5 arm 全训完)
5 组 1B/2000步(j4d512/j8d512/j6d256/j6d512/j6d1024 + baseline), 同 slimpajama 同配置:
| arm | pre-j 生成acc | post-j jump | top_acc(ΔvsBase) | h_j dim99 |
|--|--|--|--|--|
| baseline | (渐进) | — | 0.3192 | 1858 |
| j4d512 | 0.0003 | +0.057 | 0.3178(−0.001) | 426 |
| j6d256 | 0.0001 | +0.050 | 0.3141(−0.005) | **224** |
| j6d512 | 0.0001 | +0.080 | 0.3154(−0.004) | 427 |
| j6d1024 | 0.0000 | +0.087 | 0.3160(−0.003) | 774 |
| j8d512 | 0.0000 | +0.094 | 0.3150(−0.004) | 435 |
- **单点结论跨设计空间全 robust**: 分工(pre-j 生成≈0) + QCMem-friendly(dim99 从 1858 塌到 224-774) 对所有 (j,dim) 成立。
- **dim = 主导权衡轴**(fixed j=6): dim99 近线性随 funnel(256→224/512→427/1024→774); LM 代价随 dim 缩小(d256 仅损 1.6%); **d256 甜点**(最可压+最抗压缩 readout+最省)。
- **j 轴**: 每个 j 都"干净"(pre-j≈0); post-j jump 随 j 增大(深 funnel 生成更集中); LM 代价 ~j 无关; QCMem 经济性偏好浅 j(缓存层少). 实用区间 **j4-j6**。
- caveat: 弱模型只看相对趋势, 绝对 acc(~0.31)近随机。

### 3.5 跨数据 robust (wikitext, 完成)
slimpajama 外, wikitext 上复现 baseline+bottleneck(j6d512): **bottleneck 前 7 层生成 acc 全 0.000**(L8 才起), 分工核心(funnel 强制前段不做生成)跨数据 robust。post-bottleneck jump: bottleneck +0.027 vs baseline −0.0004。
- 诚实: wikitext 小数据上 next-tok acc 绝对值低(baseline 0.045/bottleneck 0.055, metric 弱), 但**相对分工结构(前 j 层=0 硬约束)跨数据一致**。bottleneck top-acc 略高于 baseline(小数据 funnel 正则化可能有益)。

### 3.6 甜点组更收敛验证 (j6d256 6000步)
2000→6000 步(loss 4.02→3.51 更收敛): bottleneck 前 7 层生成仍全 0.000, post-bottleneck jump 0.059(>baseline 0.039), LM 代价仍极小(top-acc 0.364 vs baseline 0.372, 差 0.008)。**分工结论在更收敛时保持且未削弱** → 加固"2000步趋势非收敛假象"。
- **6k 可压性更强**: bottleneck 缓存点 dim99=231(<2000步的427), rank-256 压缩 readout 完全无损(−0.0001 vs baseline +0.037)。**训练越充分, semantic-bottleneck 可压性优势越明显**(信息更集中到 rank-256 子空间)→ 加固 §3.3 QCMem-friendly。
- **可压性三点趋势(dim99)**: bottleneck 2000步=427→6000=231→12000=236(**收敛到~230≈funnel宽度256, 非无限降**); baseline 始终~1858(近满秩). rank-256 压缩 readout: bottleneck 12k 仍几乎无损(+0.0001) vs baseline +0.054. 结论: semantic-bottleneck 让缓存点可压性**收敛到 funnel 宽度**且训练充分后稳定, hypothesis 三点一致 SUPPORTED。

## 4. Novelty (6路文献检索)
- **(b) 已知组件新组合 + 一个新 primitive**。上层重算(recompute layers[j:] as readout)=最 novel。
- 缓存 hidden(KV-Direct 2603.19664 需核实/HCache 2410.05004)、检索(RAG 标准)、自蒸馏(KV-Distill)有近亲。
- 无前人用"因前k层有语义→放memory第k层"的 rationale(多数 ablation 挑层)。
- framing: depth-partitioned retrieval readout(layer-partial vs token-partial), j 作 RAG↔closed-book 旋钮。

## 5. 诚实的 limitations
- ≤64k(backbone 外推内) full-context ≥ QCMem(压缩宿命, 不宣称精度 SOTA)。
- 分水岭随 backbone 外推能力此消彼长(未来原生长上下文模型推后不可替代区间; 效率优势始终在)。
  - **实证(niah_single full-ctx vs QCMem)**: Llama-3(原生8k) full-ctx 8k=100→**16k=0崩**(QCMem 100); Qwen(原生40k) full-ctx 撑到64k=100→128k=0. **分水岭≈原生CL的~2×, backbone外推越弱QCMem不可替代区间来得越早、价值越大** → QCMem 在弱外推backbone/场景价值更突出。
- var-track(多跳)弱(64k=21); qa1 32k 需自适应 topk。
- "浅层可压缩"naive 版被证伪(浅层反最不可压, 深层低秩是 attention-sink 假象)→ 存储优势来自"只存一层"(layer-axis)非"该层可压"(feature-axis)。**但 §3.3 已证: semantic-bottleneck pretrain 能显式制造缓存点可压性(bottleneck dim99 427 vs vanilla 1859)** → 这条从软肋转为 pretrain 设计产物。

## 6. 待补 (投稿前)
- [ ] head-to-head 复现并区分 KV-Direct / HCache
- [x] ablation: 上层重算(cross-chunk+query attention)价值 **完成**(commit 2daeb9a block-diag mask+self-test). j=L closed-book(重算0层)=0. 精确block-diag(复用KV,query过全层但chunk间无cross)vs标准(full attn): niah-single 100=100(单chunk读到即可,cross无关); niah-multikey 88/92→44/40(**Δ+44/+52,消歧需cross-chunk**). **结论: 重算+full attention在多fact/干扰任务是load-bearing真设计(非过度工程), 单针可退化省算**. self-test: 单chunk block-diag≡标准(diff4.7e-5),多chunk发散(mask真起作用). **跨benchmark确认(babilong n=100)**: qa2标准36/24 vs blockdiag16/12(Δ+20/+12), qa5 68/65 vs 49/53(Δ+19/+12) — 与RULER multikey(Δ+44/+52)方向一致, cross-chunk attention在多fact任务load-bearing跨benchmark成立(babilong Δ较小因绝对分低+blockdiag仍非零)
- [x] bottleneck 模型 QCMem-friendly 验证 ✓(commit 0a68ad5: dim99 427 vs 1859, rank256压缩几乎无损)
- [ ] 更强 selector(bm25→语义/reader-attn; 但 babilong 上 reader_attn 已判负=词法够)
- [ ] scale bottleneck pretrain 到收敛(可选, 信号已清晰)
