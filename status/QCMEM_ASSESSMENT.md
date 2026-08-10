# QCMem 完整评估报告

> 生成 2026-07-07。所有分数为 RULER `string_match_all` 或 babilong 官方 `compare_answers`，n 已标注。
> 一句话结论：**QCMem 是靠谱的「效率 / 超长上下文可扩展」方法，不是「精度 SOTA」方法。** 定位对了是真贡献。

---

## 0. 方法一句话
Transformer 在深度 `j=12`(Qwen L=36) 切开。WRITE：每个 512-tok chunk 过 `layers[0:j]` 缓存 depth-j hidden `h_j`。READ：bm25 检索 topk chunk，打包 `[sink; 选中h_j; query h_j]`，重算 `layers[j:]`→logits。`j=0`=RAG 全重算，`j=L`=closed-book。训练 = 自蒸馏 LoRA(teacher j=0/student j=12，纯 PG19 零合成数据)。

---

## 1. 命题 A：超 CL 精度 — RULER scaling (Qwen3-8B, n=50/cell)

| task | 方法 | 4k | 8k | 16k | 32k | 64k | **128k** | **256k** |
|---|---|--|--|--|--|--|--|--|
| niah-single | QCMem | 100 | 100 | 100 | 100 | 100 | **100** | **98** |
| | full-ctx | – | 100 | 100 | 100 | 100 | **0** | **OOM** |
| niah-multikey | QCMem | 96 | 94 | 94 | 94 | 82 | **84** | **60** |
| | full-ctx | – | 100 | 100 | 100 | 96 | **0** | **OOM** |
| var-track | QCMem | 100 | 49 | 25 | 22 | 21 | 20 | – |
| | full-ctx | – | 100 | 100 | 100 | 98 | **0** | **OOM** |

**读法（诚实）**：
- **≤64k（Qwen 外推范围内）**：full-context ≥ QCMem（尤其 var-track 98 vs 21）。**装得下时全喂更好，QCMem 不占优**。
- **>64k（外推悬崖后）**：full-context 精度归零(128k=0)或 OOM(256k ~350GB)；**QCMem niah 仍 100/98，multikey 84/60**。这是 full-context 完全做不到的区间 → QCMem **唯一可用**。
- **分水岭 = backbone 外推极限**(Qwen rope1e6 撑到 64k，128k 崩)。

## 2. 命题 B：推理速度 / 显存 (bench_qcmem_vs_fullctx.json, median of 3)

| len | full-ctx prefill | QCMem prefill | **加速比** | full-ctx peak | QCMem peak |
|---|---|---|---|---|---|
| 8k | 0.22s | 0.22s | 0.97× | 20GB | 17GB |
| 16k | 0.52s | 0.33s | 1.59× | 25GB | 17GB |
| 32k | 1.41s | 0.57s | 2.48× | 34GB | 18GB |
| 64k | 4.39s | 1.01s | 4.36× | 52GB | 18GB |
| 128k | 15.0s | 1.92s | **7.83×** | 89GB | **18GB** |

- prefill 加速随长度增长(full O(L²) vs QCMem O(L) write + 固定 read)。
- **显存恒定 ~18GB**(full 涨到 89GB)→ QCMem 单卡能跑 full-ctx OOM 的超长档。
- 交叉点 ~16k；代价：QCMem decode 2.4s(固定，每步重算 layers[j:]) > full-ctx 0.3-0.5s，可优化(resumed-band KV cache)。

## 3. babilong SOTA (每任务最优 topk, 4000步自蒸馏, n=100 官方判分)

| task | 0k | 1k | 2k | 4k | 8k | 16k | 32k | 最优topk |
|---|---|---|---|---|---|---|---|---|
| qa1 | 98 | 79 | 68 | 66 | 63 | 57 | 28 | tk12(32k:24) |
| qa2 | 25 | 44 | 41 | 43 | 37 | 25 | 18 | tk16(32k:24) |
| qa5 | 69 | 77 | 75 | 76(tk8) | 79(tk4) | 67 | 63 | tk4-12 |
| **MemoryLLM(校准)** | qa1 –/50/49/25/30/**20**/12 ; qa5 –/47/42/40/41/**38**/34 | | | | | | | |

- qa1/16k 57 vs MemoryLLM **20 = 2.85×**；qa5/16k 63-67 vs 38 = 1.7×。中程全面超越，端点接近。
- ⚠️ MemoryLLM 数字已校准(铁律2)：qa1 长档实测 16k=20/32k=12(非曾误记的 9/7)。

## 4. 机制证据 (j-sweep + probe)

**oracle j-sweep** (完美检索, qa5/16k, n=100)：j0=69 → j6=50 → j12=39 → j18=**16**。缓存到 j≤12 保留 39-69，j18 崖跌 → 前 ~12 层是「可缓存甜蜜区上限」（单调降，非中层峰值）。

**方向 B（缓存顶层）判负**：缓存顶层丢 query-conditioning，纯前 j 层 resume 才对。

**跨 backbone 可复现**：Llama-3 自蒸馏也 work(qa1 零训 1→蒸馏 33；RULER niah-single 100)，Qwen 更强(难任务 backbone 质量差异)。

**机制分解**：qa5 好=Qwen backbone(零训 57 vs Llama 3)；qa1 好=自蒸馏训练(两 backbone 零训都崩，训练救起)。

## 5. 两个 insight 的实验裁决（probe, 诚实）

- **「浅层够检索」→ 部分支持**：recall@4 在 j=1-6 峰值(qa5/8k j6=0.54)，j8 骤降。浅层≥深层(甚至深层更差)，**但绝对 recall 中等(~0.3@4)**——"浅层≥深层"成立，"浅层高精度"不成立。
- **「浅层可压缩」→ ★证伪**：浅层 j6 反而最不可压(int4 err 0.84, 95%var 需 ~1900 PC)；深层的假性低秩来自 massive-activation/attention-sink outlier(去 magnitude 后所有深度 ~2000 维拉平)。**论文动机不能写"可压缩"，应写"只存一层就够"(省层数 layer-axis，非省维度 feature-axis)**。

## 6. Novelty (6 路文献检索裁决)
**(b) 已知组件新组合 + 一个新 primitive**。最相似 HCache(2410.05004)。上层重算(recompute layers[j:] as readout)=最 novel 无人做；缓存 hidden(KV-Direct 2603.19664 需核实)、检索(RAG 标准)、自蒸馏(KV-Distill)有近亲。最佳 framing = **depth-partitioned retrieval readout**(layer-partial vs 现有 token-partial)，j 作 RAG↔closed-book 旋钮。

## 7. 总评 & 风险

**靠谱，但必须摆正定位。** 卖点优先级：
1. **超 CL 唯一可用**(128k+ full-ctx=0/OOM，QCMem 98-100) — 最硬
2. **恒定显存 + 数倍加速**(128k 7.8×/18GB) — 机制决定
3. **纯自监督泛化 + 跨 backbone** — 守红线回报

**风险/软肋**（reviewer 会打）：
- ≤64k 打不过 full-context（压缩宿命，别宣称精度 SOTA）
- 分水岭随 backbone 外推能力此消彼长（未来原生长上下文模型会推后不可替代区间；但效率优势始终在）
- insight「可压缩」证伪 → 动机改「只存一层」
- var-track（多跳）弱(64k=21)，qa1 32k 掉(检索召回)
- 必须 head-to-head 区分 KV-Direct/HCache + ablation 证上层重算 > 复用 chunk KV

**结论**：作为效率/超长上下文方法，证据链完整、可复现、机制清晰，能发。作为精度方法会崩。
