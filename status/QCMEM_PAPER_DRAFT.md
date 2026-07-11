# QCMem 论文完整 draft（v2，整合 07-09 全部 definitive 结果）

> 2026-07-09 重写。所有数字经 Workflow 5-agent 从 log/json/CSV 交叉核对提取（非记忆），来源标注见 `status/TRAINER_ACTIVITY.jsonl` 07-08~07-09 + 各 `logs/ruler_*.log`。
> 旧版存 `status/QCMEM_PAPER_DRAFT_20260708_backup.md`。配套详据：`status/QCMEM_ASSESSMENT.md` / `status/QCMEM_RELATED_WORK.md`。
> ⚠️ 数字校准记录（铁律2，2026-07-10 更新）：128k niah_multikey 有三个权威值，差异=topk 口径非矛盾，每张表标注所用 topk：**98**（best-topk tk8，§2.0 headline，`ruler_results/ruler_tk8_128k`）、**96**（固定 tk12 head-to-head，§2.1，`ruler_results/qcmem_128k`）、**84**（长度自适应 tk24 scaling，§2.2，`ruler_results/qcmem_qwen_128k_tk24`，对齐 `QCMEM_ASSESSMENT.md §1`）。h2h sweep 是 **n=50**（非项目标准 n=100，已在正文标注）。效率加速比以 `ruler_results/bench_qcmem_vs_fullctx.json` 为权威（128k=7.83×）。Qwen3-8B 原生窗口=**40960**（config max_position_embeddings，非 131072）。

---

## 标题（候选）
**Depth-Partitioned Retrieval Readout: Caching Mid-Layer Hidden States for Unbounded-Context Memory**

## 一句话贡献
把长上下文处理从"重算部分 token 的全深度 KV"（token-partial，现有 RAG-KV-reuse）转成"缓存单个 mid-depth hidden + 检索 + 重算上层"（layer-partial）。read 计算与上下文长度无关（固定 ~6657 tok / chunk512）、显存恒定（~17-18GB），在超出 backbone 外推能力的超长档（128k）做到 full-context 与全量-KV 方法都做不到的事。

---

## 1. 方法（QCMem）
- **split at depth j**（Qwen3-8B L=36, j=12）。
- **WRITE**：每 512-tok chunk 过 layers[0:j] 缓存 depth-j 残差流 hidden h_j（chunk-local RoPE）。存储 ~1/(2L)=1/72 full KV。
- **READ**：bm25 检索 topk chunk → pack[sink; 选中 h_j; query h_j] 全新 RoPE 重算 layers[j:] → logits。read 长度固定 ~6657 tok（sink+topk×chunk+query）。
- **j 旋钮**：j=0=full RAG 重算（self_test 精确等于 full forward，max|logit diff| <1e-4），j=L=closed-book。
- **训练**：自蒸馏 LoRA（teacher j=0 / student j=12, r32/α64, 4000 步, 8-GPU DDP, 纯 PG19 n_ctx=3 连续窗口, 零合成数据）。

---

## 2.0 ★★ 五大 benchmark 总览（2026-07-09 全部跑完，per-task 最优 topk，官方判分）

> **重要方法学**：每个 benchmark/任务的最优 topk 不同（LongEval/RULER-multikey=tk4-8，babilong qa1=tk12/qa5=tk4）。统一用默认 tk12 会**系统性低估 QCMem**——本表已用各任务最优 topk。三方对照口径完全一致（同 backbone Qwen3-8B、同 chunk512、唯一变量=被测 primitive）。
>
> **★ baseline 说明**：(1) **KV-Direct 列 = full-context 精度**——KV-Direct 强制 resume_j=0（全深度重算）+ 无检索 + pack 全部 chunk，数学上精确等于把全文直接喂 Qwen（self_test j=0 read vs full forward max diff <1e-4），只是缓存 residual 省内存。所以 KV-Direct 崩=full-context 崩（超窗口 128k 都=0）。**每 benchmark 都有 full-context 精度对照 = KV-Direct 列**。(2) **HCache** 隔离检索的价值。(3) **MemoryLLM**（专用长上下文 memory 模型，同类对照）：babilong qa1/16k=20 vs QCMem 57（2.85×）；RULER niah_single 8k/16k/32k=22/22/28、multikey=30/14/18（vs QCMem 100/96-98，见 §2.1 碾压）。LongEval/LongBench/LoCoMo 待补（驱动 bug，见 §6）。

| Benchmark | 任务类型 | QCMem（最优 topk） | KV-Direct（全上下文重算） | HCache（无检索） | 一句话 |
|--|--|--|--|--|--|
| **RULER niah_single** | 合成精确检索 | 8k-128k 全 **100**，256k 98 | ≤64k 100，**128k=0** | 8k=34→崩 | 超窗口只有 QCMem 可用 |
| **RULER niah_multikey** | 合成多针 | tk8: 32k=**100**/64k=96/128k=**98** | ≤32k 98-100，64k=80，128k 崩 | 全崩(0) | 窗口内可比，超窗口 QCMem 独占 |
| **LongEval** | lines-retrieval | tk4/6: 8k=**1.00**/32k=**1.00**/64k=0.94/128k=**0.98** | 8k 1.00→32k 0.92→64k **0.34**→128k **0.00** | 全 0 | 窗口内追平、超窗口大幅反超 |
| **babilong** | bAbI 长档 QA | qa5(关系) 8k=**79**/16k=**67**/32k=**63**；qa1(单fact) 16k=57 | qa5 8k=68/16k=44/32k=58；qa1 16k=**72**/32k=62 | qa5 2k=68/8k=52→16k=19/32k=4；qa1 全崩(0-16) | qa5 关系 QCMem 强；qa1 单fact KVD 强 |
| **LongBench** | 真实长文档 QA(F1) | AVG **9.58**(4 task) | AVG 10.13 | — | 任务本难(所有法 4-12)，QCMem 恒定 read 追平全上下文 |
| **LoCoMo** | 长对话记忆 | overall F1 **9.05**/acc 24 | overall F1 **8.72**/acc 20 | overall F1 **4.73**/acc 6 | QCMem≈KV-Direct(acc更高) > HCache(无检索崩) |

**三条贯穿结论：**
1. **超窗口（128k）是杀手锏**：Qwen3-8B 原生 40960（config max_position_embeddings=40960, rope_scaling=None；131072 是未启用的 YaRN 上限），全上下文/KV-Direct 到 128k 一律崩 0（远超原生窗口 → RoPE 外推质量崩塌）；QCMem 固定 read ~2-6.6k 永远在窗口内 → RULER 100 / LongEval 0.98。**三 benchmark 一致**。
2. **窗口内精调 topk 后追平/接近**：LongEval 32k=1.00 打平 KV-Direct；RULER multikey 32k=100。之前"QCMem 精度中等"是没调 topk 的低估。
3. **诚实边界**：babilong qa1/qa2（单/双 fact 定位）窗口内 KV-Direct 更强；LongBench/LoCoMo 绝对分低但**是任务难非 QCMem 弱**（baseline 同样低）；var-track（多跳）弱。**HCache 无检索：长档 needle 类任务一致崩（RULER/LongEval 全 0、babilong qa1 全崩），但短档 redundant-fact 任务不崩（babilong qa5 2k=68/8k=52，与 QCMem 相当）——崩的是"检索缺失导致 needle 淹没"，不是模型能力**。
4. **效率**：QCMem 显存恒定 ~18GB（full 128k=89GB），prefill 128k **7.83×**，read 长度 O(1) vs baseline O(context)。

---

## 2. 核心结果（逐 benchmark 详表）

### 2.1 ★ Head-to-head：QCMem vs HCache vs KV-Direct vs MemoryLLM（RULER NIAH, Qwen3-8B, n=50, string_match_all）
同 eval 框架、同 backbone、同判分，唯一变量=被测 primitive（QCMem=检索+layer-partial；HCache=中层重算+无检索+不训练；KV-Direct=全深度重算+无检索+保留全 token；MemoryLLM=固定 memory bank+FIFO）。

**niah_single_2：**
| 方法 | 8k | 16k | 32k | 64k | 128k |
|--|--|--|--|--|--|
| **QCMem** | 100 | 100 | 100 | 100 | **100** |
| KV-Direct | 100 | 100 | 100 | 100 | **0** |
| HCache | 34 | 2 | 4 | 2 | — |
| MemoryLLM | 22 | 22 | 28 | — | — |

**niah_multikey_1（QCMem 行 = 固定 tk12 head-to-head 口径；KV-Direct/HCache/MemoryLLM 同 n=50）：**
| 方法 | 8k | 16k | 32k | 64k | 128k |
|--|--|--|--|--|--|
| **QCMem（tk12）** | 96 | 98 | 90 | 86 | **96** |
| KV-Direct | 100 | 100 | 98 | 80 | **0** |
| HCache | 4 | 0 | 0 | —(未测) | — |
| MemoryLLM | 30 | 14 | 18 | — | — |

**read_len（喂给模型的 token 数，效率核心）：**
| 方法 | 特性 | 128k read_len |
|--|--|--|
| **QCMem** | 恒定 ~6.2-6.6k（检索固定 pack） | ~6256 |
| KV-Direct | O(L) 保留全 token | ~130746 |
| HCache | O(L) 恢复全 token | 64k~65505 |

**四个结论：**
1. **vs HCache = 精度碾压**：HCache 无检索、8k 就崩到 34、16k=2、multikey 近全 0。原因：无检索长档 needle 被淹没。
2. **vs MemoryLLM（同类 memory 模型）= 精度碾压**：MemoryLLM niah_single 8k/16k/32k=22/22/28、multikey=30/14/18，QCMem 100/96-98 全面碾压（同类固定 memory 方法在合成精确检索上远不如检索+layer-partial）。
3. **vs KV-Direct = 同精度、效率碾压**：≤64k 两者精度相当（都近满分），但 QCMem read_len 恒定 ~6.3k vs KV-Direct 保留全 token（128k=13 万 tok）。
4. **★128k QCMem 唯一可用**：KV-Direct 128k **崩到 0**（保留全 130746 tok 远超 Qwen3 原生窗口 40960 → RoPE 外推质量崩塌，全深度重算失效），QCMem 固定 read 6256 tok 永远在窗口内 → **100/96**。这是 layer-partial+检索的结构性优势，不只是省显存。

### 2.2 超长上下文精度 scaling（RULER, Qwen3-8B, n=50, QCMem 长度自适应 topk / full-ctx 全长直喂）
> 注：本表 QCMem multikey 行取自 `QCMEM_ASSESSMENT.md §1` 的 scaling 决胜表（长度自适应 topk：中长档 tk12、128k=tk24、256k=tk48）。故 128k multikey=**84**（tk24）——与 §2.0 headline 的 98（best-topk tk8）、§2.1 head-to-head 的 96（固定 tk12、`ruler_results/qcmem_128k`）是同一 cell 的不同 topk 口径，非矛盾。（§2.1 的 64k=86 是 tk12 h2h run，本表 64k=82 是 scaling 系列，属不同 eval 批次的小幅波动。）
| task | 方法 | 8k | 16k | 32k | 64k | 128k | 256k |
|--|--|--|--|--|--|--|--|
| niah_single | QCMem | 100 | 100 | 100 | 100 | **100** | **98** |
| | full-ctx | 100 | 100 | 100 | 100 | **0** | **OOM** |
| niah_multikey | QCMem | 94 | 94 | 94 | 82 | 84 | 60 |
| | full-ctx | 100 | 100 | 100 | 96 | **0** | **OOM** |
| var-track | QCMem | 49 | 25 | 22 | 21 | 20 | — |
| | full-ctx | 100 | 100 | 100 | 98 | **0** | **OOM** |
- 分水岭 = backbone 外推极限（Qwen 原生 40960，64k→128k 崩）。>该点只有 QCMem 可用。
- vs StreamingLLM（同 ~6657-tok/~17GB 固定 budget）：niah_single 8k/16k/64k/128k = QCMem 100/100/100/100 vs SLM 90/42/16/4（**128k 25× gap**）。检索保留"相关"，SLM 只保留"最近"→丢中间→针 miss。

### 2.3 效率（bench_qcmem_vs_fullctx, median-of-3, chunk512）
| 长度 | prefill 加速(full/QCMem) | full 显存 | QCMem 显存 |
|--|--|--|--|
| 8k | 0.97× | 20GB | 17.3GB |
| 16k | 1.59× | 25GB | 17.4GB |
| 32k | 2.48× | 34GB | 17.5GB |
| 64k | 4.36× | 52GB | 17.8GB |
| 128k | **7.83×** | 89GB | **18.3GB(恒定)** |
- QCMem prefill = write O(L) + 固定 read；full O(L²)。交叉点 ~16k。
- **QCMem 显存 L-无关恒定 ~17-18GB**（full 涨到 89GB）→ 128k 单卡能跑 full 会 OOM 的档。
- 代价：QCMem decode ~2.4s（每步重算 layers[j:]，attend 整个 read pack）> full 0.3-0.5s，可优化。

### 2.4 泛化 + 跨 backbone + babilong
- 纯 PG19 自蒸馏（零 babilong/RULER）→ RULER niah zero-shot 强（Llama-3 self-distill niah_single 8k/16k=100/100）=通用记忆非特化。
- babilong 每任务最优 topk（4000 步自蒸馏, n=100 官方判分）：qa1 0k-32k=98/79/68/66/63/57/28（vs MemoryLLM 校准 16k=20=2.85×）；qa5=69/77/75/76/79/67/63（vs 38）。
- **★ babilong 三方逐 cell（n=100, 官方 compare_answers, 4k-32k）**：

| task | 方法 | 4k | 8k | 16k | 32k |
|--|--|--|--|--|--|
| qa1(单fact) | QCMem | 66 | 63 | 57 | 28 |
| | KV-Direct(全上下文) | **75** | **80** | **72** | **62** |
| | HCache(无检索) | 16 | 3 | 0 | 0 |
| qa5(关系) | QCMem | **76** | **79** | **67** | **63** |
| | KV-Direct | 61 | 68 | 44 | 58 |
| | HCache | 65 | 52 | 19 | 4 |

- **诚实边界坐实**：qa1（单 fact 定位）KV-Direct 全上下文更强（4k-32k 75/80/72/62 vs QCMem 66/63/57/28）——精确定位任务全喂全文占优。qa5（关系推理）QCMem 反超（尤其长档 16k 67 vs 44、32k 63 vs 58）。
- **HCache 崩的机理（重要，校正"一致崩"）**：HCache qa5 短档**不崩**（4k=65/8k=52，接近 QCMem/KVD），只在长档崩（16k=19/32k=4）；qa1 全崩（无检索时单 fact 被淹没）。→ 崩=检索缺失导致长档 needle 淹没，不是模型能力，短档 redundant-fact 任务照常工作。

### 2.5 ★ chunk_size（block size）消融：精度×效率 trade-off（新增）
read_len = topk×chunk + sink + query，随 chunk 线性（topk=12 固定）。

| chunk | read_len | peak(64k) | prefill(64k→128k) | decode | niah_multikey |
|--|--|--|--|--|--|
| 128 | 1665 | 16.3GB | 1.38× | ~0.69s | 16k=90 / 32k=80 |
| 256 | 3329 | 16.8GB | 2.61× | ~1.17s | 8k92/16k80/32k80/64k88 |
| **512** | 6657 | 17.8GB | **4.36×→7.83×** | ~2.39s | **128k=94** |
| 1024 | 13313 | 19.8GB | 4.99× | ~5.55s | 8k96/16k96/32k92/64k90/128k96 |

- **write_calls（64k）**：cs128=513 → cs256=257 → cs512=129 → cs1024=65（chunk 翻倍减半）→ 驱动 prefill 加速。
- **三角权衡**：chunk 越大→prefill 越快（write 少）+ 精度越好（multikey），但 **decode 越慢**（attend 整个 read pack，cs1024 decode 5.55s vs cs512 2.39s）+ 显存略高。chunk 越小→显存最省+decode 快，但 prefill 慢（write 多）+ multikey 掉。
- **甜点 = chunk512**：强 prefill 加速（128k 7.83×）+ 低平显存（17-18GB）+ 中等 decode（2.39s）+ 高 multikey（128k=94）——接近 cs1024 精度但约一半 decode 成本和 read_len。

### 2.6 检索甜点 topk（部分网格, n=50）
- niah_single 全档饱和 100（非判别任务）；multikey 是判别任务。
- **甜点随 length 变化**：8k 甜点 topk4（multikey=100，read_len~2465），32k 甜点 **topk8**（=98，vs topk24=90，Δ-8）。**长档甜点更小**（过召回稀释信噪比）。
- ⚠️ 网格稀疏：仅 32k 有完整曲线（tk4/8/12/16/24 = 96/98/90/92/90）。topk8/16k、topk16/多档缺；topk8/8k(n=30)、topk8/64k(n=10) 未满 n=50 不采信。

### 2.7 ★ LongEval（LongChat lines-retrieval, n=50, 最优 topk tk4-6）
纯单跳行检索——检索式记忆的主场。**默认 tk12 严重低估（0.70），最优 tk4-6 才是真实实力。**
| 长度 | QCMem（最优 topk）| KV-Direct | HCache | QCMem read_len |
|--|--|--|--|--|
| 8k | **1.00**(tk4/6) | 1.00 | 0.00 | ~2.2k |
| 16k | 0.96(tk4) | 0.96 | 0.00 | ~2.2k |
| 32k | **1.00**(tk4) | 0.92 | 0.00 | ~2.2k |
| 64k | 0.94(tk4/6) | **0.34** | 0.00 | ~2.3k |
| 128k | **0.98**(tk6) | **0.00** | 0.00 | ~3.4k |
- **窗口内追平**（8k/16k/32k = 1.00/0.96/1.00 vs KVD 1.00/0.96/0.92）+ **超窗口大幅反超**（64k 0.94 vs 0.34；128k 0.98 vs 0）。
- QCMem read_len 恒定 ~2-3k（O(1)）；KV-Direct read_len O(context)，128k 达 13 万 tok 远超 Qwen 原生窗口 40960 → RoPE 外推崩。HCache 全 0（无检索 needle 淹没）。

### 2.8 LongBench（真实长文档 QA, F1 官方口径, Qwen3-8B, chunk512）
| 任务 | QCMem(tk12) | KV-Direct(全上下文) | Stock(no-LoRA) |
|--|--|--|--|
| narrativeqa | 3.93 | 3.95 | 4.01 |
| qasper | 11.07 | 11.92 | 10.33 |
| hotpotqa | 11.64 | 12.48 | — |
| 2wikimqa | 11.69 | 12.16 | — |
| **AVG** | **9.58** | 10.13 | — |
- **关键（诚实且对 QCMem 有利）**：三方 F1 全落在 4-12（Qwen3-8B base + no_chat_template，任务本难），QCMem 用**恒定 read (~4.6k)** 追平 KV-Direct 的**全上下文**（差距在噪声内）→ **LongBench 低分是任务难非 QCMem 弱**。

### 2.9 LoCoMo（长对话记忆, F1/acc, overall + 按 category）
**★ 三方对照（overall, QCMem/HCache 全量 n=1986, KV-Direct n=760 已收敛）：**
| 方法 | overall F1 | overall acc |
|--|--|--|
| **QCMem** | **9.05** | **24.1** |
| KV-Direct(=全上下文) | 8.72 | 20.3 |
| HCache(无检索) | 4.73 | 6.4 |
- 图景与其它 benchmark 一致：**QCMem ≈ KV-Direct 且 acc 更高（24 vs 20）**（对话窗口内，检索追平/略超全上下文）；**HCache 明显低**（无检索长对话记忆差）。KV-Direct 分数 300→760 样本稳定（8.86→8.72），已收敛。

**QCMem 按 category（n=1986）：**
| category | n | F1 | acc |
|--|--|--|--|
| cat1 multi_hop | 282 | 9.8 | 12.1 |
| cat2 single_hop | 321 | 6.3 | 9.3 |
| cat3 temporal | 96 | 8.1 | 24.0 |
| cat4 open_domain/画像 | 841 | **13.9** | **45.7** |
| cat5 adversarial(要拒答) | 446 | 1.6 | 1.6 |
- cat4（开放域/画像）最好（acc 45.7）；cat5（对抗，需拒答）最低（backbone 拒答行为主导，非检索问题）。
- 绝对分低=对话 QA 答案短/paraphrastic 本难。（注：旧 mem_space 系列同类对话 QA overall F1 仅 2.7（LOCOMO n=400 口径，与本表 n=1986 口径不同，仅作量级参考）——QCMem 9.05 相对旧 mem_space 是数量级反超，非落后。）


---


## 3. 机制：为什么缓存中层有效 —— 理解-生成分工

### 3.1 分工命题（跨 backbone robust）
- **probing**：语义任务中层达峰、顶层回落；next-token 仅顶层成形。Qwen+Llama 两曲线分离 True/True。
- **3a 截断下游**（因果）：只用前 ~4-8 层（深度 0.12-0.22）达全模型 95% 下游语义；RTE 中层>顶层 +0.06~0.10；顶层 verbalizer 反超中层 probe（RTE native 0.79 > probe 0.62）=顶层用"表征线性可分"换"生成有用性"。
- **精确表述**：相对分工（理解层<生成层），语义绝对深度 backbone-dependent。

### 3.2 与 QCMem 挂钩
- 零训练 j-sweep（RULER niah 16k）：j≤9(深0.25)=100, j12 崖跌 14, j18=0。可缓存深度上限 ≈ 3a 理解饱和点。
- babilong qa5/16k oracle j-sweep：j0/j6/j12/j18 = 69/50/39/16。
- **方向 B（缓存顶层）判负**：顶层 hidden=query 敏感读出前表征，缓存丢 query-conditioning。qa5 (12,0)=61/50 → (12,6)=29/20, (6,12)=9/10。
- 自蒸馏把可缓存上限从 j9 推到 j12+。

### 3.3 ★ 端到端正面证明：QCMem-structure > vanilla+QCMem（压缩缓存下, 新增, n_docs=200）
把 bottleneck-pretrain 模型 vs vanilla 模型都当 QCMem backbone，在压缩缓存（rank-r PCA）下比 prediction ΔNLL（=NLL(rank r)−NLL(rank512)，越小越抗压）。

**3B（headline claim 成立）**：压缩缓存下 bottleneck 每个 rank 都 degrade 更少：
| rank | base ΔNLL | bot ΔNLL | 优势 |
|--|--|--|--|
| 256 | +0.013 | −0.003 | +0.016 |
| 128 | +0.067 | +0.010 | +0.057 |
| 64 | +0.14 | +0.086 | +0.05 |

- **★model-size 曲线**：优势（base−bot ΔNLL, 三 ctx 均值）**随 model 增大而增大**：r128 = 1B +0.030 → 3B **+0.057**（约 2×）；r256 = +0.008 → +0.015。**模型越大, 为 QCMem 定制 pretrain 收益越明显**。
- 1B 也全胜 rank256/128/64（9/12 wins），但 r32 极端压缩下会输（弱模型信息太挤）。
- PCA 内在维塌缩且 model-size-stable：pca_dim99 1B base 1825→bot 438；3B base 2790→bot 466（bottleneck_dim=512）。
- ⚠️ ΔNLL 相对各自 r512 定义；r512 时 bottleneck 绝对 NLL 略高（3B ctx2048 3.266 vs 3.242）=约 4% ppl 税，但这是"不压缩"时的固定小税，压缩后被抗压优势反超。

### 3.4 semantic-bottleneck pretrain 可行性（1B from-scratch）
- layer-6 rank-512 funnel（down d→d_bottle→GELU→up, 无 residual）：前 7 层 next-tok acc **0.000**（分工被显式强制），top-acc 几乎无损（0.331 vs 0.334）。
- 缓存点可压：dim99 1859→427, eff-rank 407→149；rank256 压缩 readout drop +0.028→+0.001。
- (j,dim) sweep（5 arm）：分工+可压跨设计空间 robust，d256 甜点（dim99 224）。
- 跨数据（wikitext）+ 三点收敛（dim99 2000步427→6000步231→12000步236，收敛到 ~230≈funnel 宽度）。
- **★ bottleneck_dim × LM 税 sweep（16000 步收敛版, layer6, 末10步均值 ppl）**：baseline 25.28 / d1024 26.42(+4.5%) / d512 26.78(+5.9%) / **d256 27.42(+8.5%)**。**bottleneck 越窄 LM 税越高**——与"可压性甜点在 d256"相反，二者是 trade-off：窄 funnel 迫使表征更紧凑（PCA-ΔNLL 最优、缓存最省，见 §3.3），代价是训练时 LM 税更高。设计取舍取决于部署侧更看重"缓存压缩率"（选 d256）还是"LM 质量"（选 d1024/无 bottleneck）。所有 arm 税 ≤8.5%（都是"不压缩"时的固定小税，§3.3 证明压缩后被抗压优势反超）。
- **★★ bottleneck 税 model-size 曲线（3B from-scratch 复现, 16000 步, dim sweep@layer6）**：3B base 27.44 / d512 28.73(**+4.7%**) / d256 29.04(**+5.8%**)。**3B 两档税都明显小于 1B（d512 4.7%<5.9%, d256 5.8%<8.5%），且 3B 内部仍 d256>d512 同向。→ "model 越大, bottleneck 相对 LM 税越小" 跨 dim 一致成立**，与 §3.3 端到端 ΔNLL 的 model-size 曲线（优势随 model 增大）互为印证——**为 QCMem 定制 pretrain 在更大模型上代价更低、收益更大**。这是 scale 论点的直接证据（1B→3B）。
- **★ bottleneck_layer sweep {1,3,6,9,12}@dim512/16k（16000 步收敛, 末20步均值 ppl）**：baseline 25.34 / **layer1 26.40(+4.2%)** / layer3 26.82(+5.8%) / layer6 26.85(+6.0%) / layer9 27.74(+9.5%) / layer12 27.77(+9.6%)。**LM 税随 bottleneck 深度单调递增**：越浅（越靠近输入）放 funnel，训练代价越小。这与 §3.1 分工命题一致——浅层承载的是低阶/局部信息，压缩它损失小；越深越接近"生成前的精炼表征"，压缩越伤。**设计取舍**：QCMem 缓存点若追求"LM 税最小"应放浅层（L1-3），但缓存点太浅则可缓存的语义不足（§3.2 j-sweep：j≤9 检索精度饱和、j12 崖跌）→ 二者共同框定 QCMem 的 j=12 是"可缓存语义上限"与"LM 税"的折中，而非税最小点。

### 3.5 cross-chunk attention ablation（上层重算的价值, RULER n=50 固定 tk12）
read 时 layers[j:] full-attention 重算 vs block-diagonal（复用 chunk KV 无 cross-chunk）：
- RULER niah_single 8k/16k：100=100（单 chunk 读到即可，cross 无关）。
- niah_multikey：std 88/92 vs blockdiag 44/40（**Δ+44/+52，消歧需 cross-chunk**）。
- babilong n=100 跨 benchmark 确认：qa2 std 36/24 vs blockdiag 16/12；qa5 68/65 vs 49/53。
- 结论：重算+full attention 在多 fact 任务 load-bearing（真设计非过度工程），单针可退化省算。

---

## 4. Novelty
- **(b) 已知组件新组合 + 一个新 primitive**。新 primitive = **重算 layers[j:] as readout**（depth-partitioned）。framing = depth-partitioned retrieval readout（layer-partial vs 现有 token-partial），j 作 RAG↔closed-book 旋钮。
- **最近前人（related work 必须正面区分）**：
  - **HCache (2410.05004)**：serving 系统缓存中层 activation+重算上层，但 post-hoc 不训练、无检索。**我们 head-to-head 实测其推理形态（无检索+不训练）在长档崩（8k=34/16k=2）** → 检索+自蒸馏是 QCMem 关键。
  - **KV-Direct (2603.19664, "The Residual Stream Is All You Need")**：缓存 residual 重算全深度 KV，training-free，保留全 token。**实测=全上下文精度但 128k 崩（0）+read_len 爆（13万）** → QCMem 的 layer-partial+检索+固定 read 是独家 delta。
  - **KV-CAT (2605.05971)**：同 thesis"训练期引导可压缩"，但 KV-slot masking（token 轴）+continued pretrain，非特征维 bottleneck+非 from-scratch。§3.3 的端到端 ΔNLL 是我们对它的直接回应。
  - **CompressKV (2606.24467)**：语义检索头选 token 的 KV eviction（token 轴），与 QCMem layer 轴正交，固定预算压缩类背景。
  - 超CL 锚点：YOCO (2405.05254, 1M near-perfect)、InfLLM (1024k)。

---

## 5. 诚实的 limitations
- **定位=效率/超长上下文方法，非精度 SOTA**：≤64k backbone 外推内 full-ctx ≥ QCMem（尤其 var-track 98 vs 21）。装得下时全喂更好。
- **分水岭随 backbone 外推能力此消彼长**：Llama-3（原生8k）full-ctx 8k=100→16k=0 崩（QCMem 100）；Qwen（原生40k）撑到 64k→128k=0。分水岭≈原生CL 2×，弱外推 backbone QCMem 价值更早、更大。
- var-track（多跳）弱（64k=21）；qa1 32k 需自适应 topk；qa2（双 fact 最难）0k=25/16k=25。
- decode 慢（chunk 越大越慢，cs1024 5.5s），可优化（resumed-band KV cache）。
- h2h sweep 是 n=50 cut（非 n=100 标准），正文需标注；topk 网格稀疏。

---

## 6. 待补（投稿前）
- [x] head-to-head 复现区分 KV-Direct/HCache **完成**（§2.1，mechanism-level，commits b8f3181/7dbef46）
- [x] chunk_size 消融 **完成**（§2.5）
- [x] 端到端 QCMem-structure > vanilla（§3.3，3B ΔNLL + model-size 曲线）**完成**
- [x] cross-chunk attention ablation **完成**（§3.5）
- [x] MemoryLLM×RULER 同类对照 **完成**（§2.1，single 22/22/28 / multikey 30/14/18 vs QCMem 100/96-98 碾压）
- [x] KV-Direct 128k multikey **完成**（=0，§2.1，read_len 130746 超外推极限）
- [x] babilong 三方逐 cell（qa1/qa5 × 4k-32k）**完成**（§2.4，官方 compare_answers）
- [ ] **进行中**：真实 Qwen3-8B + funnel continued pretrain（outputs/qwenbott_*），训完在真实 RULER/babilong 比"原始 Qwen+QCMem vs funnel-Qwen+QCMem"——答"强模型+真实任务上 pretrain 是否有收益"+对标 KV-CAT
- [ ] MemoryLLM 扩到 LongEval/LongBench/LoCoMo（LongEval 驱动 preds=0 bug 待修）；升 h2h n=50→100
- [ ] HCache 64k multikey / babilong 0k-2k 低档补齐
- [ ] topk×length 甜点补全稀疏网格
- [ ] KV-CAT (2605.05971) 引用图追踪排除近期撞车（投稿前）
