# QCMem Related Work + 复现清单 (Reproduction Checklist)

> 2026-07-08. 系统调研 QCMem 近邻，产出投稿必须 head-to-head 复现/对照的方法清单。
> 方法：curl arxiv abs 页逐篇核对标题/摘要（企业网络拦 WebFetch，用 hy-proxy + curl）。每个 arxiv id 均已核对，未凭记忆编号。
> 配套：`status/QCMEM_PAPER_DRAFT.md`（方法/结果）、`ops/research_notes/novelty_division_of_labor_20260708.md`（命题A/B novelty）。

QCMem 一句话回顾：深度 j 处切分。WRITE 每 512-tok chunk 过 layers[0:j] 缓存 depth-j hidden h_j（chunk-local RoPE）；READ bm25 检索 topk chunk → pack[sink; 选中 h_j; query h_j] 全新 RoPE 重算 layers[j:] → logits。特点：缓存单个 mid-depth hidden（非全 KV）、read 计算与上下文长度无关（固定 ~6657 tok）、显存恒定 ~18GB、检索式、j 作 RAG↔closed-book 旋钮。定位=**效率/超长上下文方法**（非精度 SOTA）。

---

## A. 缓存中间层 hidden 并重算上层（QCMem 推理形态最直接对手）

| arxiv id | 标题 | 机制一句 | 与 QCMem 精确 delta |
|---|---|---|---|
| **2410.05004** | Fast State Restoration in LLM Serving with HCache (Gao, Chen, Shu; 2024-10-07) | LLM serving 系统：evict KV 后从**中间层 activation** 恢复（重算上层），比 token 重算省算、比 offload 省 IO；TTFT↓1.93× vs offload / 5.73× vs token 重算，存储省 1.92–2.4×。有 bubble-free 调度 + chunk-based storage（解决 layer-before-token 存 vs token-before-layer 恢复的 layout mismatch）。 | ★**"缓存中间层 hidden、上层重算" 的系统实现，head-to-head 第一候选。** delta：(1) HCache 是 **post-hoc serving 系统、不训练、不改模型**；QCMem 自蒸馏 LoRA 学会从更深缓存 readout（j9→j12+）。(2) HCache **无检索**——它恢复被 evict 的 KV，read 成本仍与被恢复 token 数成正比；QCMem **bm25 检索 topk chunk + 固定 read 长度（~6657 tok），与上下文长度无关**。(3) HCache 目标=降 TTFT/存储（工程指标），无"超长上下文外推"主张；QCMem 卖点=128k+ full-ctx 崩而它=100。(4) HCache 存所有层？——存中间某深度 activation 后恢复上层 KV，但**不做"深度作为 RAG↔closed-book 旋钮"的语义分工 framing**。**结论：HCache 是 QCMem 推理形态最像的前人，但一个是 serving 缓存加速、一个是训练+检索的恒定显存超长上下文方法。必须正面区分。** |
| **2603.19664** | The Residual Stream Is All You Need: On the Redundancy of the KV Cache in Transformer Inference (Qasim, Zhang et al.; 2026-03-20) | **证明 KV 完全冗余**：每层 K/V 都是 residual stream 的确定性投影，从单个 residual 向量重算 K/V **零重构误差（bit-identical）**。跨 6 模型 4 架构（135M-4B）验证 cross-task residual patching D_KL=0。提出 **KV-Direct**：bounded-memory 推理，**checkpoint residual 向量（Gemma3-4B 5KB/token）替代全 KV（136KB/token）**，按需重算 K/V。20 轮对话峰值内存 42MB vs 标准 cache 103MB；vs 5 个 eviction baseline（H2O/StreamingLLM/SnapKV/TOVA/window-only）在**所有 KV 预算下 100% token match，baseline 全退化到 5-28%**；单次重算比读缓存快 5×。开源 github.com/Kaleemullahqasim/KV-Direct。 | ★★**draft 里"KV-Direct 需核实"—— id 正确，且是最危险的"缓存 hidden + 重算"近邻（2026-03 新论文）。** delta：(1) KV-Direct 缓存 **residual 向量并重算全部层的 K/V（full-depth 重算）**；QCMem 缓存 **单个 mid-layer hidden h_j 只重算 layers[j:]（depth-partial）**——QCMem 的 layer-partial 重算是 KV-Direct 没有的核心 primitive。(2) KV-Direct **保留所有 token、不检索**，内存 O(token) 虽 bounded；QCMem **检索 topk + read 长度恒定、显存恒定**，与上下文无关。(3) KV-Direct 训练-free、目标是"证明 KV 冗余 + 省内存对话"；QCMem 自蒸馏 + 目标是超长上下文外推。(4) 两者共享"cache hidden 而非 KV、按需重算"的 primitive，**这是 QCMem 必须在 related work 明确切割的最近工作**——QCMem 的增量必须落在 **depth-partition（只重算上层）+ retrieval + 固定 read + 语义分工 rationale**，不能再把"缓存 hidden 重算"当独家。 |

**A 类小结**：HCache（serving，post-hoc，无检索）+ KV-Direct（residual 全深度重算，无检索，training-free，2026-03）是 QCMem 推理形态最像的两篇。QCMem 相对二者的**独家 primitive = layer-partial 重算（只重算 layers[j:]）+ 检索式固定 read + 深度作语义旋钮 + 自蒸馏把可缓存深度推更深**。二者都 **必须 head-to-head 或至少 mechanism-level 对照**。

---

## B. 检索式长上下文 / RAG-KV-reuse（QCMem 的 READ 用检索）

| arxiv id | 标题 | 机制一句 | 与 QCMem 精确 delta |
|---|---|---|---|
| **2305.16300** | Landmark Attention: Random-Access Infinite Context Length for Transformers (EPFL; 2023-05) | 每 block 一个 **landmark token** 代表该块，**训练 attention 用 landmark 选相关块**（检索融进 attention 而非外挂机制），保留 random-access。fine-tune LLaMA-7B 扩到 32k+。开源 epfml/landmark-attention。 | 检索式长上下文的**训练版经典对手**（本项目已复现对照 faithful landmark）。delta：(1) Landmark 检索的是**块内全 KV（full-depth K/V）** 经 landmark 门控，仍存/查所有层 KV；QCMem 只缓存**单个 mid-depth hidden**。(2) Landmark 检索单元=attention 内 landmark token（可微、每步选块）；QCMem 用 **bm25 外部词法检索**（不可微、固定 topk）。(3) Landmark 无"深度分区/上层重算"。**对照价值**：同"检索相关块避免全 attention"赛道，效率表可比（Landmark 存全层 KV vs QCMem 存一层 hidden）。 |
| **2307.03170** | Focused Transformer (FoT / LongLLaMA); Contrastive Training for Context Scaling (2023-07) | 给 attention 层加**外部 (k,v) memory**，用**对比学习**训练缓解 distraction issue（相关 key 占比随文档数下降）；fine-tune OpenLLaMA 3B/7B，passkey 256k。 | 检索式外部 KV memory 的训练对手。delta：FoT memory 存 **(k,v) pairs（某层 KV）**，靠对比训练让 key 空间可区分；QCMem 存 **residual hidden h_j** 并**重算上层**（不是查 KV 做 attention）。FoT 无深度分区、无上层重算、无固定 read 长度。谱系对照。 |
| **2405.16444** | CacheBlend: Fast LLM Serving for RAG with Cached Knowledge Fusion (2024-05) | RAG 场景：预算多个 chunk 的 KV cache，复用时**选择性重算一小部分 token 的 KV** 来补 cross-attention，达到 full prefill 质量；TTFT↓2.2-3.3×。开源 LMCache。 | ★**"检索 chunk 的缓存 + 选择性重算" 的最近系统亲戚。** delta：(1) CacheBlend 缓存/复用的是 **full-depth KV**，只重算少量 token 的 KV 补 cross-attention；QCMem 缓存 **单层 hidden**，重算的是**上层全部（layers[j:]）** 而非"少数 token 的 KV"。(2) CacheBlend 是 post-hoc serving、不训练、目标=RAG prefill 提速且不掉质量；QCMem 训练 + 目标超长外推 + 恒定显存。(3) 两者都处理"多 chunk 拼接的 cross-attention"问题——QCMem 的 full-attention 重算（vs block-diag）ablation（draft §6 commit 2daeb9a）正是这个问题的 QCMem 版解法。**CacheBlend 的 cross-chunk KV fusion vs QCMem 的上层 full-attention 重算是同一痛点的两种解**，值得在 related work 对照。 |
| **2404.12457** | RAGCache: Efficient Knowledge Caching for RAG (2024-04) | 把检索知识的**中间状态（KV）** 组织成 knowledge tree 缓存在 GPU/host 层级，感知 LLM 推理+RAG 检索模式的替换策略，检索与推理 overlap；TTFT↓4×。 | 纯 serving 缓存系统：缓存**检索文档的 full KV**（intermediate states）避免重算。delta：QCMem 缓存**单层 hidden 而非全 KV**，且**每次都重算上层**（RAGCache 恰恰是为了**避免重算**）——设计目标相反：RAGCache 省重算存全 KV，QCMem 省存储只存一层 hidden 靠重算换。对照"存什么/重算什么"的权衡轴。 |

**B 类小结**：检索式赛道分两支——(1) **训练/可微检索**（Landmark、FoT）：检索融进 attention，存全层 KV；(2) **serving KV-reuse**（CacheBlend、RAGCache）：缓存检索 chunk 的全 KV 避免重算。QCMem 与全部的共同 delta = **只缓存单层 hidden（非全 KV）+ 用外部 bm25（非可微/非 attention 内）+ 每次重算上层换存储**。Landmark 是最该对照的训练版（本项目已有 faithful 复现）；CacheBlend 是 cross-chunk 问题的最近系统亲戚。

---

## C. 固定预算 / 压缩长上下文（效率对手，QCMem 卖点是恒定显存）

| arxiv id | 标题 | 机制一句 | 与 QCMem 精确 delta / 对照定位 |
|---|---|---|---|
| **2309.17453** | Efficient Streaming Language Models with Attention Sinks (StreamingLLM; 2023-09) | 只 cache **最近窗口 KV + 初始几个 attention-sink token**，使有限窗口模型外推到 4M+ token，无 fine-tune；vs sliding-window 重算快 22×。 | ★**同"固定 KV 预算"赛道的头号对照（draft §2.1 已同 budget 对比）。** delta：StreamingLLM 保留**最近 token 的全层 KV**（丢中间→niah 针 miss，128k=4 vs QCMem=100）；QCMem 用**检索保留相关 chunk 的单层 hidden**。**这是 QCMem 效率表的核心对照——同 6657-tok/~17GB 固定 budget 下，检索 vs 最近窗口的精度差（25×）**。已复现，必须保留。 |
| **2306.14048** | H2O: Heavy-Hitter Oracle (2023-06) | 观察少数 token（Heavy Hitters）贡献大部分 attention value；动态保留 recent + H2 token 的 KV eviction 策略；20% heavy hitter 吞吐↑29×。 | 固定/稀疏 KV 预算的 **token-space eviction** 经典。delta：H2O 按 attention score 保留**部分 token 的全层 KV**；QCMem 按 **bm25 检索保留部分 chunk 的单层 hidden**。同"选重要内容压 KV"赛道但选择粒度（token vs chunk）、存什么（全 KV vs 单层 hidden）、选择依据（attention vs 词法检索）全不同。效率表可作 token-eviction 代表对照。 |
| **2404.14469** | SnapKV: LLM Knows What You are Looking for Before Generation (2024-04) | 用 prompt 末尾 observation window 预测每个 head 关注的 KV 位置，**generation 前压缩 KV**；16k 输入解码快 3.6×、内存省 8.2×，NIAH 几乎无损。 | fine-tuning-free 的 **prompt-aware KV 压缩**。delta：SnapKV 仍存**被选 token 的全层 KV**且压缩发生在 prefill 后；QCMem 存**单层 hidden** 且检索发生在 read 时。SnapKV 无超长外推主张（针对已能读的长输入省内存）。同赛道效率对照。 |
| **2406.02069** | PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling (2024-06) | 观察 LLM **信息金字塔漏斗**（低层 attention 分散→高层聚焦 sink/关键 token）；据此**低层多 KV、高层少 KV** 的分层预算；保留 12% KV 匹配 full。 | ★**双重相关**：(1) 效率对手（分层 KV 预算）；(2) **机制观察与 QCMem §3 层分工呼应**——PyramidKV 说"高层信息聚焦"，QCMem 说"顶层做生成、中层语义饱和"，都在讲层间信息流变化。delta：PyramidKV 仍是 **post-hoc 存各层部分 KV**（只是每层预算不同）；QCMem **只存一层 hidden、重算上层**。PyramidKV 的金字塔观察可在 QCMem related work 作"层间信息流"的印证/区分。 |
| **2401.03462** | Long Context Compression with Activation Beacon (2024-01) | plug-in 模块，**直接压缩每层 activation（K/V）** 而非 soft prompt；细粒度渐进压缩；**compression-based auto-regression 训练**，训练时随机采样压缩比支持多档。128k 远超 20k 训练长度仍匹配 uncompressed，推理快 2×、KV 省 8×。 | ★**训练式固定预算压缩的最强对手之一（要训练、支持超长）。** delta：(1) Beacon 压**每层 activation（全深度都压）** 成 beacon token；QCMem 只存**单层 hidden、重算上层**（不压其他层）。(2) Beacon 用 beacon token 做 recurrent 压缩（RMT 家族改良）；QCMem 用**检索 + 上层重算**（非 recurrent 压缩）。(3) 都训练、都超长外推——**Activation Beacon 是 QCMem "要训练的固定预算超长上下文" 最直接的精度+效率对手，应在主表对照**。 |
| **2405.10637** | Layer-Condensed KV Cache for Efficient Inference (LCKV; 2024-05, 附带发现) | **只计算并缓存少数层的 KV**，其余层共享，吞吐↑26×，与其他省内存法正交。 | 与 QCMem "只存一层" 精神相关的**架构级 KV 层数削减**。delta：LCKV 跨层**共享 KV**（省层数）；QCMem 存**单层 hidden 并重算上层**（不共享，靠重算）。同"减少缓存的层"轴但一个共享 KV、一个存 hidden 重算。可作层轴压缩谱系背景。 |

**C 类小结**：固定预算赛道 = **token-eviction（H2O/SnapKV/StreamingLLM）** + **activation 压缩训练（Activation Beacon）** + **分层 KV 预算（PyramidKV/LCKV）**。QCMem 与全部的共同 delta = **存单层 hidden（非全 KV 也非压 KV）+ 检索式（非 eviction/window）+ 重算上层**。**效率表必须对照：StreamingLLM（已做，同 budget 头号对照）+ Activation Beacon（训练式超长压缩最强对手）**；H2O/SnapKV/PyramidKV 作 token/层-eviction 代表可选对照。

---

## D. 超长上下文 backbone（QCMem 声称 128k+ full-context 崩而它不崩，作对照锚点）

| arxiv id | 标题 | 机制一句 | 与 QCMem 精确 delta / 对照定位 |
|---|---|---|---|
| **2405.05254** | You Only Cache Once: Decoder-Decoder Architectures (YOCO; MSR, 2024-05) | **decoder-decoder** 架构：self-decoder 编码**全局 KV 一次**，上面叠 cross-decoder 经 cross-attention 复用同一份 KV；GPU 内存大降、保留全局 attention；扩到 **1M context near-perfect needle**；prefill 可 early-exit。from-scratch 架构。开源 aka.ms/YOCO。 | ★**架构级"只缓存一次"的最强超长 backbone 对照（1M niah 锚点）。** delta：(1) YOCO 省的是**层数×KV 份数**（只 cache 一层 KV 跨上层共享）；QCMem 省的是**缓存深度（存单层 hidden）+ 检索（不存全部 token）**。(2) YOCO **存所有 token 的那一份全局 KV**（内存仍 O(token)，只是常数小）；QCMem **检索 topk + 固定 read + 恒定显存**（与 token 数无关）。(3) YOCO 是 from-scratch 重训整个架构；QCMem 是现成 backbone + 自蒸馏 LoRA。**YOCO 是"别人怎么做超长 CL"的正面对照锚点——它的 1M near-perfect needle 是 QCMem 128k=100 的参照系**，related work 必须区分（YOCO cache-once vs QCMem cache-one-layer-hidden + retrieval）。 |
| **2402.04617** | InfLLM: Training-Free Long-Context Extrapolation with Efficient Context Memory (THU, 2024-02) | training-free：把远端上下文存入 **memory units**，用**高效 lookup 选 token-relevant unit** 进 attention；有限窗口模型无 fine-tune 扩到 **1024k** 仍抓长程依赖。开源 thunlp/InfLLM。 | ★**同时属 B（检索式）+ D（超长外推）的最强 training-free 对手。** delta：(1) InfLLM memory unit 存的是**block 的全层 KV**（representative token 索引 + 全 KV lookup）；QCMem 存**单层 hidden**。(2) InfLLM 检索在 **attention 内按 token 相关性**（每层 lookup）；QCMem 用**外部 bm25 一次检索 chunk + 上层重算**。(3) 都超长外推、都检索——**InfLLM 是 QCMem 精度对照的头号 training-free 检索式对手（同 niah 超长档，1024k）**。必须对照。 |
| **2402.04624** | MEMORYLLM: Towards Self-Updatable LLMs (2024-02) | transformer + latent space 内**固定大小 memory pool**，可用文本自更新、长期保留、~百万次更新无退化。开源 wangyu-ustc/MemoryLLM。 | ★**draft §2.3 已对照的固定 memory 方法（qa1/16k=57 vs 20，2.85×）。** delta：MemoryLLM 是**固定大小 latent memory pool + 参数化自更新**（memory 是模型的一部分、随更新演化）；QCMem 是**外部检索的 per-chunk hidden 缓存**（memory 是外部 KV-store、不演化、按 bm25 取）。MemoryLLM 无深度分区、无上层重算。**已是 babilong 精度对照，保留。** |
| **2308.15022** | Recursively Summarizing Enables Long-Term Dialogue Memory (2023-08) | 用 LLM **递归生成 summary/memory**（旧 memory + 新 context → 新 memory）支撑超长对话一致性；可补充长上下文/检索增强 LLM。 | 远亲：**文本空间的递归摘要 memory**（存自然语言摘要而非 hidden/KV）。delta：QCMem 存 **latent hidden**、无摘要生成、检索式。仅作"超长记忆的另一范式（text-space compression）"背景，非直接对手。 |

**D 类小结**：超长 backbone 对照锚点 = **YOCO（架构级 cache-once，1M niah）+ InfLLM（training-free 检索式 memory，1024k）+ MemoryLLM（固定 latent memory，已对照）**。QCMem 的差异化定位=**不重训架构（YOCO 要）+ 存单层 hidden 非全 KV（InfLLM/YOCO 存全 KV）+ 外部检索 + 恒定显存**。InfLLM 与 YOCO 是超长 niah 表里"别人做超 CL"最该并列的两行。

---

# ★★★ 核心交付：复现清单（Reproduction Checklist，按优先级）

**已有基建（决定复现难度）**：仓库里已存在 `scripts/eval_ruler_streamingllm.py` + `scripts/_run_streamingllm_ruler_8gpu.sh` + `scripts/merge_streamingllm_ruler.py`（StreamingLLM 已复现）；`external/landmark*` + `scripts/launch_landmark_S2.sh` + `scripts/eval_longeval_landmark.py`（faithful Landmark 已复现，⚠️仅 H20 可跑，L20A/torch2.10 跑不了）；`scripts/_launch_beacon_singlescale.sh` + `scripts/_eval_beacon_ppl.sh`（Activation Beacon 训练/eval 脚手架已在）；QCMem 自身 `eval_ruler_qcmem.py` 支持 `--resume_j 0`（=full-ctx 重算，self_test diff 0）/ `--resume_j L`（closed-book）/ `--reuse_kv_blockdiag`（block-diag ablation）/ `--selector bm25|reader_attn`。

## Tier 0 — 必须 head-to-head，reviewer 一定会问（不做站不住）

### 1. StreamingLLM (2309.17453) — 同 KV 预算精度对照【已完成，保留强化】
- **为什么必须**：QCMem 核心卖点"同固定 budget 下检索 > 最近窗口"必须有直接对照，否则 reviewer 问"你的恒定显存优势 vs 最简单的 window+sink 差在哪"。
- **复现难度**：★☆☆☆☆ 已复现（`eval_ruler_streamingllm.py`）。draft §2.1 已有 128k 100 vs 4。
- **对照指标**：同 ~6657-tok/~17GB 固定 budget 下的 niah 精度 + 显存 + 速度。
- **形态**：mechanism-level 推理侧（training-free baseline），已就位。**只需补齐 babilong 侧同 budget 数字 + 32k/256k 缺格**。

### 2. HCache (2410.05004) — "缓存中间层 hidden、上层重算"最直接前人
- **为什么必须**：这是 QCMem 推理形态最像的工作（存中间 activation、重算上层）。reviewer 会问"你和 HCache 除了加检索还有啥不同"。
- **复现难度**：★★★☆☆。HCache 是 **serving 系统（bubble-free 调度 + chunk storage），无开源明确训练模型**；**不需完整复现系统**，只需 **mechanism-level 推理侧对照**：在 QCMem 框架里跑 `--resume_j` 无检索 + 恢复全部被 evict token（=HCache 式"存中层→重算上层"但不检索、不限 read 长度），对比 QCMem 的"检索+固定 read"。QCMem eval 已支持 j-partial 重算，加个"取全部 chunk（不检索）"分支即可。
- **对照指标**：同精度下的 read 长度/速度/显存（QCMem 检索式固定 read vs HCache 式全量重算随 length 增长）——凸显 QCMem 的"read 与 context 长度无关"。
- **形态**：mechanism-level 推理侧复现（非完整 serving 系统），可在现有 eval 框架加分支。

### 3. KV-Direct / "Residual Stream Is All You Need" (2603.19664) — 缓存 hidden 重算的最新对撞（2026-03）
- **为什么必须**：2026-03 新论文，标题即"缓存 residual、按需重算 KV"，与 QCMem "缓存 hidden 重算"primitive **高度重叠且时间最近**，reviewer 极可能拿它质疑 novelty。有开源（github.com/Kaleemullahqasim/KV-Direct）。
- **复现难度**：★★☆☆☆。开源、training-free、纯推理。KV-Direct = **存 residual 重算全部层 K/V（full-depth）**；QCMem = **存单层 hidden 重算 layers[j:]（partial-depth）+ 检索**。对照只需跑 KV-Direct 的 bounded-memory 模式 vs QCMem，或在 QCMem 里设 j 对应"全深度重算"退化点。
- **对照指标**：内存增长（KV-Direct O(token) bounded vs QCMem 恒定）、超长档精度（KV-Direct 无检索→超出窗口仍会崩？QCMem 检索式不崩）、read 成本。**关键论证：KV-Direct 保留全部 token（内存随对话增长），QCMem 检索+恒定显存 → 超长档才是 QCMem 不可替代区间。**
- **形态**：mechanism-level 推理侧对照（跑开源 or 框架内退化点）。

## Tier 1 — 强烈建议，同赛道最强对手（有其一/其二才有说服力）

### 4. InfLLM (2402.04617) — training-free 检索式超长外推最强对手
- **为什么必须**：与 QCMem 同时"检索式 + 超长外推(1024k)"，training-free，开源（thunlp/InfLLM）。是"检索避免全 attention"赛道最强 training-free 对手；不对照会被问"vs InfLLM 检索 memory unit 差在哪"。
- **复现难度**：★★☆☆☆。开源、training-free、有现成 RULER/niah eval。QCMem 已有 RULER 框架，加一个 InfLLM baseline runner。
- **对照指标**：同超长档 niah 精度 + 显存（InfLLM 存 block 全层 KV vs QCMem 存单层 hidden → 存储/显存差）+ 速度。
- **形态**：mechanism-level 推理侧（跑开源 InfLLM 在同 RULER 档），不需训练。

### 5. Activation Beacon (2401.03462) — 训练式固定预算超长压缩最强对手
- **为什么必须**：QCMem 也训练（自蒸馏），Beacon 是"训练式压 activation 支持超长(128k)"的最强精度对手。仓库已有 Beacon 脚手架 → 顺手。reviewer 会问"训练式压缩 vs 你的训练式检索谁强"。
- **复现难度**：★★★☆☆。要训练（compression-based AR），但 `_launch_beacon_singlescale.sh` 已在 → 半成品。开源 FlagEmbedding。
- **对照指标**：同压缩比/预算下 niah + 长文档精度、超长外推、速度、显存。
- **形态**：完整复现权重（训练）——但脚手架已存在，成本可控。

### 6. YOCO (2405.05254) — 超长 backbone 锚点（表里引用即可，不必自训）
- **为什么必须**：1M near-perfect needle 是"别人怎么做超 CL"的最强锚点，related work 必须正面区分（cache-once vs cache-one-layer-hidden+retrieval）。
- **复现难度**：★★★★★ 完整复现要 from-scratch 重训架构 → **不复现**，直接引用论文数字作对照锚点。
- **对照指标**：超长 niah（引用其 1M 数字）+ 定性区分（架构重训 vs 现成 backbone+LoRA）。
- **形态**：论文数字引用，非复现。

## Tier 2 — 可选补充对照（同赛道代表，锦上添花）

- **Landmark Attention (2305.16300)**：检索式训练版经典，**本项目已有 faithful 复现**（`external/landmark*`，⚠️仅 H20 可跑）→ 低成本补一行对照。指标：检索式存全层 KV vs QCMem 存单层 hidden 的存储/精度。
- **CacheBlend (2405.16444)**：cross-chunk KV fusion serving 系统。QCMem 的"上层 full-attention 重算 vs block-diag"ablation（draft §6）已在 mechanism-level 回答同一痛点 → 引用 + 定性区分，不必复现系统。
- **H2O (2306.14048) / SnapKV (2404.14469) / PyramidKV (2406.02069)**：token/层-eviction 代表。选 1 个（H2O 最经典）作 token-space 固定预算对照即可，多数有开源、training-free、可入 RULER 框架。PyramidKV 的"金字塔信息漏斗"观察还可在 §3 层分工机制处引用呼应。
- **MemoryLLM (2402.04624)**：固定 latent memory，**babilong 已对照**（qa1/16k 57 vs 20）→ 保留。
- **FoT/LongLLaMA (2307.03170)**、**RAGCache (2404.12457)**、**LCKV (2405.10637)**、**递归摘要 (2308.15022)**：related work 提及+定性区分即可，不复现。

---

# 结论：QCMem 投稿最少需要复现哪几个才站得住？

**最小充分集（3 类 5 个，其中 3 个已有基建）**：

1. **StreamingLLM（已做）** — 同固定 budget 精度对照，证"检索 > 最近窗口"。【效率/精度赛道锚点】
2. **HCache（mechanism-level 推理侧）** — 证 QCMem vs "存中层重算上层"前人的 delta = **检索 + 固定 read + layer-partial**。【最像的推理形态，reviewer 必问】
3. **KV-Direct 2603.19664（跑开源 or 框架退化点）** — 证 QCMem vs "存 residual 重算 KV"最新对撞的 delta = **partial-depth + 检索 + 恒定显存（vs KV-Direct O(token) bounded）**。【2026-03 最新、novelty 防守关键】
4. **InfLLM（跑开源 baseline）** — 检索式超长外推最强 training-free 对手，证"存单层 hidden < 存 block 全 KV"。【检索赛道锚点】
5. **Activation Beacon（脚手架已在）** — 训练式固定预算超长压缩最强对手，证 QCMem 训练式检索 vs 训练式压缩。【训练式赛道锚点】

**YOCO 引用论文数字**（1M niah 锚点，不自训）。**Landmark 已有 faithful 复现**可低成本补一行。

**预判验证**：任务中"至少 HCache + 一个检索式 KV 复用 + StreamingLLM 已做"的预判**基本正确**，但需**加两条**：
- (a) **KV-Direct (2603.19664) 必须加**——它是 2026-03 最新、且与 QCMem "缓存 hidden 重算"primitive 撞得最狠的工作，是 novelty 防守的第一线（不对照会被直接质疑"和 KV-Direct 有啥区别"）。
- (b) **检索式对手首选 InfLLM 而非 CacheBlend/RAGCache**——InfLLM 是 training-free + 超长外推 + 检索式，与 QCMem 精度赛道正面可比；CacheBlend/RAGCache 是 serving 系统（TTFT 指标），mechanism-level 引用+区分即可。

**一句话**：**StreamingLLM(已做) + HCache + KV-Direct + InfLLM + Activation Beacon(脚手架在)** 是最小充分集；前 3 个防"缓存 hidden 重算"novelty（Tier 0），后 2 个防"固定预算/检索超长"赛道（Tier 1）。YOCO 引用、Landmark 已复现补格。复现负担可控：5 个里 3 个已有基建，2 个（HCache/KV-Direct）是 mechanism-level 推理侧对照可挂现有 `eval_ruler_qcmem.py`/`eval_ruler_streamingllm.py` 框架，无需从头训练。

---

## 附：所有 arxiv id 核对状态（均 curl 核对标题，无编造）
- ✅ 2410.05004 = HCache: Fast State Restoration in LLM Serving
- ✅ 2603.19664 = The Residual Stream Is All You Need (KV-Direct)【draft "需核实"→ id 正确】
- ✅ 2305.16300 = Landmark Attention
- ✅ 2307.03170 = Focused Transformer (LongLLaMA)
- ✅ 2405.16444 = CacheBlend
- ✅ 2404.12457 = RAGCache
- ✅ 2405.10637 = Layer-Condensed KV Cache (LCKV)
- ✅ 2309.17453 = StreamingLLM (Attention Sinks)
- ✅ 2306.14048 = H2O (Heavy-Hitter Oracle)
- ✅ 2404.14469 = SnapKV
- ✅ 2406.02069 = PyramidKV
- ✅ 2401.03462 = Activation Beacon
- ✅ 2405.05254 = YOCO
- ✅ 2402.04617 = InfLLM
- ✅ 2402.04624 = MemoryLLM
- ✅ 2308.15022 = Recursively Summarizing (dialogue memory)

---

## head-to-head 实现说明 (2026-07-08, mechanism-level KV-Direct & HCache)

**决策**：不跑别人 repo，在现有 `scripts/eval_ruler_qcmem.py` 里把 KV-Direct / HCache 表达成 QCMem primitive 的重参数化——同 backbone / 同 RULER 样本集 / 同 `string_match_all` 打分，唯一变量是被测的那个 primitive。commit `7dbef46`（RULER flag）+ `b8f3181`（`qcmem_generate` 的 `no_retrieval` 分支 + `read_len` stats）。

### 落地方式：`eval_ruler_qcmem.py --baseline {none,kvdirect,hcache}`
| baseline | resume_j | retrieval | LoRA | 机制含义 |
|---|---|---|---|---|
| `none`（QCMem） | 用户给的 `--resume_j`（如 12） | 是（`--selector` + `--topk`，固定 read） | 加载 | 检索 topk + layer-partial 重算 + 自蒸馏 |
| `kvdirect`（2603.19664） | **强制 0**（full-depth 重算全部层 K/V） | **否**（pack 全部 context chunk） | **忽略**（training-free） | 存 residual/embedding、按需全深度重算、保留所有 token、无检索 |
| `hcache`（2410.05004） | 保留用户 `--resume_j`（mid-layer 重算上层） | **否**（pack 全部 context chunk） | **忽略**（post-hoc 不训练） | 缓存中层 hidden、重算上层、恢复所有 token、无检索 |

- `no_retrieval=True`（两个 baseline 都开）→ `qcmem_generate` 跳过 selector/topk 筛选，按文档序 pack **全部** context chunk。read 长度 = sink + 所有 chunk 的 h_j + query，**随上下文 O(context) 增长**。
- QCMem（`none`）pack 固定 `topk` 个 chunk，read 长度**恒定**（与上下文无关）——这正是要凸显的 primitive 差异。
- `kvdirect` vs `hcache` 的唯一区别 = `resume_j`：kvdirect 强制 0（全深度重算，KV-Direct 的 "residual → 全层 K/V 零误差重算"），hcache 保留 j（只重算 layers[j:]，HCache 的 "中层 activation → 上层 KV"）。二者都不检索、都不用 LoRA。
- 现有 QCMem 行为完全不变：不传 `--baseline`（默认 `none`）即原样跑。`--reuse_kv_blockdiag` 与 `--baseline` 互斥（前者是 QCMem 内部 ablation）。

### 接口适配说明（实际 vs 任务描述）
- 任务描述里 "resume_j=0 = 从头全深度重算" **属实**：`QCMemModel.resume_j=0` 时 write=embed、read=从 layer 0 resume 全部层，self_test 已证 j=0 read == 全量 forward（fp32 diff<1e-4）。所以 KV-Direct ≈ resume_j=0 + 无检索，映射干净。
- 任务描述里 "HCache = 保持 resume_j + 无检索 + 不用 LoRA" **属实**：直接保留 `--resume_j` + `no_retrieval` + 丢弃 lora_adapter。
- 唯一实现细节：QCMem 没有现成 `--no_retrieval`，我在 `qcmem_generate` 加了 `no_retrieval` kwarg（pack 全 chunk）+ `stats` out-dict（记 `read_len`/`n_selected_chunks`/`n_context_chunks`），RULER 驱动把 `--baseline` 翻译成 `(resume_j, no_retrieval, lora)` 三元组。**未破坏 babilong 驱动的既有调用**（新参默认 `no_retrieval=False, stats=None`）。

### 对照什么指标
1. **精度**（RULER niah_single/niah_multi/vt recall）：同 read 预算下，QCMem 的**检索**让相关 chunk 进 read → 精度应 ≥ 无检索 baseline（尤其超长档 needle 分散时）。
2. **read 长度 / 显存 / 速度**（新增 `avg_read_len` 记入每 cell 的 summary.json）：QCMem `read_len` 恒定（≈ sink + topk×chunk_size + query）；kvdirect/hcache 的 `read_len` 随 length 线性增长（O(context)）——**这是 QCMem "read 与上下文长度无关" 的卖点数字**。超长档 baseline 会 OOM / read 爆炸，QCMem 恒定。
3. **kvdirect vs hcache**：隔离 "全深度重算 vs layer-partial 重算" 的精度/算力差（QCMem 的第二个 primitive）。

### 全量 eval 启动命令（等 seq4k 训完 GPU 空了 main 来跑）
三条命令共用同一 RULER 样本集（seed/shard 逐位一致），可挂 `scripts/_eval_taskpool_2group.sh` 分片，也可单卡直跑。QCMem arm 用已训 LoRA；两个 baseline arm training-free。

```bash
MODEL=/apdcephfs_wzc1/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b
PYBIN=.venv/bin/python   # diskB/B200 用 .venv
COMMON="--model_path $MODEL --chunk_size 512 --ruler_tasks niah_single niah_multi vt \
        --lengths 4k 8k 16k 32k --limit 50 --dtype bfloat16"

# 1) QCMem（检索 + layer-partial + LoRA）—— 主方法
$PYBIN scripts/eval_ruler_qcmem.py $COMMON \
    --baseline none --resume_j 12 --selector bm25 --topk 12 --sink_tokens bos \
    --lora_adapter outputs/qcmem_distill_qwen_j12_r32_4k/final \
    --output_name ruler_qcmem_j12 --results_folder ruler_results/h2h_qcmem_j12

# 2) HCache（中层重算上层 + 无检索 + 不训练）—— resume_j 与 QCMem 同深度做对照
$PYBIN scripts/eval_ruler_qcmem.py $COMMON \
    --baseline hcache --resume_j 12 --sink_tokens bos \
    --output_name ruler_hcache_j12 --results_folder ruler_results/h2h_hcache_j12

# 3) KV-Direct（全深度重算 + 无检索 + 不训练）—— resume_j 强制 0
$PYBIN scripts/eval_ruler_qcmem.py $COMMON \
    --baseline kvdirect --sink_tokens bos \
    --output_name ruler_kvdirect --results_folder ruler_results/h2h_kvdirect
```
（如用 2-组 task-pool：把上面每条包成 `CKPT`/`ADAPTER_CONFIG` 传给 `scripts/_eval_taskpool_2group.sh`，`EXTRA_ARGS` 携带 `--baseline ...`；或加 `--num_shards 4 --shard_index {0..3}` 单组 4 卡分片后用 `score_nested_babilong.py` 合并。）

### 预期结论（拉开点）
- **超长档（16k/32k）显存 & read 长度**：QCMem 恒定（read_len ≈ 1 + 12×512 + query ≈ 6.2k），HCache/KV-Direct 的 read_len 随 length 线性增长（32k 档 ≈ 全 chunk ≈ 60+ chunk → read_len 数万 tok），显存暴涨甚至 OOM。**这是 QCMem 最硬的卖点**。
- **同 read 预算精度**：QCMem 检索把 needle chunk 拉进固定 read → niah recall 应显著高于"无检索但被迫截断/爆显存"的 baseline；无检索 baseline 若不截断则算力爆炸，若截断则丢 needle。
- **kvdirect vs hcache**：二者精度接近（都无检索、都重算），差异在算力——kvdirect 重算全部层（贵），hcache 只重算上层（省），印证 QCMem "layer-partial 重算" 的算力价值；但两者都输在 read 随上下文增长。
