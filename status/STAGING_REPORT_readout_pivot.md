# 阶段性报告：从「逼 memory」到「读出瓶颈」（2026-06-14）

> 本报告承接上一阶段 harder-objective 实验，给出阶段性结论，并把下一步焦点正式锁定在**读出瓶颈**上。
> 配套数据细节见 `status/HARDOBJ_FINAL_REPORT.md`（depth×seed×容量全矩阵）。

## 1. 上一阶段做了什么

针对「纯 LM next-token 训练没给 memory 学长程检索的压力」这个根因，实现并验证了 harder-objective `--last_chunk_loss_only`：
- context chunk 全程 `no_grad` 流式写入 memory bank
- LM loss **只**压在最后的 target chunk
- → target 的预测无法靠局部注意力，必须从 memory 回忆前文

在通用 dolmino 文本上训练（不混 BABILong，避免拟合 eval），全维度扫了 depth(ctx3/5/7) × seed(42/1234/2026) × 容量(N96/128/192/256)。

## 2. 阶段性结论（已 seed-robust 坐实）

**判据（用户校准）= W0（纯 memory 读出，生成窗口仅最后 1 块）相对 BASE_mix0 的提升**，不拿 SWA gap 当成败线。

| 维度 | 结论 |
|---|---|
| 有效性 | 纯 memory 长程 qa5 从基线 5/5/3 → **~13/11/8（2-3×）** |
| seed | **稳**：ctx3 两 seed 13/9/6 完全一致，ctx5/ctx7 多 seed 全复现 |
| 深度 | 温和单调正效应（16k 档 ctx3=9 → ctx7=11-14），ctx7 最强，ctx10 结构性 hang |
| 容量 | **完全无关**（N96→N256 全程持平）——扩 memory 不是杠杆，再次坐实 |

**方向正确，但撞 plateau ~11-15，无质变。**

## 3. ★核心认知：瓶颈在「读出」不在「存储」

把 W0（纯 memory）和 W6（SWA，开卷直接看上下文原始 KV）并排看：

| 口径 | qa5 8k / 16k / 32k | 含义 |
|---|---|---|
| W0 纯 memory | ~13 / 11 / 8 | memory 通道自己能读出的 |
| W6 SWA 上限 | ~50-60 / ~40-50 / ~30-40 | 信息「在上下文里可达」的上限 |

**Gap 巨大（13 vs 50+）。** 关键解读：
- 这 ~13 分**不是随机**（远超基线 5/5/3）→ memory **确实承载了**长程信息，存储侧 work。
- 但离 SWA 上限差一大截 → **大部分被 memory 编码进去的信息，读出阶段没能取回来。**
- harder-objective 把存储侧压力拉满了（target 只能靠 memory），却**没能把读出效率提上去** → 说明瓶颈不在「逼不逼 memory 去存」，而在「存进去之后能不能精确取回」。

**这就是下一阶段要攻的点：读出（readout）。**

## 4. 为什么 readout 是真瓶颈（机制层面）

回顾此前写入/读出侧诊断（`status/MEMORY_UPDATE_DIAGNOSIS.md` / `MEMORY_REDESIGN_DIRECTIONS.md`）：
- 读路径 95% 注意力落在「从未写入」的槽（chunk-0 token 快照），活槽只承载 ~5%（dead_slot_read_mass 诊断）。
- 读出是 content-based softmax over all-N slots：query 和 slot key 的相似度决定取哪个槽。
- v20 读基生命周期改造（soft-decay / hard-eviction）已证伪——**强行干预读分布伤长程**。

→ 当前读出是「按 key 相似度均匀铺在一大片槽上」，**没有结构、没有精确寻址**。长程检索（qa5：找某个特定事实）需要的恰恰是**精确寻址**——query 能定位到「写入那条信息的那个槽」。harder-objective 让信息进了 memory，但读出时找不准。

## 5. 下一阶段方案：T2 合成 recall 任务 — 专门施加「精确读出」压力

**核心思路**：纯 LM / 通用文本里，target 往往能靠「上下文语义连贯性」蒙对，对精确寻址的压力不足。合成 associative-recall 任务**消除蒙对空间**：答案是随机 key→value 映射，只能靠从 memory 精确取回写入时的那条 (key,value)。

### 5.1 与现有 NIAH 数据集的关系（关键设计点）
`src/memory/mem_space/niah_dataset.py` 已有 NIAH（单 needle 找 5 位密码），但它是**旧格式**：yield 扁平 `input_ids`+`labels`，走普通 forward → **局部注意力能直接看到 needle，没强制走 memory**。这正是要修的。

**T2 = NIAH 的内容 × harder-objective 的机制**：
- 复用 NIAH 的合成逻辑（随机 name→code 映射、needle 句、question+answer）
- 但改成 **chunked 格式**（`context_chunks` + `target_ids`），喂给 `dolmino_train_step`：
  - needle 散在 context chunks（`no_grad` 入 memory）
  - question 在 target chunk，loss **只**压在 answer 的 5 位 digit
  - → 答案**只能**从 memory 取回，局部注意力看不到 needle（它在 no_grad 的 context 里，不在 target 窗口）

### 5.2 任务难度阶梯（由易到难）
- **T2a associative recall**：context 散布 K 条 (name_i → code_i)，target 问其中一个 name 的 code。判别「精确寻址」最干净。
- **T2b multi-key**：增加干扰 key 数量（写 8 条，问 1 条），逼读出在多候选中精确选。
- **T2c multi-hop**（后续）：答案需串联两条 fact。

### 5.3 验证判据
- **训练信号有效性**：用 P11 / harder-objective step500 ckpt 对 T2a 做 chunked(W0) eval —— 应该差（现 readout 取不准）；SWA eval 应该好（信息在）。gap 越大说明这个任务越能逼读出。
- **训练后**：T2 训出的 ckpt 在 **BABILong W0** 上长程 qa5 是否**突破 plateau ~11-15** → 突破=施加精确读出压力有效，readout 是可训的；不破=读出瓶颈是架构性的，需换读机制（而非换训练信号）。
- ⚠️ T2 是合成训练数据，**eval 仍用 held-out BABILong**（T2 的 name/code 是随机的，与 BABILong 无重叠，不构成拟合）。

### 5.4 ★chunk_size 是 readout 的直接杠杆（用户 2026-06-14 提点，必须细致处理）

chunk_size 决定**每次 memory 写入压缩的文本粒度**，直接影响读出精度：

- 一条 needle ~20-30 token。在 **chunk_size=512** 下只占 ~5%，写入时被其余 ~480 个背景 token 稀释 → 编码进 slot 的向量不干净 → 读出取不准。
- chunk_size 越小 → needle 在「单次写入覆盖的文本」里占比越大 → (key→value) 编码越干净、越接近「一个槽承载一条 fact」→ **精确寻址越可能成立**。
- 极端假设：若 chunk_size ≈ needle 长度，每次写入几乎只编码一条信息，readout 退化为「查表」——这正是我们想要的精确寻址。

**因此 chunk_size 不是固定 512，而是 T2 的一等实验变量**，与现有 chunk512 体系并列扫：

| chunk_size | 每 chunk 写入粒度 | 预期 readout | 代价 |
|---|---|---|---|
| 512（现行） | 粗（needle 占 ~5%） | 稀释严重，基线 | 序列短、chunk 数少 |
| 256 | 中 | needle 占 ~10%，应更干净 | chunk 数 ×2 |
| 128 | 细 | needle 占 ~20%，最接近查表 | chunk 数 ×4，写入次数多 |

**实现注意（关键约束）**：
1. **train/eval chunk_size 必须一致** —— adapter 在某 chunk_size 下训练，BABILong eval 必须同 chunk_size（`run_babilong_mem_space.py --chunk_size` 对齐），否则 memory 写入粒度错配，结果不可比。已有 chunk256/chunk1024 的 launch 脚本可参考。
2. **needle 不能跨 chunk 边界被切断** —— 小 chunk_size 下 needle 句（~25 token）仍应整条落在单个 chunk 内（数据生成时保证 insert_at + len(needle) ≤ chunk_size）。
3. **固定总上下文长度做公平对照** —— 比较 chunk_size 时，固定「needle 到 query 的 token 距离」（而非固定 chunk 数），否则混淆「粒度效应」和「距离效应」。即 chunk128 用 4× chunk 数覆盖与 chunk512 相同的 token gap。
4. **gradient/显存**：小 chunk_size → 同样 token gap 需更多 chunk → 更多 memory 写入 forward → 显存/速度成本上升，可能需降 batch_size。

**首批 T2 设计矩阵**：T2a(associative recall) × chunk_size{512, 256, 128} × ctx7-等效深度，固定 token gap。先用 P11/harder-objective ckpt 做零成本任务甄别（5.3），再训。

### 5.5 实现范围（coder 任务）
1. 新增 `niah_chunked_dataset.py`（或给 niah_dataset 加 `chunked=True` 模式）：yield `{context_chunks: [..], target_ids, answer_mask}`，**chunk_size 参数化**，保证 needle 整条落单 chunk + 固定 token gap。
2. `dolmino_train_step` 已支持 context no_grad + target loss；需让它接受 answer-position label mask（只在 answer digit 上算 loss，而非整个 target chunk）。
3. 新 flag `--t2_recall_mix_fraction` / `--t2_num_keys` / `--t2_max_gap` + 复用 `--chunk_size`（扫 512/256/128），与 dolmino 文本混合训练（避免纯合成任务破坏 LM）。
4. 启动脚本 `launch_T2_recall_*.sh`（按 chunk_size 分脚本，对齐 eval chunk_size）。

## 5b. 阶段性/课程式训练想法（用户 2026-06-14 提点，待 T2 静态基线后探索）

用户提出：除了固定配置，还可以用**分阶段/课程**的方式训练，几个具体方向：

### (i) L1 / L3 互相影响 → 分阶段解耦训练
- 已知：**L3（Q-Former 64 summary token）是长程主力**，L1-only 关掉 L3 后 qa5 1k 从 89 崩到 4 → L1 单独几乎不 work。但 L1（离散槽 top-k 写 + all-N 读）和 L3 同时训练，**两者可能互相干扰**（梯度互相拉扯、读出注意力被两条通道瓜分）。
- 课程式方案候选：**先训 L3（建立长程主干）→ 再加 L1（精修局部/精确寻址）**，或反过来；或交替冻结其一。目的：看读出瓶颈是「L1/L3 耦合训练导致谁都没学好读出」还是「单通道本身的固有上限」。这正好与 readout 焦点对齐——如果分阶段能让某一通道专注精确寻址，plateau 可能松动。
- ⚠️ 需先确认当前是否有 freeze/分阶段开关（查 config.py 的 disable_l1_inject / use_l3_summary 等），可能已有部件可复用。

### (ii) 渐进式 chunk_size 课程（progressive chunk_size）
- 在 §5.4 静态扫（512/256/128）的基础上，进一步做**课程**：训练**先用小 chunk_size**（needle 占比大、编码干净、读出容易——让模型先学会精确寻址的「能力」）→ **逐步增大 chunk_size**（稀释加重，逼模型在更难的粒度下**保持**读出精度）。
- 直觉：先教会「精确取回」这个技能，再让它在越来越脏的写入下不退化——类似「先学会再加噪」。与现有 curriculum（n_ctx 深度递增）正交，可叠加。
- 判据：渐进式 vs 固定 chunk_size 的 W0 长程，看课程是否把 plateau 顶上去。

### (iii) 渐进式 training 难度（更一般）
- 把「难度」抽象成多个旋钮：n_ctx 深度（已有 curriculum）、chunk_size 粒度、干扰 key 数（T2 的 num_keys）、needle→query 距离（gap_tokens）。课程可在任意维度递增。
- 实施次序建议：**先拿到 T2 静态基线**（固定各旋钮，确认「chunked 难/SWA 易」+ chunk_size 对 readout 的单调影响），**再设计课程**——否则不知道往哪个方向「递增」才有意义。

**优先级**：T2 静态实现（进行中）→ 静态 chunk_size 扫 + 任务甄别 → 据结果选 (i)/(ii)/(iii) 中最有信号的做课程。不一次性全上，避免多变量混淆。

## 6. 当前集群状态
harder-objective 收尾 run 仍在跑（N192 2nd-seed 训练 + N96/N256 swa6 eval）。T2 实现就绪后，下一个空闲节点即起 T2a 首跑。

## 7. 训练长度 confound + 长数据对照（2026-06-14 用户提点）

### 7.1 当前训练序列长度（关键发现）
- **T2 512/256/128 档（固定 gap=3584）**：总序列仅 ~3.7-4.1k tokens（n_ctx×chunk+target）。
- **T2 1024/2048/4096 档（固定 n_ctx=7）**：8k / 16k / 32k tokens。
- **dolmino 背景（50%, curriculum 0:3, chunk512）**：~2k tokens。
- ⚠️ **结论**：512/256/128 档训练时模型**从没见过 16k/32k**，BABILong 长程 eval 是纯外推 → plateau 可能部分源于**训练长度不足**，非纯读出机制问题。

### 7.2 confound：大 chunk 好 vs 训练序列长就好
chunk2048/4096 训练序列恰好是 16k/32k（首次覆盖 eval 长程区间）。若它们长程 W0 改善，无法区分是「大 chunk 粒度」还是「训练序列终于够长」。
**对照实验（用户思路）**：固定 chunk512，把 n_ctx 加大让训练序列也到 16k/32k（n_ctx=32→16k, n_ctx=64→32k），与 chunk2048/4096 同序列长度但小 chunk。
- chunk512×n_ctx64（32k 序列）≈ chunk4096×n_ctx7（32k 序列）：同总长，差在 chunk 粒度 + 写入次数（64 vs 7）。
- 这样能三方分离：序列长度 / chunk 粒度 / 写入次数。

### 7.3 长训练数据现状（待解决）
- `MemLong/data/processed/dolmino_longdoc_wiki_min4k`：99899 篇，min4k **median 6.3k / max 16.3k / 0% 达 32k**。→ 只够 ~16k，**不足以喂 chunk512×32k**。
- T2 合成任务**不依赖长文档**（needle 嵌入合成背景，gap 任意设）→ T2 路线可直接用大 n_ctx 把序列拉到 32k，无需新数据。
- 若要 dolmino 真实长文本到 32k：需重筛（dolmino 原始多为短文；wiki min4k 已是筛过的上限）或换源（pg19/books 长文）。

### 7.4 下一步（T2 内做训练长度对照，零新数据成本）
T2 是合成任务，gap_tokens 任意 → **固定 chunk512，扫 gap_tokens=3584/7168/14336/28672（n_ctx=7/14/28/56，序列 4k/8k/16k/32k）**。
- 与现有 chunk512×4k 同 chunk 粒度，纯变训练序列长度 → 直接测「训练长度」单变量效应。
- 与 chunk2048×16k / chunk4096×32k 对比 → 分离「大 chunk」vs「长序列」。

## 8. ≥32k 长文档数据调研（2026-06-14）

需求：训练序列覆盖 BABILong eval 长程区间（16k/32k），现有 dolmino 全部 ≤4-16k 不够。

### 本地数据盘点
| 数据 | 长度 | Llama-3 token? | 可用性 |
|---|---|---|---|
| dolmino_per_doc | median 1k, max 4k | ✓ | ✗ 太短 |
| dolmino_0.5B_1024 | 固定 1024 | ✓ | ✗ 太短 |
| dolmino_longdoc_wiki_min4k | median 6.3k, max 16k | ✓ | △ 最多 16k |
| **slimpajama-per-source-length-upsample** | **每行 131072 (128k)!** | ✗ **非Llama3(max token id 31979,~32k vocab)** | ✗ tokenizer不匹配,直接喂是垃圾 |
| dolmino_pes2o_wiki/raw (pes2o) | abstracts median 247 tok, max ~1.4k | 原始文本 | ✗ 是摘要非全文 |
| **data/pg19_train.jsonl (11.4GB raw text)** | **book-length,天然60k-100k+/本** | 原始文本(可自己tokenize) | ✅ **最佳候选** |
| data/pg19_chunks_llama3.npy | 5916×4096(已切块) | ✓ | △ 已切4k块,丢了per-book长度 |

### 结论 & 推荐
1. **slimpajama-length-upsample 是128k长文但tokenizer错配**(非Llama-3)→不能直接用。若要用需拿原始文本重tokenize(未找到原始文本,只有token)。
2. **pg19_train.jsonl(11.4GB原始书籍文本)= 最佳现成源**：PG19是book-length语料,单本天然60k-100k+ tokens,且是raw text→用Llama-3自己tokenize无错配。需写预处理:按书切分(当前是concatenated text)→per-book tokenize→存per-doc arrow(类似dolmino_per_doc流程)。
3. 备选:重新下载peS2o **full-text**版(非本地的abstracts版)或arxiv,但需联网+大量处理。

### 推荐下一步
派 coder 写 pg19 per-book 预处理(raw jsonl→Llama-3 tokenized per-doc,筛 ≥某长度),产出 pg19_perbook_min8k 之类,用于:
- chunk512 × 长序列(n_ctx 拉到 32/64)真实文本对照(分离"训练长度"vs"大chunk")
- 也可作为 T2 的真实长背景(替代合成/短dolmino背景)

---

## 9. chunk_size × seq_len 网格结果（2026-06-15）— 长度效应 + curriculum 启动

### 9.1 完整网格（W0 纯 memory readout，babilong qa5，FINAL ckpt）

总数据量全部对齐 16000 样本（H20 bs1×1000 / B200 bs8×125），可比。

| chunk × seq | 0k | 1k | 2k | 4k | 8k | 16k | 32k |
|-------------|----|----|----|----|----|-----|-----|
| **c512 × 8k** | 73 | 54 | 54 | 23 | 16 | 10 | 9 |
| **c512 × 16k** | 72 | 31 | 50 | 21 | 17 | 10 | 10 | (step500;final评估中) |
| **c512 × 32k** | 66 | 29 | 38 | 19 | 12 | 8 | 8 |
| **c1024 × 8k** | （.196 评估中） | | | | | | |
| **c1024 × 16k** | 25 | 52 | **55** | **48** | **30** | 15 | 12 |
| **c1024 × 32k** | 74 | 52 | 48 | 23 | 14 | 12 | 10 |

### 9.2 与历史基线对比（qa5 W0，8k/16k/32k）

| 训练 | 8k | 16k | 32k |
|------|----|-----|-----|
| BASE_mix0（纯 LM 锚点） | 5 | 5 | 3 |
| harder-obj ctx3/7（旧线 plateau） | 13-15 | 11-14 | 8-9 |
| **T2 c1024×16k（当前最佳）** | **30** | **15** | **12** |
| harder-obj SWA 上限（开卷参考） | 51-63 | 52 | 39 |

**T2 合成 recall 训练是真突破**：中短程（4k）从 harder-obj 的 ~28 抬到 48，8k 从 13-15 抬到 30，约翻倍，逼近 SWA 开卷上限一半。证明合成 needle 施加了纯 LM 缺失的"精确寻址"压力。但 32k 仍只 10-12，**长程未攻克**。

### 9.3 长度效应（修正先前"长度无用"的误判）

固定 chunk512 看 seq 长度：4k→8k qa5 全档抬升（1k:31→54, 8k:13→16, 32k:6→9）——**长度有用**。先前"无用"是被 c1024 跨 chunk 的退化点误导。

但 **8k 之上出现退化**：c512 的 8k→32k，中长程反降（4k:23→19, 8k:16→12）；c1024 的 16k→32k 同样（4k:48→23, 8k:30→14）。**说明存在最优训练长度（~8-16k），直接堆 32k 反而学不好**——模型一上来面对超长上下文抓不到 readout 结构。这正是 curriculum 的动机。

### 9.4 ★Curriculum 已实现并启动（用户 2026-06-15 拍板）

- **代码改造（commit 2c22d80）**：`NIAHChunkedDataset.set_n_ctx()` + 训练循环中紧随 dolmino `set_n_context` 调用，让 **T2 needle→query 距离也随 curriculum 拉长**（先前只 dolmino 背景跟随，T2 gap 固定）。num_workers=0（默认）保证 setter 传播。
- **B200 run**：`launch_T2_recall_chunk1024_CURRIC_4to32k_B200.sh`，chunk1024，curriculum `0:4,125:8,250:16,375:32`（n_ctx 4→8→16→32 = 4K→8K→16K→32K），每阶段 16000 样本（bs8×125步），save_interval=125 → 每阶段末出 ckpt 看轨迹。**待 B200 当前 eval 收尾后启动**。
- **预期验证**：若 curriculum 的 32k-final 在 16k/32k 长程上超过直接训 c1024×32k（14/12/10）甚至 c1024×16k（30/15/12），则证明渐进式是攻克长程 readout 的有效路径。

