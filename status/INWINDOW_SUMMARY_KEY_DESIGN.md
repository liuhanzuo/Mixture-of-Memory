# 选择侧修复：in-window summary key（破墙第二步设计草案）

**作者**: landmark-repro (Landmark 机制) + methodA-eval (mem_space 接入)  **日期**: 2026-06-20
**状态**: DRAFT，待 methodA-eval 评估接入可行性 + chunk512 单轴消费结果。
**前置**: chunk512 单轴 grouped-readout 重训(PID 4178815)在测「消费侧」；本设计攻「选择侧」。

---

## 0. 为什么需要这个（两难的解）

- **H2**：显式训练选择器(gist scorer，跨块选择折进 LM loss)→ 学不出(LM loss 对"选含 needle 的块"无梯度压力)。
- **变体B**：未训练的 reader-attn 选择(no_grad 固定)→ 选不准,硬 bias 到错 sub-block 比等权更糟。
- **两难**：选择不能直接训(H2)、不训也不准(变体B)。

**Landmark 的解(代码级证实,llama_mem.py + train.py)**：
- 无任何 aux/selection loss——只有 LM loss。
- `<landmark>` 是每 mem_freq 插入的**可训练 special token**,训练时**在窗口内**学:窗口内后续 token 经 grouped-softmax 通过 landmark-token 这个 bottleneck attend 回它代表的 mem_freq 段 → landmark-token 收**密集的 in-window 梯度**,学会"概括本段"。
- 训练**不跑**跨块检索(cache_top_k=None,past_key_value=None)。跨块选择是**推理期涌现**:用 in-window 训好的概括 key 做 query·landmark-key。
- **关键洞察**:选择 key 不是从"跨块选择"学的(H2,无梯度),是从"in-window 概括"学的(密集梯度)。绕开 H2 靠**代理目标**,不靠架构。

## 1. 设计：可训练 per-block summary key + in-window 概括目标

**与 Landmark 的对应(同构,不插 token 避免改 tokenize/label)**：
- 不在 input_ids 插 `<landmark>`(那要改 tokenize + label 移位,大改 + 影响 dolmino/T2 数据管线)。
- 改为**可训练投影 `summary_proj`**:per-block summary key = `summary_proj(block hidden 的 pool)`。`summary_proj` 是新可训练参数(类比 gist_proj,但训练目标不同)。

**in-window 概括目标(绕 H2 的核心)**：
- 训练时,**当前窗口**的 self-attention 里,块内后续 token 对**本块自己的 summary key** 做 grouped-softmax(bottleneck)→ summary_proj 收 in-window 密集梯度,学"概括本块"。
- 梯度来源 = LM loss(经 in-window grouped-softmax),**无独立 aux loss**(保持 Landmark-faithful 纯 LM loss)。

**推理涌现**：
- 跨块检索:query · 各 block 的 summary key(in-window 训好的概括 key)→ 外层选 block。
- 这就把外层选择 key 从"未训练 reader native q·k(73% ceiling)"换成"in-window 训练的 summary key",**预期外层命中率超 73%**。

## 2. 两层最终架构（消费 + 选择都解）

| 层 | 机制 | 训练 | 守 H2? |
|---|---|---|---|
| 外层选择(选哪些 block) | query · **summary key** topk + gather(硬) | summary_key 由 **in-window 概括**训(密集梯度);gather 硬 argmax 不进 loss | ✅ 不训跨块选择 |
| 内层读出(block 内 token mass) | group_lse 顶层 + within-block softmax | answer 梯度经 grouped-softmax(= chunk512 单轴在测的) | ✅ 涌现 |

## 3. 接入点（methodA-eval 评估改动量,2026-06-20）

1. **store**:写入加 `summary_proj(chunk_hidden_pool)` 存 per-block summary key。**低改动 ✓ ~20-30 行**(RawKVReadoutStore 加 Linear + 写入存字段,类比 gist_src)。
2. **in-window 概括目标**:训练时当前 chunk self-attention 加对"本 chunk summary key"的 grouped-softmax bottleneck。⚠️ **高改动 + 深层可行性问题（核心,见 §4）**。inattn forward 只在 readout/injection 路径触发,当前 chunk 自己走原生 frozen self_attn,要加 in-window bottleneck 必须新 hook 进 frozen self_attn,碰当前窗口 attention(不只 retrieved)。
3. **推理**:外层 topk 的 score 从 reader native q·k 换成 query·summary_key。**低改动 ✓ ~10 行**(layer.py:1135 _reader_attn_keep_set 换 score 来源)。

**★1+3 是壳,2 是魂**:点1+3 单独没用——只换 summary key 来源,但若 summary_proj 没被 in-window 目标训过,它就是**随机投影**,选不准。点2(用 in-window 概括目标训 summary_proj)是让 summary key 有意义的前提。B 的价值全压在点2 能否做出**真 bottleneck**。

## 4. ★核心结论：旁路必败,必须真 bottleneck（2026-06-20,landmark-repro+methodA-eval 敲定）

**Landmark 的 bottleneck 是「物理插 token + grouped-softmax 截断直连」实现的,不是「加一条旁路 attention」。** `<landmark>` 真在序列里,grouped-softmax(llama_mem.py:241 `full_access_mask`/`last_section_mask`)把"非当前段 token 直接 attend 更早段"的路径**截断**,强制经 landmark 中转 → landmark-token 收到**密集 in-window 梯度**(它是必经瓶颈)。

→ **关键:bottleneck = 切断直连、只留经 summary 的路;NOT 加一条到 summary 的旁路。**
- **summary_proj 旁路(加 attention 到 summary key,但 native 块内直连还在)= token 走 native 绕过 → summary_proj 拿不到密集梯度 = H2 旁路风险翻版。必败,别做。**
- 真 bottleneck 两条路(都比旁路重,等 step500 消费结果定 + 优先 B-插token):
  - **(B-插token)**:像 Landmark 物理插 summary token + grouped-softmax 截断块内直连。碰 tokenize + label 移位(大),但真 bottleneck,**优先试这个**。
  - **(B-截断)**:不插 token,但在当前 chunk attention 里用 grouped-softmax 截断"token 直接 attend 块内更早 token"强制经 summary key。碰 frozen self_attn 结构,且要 ablation 确认不破坏正常 LM(短程不退化)。

## 5. 风险 / 开放问题
- in-window 概括目标怎么不破坏当前窗口正常 LM——需 ablation 确认不退化短程(B-截断 尤其要测)。
- summary_proj pool 方式(mean / attention-pool / 学习 query)。
- 改动量大(碰当前窗口 attention)→ **先等 chunk512 单轴消费结果**:若消费侧训练已把 4k 拉到 ~57.5(逼近 chunk-oracle 上限),说明内层够、瓶颈纯在外层选择 → summary key 值得做(走 B-插token);若消费侧仍~14,问题更深,先排查消费,B 缓。

## 5. 执行顺序
1. **[RUNNING]** chunk512 单轴 grouped-readout 重训(消费侧,PID 4178815)。
2. step500 W0 → 判消费侧贡献(4k 14→?）。
3. 若消费侧有效 + 本设计 methodA-eval 评估可行 → 实装 summary key(选择侧)→ 第二个重训。
4. 仍不够 → (A) per-layer per-head 投票(我已给逻辑)进一步提外层命中。
