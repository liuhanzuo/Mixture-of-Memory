# 32K 长程精确读出墙 —— 一个干净、完整的负结果

> 整合自 `status/RUN_REGISTRY.md`（§3–§4）+ `status/SESSION_HANDOFF.md` + `status/TRAINER_ACTIVITY.jsonl` + git log。
> 所有数字均从上述总账逐条核对，未核实处显式标注「待补」。
> 撰写：2026-06-18。架构 = frozen Meta-Llama-3-8B + 可训练 memory adapter（mem_space，128 slot），研究方向 = 固定大小 memory buffer 压缩长上下文做长程精确事实读出。

---

## 摘要

经过约 25 个训练/评测实验，我们得到一个**干净、完整、有价值的负结果**：

> **在「冻结 backbone（reader）+ 固定 128-slot 压缩 memory」这一范式下，32k 长程精确事实读出存在一堵对几乎所有干预免疫的墙。其根因不是写入/容量/路由/检索源，而是冻结 reader 在固定压缩瓶颈下的信息论与「消费能力」限制。**

证据强度逐级递进：

1. **诊断层**：slot 写入侧健康（0 dead、用满 128），但读出侧失效——多个独立 benchmark 一致指向「保 gist、丢精确事实」。
2. **训练侧旋钮全证伪**：训练窗口（4k→8k→32k）、压缩比（128→256 slot）、训练侧 mass bias —— 没有一个能抬高 32k。加大训练窗口甚至与长程性能**单调负相关**。
3. **蒸馏**：dolmino（短窗）蒸馏中长程塌；唯有 pg19 真长文蒸馏在 **16k 档破天花板（+3）**，但 **32k 仍 = 9**，且不迁移到真实长文档（LongBench）。
4. **evidence / retrieval injection 全证伪**：slot-routed evidence、oracle 完美证据注入、raw-KV 检索、★TRUE in-attention native K/V concat（完美 needle 100% 命中）—— 四条注入路线全部 ≈ OFF。**给冻结 reader 完美的正确证据，它也用不上。**
5. **外部文献佐证**：recall vs state-size 的 Pareto 前沿（Based/Zoology）、gist-token 压缩比与重建率（ACL2025）—— 均与我们观察到的压缩瓶颈一致。

**结论**：固定 slot 压缩 + 冻结 reader 无法做长程精确读出，已经过两种 readout 机制（prefix 注入 + native in-attention）× 多种检索源（slot / raw-KV / oracle-perfect）双重证伪。**唯一尚未关闭的路径 = 解冻 reader 部分层 finetune，或换范式。**

---

## 1. 诊断：墙的形状

固定 128-slot bank 在「需要全局总结」的任务上弱，在「精确单条检索」的任务上更是断崖式失效。多个口径互证，确认问题在**读出端精确事实绑定**，而非写入/容量。

| benchmark | 现象 | 数字（mem_space P11 SOTA W0 vs 参照） |
|---|---|---|
| **BABILong qa5（W0 纯 memory）** | 长程随长度衰减，32k 卡死 | P11 chunk512 step500 = 74/89/81/60/48/45/44（0k→32k）；**32k≈44 远超随机，但所有方法 32k qa5 天花板 ≈ 9（n_ctx7 W0 口径）** |
| **LongEval lines-retrieval（W0 纯检索探针）** | **≥8k 精确单条检索直接归零（非渐进衰减）** | 1k/2k/4k/8k/16k/32k = 8%/14%/10%/**0%**/**0%**/**0%**（n=50，精确 6 位数）|
| **对话记忆 LongMemEval（n=500 全 6 题型对齐）** | 保 gist 丢精确事实 | mem 10.4/5.5 vs base 39.2/10.9，**base ≈ 3.8×**；single-user 18.6 vs 74.3 |
| **对话记忆 LOCOMO（n=400）** | 同上，差距更大 | mem 2.75 vs base 19.25，**base ≈ 7×**；时序类 0 vs 17.8 |
| **Sliding-window 32k PPL** | 长程依赖越强、退化越重 | codeparrot +44% / proofpile +41% / **pg19 +109%**（commit fe0f28d）|

**关键诊断事实**：
- **写入侧健康**：slot 0 dead、用满 128（slot 分布诊断）。墙不在写入端。
- **读出侧失效是开关性的**：LongEval 在 8k（16 chunk）处突变归零，暗示 readout query 在 chunk 数超阈后分辨不出目标 slot，而非压缩比连续退化。
- **输出非崩溃**：mem_space 生成连贯、主题相关，但事实细节错（如「4年3个月」vs gold「4年9个月」）—— 典型的「记 gist、丢精确」。

---

## 2. 三大证伪家族

### 家族 A — 训练侧旋钮：窗口 / 压缩比 / mass 全部免疫

主线假设链「加大训练窗口 → 见过真长程依赖 → 破 32k」被三个独立负结果逐一证伪。所有实验配方一致（pg19 真长文蒸馏，chunk512，128 slot，distill λ0.6 layers 12/20/28，seed42，500 步），仅改一个变量。

| 实验 | 变量 | qa5 8k / 16k / 32k（W0） | 裁决 | commit |
|---|---|---|---|---|
| pg19 n_ctx7（基线，唯一破墙者）| 训练窗口 4096 | 19 / **16** / 9 | **16k 破天花板（+3 vs 13）** | 93f5d77 谱系 |
| pg19 n_ctx15 | 训练窗口 8192 | 17 / 15 / 8 | 32k 未破（8≤9），16k 反降 | 待补* |
| pg19 N256 | 压缩比 250:1→125:1 | 17 / 16 / 9 | 32k 无效，16k 持平 | 待补* |
| **pg19 n_ctx63** | 训练窗口 **32768 = 推理长度** | 14 / **12** / 9 | **决定性：32k 持平、16k 反低于 n_ctx7** | **4812d70** |
| pg19 n_ctx63 + weak-mass 0.5 | 训练侧 mass 杠杆 | 15 / 13 / 7 | 与 plain 噪声内持平，32k 反降 | **db584a6** |

**两个硬结论**：
1. **「训练窗口 = 推理长度」假设彻底证伪，且呈单调负相关**：16k 档 n_ctx7(4k窗口)=16 > n_ctx15(8k)=15 > n_ctx63(32k窗口)=12。把训练窗口推到完整 32768 token 也没破墙——机理推断为：长窗口让每样本上下文更长、有效梯度信号更稀疏（同 500 步内 token-level 监督密度下降），中程反而退化。**16k 甜点是 n_ctx7（约 8× 短于推理长度）。**
2. **训练侧 mass 杠杆失效**：readout-attack 中弱 mass 是长程最优，但其增益是**机理侧（推理时 readout 软化）**现象，无法通过训练侧 `mass_coef` 复现到蒸馏配方——印证「机理侧 > 训练侧」。
3. **n_ctx15 已双 seed 确认**（seed42 + seed1234，32k 均值 ≈9 = n_ctx7 的 9，噪声内持平）。负结果稳健。

> *注：n_ctx15 / N256 的具体 launch commit 在 RUN_REGISTRY §3 有结果表但未逐一记 hash，需从 git log 补；结果数字本身已核实。

> **历史佐证（mem_space 谱系，非 pg19 蒸馏）**：训练窗口 nctx7→15→63 的单调负相关与早期 chunk-size / 容量 sweep 一致——chunk512 是甜区（chunk1024 1k 后断崖、chunk256 中长档不及）；num_slots 128→256→384→512→768 sweep 经多 seed 复核后**坐实为噪声**（N384「全面超基准」三 seed 形态全异，不复刻）。「扩容量/加窗口」线性外推到此为止。

### 家族 B — 蒸馏：唯一动过 16k 的路线，但不迁移、32k 仍墙

| 实验 | 数据 | qa5 8k / 16k / 32k | LongBench AVG | 裁决 |
|---|---|---|---|---|
| dolmino self-study 蒸馏 AB（A=logits KL + B=hidden MSE）| dolmino ≤16k，n_ctx=3（窗 2048）| 11 / 11 / 8 | 8.87（3任务）| 中长程全面塌；LongBench < P11 baseline 10.20，无正迁移 |
| dolmino 蒸馏 + weak-mass 0.5 | 同上 | 11 / 14 / 9 | 8.98（3任务）| ≈ AB，弱 mass 无加成（BABILong+LongBench 同向）|
| L2 分层 + pg19 长文 | pg19 | 18 / 13 / 5 | 待补 | 仅 8k 微正，32k 最差 |
| **pg19 真长文蒸馏（n_ctx7 final）** | **pg19 78% ≥32k** | 19 / **16** / 9 | **6.5** | **★16k 破天花板（唯一）** |

> dolmino 两 arm 的 LongBench AVG 为 3-任务子集（multifieldqa_en/2wikimqa/musique，n=100，W0，口径同 P11 baseline=10.20、base 开卷上界=14.67）；pg19 的 6.5 为 6-任务口径，二者不直接可比，但两 dolmino arm < 同口径 P11 baseline 已足够定性。

**裁决**：
- **抬 readout 靠真长文训练数据，不靠蒸馏目标 / mass 旋钮**：dolmino（短窗）蒸馏只复刻短档；真长文（pg19）才把 16k 抬到 16（vs 所有现有方法天花板 13）。
- **★dolmino 蒸馏对真实长文档任务无正迁移（2026-06-18 补评闭环）**：AB / MASS0p5 在 LongBench 3 任务 AVG=8.87/8.98，**低于未蒸馏的 P11 mem_space baseline（10.20）**，更远低于 base 开卷上界（14.67，仅 ~61%）。蒸馏既没补上 BABILong 中长程 readout，也没迁移到真实长文档——印证瓶颈是 readout/数据而非蒸馏目标。MASS0p5≈AB 再次确认弱 mass 无加成。
- **★BABILong 长程突破 ≠ 真实长文档能力**：pg19 在 BABILong 16k 破墙，但 LongBench 仅 **6.5**（≈ mass_coef1 的 6.56，远低于 weak-mass+蒸馏的 10.4）。pg19 的 16k 突破是 **BABILong 合成事实链检索任务特定**，未迁移到真实长文档理解。两个 benchmark 测不同能力。
- **32k 在所有蒸馏变体下仍 = 8/9/5，无一突破天花板 9。**

> **过训单调退化铁律（贯穿全程）**：几乎所有 run step500 ≫ step5000（0k 除外）。P11 chunk512 step5000 在 LongBench 全 6 任务一致劣于 step500（AVG 6.06 vs 8.87），含三个全局语义任务——**反驳「检索→语义压缩」能力迁移假说，证明是单调过训退化**。早停（step500/250）是正确交付策略。

### 家族 C — evidence / retrieval injection：给完美证据也救不动

这是最强、最具决定性的证伪家族。三代注入接口 × 多种检索源（含 oracle 100% 完美命中），**全部落在 OFF 噪声内**。

| 接口 | 检索源 | 探针 | OFF | best arm | Δ | commit |
|---|---|---|---|---|---|---|
| **evidence-prefix**（hidden-state prefix 单层注入）| slot-routed 启发式 + **oracle 100% 命中** | niah_single_1 4k，n=200 | 23.5 | oracle_pos0 = 26.0 | **+2.5（<3pt 噪声）** | a10cb46 / 657f43c |
| **raw-KV-prefix**（未压缩原始 token，query 相似度检索）| raw-KV top-64 / top-256 | niah_single_1 4k，n=100 | 21.0 | 22.0 | **+1.0** | **b4b7183** |
| **★TRUE in-attention K/V concat**（retrieved K,V 经 native k/v_proj + 真实 RoPE 位置直接拼进 L16 self-attn，ONE softmax）| in-attn top-64 / top-256 / **oracle-only** | niah_single_1 4k，n=100 | 22.0 | oracle-only = 21.0 | **−1.0** | **b2fe183 / d9b0945 / 76efbd4** |

**决定性证据 = in-attention oracle-only 臂**：
- 直接把真实 needle 经 TRUE in-attention 注入 L16，`QUERY_DIAG` 实测 top1_sim=0.94 / slot_content_cos=0.50，确认 **perfect content 真进了 self-attention**。
- 结果仍 **21.0 ≈ OFF 22.0** → **冻结 reader 即便拿到完美注入的正确 KV 内容，也无法 CONSUME 来答题。**
- topk64=29(+7) 判为 noise/artifact：topk256（更多检索）反而 20 < OFF，且 scorer needle-precision 实测 0%，oracle-perfect 都只 21——没有机制理由让 topk64 真 +7。

**三方互证**（与诊断层完全一致）：
- LongEval（≥8k 精确检索归零）+ 对话记忆（保 gist 丢精确事实）+ evidence/raw-KV/in-attn 全 ≈ OFF。
- 瓶颈**不是「检索源 / 找不到精确信息 / 注错位置」**，而是 **readout 端 frozen reader 即便逐字拿到原始 KV / 完美 oracle 内容也用不上精确事实**。
- 三代接口（evidence-prefix +2.5 / raw-KV-prefix +1.0 / TRUE-in-attn-oracle −1.0）覆盖了 prefix 与 native-attn 两类注入、slot/raw/oracle-perfect 三类内容 —— **slot + injection 范式的最后一个未证伪 readout 机制已证伪，frozen-reader 框架穷尽。**

---

## 3. 外部文献佐证

> 文献数字为团队此前调研引用（见 `docs/SLOT_ROUTED_EVIDENCE_ROADMAP.md` 与 RESEARCH_LITERATURE），用于定位我们的观察在已知 Pareto 前沿上的位置；具体引用 bibkey 待补。

- **Based / Zoology（recall vs state-size Pareto 前沿）**：固定 recurrent state 大小与 associative recall 能力存在硬 Pareto 权衡——固定 128-slot state 在精确 recall 上必然受限，与我们「容量 sweep 不是有效杠杆」一致。
- **ACL2025 gist-token 压缩**：压缩比 8× 时重建 39.9%，压缩比 16× 时降到 9.6%——压缩比与可恢复精确信息量强相关。我们的 32k→128 slot ≈ 250:1 压缩比远超此区间，精确读出受限符合预期。

---

## 4. 结论与未来方向

### 结论

1. **墙是真实的、形状清晰的**：32k 长程精确事实读出对训练窗口、压缩比、路由旋钮、训练侧 mass、以及三代 evidence/KV 注入接口**全部免疫**。
2. **根因 = 冻结 reader 在固定压缩瓶颈下的「消费能力」限制**：写入侧健康、容量非瓶颈、检索源非瓶颈（oracle 100% 命中也救不动）。问题在冻结 backbone 无法消费注入的精确 KV/内容。
3. **memory 范式本身并未被证伪**：W0 qa5 32k≈44 远超随机，这部分分数完全来自 memory bank（SWA 在 32k 只直接覆盖 ~11% 上下文）。**已证伪的是「在冻结 reader 下用 slot 压缩 + injection 做精确读出」这条具体路线，而非「memory 承载长程」本身。**
4. **唯一相对亮点 = pg19 真长文蒸馏在 16k 破天花板（+3）**，但不迁移到真实长文档，且 32k 仍墙。

### 未来方向（按 confidence 排序）

1. **解冻 reader 部分层 finetune**（唯一未关闭路径）：让 reader 学会「消费」注入的 KV / slot 内容。所有 frozen-reader 实验都指向这是必要条件。**这是架构方向决策，超出 heartbeat 自主权限，需主会话 sign-off。**
2. **换范式**：放弃「固定 slot 压缩 + frozen reader」，转向 retrieval-augmented + 可训练消费层 / 分层记忆 / 动态预算。
3. **明确不再投入**：加训练窗口（4k→8k→32k 单调变差）、扩 slot 容量（多 seed 证伪）、调路由均衡四旋钮、调写入侧（top_k/回收/dense 全谱系证伪）、换检索源做 injection（三代接口全 ≈ OFF）。

### 方法论沉淀（避免后人重走）

- **单 run 分数有高 run-to-run 方差**：容量 sweep 的「N384 突破」被多 seed 证伪为运气。**候选好配置必须 2–3 seed 重复，单 run 只作筛选不作定论。**
- **判据 = W0（纯 memory 读出）相对 baseline 的提升**，不要拿「闭合 SWA gap」当成败线（SWA 直接注意原始 KV ≈ 开卷，是过高标尺）。
- **BABILong（合成事实链）与 LongBench/对话记忆（真实长文档）测不同能力**，优化目标需分清——BABILong 上的突破未必迁移。

---

## 附录：证据缺口（数字对不上 / 缺失）

1. ~~**LongBench AVG 缺 dolmino 蒸馏（AB / MASS0p5 / L2pg19）三 arm 的分**~~ **【已补 AB/MASS0p5，2026-06-18】**：dolmino 蒸馏 AB / MASS0p5 的 LongBench W0 已补评（3 任务 multifieldqa_en/2wikimqa/musique，n=100，口径同 P11 baseline）：AB AVG=8.87、MASS0p5 AVG=8.98，均 < P11 baseline 10.20 < base 开卷上界 14.67 → dolmino 蒸馏对真实长文档无正迁移。结果已写入 RUN_REGISTRY §「self-study 蒸馏 AB」LongBench 小节。**L2pg19 仍待补**（在 .249 diskB，本轮未评）。
2. **n_ctx15 / N256 的 launch commit hash**：RUN_REGISTRY §3 有结果但未逐一记 hash（仅 n_ctx63=4812d70、mass0.5=db584a6 明确）。结果数字已核实，commit 待从 git log 回填。
3. **外部文献 bibkey**：Based/Zoology、ACL2025 gist-token 的精确引用未在 RUN_REGISTRY 落 bibkey，需从 RESEARCH_LITERATURE.md 补全。
4. **32k qa5 「天花板≈9 / 13」两个数字并存**：RUN_REGISTRY 在不同表用「现有方法天花板 ~15/13/9」（8k/16k/32k）作参照，文中 16k 天花板取 13、32k 取 9，与原表一致；若需统一口径建议在 RUN_REGISTRY 固化一处。
</content>
</invoke>
