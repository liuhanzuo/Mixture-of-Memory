# B11 — Generative long-context scorers can encode output format strongly enough to destroy ranking

## 状态

**BACKLOG。NOVELTY 未核查（这是启动前的第一道门），claim 未跨家族复现。**
**不得在通过 novelty check 之前花任何 GPU。**

来源：A02 的 read-tax gate 产出（`proposal/backlog/A02-comem-write-read-repair/A02_BABILONG_MISORDER_VERDICT.md`
§1）。A02 只保留 provenance，不拥有这个 claim —— 它对任何被 `babilong.metrics` 打分的
base LM 都成立，与 CoMem / depth knob 无关，所以单独立项。

## Claim under test

生成式长上下文 benchmark 的 scorer，其**文本预处理**可以把「输出格式习惯」编码进分数，
强到足以**破坏一个真实效应量 +70pp 的排序**。而且失效点是**可定位到具体代码行的**，
不是笼统的「benchmark 有噪声」。

## 已成立（A02 实测，n=100/cell，canonical scorer，两盘 md5 一致）

在一个 RULER 上真实差距 **+70 到 +84pp** 的 depth 操作（A4 j=12 vs A5 j=18）上：

- 6 个 BABILong cell **全部无法恢复该排序**；其中 **4 个点估计反向**。
- **反向不显著**：最好的一个 exact McNemar p = 0.0703（判别项仅 8 个：b=1,c=7），
  Holm 校正后 0.4219。**所以 claim 是 ranking failure，不是 demonstrated sign error。**
- 机制**不是** retrieval 瓶颈：条件在 retrieval-HIT 子集上，反向**更大**
  （qa1×16k HIT −6.67 vs MISS +8.57）。预注册的 retrieval-domination 假设被**证伪**。
- 机制是 **metric 预处理 + floor**：
  `preprocess_output` 在**第一个句号**处截断，`compare_answers` 又要求 target 是
  **唯一**存活 label。于是 `'Choices: A. In the kitchen B. ...'` → `'choices: a'` → 判 0,
  **无论模型是否找到了事实**。
  A4 有 **22–43%** 的 item 属于「target 在原始输出里但被截断杀掉」，A5 只有 5–16%。
- **一行消融**（只删截断，保留唯一性要求，故 multiple-choice 仍判 0，无 chance inflation）：
  修复 **2/6** cell 的反向，qa1×32k 的 ladder 变成 **ρ = −1.000 (p=0.0167)**。
  另 2 个（qa2）修不动 —— 那是 floor（A4 = 1%）。
- cell 级 dissociation 完美：反向的 4 个 cell **正是** list-format 率高的 4 个
  (60–75% vs qa5 的 2%)，`[[4,0],[0,2]]`，Fisher p = 0.0667（6 cell 的下界，描述性）。

## 未成立 / 必须先做

1. **NOVELTY（第一道 gate，0 GPU）**：answer-extraction / metric-robustness 文献很大。
   必须核查 lm-evaluation-harness 的 answer extraction、`exact_match` 变体、
   MCQ answer-parsing robustness、"LLM-as-judge vs string match" 这几条线。
   **很可能已被做过**；若已被完整覆盖 → 直接 archive，不烧 GPU。
   注意 venue 核实两套家族规则：OpenReview 系（ICLR/NeurIPS/ICML）看 `venueid`；
   ACL 系（含 Findings）必须 aclanthology + DBLP。
2. **跨家族复现**：目前只有 Qwen3-8B 一个家族。输出格式习惯是**家族特性**，
   单家族无法区分「scorer 脆弱」与「这个模型爱输出选择题」。至少再要一个家族。
3. **跨 benchmark**：目前只有 `babilong.metrics`。若只此一家，claim 退化为
   「BABILong 的 scorer 有一个 bug」——那是 issue report，不是论文。
   需要检查 LongBench / RULER / LongEval 的 scorer 是否有同类预处理。
4. **显著性**：反向本身不显著。若要主张「sign error」，需要更大 n 或更强的效应。
   当前只能主张 ranking failure。

## Kill gates

- **K1（novelty）**：若 answer-extraction robustness 文献已覆盖「预处理截断改变模型排序」
  → **archive**。
- **K2（跨家族）**：若第二个家族上 list-format 率不再有臂间差异，且反向消失
  → claim 退化为「Qwen3 的格式习惯 + 一个 scorer bug」→ **archive**，
  作为 A02 的 appendix negative result。
- **K3（单 benchmark）**：若其他生成式长上下文 scorer 都没有同类预处理
  → 降级为对 BABILong 的 upstream bug report，不成篇。

## 不能声称的

- **不能**说「BABILong 会让结论反向」——反向不显著（p=0.0703，Holm 0.4219）。
- **不能**说 retrieval 是机制 —— 已证伪（HIT 上反向更大）。
- **不能**把 retrieval 与 floor 分开归因 —— 6 个 cell 上两者共线（ρ=+0.714），
  **mechanism NOT identified**。
- **不能**用 `target_in_raw`（宽松 substring）的 4/6 数字 —— 它对 multiple-choice
  有 chance inflation，已被严格消融的 **2/6** 取代。
- **不能**与 B04 合并。B04 是 *per-item `acc_norm` margin compression under damage*
  （likelihood ranking，无生成、无检索、无字符串匹配），当前 `NARROWED_TO_OLMO_2_ONLY`。
  同一个「eval fragility」词，**不同 construct、不同机制、不同失效面**。
  合并会让 OLMo-2-only 的 margin claim 从一个无关机制借力。

## 证据位置

`proposal/backlog/A02-comem-write-read-repair/evidence/babilong_misorder/`
（7 个 JSON，wzc1 与 zwfy6 md5 一致）+ `code/analyze_a02_truncation_ablation.py`
（一行消融）、`code/analyze_a02_format_mechanism.py`（per-item 格式统计）。
原始 generation 在 zwfy6 `babilong_results/a02_{rtax,dvr,babilong_c2}_*`。
