# A01 — What Survives Null Calibration?

## 状态

**ACTIVE。所有 gate 已跑完；第三条 kill clause（novelty）已核查、未触发。**
当前范围 = `STATUS.json:claim_scope_after_gates` 的 `confirmed_general`。
本文件的 claims 部分已于 2026-08-09 重写以对齐该范围
（此前它仍停留在一个**已被撤回的收窄范围**上）。

## 一句话主张

一个评测量在被解释为“能力”之前，必须与该 construct 自己的
**input-blind null** 比较，而不是与泛化的 chance line 或一个过弱 baseline 比较。
我们统一报告：

```text
reported value / construct-appropriate null / null 的 convention /
calibrated residual / residual fraction
```

并展示这个协议不仅改变误差条，还会**推翻我们自己的 headline**——包括推翻我们
自己对这个 headline 的第一次撤回。

> ⚠️ **`null 的 convention` 是 2026-08-09 新增的必报字段。**
> 见下方 §「协议本身也需要被校准」。这不是格式要求，是一个实测结论：
> 同一个 null 的 tie convention 单独就能把 6 个 arm 里 5 个的判定从
> “above null” 翻成 “significantly BELOW null”。

---

## Claims — 当前范围（2026-08-09 重写）

### 撤回史（必须保留，不得静默删除）

A01 经历了**一次撤回 + 一次对该撤回的反撤回**。读者必须能看到claim过什么、
撤回了什么、为什么。这不是难堪，这是本文的方法论论点本身。

| # | 时间 | 事件 | 依据 |
|---|---|---|---|
| 1 | 2026-08-06 | **原 headline（Obs4）：MC scoring interface 会翻转 model ranking。** 45 arm-pair 里 7 个 sign-opposite、2 个两口径都显著且过 BH。 | `evidence/C5_self_falsification.md` §1 |
| 2 | 2026-08-06 | **自我撤回 #1（成立，保留）**：3/3 参与 flip 的 arm 在 letter 口径上都**处于或低于**自己的 best-constant floor。限制到两口径都显著高于 floor 的 4 个 arm → 6 pair、**0 sign flip、0 显著 flip**。flip 真实存在，但完全发生在仪器已失效的区域。 | `evidence/C5_self_falsification.md` §4 |
| 3 | 2026-08-09 | **撤回 #2（❌ 本身已被撤回）**：`GATE1_VERDICT.md` §1 曾判 `KILL_CONDITION_CLAUSE_2_TRIGGERED`，结论是“把 A01 收窄为 *letter interface 只在结构损伤的 OLMo-2 上退化*”，并要求 drop “letter MC interface 一般而言是不可靠仪器”这一 claim。 | `GATE1_VERDICT.md` §1（原文保留，标注 RETRACTED） |
| 4 | 2026-08-09 | **反撤回（当前有效）**：撤回 #2 测的是**错误的条件**。A01 的 kill clause 讲的是**受损**模型；`GATE1_VERDICT.md` 测的是**完好** base，完好 base 从未被预期出现该病理，因此既不能触发也不能解除该 clause。受损臂实验（**6/6** 非 OLMo 受损臂在自己的 floor 之下）**确认**了一般性 claim。 | `GATE1_DAMAGED_VERDICT.md`、`STATUS.json:claim_scope_after_gates.RETRACTED_must_narrow` |

**从 #3→#4 学到的、要写进论文的教训**：判定一条 kill clause 之前，先确认
自己测的是不是该 clause 说的那个条件。撤回 #2 是在 intact base 上做的，
而 clause 说的是 damaged——这是 A01 自己的协议（“先确认仪器在该条件下有效”）
在 meta 层面的又一次应验。

### ✅ CONFIRMED — 可以作为一般性结论主张

来源：`STATUS.json:claim_scope_after_gates.confirmed_general`（六条），逐条附证据。

1. **结构损伤下，letter MC interface 退化到自己 best-constant floor 之上够不到、
   甚至更低。** 四个家族：OLMo-2-7B（healed keep8，letter `0.2550`）、
   Llama-2-7B（截断，`0.2295`–`0.2415`）、Llama-3-8B（截断，`0.2329`–`0.2527`）、
   Qwen3-8B-Base（截断，`0.2286`–`0.2301`）。floor = always-D `0.2689`，n=14042/臂。
   → `GATE1_DAMAGED_VERDICT.md`

2. **这个塌陷是深度上的 SHARP PHASE TRANSITION，不是渐进衰减。**
   Qwen3-8B（36L）在 **k=4…24 共 12 个实测深度**上 letter 钉在
   `0.229454`–`0.230309`（**跨度 0.0855 pp**），随后在 **k24→k25 一层之内跳
   `+48.02` pp**（`0.229739` → `0.709942`），此后 plateau `0.647`–`0.730`。
   同一批 forward pass 上 content_norm 在 k24→k25 只动 **`+1.35` pp**
   （`0.3045` → `0.3180`）且全程单调平滑。
   Llama-3-8B 在 k17→k18 跳 `+30.34` pp；OLMo-2-7B 在 k18→k19 跳 `+26.68` pp。
   → `GATE1_DEPTHCURVE_VERDICT.md`；数值本次复核自
   `olmo2_mmlu_content_results/gate1_dmg_{qwen3_8b,llama3_8b,olmo2_7b}_depth*_k*/summary.json`（zwfy6）

   > **⚠️ 已修正的内部矛盾（2026-08-09）**：`STATUS.json` 曾在两处给出不同的
   > Qwen3 transition：`confirmed_general[1]` 写 “keep30 跳 +45.74pp”，
   > `gate_results.gate1_depth_curve` 写 “k24→k25 跳 +48.0pp”。
   > **正确的是 k24→k25 / +48.02pp。** `+45.74pp` 是 **k30 相对 floor 0.2689 的
   > 残差**（`0.726321 − 0.268908 = +45.7413` pp），不是任何一层的跳变，也不是
   > transition 点——k25 已经在 plateau 上（`0.709942`，相对 floor `+44.10` pp）。
   > `STATUS.json` 已改。

3. **换 interface 救不了受损臂。** 每个受损臂的 content_norm 都落在自己 letter
   accuracy 的 ±3pp 内，且两者都贴着各自的 floor。
   → `GATE1_DAMAGED_VERDICT.md` §4

4. **任何 construct 在跨 arm 比较之前，必须先对齐一个 construct-appropriate null
   （best-constant，而不是 chance）。** 三个可验证的演示：
   * keep8 letter `0.2550` 相对 `.25` chance 读起来“高于随机”，相对 floor
     `0.2689` 则是**低于 floor**（fp32 下 `−1.538` pp，boot p `0.0062`）；
   * BoolQ raw 口径上 keep12 `0.6101` / keep10 `0.6086` / keep8 `0.5948` 相对
     `.50` 读起来“及格”，相对 always-B floor `0.6217` 则在其**之下**；keep14
     `0.6382` 虽 `+0.0165` 但 McNemar `p = 0.20` **不显著** → **6 臂里 4 臂
     无法与一个常数预测器区分**；
   * OBQA 的 residual fraction 在 chance 线下被夸大 **2.15×**（`45.9%` → `21.3%`），
     且 keep10 的 `acc_norm 0.3560` 低于 longest-option floor `0.3635`。
   → `status/scout_21/lane2_a01_gate2.md`（仅 wzc1；zwfy6 无此文件）、
     `STATUS.json:gate2_second_mc_benchmark`、`GATE3_VERDICT.md`

5. **wrong-null 问题不是 MMLU 的产物。** gate-2 在非 MMLU 的 MC benchmark 上复现。
   其中 winogrande 结构性退化（两个选项共享 continuation → `acc == acc_norm`
   恒等、tie rate 100%），**作为 negative control 报告，不作为证据**。
   → `STATUS.json:gate2_second_mc_benchmark`

6. **self-falsification narrative**：用自己的协议撤回自己的 headline——
   现在还包括撤回自己对该 headline 的第一次撤回（见上表 #3/#4），
   以及 `GATE1_HEALEDARMS_VERDICT.md` §7 那次“我在没测 intact base 的
   content_norm 的情况下写了一条关于它的 claim，测完发现符号相反”的更正。

### ⚠️ FAMILY-SPECIFIC — 不得作为一般性结论主张

来源：`STATUS.json:claim_scope_after_gates.family_specific_not_general`。

1. **塌陷的 MECHANISM（机制）是 family-specific 的，OUTCOME 才是一般的。**
   OLMo-2 走的是 **bf16 exact ties**（keep8 上 `30.64%`）被 argmax 的 index bias
   打破这条路；Llama / Qwen 走的是**直接 modal collapse、ties 近零**这条路
   （Llama-2 k12 是 **100.0% modal / 0.00% ties**；Qwen3 k12 是
   **99.1% modal / 0.03% ties**）。**同一个终点，不同的路。**
   论文里必须写成“damage 把 letter interface 压向一个常数预测；在 bf16 下这有时
   表现为 exact ties（OLMo），有时表现为 sharp modal collapse（非 OLMo）”，
   **不得**写成一个统一机制。
   → `GATE1_DAMAGED_VERDICT.md` §3、`GATE3_VERDICT.md` §3

2. **“content 才是公平的 interface” 不是关于 MC interface 的一般陈述，
   它只是关于受损模型的陈述。** 在**健康且 letter 有竞争力**的模型上 letter
   反而比 label-free content **好 13–23 pp**（OLMo-2-7B `−13.59` pp、
   Llama-3-8B `−15.96` pp、Qwen3-8B `−22.91` pp，符号为 `cn − l`）。
   在中等能力档（letter `0.37`–`0.41`）则是 family-dependent：OLMo-2-1B 与
   Llama-2-7B 是 TIE，Llama-3.2-1B 显著但只 `2.4` pp。
   因此**“总是报 content_norm”是过强的建议**。
   → `GATE1_VERDICT.md` §4（verdict 已撤回但数字有效）、
     `GATE1_HEALEDARMS_VERDICT.md` §7.1.3

### ❌ MUST DROP — 已被证伪，不得以任何形式出现

**“Exact ties 是 interface failure 的机制”作为 family-general 的 CAUSAL claim。
四重证伪：**

| # | 证据 | 结果 |
|---|---|---|
| a | gate-3 base：ties `18 → 0`（fp32） | letter acc 完全不动，McNemar **p = 1.000** |
| b | gate-3 keep8：ties `4303 → 0`，**18.03%** 的 argmax 翻转 | letter acc 不动（p = 0.570），**仍在 floor 之下** |
| c | 非 OLMo 受损臂 ties `0.00%`–`1.35%` | 却塌陷到同一个地方 |
| d | **（2026-08-09 新增第四条）** 六个 OLMo-2 臂全部有 fp32-vs-bf16 结果：fp32 把 exact-tie rate 打到 `0.00%`（shortgpt16 `0.01%`），而 **6 个臂里 5 个的 floor 判定一动不动**；keep8 **仍在 floor 之下且更显著**（bf16 `−1.389` pp p `0.0192` → fp32 `−1.538` pp p `0.0062`） | ties 是 bf16 artifact，移除它不恢复任何可测精度 |

正确的（更弱、更干净的）表述：受损模型的四个 letter logit 在 bf16 精度下已无法
分辨（gap median 从 base 的 `1.1185` 压到 keep8 的 `0.2500`，压缩 4.5×）；
tie count 是这种压缩的**读数**，不是它的**原因**。**interface 丢的是信息，
不是打破平局的能力。**

唯一的 dtype 边界情形，**不得过度解读**：keep12 的判定是 dtype-fragile 的
（bf16 `+0.385` pp p `0.3736` = AT floor；fp32 `+0.819` pp p `0.0272` = above floor）。
这是 6 个臂里的 1 个，fp32 的 CI 只以 `0.078` pp 越过零，`p = 0.0272` **未做多重
比较校正**、过不了六臂 BH。且 keep12 在 fp32 下 modal 集中度**升高**
（D `49.3%` → `61.3%`），即精度提高让它**更像**一个常数预测器。
写成“六臂中有一臂的判定在边界上对 dtype 敏感，故每个判定都应连同 dtype 一起报”
（这正是 A01 自己的主张），**不得**写成“fp32 救了 keep12”。

→ `GATE3_VERDICT.md`、`STATUS.json:gate3_fp32_causal_tie_test`

---

## 协议本身也需要被校准（2026-08-09 新增，A01 目前最强的自有 claim）

A01 让别人用 construct-appropriate null 取代 chance line。**但 A01 自己的 MC
content null 存在同一类未申明的自由度。** “null 取最长选项”这句话没有规定
多个选项在 token 数上打平时怎么办，而 MMLU 上 **34.22%** 的 item 存在这种平局
（其中 **13.37%（1877 item）四个选项全平**）。同一句英文的五种合理读法给出：

| convention | null | 六臂判定（bf16 content_norm） |
|---|---:|---|
| `split`（预注册 canonical） | **0.284450** | 6/6 above |
| `first`（最低索引） | 0.281085 | 6/6 above |
| `last`（最高索引） | 0.282154 | 6/6 above |
| **`credit`（乐观/oracle）** | **0.453710** | **1/6 above，5/6 显著 BELOW** |
| **`wrong`（悲观）** | **0.196126** | 6/6 above |

**null 本身摆动 25.76 pp，比 chance `0.25` 到 intact base content `0.4706` 的
全部距离还大。** intact base 的 residual fraction 在两个极端 convention 下是
`0.0359` vs `0.5832`——**16.26×**，单臂、单一个未申明的 convention。
即便完全弃用 `credit`，`wrong`-vs-`split` 在 base 上仍差 `1.47×`、在 keep8 上
`2.53×`。

**结论：协议必须写成“报 construct-appropriate null **并且打印该 null 的
convention**”。** 这与 gate-4 是同一种纪律，只是深了一层：gate-4 发现
**aggregation** 选择让 headline span 在 `6.86×`–`10.04×` 之间移动，所以 headline
必须是区间；这里发现 **null 自己**有 convention 参数，其影响（单臂 residual
fraction `16.26×`）比 gate-4 担心的 aggregation 效应更大。

**并且这个 null 还依赖 tokenizer。** “最长选项”是按 **continuation token 数**
量的，各家族分词不同，所以同一个 `split` convention 在同一批 item 上给出
Llama-2-7B `0.2757` / Qwen3-8B `0.2833` / Llama-3-8B `0.2847` / OLMo-2 `0.2845`
（`evidence/a01_gate1_third_family.json` 的 `longest_option_split_tie_null`
字段，跨度 `0.90` pp）。因此**任何跨家族的 content 比较都必须用 per-family、
per-convention 的 null**，不能共用一个 `0.2845`。这一条与 tie convention 相互
独立，两者叠加。

→ `GATE3_CONVENTIONS_VERDICT.md`、`evidence/gate3_content_null_conventions.{json,csv}`、
  `code/a01_gate3_content_conventions.py`。所有已发表的 A01 数字都用的是 `split`
  且已标注，**as published 全部正确**（对档案 summary 复核到 `<1e-12`）。

---

## 已完成的四类 construct

| Construct | Reported | Null | convention | Residual fraction |
|---|---:|---:|---|---:|
| C1 MC content scoring | 0.3598 | longest-option 0.2845 | **`split`（tie 平分）** | 20.9% |
| C2 Generative majority prior | 0.6590 | constant refusal 0.4985 | — | 24.4% |
| C3 Representation similarity | 0.4907 | layer-order shuffle 0.4529 | 2000-perm | 7.69% |
| C4 Probe readout depth | 0.6610 | native readout 0.1505 | V1 预注册 | 77.2% |

稳妥表述是：残余比例约 **8%–77%**；**不要**把“恰好超过 10×”作为 headline，
因为 C4 aggregation 的合理变体给出约 **6.86×–10.04×**（gate-4 五个变体里只有
预注册的 V1 达到 10.04×，最近的替代变体给 9.98×）。可写的句子是
“residual fractions span 约 6.9–10.0×，取决于 C4 aggregation convention；在预注册
primary 下为 10.0×”。

→ `GATE4_VERDICT.md`、`evidence/gate4_c4_prereg.json`

## Representation leg（C3）

- observed mean midband z-CKA：`0.490672`
- 2000-permutation layer-order null：`0.452936`
- calibrated residual：`0.037737`
- residual / reported：`7.69%`
- BH `q=.05`：`52/91` pair（raw `p<.05` 为 `57/91`）

random-init `0.0912` 是错误 null；使用它会把可用 correspondence signal
夸大约 10.6×。

---

## 新颖性边界（2026-08-09 经正式 venue 核实后更新）

完整核查见 **`NOVELTY_CHECK.md`**（含每篇的 verified venue + 验证来源）。
**第三条 kill clause 未触发。**

### 不能主张（必须引用，不得据为己有）

- 首创 permutation null calibration / 首创 BH / “表征相似性文献没有 null”；
- “MCQA 应该用比 chance 更强的 baseline” — **Balepur et al., ACL 2024 main**
  (`10.18653/v1/2024.acl-long.555`) 已明确提出；
- “null model / 常数输出能在 benchmark 上拿高分” — **Zheng et al., ICLR 2025 Oral**
  (arXiv:2410.07137) 已提出；
- **`acc` vs `acc_norm` 的 length 敏感性 —— 这一条必须 DROP。**
  **Oostermeijer, ICML 2026**（arXiv:2607.12767，OpenReview `venueid=ICML.cc/2026/Conference`
  + `Camera_Ready_Revision`）已证明 length normalisation 会 over-correct。
  A01 的 OBQA sign-flip 只能改写成“在受损模型这一新设定下的复现”；
- letter/cloze 两种 MC interface 的存在及其改变结果 — **OLMES, Findings of NAACL 2025**
  (`10.18653/v1/2025.findings-naacl.282`) 已标准化；
- “construct validity 是 LLM benchmark 的问题” — **Bean et al., NeurIPS 2025 D&B**
  (arXiv:2511.04703) 已系统综述 445 个 benchmark；
- “MC scoring interface 的有效性令人担忧” — **Cho et al., ICLR 2026**
  (arXiv:2502.18798) 已从 choice sensitivity 角度提出。

### 可以主张

1. 跨多个无关 construct 的统一 null-calibrated reporting（含 convention 字段）；
2. **null 自身的 convention 自由度会反转判定**（`credit` 翻 5/6 臂，
   单臂 residual fraction 16.26×）——未在任何已核实的 prior art 中出现；
3. 针对 layer correspondence 问题的 **layer-order null**
   （措辞降级为“我们未发现更早的 layer-order null”，而非“首创”）；
4. 把该协议先用于**撤回自己的结果**，并且用它撤回自己的第一次撤回；
5. 给出 calibrated residual fraction，而非只给显著性；
6. **对 OLMES 的一处 defect 做 follow-up 修正**：按**模型规模**选 interface 的规则
   在**读出能力已受损**的模型上失效（受损 7B 会被规则派到 letter 口径，而它在那里
   位于自己的 best-constant floor 之下）；正确的键是该臂自己的 floor test。

### 必须补的反向 caveat

**Feng, Wallace, Boyd-Graber, “Misleading Failures of Partial-input Baselines”**
（arXiv:1905.05778，venue 待按 Anthology 核实）证明 partial-input baseline
**失败**并不能证明数据集没有 artifact。因此 A01 必须明写：**越过 floor 是必要
条件，不是充分条件。** A01 现有 claim 的方向本来就是安全的（只用 floor-failure
去**取消资格**，从不用 floor-success 去**认证有效**），但论文里要把这句说出来。

---

## Gate 状态

| gate | 状态 | 结论 |
|---|---|---|
| 1. 第三个模型家族的 MC interface case | ✅ DONE | intact 腿测错了条件（已撤回）；damaged 腿 **6/6 确认一般性 claim** |
| 1b. depth curve（四家族，单层分辨率） | ✅ DONE | letter 是 step function，content 平滑单调；三家族 transition 钉到单层 |
| 1c. healed arms | ✅ DONE | heal 软化 step 但不闭合 interface gap；shortgpt16 保住 intact 的排序 |
| 2. 非 MMLU 的 MC benchmark | ✅ DONE | **复现**（所以 AND-gate 不可能触发） |
| 3. OLMo full-fp32 forward | ✅ DONE（六臂） | ties 是 bf16 artifact；移除后判定 5/6 不变 → 机制被证伪 |
| 3b. longest-option convention 表 | ✅ DONE（2026-08-09） | **null 自己有 convention 自由度，翻 5/6 臂** |
| 4. C4 aggregation 预注册 | ✅ DONE | headline 必须是区间 `6.86×–10.04×` |
| 5. novelty / prior-art 边界 | ✅ DONE（2026-08-09） | **kill clause 3 未触发**；6 条引用义务 + 1 条 sub-claim 撤回 |

### 成功条件（对照）

- ✅ 至少三个 construct 的 null calibration 改变科学结论：C1（ranking flip 全部
  发生在 floor 之下）、C2（BoolQ 4/6 臂从“及格”变成不可与常数预测器区分）、
  C3（residual fraction 从错误 null 的 10.6× 夸大回落到 7.69%）。
- ✅ 第三模型 / 第二 benchmark 保持 “instrument validity before comparison”：
  damaged 腿 6/6 + gate-2 复现。
- ✅ 与已有 similarity-null prior art 的边界经正式 venue 核实：见 `NOVELTY_CHECK.md`
  （ACL 系走 Anthology，OpenReview 系走 `venueid` + `Camera_Ready_Revision`）。

### Kill 条件（逐条对照，全部未触发）

1. ❌ 未触发 — “除 representation 外其他 construct 在严格 null 下结论都不变”：
   假。C1/C2 的结论都被 null 改变（见上）。
2. ❌ 未触发 — “第三家族和第二 benchmark 均不复现 interface failure”：
   假。damaged 第三/第四家族 **6/6 复现**，gate-2 也复现。
   （2026-08-09 曾被误判为触发，因为测的是 intact base 而 clause 说的是 damaged。）
3. ❌ 未触发 — “论文只能退化为已有 similarity-null 方法的案例集合”：
   假。存活的核心发现（四家族 floor 塌陷、letter-as-step-function、fp32 机制证伪、
   null convention 敏感性）都在 MC-accuracy construct 上，不是 similarity-null 的
   实例。见 `NOVELTY_CHECK.md` §1/§7。

---

## 不得复活的旧数字

来源 `STATUS.json:must_not_resurrect`，逐条给出更正后的正确值。

| 旧数字 | 为什么错 | 正确值 |
|---|---|---|
| `4.8×` | 口径不明 | 相对 `.25` 为 **`4.69×`**；相对 content 自己的 floor 为 **`3.22×`** |
| longest-option `.2822` | **不是算错**——它是 `last`-of-maximal convention（精确 `0.282154`）。它是一个**不同且较难辩护的 convention**，被预注册的 `split` 取代。 | canonical 为 split-tie **`0.2845`**（`0.284450`）。⚠️ 现在应把它放进 convention 表里公开讨论，而不是单纯禁用 |
| `58/91 significant` | 未做 BH | canonical **`52/91`**（BH q=.05）；raw `p<.05` 为 `57/91` |
| 把 `.25` 当 MMLU 的 null | chance 不是 construct-appropriate null | letter null = always-D **`0.2689`**；content null = **`0.2845`**（`split`） |
| 把 `.50` 当 BoolQ 的 null | 同上 | always-B **`0.6217`**（gold 2033 B / 1237 A） |
| “Qwen3 在 keep30 跳 +45.74pp” | 混淆了**跳变**与**相对 floor 的残差** | transition 是 **k24→k25，+48.02 pp**（`0.229739`→`0.709942`）；`+45.74 pp` 是 **k30 相对 floor 的残差**（`0.726321−0.268908`）|
